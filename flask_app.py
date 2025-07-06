#!/usr/bin/env python3
import pandas as pd
import yfinance as yf
import requests
from pathlib import Path
import warnings
import numpy as np
from flask import Flask, render_template, jsonify
import logging
from scipy import stats
from functools import lru_cache
from datetime import datetime

# --- Configuração Básica ---
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

# --- Configurações Centralizadas ---
CONFIG = {
    "DIRETORIO_DADOS_CONSOLIDADOS": BASE_DIR / "consolidated_data",
    "CAMINHO_MAPA_TICKER_CVM": BASE_DIR / "mapeamento_tickers.csv",
    "PERIODO_BETA_IBOV": "5y",
    "CONTAS_CVM": {
        "EBIT": "3.05", "IMPOSTO_DE_RENDA_CSLL": "3.10", "LUCRO_ANTES_IMPOSTOS": "3.09",
        "ATIVO_NAO_CIRCULANTE": "1.02", "CAIXA_EQUIVALENTES": "1.01.01",
        "DIVIDA_CURTO_PRAZO": "2.01.04", "DIVIDA_LONGO_PRAZO": "2.02.01",
        "CONTAS_A_RECEBER": "1.01.03", "ESTOQUES": "1.01.04", "FORNECEDORES": "2.01.02",
        "DESPESAS_FINANCEIRAS": "3.07" 
    },
}

# --- Funções Utilitárias de Carregamento de Dados ---

def carregar_csv_robusto(caminho_arquivo, **kwargs):
    separators = [",", ";"]
    encodings = ["utf-8", "latin-1", "cp1252"]
    kwargs.pop("low_memory", None)
    for sep in separators:
        for encoding in encodings:
            try:
                df = pd.read_csv(caminho_arquivo, sep=sep, encoding=encoding, engine='python', **kwargs)
                if len(df.columns) > 1:
                    return df
            except Exception:
                continue
    raise Exception(f"Não foi possível carregar o arquivo '{caminho_arquivo.name}' corretamente.")

@lru_cache(maxsize=1)
def carregar_mapeamento_ticker_cvm():
    caminho_arquivo = CONFIG["CAMINHO_MAPA_TICKER_CVM"]
    try:
        if not caminho_arquivo.exists():
            raise FileNotFoundError(f"Arquivo de mapeamento '{caminho_arquivo.name}' não encontrado.")
        df = carregar_csv_robusto(caminho_arquivo, header=0, dtype=str)
        df.columns = [str(col).strip().upper() for col in df.columns]
        required = ["CD_CVM", "TICKER", "NOME_EMPRESA"]
        if not all(col in df.columns for col in required):
            raise ValueError(f"As colunas necessárias {required} não foram encontradas.")
        df = df[required].copy()
        df["TICKER"] = df["TICKER"].str.strip()
        df["CD_CVM"] = pd.to_numeric(df["CD_CVM"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["CD_CVM", "TICKER"])
        return df, None
    except Exception as e:
        logging.error(f"Erro CRÍTICO ao carregar mapeamento de tickers: {e}", exc_info=True)
        return None, str(e)

@lru_cache(maxsize=4)
def carregar_dados_preparados():
    try:
        demonstrativos = {}
        tipos_necessarios = ["dre", "bpa", "bpp", "dfc_mi"]
        for tipo in tipos_necessarios:
            caminho_arquivo = CONFIG["DIRETORIO_DADOS_CONSOLIDADOS"] / f"{tipo}_consolidado.csv"
            if not caminho_arquivo.exists():
                if tipo == "dfc_mi":
                    demonstrativos[tipo] = pd.DataFrame()
                    continue
                raise FileNotFoundError(f"Arquivo de dados essencial não encontrado: {caminho_arquivo.name}.")
            df = carregar_csv_robusto(caminho_arquivo, dtype={"CD_CONTA": str})
            if "CD_CVM" in df.columns:
                df["CD_CVM"] = pd.to_numeric(df["CD_CVM"], errors="coerce").astype("Int64")
            if "VL_CONTA" in df.columns:
                df["VL_CONTA"] = pd.to_numeric(df["VL_CONTA"], errors="coerce").fillna(0) * 1000
            demonstrativos[tipo] = df
        return demonstrativos, None
    except Exception as e:
        logging.error(f"Erro CRÍTICO ao carregar dados consolidados: {e}", exc_info=True)
        return None, f"Erro ao carregar arquivos de dados consolidados: {e}"

@lru_cache(maxsize=1)
def obter_dados_mercado():
    """Obtém premissas de mercado (taxa livre de risco, prêmio) e dados do IBOV para cálculo do Beta."""
    dados = {"risk_free_rate": 0.105, "premio_risco_mercado": 0.08, "cresc_perpetuo": 0.03}
    
    # Fetch SELIC (Risk-Free Rate)
    try:
        selic_url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados/ultimos/1?formato=json"
        response = requests.get(selic_url, timeout=10)
        response.raise_for_status()
        selic_value = float(response.json()[0]["valor"])
        dados["risk_free_rate"] = selic_value / 100.0
    except Exception as e:
        logging.warning(f"Não foi possível obter a SELIC do BCB. Erro: {e}")

    # Fetch IPCA (Inflation)
    try:
        ipca_url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados/ultimos/1?formato=json"
        response = requests.get(ipca_url, timeout=10)
        response.raise_for_status()
        ipca_data = response.json()[0]
        # Formata a data para o padrão brasileiro
        data_obj = datetime.strptime(ipca_data['data'], '%d/%m/%Y')
        mes_ano = data_obj.strftime('%m/%Y')
        dados["ipca_mes"] = f"{float(ipca_data['valor'])}% (ref. {mes_ano})"
    except Exception as e:
        logging.warning(f"Não foi possível obter o IPCA do BCB. Erro: {e}")
        dados["ipca_mes"] = "N/A"

    # Fetch Exchange Rate (Dolar)
    try:
        cambio_url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.1/dados/ultimos/1?formato=json"
        response = requests.get(cambio_url, timeout=10)
        response.raise_for_status()
        cambio_data = response.json()[0]
        dados["cambio_dolar"] = f"R$ {float(cambio_data['valor'])}"
    except Exception as e:
        logging.warning(f"Não foi possível obter o Câmbio do BCB. Erro: {e}")
        dados["cambio_dolar"] = "N/A"

    # Fetch IBOV data
    try:
        dados["ibov_data"] = yf.download("^BVSP", period=CONFIG["PERIODO_BETA_IBOV"], progress=False, timeout=15)
        if dados["ibov_data"].empty:
            raise ValueError("Download do IBOV retornou um DataFrame vazio.")
    except Exception as e:
        logging.error(f"Falha ao baixar dados do IBOV. O cálculo do Beta usará o valor padrão 1.0. Erro: {e}")
        dados["ibov_data"] = pd.DataFrame() 
    return dados

def obter_valor_recente(df_empresa, codigo_conta):
    historico = obter_historico_metrica(df_empresa, codigo_conta)
    return historico.iloc[-1] if not historico.empty else 0

def obter_historico_metrica(df_empresa, codigo_conta):
    metric_df = df_empresa[(df_empresa["CD_CONTA"] == codigo_conta) & (df_empresa["ORDEM_EXERC"] == "ÚLTIMO")]
    if metric_df.empty:
        return pd.Series(dtype=float)
    metric_df = metric_df.copy()
    metric_df["DT_REFER"] = pd.to_datetime(metric_df["DT_REFER"])
    return metric_df.set_index("DT_REFER")["VL_CONTA"].sort_index()

def calcular_beta(ticker, ibov_data):
    if ibov_data.empty: return 1.0
    try:
        dados_acao = yf.download(ticker, period=CONFIG["PERIODO_BETA_IBOV"], progress=False, timeout=15)
        if dados_acao.empty or len(dados_acao) < 60: return 1.0
        dados_combinados = pd.concat([dados_acao["Adj Close"], ibov_data["Adj Close"]], axis=1).dropna()
        retornos = dados_combinados.pct_change().dropna()
        if len(retornos) < 50: return 1.0
        retornos.columns = ['Acao', 'Ibov']
        slope, _, _, _, _ = stats.linregress(retornos['Ibov'], retornos['Acao'])
        beta_ajustado = 0.67 * slope + 0.33 * 1.0
        return beta_ajustado if not np.isnan(beta_ajustado) else 1.0
    except Exception:
        return 1.0

def processar_valuation_empresa(ticker_sa, codigo_cvm, demonstrativos, market_data):
    try:
        dre, bpa, bpp = demonstrativos["dre"], demonstrativos["bpa"], demonstrativos["bpp"]
        empresa_dre = dre[dre["CD_CVM"] == codigo_cvm]
        empresa_bpa = bpa[bpa["CD_CVM"] == codigo_cvm]
        empresa_bpp = bpp[bpp["CD_CVM"] == codigo_cvm]

        if any(df.empty for df in [empresa_dre, empresa_bpa, empresa_bpp]): return None

        info = yf.Ticker(ticker_sa).info
        market_cap = info.get("marketCap")
        preco_atual = info.get("currentPrice", info.get("previousClose"))
        n_acoes = info.get("sharesOutstanding")

        if not all([market_cap, preco_atual, n_acoes, n_acoes > 0, market_cap > 0]): return None

        C = CONFIG["CONTAS_CVM"]
        hist_ebit = obter_historico_metrica(empresa_dre, C["EBIT"])
        if hist_ebit.empty or hist_ebit.iloc[-1] == 0: return None

        imposto_total = obter_historico_metrica(empresa_dre, C["IMPOSTO_DE_RENDA_CSLL"]).sum()
        lucro_antes_ir = obter_historico_metrica(empresa_dre, C["LUCRO_ANTES_IMPOSTOS"]).sum()
        aliquota_efetiva = abs(imposto_total / lucro_antes_ir) if lucro_antes_ir != 0 else 0.34
        aliquota_efetiva = max(0, min(aliquota_efetiva, 0.45))

        nopat_recente = hist_ebit.iloc[-1] * (1 - aliquota_efetiva)

        ncg = (obter_valor_recente(empresa_bpa, C["CONTAS_A_RECEBER"]) + 
               obter_valor_recente(empresa_bpa, C["ESTOQUES"]) - 
               obter_valor_recente(empresa_bpp, C["FORNECEDORES"]))
        
        ativo_nao_circulante = obter_valor_recente(empresa_bpa, C["ATIVO_NAO_CIRCULANTE"])
        capital_empregado = ncg + ativo_nao_circulante

        if capital_empregado <= 0: return None

        roic = nopat_recente / capital_empregado
        beta = calcular_beta(ticker_sa, market_data["ibov_data"])
        ke = market_data["risk_free_rate"] + beta * market_data["premio_risco_mercado"]

        divida_total = (obter_valor_recente(empresa_bpp, C["DIVIDA_CURTO_PRAZO"]) + 
                        obter_valor_recente(empresa_bpp, C["DIVIDA_LONGO_PRAZO"]))
        despesa_financeira = abs(obter_valor_recente(empresa_dre, C["DESPESAS_FINANCEIRAS"]))
        
        if divida_total > 0 and despesa_financeira > 0:
            kd_calculado = despesa_financeira / divida_total
            kd = min(kd_calculado, 0.35) 
        else:
            kd = ke * 0.7

        valor_total = market_cap + divida_total
        if valor_total <= 0: return None
            
        w_e = market_cap / valor_total
        w_d = divida_total / valor_total
        wacc = (w_e * ke) + (w_d * kd * (1 - aliquota_efetiva))

        g = market_data["cresc_perpetuo"]
        if wacc <= g: return None

        eva = (roic - wacc) * capital_empregado
        valor_firma = capital_empregado + (eva * (1 + g)) / (wacc - g)
        
        divida_liquida = divida_total - obter_valor_recente(empresa_bpa, C["CAIXA_EQUIVALENTES"])
        equity_value = valor_firma - divida_liquida
        preco_justo = equity_value / n_acoes
        
        upside = (preco_justo / preco_atual) - 1 if preco_atual > 0 else 0

        eva_percent = roic - wacc
        riqueza_atual = eva / wacc if wacc > 0 else 0.0
        riqueza_futura_esperada = market_cap - capital_empregado
        efv = riqueza_futura_esperada - riqueza_atual
        efv_percent = efv / market_cap if market_cap > 0 else 0.0

        return {
            'Nome': info.get('shortName', ticker_sa.replace('.SA', ''))[:30], 
            'Ticker': ticker_sa.replace('.SA', ''),
            'Upside': upside, 'ROIC': roic, 'WACC': wacc, 'Spread': roic - wacc,
            'EVA_percent': eva_percent, 'EFV_percent': efv_percent,
            'Preco_Atual': preco_atual, 'Preco_Justo': preco_justo,
            'Market_Cap': market_cap, 'EVA': eva,
            'Capital_Empregado': capital_empregado, 'NOPAT': nopat_recente
        }
    except Exception:
        return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/market_info")
def market_info():
    """Endpoint para fornecer as premissas de mercado usadas nos cálculos."""
    try:
        dados = obter_dados_mercado()
        premissas = {
            "Taxa Livre de Risco (Selic)": f"{dados.get('risk_free_rate', 0):.2%}",
            "Inflação (IPCA)": dados.get("ipca_mes", "N/A"),
            "Câmbio (Dólar Comercial)": dados.get("cambio_dolar", "N/A"),
            "Prêmio de Risco de Mercado": f"{dados.get('premio_risco_mercado', 0):.2%}",
            "Crescimento Perpétuo (g)": f"{dados.get('cresc_perpetuo', 0):.2%}"
        }
        return jsonify(premissas)
    except Exception as e:
        return jsonify({"error": f"Não foi possível carregar as premissas de mercado: {e}"}), 500

@app.route("/run_analysis")
def run_analysis():
    """Rota principal que dispara a análise de todas as empresas."""
    logging.info(">>>>>> ANÁLISE INICIADA <<<<<<")
    carregar_mapeamento_ticker_cvm.cache_clear()
    carregar_dados_preparados.cache_clear()
    
    demonstrativos, error_msg = carregar_dados_preparados()
    if error_msg: return jsonify({"error": f"Erro ao carregar dados preparados: {error_msg}"}), 500
        
    ticker_map, error_msg = carregar_mapeamento_ticker_cvm()
    if error_msg: return jsonify({"error": f"Falha ao carregar mapeamento de tickers: {error_msg}"}), 500
        
    market_data = obter_dados_mercado()
    
    resultados_brutos = []
    empresas_excluidas = ['ITUB4', 'BBDC4', 'BBAS3', 'SANB11', 'B3SA3']
    
    for _, row in ticker_map.drop_duplicates(subset=['TICKER']).iterrows():
        ticker = row['TICKER']
        if ticker in empresas_excluidas: continue
        
        resultado = processar_valuation_empresa(f"{ticker.upper()}.SA", row['CD_CVM'], demonstrativos, market_data)
        if resultado: resultados_brutos.append(resultado)
    
    resultados_filtrados = []
    for r in resultados_brutos:
        if r is None: continue
        wacc_ok = 0.01 < r.get('WACC', 1) < 0.40
        upside_ok = -0.99 < r.get('Upside', 0) < 10.0
        if wacc_ok and upside_ok:
            resultados_filtrados.append(r)
        else:
            logging.warning(f"Filtrando empresa {r['Ticker']} por resultados extremos: WACC={r.get('WACC', 'N/A'):.2%}, Upside={r.get('Upside', 'N/A'):.2%}")

    total_calculado = len(resultados_brutos)
    total_filtrado = len(resultados_filtrados)
    logging.info(f">>>>>> ANÁLISE CONCLUÍDA: {total_filtrado} de {total_calculado} empresas passaram no filtro. <<<<<<")
    
    return jsonify(resultados_filtrados)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)