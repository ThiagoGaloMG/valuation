#!/usr/bin/env python3
"""
Script Aprimorado de Atualização de Dados da CVM (Otimizado para Cota de Disco)

Esta versão foi otimizada para ambientes com limitação de disco, como o PythonAnywhere.
Ela processa um histórico de 2 anos para garantir que os arquivos finais não excedam a cota.

Lógica:
1. Carrega o mapeamento de tickers.
2. Para cada tipo de demonstrativo (DRE, BPA, etc.):
   a. Cria um arquivo CSV final vazio, apenas com o cabeçalho.
   b. Itera sobre cada ano, processando um de cada vez.
   c. Dentro do ano, combina os dados CONSOLIDADOS e INDIVIDUAIS.
   d. Anexa o resultado do ano ao arquivo CSV final.
3. Ao final de tudo, apaga a pasta CVM_DATA para liberar espaço.
"""
import pandas as pd
from pathlib import Path
import logging
import time
import sys
from zipfile import ZipFile, BadZipFile
from datetime import datetime
import shutil

# === CONFIGURAÇÃO ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = Path(__file__).resolve().parent
CAMINHO_MAPA_TICKER_CVM = BASE_DIR / "mapeamento_tickers.csv"
DIRETORIO_DADOS_CVM = BASE_DIR / "CVM_DATA"
DIRETORIO_DADOS_CONSOLIDADOS = BASE_DIR / "consolidated_data"
# AJUSTE FINAL: Reduzido para 2 anos para garantir a execução no PythonAnywhere.
HISTORICO_ANOS_CVM = 2 

def carregar_mapeamento_robusto(caminho_arquivo):
    """Carrega o arquivo de mapeamento de tickers de forma robusta."""
    logging.info("Carregando mapeamento de tickers...")
    if not caminho_arquivo.exists():
        logging.error(f"ERRO CRÍTICO: Arquivo de mapeamento '{caminho_arquivo.name}' não encontrado.")
        return None
    try:
        df_map = pd.read_csv(caminho_arquivo, sep=';', encoding='latin-1', dtype=str)
        if len(df_map.columns) <= 1:
            df_map = pd.read_csv(caminho_arquivo, sep=',', encoding='latin-1', dtype=str)
        
        df_map.columns = [str(col).strip().upper() for col in df_map.columns]
        
        if 'CD_CVM' not in df_map.columns:
            raise ValueError("Coluna 'CD_CVM' não encontrada no arquivo de mapeamento.")

        df_map.dropna(subset=['CD_CVM'], inplace=True)
        codigos_cvm = pd.to_numeric(df_map['CD_CVM'], errors='coerce').dropna().astype(int).unique()
        
        codigos_cvm_set = set(codigos_cvm)
        logging.info(f"✓ Mapeamento carregado: {len(codigos_cvm_set)} códigos CVM únicos a processar.")
        return codigos_cvm_set
    except Exception as e:
        logging.error(f"ERRO CRÍTICO ao carregar o arquivo de mapeamento: {e}")
        return None

def processar_e_anexar_por_tipo(tipo_demonstrativo, cvm_codes_filtrar, anos_a_processar):
    """
    Processa os dados ano a ano e anexa a um arquivo CSV final para economizar disco.
    """
    logging.info(f"\n=== PROCESSANDO TIPO: {tipo_demonstrativo.upper()} ===")
    caminho_salvar = DIRETORIO_DADOS_CONSOLIDADOS / f"{tipo_demonstrativo.lower()}_consolidado.csv"
    
    # Apaga o arquivo antigo para garantir que estamos começando do zero
    if caminho_salvar.exists():
        caminho_salvar.unlink()

    cabecalho_escrito = False

    for ano in anos_a_processar:
        nome_zip = f'dfp_cia_aberta_{ano}.zip'
        caminho_zip = DIRETORIO_DADOS_CVM / nome_zip
        
        if not caminho_zip.exists():
            logging.warning(f"--- Pulando ano {ano}: Arquivo '{nome_zip}' não encontrado. ---")
            continue
            
        logging.info(f"--- Processando ano {ano} do arquivo '{nome_zip}' ---")
        
        df_con_ano = pd.DataFrame()
        df_ind_ano = pd.DataFrame()

        try:
            with ZipFile(caminho_zip, 'r') as z:
                # Processa dados consolidados do ano
                nome_csv_con = f'dfp_cia_aberta_{tipo_demonstrativo.upper()}_con_{ano}.csv'
                if nome_csv_con in z.namelist():
                    with z.open(nome_csv_con) as csv_file:
                        df_con_ano = pd.read_csv(csv_file, sep=';', encoding='latin-1', low_memory=False, dtype=str)
                
                # Processa dados individuais do ano
                nome_csv_ind = f'dfp_cia_aberta_{tipo_demonstrativo.upper()}_ind_{ano}.csv'
                if nome_csv_ind in z.namelist():
                    with z.open(nome_csv_ind) as csv_file:
                        df_ind_ano = pd.read_csv(csv_file, sep=';', encoding='latin-1', low_memory=False, dtype=str)

        except Exception as e:
            logging.error(f"  ✗ ERRO inesperado ao processar {nome_zip}: {e}")
            continue

        if df_con_ano.empty and df_ind_ano.empty:
            logging.warning(f"  -> Nenhum dado (con ou ind) encontrado para {tipo_demonstrativo.upper()} em {ano}.")
            continue

        if not df_con_ano.empty:
            df_con_ano['CD_CVM'] = pd.to_numeric(df_con_ano['CD_CVM'], errors='coerce').dropna().astype(int)
            df_con_ano = df_con_ano[df_con_ano['CD_CVM'].isin(cvm_codes_filtrar)].copy()
        
        if not df_ind_ano.empty:
            df_ind_ano['CD_CVM'] = pd.to_numeric(df_ind_ano['CD_CVM'], errors='coerce').dropna().astype(int)
            df_ind_ano = df_ind_ano[df_ind_ano['CD_CVM'].isin(cvm_codes_filtrar)].copy()

        cvm_em_con = set(df_con_ano['CD_CVM'].unique()) if not df_con_ano.empty else set()
        df_ind_fallback = df_ind_ano[~df_ind_ano['CD_CVM'].isin(cvm_em_con)] if not df_ind_ano.empty else pd.DataFrame()
        
        df_final_ano = pd.concat([df_con_ano, df_ind_fallback], ignore_index=True)

        if df_final_ano.empty:
            continue

        try:
            mode = 'a' if cabecalho_escrito else 'w'
            header = not cabecalho_escrito
            df_final_ano.to_csv(caminho_salvar, index=False, encoding='utf-8', sep=',', mode=mode, header=header)
            if not cabecalho_escrito:
                cabecalho_escrito = True
            
            logging.info(f"  -> Dados de {ano} para {tipo_demonstrativo.upper()} anexados com sucesso.")

        except Exception as e:
            logging.error(f"✗ ERRO CRÍTICO ao anexar dados para {tipo_demonstrativo.lower()}: {e}")
            return False

    if not cabecalho_escrito:
        logging.warning(f"✗ Nenhum dado foi processado ou salvo para {tipo_demonstrativo.upper()}.")
        pd.DataFrame().to_csv(caminho_salvar, index=False)
        return False

    logging.info(f"✓ Arquivo final para {tipo_demonstrativo.upper()} gerado com sucesso em '{caminho_salvar.name}'.")
    return True

def main():
    print(f"CVM DATA UPDATER (Otimizado para {HISTORICO_ANOS_CVM} Anos de Histórico)")
    print("=" * 60)

    cvm_codes_filtrar = carregar_mapeamento_robusto(CAMINHO_MAPA_TICKER_CVM)
    if cvm_codes_filtrar is None:
        return False

    DIRETORIO_DADOS_CONSOLIDADOS.mkdir(exist_ok=True)

    ano_atual = datetime.today().year
    anos_a_processar = range(ano_atual - HISTORICO_ANOS_CVM, ano_atual + 1)
    print(f"\n1. Processando arquivos ZIP locais para o período de {min(anos_a_processar)} a {max(anos_a_processar)}...")

    if not DIRETORIO_DADOS_CVM.exists():
        logging.error(f"ERRO: O diretório '{DIRETORIO_DADOS_CVM.name}' não foi encontrado. Crie-o e coloque os arquivos .zip dentro dele.")
        return False

    tipos_demonstrativos = ['DRE', 'BPA', 'BPP', 'DFC_MI']
    sucessos = 0
    for tipo in tipos_demonstrativos:
        if processar_e_anexar_por_tipo(tipo, cvm_codes_filtrar, anos_a_processar):
            sucessos += 1
    
    print("\n2. Limpando arquivos ZIP temporários...")
    try:
        shutil.rmtree(DIRETORIO_DADOS_CVM)
        print(f"✓ Diretório de dados temporários '{DIRETORIO_DADOS_CVM.name}' removido com sucesso.")
    except Exception as e:
        print(f"✗ Erro ao limpar diretório temporário: {e}")

    print("\n" + "=" * 60)
    if sucessos == len(tipos_demonstrativos):
        print(f"✓ ATUALIZAÇÃO CONCLUÍDA COM SUCESSO! Todos os {len(tipos_demonstrativos)} arquivos foram gerados.")
    else:
        print(f"✗ ATENÇÃO: {sucessos} de {len(tipos_demonstrativos)} arquivos foram gerados. Verifique os logs de erro.")
    
    return sucessos > 0

if __name__ == "__main__":
    inicio = time.time()
    sucesso = main()
    fim = time.time()
    print(f"\nTempo total de execução: {fim - inicio:.2f} segundos.")
    sys.exit(0 if sucesso else 1)