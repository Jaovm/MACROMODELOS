import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier

# --- Funções auxiliares ---
def obter_preco_fmp(ticker, data_inicio, api_key):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={data_inicio}&apikey={api_key}"
    r = requests.get(url)
    dados = r.json()
    if "historical" in dados:
        df = pd.DataFrame(dados["historical"])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df['close'].sort_index().rename(ticker)
    return pd.Series(name=ticker)

def carregar_precos(tickers, data_inicio, api_key):
    precos = pd.concat(
        [obter_preco_fmp(ticker, data_inicio, api_key) for ticker in tickers],
        axis=1
    )
    return precos.dropna(how='all')

def calcular_probabilidades_altas(precos):
    retornos_diarios = precos.pct_change().dropna()
    probabilidades = (retornos_diarios > 0).sum() / len(retornos_diarios)
    return probabilidades.round(2)

# --- Configuração inicial ---
st.title("Gestor Inteligente - Carteira Otimizada com Valuation e Cenário Econômico")

tickers = ["AGRO3.SA", "BBAS3.SA", "BBSE3.SA", "BPAC11.SA", "EGIE3.SA",
           "ITUB3.SA", "PRIO3.SA", "PSSA3.SA", "SAPR4.SA", "SBSP3.SA",
           "VIVT3.SA", "WEGE3.SA", "TOTS3.SA", "B3SA3.SA", "TAEE3.SA", "CMIG3.SA"]

pesos_atuais = np.array([0.07, 0.05, 0.13, 0.06, 0.07, 0.07, 0.11, 0.08, 0.05, 0.03, 0.05, 0.15, 0.03, 0.01, 0.05, 0.0])
pesos_atuais_dict = dict(zip(tickers, pesos_atuais))

api_key = st.secrets["FMP_API_KEY"] if "FMP_API_KEY" in st.secrets else st.text_input("Informe sua API KEY do FMP", type="password")
usar_sugestao = st.checkbox("Deseja receber sugestão de compra para novo aporte?", value=True)

if api_key:
    data_inicio = "2020-01-01"
    precos = carregar_precos(tickers, data_inicio, api_key)

    # --- Otimização com Markowitz ---
    retornos = mean_historical_return(precos)
    cov = CovarianceShrinkage(precos).ledoit_wolf()
    ef = EfficientFrontier(retornos, cov)
    pesos_otimizados = ef.max_sharpe()
    pesos_limpos = ef.clean_weights()

    # --- Probabilidades de alta histórica ---
    probabilidades = calcular_probabilidades_altas(precos)

    # --- Tabela de comparação ---
    df_resultado = pd.DataFrame({
        "Peso Atual": pd.Series(pesos_atuais_dict),
        "Peso Otimizado": pd.Series(pesos_limpos),
        "Probabilidade de Alta": probabilidades
    }).fillna(0)

    df_resultado["Delta Aporte (%)"] = ((df_resultado["Peso Otimizado"] - df_resultado["Peso Atual"]).clip(lower=0) * 100).round(2)
    
    st.subheader("Resumo da Carteira")
    st.dataframe(df_resultado.style.format("{:.2%}"))

    if usar_sugestao:
        recomendados = df_resultado[df_resultado["Delta Aporte (%)"] > 0].sort_values("Delta Aporte (%)", ascending=False)
        st.subheader("Recomendações de Compra (sem vendas)")
        st.table(recomendados[["Delta Aporte (%)", "Probabilidade de Alta"]])
else:
    st.warning("Insira sua API Key para continuar.")
