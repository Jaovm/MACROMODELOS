import streamlit as st
import requests
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime

api_key = "rd6uBzkLLSPG68s9GcSx3folN76IxRhV"

# ---------- Funções utilitárias ---------- #

def get_stock_data(tickers, data_inicio):
    tickers_str = ",".join(tickers)
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{tickers_str}?from={data_inicio}&apikey={api_key}&serietype=line"

    response = requests.get(url)
    data = response.json()
    
    historico = {}

    if isinstance(data, dict) and 'historicalStockList' in data:
        for item in data['historicalStockList']:
            symbol = item['symbol']
            if 'historical' in item:
                dates = [entry['date'] for entry in item['historical']]
                adj_close = [entry['adjClose'] for entry in item['historical']]
                historico[symbol] = pd.Series(data=adj_close, index=pd.to_datetime(dates))
    elif isinstance(data, dict) and 'symbol' in data:
        symbol = data['symbol']
        dates = [entry['date'] for entry in data['historical']]
        adj_close = [entry['adjClose'] for entry in data['historical']]
        historico[symbol] = pd.Series(data=adj_close, index=pd.to_datetime(dates))
    else:
        raise ValueError("Formato inesperado na resposta da API")

    precos = pd.DataFrame(historico).sort_index()
    return precos.dropna(how='all', axis=1)

def calcular_probabilidade_alta(precos):
    retornos = precos.pct_change().dropna()
    media_anual = retornos.mean() * 252
    desvio_anual = retornos.std() * np.sqrt(252)

    # Probabilidade de retorno > 0 usando distribuição normal padrão
    prob = 1 - (0.5 * (1 - (media_anual / desvio_anual)))
    return prob.clip(0, 1)

def fronteira_eficiente(retornos, n_portfolios=5000, risco_livre=0.0):
    num_assets = retornos.shape[1]
    resultados = {'retorno': [], 'volatilidade': [], 'sharpe': [], 'pesos': []}
    
    media_retornos = retornos.mean() * 252
    cov_matrix = retornos.cov() * 252

    for _ in range(n_portfolios):
        pesos = np.random.dirichlet(np.ones(num_assets))
        ret_esperado = np.dot(pesos, media_retornos)
        volatilidade = np.sqrt(np.dot(pesos.T, np.dot(cov_matrix, pesos)))
        sharpe = (ret_esperado - risco_livre) / volatilidade
        resultados['retorno'].append(ret_esperado)
        resultados['volatilidade'].append(volatilidade)
        resultados['sharpe'].append(sharpe)
        resultados['pesos'].append(pesos)

    df_resultados = pd.DataFrame(resultados)
    return df_resultados

def melhor_portfolio(df_resultados, tickers):
    idx_max_sharpe = df_resultados['sharpe'].idxmax()
    melhor = df_resultados.iloc[idx_max_sharpe]
    return pd.Series(data=melhor['pesos'], index=tickers, name='Peso ótimo')

def sugerir_compras(carteira_atual, prob_alta, limite_max=0.2):
    # Aumentar peso de ativos com alta probabilidade de valorização
    pesos = prob_alta / prob_alta.sum()
    pesos = pesos.clip(upper=limite_max)
    pesos = pesos / pesos.sum()  # Normalizar

    return pesos.rename('Sugestão de Compra')

# ---------- App principal ---------- #

st.title("Modelo Quantitativo de Carteira com API da FMP")

tickers = st.text_input("Tickers separados por vírgula (ex: AAPL,MSFT,GOOG)", value="AAPL,MSFT,GOOG").split(",")
tickers = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
data_inicio = "2018-01-01"

try:
    precos = get_stock_data(tickers, data_inicio)
    st.success("Dados carregados com sucesso.")
    
    retornos = precos.pct_change().dropna()
    
    # Probabilidade de alta
    prob_alta = calcular_probabilidade_alta(precos)
    st.subheader("Probabilidade de Alta em 1 ano")
    st.dataframe(prob_alta.sort_values(ascending=False).to_frame(style="color: green;"))

    # Fronteira eficiente
    st.subheader("Carteira na Fronteira Eficiente")
    df_fronteira = fronteira_eficiente(retornos)
    pesos_otimos = melhor_portfolio(df_fronteira, retornos.columns)
    st.dataframe(pesos_otimos.to_frame(style="color: green;"))

    # Sugestão de compra com base no cenário macro (aqui simplificada pela probabilidade)
    st.subheader("Sugestão de Alocação de Aporte")
    sugestao = sugerir_compras(pesos_otimos, prob_alta)
    st.dataframe(sugestao.to_frame(style="color: blue;"))

except Exception as e:
    st.error(f"Erro ao carregar dados ou calcular: {e}")
