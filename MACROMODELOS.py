
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.hierarchical_risk_parity import HRPOpt
from io import BytesIO

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Gestor Inteligente", layout="wide")
st.title("Gestor Inteligente: Otimizador de Carteiras com Valuation e Macro")

# -------------------- INPUTS --------------------
st.sidebar.header("Configurações")
uso_macro = st.sidebar.checkbox("Usar Score Macro + Valuation", value=True)
uso_otimizacao = st.sidebar.checkbox("Usar Otimização Quantitativa (Sharpe, HRP)", value=True)
aporte = st.sidebar.number_input("Valor do novo aporte (R$)", min_value=0.0, value=1000.0, step=100.0)

# -------------------- DADOS DA CARTEIRA ATUAL --------------------
tickers = ['AGRO3.SA','BBAS3.SA','BBSE3.SA','BPAC11.SA','EGIE3.SA','ITUB3.SA','PRIO3.SA','PSSA3.SA',
           'SAPR4.SA','SBSP3.SA','VIVT3.SA','WEGE3.SA','TOTS3.SA','B3SA3.SA','TAEE3.SA','CMIG3.SA']
pesos_atuais = np.array([0.07, 0.05, 0.13, 0.06, 0.07, 0.07, 0.11, 0.08, 0.05, 0.03,
                         0.05, 0.15, 0.03, 0.01, 0.05, 0.00])

# -------------------- DOWNLOAD DE PREÇOS --------------------
st.subheader("Histórico de Preços")
data_inicio = datetime.today() - timedelta(days=365*5)
precos = yf.download(tickers, start=data_inicio)["Adj Close"].dropna(how="all", axis=1)
st.line_chart(precos)

# -------------------- RETORNOS E RISCO --------------------
st.subheader("Análise Quantitativa")
retornos = mean_historical_return(precos)
cov = CovarianceShrinkage(precos).ledoit_wolf()

# Otimizações
pesos_sugeridos = pd.Series(index=tickers, data=pesos_atuais)

if uso_otimizacao:
    ef = EfficientFrontier(retornos, cov)
    sharpe_pesos = ef.max_sharpe()
    pesos_sharpe = pd.Series(sharpe_pesos).drop("Expected Return", errors='ignore')
    hrp = HRPOpt(precos.pct_change().dropna())
    pesos_hrp = hrp.optimize()
    pesos_otimizados = (pesos_sharpe + pesos_hrp) / 2
    pesos_sugeridos = pesos_otimizados if not uso_macro else (pesos_otimizados + pesos_atuais) / 2

# -------------------- PROBABILIDADE DE ALTA --------------------
st.subheader("Probabilidade Histórica de Alta")
retornos_mensais = precos.resample('M').last().pct_change().dropna()
prob_alta = (retornos_mensais > 0).sum() / len(retornos_mensais)
st.dataframe(prob_alta.sort_values(ascending=False).rename("Probabilidade de Alta"))

# -------------------- RECOMENDAÇÃO DE COMPRA --------------------
st.subheader("Sugestão de Compra com Aporte")
valores_atuais = pesos_atuais * 100000  # Supondo R$ 100.000 de carteira atual
valores_futuros = valores_atuais + pesos_sugeridos * aporte
alocacao_final = valores_futuros / valores_futuros.sum()

df_recomendacao = pd.DataFrame({
    "Ticker": tickers,
    "Peso Atual": pesos_atuais,
    "Peso Sugerido": pesos_sugeridos,
    "Alocação Final (%)": alocacao_final
})
df_recomendacao["Compra Recomendada (R$)"] = alocacao_final * (aporte + valores_atuais.sum()) - valores_atuais

st.dataframe(df_recomendacao.set_index("Ticker"))

# -------------------- GRÁFICOS --------------------
st.subheader("Distribuição da Carteira")
col1, col2 = st.columns(2)
with col1:
    st.caption("Antes do Aporte")
    st.pyplot(plt.pie(pesos_atuais, labels=tickers, autopct='%1.1f%%')[0].figure)
with col2:
    st.caption("Depois do Aporte")
    st.pyplot(plt.pie(alocacao_final, labels=tickers, autopct='%1.1f%%')[0].figure)

# -------------------- EXPORTAÇÃO --------------------
st.subheader("Exportar para Excel")
excel = BytesIO()
df_recomendacao.to_excel(excel, index=False)
st.download_button("Baixar Excel", data=excel.getvalue(), file_name="recomendacao_carteira.xlsx")
