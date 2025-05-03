import streamlit as st
import pandas as pd
import requests

st.set_page_config(layout="wide", page_title="Alocação Inteligente de Ações")

# ------------------------------ CONFIGURAÇÕES INICIAIS ------------------------------

st.title("Gestão de Carteira Baseada em Macroeconomia e Valuation")
api_key = st.text_input("Insira sua API Key do Financial Modeling Prep (FMP)", type="password")
if not api_key:
    st.stop()

setores_por_ticker = {
    "AGRO3.SA": "Agronegócio",
    "BBAS3.SA": "Bancos",
    "BBSE3.SA": "Seguradoras",
    "BPAC11.SA": "Bancos",
    "EGIE3.SA": "Energia Elétrica",
    "ITUB3.SA": "Bancos",
    "PRIO3.SA": "Petróleo, Gás e Biocombustíveis",
    "PSSA3.SA": "Seguradoras",
    "SAPR4.SA": "Utilidades Públicas",
    "SBSP3.SA": "Utilidades Públicas",
    "VIVT3.SA": "Comunicação",
    "WEGE3.SA": "Indústria e Bens de Capital",
    "TOTS3.SA": "Tecnologia",
    "B3SA3.SA": "Bolsas e Serviços Financeiros",
    "TAEE3.SA": "Energia Elétrica",
    "CMIG3.SA": "Energia Elétrica"
}

setores_por_cenario = {
    "Expansionista": ["Consumo Discricionário", "Tecnologia", "Indústria e Bens de Capital", "Agronegócio"],
    "Neutro": ["Saúde", "Bancos", "Seguradoras", "Bolsas e Serviços Financeiros", "Utilidades Públicas"],
    "Restritivo": ["Energia Elétrica", "Petróleo, Gás e Biocombustíveis", "Mineração e Siderurgia", "Consumo Básico", "Comunicação"]
}

# ------------------------------ FUNÇÕES AUXILIARES ------------------------------

@st.cache_data
def classificar_cenario_macro():
    try:
        df = pd.read_csv("https://www3.bcb.gov.br/expectativas2/estatisticas/consultaExpectativas.csv", sep=";")
        df = df[df["Indicador"] == "Selic"]  # Foco na Selic
        proj = float(df[df["Data"] == df["Data"].max()]["Mediana"].values[0])
        if proj <= 9:
            return "Expansionista"
        elif proj <= 11:
            return "Neutro"
        else:
            return "Restritivo"
    except:
        return "Neutro"

def get_macro_scores(cenario):
    score_setorial = {}
    for ticker, setor in setores_por_ticker.items():
        if setor in setores_por_cenario[cenario]:
            score_setorial[ticker] = 3
        elif any(setor in v for k,v in setores_por_cenario.items() if k != cenario):
            score_setorial[ticker] = 2
        else:
            score_setorial[ticker] = 1
    return score_setorial

@st.cache_data(show_spinner=False)
def get_fmp_data(ticker, api_key):
    base = f"https://financialmodelingprep.com/api/v3/"
    endpoints = {
        "ratios": f"{base}ratios-ttm/{ticker}?apikey={api_key}",
        "quote": f"{base}quote/{ticker}?apikey={api_key}",
        "dcf": f"{base}discounted-cash-flow/{ticker}?apikey={api_key}"
    }
    try:
        ratios = requests.get(endpoints["ratios"]).json()[0]
        quote = requests.get(endpoints["quote"]).json()[0]
        dcf = requests.get(endpoints["dcf"]).json()[0]
        return {
            "P/L": ratios.get("peRatioTTM"),
            "ROE": ratios.get("roeTTM"),
            "DY": ratios.get("dividendYieldTTM"),
            "FairValue": dcf.get("dcf"),
            "Price": quote.get("price")
        }
    except:
        return None

@st.cache_data(show_spinner=True)
def get_valorations_scores(api_key):
    scores = {}
    for ticker in setores_por_ticker.keys():
        dados = get_fmp_data(ticker.replace('.SA', ''), api_key)
        if not dados:
            scores[ticker] = 2
            continue

        score = 0
        if dados["P/L"] and dados["P/L"] < 15:
            score += 1
        if dados["ROE"] and dados["ROE"] > 0.15:
            score += 1
        if dados["DY"] and dados["DY"] > 0.04:
            score += 1
        if dados["FairValue"] and dados["Price"] and dados["FairValue"] > dados["Price"]:
            score += 1
        scores[ticker] = score
    return scores

def sugerir_alocacao(pesos_macro, pesos_val):
    df = pd.DataFrame({
        "ScoreMacro": pesos_macro,
        "ScoreValuation": pesos_val
    })
    df["ScoreTotal"] = df["ScoreMacro"] * 0.6 + df["ScoreValuation"] * 0.4
    df["AlocacaoSugerida"] = df["ScoreTotal"] / df["ScoreTotal"].sum()
    return df

# ------------------------------ EXECUÇÃO ------------------------------

st.subheader("1. Cenário Macroeconômico")
modo = st.radio("Selecione o modo de classificação do cenário", ["Automático", "Manual"])
if modo == "Automático":
    cenario = classificar_cenario_macro()
else:
    cenario = st.selectbox("Escolha o cenário", ["Expansionista", "Neutro", "Restritivo"])

st.success(f"Cenário selecionado: **{cenario}**")

# Scores
with st.spinner("Calculando pontuações macro e de valuation..."):
    macro_scores = get_macro_scores(cenario)
    valuation_scores = get_valorations_scores(api_key)
    resultado = sugerir_alocacao(macro_scores, valuation_scores)

# ------------------------------ SAÍDA ------------------------------
st.subheader("2. Alocação sugerida com base no cenário + valuation")
st.dataframe(resultado[["ScoreMacro", "ScoreValuation", "AlocacaoSugerida"]].style.format({
    "AlocacaoSugerida": "{:.1%}"
}), use_container_width=True)

st.caption("Os scores vão de 0 a 4 e combinam os setores favorecidos pelo cenário com fundamentos da empresa.")
