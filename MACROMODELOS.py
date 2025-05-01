import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import datetime
import os
import time
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.optimize import minimize

st.set_page_config(page_title="Sugestão de Carteira", layout="wide")

def get_bcb_hist(code, start, end):
    """Baixa série histórica mensal do BCB para um código SGS."""
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados?formato=json&dataInicial={start}&dataFinal={end}"
    r = requests.get(url)
    if r.status_code == 200:
        df = pd.DataFrame(r.json())
        df['data'] = pd.to_datetime(df['data'], dayfirst=True)
        df['valor'] = df['valor'].str.replace(",", ".").astype(float)
        return df.set_index('data')['valor']
    else:
        return pd.Series(dtype=float)

def obter_preco_petroleo_hist(start, end):
    """Baixa preço histórico mensal do petróleo Brent (BZ=F) do Yahoo Finance."""
    df = yf.download("BZ=F", start=start, end=end, interval="1mo", progress=False)
    if not df.empty:
        df.index = pd.to_datetime(df.index)
        return df['Close']
    return pd.Series(dtype=float)

def montar_historico_7anos(tickers, setores_por_ticker, start='2018-01-01'):
    """Adiciona delay entre requisições no loop."""
    hoje = datetime.date.today()
    inicio = pd.to_datetime(start)
    final = hoje
    datas = pd.date_range(inicio, final, freq='M').normalize()
    
    # Baixar séries macro históricas do BCB
    selic_hist = get_bcb_hist(432, inicio.strftime('%d/%m/%Y'), final.strftime('%d/%m/%Y'))
    ipca_hist = get_bcb_hist(433, inicio.strftime('%d/%m/%Y'), final.strftime('%d/%m/%Y'))
    dolar_hist = get_bcb_hist(1, inicio.strftime('%d/%m/%Y'), final.strftime('%d/%m/%Y'))
    petroleo_hist = obter_preco_petroleo_hist(inicio.strftime('%Y-%m-%d'), final.strftime('%Y-%m-%d'))
    
    # Normalizar todos os índices para garantir compatibilidade
    selic_hist.index = pd.to_datetime(selic_hist.index).normalize()
    ipca_hist.index = pd.to_datetime(ipca_hist.index).normalize()
    dolar_hist.index = pd.to_datetime(dolar_hist.index).normalize()
    petroleo_hist.index = pd.to_datetime(petroleo_hist.index).normalize()
    
    macro_df = pd.DataFrame(index=datas)
    macro_df['selic'] = selic_hist.reindex(datas, method='ffill')
    macro_df['ipca'] = ipca_hist.reindex(datas, method='ffill')
    macro_df['dolar'] = dolar_hist.reindex(datas, method='ffill')
    macro_df['petroleo'] = petroleo_hist.reindex(datas, method='ffill')
    macro_df = macro_df.ffill().bfill()  # Atualização para o pandas moderno

    historico = []
    for data in datas:
        time.sleep(3)  # Adiciona um atraso de 1 segundo entre as iterações do loop
        macro = {
            "ipca": macro_df.loc[data, "ipca"],
            "selic": macro_df.loc[data, "selic"],
            "dolar": macro_df.loc[data, "dolar"],
            "pib": 2,
            "petroleo": macro_df.loc[data, "petroleo"],
            "soja": None,
            "milho": None,
            "minerio": None
        }
        cenario = classificar_cenario_macro(
            ipca=macro["ipca"],
            selic=macro["selic"],
            dolar=macro["dolar"],
            pib=macro["pib"],
            preco_soja=macro["soja"],
            preco_milho=macro["milho"],
            preco_minerio=macro["minerio"],
            preco_petroleo=macro["petroleo"]
        )
        score_macro = pontuar_macro(macro)
        for ticker in tickers:
            setor = setores_por_ticker.get(ticker, None)
            favorecido = calcular_favorecimento_continuo(setor, score_macro)
            historico.append({
                "data": str(data.date()),
                "cenario": cenario,
                "ticker": ticker,
                "setor": setor,
                "favorecido": favorecido
            })
    df_hist = pd.DataFrame(historico)
    return df_hist

# ========= DICIONÁRIOS ==========

setores_por_ticker = {
    # Bancos
    'ITUB4.SA': 'Bancos',
    'BBDC4.SA': 'Bancos',
    'SANB11.SA': 'Bancos',
    'BBAS3.SA': 'Bancos',
    'ABCB4.SA': 'Bancos',
    'BRSR6.SA': 'Bancos',
    'BMGB4.SA': 'Bancos',
    'BPAC11.SA': 'Bancos',
    'ITSA4.SA': 'Bancos',

    # Seguradoras
    'BBSE3.SA': 'Seguradoras',
    'PSSA3.SA': 'Seguradoras',
    'SULA11.SA': 'Seguradoras',
    'CXSE3.SA': 'Seguradoras',

    # Bolsas e Serviços Financeiros
    'B3SA3.SA': 'Bolsas e Serviços Financeiros',
    'XPBR31.SA': 'Bolsas e Serviços Financeiros',

    # Energia Elétrica
    'EGIE3.SA': 'Energia Elétrica',
    'CPLE6.SA': 'Energia Elétrica',
    'TAEE11.SA': 'Energia Elétrica',
    'CMIG4.SA': 'Energia Elétrica',
    'AURE3.SA': 'Energia Elétrica',
    'CPFE3.SA': 'Energia Elétrica',
    'AESB3.SA': 'Energia Elétrica',

    # Petróleo, Gás e Biocombustíveis
    'PETR4.SA': 'Petróleo, Gás e Biocombustíveis',
    'PRIO3.SA': 'Petróleo, Gás e Biocombustíveis',
    'RECV3.SA': 'Petróleo, Gás e Biocombustíveis',
    'RRRP3.SA': 'Petróleo, Gás e Biocombustíveis',
    'UGPA3.SA': 'Petróleo, Gás e Biocombustíveis',
    'VBBR3.SA': 'Petróleo, Gás e Biocombustíveis',

    # Mineração e Siderurgia
    'VALE3.SA': 'Mineração e Siderurgia',
    'CSNA3.SA': 'Mineração e Siderurgia',
    'GGBR4.SA': 'Mineração e Siderurgia',
    'CMIN3.SA': 'Mineração e Siderurgia',
    'GOAU4.SA': 'Mineração e Siderurgia',
    'BRAP4.SA': 'Mineração e Siderurgia',

    # Indústria e Bens de Capital
    'WEGE3.SA': 'Indústria e Bens de Capital',
    'RANI3.SA': 'Indústria e Bens de Capital',
    'KLBN11.SA': 'Indústria e Bens de Capital',
    'SUZB3.SA': 'Indústria e Bens de Capital',
    'UNIP6.SA': 'Indústria e Bens de Capital',
    'KEPL3.SA': 'Indústria e Bens de Capital',
    'TUPY3.SA': 'Indústria e Bens de Capital',

    # Agronegócio
    'AGRO3.SA': 'Agronegócio',
    'SLCE3.SA': 'Agronegócio',
    'SMTO3.SA': 'Agronegócio',
    'CAML3.SA': 'Agronegócio',

    # Saúde
    'HAPV3.SA': 'Saúde',
    'FLRY3.SA': 'Saúde',
    'RDOR3.SA': 'Saúde',
    'QUAL3.SA': 'Saúde',
    'RADL3.SA': 'Saúde',

    # Tecnologia
    'TOTS3.SA': 'Tecnologia',
    'POSI3.SA': 'Tecnologia',
    'LINX3.SA': 'Tecnologia',
    'LWSA3.SA': 'Tecnologia',

    # Consumo Discricionário
    'MGLU3.SA': 'Consumo Discricionário',
    'LREN3.SA': 'Consumo Discricionário',
    'RENT3.SA': 'Consumo Discricionário',
    'ARZZ3.SA': 'Consumo Discricionário',
    'ALPA4.SA': 'Consumo Discricionário',

    # Consumo Básico
    'ABEV3.SA': 'Consumo Básico',
    'NTCO3.SA': 'Consumo Básico',
    'PCAR3.SA': 'Consumo Básico',
    'MDIA3.SA': 'Consumo Básico',

    # Comunicação
    'VIVT3.SA': 'Comunicação',
    'TIMS3.SA': 'Comunicação',
    'OIBR3.SA': 'Comunicação',

    # Utilidades Públicas
    'SBSP3.SA': 'Utilidades Públicas',
    'SAPR11.SA': 'Utilidades Públicas',
    'SAPR3.SA': 'Utilidades Públicas',
    'SAPR4.SA': 'Utilidades Públicas',
    'CSMG3.SA': 'Utilidades Públicas',
    'ALUP11.SA': 'Utilidades Públicas',
    'CPLE6.SA': 'Utilidades Públicas',

    # Adicionando ativos novos conforme solicitado
    'CRFB3.SA': 'Consumo Discricionário',
    'COGN3.SA': 'Tecnologia',
    'OIBR3.SA': 'Comunicação',
    'CCRO3.SA': 'Utilidades Públicas',
    'BEEF3.SA': 'Consumo Discricionário',
    'AZUL4.SA': 'Consumo Discricionário',
    'POMO4.SA': 'Indústria e Bens de Capital',
    'RAIL3.SA': 'Indústria e Bens de Capital',
    'CVCB3.SA': 'Consumo Discricionário',
    'BRAV3.SA': 'Bancos',
    'PETR3.SA': 'Petróleo, Gás e Biocombustíveis',
    'VAMO3.SA': 'Consumo Discricionário',
    'CSAN3.SA': 'Energia Elétrica',
    'USIM5.SA': 'Mineração e Siderurgia',
    'RAIZ4.SA': 'Agronegócio',
    'ELET3.SA': 'Energia Elétrica',
    'CMIG4.SA': 'Energia Elétrica',
    'EQTL3.SA': 'Energia Elétrica',
    'ANIM3.SA': 'Saúde',
    'MRVE3.SA': 'Consumo Discricionário',
    'AMOB3.SA': 'Tecnologia',
    'RAPT4.SA': 'Consumo Discricionário',
    'CSNA3.SA': 'Mineração e Siderurgia',
    'RENT3.SA': 'Consumo Discricionário',
    'MRFG3.SA': 'Consumo Básico',
    'JBSS3.SA': 'Consumo Básico',
    'VBBR3.SA': 'Petróleo, Gás e Biocombustíveis',
    'BBDC3.SA': 'Bancos',
    'IFCM3.SA': 'Tecnologia',
    'BHIA3.SA': 'Bancos',
    'LWSA3.SA': 'Tecnologia',
    'SIMH3.SA': 'Saúde',
    'CMIN3.SA': 'Mineração e Siderurgia',
    'UGPA3.SA': 'Petróleo, Gás e Biocombustíveis',
    'MOVI3.SA': 'Consumo Discricionário',
    'GFSA3.SA': 'Consumo Discricionário',
    'AZEV4.SA': 'Saúde',
    'RADL3.SA': 'Saúde',
    'BPAC11.SA': 'Bancos',
    'PETZ3.SA': 'Saúde',
    'AURE3.SA': 'Energia Elétrica',
    'ENEV3.SA': 'Energia Elétrica',
    'WEGE3.SA': 'Indústria e Bens de Capital',
    'CPLE3.SA': 'Energia Elétrica',
    'SRNA3.SA': 'Indústria e Bens de Capital',
    'BRFS3.SA': 'Consumo Básico',
    'SLCE3.SA': 'Agronegócio',
    'CBAV3.SA': 'Consumo Básico',
    'ECOR3.SA': 'Tecnologia',
    'KLBN11.SA': 'Indústria e Bens de Capital',
    'EMBR3.SA': 'Indústria e Bens de Capital',
    'MULT3.SA': 'Bancos',
    'CYRE3.SA': 'Indústria e Bens de Capital',
    'RDOR3.SA': 'Saúde',
    'TIMS3.SA': 'Comunicação',
    'SUZB3.SA': 'Indústria e Bens de Capital',
    'ALOS3.SA': 'Saúde',
    'SMFT3.SA': 'Tecnologia',
    'FLRY3.SA': 'Saúde',
    'IGTI11.SA': 'Tecnologia',
    'AMER3.SA': 'Consumo Discricionário',
    'YDUQ3.SA': 'Tecnologia',
    'STBP3.SA': 'Bancos',
    'GMAT3.SA': 'Indústria e Bens de Capital',
    'TOTS3.SA': 'Tecnologia',
    'CEAB3.SA': 'Indústria e Bens de Capital',
    'EZTC3.SA': 'Consumo Discricionário',
    'BRAP4.SA': 'Mineração e Siderurgia',
    'RECV3.SA': 'Petróleo, Gás e Biocombustíveis',
    'VIVA3.SA': 'Saúde',
    'DXCO3.SA': 'Tecnologia',
    'SANB11.SA': 'Bancos',
    'BBSE3.SA': 'Seguradoras',
    'LJQQ3.SA': 'Tecnologia',
    'PMAM3.SA': 'Saúde',
    'SBSP3.SA': 'Utilidades Públicas',
    'ENGI11.SA': 'Energia Elétrica',
    'JHSF3.SA': 'Indústria e Bens de Capital',
    'INTB3.SA': 'Indústria e Bens de Capital',
    'RCSL4.SA': 'Tecnologia',
    'GOLL4.SA': 'Consumo Discricionário',
    'CXSE3.SA': 'Seguradoras',
    'QUAL3.SA': 'Saúde',
    'BRKM5.SA': 'Indústria e Bens de Capital',
    'HYPE3.SA': 'Saúde',
    'IRBR3.SA': 'Tecnologia',
    'MDIA3.SA': 'Consumo Básico',
    'BEEF3.SA': 'Consumo Discricionário',
    'MMXM3.SA': 'Indústria e Bens de Capital',
    'USIM5.SA': 'Mineração e Siderurgia',
}

empresas_exportadoras = [
    'VALE3.SA',  # Mineração
    'SUZB3.SA',  # Celulose
    'KLBN11.SA', # Papel e Celulose
    'AGRO3.SA',  # Agronegócio
    'PRIO3.SA',  # Petróleo
    'SLCE3.SA',  # Agronegócio
    'SMTO3.SA',  # Açúcar e Etanol
    'CSNA3.SA',  # Siderurgia
    'GGBR4.SA',  # Siderurgia
    'CMIN3.SA',  # Mineração
    'TUPY3.SA',
]


# Mapeamento dos setores mais favorecidos em cada fase do ciclo macroeconômico.
# Ajuste conforme mudanças de conjuntura ou inclusão de novos setores.
setores_por_cenario = {
    # Crescimento acelerado, demanda forte e apetite a risco.
    "Expansão Forte": [
        'Consumo Discricionário',  # Ex: varejo, turismo, educação privada
        'Tecnologia',
        'Indústria e Bens de Capital',
        'Agronegócio',
        'Mineração e Siderurgia',
        'Petróleo, Gás e Biocombustíveis'
    ],
    # Crescimento moderado, ainda com bom apetite, mas já com busca por qualidade.
    "Expansão Moderada": [
        'Consumo Discricionário',
        'Tecnologia',
        'Indústria e Bens de Capital',
        'Agronegócio',
        'Mineração e Siderurgia',
        'Petróleo, Gás e Biocombustíveis',
        'Saúde'  # Começa a ganhar tração em cenários menos exuberantes
    ],
    # Economia estável, equilíbrio entre risco e proteção, preferência por setores defensivos.
    "Estável": [
        'Saúde',
        'Bancos',
        'Seguradoras',
        'Bolsas e Serviços Financeiros',
        'Consumo Básico',
        'Utilidades Públicas',
        'Comunicação'
    ],
    # Início de desaceleração, foco em proteção e estabilidade de receita.
    "Contração Moderada": [
        'Bancos',
        'Seguradoras',
        'Consumo Básico',
        'Utilidades Públicas',
        'Saúde',
        'Energia Elétrica',
        'Comunicação'
    ],
    # Contração severa, recessão; apenas setores mais resilientes.
    "Contração Forte": [
        'Utilidades Públicas',
        'Consumo Básico',
        'Energia Elétrica',
        'Saúde'
    ]
}

# DICA: Para obter todos os setores únicos usados em qualquer cenário:
todos_setores = set()
for setores in setores_por_cenario.values():
    todos_setores.update(setores)
# todos_setores agora contém todos os setores possíveis


# ========= MACRO ==========

# Funções para obter dados do BCB

@st.cache_data(ttl=86400)
def buscar_projecoes_focus(indicador, ano=datetime.datetime.now().year):
    indicador_map = {
        "IPCA": "IPCA",
        "Selic": "Selic",
        "PIB Total": "PIB Total",
        "Câmbio": "Câmbio"
    }
    nome_indicador = indicador_map.get(indicador)
    if not nome_indicador:
        return None
    base_url = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/"
    url = f"{base_url}ExpectativasMercadoTop5Anuais?$top=100000&$filter=Indicador eq '{nome_indicador}'&$format=json&$select=Indicador,Data,DataReferencia,Mediana"
    try:
        response = requests.get(url)
        response.raise_for_status()
        dados = response.json()["value"]
        df = pd.DataFrame(dados)
        df = df[df["DataReferencia"].str.contains(str(ano))]
        df = df.sort_values("Data", ascending=False)
        if df.empty:
            raise ValueError(f"Nenhum dado encontrado para {indicador} em {ano}.")
        return float(df.iloc[0]["Mediana"])
    except Exception as e:
        print(f"Erro ao buscar {indicador} no Boletim Focus: {e}")
        return None

def obter_macro():
    macro = {
        "ipca": buscar_projecoes_focus("IPCA"),
        "selic": buscar_projecoes_focus("Selic"),
        "pib": buscar_projecoes_focus("PIB Total"),
        "petroleo": obter_preco_petroleo(),
        "dolar": buscar_projecoes_focus("Câmbio"),
        "soja": obter_preco_commodity("ZS=F", nome="Soja"),
        "milho": obter_preco_commodity("ZC=F", nome="Milho"),
        "minerio": obter_preco_commodity("BZ=F", nome="Minério de Ferro")
    }
    return macro

@st.cache_data(ttl=86400)
def obter_preco_yf(ticker, nome="Ativo"):
    """Adiciona delay para evitar limitação de requisições."""
    try:
        time.sleep(2)  # Adiciona um atraso de 1 segundo antes da requisição
        dados = yf.Ticker(ticker).history(period="5d")
        if not dados.empty and 'Close' in dados.columns:
            return float(dados['Close'].dropna().iloc[-1])
        else:
            st.warning(f"Preço indisponível para {nome}.")
            return None
    except Exception as e:
        st.error(f"Erro ao obter preço de {nome} ({ticker}): {e}")
        return None

@st.cache_data(ttl=86400)
def obter_preco_commodity(ticker, nome="Commodity"):
    return obter_preco_yf(ticker, nome)

@st.cache_data(ttl=86400)
def obter_preco_petroleo():
    return obter_preco_yf("BZ=F", "Petróleo")

# Funções de pontuação individual

PARAMS = {
    "selic_neutra": 7.0,
    "ipca_meta": 3,
    "ipca_tolerancia": 1.5,
    "dolar_ideal": 5.30,
}

# ==================== FUNÇÕES DE PONTUAÇÃO ====================

def pontuar_ipca(ipca):
    if ipca is None or pd.isna(ipca):
        return 0
    meta = PARAMS["ipca_meta"]
    tolerancia = PARAMS["ipca_tolerancia"]
    # Dentro da meta: 10 pontos
    if meta - tolerancia <= ipca <= meta + tolerancia:
        return 10
    # Até 1% acima da tolerância: 5 pontos
    elif ipca <= meta + tolerancia + 1:
        return 5
    # Muito acima do teto da meta: penalização pesada
    elif ipca > meta + tolerancia + 1:
        return 0
    # Abaixo da banda: 3 pontos (deflação)
    else:
        return 3

def pontuar_selic(selic):
    if selic is None or pd.isna(selic):
        return 0
    neutra = PARAMS["selic_neutra"]
    # Dentro da neutralidade: 10 pontos
    if abs(selic - neutra) <= 0.5:
        return 10
    # Até 2% acima: 4 pontos
    elif selic > neutra and selic <= neutra + 2:
        return 4
    # Muito acima da neutra: 0 pontos
    elif selic > neutra + 2:
        return 0
    # Abaixo da neutra: 6 pontos (ainda expansionista)
    else:
        return 6


def pontuar_dolar(dolar):
    if dolar is None or pd.isna(dolar):
        return 0
    ideal = PARAMS["dolar_ideal"]
    desvio = abs(dolar - ideal)
    return max(0, 10 - desvio * 2)

def pontuar_pib(pib):
    if pib is None or pd.isna(pib):
        return 0
    ideal = 2.0
    if pib >= ideal:
        return min(10, 8 + (pib - ideal) * 2)
    else:
        return max(0, 8 - (ideal - pib) * 3)

# Atualize as funções de preço ideal para usar médias móveis

@st.cache_data(ttl=86400)
def calcular_media_movel(ticker, periodo="12mo", intervalo="1mo"):
    """
    Calcula a média móvel do preço de um ativo (ex.: soja, milho, petróleo, minério).
    Adiciona delay para evitar limitação de requisições.
    """
    try:
        time.sleep(3)  # Adiciona um atraso de 1 segundo antes da requisição
        dados = yf.download(ticker, period=periodo, interval=intervalo, progress=False)
        if not dados.empty:
            media_movel = float(dados['Close'].mean())
            return media_movel
        else:
            st.warning(f"Dados históricos indisponíveis para {ticker}.")
            return None
    except Exception as e:
        st.error(f"Erro ao calcular média móvel para {ticker}: {e}")
        return None

# --- Função para obter preços ideais dinâmicos usando médias móveis ---
def obter_precos_ideais():
    return {
        "soja_ideal": calcular_media_movel("ZS=F", periodo="12mo", intervalo="1mo"),    # Soja
        "milho_ideal": calcular_media_movel("ZC=F", periodo="12mo", intervalo="1mo"),   # Milho
        "minerio_ideal": calcular_media_movel("TIO=F", periodo="12mo", intervalo="1mo"), # Minério de ferro (use o ticker correto para o seu caso)
        "petroleo_ideal": calcular_media_movel("BZ=F", periodo="12mo", intervalo="1mo") # Petróleo Brent
    }

# --- Atualize os parâmetros globais de commodities ---
def atualizar_parametros_com_medias_moveis():
    precos_ideais = obter_precos_ideais()
    PARAMS.update(precos_ideais)
    return PARAMS

# --- Garanta que as funções de pontuação aceitem apenas valores escalares ---
def pontuar_soja(soja):
    if soja is None or pd.isna(soja):
        return 0
    if isinstance(soja, pd.Series):
        soja = float(soja.iloc[0])
    ideal = PARAMS.get("soja_ideal", 13.0)
    if ideal is None or pd.isna(ideal):
        return 0
    desvio = abs(soja - ideal)
    return max(0, 10 - desvio * 1.5)

def pontuar_milho(milho):
    if milho is None or pd.isna(milho):
        return 0
    if isinstance(milho, pd.Series):
        milho = float(milho.iloc[0])
    ideal = PARAMS.get("milho_ideal", 5.5)
    if ideal is None or pd.isna(ideal):
        return 0
    desvio = abs(milho - ideal)
    return max(0, 10 - desvio * 2)

def pontuar_soja_milho(soja, milho):
    """Pontua a média entre soja e milho, para commodities agro."""
    return (pontuar_soja(soja) + pontuar_milho(milho)) / 2
    
def pontuar_minerio(minerio):
    if minerio is None or pd.isna(minerio):
        return 0
    if isinstance(minerio, pd.Series):
        minerio = float(minerio.iloc[0])
    ideal = PARAMS["minerio_ideal"]
    if ideal is None or pd.isna(ideal):
        return 0
    desvio = abs(minerio - ideal)
    return max(0, 10 - desvio * 0.1)

def pontuar_petroleo(petroleo):
    if petroleo is None or pd.isna(petroleo):
        return 0
    if isinstance(petroleo, pd.Series):
        petroleo = float(petroleo.iloc[0])
    ideal = PARAMS["petroleo_ideal"]
    if ideal is None or pd.isna(ideal):
        return 0
    desvio = abs(petroleo - ideal)
    return max(0, 10 - desvio * 0.2)



# --- Garanta que os parâmetros estejam atualizados antes do uso ---
PARAMS = atualizar_parametros_com_medias_moveis()

# Atualize o Streamlit para mostrar os preços ideais
def validar_macro(macro):
    obrigatorios = ["selic", "ipca", "dolar", "pib", "soja", "milho", "minerio", "petroleo"]
    for k in obrigatorios:
        if k not in macro or macro[k] is None or (isinstance(macro[k], float) and pd.isna(macro[k])):
            macro[k] = 0.0  # Preencha com zero se ausente ou inválido
            

def pontuar_macro(m, pesos=None):
    """
    Calcula scores macroeconômicos normalizados e média ponderada.
    m: dict com indicadores macroeconômicos
    pesos: dict opcional com pesos dos indicadores
    """
    validar_macro(m)
    score = {
        "juros": pontuar_selic(m["selic"]),
        "inflação": pontuar_ipca(m["ipca"]),
        "dolar": pontuar_dolar(m["dolar"]),
        "pib": pontuar_pib(m["pib"]),
        "commodities_agro": pontuar_soja_milho(m["soja"], m["milho"]),
        "commodities_minerio": pontuar_minerio(m["minerio"]),
        "commodities_petroleo": pontuar_petroleo(m["petroleo"]),
    }
    # Normalização e pesos
    pesos = pesos or {k: 1 for k in score}
    total_peso = sum(pesos.values())
    media_global = sum(score[k] * pesos.get(k, 1) for k in score) / total_peso
    score["media_global"] = media_global
    return score



# Funções para preço-alvo e preço atual

def obter_preco_alvo(ticker):
    try:
        return yf.Ticker(ticker).info.get('targetMeanPrice', None)
    except Exception as e:
        st.warning(f"Erro ao obter preço-alvo de {ticker}: {e}")
        return None

def obter_preco_atual(ticker):
    try:
        dados = yf.Ticker(ticker).history(period="1d")
        if not dados.empty:
            return dados['Close'].iloc[-1]
    except Exception as e:
        st.warning(f"Erro ao obter preço atual de {ticker}: {e}")
    return None

def gerar_ranking_acoes(carteira, macro, usar_pesos_macro=True):
    score_macro = pontuar_macro(macro)
    resultados = []

    for ticker in carteira.keys():
        setor = setores_por_ticker.get(ticker)
        if setor is None:
            st.warning(f"Setor não encontrado para {ticker}. Ignorando.")
            continue

        preco_atual = obter_preco_atual(ticker)
        preco_alvo = obter_preco_alvo(ticker)

        if preco_atual is None or preco_alvo is None or preco_atual == 0:
            st.warning(f"Dados insuficientes para {ticker}. Ignorando.")
            continue

        favorecimento_score = calcular_favorecimento_continuo(setor, score_macro)
        score, detalhe = calcular_score(preco_atual, preco_alvo, favorecimento_score, ticker, setor, macro, usar_pesos_macro, return_details=True)

        resultados.append({
            "ticker": ticker,
            "setor": setor,
            "preço atual": preco_atual,
            "preço alvo": preco_alvo,
            "favorecimento macro": favorecimento_score,
            "score": score,
            "detalhe": detalhe
        })

    df = pd.DataFrame(resultados).sort_values(by="score", ascending=False)

    # Garantir exibição mesmo se algumas colunas estiverem ausentes
    colunas_desejadas = ["ticker", "setor", "preço atual", "preço alvo", "favorecimento macro", "score"]
    colunas_existentes = [col for col in colunas_desejadas if col in df.columns]
    st.dataframe(df[colunas_existentes], use_container_width=True)

    
    with st.expander("🔍 Ver detalhes dos scores"):
        st.dataframe(df[["ticker", "detalhe"]], use_container_width=True)
    
    return df

     


def calcular_score(
    preco_atual, preco_alvo, favorecimento_score, ticker, setor, macro,
    usar_pesos_macroeconomicos=True, return_details=False
):
    """
    Calcula o score de atratividade do ativo considerando upside, macro, favorecimento setorial e bônus de exportadora.
    O score final é limitado a [-10, +10] para facilitar comparação.
    """

    if preco_atual == 0:
        return -float("inf"), "Preço atual igual a zero"

    upside = (preco_alvo - preco_atual) / preco_atual

    # Upside: peso reduzido, logarítmico (para não distorcer por outlier)
    base_score = np.sign(upside) * np.log1p(abs(upside)) * 3  # máximo prático ~3 a 4

    # Macro: agora pesa mais
    score_macro = 0
    if setor in sensibilidade_setorial and usar_pesos_macroeconomicos:
        s = sensibilidade_setorial[setor]
        score_indicadores = pontuar_macro(macro)
        for indicador, peso in s.items():
            score_macro += peso * score_indicadores.get(indicador, 0)
    score_macro = np.clip(score_macro, -10, 10)

    # Favorecimento setorial: peso relevante
    favorecimento_peso = 2.0 if usar_pesos_macroeconomicos else 0

    # Bônus para exportadoras
    bonus = 0
    if ticker in empresas_exportadoras:
        if macro.get('dolar') and macro['dolar'] > PARAMS["dolar_ideal"]:
            bonus += 0.10
        if macro.get('petroleo') and macro['petroleo'] > PARAMS["petroleo_ideal"]:
            bonus += 0.05
    bonus = np.clip(bonus, 0, 0.15)

    # Score final: pesos calibrados para que nenhum fator domine
    score_total = (
        base_score                 # -3 a +3 (upside)
        + (0.20 * score_macro)     # -2 a +2 (macro)
        + bonus                    # até +0.15
        + (favorecimento_score * favorecimento_peso)  # -4 a +4 (favorecimento)
    )
    score_total = np.clip(score_total, -10, 10)

    detalhe = (
        f"upside={upside:.2f}, base={base_score:.2f}, macro={score_macro:.2f}, "
        f"bonus={bonus:.2f}, favorecimento={favorecimento_score:.2f}, score_final={score_total:.2f}"
    )

    return (score_total, detalhe) if return_details else score_total


def classificar_cenario_macro(
    ipca, selic, dolar, pib,
    preco_soja=None, preco_milho=None,
    preco_minerio=None, preco_petroleo=None
):
    score_ipca = pontuar_ipca(ipca)
    score_selic = pontuar_selic(selic)
    score_dolar = pontuar_dolar(dolar)
    score_pib = pontuar_pib(pib)
    
    # Soma apenas dos 4 principais indicadores macro
    core_score = score_ipca + score_selic + score_dolar + score_pib

    # Commodities - peso quase simbólico (apenas 0.1x)
    commodities = [
        ('soja', preco_soja, pontuar_soja),
        ('milho', preco_milho, pontuar_milho),
        ('minerio', preco_minerio, pontuar_minerio),
        ('petroleo', preco_petroleo, pontuar_petroleo)
    ]
    commodities_score = 0
    for nome, preco, func in commodities:
        if preco is not None and not pd.isna(preco):
            commodities_score += 0.1 * func(preco)

    total_score = core_score + commodities_score

    # ESCALA SUPER CONSERVADORA: "Estável" só se tudo está ótimo
    if total_score >= 38:
        return "Expansão Forte"
    elif total_score >= 32:
        return "Expansão Moderada"
    elif total_score >= 26:
        return "Estável"
    elif total_score >= 14:
        return "Contração Moderada"
    else:
        return "Contração Forte"



def get_macro_adjusted_returns(retornos, score_dict):
    """
    Ajusta o retorno esperado anualizado de cada ativo usando o score macro/setorial.
    Os scores são normalizados para evitar distorção excessiva.
    """
    media_retorno = retornos.mean() * 252
    tickers = retornos.columns.tolist()
    # Normaliza scores para o intervalo [0.5, 1.5]
    min_score = min(score_dict.values()) if len(score_dict) > 0 else 0
    max_score = max(score_dict.values()) if len(score_dict) > 0 else 1
    def norm(s): return 0.5 + (s - min_score) / (max_score - min_score + 1e-9)
    ajuste_score = np.array([norm(score_dict.get(t, 0)) for t in tickers])
    return media_retorno * ajuste_score

def macro_bounds(tickers, score_dict, limite_base=0.20, bonus=0.10):
    """
    Limita os pesos máximos de cada ativo conforme o score macro.
    Ativos favorecidos podem receber limite maior.
    """
    max_score = max(score_dict.values()) if len(score_dict) > 0 else 1
    bounds = []
    for t in tickers:
        score = score_dict.get(t, 0)
        bonus_pct = bonus * (score / max_score) if max_score > 0 else 0
        bounds.append((0, min(1, limite_base + bonus_pct)))
    return tuple(bounds)


#===========PESOS FALTANTES======
def completar_pesos(tickers_originais, pesos_calculados):
    """
    Garante que todos os ativos originais estejam presentes nos pesos finais,
    atribuindo 0 para os que foram excluídos na otimização.
    """
    pesos_completos = pd.Series(0.0, index=tickers_originais)
    for ticker in pesos_calculados.index:
        pesos_completos[ticker] = pesos_calculados[ticker]
    return pesos_completos

        

# ========= FILTRAR AÇÕES ==========
# Novo modelo com commodities separadas
sensibilidade_setorial = {
    # Setores pró-cíclicos (beneficiam muito de crescimento/expansão, sensíveis ao PIB e commodities)
    'Consumo Discricionário': {
        'juros': -2, 'inflação': -1, 'dolar': -1, 'pib': 2.5,
        'commodities_agro': -0.5, 'commodities_minerio': -0.5, 'commodities_petroleo': -0.2
    },
    'Tecnologia': {
        'juros': -1.5, 'inflação': 0, 'dolar': -1, 'pib': 2,
        'commodities_agro': -0.2, 'commodities_minerio': -0.2, 'commodities_petroleo': 0
    },
    'Indústria e Bens de Capital': {
        'juros': -1, 'inflação': -0.5, 'dolar': -0.5, 'pib': 2.2,
        'commodities_agro': 0, 'commodities_minerio': 0.2, 'commodities_petroleo': 0
    },
    'Mineração e Siderurgia': {
        'juros': 0, 'inflação': 0, 'dolar': 2, 'pib': 1.2,
        'commodities_agro': 0, 'commodities_minerio': 2.5, 'commodities_petroleo': 0.6
    },
    'Petróleo, Gás e Biocombustíveis': {
        'juros': 0, 'inflação': 0, 'dolar': 1.5, 'pib': 1,
        'commodities_agro': 0, 'commodities_minerio': 0, 'commodities_petroleo': 2.7
    },
    'Agronegócio': {
        'juros': -0.5, 'inflação': -0.6, 'dolar': 1.7, 'pib': 1.1,
        'commodities_agro': 2.7, 'commodities_minerio': 0, 'commodities_petroleo': 0.4
    },

    # Setores defensivos (resilientes em qualquer ciclo, mas pouco sensíveis positivamente ao PIB)
    'Saúde': {
        'juros': 0, 'inflação': 0, 'dolar': 0, 'pib': 0.6,
        'commodities_agro': 0, 'commodities_minerio': 0, 'commodities_petroleo': 0
    },
    'Consumo Básico': {
        'juros': 0.7, 'inflação': -1.2, 'dolar': -0.7, 'pib': 0.6,
        'commodities_agro': -0.2, 'commodities_minerio': -0.2, 'commodities_petroleo': -0.1
    },
    'Utilidades Públicas': {
        'juros': 1.2, 'inflação': 0.7, 'dolar': -0.6, 'pib': -0.6,
        'commodities_agro': -0.2, 'commodities_minerio': -0.2, 'commodities_petroleo': 0
    },
    'Energia Elétrica': {
        'juros': 0.5, 'inflação': 0.5, 'dolar': -0.7, 'pib': -0.7,
        'commodities_agro': -0.3, 'commodities_minerio': -0.2, 'commodities_petroleo': 0.1
    },

    # Setores financeiros (Bancos, Seguradoras, Bolsas)
    'Bancos': {
        'juros': 1.6, 'inflação': -0.1, 'dolar': -0.3, 'pib': 1.1,
        'commodities_agro': 0.3, 'commodities_minerio': 0.2, 'commodities_petroleo': 0
    },
    'Seguradoras': {
        'juros': 2, 'inflação': 0.2, 'dolar': 0, 'pib': 0.7,
        'commodities_agro': 0, 'commodities_minerio': 0, 'commodities_petroleo': 0
    },
    'Bolsas e Serviços Financeiros': {
        'juros': 1, 'inflação': 0, 'dolar': 0, 'pib': 1.5,
        'commodities_agro': 0, 'commodities_minerio': 0, 'commodities_petroleo': 0
    },

    # Outros setores típicos
    'Comunicação': {
        'juros': 0, 'inflação': 0, 'dolar': -0.3, 'pib': 0.5,
        'commodities_agro': 0, 'commodities_minerio': 0, 'commodities_petroleo': 0
    }
}
# Sugestão: documente/calcule a origem destes valores, e revise-os periodicamente.

def calcular_favorecimento_continuo(setor, score_macro):
    """
    Calcula o favorecimento contínuo do setor com base na sensibilidade setorial e scores macro.
    """
    if setor not in sensibilidade_setorial:
        return 0
    sens = sensibilidade_setorial[setor]
    bruto = sum(score_macro.get(k, 0) * peso for k, peso in sens.items())
    return np.tanh(bruto / 5) * 2


def filtrar_ativos_validos(carteira, setores_por_ticker, setores_por_cenario, macro, calcular_score):
    # Extrair valores individuais do dicionário de pontuação
    score_macro = pontuar_macro(macro)
    ipca = score_macro.get("inflação")
    selic = score_macro.get("juros")
    dolar = score_macro.get("dolar")
    pib = score_macro.get("pib")

    # Agora chama a função passando os parâmetros individuais
    cenario = classificar_cenario_macro(ipca, selic, dolar, pib, 
                                        preco_soja=macro.get("soja"), 
                                        preco_milho=macro.get("milho"), 
                                        preco_minerio=macro.get("minerio"), 
                                        preco_petroleo=macro.get("petroleo"))
    
    # Exibir as pontuações e o cenário


    # Obter os setores válidos conforme o cenário
    setores_cidos = setores_por_cenario.get(cenario, [])

    # Inicializar a lista de ativos válidos
    ativos_validos = []
    for ticker in carteira:
        setor = setores_por_ticker.get(ticker, None)
        preco_atual = obter_preco_atual(ticker)
        preco_alvo = obter_preco_alvo(ticker)

        if preco_atual is None or preco_alvo is None:
            continue

        favorecimento_score = calcular_favorecimento_continuo(setor, macro)
        score = calcular_score(preco_atual, preco_alvo, favorecimento_score, ticker, setor, macro, usar_pesos_macroeconomicos=True, return_details=False)

        # Adicionar o ativo à lista de ativos válidos
        ativos_validos.append({
            "ticker": ticker,
            "setor": setor,
            "cenario": cenario,
            "preco_atual": preco_atual,
            "preco_alvo": preco_alvo,
            "score": score,
            "favorecido": favorecimento_score
        })

    # Ordenar os ativos válidos pelo score
    ativos_validos.sort(key=lambda x: x['score'], reverse=True)

    return ativos_validos


# ========= OTIMIZAÇÃO ==========



@st.cache_data(ttl=86400)
def obter_preco_diario_ajustado(tickers):
    dados_brutos = yf.download(tickers, period="7y", auto_adjust=False)

    # Forçar tickers a ser lista, mesmo se for string
    if isinstance(tickers, str):
        tickers = [tickers]

    if isinstance(dados_brutos.columns, pd.MultiIndex):
        if 'Adj Close' in dados_brutos.columns.get_level_values(0):
            return dados_brutos['Adj Close']
        elif 'Close' in dados_brutos.columns.get_level_values(0):
            return dados_brutos['Close']
        else:
            raise ValueError("Colunas 'Adj Close' ou 'Close' não encontradas nos dados.")
    else:
        # Apenas um ticker e colunas simples
        if 'Adj Close' in dados_brutos.columns:
            return dados_brutos[['Adj Close']].rename(columns={'Adj Close': tickers[0]})
        elif 'Close' in dados_brutos.columns:
            return dados_brutos[['Close']].rename(columns={'Close': tickers[0]})
        else:
            raise ValueError("Coluna 'Adj Close' ou 'Close' não encontrada nos dados.")
            
def calcular_fronteira_eficiente_macro(retornos, score_dict, n_portfolios=100, taxa_risco_livre=0.0):
    """
    Gera portfolios aleatórios usando retornos ajustados pelo score macro.
    """
    media_retorno = get_macro_adjusted_returns(retornos, score_dict)
    cov = retornos.cov() * 252
    num_ativos = len(media_retorno)
    resultados = []

    for _ in range(n_portfolios):
        pesos = np.random.random(num_ativos)
        pesos /= np.sum(pesos)
        ret = np.dot(pesos, media_retorno)
        vol = np.sqrt(np.dot(pesos.T, np.dot(cov, pesos)))
        sharpe = (ret - taxa_risco_livre) / vol if vol > 0 else 0
        resultados.append((vol, ret, sharpe, pesos.copy()))

    df = pd.DataFrame(resultados, columns=['Volatilidade', 'Retorno', 'Sharpe', 'Pesos'])
    return df

def otimizar_carteira_sharpe(tickers, carteira_atual, taxa_risco_livre=0.0001, favorecimentos=None):
    """
    Otimiza a carteira com base no índice de Sharpe, agora ajustando retornos, limites e pesos iniciais
    conforme o score macro/setorial de cada ativo.
    """
    dados = obter_preco_diario_ajustado(tickers)
    dados = dados.ffill().bfill()

    # Retornos logarítmicos
    retornos = np.log(dados / dados.shift(1)).dropna()
    tickers_validos = retornos.columns.tolist()
    n = len(tickers_validos)
    if n == 0:
        st.error("Nenhum dado de retorno válido disponível para os ativos selecionados.")
        return pd.Series(0.0, index=tickers)

    # 1. Normalizar scores para range (ex: 0.7 a 1.3)
    if favorecimentos:
        fav_array_raw = np.array([favorecimentos.get(t, 0) for t in tickers_validos])
        min_fav, max_fav = fav_array_raw.min(), fav_array_raw.max()
        if max_fav == min_fav:
            fav_norm = np.ones(n)
        else:
            fav_norm = 0.7 + 0.6 * (fav_array_raw - min_fav) / (max_fav - min_fav)
    else:
        fav_norm = np.ones(n)

    # 2. Ajuste do retorno esperado pelo score macro
    media_retorno = retornos.mean() * 252
    media_retorno_ajustado = media_retorno * fav_norm

    # 3. Limites máximos por ativo ajustados pelo score macro
    limite_base = 0.20
    bonus_limite = 0.10
    limites = []
    for f in fav_norm:
        limites.append((0.01, min(1, limite_base + bonus_limite * (f - 1))))
    limites = tuple(limites)

    # 4. Pesos iniciais proporcionais ao score macro
    if favorecimentos and fav_norm.sum() > 0:
        pesos_iniciais = fav_norm / fav_norm.sum()
    else:
        pesos_iniciais = np.ones(n) / n

    # 5. Matriz de covariância robusta
    cov_matrix = LedoitWolf().fit(retornos).covariance_
    cov = pd.DataFrame(cov_matrix, index=retornos.columns, columns=retornos.columns)

    def sharpe_neg(pesos):
        ret = np.dot(pesos, media_retorno_ajustado) - taxa_risco_livre
        vol = np.sqrt(pesos @ cov.values @ pesos.T)
        return -ret / vol if vol > 0 else 0

    restricoes = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    resultado = minimize(
        sharpe_neg,
        pesos_iniciais,
        method='SLSQP',
        bounds=limites,
        constraints=restricoes,
        options={'disp': False, 'maxiter': 1000}
    )

    if resultado.success and not np.isnan(resultado.fun):
        pesos_otimizados = pd.Series(resultado.x, index=tickers_validos)
        return completar_pesos(tickers, pesos_otimizados)
    else:
        st.warning("Otimização falhou ou retornou valor inválido. Usando pesos uniformes.")
        pesos_uniformes = pd.Series(np.ones(n) / n, index=tickers_validos)
        return completar_pesos(tickers, pesos_uniformes)


def otimizar_carteira_retorno_maximo(tickers, carteira_atual, favorecimentos=None):
    """
    Otimiza a carteira para máximo retorno esperado com limitação máxima de 20% por ativo.
    """
    dados = obter_preco_diario_ajustado(tickers)
    dados = dados.ffill().bfill()

    retornos = np.log(dados / dados.shift(1)).dropna()
    tickers_validos = retornos.columns.tolist()
    n = len(tickers_validos)

    if n == 0:
        st.error("Nenhum dado de retorno válido disponível para os ativos selecionados.")
        return pd.Series(0.0, index=tickers)

    media_retorno = retornos.mean()
    # Pesos iniciais proporcionais ao retorno esperado (ou uniformes)
    if (media_retorno > 0).any():
        pesos_iniciais = np.maximum(media_retorno, 0)
        if pesos_iniciais.sum() > 0:
            pesos_iniciais = pesos_iniciais / pesos_iniciais.sum()
        else:
            pesos_iniciais = np.ones(n) / n
    else:
        pesos_iniciais = np.ones(n) / n

    # Limite estrito de 20% por ativo
    limites = [(0.0, 0.20) for _ in range(n)]

    # Restrição: soma dos pesos = 1
    restricoes = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    # Função objetivo: maximize retorno esperado (minimize negativo do retorno)
    def neg_retorno(pesos):
        return -np.dot(pesos, media_retorno)

    resultado = minimize(
        neg_retorno,
        pesos_iniciais,
        method='SLSQP',
        bounds=limites,
        constraints=restricoes,
        options={'disp': False, 'maxiter': 1000}
    )

    if resultado.success and not np.isnan(resultado.fun):
        pesos_otimizados = pd.Series(resultado.x, index=tickers_validos)
        return completar_pesos(tickers, pesos_otimizados)
    else:
        st.warning("Otimização falhou ou retornou valor inválido. Usando pesos uniformes.")
        pesos_uniformes = pd.Series(np.ones(n) / n, index=tickers_validos)
        return completar_pesos(tickers, pesos_uniformes)


def otimizar_carteira_hrp(tickers, carteira_atual, favorecimentos=None):
    """
    Otimiza a carteira com HRP, ajustando os pesos finais com base nos ativos válidos.
    """
    dados = obter_preco_diario_ajustado(tickers)
    dados = dados.dropna(axis=1, how='any')
    tickers_validos = dados.columns.tolist()

    if len(tickers_validos) < 2:
        st.error("Número insuficiente de ativos com dados válidos para otimização.")
        return pd.Series(0.0, index=tickers)

    retornos = dados.pct_change().dropna()
    correlacao = retornos.corr()
    dist = np.sqrt((1 - correlacao) / 2)

    dist_condensada = squareform(dist.values, checks=False)
    linkage_matrix = linkage(dist_condensada, method='single')

    def get_quasi_diag(link):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0]*2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df1 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df1])
            sort_ix = sort_ix.sort_index()
        return sort_ix.tolist()

    def get_recursive_bisection(cov, sort_ix):
        w = pd.Series(1, index=sort_ix)
        cluster_items = [sort_ix]

        while len(cluster_items) > 0:
            cluster_items = [i[j:k] for i in cluster_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for cluster in cluster_items:
                c_items = cluster
                c_var = cov.loc[c_items, c_items].values
                inv_diag = 1. / np.diag(c_var)
                parity_w = inv_diag / inv_diag.sum()
                alloc = parity_w.sum()
                w[c_items] *= parity_w * alloc
        return w / w.sum()

    cov_matrix = LedoitWolf().fit(retornos).covariance_
    cov_df = pd.DataFrame(cov_matrix, index=retornos.columns, columns=retornos.columns)
    sort_ix = get_quasi_diag(linkage_matrix)
    ordered_tickers = [retornos.columns[i] for i in sort_ix]
    pesos_hrp = get_recursive_bisection(cov_df, ordered_tickers)

        # --- NOVO: ajuste final pelo favorecimento ---
    if favorecimentos:
        fav_array = np.array([1 + max(0, favorecimentos.get(t, 0)) for t in pesos_hrp.index])
        pesos_hrp = pesos_hrp * fav_array
        pesos_hrp = pesos_hrp / pesos_hrp.sum()

    return completar_pesos(tickers, pesos_hrp)

macro = obter_macro()

historico_7anos = montar_historico_7anos(
    tickers=list(setores_por_ticker.keys()),
    setores_por_ticker=setores_por_ticker,
    start='2018-01-01'
)



import matplotlib.pyplot as plt

def calcular_cagr(valor_final, valor_inicial, anos):
    return (valor_final / valor_inicial) ** (1 / anos) - 1

def backtest_portfolio_vs_ibov_duplo(tickers, pesos, start_date='2018-01-01'):
    df_adj = yf.download(tickers, start=start_date, auto_adjust=True, progress=False)['Close']
    df_close = yf.download(tickers, start=start_date, auto_adjust=False, progress=False)['Close']

    df_adj = df_adj.ffill().dropna()
    df_close = df_close.ffill().dropna()

    ibov_adj = yf.download('^BVSP', start=start_date, auto_adjust=True, progress=False)['Close']
    ibov_close = yf.download('^BVSP', start=start_date, auto_adjust=False, progress=False)['Close']

    ibov_adj = ibov_adj.ffill().dropna()
    ibov_close = ibov_close.ffill().dropna()

    idx = df_adj.index.intersection(df_close.index).intersection(ibov_adj.index).intersection(ibov_close.index)
    df_adj, df_close = df_adj.loc[idx], df_close.loc[idx]
    ibov_adj, ibov_close = ibov_adj.loc[idx], ibov_close.loc[idx]

    df_adj_norm = df_adj / df_adj.iloc[0]
    df_close_norm = df_close / df_close.iloc[0]
    ibov_adj_norm = ibov_adj / ibov_adj.iloc[0]
    ibov_close_norm = ibov_close / ibov_close.iloc[0]

    pesos = np.array(pesos)
    if len(pesos) != df_adj.shape[1]:
        pesos = np.ones(df_adj.shape[1]) / df_adj.shape[1]

    port_adj = (df_adj_norm * pesos).sum(axis=1)
    port_close = (df_close_norm * pesos).sum(axis=1)

    anos = (port_adj.index[-1] - port_adj.index[0]).days / 365.25
    cagr_port_adj = calcular_cagr(float(port_adj.iloc[-1]), float(port_adj.iloc[0]), anos)
    cagr_port_close = calcular_cagr(float(port_close.iloc[-1]), float(port_close.iloc[0]), anos)
    cagr_ibov_adj = calcular_cagr(float(ibov_adj_norm.iloc[-1]), float(ibov_adj_norm.iloc[0]), anos)
    cagr_ibov_close = calcular_cagr(float(ibov_close_norm.iloc[-1]), float(ibov_close_norm.iloc[0]), anos)

    st.markdown(f"**CAGR Carteira Recomendada (Ajustado):** {100*float(cagr_port_adj):.2f}% ao ano")
    st.markdown(f"**CAGR Carteira Recomendada (Close):** {100*float(cagr_port_close):.2f}% ao ano")
    st.markdown(f"**CAGR IBOV (Ajustado):** {100*float(cagr_ibov_adj):.2f}% ao ano")
    st.markdown(f"**CAGR IBOV (Close):** {100*float(cagr_ibov_close):.2f}% ao ano")

    fig, ax = plt.subplots(figsize=(10, 6))
    port_adj.plot(ax=ax, label='Carteira Recomendada (Ajustado)')
    port_close.plot(ax=ax, label='Carteira Recomendada (Close)')
    ibov_adj_norm.plot(ax=ax, label='IBOV (Ajustado)')
    ibov_close_norm.plot(ax=ax, label='IBOV (Close)')
    ax.set_title('Backtest: Carteira Recomendada vs IBOV (7 anos)')
    ax.set_ylabel('Retorno acumulado')
    ax.set_xlabel('Ano')
    ax.legend()
    st.pyplot(fig)


# ========= STREAMLIT ==========
# ========= STREAMLIT ==========


st.title("📊 Sugestão e Otimização de Carteira: Cenário Projetado")

st.markdown("---")

# Sempre use o macro atualizado
macro = obter_macro()

with st.sidebar:
    st.header("Ajuste Manual dos Indicadores Macro")
    macro_manual = {}
    for indicador in ["ipca", "selic", "pib", "dolar"]:
        macro_manual[indicador] = st.number_input(
            f"{indicador.upper()} (ajuste, opcional)", 
            value=macro[indicador] if macro[indicador] else 0.0,
            step=0.01
        )
    usar_macro_manual = st.checkbox("Usar ajustes manuais acima?")
    if usar_macro_manual:
        macro.update(macro_manual)

cenario_atual = classificar_cenario_macro(
    ipca=macro.get("ipca"),
    selic=macro.get("selic"),
    dolar=macro.get("dolar"),
    pib=macro.get("pib"),
    preco_soja=macro.get("soja"),
    preco_milho=macro.get("milho"),
    preco_minerio=macro.get("minerio"),
    preco_petroleo=macro.get("petroleo")
)

score_macro = pontuar_macro(macro)
score_medio = round(np.mean(list(score_macro.values())), 2)
st.markdown(f"### 🧭 Cenário Macroeconômico Atual: **{cenario_atual}**")
st.markdown("### 📉 Indicadores Macroeconômicos")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Selic (%)", f"{macro['selic']:.2f}" if macro.get("selic") is not None else "N/A")
col2.metric("IPCA (%)", f"{macro['ipca']:.2f}" if macro.get("ipca") is not None else "N/A")
col3.metric("PIB (%)", f"{macro['pib']:.2f}" if macro.get("pib") is not None else "N/A")
col4.metric("Dólar (R$)", f"{macro['dolar']:.2f}" if macro.get("dolar") is not None else "N/A")
col5.metric("Petróleo (US$)", f"{macro['petroleo']:.2f}" if macro.get("petroleo") else "N/A")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Parâmetros")
    st.markdown("### Dados dos Ativos")
    tickers_default = [
        "AGRO3.SA", "BBAS3.SA", "BBSE3.SA", "BPAC11.SA", "EGIE3.SA",
        "ITUB4.SA", "PRIO3.SA", "PSSA3.SA", "SAPR11.SA", "SBSP3.SA",
        "VIVT3.SA", "WEGE3.SA", "TOTS3.SA", "B3SA3.SA", "TAEE11.SA"
    ]
    pesos_default = [
        0.07, 0.06, 0.07, 0.07, 0.08,
        0.07, 0.12, 0.09, 0.06, 0.04,
        0.1, 0.18, 0.04, 0.01, 0.02
    ]
    if "num_ativos" not in st.session_state:
        st.session_state.num_ativos = len(tickers_default)
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("( + )", key="add_ativo"):
            st.session_state.num_ativos += 1
    with col2:
        if st.button("( - )", key="remove_ativo") and st.session_state.num_ativos > 1:
            st.session_state.num_ativos -= 1
    tickers = []
    pesos = []
    for i in range(st.session_state.num_ativos):
        col1, col2 = st.columns(2)
        with col1:
            ticker_default = tickers_default[i] if i < len(tickers_default) else ""
            ticker = st.text_input(f"Ticker do Ativo {i+1}", value=ticker_default, key=f"ticker_{i}").upper()
        with col2:
            peso_default = pesos_default[i] if i < len(pesos_default) else 1.0
            peso = st.number_input(f"Peso do Ativo {i+1}", min_value=0.0, step=0.01, value=peso_default, key=f"peso_{i}")
        if ticker:
            tickers.append(ticker)
            pesos.append(peso)
    pesos_array = np.array(pesos)
    if pesos_array.sum() > 0:
        pesos_atuais = pesos_array / pesos_array.sum()
    else:
        st.error("A soma dos pesos deve ser maior que 0.")
        st.stop()

# ==================== GERAÇÃO DO RANKING E OTIMIZAÇÃO ====================

st.subheader("🏆 Ranking Geral de Ações (com base no score)")
carteira = dict(zip(tickers, pesos_atuais))
ranking_df = gerar_ranking_acoes(carteira, macro, usar_pesos_macro=True)

aporte = st.number_input("💰 Valor do aporte mensal (R$)", min_value=100.0, value=1000.0, step=100.0)

# Novo: seleção do método de otimização

ativos_validos = filtrar_ativos_validos(carteira, setores_por_ticker, setores_por_cenario, macro, calcular_score)
favorecimentos = {a['ticker']: a['favorecido'] for a in ativos_validos}






# Função para exibir histórico ajustado (lembre-se de definir montar_historico_7anos!)

# Função para exibir histórico ajustado (lembre-se de definir montar_historico_7anos!)
def mostrar_historico_ajustado(tickers, setores_por_ticker, cenario_atual, anos=7):
    if 'montar_historico_7anos' not in globals():
        st.warning("Função montar_historico_7anos não definida.")
        return

    historico_7anos_df = montar_historico_7anos(
        tickers=tickers,
        setores_por_ticker=setores_por_ticker,
        start=(pd.Timestamp.today() - pd.DateOffset(years=anos)).strftime('%Y-%m-%d')
    )
    historico_cenario = historico_7anos_df[historico_7anos_df["cenario"] == cenario_atual]
    if not historico_cenario.empty:
        destaque_hist = (
            historico_cenario.groupby(["ticker", "setor"])
            .agg(
                media_favorecido=("favorecido", "mean"),
                ocorrencias=("favorecido", "count")
            )
            .reset_index()
            .sort_values(by=["media_favorecido", "ocorrencias"], ascending=False)
        )
        st.subheader(f"🏅 Empresas da sua carteira que mais se destacaram em cenários '{cenario_atual}' nos últimos {anos} anos")
        st.dataframe(destaque_hist.head(100), use_container_width=True)
    else:
        st.info(f"Sem dados históricos para o cenário '{cenario_atual}' nos últimos {anos} anos.")

# Função para calcular métricas da carteira (CAGR, risco, Sharpe)
def calcular_metricas_carteira(tickers, pesos, start_date='2018-01-01', rf=0):
    import yfinance as yf
    df_adj = yf.download(tickers, start=start_date, auto_adjust=True, progress=False)['Close']
    df_adj = df_adj.ffill().dropna()
    df_adj = df_adj.loc[:, ~df_adj.columns.duplicated()]  # Remove duplicadas
    df_pct = df_adj.pct_change().dropna()
    port_retorno = (df_pct * pesos).sum(axis=1)
    port_valor = (1 + port_retorno).cumprod()
    anos = (port_valor.index[-1] - port_valor.index[0]).days / 365.25
    cagr = (float(port_valor.iloc[-1]) / float(port_valor.iloc[0])) ** (1 / anos) - 1
    risco = port_retorno.std() * np.sqrt(252)
    sharpe = (port_retorno.mean() * 252 - rf) / risco if risco > 0 else 0
    return cagr, risco, sharpe

# Função para backtest plug and play (ajuste para seu fluxo)
def backtest_portfolio_vs_ibov_duplo(tickers, pesos, start_date='2018-01-01'):
    import yfinance as yf
    df_adj = yf.download(tickers, start=start_date, auto_adjust=True, progress=False)['Close']
    ibov_adj = yf.download('^BVSP', start=start_date, auto_adjust=True, progress=False)['Close']
    df_adj = df_adj.ffill().dropna()
    ibov_adj = ibov_adj.ffill().dropna()
    idx = df_adj.index.intersection(ibov_adj.index)
    df_adj, ibov_adj = df_adj.loc[idx], ibov_adj.loc[idx]
    df_adj_norm = df_adj / df_adj.iloc[0]
    ibov_adj_norm = ibov_adj / ibov_adj.iloc[0]
    pesos = np.array(pesos)
    if len(pesos) != df_adj.shape[1]:
        pesos = np.ones(df_adj.shape[1]) / df_adj.shape[1]
    port_adj = (df_adj_norm * pesos).sum(axis=1)
    anos = (port_adj.index[-1] - port_adj.index[0]).days / 365.25
    cagr_port_adj = (float(port_adj.iloc[-1]) / float(port_adj.iloc[0])) ** (1 / anos) - 1
    cagr_ibov_adj = (float(ibov_adj_norm.iloc[-1]) / float(ibov_adj_norm.iloc[0])) ** (1 / anos) - 1
    st.markdown(f"**CAGR Carteira Recomendada:** {100*float(cagr_port_adj):.2f}% ao ano")
    st.markdown(f"**CAGR IBOV:** {100*float(cagr_ibov_adj):.2f}% ao ano")
    fig, ax = plt.subplots(figsize=(10, 6))
    port_adj.plot(ax=ax, label='Carteira Recomendada')
    ibov_adj_norm.plot(ax=ax, label='IBOV')
    ax.set_title('Backtest: Carteira Recomendada vs IBOV (7 anos)')
    ax.set_ylabel('Retorno acumulado')
    ax.set_xlabel('Ano')
    ax.legend()
    st.pyplot(fig)

# --- Lista dos métodos de alocação ---
metodos_aporte = [
    "Sharpe (macro)",
    "Sharpe (Monte Carlo)",
    "HRP",
    "Monte Carlo (Melhor Simulada)",
    "HRP + Monte Carlo"
]
metodo_escolha = st.selectbox(
    "Qual carteira usar para recomendação de aporte?",
    metodos_aporte,
    key="metodo_aporte"
)

# Slider para o alpha HRP+MonteCarlo somente se selecionado
if st.session_state.get("metodo_aporte") == "HRP + Monte Carlo":
    alpha = st.slider(
        "Ajuste a combinação: 0 = só Monte Carlo • 1 = só HRP",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="Ajuste o quanto a alocação final deve ser influenciada pelo HRP ou pelo Monte Carlo"
    )
else:
    alpha = 0.5  # valor padrão

if st.button("Gerar Alocação Otimizada"):
    try:
        # --- Coletar ativos válidos e retornos ---
        ativos_validos = filtrar_ativos_validos(
            carteira, setores_por_ticker, setores_por_cenario, macro, calcular_score
        )
        if not ativos_validos:
            st.warning("Nenhum ativo com preço atual abaixo do preço-alvo dos analistas.")
            st.session_state['pesos_opcoes'] = None
            st.session_state['ativos_validos_aporte'] = None
            st.session_state['aporte_valor'] = None
            st.stop()
        
        favorecimentos = {a['ticker']: a['favorecido'] for a in ativos_validos}
        tickers_validos = [a['ticker'] for a in ativos_validos]
        retornos = obter_preco_diario_ajustado(tickers_validos).pct_change().dropna()
        media_retorno = retornos.mean() * 252
        cov = retornos.cov() * 252

        # --- Simulação Monte Carlo (Fronteira Eficiente) ---
        df_front = calcular_fronteira_eficiente_macro(
            retornos=retornos,
            score_dict=favorecimentos,
            n_portfolios=100000
        )
        melhor_carteira = df_front.loc[df_front['Sharpe'].idxmax()]

        # --- Otimização Sharpe padrão ---
        pesos_sharpe = otimizar_carteira_sharpe(
            tickers_validos, carteira, favorecimentos=favorecimentos
        )

        # --- Otimização Sharpe usando seed Monte Carlo ---
        from scipy.optimize import minimize
        def sharpe_neg(pesos):
            retorno = np.dot(pesos, media_retorno)
            risco = np.sqrt(np.dot(pesos.T, np.dot(cov, pesos)))
            return - (retorno / risco) if risco > 0 else 0
        limites = tuple((0, 1) for _ in range(len(tickers_validos)))
        restricoes = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        pesos_seed_mc = np.array(melhor_carteira['Pesos'])
        res_mc = minimize(
            sharpe_neg,
            pesos_seed_mc,
            method='SLSQP',
            bounds=limites,
            constraints=restricoes,
            options={'disp': False, 'maxiter': 1000}
        )
        if res_mc.success:
            pesos_sharpe_mc = res_mc.x
        else:
            pesos_sharpe_mc = pesos_sharpe.to_numpy()

        # --- HRP ---
        pesos_hrp = otimizar_carteira_hrp(
            tickers_validos, carteira, favorecimentos=favorecimentos
        )

        # --- Monte Carlo puro (Fronteira) ---
        pesos_mc = pd.Series(melhor_carteira['Pesos'], index=retornos.columns)
        pesos_hrp_series = pesos_hrp if isinstance(pesos_hrp, pd.Series) else pd.Series(pesos_hrp, index=retornos.columns)

        # --- HRP + Monte Carlo combinado ---
        pesos_combinados = alpha * pesos_hrp_series + (1 - alpha) * pesos_mc
        pesos_combinados /= pesos_combinados.sum()

        # --- Dicionário de opções ---
        pesos_opcoes = {
            "Sharpe (macro)": pesos_sharpe,
            "Sharpe (Monte Carlo)": pd.Series(pesos_sharpe_mc, index=retornos.columns),
            "HRP": pesos_hrp_series,
            "Monte Carlo (Melhor Simulada)": pesos_mc,
            "HRP + Monte Carlo": pesos_combinados
        }
        st.session_state['pesos_opcoes'] = pesos_opcoes
        st.session_state['ativos_validos_aporte'] = ativos_validos
        st.session_state['aporte_valor'] = aporte
        st.session_state['pesos_mc'] = pesos_mc
        st.session_state['melhor_carteira'] = melhor_carteira
    except Exception as e:
        st.error(f"Erro na otimização: {str(e)}")
        st.session_state['pesos_opcoes'] = None
        st.session_state['ativos_validos_aporte'] = None
        st.session_state['aporte_valor'] = None

# --- Exibir tabela de aportes sempre que houver resultado salvo ---
if (
    st.session_state.get('pesos_opcoes')
    and st.session_state.get('ativos_validos_aporte')
    and st.session_state.get('aporte_valor')
):
    pesos_recomendados = st.session_state['pesos_opcoes'][st.session_state['metodo_aporte']]
    ativos_validos = st.session_state['ativos_validos_aporte']
    aporte = st.session_state['aporte_valor']

    df_validos = pd.DataFrame(ativos_validos)
    df_resultado = df_validos[['ticker', 'setor', 'preco_atual', 'preco_alvo', 'score']].copy()
    df_resultado["peso_otimizado"] = df_resultado["ticker"].map(pesos_recomendados).fillna(0)
    df_resultado["Valor Alocado Bruto (R$)"] = df_resultado["peso_otimizado"] * aporte
    df_resultado["Qtd. Ações"] = (df_resultado["Valor Alocado Bruto (R$)"] / df_resultado["preco_atual"])\
        .replace([np.inf, -np.inf], 0).fillna(0).apply(np.floor)
    df_resultado["Valor Alocado (R$)"] = (df_resultado["Qtd. Ações"] * df_resultado["preco_atual"]).round(2)
    df_resultado = df_resultado[df_resultado["Qtd. Ações"] > 0]

    st.subheader("📈 Ativos Recomendados para Novo Aporte")
    st.dataframe(df_resultado[[
        "ticker", "setor", "preco_atual", "preco_alvo", "score", "Qtd. Ações",
        "Valor Alocado (R$)"
    ]], use_container_width=True)
    valor_utilizado = df_resultado["Valor Alocado (R$)"].sum()
    troco = aporte - valor_utilizado

    # --- Métricas da carteira após aporte ---
    tickers_aporte = df_resultado["ticker"].tolist()
    pesos_aporte = df_resultado["peso_otimizado"].values
    if len(tickers_aporte) >= 2:
        cagr, risco, sharpe = calcular_metricas_carteira(tickers_aporte, pesos_aporte)
        st.markdown(f"💰 **Valor utilizado no aporte:** R$ {valor_utilizado:,.2f}")
        st.markdown(f"🔁 **Troco (não alocado):** R$ {troco:,.2f}")
        st.markdown(f"**CAGR estimado (7 anos):** {100*cagr:.2f}% ao ano")
        st.markdown(f"**Risco anualizado:** {100*risco:.2f}%")
        st.markdown(f"**Índice de Sharpe:** {sharpe:.2f}")
    else:
        st.markdown(f"💰 **Valor utilizado no aporte:** R$ {valor_utilizado:,.2f}")
        st.markdown(f"🔁 **Troco (não alocado):** R$ {troco:,.2f}")
        st.info("Métricas da carteira precisam de pelo menos 2 ativos.")

    # --- Exibe composição dos pesos HRP/MonteCarlo/Combinado ---
    if st.session_state['metodo_aporte'] == "HRP + Monte Carlo":
        st.subheader("🔗 Pesos Individuais e Combinados")
        st.write(pd.DataFrame({
            "HRP": pesos_hrp_series,
            "Monte Carlo": pesos_mc,
            "Combinado": pesos_combinados
        }).round(4))

    # --- Mostra a carteira da fronteira eficiente (Monte Carlo) ---
    if 'Monte Carlo (Melhor Simulada)' in st.session_state['pesos_opcoes']:
        st.subheader("⭐ Carteira da Fronteira Eficiente (Monte Carlo)")
        pesos_mc = st.session_state['pesos_opcoes']["Monte Carlo (Melhor Simulada)"]
        df_resultado_mc = df_validos[['ticker', 'setor', 'preco_atual', 'preco_alvo', 'score']].copy()
        df_resultado_mc["peso_monte_carlo"] = df_resultado_mc["ticker"].map(pesos_mc).fillna(0)
        df_resultado_mc["Valor Alocado Bruto (R$)"] = df_resultado_mc["peso_monte_carlo"] * aporte
        df_resultado_mc["Qtd. Ações"] = (df_resultado_mc["Valor Alocado Bruto (R$)"] / df_resultado_mc["preco_atual"])\
            .replace([np.inf, -np.inf], 0).fillna(0).apply(np.floor)
        df_resultado_mc["Valor Alocado (R$)"] = (df_resultado_mc["Qtd. Ações"] * df_resultado_mc["preco_atual"]).round(2)
        df_resultado_mc = df_resultado_mc[df_resultado_mc["Qtd. Ações"] > 0]
        st.dataframe(df_resultado_mc[[
            "ticker", "setor", "preco_atual", "preco_alvo", "score", "peso_monte_carlo",
            "Qtd. Ações", "Valor Alocado (R$)"
        ]], use_container_width=True)
        valor_utilizado_mc = df_resultado_mc["Valor Alocado (R$)"].sum()
        troco_mc = aporte - valor_utilizado_mc

        # --- Métricas da carteira Monte Carlo (Fronteira) ---
        tickers_mc = df_resultado_mc["ticker"].tolist()
        pesos_mc_vec = df_resultado_mc["peso_monte_carlo"].values
        if len(tickers_mc) >= 2:
            cagr_mc, risco_mc, sharpe_mc = calcular_metricas_carteira(tickers_mc, pesos_mc_vec)
            st.markdown(f"💰 **Valor utilizado na carteira eficiente:** R$ {valor_utilizado_mc:,.2f}")
            st.markdown(f"🔁 **Troco (não alocado):** R$ {troco_mc:,.2f}")
            st.markdown(f"**CAGR estimado (7 anos):** {100*cagr_mc:.2f}% ao ano")
            st.markdown(f"**Risco anualizado:** {100*risco_mc:.2f}%")
            st.markdown(f"**Índice de Sharpe:** {sharpe_mc:.2f}")
        else:
            st.markdown(f"💰 **Valor utilizado na carteira eficiente:** R$ {valor_utilizado_mc:,.2f}")
            st.markdown(f"🔁 **Troco (não alocado):** R$ {troco_mc:,.2f}")
            st.info("Métricas da carteira precisam de pelo menos 2 ativos.")



    # --- Mostra a carteira integral após o aporte (com pesos iniciais, finais e recomendados) ---
    # --- Mostra a carteira integral após o aporte (com pesos iniciais, finais e recomendados corrigidos) ---
       # --- Mostra a carteira integral após o aporte (com pesos iniciais, finais e recomendados corrigidos) ---
    # --- Mostra a carteira inicial e os ajustes após o aporte ---
    st.subheader("📦 Carteira Inicial e Ajustada Após o Aporte")
    
    # Calcula o valor total inicial com base nos pesos fornecidos pelo usuário
    valor_total_inicial = sum(
        pesos_atuais[i] * carteira_integral[tickers[i]].get("preco_atual", 0)
        for i in range(len(tickers))
        if carteira_integral[tickers[i]].get("preco_atual", 0) > 0
    )
    
    # Calcula o valor total final com base nas quantidades e preços após o aporte
    valor_total_final = sum(
        v["quantidade"] * v.get("preco_atual", 0)
        for v in carteira_integral.values()
        if isinstance(v, dict) and v.get("quantidade", 0) > 0 and v.get("preco_atual", 0) > 0
    )
    
    # Preenche os dados para todos os ativos, incluindo os que não receberam aportes
    dados_integral = []
    for i, t in enumerate(tickers):
        v = carteira_integral.get(t, {"quantidade": 0, "preco_atual": 0})
        preco_atual = v.get("preco_atual", 0)
        peso_recomendado = pesos_recomendados.get(t, 0)  # Peso recomendado pela carteira otimizada
        peso_inicial = pesos_atuais[i] if i < len(pesos_atuais) else 0  # Peso inicial fornecido pelo usuário
        peso_final = (v.get("quantidade", 0) * preco_atual) / valor_total_final if valor_total_final > 0 else 0
        dados = {
            "ticker": t,
            "setor": v.get("setor", ""),
            "quantidade_inicial": round((peso_inicial * valor_total_inicial) / preco_atual) if preco_atual > 0 else 0,
            "quantidade_final": v.get("quantidade", 0),
            "preco_atual": preco_atual,
            "preco_alvo": v.get("preco_alvo", 0),
            "score": v.get("score", 0),
            "peso_inicial": peso_inicial,  # Peso inicial do usuário
            "peso_recomendado": peso_recomendado,  # Peso recomendado pela carteira otimizada
            "peso_final": peso_final,  # Peso final após o aporte
        }
        dados_integral.append(dados)
    
    df_carteira_integral = pd.DataFrame(dados_integral)
    
    # Seleciona as colunas desejadas
    colunas = [
        "ticker", "setor", "quantidade_inicial", "quantidade_final", "preco_atual",
        "preco_alvo", "score", "peso_inicial", "peso_recomendado", "peso_final"
    ]
    
    # Exibe a tabela com pesos iniciais, recomendados e finais
    st.dataframe(
        df_carteira_integral[colunas].sort_values(by="peso_final", ascending=False),
        use_container_width=True
    )

    # --- Backtest plug and play ---
    if len(df_resultado) >= 2:
        st.subheader("📊 Backtest: Carteira Recomendada vs IBOV (7 anos)")
        tickers_bt = df_resultado["ticker"].tolist()
        pesos_bt = df_resultado["peso_otimizado"].values
        backtest_portfolio_vs_ibov_duplo(tickers_bt, pesos_bt)

    # --- Histórico ajustado plug and play ---
    if "cenario_atual" in globals():
        mostrar_historico_ajustado(
            tickers=list(setores_por_ticker.keys()),
            setores_por_ticker=setores_por_ticker,
            cenario_atual=cenario_atual,
            anos=7
        )
