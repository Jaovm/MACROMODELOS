import pandas as pd
import numpy as np
import requests
import streamlit as st

# API key para a FMP
api_key = "rd6uBzkLLSPG68s9GcSx3folN76IxRhV"

# Função para obter os preços históricos das ações a partir da API FMP
def get_stock_data(tickers, data_inicio):
    # Converte a lista de tickers em uma string separada por vírgula
    tickers_str = ",".join(tickers)
    
    # URL da API da FMP para obter os dados históricos
    url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{tickers_str}?from={data_inicio}&apikey={api_key}'
    
    # Fazendo a requisição à API
    response = requests.get(url)
    data = response.json()
    
    # Extraindo os dados históricos de cada ticker
    historico = {}
    
    for ticker in data['symbol']['historical']:
        # Extraindo a data e o fechamento ajustado
        dates = [entry['date'] for entry in ticker['historical']]
        adj_close = [entry['adjClose'] for entry in ticker['historical']]
        historico[ticker['symbol']] = pd.Series(data=adj_close, index=pd.to_datetime(dates))
    
    # Transformando em DataFrame
    precos = pd.DataFrame(historico)
    
    return precos

# Função para calcular probabilidade de alta nos próximos 12 meses
def calcular_probabilidade_alta(precos):
    # Calculando retornos diários
    retornos = precos.pct_change().dropna()
    
    # Calculando a probabilidade de alta (simples, assumindo distribuição normal)
    probabilidade_alta = (retornos.mean() + 2 * retornos.std()).mean()  # Aproximando probabilidade de alta
    
    return probabilidade_alta

# Função para calcular a alocação ótima de carteira (usando uma técnica de otimização simples)
def alocacao_otima(precos):
    retornos = precos.pct_change().dropna()
    cov_matrix = retornos.cov()
    media_retornos = retornos.mean()
    
    # Alocação ótima por meio da maximização de retorno (simplificada)
    alocacao = media_retornos / media_retornos.sum()  # Alocação proporcional ao retorno esperado
    
    return alocacao

# Função para obter dados da API FMP para o cenário macroeconômico
def get_macroeconomic_data():
    url = f'https://financialmodelingprep.com/api/v3/economic_indicators?apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    return data

# Função para classificar o cenário macroeconômico com base em PIB e Selic
def classificar_cenario_macro():
    data = get_macroeconomic_data()
    
    # Simulação simples de classificação do cenário baseado no PIB e na Selic
    pib = data['economicIndicators'][0]['value']  # PIB em valor (exemplo)
    selic = data['economicIndicators'][1]['value']  # Taxa Selic
    
    if selic > 10 and pib < 2:
        return 'Restritivo'
    elif selic < 5 and pib > 4:
        return 'Expansionista'
    else:
        return 'Neutro'

# Função para sugerir compras com base no cenário macroeconômico
def sugerir_compras(ativos, cenario):
    # Exemplo simples de sugestão de compras com base no cenário
    if cenario == 'Expansionista':
        ativos_recomendados = ativos[ativos['setor'] == 'Tecnologia']
    elif cenario == 'Restritivo':
        ativos_recomendados = ativos[ativos['setor'] == 'Energia']
    else:
        ativos_recomendados = ativos[ativos['setor'] == 'Saúde']
    
    return ativos_recomendados

# Dados de exemplo: tickers e seus setores
ativos = pd.DataFrame({
    'ticker': ['AGRO3.SA', 'BBAS3.SA', 'BBSE3.SA', 'BPAC11.SA', 'EGIE3.SA', 'ITUB3.SA', 'PRIO3.SA', 'PSSA3.SA', 'SAPR4.SA', 'SBSP3.SA', 'VIVT3.SA', 'WEGE3.SA', 'TOTS3.SA', 'B3SA3.SA', 'TAEE3.SA'],
    'setor': ['Agronegócio', 'Bancos', 'Seguradoras', 'Financeiro', 'Energia', 'Bancos', 'Petróleo', 'Seguradoras', 'Energia', 'Utilidades', 'Comunicação', 'Indústria', 'Tecnologia', 'Bolsas', 'Energia']
})

# Exemplo de tickers
tickers = ativos['ticker'].tolist()

# Obtendo dados históricos
data_inicio = "2015-01-01"
precos = get_stock_data(tickers, data_inicio)

# Calculando a probabilidade de alta
probabilidade_alta = calcular_probabilidade_alta(precos)

# Calculando a alocação ótima
alocacao = alocacao_otima(precos)

# Classificando o cenário macroeconômico
cenario_macro = classificar_cenario_macro()

# Sugerindo compras com base no cenário
compras_recomendadas = sugerir_compras(ativos, cenario_macro)

# Exibindo resultados
print("Probabilidade de Alta nos Próximos 12 Meses:", probabilidade_alta)
print("Alocação Ótima da Carteira:")
print(alocacao)
print(f"Cenário Macroeconômico: {cenario_macro}")
print("Ativos Recomendados para Compra:")
print(compras_recomendadas)

# Streamlit display
st.write(f"**Cenário Macroeconômico Atual**: {cenario_macro}")
st.write("**Probabilidade de Alta (12 meses)**:", probabilidade_alta)
st.write("**Alocação Ótima da Carteira**:")
st.write(alocacao)
st.write("**Ativos Recomendados para Compra**:")
st.write(compras_recomendadas)
