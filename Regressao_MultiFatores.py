import functions as fn
import os

# DEFINIÇÃO DE VARIAVEIS GLOBAIS
TICKER = '' # Nome do ativo que irá no modelo (y)
SETOR = [] # Nome dos ativos do mesmo setor
INICIO, FIM = '2014-12-31', '2025-06-30'

# Pega chaves das API's nas variaveis do sitema
# Define todas as APIs usadas no projeto
REQUIRED_KEYS = [
    "FRED_API_KEY",
    "exchangeRate_API_KEY",
]

api_keys = fn.check_api_keys(REQUIRED_KEYS)

# Obtem os dados do ativo alvo (y)


# Obtem os dados dos fatores (x)
# Usar reindex e ffill nos dados originais para alinhar com o indice do ativo alvo
brent = fn.filter_data(fn.get_api_fred(api_key=api_keys["FRED_API_KEY"], series_id='DCOILBRENTEU'),start=INICIO,end=FIM)
brent_ret = brent.pct_change()

UST_10Y = fn.filter_data(fn.get_api_fred(series_id='DGS10',api_key=api_keys["FRED_API_KEY"])/100,start=INICIO,end=FIM)
UST_10Y_diff = UST_10Y.diff()

vix = fn.filter_data(fn.get_api_fred(series_id='VIXCLS', api_key=api_keys['FRED_API_KEY']), start=INICIO, end=FIM)
vix = fn.standardize(vix)
