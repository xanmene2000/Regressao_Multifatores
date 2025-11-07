from functions import *
import os

# DEFINIÇÃO DE VARIAVEIS GLOBAIS
TICKER = '' # Nome do ativo que irá no modelo (y)
SETOR = [] # Nome dos ativos do mesmo setor
INICIO, FIM = '2014-12-31', '2025-06-30'

# Pega chaves das API's nas variaveis do sitema
fred_api_key = os.getenv("FRED_API_KEY")
if fred_api_key is None:
    raise ValueError("Variável FRED_API_KEY não encontrada. Configure no sistema ou .env")


# Obtem os dados do ativo alvo (y)


# Obtem os dados dos fatores (x)
# Usar reindex e ffill nos dados originais para alinhar com o indice do ativo alvo
brent = filter_data(get_api_fred(api_key=fred_api_key, series_id='DCOILBRENTEU'),start=INICIO,end=FIM)
brent_ret = brent.pct_change()

UST_10Y = filter_data(get_api_fred(series_id='DGS10',api_key=fred_api_key)/100,start=INICIO,end=FIM)
UST_10Y_diff = UST_10Y.diff()

