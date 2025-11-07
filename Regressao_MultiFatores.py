from functions import *
import os

# DEFINIÇÃO DE VARIAVEIS GLOBAIS
TICKER = '' # Nome do ativo que irá no modelo (y)
SETOR = [] # Nome dos ativos do mesmo setor
INICIO, FIM = '2015-01-01', '2025-06-30'

# Pega chaves das API's nas variaveis do sitema
fred_api_key = os.getenv("FRED_API_KEY")
if fred_api_key is None:
    raise ValueError("Variável FRED_API_KEY não encontrada. Configure no sistema ou .env")


# Obtem os dados do ativo alvo (y)


# Obtem os dados dos fatores (x)
brent = filter_data(get_api_fred(api_key=fred_api_key, series_id='DCOILBRENTEU'),start=INICIO,end=FIM)
print(brent)