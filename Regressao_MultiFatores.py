import functions as fn
import pandas as pd
import os
from statsmodels.api import OLS, add_constant


# DEFINIÇÃO DE VARIAVEIS GLOBAIS
TICKER = '' # Nome do ativo que irá no modelo (y)
SETOR = [] # Nome dos ativos do mesmo setor
INICIO, FIM = '2014-11-30', '2025-10-30'

# Pega chaves das API's nas variaveis do sitema
# Define todas as APIs usadas no projeto
REQUIRED_KEYS = [
    "FRED_API_KEY",
    "NASDAQ_API_KEY",
]

api_keys = fn.check_api_keys(REQUIRED_KEYS)

# Obtem os dados do ativo alvo (y)
copper_3M = fn.filter_data(fn.get_datasets_csv(path="datasets/3M_Copper_LME.csv", name="COPPER_3M"), start=INICIO, end=FIM)
copper_3M = copper_3M.pct_change()

aluminium_3M = fn.filter_data(fn.get_datasets_csv(path="datasets/3M_Aluminium_LME.csv", name="ALUMINIUM_3M"), start=INICIO, end=FIM)
aluminium_3M = aluminium_3M.pct_change()

# Obtem os dados dos fatores (x)
# Usar reindex e ffill nos dados originais para alinhar com o indice do ativo alvo
brent = fn.filter_data(fn.get_api_fred(api_key=api_keys["FRED_API_KEY"], series_id='DCOILBRENTEU', name="BRENT"),start=INICIO,end=FIM)
brent = brent.pct_change()

UST_10Y = fn.filter_data(fn.get_api_fred(series_id='DGS10',api_key=api_keys["FRED_API_KEY"], name="UST_10Y")/100,start=INICIO,end=FIM)
UST_10Y= UST_10Y.diff()

vix = fn.filter_data(fn.get_api_fred(series_id='VIXCLS', api_key=api_keys['FRED_API_KEY'], name="VIX"), start=INICIO, end=FIM)
vix = fn.standardize(vix)

dxy = fn.filter_data(fn.get_api_fred(api_key=api_keys["FRED_API_KEY"], series_id='DTWEXBGS', name="DXY"),start=INICIO,end=FIM)
dxy = dxy.pct_change()

usd_cny = fn.filter_data(fn.get_api_fred(api_key=api_keys["FRED_API_KEY"], series_id='DEXCHUS', name="USD/CNY"),start=INICIO,end=FIM)
usd_cny = usd_cny.pct_change()

ind_prod = fn.filter_data(fn.get_api_fred(series_id='INDPRO', api_key=api_keys['FRED_API_KEY'], name="IND_PROD"), start=INICIO, end=FIM)
ind_prod = fn.align_index(daily_series=dxy, monthly_series=fn.standardize(ind_prod).shift(1)).dropna()

pmi_china = fn.filter_data(fn.read_pmi_china(path='datasets\china-caixin-manufacturing-pmi.csv'),start=INICIO, end=FIM)
pmi_china = fn.align_index(daily_series=dxy, monthly_series=fn.standardize(pmi_china).shift(1)).dropna()

cftc_mm_copper = fn.filter_data(fn.get_cftc_mm_nasdaq(api_key=api_keys['NASDAQ_API_KEY'],code='085692', name="MM_COPPER"),start=INICIO, end=FIM)
cftc_mm_copper = fn.align_index(daily_series=dxy, monthly_series=fn.standardize(cftc_mm_copper).shift(1)).dropna()

copper_cash = fn.filter_data(fn.get_datasets_csv(path="datasets/Cash_Copper_LME.csv", name="COPPER_CASH"), start=INICIO, end=FIM)
cash_3M_copper = fn.standardize(copper_3M['COPPER_3M'] - copper_cash['COPPER_CASH']).rename('CASH_3M_COPPER')

aluminium_cash = fn.filter_data(fn.get_datasets_csv(path="datasets/Cash_Aluminium_LME.csv", name="ALUMINIUM_CASH"), start=INICIO, end=FIM)
cash_3M_aluminium = fn.standardize(aluminium_3M['ALUMINIUM_3M'] - aluminium_cash['ALUMINIUM_CASH']).rename('CASH_3M_ALUMINIUM')

inventory_lme_copper = fn.filter_data(fn.get_datasets_csv(path="datasets/Stock_Copper_LME.csv", name="INVENTORY_COPPER_LME"), start=INICIO, end=FIM)
inventory_lme_copper = fn.standardize(inventory_lme_copper)

inventory_shfe_copper = fn.filter_data(fn.get_datasets_csv(path="datasets/Stock_Copper_SHFE.csv", name="INVENTORY_COPPER_SHFE"), start=INICIO, end=FIM)
inventory_shfe_copper = fn.standardize(inventory_shfe_copper)


# Cria a variavel X_all do modelo, contendo todos fatores
# Depois, para cada commoditie, usará fatores especificos desta variavel
panel = pd.concat([
    copper_3M,
    brent,
    UST_10Y,
    vix,
    dxy,
    usd_cny,
    ind_prod,
    pmi_china,
    cftc_mm_copper
], axis=1).dropna()

panel_copper = pd.concat([
    copper_3M,
    cash_3M_copper,
    inventory_lme_copper, inventory_shfe_copper,
    cftc_mm_copper,
    brent,
    vix,
    UST_10Y,
    ind_prod,
    dxy,
    pmi_china,
], axis=1).dropna()

# Estimação OLS multifatorial
Y = panel_copper['COPPER_3M']
X = add_constant(panel_copper.drop(columns='COPPER_3M'), has_constant="add")
ols = OLS(Y, X).fit()
print(ols.summary())

