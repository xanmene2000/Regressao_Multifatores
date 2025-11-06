import numpy as np, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.gofplots import qqplot
import requests
import datetime as dt
from urllib.parse import quote


def get_nefin_br_values(path='datasets\nefin_factors_br.csv'):
    # Leitura do CSV
    nefin_daily = pd.read_csv(path)

    # Padroniza colunas (garante nomes consistentes)
    nefin_daily.columns = [
        "Index", "Date", "Rm_minus_Rf", "SMB", "HML", "WML", "IML", "Risk_Free"
    ]

    # Converte datas e define como índice
    nefin_daily["Date"] = pd.to_datetime(nefin_daily["Date"])
    nefin_daily = nefin_daily.set_index("Date").sort_index()

    nefin_daily = nefin_daily.drop(columns=["Index", "Rm_minus_Rf", "Risk_Free"])


    # Remove duplicatas e converte para float
    cols_num = ["SMB", "HML", "WML", "IML"]
    nefin_daily[cols_num] = nefin_daily[cols_num].astype(float)
    return nefin_daily
    