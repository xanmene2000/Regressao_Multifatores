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

def filter_data(df, start, end):
    return df.loc[(df.index >= start) & (df.index <= end)]


def _fmt_bcb_date(s):
    return pd.to_datetime(s).strftime("%d/%m/%Y")

def get_bcb_series(codigo_sgs: int, start, end):
    """
    Busca série do SGS/BCB. Datas devem ser dd/mm/aaaa.
    Se der 400, tenta sem datas e filtra localmente.
    Retorna DataFrame com índice datetime e coluna 'valor' (float).
    """
    start_bcb = _fmt_bcb_date(start)
    end_bcb   = _fmt_bcb_date(end)
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_sgs}/dados?formato=json&dataInicial={quote(start_bcb)}&dataFinal={quote(end_bcb)}"

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = pd.DataFrame(r.json())
    except requests.HTTPError:
        # Fallback: busca tudo e filtra localmente
        url_all = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo_sgs}/dados?formato=json"
        r = requests.get(url_all, timeout=30)
        r.raise_for_status()
        data = pd.DataFrame(r.json())

    if data.empty:
        raise ValueError(f"Série {codigo_sgs} sem dados.")

    data["data"] = pd.to_datetime(data["data"], dayfirst=True)
    data["valor"] = pd.to_numeric(data["valor"].str.replace(",", ".", regex=False), errors="coerce")
    data = data.dropna().set_index("data").sort_index()

    # Filtra localmente
    data = filter_data(data, start, end)
    return data

def prepare_ipca(start, end):
    """
    IPCA var. mensal (%), código 433. Converte para DECIMAL ao mês.
    """
    df = get_bcb_series(433, start, end)
    s = df["valor"] / 100.0
    s.name = "IPCA"
    return s


def monthly_to_daily_equivalent(data_monthly: pd.Series, target_index, columnName) -> pd.Series:
    """
    Converte dados mensais (decimal ao mês) para taxa diária equivalente,
    distribuindo por *exatamente* os dias presentes em `target_index`.
    A composição diária de cada mês reproduz o IPCA mensal original.
    """
    # 1) índice diário alvo, ordenado
    idx = pd.DatetimeIndex(target_index).sort_values()
    df = pd.DataFrame(index=idx)
    df["month"] = df.index.to_period("M")

    # 2) traz a taxa mensal para cada linha (mês correspondente)
    m = data_monthly.copy().dropna()
    m.index = m.index.to_period("M")
    df = df.join(m.rename("m_rate"), on="month")

    # 3) conta de dias (do SEU índice) em cada mês
    df["n_in_month"] = df.groupby("month")["month"].transform("size")

    # 4) taxa diária equivalente que recompõe exatamente o mês
    df["data_daily"] = (1.0 + df["m_rate"]).pow(1.0 / df["n_in_month"]) - 1.0

    out = df["data_daily"].rename(columnName)
    return out


def prepare_pib_proxy(start, end):
    """
    Proxy de PIB: IBC-Br dessazonalizado (cód. 24363, nível-índice).
    Transformamos em CRESCIMENTO M/M: pct_change mensal (decimal).
    """
    df = get_bcb_series(24363, start, end)
    level = df["valor"]
    s = level.pct_change()
    s.name = "PIB"
    return s


def get_usd_ptax(start, end):
    """
    Retorna a variação percentual diária (USD_PTAX_RET) da PTAX (venda) entre 'start' e 'end'.
    Mantém apenas a data (sem hora) e um valor por dia (último do dia).
    """
    # converte datas para formato aceito pela API (MM-DD-YYYY)
    def fmt(d):
        if isinstance(d, str):
            for f in ("%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y"):
                try:
                    d = datetime.strptime(d, f)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError("Data inválida. Use 'YYYY-MM-DD' ou 'DD/MM/YYYY'.")
        return d.strftime("%m-%d-%Y")

    start_fmt = fmt(start)
    end_fmt   = fmt(end)

    # chamada da API PTAX
    url = (
        "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/"
        f"CotacaoDolarPeriodo(dataInicial=@dataInicial,dataFinalCotacao=@dataFinalCotacao)?"
        f"@dataInicial='%s'&@dataFinalCotacao='%s'&$top=10000&$format=json&"
        "$select=cotacaoVenda,dataHoraCotacao"
    ) % (start_fmt, end_fmt)

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = pd.DataFrame(r.json().get("value", []))
    if data.empty:
        raise ValueError("Sem dados PTAX no período informado.")

    # === trata datas como string, remove horas ===
    data["dataHoraCotacao"] = data["dataHoraCotacao"].astype(str)
    data["date"] = data["dataHoraCotacao"].str.split("T| ").str[0]
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")

    # === converte cotação e mantém último valor do dia ===
    data["cotacaoVenda"] = pd.to_numeric(data["cotacaoVenda"], errors="coerce")
    daily = data.groupby("date")["cotacaoVenda"].last().sort_index()

    # === calcula retorno diário ===
    ptax_ret = daily.pct_change()
    ptax_ret.iloc[0] = ptax_ret.iloc[1]  # define o primeiro dia como 0
    ptax_ret.name = "USD_PTAX"

    return ptax_ret
    