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
import os
import sys


def standardize(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std()

def check_api_keys(required_keys: list[str]) -> dict[str, str]:
    """
    Verifica se todas as variáveis de ambiente das APIs foram definidas.
    Retorna um dicionário {nome_da_api: chave} ou encerra o programa com erro amigável.

    Parâmetros
    ----------
    required_keys : list
        Lista com os nomes exatos das variáveis de ambiente (ex: ["FRED_API_KEY", "EXCHANGE_API_KEY"]).

    Retorno
    -------
    dict : {key_name: key_value}
        Dicionário contendo as chaves lidas do ambiente.
    """
    missing = []
    keys = {}

    for k in required_keys:
        val = os.getenv(k)
        if not val:
            missing.append(k)
        else:
            keys[k] = val

    if missing:
        print("\nERRO: Uma ou mais variáveis de API não foram configuradas:\n")
        for m in missing:
            print(f"   - {m} (adicione no seu sistema ou .env)")
        print("\n💡 Exemplo de como configurar:")
        print('   setx FRED_API_KEY "sua_chave_aqui"   (Windows)')
        sys.exit(1)

    return keys

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

def periodic_to_daily_equivalent(
    data_periodic: pd.Series,
    target_index,
    column_name: str,
    return_type: str = "simple",  # "simple" ou "log"
    freq: str = "M",              # "M" para mensal, "W" para semanal
    release_lag: str = "shift1"   # "shift1" aplica a taxa no período seguinte (evita look-ahead)
) -> pd.Series:
    """
    Converte retornos periódicos (mensais ou semanais) em equivalentes diários,
    distribuídos apenas pelos dias do período-alvo. Opcionalmente aplica defasagem
    de divulgação para evitar look-ahead.
    - return_type: "simple" usa (1+r)^(1/n)-1 ; "log" usa g/n.
    - release_lag: "none" aplica no próprio período; "shift1" aplica no período seguinte.
    """
    # 1) índice diário alvo, ordenado
    idx = pd.DatetimeIndex(target_index).sort_values()
    df = pd.DataFrame(index=idx)

    # 2) período conforme freq
    if freq.upper().startswith("M"):
        period = df.index.to_period("M")
    elif freq.upper().startswith("W"):
        period = df.index.to_period("W-SUN")  # ajuste a âncora se quiser
    else:
        raise ValueError("freq deve ser 'M' (mensal) ou 'W' (semanal).")

    df["period"] = period

    # 3) prepara a série periódica
    s = data_periodic.copy().dropna()
    s.index = s.index.to_period("M" if freq.upper().startswith("M") else "W-SUN")

    # 4) aplica defasagem de divulgação para evitar look-ahead
    if release_lag == "shift1":
        s = s.shift(1)  # usa a taxa do mês/semana anterior no período corrente
    elif release_lag == "none":
        pass
    else:
        raise ValueError("release_lag deve ser 'none' ou 'shift1'.")

    # 5) traz a taxa do período para cada dia, conta n por período
    df = df.join(s.rename("r_period"), on="period")
    df["n_in_period"] = df.groupby("period")["period"].transform("size")

    # 6) distribui: simple vs log
    if return_type == "simple":
        # r_d = (1+r_m)^(1/n) - 1
        df[column_name] = (1.0 + df["r_period"]).pow(1.0 / df["n_in_period"]) - 1.0
    elif return_type == "log":
        # g_d = g_m / n ; se quiser simples depois: np.expm1(g_d)
        g_d = (df["r_period"] / df["n_in_period"])
        df[column_name] = g_d  # permaneça em log-return diário
    else:
        raise ValueError("return_type deve ser 'simple' ou 'log'.")

    return df[column_name]

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

def get_api_fred(series_id: str, api_key: str) -> pd.Series:
    """
    Baixa dados de uma série do FRED via API e retorna um pandas.Series limpo.

    Parâmetros:
    -----------
    series_id : str
        Código da série no FRED (ex: "DCOILBRENTEU" para Brent).
    api_key : str
        Chave da API FRED (pode vir de os.getenv("FRED_API_KEY")).
    
    Retorno:
    --------
    pandas.Series com índice datetime e valores float.
    """

    url = (
        f"https://api.stlouisfed.org/fred/series/observations?"
        f"series_id={series_id}&api_key={api_key}&file_type=json&sort_order=asc"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json().get("observations", [])
    except Exception as e:
        raise RuntimeError(f"Erro ao acessar a API FRED ({series_id}): {e}")

    if not data:
        raise ValueError(f"Nenhum dado retornado para a série {series_id}")

    # Converte em DataFrame
    df = pd.DataFrame(data)[["date", "value"]]

    # Trata valores
    df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "value"])

    # Define índice e frequência
    s = df.set_index("date")["value"].sort_index()
    
    s.name = series_id
    return s

def align_index(daily_series, monthly_series) -> pd.Series:
    # Alinhar com merge_asof
    df_ajusted = pd.merge_asof(
        daily_series.sort_index(),
        monthly_series.sort_index(),
        left_index=True,
        right_index=True,
        direction='backward'
    )
    return df_ajusted[monthly_series.name]

def read_pmi_china(path: str, column: str = 'ActualValue') -> pd.Series:
    df = pd.read_csv(path, sep='\t')

    # Converte datas e define como índice
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    df = df.drop(columns=["ForecastValue", "PreviousValue"])

    # Remove duplicatas e converte para float
    df[column] = df[column].astype(float)

    s = df[column].dropna()
    
    return s

def get_cftc_mm_nasdaq(api_key: str, code: str) -> pd.Series:
    """
    Retorna série semanal padronizada (z-score) do Net Managed Money
    para um contrato CFTC via Nasdaq Data Link (QDL/FON).

    - code: contract_code (string), ex: "085692" para cobre
    - Usa type=F_ALL (Futures Only, categoria ALL)
    - Faz: Net = longs - shorts
           shift(1) para evitar look-ahead
           padronização (média 0, desvio 1)
    """

    base_url = "https://data.nasdaq.com/api/v3/datatables/QDL/FON"

    params = {
        "contract_code": code,
        "type": "F_ALL",  # Futures Only, ALL
        "qopts.columns": "date,money_manager_longs,money_manager_shorts",
        "api_key": api_key,
    }

    r = requests.get(base_url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()["datatable"]

    if not j["data"]:
        raise ValueError(f"Nenhum dado retornado para contract_code={code}")

    cols = [c["name"] for c in j["columns"]]
    df = pd.DataFrame(j["data"], columns=cols)

    # datas e ordenação
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_values("date")

    # garante numérico
    df["money_manager_longs"] = pd.to_numeric(df["money_manager_longs"], errors="coerce")
    df["money_manager_shorts"] = pd.to_numeric(df["money_manager_shorts"], errors="coerce")

    # Net Managed Money
    df["mm_net"] = df["money_manager_longs"] - df["money_manager_shorts"]

    s = df["mm_net"]
    
    return s
