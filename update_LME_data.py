import pandas as pd

def fetch_westmetall_lme(metal: str) -> pd.DataFrame:
    """
    Baixa e limpa a tabela completa do Westmetall (LME Copper)
    retornando um DataFrame com índice Date e colunas:
    ['Cash', '3M', 'Stock'].
    """

    url = f"https://www.westmetall.com/en/markdaten.php?action=table&field={metal}"

    # Lê todas as tabelas da página (uma por ano)
    dfs = pd.read_html(url, decimal=".", thousands=",")

    # Concatena todas em uma única tabela
    df = pd.concat(dfs, ignore_index=True)

    # Força nomes de colunas (Westmetall sempre usa esta estrutura)
    df.columns = ["Date", "Cash", "3M", "Stock"]

    # -------- Remoção das linhas de cabeçalho repetidas ----------
    # Filtra apenas as linhas cuja 'Date' parece uma data real
    # Formato típico: "14. November 2025"
    mask = df["Date"].astype(str).str.match(r"^\d{2}\.\s+[A-Za-z]+\s+\d{4}$")
    df = df[mask].copy()

    # -------- Conversão de tipos ----------
    # Converte coluna Date para datetime
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])  # remove datas inválidas

    # Converte valores numéricos
    for col in ["Cash", "3M", "Stock"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # -------- Finalização ----------
    df = df.dropna(subset=["Cash", "3M", "Stock"], how="all")  # caso venha alguma linha vazia
    df = df.set_index("Date").sort_index()

    return df

df_copper_lme = fetch_westmetall_lme(metal='LME_Cu_cash')
df_aluminium_lme = fetch_westmetall_lme(metal='LME_Al_cash')
dfs_lme = {
    'Copper': df_copper_lme,
    'Aluminium': df_aluminium_lme
}
for key in dfs_lme:
    for column in dfs_lme[key].columns:
        dfs_lme[key][column].to_csv(f'datasets\{column}_{key}_LME.csv') 