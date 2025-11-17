import time
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# ---------------------------------------------------------------------
# CONFIGURAÇÃO DOS ESTOQUES (SHFE / COMEX)
# ---------------------------------------------------------------------

INVENTORY_SOURCES = {
    "Copper_SHFE":  ("https://commoditieschart.net/metals/copper/shfe-copper-stocks",  "Stock_Copper_SHFE.csv"),
    "Copper_COMEX": ("https://commoditieschart.net/metals/copper/comex-copper-stocks", "Stock_Copper_COMEX.csv"),
    "Aluminium_SHFE":  ("https://commoditieschart.net/metals/aluminium/shfe-aluminium-stocks",  "Stock_Aluminium_SHFE.csv"),
    "Aluminium_COMEX": ("https://commoditieschart.net/metals/aluminium/comex-aluminium-stocks", "Stock_Aluminium_COMEX.csv"),
}


# ---------------------------------------------------------------------
# SELENIUM HELPER: inicia driver
# ---------------------------------------------------------------------
def start_driver():
    chrome_path = r"C:\Users\xanme\AppData\Local\SeleniumBasic\chromedriver.exe"  # <<<< coloque o seu caminho aqui

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(executable_path=chrome_path)
    driver = webdriver.Chrome(service=service, options=options)
    return driver


# ---------------------------------------------------------------------
# FUNÇÃO PRINCIPAL DE SCRAPING DO COMMODITIESCHART
# ---------------------------------------------------------------------
def fetch_inventory_commoditieschart(url: str, last_date=None) -> pd.DataFrame:
    """
    - Abre página
    - Clica no botão 'Table'
    - Extrai tabela
    - Percorre paginação ('Next') se existir
    - Retorna DataFrame: Date, Value
    """

    driver = start_driver()
    driver.get(url)

    wait = WebDriverWait(driver, 3)

    # 1. Clicar no botão TABLE para mostrar a tabela
    table_btn = wait.until(
        EC.element_to_be_clickable((By.XPATH, "//div[contains(@class,'cursor-pointer') and normalize-space(text())='Table']"))
    )
    table_btn.click()
    time.sleep(1)  # deixa carregar tabela

    rows = []

    def extract_current_page(last_date):
        
        """Extrai as linhas da página atual (grid de 4 colunas em divs)."""
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        # container scrollável da tabela
        container = soup.find(
            "div",
            class_=lambda c: c
            and "flex" in c.split()
            and "flex-col" in c.split()
            and "overflow-y-auto" in c,
        )
        if container is None:
            print("Não encontrou o container")
            return []

        new_rows = []
        reached_old = False

        # cada linha é um div grid grid-cols-4 py-1 ...
        for row_div in container.find_all(
            "div",
            class_=lambda c: c
            and "grid" in c.split()
            and "grid-cols-4" in c.split()
            and "py-1" in c,
        ):
            cells = [d.get_text(strip=True) for d in row_div.find_all("p")]
            if len(cells) < 2:
                continue

            date_str_raw, value_str = cells[0], cells[1]
            date_str = pd.Series([date_str_raw]).str.replace(r",(\d{4})$", r", \1", regex=True).iloc[0]
            dt = pd.to_datetime(date_str, format="%b %d, %Y", errors="coerce")

            if pd.isna(dt):
                continue

            # se temos last_date, checa se já chegou em dado antigo
            if last_date is not None and dt <= last_date:
                reached_old = True
                return new_rows, reached_old  # não adiciona; já existe no CSV

            new_rows.append((dt, value_str))

        return new_rows, reached_old

    # 2. Extrair primeira página
    page_rows, reached_old = extract_current_page(last_date)
    rows.extend(page_rows)

    # 3. Paginação via botão NEXT
    while not reached_old:
        print("Entrou no loop")
        try:
            # botão Next é <button>, não <a>
            next_btn = driver.find_element(By.XPATH, "//button[normalize-space()='Next']")
        except Exception:
            break  # não existe botão → fim

        # checar se o botão realmente está desabilitado
        if next_btn.get_attribute("disabled") is not None:
            break  # acabou as páginas

        # clicar via javascript (mais robusto)
        driver.execute_script("arguments[0].click();", next_btn)
        time.sleep(1)

        rows.extend(extract_current_page())

    driver.quit()

    if not rows:
        return pd.DataFrame(columns=["Date", "Value"])

    # 4. Converte rows → DataFrame
    df = pd.DataFrame(rows, columns=["Date", "Value"])

    df["Value"] = (
         df["Value"]
         .str.replace(",", "")
         .astype(float)
    )

    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    return df


# ---------------------------------------------------------------------
# SALVAR CSV
# ---------------------------------------------------------------------
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)


def load_existing_inventory(filename: str) -> pd.DataFrame:
    """
    Lê CSV existente em datasets/<filename> se houver.
    Retorna DataFrame com colunas ['Date','Value'] ordenado.
    """
    path = DATASETS_DIR / filename
    if not path.exists():
        return pd.DataFrame(columns=["Date", "Value"])

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    return df


def save_inventory(df: pd.DataFrame, filename: str):
    """
    Salva DataFrame ['Date','Value'] em datasets/<filename>.
    """
    path = DATASETS_DIR / filename
    out = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    out.to_csv(path, index=False)
    print(f"[OK] Salvo: {filename} ({len(out)} linhas)")



def update_inventory(name: str, url: str, filename: str):
    """
    name      -> só pra print (ex.: 'Copper_SHFE')
    url       -> URL do commoditieschart
    filename  -> nome do CSV em datasets/ (ex.: 'Stock_Copper_SHFE.csv')
    """
    print(f">>> Atualizando {name} ...")

    df_old = load_existing_inventory(filename)

    last_date = df_old["Date"].max() if not df_old.empty else None
    if last_date is not None:
        print(f"Última data no CSV: {last_date.date()}")

    df_new = fetch_inventory_commoditieschart(url, last_date=last_date)

    if df_new.empty:
        print(f"Nenhum dado novo para {name}.")
        return

    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["Date"], keep="last")
    df_all = df_all.sort_values("Date")

    save_inventory(df_all, filename)


# ---------------------------------------------------------------------
# SCRIPT FINAL
# ---------------------------------------------------------------------
if __name__ == "__main__":
    for name, (url, filename) in INVENTORY_SOURCES.items():
        update_inventory(name, url, filename)
