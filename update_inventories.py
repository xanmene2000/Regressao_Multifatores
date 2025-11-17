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
    "Copper_SHFE": "https://commoditieschart.net/metals/copper/shfe-copper-stocks",
    "Copper_COMEX": "https://commoditieschart.net/metals/copper/comex-copper-stocks",
    "Aluminum_SHFE": "https://commoditieschart.net/metals/aluminium/shfe-aluminium-stocks",
    "Aluminum_COMEX": "https://commoditieschart.net/metals/aluminium/comex-aluminium-stocks",
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
def fetch_inventory_commoditieschart(url: str) -> pd.DataFrame:
    """
    - Abre página
    - Clica no botão 'Table'
    - Extrai tabela
    - Percorre paginação ('Next') se existir
    - Retorna DataFrame: Date, Value
    """

    driver = start_driver()
    driver.get(url)

    wait = WebDriverWait(driver, 5)

    # 1. Clicar no botão TABLE para mostrar a tabela
    table_btn = wait.until(
        EC.element_to_be_clickable((By.XPATH, "//div[contains(@class,'cursor-pointer') and normalize-space(text())='Table']"))
    )
    table_btn.click()
    print('Encontrou o botao Table')
    time.sleep(2)  # deixa carregar tabela

    rows = []

    def extract_current_page():
        
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

            date_str, value_str = cells[0], cells[1]
            new_rows.append((date_str, value_str))

        return new_rows

    # 2. Extrair primeira página
    rows.extend(extract_current_page())

    # 3. Paginação via botão NEXT
    while True:
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

    # 4. Converte rows → DataFrame
    df = pd.DataFrame(rows, columns=["Date", "Value"])

    # Converte tipos
    # transforma "Aug 20,2024" → "Aug 20, 2024"
    df["Date"] = df["Date"].str.replace(r",(\d{4})$", r", \1", regex=True)

    # agora podemos usar o formato explícito
    df["Date"] = pd.to_datetime(df["Date"], format="%b %d, %Y", errors="coerce")
    df["Value"] = (
         df["Value"]
         .str.replace(",", "")
         .astype(float)
    )

    df = df.dropna(subset=["Date"]).sort_values("Date")

    return df


# ---------------------------------------------------------------------
# SALVAR CSV
# ---------------------------------------------------------------------
def save_inventory(df: pd.DataFrame, filename: str):
    datasets = Path(__file__).resolve().parent / "datasets"
    datasets.mkdir(exist_ok=True)

    out_path = datasets / filename

    df = df.sort_values("Date")
    df.to_csv(out_path, index=False)

    print(f"[OK] Salvo: {filename} ({len(df)} linhas)")


# ---------------------------------------------------------------------
# SCRIPT FINAL
# ---------------------------------------------------------------------
if __name__ == "__main__":
    for key, url in INVENTORY_SOURCES.items():
        print(f"\n>>> Extraindo {key} ...")
        df = fetch_inventory_commoditieschart(url)
        save_inventory(df, f"Stock_{key}.csv")
