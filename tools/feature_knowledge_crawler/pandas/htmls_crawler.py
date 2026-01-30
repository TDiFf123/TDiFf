from dev1.sql2py.tools.feature_knowledge_crawler.crawler.crawler_options import set_options
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from selenium.webdriver.chrome.service import Service
def htmls_crawler(html):
    timeout = 5
    chrome_driver_path = "../chromedriver-linux64/chromedriver"
    service = Service(chrome_driver_path)
    options = set_options()
    driver = webdriver.Chrome(service=service, options=options)
    # driver.get("https://example.com")
    # print(driver.title)
    htmls_table = {}  #
    # skip_htmls = [
    #     "https://duckdb.org/docs/sql/functions/dateformat",
    #     "https://duckdb.org/docs/sql/functions/nested"
    # ]
    driver.get(html)
    WebDriverWait(driver, timeout)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    sql_functions_li = soup.find("li", class_="toctree-l1 current active has-children")
    if (sql_functions_li == None):
        sql_functions_li = soup.find("li", class_="toctree-l2 current active has-children")
        sub_items = sql_functions_li.find_all("li", class_="toctree-l3")
    else:
        sub_items = sql_functions_li.find_all("li", class_="toctree-l2 has-children")

    for soup_li in sub_items:
        a_tag = soup_li.find("a")
        if a_tag:
            name = a_tag.text.strip()
            href = urljoin(html, a_tag.get("href"))
            htmls_table[name] = href
        # if href in skip_htmls:
        #     continue
    return {"No Category": htmls_table}
