import json
import os
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup, Tag
from dev1.sql2py.tools.feature_knowledge_crawler.crawler.crawler_options import set_options
from selenium.webdriver.chrome.service import Service
import re
from urllib.parse import urljoin

def get_table_column_names(soup_thead):
    soup_thead_names = []
    if not soup_thead:
        return soup_thead_names
    ths = soup_thead.find_all("th")
    for th in ths:
        soup_thead_names.append(th.text.strip())
    return soup_thead_names

def get_table_column_contents(soup_tbody):
    soup_tbody_contents = []
    if not soup_tbody:
        return soup_tbody_contents
    trs = soup_tbody.find_all("tr")
    for tr in trs:
        td_contents = []
        if tr.find("td").text.strip().lower() in ["alias(es)", "aliases", "alias"]:
            td_contents.append(tr.find("td").text.strip())
            code_temp = ""
            for td in tr.find_all("td"):
                if td.text.strip() in td_contents or "-" == td.text.strip():
                    continue
                codes = td.find_all("code")
                for code in codes:
                    code_temp += code.text.strip() + ";"
            td_contents.append(code_temp[:-1])
        else:
            for td in tr.find_all("td"):
                td_contents.append(td.text.strip())
        soup_tbody_contents.append(td_contents)
    return soup_tbody_contents


def function_crawler(origin_category, title, html, dic_filename):
    result = {}
    timeout = 5
    options = set_options()
    chrome_driver_path = "../chromedriver-linux64/chromedriver"
    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(html)
    WebDriverWait(driver, timeout)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    soup_section = soup.find("section", id=origin_category.lower().replace("_", "-"))
    soup_tables = soup_section.find_all("table")

    functions_dic = {}


    for table in soup_tables:
        column_names = get_table_column_names(table.find("thead"))
        if ("Function" in column_names or "Name" in column_names) and "Description" in column_names:
            table_contents = get_table_column_contents(table.find("tbody"))
            for content in table_contents:
                feature_temp = [content[column_names.index("Function")]] if "Function" in column_names else [content[column_names.index("Name")]]
                description_temp = [content[column_names.index("Description")]] if "Description" in column_names else []
                example_temp = [content[column_names.index("Example")]] if "Example" in column_names else []

                function_res = {
                    "HTML": [html],
                    "Title": feature_temp,
                    "Feature": feature_temp,
                    "Description": description_temp,
                    "Examples": example_temp,
                    "Category": [origin_category]
                }

                functions_dic[feature_temp[0]] = function_res

    for function_name, function_info in functions_dic.items():
        title = function_info["Title"][0]
        cleaned_title = re.sub(r"\[.*?\]", "", title).strip()
        id = re.sub(r"\s+", "-", cleaned_title.lower())
        soup_section = soup.find("section", id=id)
        if not soup_section:
            soup_section = soup.find("p", id=id)

        example_temp = []
        if soup_section:
            example_tag = soup_section.find("p", string=lambda x: x and "Example:" in x)
            if example_tag:
                code_div = example_tag.find_next(
                    lambda tag: (
                            tag.name == "div" and
                            tag.get("class") and
                            any(cls.startswith("highlight-") for cls in tag.get("class"))
                    )
                )
                if code_div:
                    pre_tag = code_div.find("pre")
                    if pre_tag:
                        code_text = pre_tag.get_text().strip()
                    else:
                        code_text = code_div.get_text().strip()
                    example_temp = [code_text]

        function_info["Examples"] = example_temp

    if not os.path.exists(dic_filename):
        os.makedirs(dic_filename)

    for key, value in functions_dic.items():
        file_cnt = len(os.listdir(dic_filename))
        filename = f"{file_cnt}.json"
        filepath = os.path.join(dic_filename, filename)
        if os.path.exists(filepath):
            print(f"{filename} already exists")
        with open(filepath, "w", encoding="utf-8") as w:
            json.dump(value, w, indent=4, ensure_ascii=False)
    driver.quit()


def python_crawler(origin_category, title, html, dic_filename):
    result = {}
    functions_dic = {}
    timeout = 5
    options = set_options()
    if ((origin_category == 'Plot') or (origin_category == 'Style')):
        detial = extract_description_and_examples(html, origin_category.lower())
        function_res = {
            "HTML": [html],
            "Title": origin_category,
            "Feature": origin_category,
            "Description": detial["description"],
            "Examples": detial["examples"],
            "Category": [origin_category]
        }
        functions_dic[origin_category] = function_res
    else:
        chrome_driver_path = "../chromedriver-linux64/chromedriver"
        service = Service(chrome_driver_path)
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(html)
        WebDriverWait(driver, timeout)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        soup_section = soup.find("section", id=origin_category.lower().replace("_", "-").replace("/", "-"))

        rows = soup_section.find_all("tr")

        for row in rows:
            cols = row.find_all("td")
            if len(cols) != 2:
                continue

            func_col = cols[0]
            desc_col = cols[1]

            a_tag = func_col.find("a")
            if not a_tag or not a_tag.get("href"):
                continue

            func_name = a_tag.text.strip()
            print(func_name)
            href = a_tag["href"]
            full_url = urljoin(html, href)

            description_text = desc_col.get_text(strip=True)

            section_id = f"pandas-{func_name.lower().replace('.', '-').replace('_', '-')}"
            section_id = re.sub(r"-{2,}", "-", section_id)
            section_id = section_id.strip("-")
            detail_description = extract_description_and_examples(full_url, section_id)
            detail = extract_function_detail(full_url, func_name, href)

            function_res = {
                "HTML": [full_url],
                "Title": [func_name],
                "Feature": [func_name],
                "Description": detail_description["description"],
                "Examples": detail["Examples"],
                "Parameters": detail["Parameters"],
                "Returns": detail["Returns"],
                "Category": [origin_category.replace("/", "_")]
            }

            functions_dic[func_name] = function_res
            driver.quit()

    if not os.path.exists(dic_filename):
        os.makedirs(dic_filename)

    for key, value in functions_dic.items():
        file_cnt = len(os.listdir(dic_filename))
        filename = f"{file_cnt}.json"
        filepath = os.path.join(dic_filename, filename)
        with open(filepath, "w", encoding="utf-8") as w:
            json.dump(value, w, indent=4, ensure_ascii=False)

def extract_description_and_examples(url, section_id):
    timeout = 5
    options = set_options()
    chrome_driver_path = "../chromedriver-linux64/chromedriver"
    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    WebDriverWait(driver, timeout)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    if not soup:
        print(f"soup not found in {url}")
    section = soup.find("section", {"id": section_id})
    if not section:
        print(f"Section '{section_id}' not found in {url}, not description")
        return {
            "url": url,
            "description": None,
            "examples": None
        }

    dt = section.find("dt")
    dd = dt.find_next_sibling("dd") if dt else None
    if not dd:
        print(f"Description block not found in {url}")
        return None

    description_parts = []
    examples_code = []

    reached_examples = False
    for tag in dd.children:
        if not isinstance(tag, Tag):
            continue

        if tag.name == "p" and "Examples" in tag.text:
            reached_examples = True
            continue

        if not reached_examples:
            if tag.name in ["p", "div", "ul"]:
                description_parts.append(tag.get_text(strip=True))
        else:
            code_blocks = tag.find_all("pre")
            for pre in code_blocks:
                examples_code.append(pre.get_text())
    driver.quit()
    return {
        "url": url,
        "description": "\n".join(description_parts),
        "examples": examples_code
    }

def extract_function_detail(url, feature_name, href):
    options = set_options()
    chrome_driver_path = "../chromedriver-linux64/chromedriver"
    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    WebDriverWait(driver, 5)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    if not soup:
        print(f"soup not found in {url}")

    title = href.split("#")[-1]
    cleaned_title = re.sub(r"\[.*?\]", "", title).strip()
    cleaned_title = cleaned_title.replace(".", "-").replace("_", "-")
    id = re.sub(r"\s+", "-", cleaned_title.lower())
    id = re.sub(r"-{2,}", "-", id)
    id = id.strip("-")
    section = soup.find("section", id=id)
    if not section:
        print(f"Section '{id}' not found in {url}")
        return {"Examples": [], "Parameters": [], "Returns": []}

    examples, parameters, returns = [], [], []

    examples_header = section.find("p", class_="rubric", string="Examples")
    if examples_header:
        code_block = examples_header.find_next("div", class_="highlight")
        if code_block:
            examples.append(code_block.get_text().strip())

    dt_tag = section.find("dt", class_="sig sig-object py")
    dd_tag = dt_tag.find_next_sibling("dd") if dt_tag else None

    parameters = extract_section_by_title(dd_tag, "Parameters")
    returns = extract_section_by_title(dd_tag, "Returns")
    driver.quit()
    return {
        "Examples": examples,
        "Parameters": parameters,
        "Returns": returns
    }


def extract_section_by_title(dd_tag, title_text):
    result = []
    dt_tags = dd_tag.find_all("dt")
    for dt_tag in dt_tags:
        if dt_tag.get_text(strip=True).startswith(title_text):
            dd = dt_tag.find_next_sibling("dd")
            if not dd:
                continue
            dl = dd.find("dl")
            if not dl:
                continue
            for dt, dd in zip(dl.find_all("dt"), dl.find_all("dd")):
                name = dt.get_text(" ", strip=True)
                desc = dd.get_text(" ", strip=True)
                result.append((name, desc))
            break
    return result

def op_crawler(origin_category, title, html, dic_filename):
    result = {}
    timeout = 5
    options = set_options()
    chrome_driver_path = "../chromedriver-linux64/chromedriver"
    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(html)
    WebDriverWait(driver, timeout)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    soup_div = soup.find("div", id="main_content_wrap")
    soup_tables = soup_div.find_all("table")

    functions_dic = {}
    for table in soup_tables:
        column_names = get_table_column_names(table.find("thead"))
        if "Operator" in column_names:
            table_contents = get_table_column_contents(table.find("tbody"))
            for content in table_contents:
                feature_temp = [content[column_names.index("Operator")]] if "Operator" in column_names else []
                description_temp = [content[column_names.index("Description")]] if "Description" in column_names else []
                example_temp = [content[column_names.index("Example")]] if "Example" in column_names else []

                function_res = {
                    "HTML": [html],
                    "Title": feature_temp,
                    "Feature": feature_temp,
                    "Description": description_temp,
                    "Examples": example_temp,
                    "Category": [origin_category]
                }
                functions_dic[function_res["Description"][0]] = function_res

    for key, value in functions_dic.items():
        file_cnt = len(os.listdir(dic_filename))
        filename = str(file_cnt) + ".json"
        if os.path.exists(os.path.join(dic_filename, filename)):
            print(filename+":already exists")
        with open(os.path.join(dic_filename, filename), "w", encoding="utf-8") as w:
            json.dump(value, w, indent=4)

def data_types_crawler(category_key, statement_key, statement_value, dic_filename):
    detailed = {
        "HTML": [statement_value],
        "Title": [statement_key],
        "Feature": [statement_key],
        "Description": [],
        "Examples": [],
        "Category": [statement_key]
    }
    timeout = 5
    options = set_options()
    chrome_driver_path = "../chromedriver-linux64/chromedriver"
    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(statement_value)
    WebDriverWait(driver, timeout)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    soup_div = soup.find("div", id="main_content_wrap")
    for item in soup_div:
        if len(item.text.strip()):
            detailed["Description"].append(item.text)
    soup_pres = soup.find_all("pre", class_="highlight")
    for item in soup_pres:
        soup_codes = item.find_all("code")
        for code in soup_codes:
            if len(code.text.strip()):
                detailed["Examples"].append(code.text)
    file_cnt = len(os.listdir(dic_filename))
    with open(os.path.join(dic_filename, str(file_cnt) + ".json"), "w", encoding="utf-8") as w:
        json.dump(detailed, w, indent=4)

def crawler_results(feature_type, htmls_filename, dic_filename):
    if len(os.listdir(dic_filename)):
        print(dic_filename + ":Crawler finished")
        return
    with open(htmls_filename, "r", encoding="utf-8") as rf:
        html_contents = json.load(rf)
        for category_key, value in html_contents.items():
            for statement_key, statement_value in value.items():
                print(statement_key+":"+str(statement_value))
                if feature_type == "dataframe":
                    python_crawler(statement_key, statement_key, statement_value, dic_filename)
                print('----------------------')


