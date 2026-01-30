from selenium.webdriver.chrome.options import Options
import re
import glob
import os
import json
def set_options():
    option = Options()
    option.page_load_strategy = 'eager'
    option.add_argument("--window-size=1920,1080")
    option.add_argument("--disable-extensions")
    option.add_argument('--no-sandbox')
    option.add_argument('--ignore-certificate-errors')
    option.add_argument('--allow-running-insecure-content')


    option.add_argument('--headless')
    option.add_argument("--disable--gpu")
    option.add_argument("--disable-software-rasterizer")
    option.add_argument("blink-settings=imagesEnabled=false")
    option.add_argument('--disable-plugins')
    option.add_argument("--disable-extensions")
    return option

def sanitize_title(title):
    title = re.sub(r'<', ' less ', title)
    title = re.sub(r'>', ' greater ', title)
    title = re.sub(r':', ' colon ', title)
    title = re.sub(r'\*', ' multiply ', title)
    title = re.sub(r'\/', ' divide ', title)
    title = re.sub(r'\"', "'", title)
    title = re.sub(r'\n', " ", title)
    title = re.sub(r'[\\|?]', '_', title)
    return title

def category_classifier(results_dicname, results_category_dicname):
    json_files = glob.glob(os.path.join(results_dicname, '*.json'))
    if len(os.listdir(results_category_dicname)):
        print("Category OK.")
        return
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as r:
            data = json.load(r)

        for category in data["Category"]:
            with open(os.path.join(results_category_dicname, category + ".jsonl"), "a",encoding="utf-8") as w:
                json.dump(data, w)
                w.write('\n')