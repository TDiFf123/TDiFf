import os
import json
from dev1.sql2py.tools.feature_knowledge_crawler.duckdb.htmls_crawler import htmls_crawler
from dev1.sql2py.tools.feature_knowledge_crawler.duckdb.info_crawler import crawler_results
from dev1.sql2py.tools.feature_knowledge_crawler.crawler.crawler_options import category_classifier
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

function_list = ["dataframe"]
def duckdb_crawler():
    dic_path = os.path.join(current_dir, "..", "..", "FeatureKnowledgeBase", "duckdb")
    feature_types = ["dataframe"]
    sub_dic = ["results", "results_category"]
    htmls_list = {
        "sql": "https://duckdb.org/docs/sql",
    }
    # htmls Crawler
    for feature in feature_types:
        # make dictionaries
        feature_dic = os.path.join(dic_path, feature)
        if not os.path.exists(feature_dic):
            os.makedirs(feature_dic)
        for sub in sub_dic:
            sub_dic_path = os.path.join(feature_dic, sub)
            if not os.path.exists(sub_dic_path):
                os.makedirs(sub_dic_path)
        # crawl the htmls list
        html_path = os.path.join(feature_dic, "HTMLs.json")
        if os.path.exists(html_path):
            print("File " + html_path + " exists!")
            continue
        if (feature not in function_list):
            htmls = htmls_crawler(htmls_list[feature])
        else:
            htmls = {
                "No Category": {
                    feature.title(): htmls_list[feature]
                }
            }
        with open(html_path, 'w', encoding='utf-8') as f:
            json.dump(htmls, f, indent=4, ensure_ascii=False)


    # information Crawler and classification
    for feature in feature_types:
        htmls_filename = os.path.join(dic_path, feature, "HTMLs.json")
        results_dic = os.path.join(dic_path, feature, "results")
        results_category_dic = os.path.join(dic_path, feature, "results_category")
        crawler_results(feature, htmls_filename, results_dic)
        category_classifier(results_dic, results_category_dic)

duckdb_crawler()

