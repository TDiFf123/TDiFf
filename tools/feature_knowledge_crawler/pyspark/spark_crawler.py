import os
import json
from dev1.sql2py.tools.feature_knowledge_crawler.spark.htmls_crawler import htmls_crawler
from dev1.sql2py.tools.feature_knowledge_crawler.spark.info_crawler import crawler_results
from dev1.sql2py.tools.feature_knowledge_crawler.crawler.crawler_options import category_classifier
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

function_list = ["dataframe"]
def spark_crawler():
    dic_path = os.path.join(current_dir, "..", "..", "FeatureKnowledgeBase", "pyspark1")
    feature_types = ["spark_sql"] #spark_sql pandas_api datatype operator function clause datatype
    sub_dic = ["results", "results_category"]
    htmls_list = {
        "spark_sql": "https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/index.html#",
        "pandas_api": "https://spark.apache.org/docs/latest/api/python/reference/pyspark.pandas/index.html#",
        "datatype": "https://spark.apache.org/docs/latest/sql-ref-datatypes.html",
        "operator": "https://spark.apache.org/docs/latest/sql-ref-operators.html",
        "function": "https://spark.apache.org/docs/latest/sql-ref-functions.html",
        "clause": "https://spark.apache.org/docs/latest/sql-ref-syntax.html",
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
            print("File " + html_path + " exists！")
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

spark_crawler()