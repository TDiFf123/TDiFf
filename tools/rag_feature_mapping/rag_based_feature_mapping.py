import json
import os
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import os
import json

from JSONLoader import JSONLoader

llm = ChatOpenAI(
    model="gpt-5-mini-2025-08-07",
    base_url="https://api.openai.com/v1",
    api_key="sk-xxxxxxxx",
    temperature=0,
)

def feature_knowledge_merge(input_dir: str, output_dir: str, feature_type: str):
    merge_data_filename = os.path.join(output_dir, feature_type + ".jsonl")

    if not os.path.exists(merge_data_filename):
        feature_filepath = os.path.join(input_dir, feature_type, "results")

        if not os.path.exists(feature_filepath):
            raise FileNotFoundError(f"The input directory does not exist: {feature_filepath}")

        filenames = os.listdir(feature_filepath)

        os.makedirs(output_dir, exist_ok=True)

        for filename in filenames:
            file_path = os.path.join(feature_filepath, filename)
            if not os.path.isfile(file_path):
                continue

            with open(file_path, "r", encoding="utf-8") as r:
                value = json.load(r)

            with open(merge_data_filename, "a", encoding="utf-8") as a:
                json.dump(value, a, ensure_ascii=False)
                a.write("\n")
    else:
        print(merge_data_filename + ": already exists！")


def feature_type_merge(input_dir: str, output_dir: str, feature_types: list[str]) -> str:
    names = "merge"
    for feature_type in feature_types:
        names += "_" + feature_type
    merge_feature_filename = os.path.join(output_dir, names + ".jsonl")

    if not os.path.exists(merge_feature_filename):
        os.makedirs(output_dir, exist_ok=True)
        index_all = 0

        for feature_type in feature_types:
            embedding_data_filename = os.path.join(input_dir, feature_type + ".jsonl")

            if not os.path.exists(embedding_data_filename):
                raise FileNotFoundError(f"{embedding_data_filename} does not exist")

            with open(embedding_data_filename, "r", encoding="utf-8") as r:
                lines = r.readlines()

            with open(merge_feature_filename, "a", encoding="utf-8") as a:
                for line in lines:
                    value = json.loads(line)
                    value["index"] = index_all
                    index_all += 1
                    json.dump(value, a, ensure_ascii=False)
                    a.write("\n")

    else:
        print(f"{merge_feature_filename} already exists！")

    return merge_feature_filename

def safe_join(value):
    if isinstance(value, str):
        return value
    elif isinstance(value, list):
        return " ".join([safe_join(v) for v in value])  # flatten
    elif isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    else:
        return str(value)

def load_feature_knowledge_embedding(embedding_data_filename, merge_feature_filename, content_keys):
    if not os.path.exists(embedding_data_filename):
        with open(merge_feature_filename, "r", encoding="utf-8") as r:
            lines = r.readlines()
        for line in lines:
            value = json.loads(line)
            vector_txt = str(value["index"]) + ":"
            for content_key in content_keys:
                if content_key in value:
                    vector_txt += content_key + ": " + safe_join(value[content_key]) + "\n"
            value["vector_txt"] = vector_txt
            with open(embedding_data_filename, "a", encoding="utf-8") as a:
                json.dump(value, a)
                a.write("\n")
    else:
        print(f"{embedding_data_filename}: already exists!")

    loader = JSONLoader(file_path=embedding_data_filename, content_key="vector_txt", json_lines=True)
    return loader.load()


def rag_feature_mapping_llm(search_k, db, dir_filename,
                               python_embedding_data_filename, python_merge_feature_filename,
                               sql_embedding_data_filename, sql_merge_feature_filename,
                               content_keys):
    data_python = load_feature_knowledge_embedding(
        python_embedding_data_filename, python_merge_feature_filename, content_keys
    )

    embeddings = SentenceTransformerEmbeddings(model_name="")
    vectorstore = Chroma.from_documents(data_python, embeddings)

    data_sql = load_feature_knowledge_embedding(
        sql_embedding_data_filename, sql_merge_feature_filename, content_keys
    )

    with open(sql_merge_feature_filename, "r", encoding="utf-8") as read_lines:
        query_data = read_lines.readlines()

    feature_to_index = {}
    with open(python_merge_feature_filename, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            for feat in obj.get("Feature", []):
                feature_to_index[feat] = obj.get("index", -1)

    valid_features = set(feature_to_index.keys())

    processed_lines_cnt = 0
    if os.path.exists(dir_filename):
        with open(dir_filename, "r", encoding="utf-8") as direct_file_r:
            processed_lines_cnt = sum(1 for line in direct_file_r)

    response_schemas = [
        ResponseSchema(type="string", name="Feature", description='The mapping feature name'),
        ResponseSchema(type="string", name="Explanation", description='Explain the mapping reason.')
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    query_merged = """You are doing feature mapping between SQL-style features and DataFrame-style features.  
    Use the following pieces of retrieved context (knowledge base entries) to answer the question.  

    Require:  
    - The "feature" you return must exactly match the "feature" field of the feature provided in the knowledge base context. 
    - Do not create new features, abbreviations, or prefixed versions.
    - If no corresponding feature is found in the knowledge base, return "None".
    Question: {question}
    Context: {context}
    Answer the mapping feature name and reason in json format: {{"Feature":"", "Explanation":"...."}}
    """

    for query in query_data:
        query_json = json.loads(query)
        print(query_json["index"])
        if query_json["index"] < processed_lines_cnt:
            continue

        feature_name = "".join(query_json["Feature"])

        prompt = ChatPromptTemplate.from_template(query_merged)

        ks = [search_k, search_k * 5, search_k * 10]
        resp_json = None

        for k in ks:
            retriever = vectorstore.as_retriever(search_k=k)
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            with get_openai_callback() as cb:
                resp = chain.invoke(
                    "About the feature " + feature_name + " in " + db + "SQL style" +
                    ", what is the similar feature in " + db + "DataFrame style?\n" +
                    "Here is the relevant information about " + feature_name +
                    data_sql[query_json["index"]].page_content
                )

            try:
                resp_json = output_parser.parse(resp)
            except Exception:
                resp_json = {"Feature": "None", "Explanation": "Parse error"}

            if resp_json.get("Feature") and resp_json["Feature"].lower() != "none":
                break

        pred_feat = resp_json.get("Feature", "None")
        if not pred_feat or pred_feat not in valid_features:
            resp_json["Feature"] = "None"
            resp_json["index"] = -1
        else:
            resp_json["index"] = feature_to_index.get(pred_feat, -1)
            resp_json["Feature"] = [pred_feat]

        mapping_result = {
            "sql": {
                "index": query_json["index"],
                "Feature": query_json["Feature"]
            },
            "python": resp_json
        }

        print(mapping_result)

        with open(dir_filename, "a", encoding="utf-8") as a:
            json.dump(mapping_result, a, ensure_ascii=False)
            a.write("\n")


rag_feature_mapping_llm(50, "spark", "spark/clause_feature_map.jsonl",
                           "../../../../feature_knowledge_base/spark/python/RAG_Embedding_Data/pandas_api.jsonl",
                           "../../../feature_knowledge_base/spark/python/RAG_Embedding_Data/merge_pandas_api.jsonl",
                           "../../../feature_knowledge_base/spark/sql/RAG_Embedding_Data/clause.jsonl",
                           "../../../feature_knowledge_base/spark/sql/RAG_Embedding_Data/merge_clause.jsonl",
                           ["Title", "Feature", "Description", "Examples", "Parameters", "Returns", "index"])
