import os
import re
import json
from typing import List, Optional, Any
import argparse
from pyspark.sql import SparkSession, Row
import pandas as pd
from duckdb.experimental.spark.sql import SparkSession as DuckdbSession
import duckdb
from pyspark.sql import functions as F
from duckdb.experimental.spark.sql import functions as duckdb_F


def _row_to_py(obj):
    if isinstance(obj, Row):
        return [_row_to_py(v) for v in obj.asDict(recursive=False).values()]
    if isinstance(obj, (list, tuple)):
        return [_row_to_py(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _row_to_py(v) for k, v in obj.items()}
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def rows_to_list(rows: Optional[List[Row]]) -> Optional[List[Any]]:
    if rows is None:
        return None
    out = []
    for r in rows:
        if isinstance(r, Row):
            out.append([_row_to_py(v) for v in r])
        else:
            out.append(_row_to_py(r))
    return out

def exec_api_spark_statement(spark, transfer_code):
    try:
        global_env = {"spark": spark, "F": F}
        exec(transfer_code, global_env)
        transfer_result = global_env["result"].collect()
        return transfer_result, None
    except Exception as e:
        print("python error message:")
        print(e)
        # traceback.print_exc()
        return None, e


def exec_api_duckdb_statement(duckdb_spark, transfer_code):
    try:
        global_env = {"spark": duckdb_spark, "F": duckdb_F}
        exec(transfer_code, global_env)
        transfer_result = global_env["result"].collect()
        return transfer_result, None
    except Exception as e:
        print("python error message:")
        print(e)
        # traceback.print_exc()
        return None, e

def process_single_json(json_path, output_dir, spark, duckdb_spark):
    fname = os.path.basename(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    python_code = data.get("FinalPythonCode")

    spark_result, spark_error = exec_api_spark_statement(spark, python_code)
    duckdb_result, duckdb_error = exec_api_duckdb_statement(duckdb_spark, python_code)

    res = {
        "Python Code": python_code,
        "Spark": {
            "version": spark.version,
            "Result": rows_to_list(spark_result),
            "Error": str(spark_error) if spark_error else None
        },
        "DuckDB_Spark": {
            "version": duckdb.__version__,
            "Result": rows_to_list(duckdb_result),
            "Error": str(duckdb_error) if duckdb_error else None
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.splitext(fname)[0] + ".json"
    out_path = os.path.join(output_dir, out_file)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)

    print(f"Saved → {out_path}")

def run_diff_test(json_dir, spark, duckdb_spark, output_dir="output_results"):
    for fname in os.listdir(json_dir):
        if fname.endswith(".json"):
            json_path = os.path.join(json_dir, fname)
            print(f"Processing {fname} ...")
            process_single_json(json_path, output_dir, spark, duckdb_spark)

    print("\n=== All files processed ===")


def init_spark_tables(csv_dir, spark):
    for fname in os.listdir(csv_dir):
        if fname.lower().endswith(".csv"):
            table_name = os.path.splitext(fname)[0]
            path = os.path.join(csv_dir, fname)
            print(f"Spark: Loading {path} as table {table_name} ...")
            df = spark.read.csv(path, header=True, inferSchema=True)
            df.createOrReplaceTempView(table_name)


def init_duckdb_tables(csv_dir, duckdb_spark):
    for fname in os.listdir(csv_dir):
        if fname.lower().endswith(".csv"):
            table_name = os.path.splitext(fname)[0]
            path = os.path.join(csv_dir, fname)
            print(f"DuckDB Spark: Loading {path} as table {table_name} ...")
            pandas_df = pd.read_csv(path)
            df = duckdb_spark.createDataFrame(pandas_df)
            df.createOrReplaceTempView(table_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", default="../../../llm_transfer/pyspark/transfer_result/deepseek-reasoner/WHERE2/LLM+Err_Iter+RAG+DFG/result")
    parser.add_argument("--output_dir", default="./result/deepseek-reasoner/WHERE2/LLM+Err_Iter+RAG+DFG/result")
    parser.add_argument("--csv_file", default="../../../data/sqlancer/WHERE2/table")
    args = parser.parse_args()

    spark = SparkSession.builder \
        .appName("Spark API Test") \
        .master("local[*]") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "8g") \
        .getOrCreate()

    duckdb_spark = DuckdbSession.builder \
        .appName("DuckDB Spark API Test") \
        .master("local[*]") \
        .getOrCreate()

    init_spark_tables(args.csv_file, spark)
    init_duckdb_tables(args.csv_file, duckdb_spark)

    run_diff_test(args.json_file, spark, duckdb_spark, args.output_dir)
