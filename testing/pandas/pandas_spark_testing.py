import json
import pandas as pd
import numpy as np
import os
import re
import pyspark
from pyspark.sql import SparkSession
import pyspark.pandas as ps
import datetime
import argparse

def init_pandas_tables(csv_dir):
    pandas_tables = {}
    for fname in os.listdir(csv_dir):
        if fname.lower().endswith(".csv"):
            table_name = os.path.splitext(fname)[0]
            path = os.path.join(csv_dir, fname)
            pandas_tables[table_name] = pd.read_csv(path)
    return pandas_tables

def init_ps_tables(csv_dir):
    ps_tables = {}
    for fname in os.listdir(csv_dir):
        if fname.lower().endswith(".csv"):
            table_name = os.path.splitext(fname)[0]
            path = os.path.join(csv_dir, fname)
            ps_tables[table_name] = ps.read_csv(path)
    return ps_tables

def exec_pandas_statement(transfer_code, tables):
    try:
        local_vars = {name: df for name, df in tables.items()}
        local_vars["pd"] = pd
        local_vars["np"] = np
        exec(transfer_code, local_vars, local_vars)
        transfer_result = local_vars["result"]
        return transfer_result, None
    except Exception as e:
        print("python error message (pandas): ")
        print(e)
        return None, e

# def replace_pd_with_ps(code: str):
#     pattern = r'(?<![A-Za-z0-9_])pd(?=\s*[\.\(\[])'
#     return re.sub(pattern, "ps", code)
def replace_pd_with_ps(code: str):
    pattern = r'(?<![A-Za-z0-9_])pd(?!\s*\.Timestamp)(?=\s*[\.\(\[])'
    return re.sub(pattern, "ps", code)

def exec_ps_statement(transfer_code, ps_tables):
    try:
        transfer_code_ps = replace_pd_with_ps(transfer_code)
        local_vars = {name: df for name, df in ps_tables.items()}
        local_vars["pd"] = pd
        local_vars["ps"] = ps
        local_vars["np"] = np
        exec(transfer_code_ps, local_vars, local_vars)
        transfer_result = local_vars["result"].to_pandas()
        return transfer_result, None
    except Exception as e:
        print("python error message: ")
        print(e)
        return None, e

def default_serializer(obj):
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, Exception):
        return str(obj)
    return repr(obj)

def process_single_json(
    json_path,
    pandas_tables,
    ps_tables,
    output_dir,
):
    fname = os.path.basename(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transfer_code = data.get("FinalPythonCode")

    pandas_out, pandas_err = exec_pandas_statement(transfer_code, pandas_tables)
    ps_out, ps_err = exec_ps_statement(transfer_code, ps_tables)

    output = {
        "PythonCode": transfer_code,
        "Numpy Version": np.__version__,
        "Pandas": {
            "Version": pd.__version__,
            "Result": pandas_out,
            "Error": str(pandas_err),
        },
        "Pyspark_Pandas": {
            "Version": pyspark.__version__,
            "Result": ps_out,
            "Error": str(ps_err),
        }
    }


    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.splitext(fname)[0] + ".json"
    out_path = os.path.join(output_dir, out_file)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False, default=default_serializer)

    print(f"Saved → {out_path}")
    return output

def run_diff_test(json_dir, csv_dir, output_dir="output_results"):
    pandas_tables = init_pandas_tables(csv_dir)
    ps_tables = init_ps_tables(csv_dir)

    for fname in os.listdir(json_dir):
        if fname.endswith(".json"):
            json_path = os.path.join(json_dir, fname)
            print(f"Processing {fname} ...")
            process_single_json(
                json_path,
                pandas_tables,
                ps_tables,
                output_dir
            )

    print("\n=== All files processed ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", default="")
    parser.add_argument("--csv_dir", default="")
    parser.add_argument("--out", default="")
    spark = (
        SparkSession.builder
        .config("spark.sql.ansi.enabled", "false")
        .getOrCreate()
    )
    ps.set_option("compute.ops_on_diff_frames", True)
    args = parser.parse_args()
    run_diff_test(
        args.json_dir,
        args.csv_dir,
        args.out
    )
