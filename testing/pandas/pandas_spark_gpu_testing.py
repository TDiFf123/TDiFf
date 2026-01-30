import os
rapids = "rapids-4-spark_2.13-25.10.0.jar"
cudf = "cudf-25.10.0-cuda12.jar"
os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-17-openjdk-amd64"
os.environ['SPARK_HOME'] = "./spark-4.0.1-bin-hadoop3"
os.environ['PYSPARK_SUBMIT_ARGS'] = f"--jars ./{rapids},./{cudf} --master local[*] pyspark-shell"

import json
import pandas as pd
import numpy as np
import re
import findspark
findspark.init()
import pyspark
from pyspark.sql import SparkSession
import pyspark.pandas as ps
import datetime
import argparse

def init_ps_tables(csv_dir):
    ps_tables = {}
    for fname in os.listdir(csv_dir):
        if fname.lower().endswith(".csv"):
            table_name = os.path.splitext(fname)[0]
            path = os.path.join(csv_dir, fname)
            ps_tables[table_name] = ps.read_csv(path)
    return ps_tables

def replace_pd_with_ps(code: str):
    pattern = r'(?<![A-Za-z0-9_])pd(?!\s*\.Timestamp)(?=\s*[\.\(\[])'
    return re.sub(pattern, "ps", code)

def exec_ps_cpu_statement(transfer_code, ps_tables):
    try:
        spark.conf.set('spark.rapids.sql.enabled', 'false')
        transfer_code_ps = replace_pd_with_ps(transfer_code)
        local_vars = {name: df for name, df in ps_tables.items()}
        local_vars["ps"] = ps
        local_vars["np"] = np
        local_vars["pd"] = pd
        exec(transfer_code_ps, local_vars, local_vars)
        transfer_result = local_vars["result"].to_pandas()
        return transfer_result, None
    except Exception as e:
        print("python error message: ")
        print(e)
        return None, e

def exec_ps_gpu_statement(transfer_code, ps_tables):
    try:
        spark.conf.set('spark.rapids.sql.enabled', 'true')
        transfer_code_ps = replace_pd_with_ps(transfer_code)
        local_vars = {name: df for name, df in ps_tables.items()}
        local_vars["ps"] = ps
        local_vars["np"] = np
        local_vars["pd"] = pd
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
    ps_tables,
    output_dir,
):
    fname = os.path.basename(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transfer_code = data.get("FinalPythonCode")

    ps_cpu_out, ps_cpu_err = exec_ps_cpu_statement(transfer_code, ps_tables)
    ps_gpu_out, ps_gpu_err = exec_ps_gpu_statement(transfer_code, ps_tables)

    output = {
        "PythonCode": transfer_code,
        "Numpy Version": np.__version__,
        "Spark Version": pyspark.__version__,
        "Cudf Version": cudf,
        "Rapids Version": rapids,
        "Pyspark_Pandas_CPU": {
            "Result": ps_cpu_out,
            "Error": str(ps_cpu_err),
        },
        "Pyspark_Pandas_GPU": {
            "Result": ps_gpu_out,
            "Error": str(ps_gpu_err),
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
    ps_tables = init_ps_tables(csv_dir)

    for fname in os.listdir(json_dir):
        if fname.endswith(".json"):
            json_path = os.path.join(json_dir, fname)
            print(f"Processing {fname} ...")
            process_single_json(
                json_path,
                ps_tables,
                output_dir
            )

    print("\n=== All files processed ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", default="")
    parser.add_argument("--csv_dir", default="")
    parser.add_argument("--out", default="")
    spark = SparkSession.builder.appName('SparkRAPIDS').config("spark.sql.ansi.enabled", "false")\
        .config('spark.plugins', 'com.nvidia.spark.SQLPlugin')\
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "8g").getOrCreate()

    spark.sparkContext.addPyFile(f'./{rapids}')
    spark.sparkContext.addPyFile(f'./{cudf}')
    # spark.conf.set('spark.rapids.sql.enabled', 'true')
    spark.conf.set('spark.rapids.sql.incompatibleOps.enabled', 'true')
    spark.conf.set('spark.rapids.sql.format.csv.read.enabled', 'true')
    spark.conf.set('spark.rapids.sql.format.csv.enabled', 'true')
    spark.conf.set("spark.executor.resource.gpu.amount", "1")
    spark.conf.set("spark.task.resource.gpu.amount", "1")
    spark.conf.set("spark.rapids.sql.concurrentGpuTasks", "1")
    spark.conf.set("spark.rapids.sql.exec.CollectLimitExec", "true")
    # spark.conf.set("spark.rapids.sql.explain", "ALL")
    spark.conf.set("spark.rapids.sql.stableSort.enabled", "true")
    ps.set_option("compute.ops_on_diff_frames", True)
    args = parser.parse_args()
    run_diff_test(
        args.json_dir,
        args.csv_dir,
        args.out
    )
