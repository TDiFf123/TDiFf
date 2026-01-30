import os

rapids = "rapids-4-spark_2.13-25.10.0.jar"
cudf = "cudf-25.10.0-cuda12.jar"
os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-17-openjdk-amd64"
os.environ['SPARK_HOME'] = "/home/library/spark-4.0.1-bin-hadoop3"
os.environ['PYSPARK_SUBMIT_ARGS'] = f"--jars /home/library/{rapids},/home/library/{cudf} --master local[*] pyspark-shell"

import re
import json
from typing import List, Optional, Any
import argparse
import os
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Row


def init_spark_tables(csv_dir, spark):
    for fname in os.listdir(csv_dir):
        if fname.lower().endswith(".csv"):
            table_name = os.path.splitext(fname)[0]
            path = os.path.join(csv_dir, fname)
            print(f"Loading {path} as table {table_name} ...")

            df = spark.read.csv(path, header=True, inferSchema=True)
            df.createOrReplaceTempView(table_name)

def exec_api_cpu_statement(spark, transfer_code):
    try:
        spark.conf.set('spark.rapids.sql.enabled', 'false')
        global_env = {"spark": spark, "F": F}
        exec(transfer_code, global_env)
        transfer_result = global_env["result"].collect()
        return transfer_result, None
    except Exception as e:
        print("python error message:")
        print(e)
        # traceback.print_exc()
        return None, e

def exec_api_gpu_statement(spark, transfer_code):
    try:
        spark.conf.set('spark.rapids.sql.enabled', 'true')
        global_env = {"spark": spark, "F": F}
        exec(transfer_code, global_env)
        transfer_result = global_env["result"].collect()
        return transfer_result, None
    except Exception as e:
        print("python error message:")
        print(e)
        # traceback.print_exc()
        return None, e

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

def rows_to_list(rows: Optional[List[Row]]) -> Optional[List[List[Any]]]:
    if rows is None:
        return None
    out = []
    for r in rows:
        if isinstance(r, Row):
            out.append([_row_to_py(v) for v in r])
        else:
            out.append(_row_to_py(r))
    return out

def process_single_json(json_path, output_dir, spark):
    fname = os.path.basename(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    python_code = data.get("FinalPythonCode")

    cpu_result, cpu_error = exec_api_cpu_statement(spark, python_code)
    gpu_result, gpu_error = exec_api_gpu_statement(spark, python_code)

    res = {
        "Spark Version": spark.version,
        "Cudf Version": cudf,
        "Rapids Version": rapids,
        "Python Code": python_code,
        "CPU": {"Result": rows_to_list(cpu_result), "Error": str(cpu_error)},
        "GPU": {"Result": rows_to_list(gpu_result), "Error": str(gpu_error)}
    }

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.splitext(fname)[0] + ".json"
    out_path = os.path.join(output_dir, out_file)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)

    print(f"Saved → {out_path}")

def run_diff_test(json_dir, spark, csv_dir, output_dir="output_results"):
    init_spark_tables(csv_dir, spark)

    for fname in os.listdir(json_dir):
        if fname.endswith(".json"):
            json_path = os.path.join(json_dir, fname)
            print(f"Processing {fname} ...")
            process_single_json(json_path, output_dir, spark)

    print("\n=== All files processed ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", default="")
    parser.add_argument("--csv_dir", default="")
    parser.add_argument("--out", default="")

    args = parser.parse_args()
    spark = SparkSession.builder.appName('SparkRAPIDS').config('spark.plugins', 'com.nvidia.spark.SQLPlugin').config(
        "spark.executor.memory", "8g") \
        .config("spark.driver.memory", "8g").getOrCreate()

    spark.sparkContext.addPyFile(f'/home/library/{rapids}')
    spark.sparkContext.addPyFile(f'/home/library/{cudf}')
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

    run_diff_test(args.json_dir, spark, args.csv_dir, args.out)
