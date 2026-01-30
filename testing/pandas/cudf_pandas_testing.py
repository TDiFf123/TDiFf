import json
import cudf.pandas
cudf.pandas.install()
import pandas as pd
import cudf
import os
import datetime
import argparse
import numpy as np

def init_pandas_tables(csv_dir):
    pandas_tables = {}
    for fname in os.listdir(csv_dir):
        if fname.lower().endswith(".csv"):
            table_name = os.path.splitext(fname)[0]
            path = os.path.join(csv_dir, fname)
            pandas_tables[table_name] = pd.read_csv(path)
    return pandas_tables


def exec_pandas_statement(transfer_code, tables):
    try:
        local_vars = {name: df for name, df in tables.items()}
        local_vars["pd"] = pd
        local_vars["np"] = np
        exec(transfer_code, local_vars, local_vars)
        transfer_result = local_vars["result"]
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


def merge_with_existing(out_path, new_block):
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            try:
                old_data = json.load(f)
            except Exception:
                old_data = {}

        old_data.update(new_block)

        return old_data

    else:
        return new_block


def process_single_json(json_path, pandas_tables, output_dir):
    fname = os.path.basename(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transfer_code = data.get("FinalPythonCode")

    pandas_out, pandas_err = exec_pandas_statement(transfer_code, pandas_tables)

    new_block = {
        "Cudf_Pandas": {
            "Pandas Version": pd.__version__,
            "Cudf Version": cudf.__version__,
            "Result": pandas_out,
            "ERROR": str(pandas_err),
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.splitext(fname)[0] + ".json"
    out_path = os.path.join(output_dir, out_file)

    final_output = merge_with_existing(out_path, new_block)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False, default=default_serializer)

    print(f"Updated → {out_path}")
    return final_output


def run_diff_test(json_dir, csv_dir, output_dir="output_results"):
    pandas_tables = init_pandas_tables(csv_dir)

    for fname in os.listdir(json_dir):
        if fname.endswith(".json"):
            json_path = os.path.join(json_dir, fname)
            print(f"Processing {fname} ...")
            process_single_json(json_path, pandas_tables, output_dir)

    print("\n=== All files processed ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", default="")
    parser.add_argument("--csv_dir", default="e")
    parser.add_argument("--out", default="")

    args = parser.parse_args()
    run_diff_test(args.json_dir, args.csv_dir, args.out)
