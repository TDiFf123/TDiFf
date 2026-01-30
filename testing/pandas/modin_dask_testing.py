import json
import os
os.environ["MODIN_ENGINE"] = "dask"
import modin.pandas as md
import modin
import datetime
import argparse
import re
import pandas as pd
import numpy as np


def init_modin_tables(csv_dir):
    modin_tables = {}
    for fname in os.listdir(csv_dir):
        if fname.lower().endswith(".csv"):
            table_name = os.path.splitext(fname)[0]
            path = os.path.join(csv_dir, fname)
            modin_tables[table_name] = md.read_csv(path)
    return modin_tables

def replace_pd_with_md(code: str):
    pattern = r'(?<![A-Za-z0-9_])pd(?=\s*[\.\(\[])'
    return re.sub(pattern, "md", code)

def exec_modin_statement(transfer_code, modin_tables):
    try:
        transfer_code_md = replace_pd_with_md(transfer_code)
        local_vars = {name: df for name, df in modin_tables.items()}
        local_vars["md"] = md
        local_vars["np"] = np
        exec(transfer_code_md, {}, local_vars)
        transfer_result = local_vars["result"]._to_pandas()
        return transfer_result, None
    except Exception as e:
        print("python error message (modin): ")
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


def process_single_json(json_path, modin_tables, output_dir):
    fname = os.path.basename(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transfer_code = data.get("FinalPythonCode")

    modin_out, modin_err = exec_modin_statement(transfer_code, modin_tables)

    new_block = {
        "Modin_Dask": {
            "Version": modin.__version__,
            "Backend": "dask",
            "Result": modin_out,
            "Error": str(modin_err),
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
    modin_tables = init_modin_tables(csv_dir)

    for fname in os.listdir(json_dir):
        if fname.endswith(".json"):
            json_path = os.path.join(json_dir, fname)
            print(f"Processing {fname} ...")
            process_single_json(json_path, modin_tables, output_dir)

    print("\n=== All files processed ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", default="")
    parser.add_argument("--csv_dir", default="")
    parser.add_argument("--out", default="")

    args = parser.parse_args()
    run_diff_test(args.json_dir, args.csv_dir, args.out)
