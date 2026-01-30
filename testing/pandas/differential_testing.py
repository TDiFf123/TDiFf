import json
import pandas as pd
import numpy as np
import dask
import dask.dataframe as dd
import os
os.environ["MODIN_ENGINE"] = "ray"
import modin.pandas as md
import modin
import vaex
import re
import datetime
import argparse
import cudf
import dask_cudf

def init_pandas_tables(csv_dir):
    pandas_tables = {}
    for fname in os.listdir(csv_dir):
        if fname.lower().endswith(".csv"):
            table_name = os.path.splitext(fname)[0]
            path = os.path.join(csv_dir, fname)
            pandas_tables[table_name] = pd.read_csv(path)
    return pandas_tables

def init_dask_tables(csv_dir):
    dask_tables = {}
    for fname in os.listdir(csv_dir):
        if fname.lower().endswith(".csv"):
            table_name = os.path.splitext(fname)[0]
            path = os.path.join(csv_dir, fname)
            dask_tables[table_name] = dd.read_csv(path)
    return dask_tables

def init_modin_tables(csv_dir):
    modin_tables = {}
    for fname in os.listdir(csv_dir):
        if fname.lower().endswith(".csv"):
            table_name = os.path.splitext(fname)[0]
            path = os.path.join(csv_dir, fname)
            modin_tables[table_name] = md.read_csv(path)
    return modin_tables

def init_vaex_tables(csv_dir):
    vaex_tables = {}
    for fname in os.listdir(csv_dir):
        if fname.lower().endswith(".csv"):
            table_name = os.path.splitext(fname)[0]
            path = os.path.join(csv_dir, fname)
            vaex_tables[table_name] = vaex.from_csv(path)
    return vaex_tables

def init_cudf_tables(csv_dir):
    cudf_tables = {}
    for fname in os.listdir(csv_dir):
        if fname.lower().endswith(".csv"):
            table_name = os.path.splitext(fname)[0]
            path = os.path.join(csv_dir, fname)
            cudf_tables[table_name] = cudf.read_csv(path)
    return cudf_tables

def init_dask_cudf_tables(csv_dir):
    dask_cudf_tables = {}
    for fname in os.listdir(csv_dir):
        if fname.lower().endswith(".csv"):
            table_name = os.path.splitext(fname)[0]
            path = os.path.join(csv_dir, fname)
            dask_cudf_tables[table_name] = dask_cudf.read_csv(path)
    return dask_cudf_tables

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

def replace_pd_with_dd(code: str):
    pattern = r'(?<![A-Za-z0-9_])pd(?=\s*[\.\(\[])'
    return re.sub(pattern, "dd", code)

def exec_dask_statement(transfer_code, dask_tables):
    try:
        transfer_code_dd = replace_pd_with_dd(transfer_code)
        local_vars = {name: df for name, df in dask_tables.items()}
        local_vars["dd"] = dd
        local_vars["np"] = np
        exec(transfer_code_dd, local_vars, local_vars)
        transfer_result = local_vars["result"]
        if not isinstance(transfer_result, pd.DataFrame):
            transfer_result = transfer_result.compute()
        return transfer_result, None
    except Exception as e:
        print("python error message (dask): ")
        print(e)
        return None, e

def replace_pd_with_md(code: str):
    pattern = r'(?<![A-Za-z0-9_])pd(?=\s*[\.\(\[])'
    return re.sub(pattern, "md", code)

def exec_modin_statement(transfer_code, modin_tables):
    try:
        transfer_code_md = replace_pd_with_md(transfer_code)
        local_vars = {name: df for name, df in modin_tables.items()}
        local_vars["md"] = md
        local_vars["np"] = np
        exec(transfer_code_md, local_vars, local_vars)
        transfer_result = local_vars["result"]._to_pandas()
        return transfer_result, None
    except Exception as e:
        print("python error message (modin): ")
        print(e)
        return None, e

def replace_pd_with_vaex(code: str):
    pattern = r'(?<![A-Za-z0-9_])pd(?=\s*[\.\(\[])'
    return re.sub(pattern, "vaex", code)

def exec_vaex_statement(transfer_code, vaex_tables):
    try:
        transfer_code_vaex = replace_pd_with_vaex(transfer_code)
        local_vars = {name: df for name, df in vaex_tables.items()}
        local_vars["vaex"] = vaex
        local_vars["np"] = np
        exec(transfer_code_vaex, local_vars, local_vars)
        transfer_result = local_vars["result"].to_pandas_df()
        return transfer_result, None
    except Exception as e:
        print("python error message (vaex): ")
        print(e)
        return None, e

def replace_pd_with_cudf(code: str):
    pattern = r'(?<![A-Za-z0-9_])pd(?=\s*[\.\(\[])'
    return re.sub(pattern, "cudf", code)

def exec_cudf_statement(transfer_code, cudf_tables):
    try:
        transfer_code_cudf = replace_pd_with_cudf(transfer_code)
        local_vars = {name: df for name, df in cudf_tables.items()}
        local_vars["cudf"] = cudf
        local_vars["np"] = np
        exec(transfer_code_cudf, local_vars, local_vars)
        transfer_result = local_vars["result"].to_pandas()
        return transfer_result, None
    except Exception as e:
        print("python error message (cudf): ")
        print(e)
        return None, e

def replace_pd_with_dask_cudf(code: str):
    pattern = r'(?<![A-Za-z0-9_])pd(?=\s*[\.\(\[])'
    return re.sub(pattern, "dask_cudf", code)

def exec_dask_cudf_statement(transfer_code, dask_cudf_tables):
    try:
        transfer_code_dask_cudf = replace_pd_with_dask_cudf(transfer_code)
        local_vars = {name: df for name, df in dask_cudf_tables.items()}
        local_vars["dask_cudf"] = dask_cudf
        local_vars["np"] = np
        exec(transfer_code_dask_cudf, local_vars, local_vars)
        transfer_result = local_vars["result"].to_pandas()
        return transfer_result, None
    except Exception as e:
        print("python error message (cudf): ")
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
    dask_tables,
    modin_tables,
    vaex_tables,
    cudf_tables,
    dask_cudf_tables,
    output_dir,
    enable_vaex: bool,
):
    fname = os.path.basename(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transfer_code = data.get("FinalPythonCode")

    pandas_out, pandas_err = exec_pandas_statement(transfer_code, pandas_tables)
    dask_out, dask_err = exec_dask_statement(transfer_code, dask_tables)
    modin_out, modin_err = exec_modin_statement(transfer_code, modin_tables)
    cudf_out, cudf_err = exec_cudf_statement(transfer_code, cudf_tables)
    dask_cudf_out, dask_cudf_err = exec_dask_cudf_statement(transfer_code, dask_cudf_tables)

    output = {
        "PythonCode": transfer_code,
        "Numpy Version": np.__version__,
        "Pandas": {
            "Version": pd.__version__,
            "Result": pandas_out,
            "Error": str(pandas_err),
        },
        "Dask": {
            "Version": dask.__version__,
            "Result": dask_out,
            "Error": str(dask_err),
        },
        "Modin_Ray": {
            "Version": modin.__version__,
            "Backend": "ray",
            "Result": modin_out,
            "Error": str(modin_err),
        },
        "Cudf": {
            "Version": cudf.__version__,
            "Result": cudf_out,
            "Error": str(cudf_err),
        },
        "Dask_Cudf": {
            "Version": dask_cudf.__version__,
            "Result": dask_cudf_out,
            "Error": str(dask_cudf_err)
        }
    }

    if enable_vaex:
        vaex_out, vaex_err = exec_vaex_statement(transfer_code, vaex_tables)
        output["Vaex"] = {
            "Version": vaex.__version__["vaex"]
            if isinstance(vaex.__version__, dict) and "vaex" in vaex.__version__
            else str(vaex.__version__),
            "Result": vaex_out,
            "Error": str(vaex_err),
        }

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.splitext(fname)[0] + ".json"
    out_path = os.path.join(output_dir, out_file)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False, default=default_serializer)

    print(f"Saved → {out_path}")
    return output

def run_diff_test(json_dir, csv_dir, output_dir="output_results", enable_vaex: bool = False):
    pandas_tables = init_pandas_tables(csv_dir)
    dask_tables = init_dask_tables(csv_dir)
    modin_tables = init_modin_tables(csv_dir)
    cudf_tables = init_cudf_tables(csv_dir)
    dask_cudf_tables = init_dask_cudf_tables(csv_dir)

    vaex_tables = None
    if enable_vaex:
        vaex_tables = init_vaex_tables(csv_dir)

    for fname in os.listdir(json_dir):
        if fname.endswith(".json"):
            json_path = os.path.join(json_dir, fname)
            print(f"Processing {fname} ...")
            process_single_json(
                json_path,
                pandas_tables,
                dask_tables,
                modin_tables,
                vaex_tables,
                cudf_tables,
                dask_cudf_tables,
                output_dir,
                enable_vaex=enable_vaex,
            )

    print("\n=== All files processed ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", default="")
    parser.add_argument("--csv_dir", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--enable_vaex", default=True)
    args = parser.parse_args()
    run_diff_test(
        args.json_dir,
        args.csv_dir,
        args.out,
        enable_vaex=args.enable_vaex,
    )
