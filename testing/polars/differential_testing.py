import os
import argparse
import json
import datetime
import warnings

import pandas as pd
import polars as pl
from polars.exceptions import PerformanceWarning
import importlib.metadata


def init_tables(csv_dir: str):
    dataframe_tables = {}
    lazyframe_tables = {}
    for fname in sorted(os.listdir(csv_dir)):
        if fname.endswith(".csv"):
            table_name = os.path.splitext(fname)[0]
            path = os.path.join(csv_dir, fname)
            dataframe_tables[table_name] = pl.read_csv(path)
            lazyframe_tables[table_name] = pl.scan_csv(path)
    return dataframe_tables, lazyframe_tables


# =======================================================
#             EXECUTION MODES
# =======================================================

def exec_python_eager_statement(dataframe_tables, python_code):
    try:
        local_vars = {name: df.clone() for name, df in dataframe_tables.items()}
        local_vars["pl"] = pl
        exec(python_code, {}, local_vars)
        result = local_vars["result"]
        return result, None
    except Exception as e:
        print("python eager error: ")
        print(e)
        return None, e


def exec_python_lazy_statement(lazyframe_tables, python_code):
    try:
        local_vars = {name: df.clone() for name, df in lazyframe_tables.items()}
        local_vars["pl"] = pl
        exec(python_code, {}, local_vars)
        result = local_vars["result"]
        if isinstance(result, pl.DataFrame):
            result = result.lazy()
        result = result.collect()
        return result, None
    except Exception as e:
        print("python lazy error: ")
        print(e)
        return None, e


def exec_python_lazy_streaming_statement(lazyframe_tables, python_code):
    try:
        local_vars = {name: df.clone() for name, df in lazyframe_tables.items()}
        local_vars["pl"] = pl
        exec(python_code, {}, local_vars)
        result = local_vars["result"]
        if isinstance(result, pl.DataFrame):
            result = result.lazy()
        result = result.collect(engine="streaming")
        return result, None
    except Exception as e:
        print("python lazy streaming error: ")
        print(e)
        return None, e


def exec_python_lazy_gpu_statement(lazyframe_tables, python_code, raise_on_fail):
    try:
        local_vars = {name: df.clone() for name, df in lazyframe_tables.items()}
        local_vars["pl"] = pl
        exec(python_code, {}, local_vars)
        result = local_vars["result"]
        if isinstance(result, pl.DataFrame):
            result = result.lazy()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", PerformanceWarning)
            with pl.Config() as cfg:
                cfg.set_verbose(True)
                result = result.collect(engine=pl.GPUEngine(raise_on_fail=raise_on_fail))

        msgs = [str(warn.message) for warn in w if isinstance(warn.message, PerformanceWarning)]
        return result, None, msgs

    except Exception as e:
        print("python lazy gpu error: ")
        print(e)
        return None, e, None


def exec_python_lazy_gpu_streaming_statement(lazyframe_tables, python_code, raise_on_fail):
    try:
        local_vars = {name: df.clone() for name, df in lazyframe_tables.items()}
        local_vars["pl"] = pl
        exec(python_code, {}, local_vars)
        result = local_vars["result"]
        if isinstance(result, pl.DataFrame):
            result = result.lazy()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", PerformanceWarning)
            with pl.Config() as cfg:
                cfg.set_verbose(True)
                result = result.collect(
                    engine=pl.GPUEngine(
                        raise_on_fail=raise_on_fail,
                        executor="streaming",
                        executor_options={"fallback_mode": "raise"},
                    )
                )

        msgs = [str(warn.message) for warn in w if isinstance(warn.message, PerformanceWarning)]
        return result, None, msgs

    except Exception as e:
        print("python lazy gpu streaming error: ")
        print(e)
        return None, e, None


# =======================================================
# Serializer
# =======================================================

def default_serializer(obj):
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pl.DataFrame):
        return obj.to_dicts()
    elif isinstance(obj, Exception):
        return str(obj)
    return str(obj)


def save_result(result_dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False, default=default_serializer)


# =======================================================
#                   MAIN LOGIC
# =======================================================

def main(json_dir, csv_dir, output_dir, enable_gpu, enable_gpu_streaming,
         raise_on_fail, streaming, enable_eager):

    dataframe_tables, lazyframe_tables = init_tables(csv_dir)

    for fname in sorted(os.listdir(json_dir)):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(json_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        python_code = data.get("FinalPythonCode")
        if python_code is None:
            print(f"[Skip] No FinalPythonCode in {fname}")
            continue

        print(f"Processing {fname}: {python_code}")

        # ------------------ EAGER ------------------
        if enable_eager:
            eager_result, eager_error = exec_python_eager_statement(dataframe_tables, python_code)
        else:
            eager_result, eager_error = None, None

        # ------------------ LAZY CPU ------------------
        lazy_result, lazy_error = exec_python_lazy_statement(lazyframe_tables, python_code)

        # ------------------ LAZY STREAMING ------------------
        if streaming:
            lazy_stream_res, lazy_stream_err = exec_python_lazy_streaming_statement(lazyframe_tables, python_code)
        else:
            lazy_stream_res, lazy_stream_err = None, None

        # ------------------ LAZY GPU ------------------
        if enable_gpu:
            lazy_gpu_result, lazy_gpu_error, lazy_gpu_warn = exec_python_lazy_gpu_statement(lazyframe_tables, python_code, raise_on_fail)
        else:
            lazy_gpu_result, lazy_gpu_error, lazy_gpu_warn = None, None, None

        # ------------------ LAZY GPU STREAMING ------------------
        if enable_gpu_streaming:
            lazy_gpu_stream_res, lazy_gpu_stream_err, lazy_gpu_stream_warn = \
                exec_python_lazy_gpu_streaming_statement(lazyframe_tables, python_code, raise_on_fail)
        else:
            lazy_gpu_stream_res, lazy_gpu_stream_err, lazy_gpu_stream_warn = None, None, None

        # ==========================================================
        #                    BUILD JSON OUTPUT
        # ==========================================================
        output = {
            "DataBase Version": pl.__version__,
            "Python Code": python_code,
            "Lazy": {
                "Result": lazy_result,
                "Error": lazy_error
            }
        }

        # EAGER
        if enable_eager:
            output["Eager"] = {
                "Result": eager_result,
                "Error": eager_error,
            }

        # Lazy Streaming
        if streaming:
            output["Lazy_Streaming"] = {
                "Result": lazy_stream_res,
                "Error": lazy_stream_err,
            }

        # Lazy GPU
        if enable_gpu:
            output["Lazy_GPU"] = {
                "Cudf-Polars": importlib.metadata.version("cudf-polars"),
                "Result": lazy_gpu_result,
                "Error": lazy_gpu_error,
                "Warning": lazy_gpu_warn or []
            }

        # Lazy GPU Streaming
        if enable_gpu_streaming:
            output["Lazy_GPU_Streaming"] = {
                "Cudf-Polars": importlib.metadata.version("cudf-polars"),
                "Result": lazy_gpu_stream_res,
                "Error": lazy_gpu_stream_err,
                "Warning": lazy_gpu_stream_warn or []
            }

        # Save the output
        out_path = os.path.join(output_dir, fname)
        save_result(output, out_path)
        print(f"[Saved] {out_path}")


# =======================================================
# Entry
# =======================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", default="")
    parser.add_argument("--csv_dir", default="")
    parser.add_argument("--output_dir", default="")

    parser.add_argument("--gpu", type=bool, default=True)
    parser.add_argument("--gpu_streaming", type=bool, default=True)
    parser.add_argument("--raise_on_fail", type=bool, default=True)

    parser.add_argument("--streaming", type=bool, default=False)
    parser.add_argument("--eager", type=bool, default=False)

    args = parser.parse_args()

    main(
        json_dir=args.json_dir,
        csv_dir=args.csv_dir,
        output_dir=args.output_dir,
        enable_gpu=args.gpu,
        enable_gpu_streaming=args.gpu_streaming,
        raise_on_fail=args.raise_on_fail,
        streaming=args.streaming,
        enable_eager=args.eager,
    )
