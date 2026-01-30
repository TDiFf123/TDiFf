import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import re


DATETIME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}")

def normalize_datetime(v):
    """ Normalize datetime strings but do NOT convert pure date to datetime """
    if not isinstance(v, str):
        return v

    # Pure date → must remain distinct (strict rule)
    try:
        dt = datetime.strptime(v, "%Y-%m-%d")
        return v
    except Exception:
        pass

    cleaned = v.replace("Z", "").split("+")[0]

    for fmt in [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
    ]:
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

    return v


def normalize_null(v):
    """ Normalize null-equivalent values """
    if v is None:
        return None
    if v is pd.NaT:
        return None
    if isinstance(v, np.datetime64) and np.isnat(v):
        return None
    if isinstance(v, float) and (v != v):  # NaN
        return None
    if isinstance(v, str) and v.lower() in ("nan", "null", "nat"):
        return None
    return v


def normalize_value(v):
    v = normalize_null(v)
    v = normalize_datetime(v)
    return v

def compare_values_only(sql_res, py_res):
    if not isinstance(sql_res, list) or not isinstance(py_res, list):
        return False

    if len(sql_res) != len(py_res):
        return False

    sql_norm = []
    py_norm = []

    for row_sql in sql_res:
        if not isinstance(row_sql, dict):
            return False
        sql_norm.append([normalize_value(v) for v in row_sql.values()])

    for row_py in py_res:
        if not isinstance(row_py, dict):
            return False
        py_norm.append([normalize_value(v) for v in row_py.values()])

    sql_norm_sorted = sorted(sql_norm, key=lambda row: json.dumps(row, sort_keys=True, default=str))
    py_norm_sorted = sorted(py_norm, key=lambda row: json.dumps(row, sort_keys=True, default=str))

    return sql_norm_sorted == py_norm_sorted


def analyze_json_files(directory, output_path, max_iter=1):
    total_files = 0
    correct_count = 0
    logic_count = 0
    error_count = 0

    logic_files = []
    error_files = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            total_files += 1
            file_path = os.path.join(directory, filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                sql_execution_result = data.get("SQLExecutionResult")
                final_python_result = data.get("FinalPythonResult")
                iteration_history = data.get("IterationHistory", [])

                # ---------- Correct ----------
                if compare_values_only(sql_execution_result, final_python_result) and \
                   len(iteration_history) <= max_iter:
                    correct_count += 1

                # ---------- Error ----------
                elif (final_python_result is None and sql_execution_result is not None) or \
                     len(iteration_history) > max_iter:
                    error_count += 1
                    error_files.append(filename)

                # ---------- Logic ----------
                else:
                    logic_count += 1
                    logic_files.append(filename)

            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {filename}")
                continue

    correct_ratio = correct_count / total_files if total_files else 0
    logic_ratio = logic_count / total_files if total_files else 0
    error_ratio = error_count / total_files if total_files else 0

    result = {
        "total_files": total_files,
        "max_iter": max_iter,
        "correct": {
            "count": correct_count,
            "ratio": correct_ratio
        },
        "mismatch": {
            "count": logic_count,
            "ratio": logic_ratio,
            "files": logic_files
        },
        "error": {
            "count": error_count,
            "ratio": error_ratio,
            "files": error_files
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"Analysis complete. Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--max_iter", type=int, default=1)

    args = parser.parse_args()

    analyze_json_files(args.dir, args.out, max_iter=args.max_iter)
