import os
import json
import argparse
from datetime import datetime
import math


def _is_null_like(x):
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if isinstance(x, str) and x.strip().lower() in ("null", "nan", "nat", ""):
        return True
    return False


def _normalize_timestamp(value):
    if not isinstance(value, str):
        return value
    try:
        dt = datetime.fromisoformat(value.replace("T", " "))
        return dt.isoformat(sep=" ")
    except Exception:
        return value

def normalize_value(v):
    if _is_null_like(v):
        return None

    # timestamp normalization only
    if isinstance(v, str):
        ts = _normalize_timestamp(v)
        if ts != v:
            return ts
        return v  # <-- keep string, DO NOT convert to number

    if isinstance(v, float) and math.isnan(v):
        return None

    return v

def normalize_result(res):
    if res is None:
        return None

    # list results
    if isinstance(res, list):

        # list[dict] → normalize + sort
        if all(isinstance(row, dict) for row in res):
            normalized = [
                {k: normalize_value(v) for k, v in row.items()}
                for row in res
            ]
            return sorted(normalized, key=lambda r: json.dumps(r, sort_keys=True))

        # list[list] → normalize + sort (fix for Spark/DuckDB/Polars cross join)
        if all(isinstance(row, list) for row in res):
            normalized = [
                [normalize_value(v) for v in row]
                for row in res
            ]
            return sorted(normalized, key=lambda r: json.dumps(r, sort_keys=True))

        # list[scalar] or mixed
        return [normalize_value(row) for row in res]

    # dict results
    if isinstance(res, dict):
        return {k: normalize_value(v) for k, v in res.items()}

    # scalar values
    return normalize_value(res)

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
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                sql_res = normalize_result(data.get("SQLExecutionResult"))
                py_res = normalize_result(data.get("FinalPythonResult"))
                iteration_history = data.get("IterationHistory", [])

                # correct
                if sql_res == py_res and len(iteration_history) <= max_iter:
                    correct_count += 1

                # error
                elif py_res is None and sql_res is not None:
                    error_count += 1
                    error_files.append(filename)

                elif len(iteration_history) > max_iter:
                    error_count += 1
                    error_files.append(filename)

                # mismatch
                else:
                    logic_count += 1
                    logic_files.append(filename)

            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {filename}")
                continue

    result = {
        "total_files": total_files,
        "max_iter": max_iter,
        "correct": {
            "count": correct_count,
            "ratio": correct_count / total_files if total_files else 0
        },
        "mismatch": {
            "count": logic_count,
            "ratio": logic_count / total_files if total_files else 0,
            "files": logic_files
        },
        "error": {
            "count": error_count,
            "ratio": error_count / total_files if total_files else 0,
            "files": error_files
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"Analysis complete. Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--max_iter", type=int, default=1)
    args = parser.parse_args()

    analyze_json_files(args.dir, args.out, max_iter=args.max_iter)
