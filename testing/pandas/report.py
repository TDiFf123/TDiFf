import os
import json
import argparse
import re
from datetime import datetime

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"LOAD_ERROR": str(e)}


DATETIME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}")

def normalize_datetime_equiv(v):
    if not isinstance(v, str):
        return v

    if not DATETIME_PATTERN.match(v):
        return v

    cleaned = v.replace("Z", "").split("+")[0]

    for fmt in [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d",
    ]:
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

    return v


def normalize_null_equiv(v):
    """
    Normalize null-equivalent values:
    NaN, null → None
    """
    if v is None:
        return None
    if isinstance(v, float) and (v != v):  # NaN check
        return None
    if isinstance(v, str) and v.lower() in ("nan", "null"):
        return None
    return v


def normalize_structure(obj):
    obj = normalize_datetime_equiv(obj)
    obj = normalize_null_equiv(obj)

    if isinstance(obj, dict):
        return {k: normalize_structure(obj[k]) for k in sorted(obj.keys())}

    if isinstance(obj, list):
        normalized_list = [normalize_structure(x) for x in obj]
        try:
            return sorted(normalized_list, key=lambda x: json.dumps(x, sort_keys=True))
        except Exception:
            return normalized_list

    return obj


def normalize_result(res):
    if res is None:
        return None
    return normalize_structure(res)


def analyze_one(data, available_engines):
    pandas_err = data.get("Pandas", {}).get("Error") or data.get("Pandas", {}).get("ERROR")
    pandas_res = normalize_result(data.get("Pandas", {}).get("Result"))

    result = {"Pandas": None}

    if pandas_err not in [None, "None", ""]:
        result["Pandas"] = "error"
    else:
        result["Pandas"] = "correct"

    def check_engine(name):
        eng = data.get(name, {})
        err = eng.get("Error") or eng.get("ERROR")
        res = normalize_result(eng.get("Result"))

        if err not in [None, "None", ""]:
            return "error"

        if eng.get("Result") is None and err in [None, ""]:
            return "error"

        if res == pandas_res:
            return "correct"

        return "mismatch"

    for eng in available_engines:
        result[eng] = check_engine(eng)

    return result


def analyze(input_dir, output_path):
    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    total = len(files)

    sample = load_json(os.path.join(input_dir, files[0]))

    possible_engines = [
        "Dask",
        "Modin_Ray",
        "Vaex",
        "Cudf",
        "Cudf_Pandas",
        "Dask_Cudf",
        "Modin_Dask"
    ]

    available_engines = [e for e in possible_engines if e in sample]

    print("Detected engines:", available_engines)

    stats = {
        "total_files": total,
        "Pandas": {"correct": [], "error": []},
    }

    for eng in available_engines:
        stats[eng] = {"correct": [], "mismatch": [], "error": []}

    for fname in files:
        path = os.path.join(input_dir, fname)
        data = load_json(path)
        result = analyze_one(data, available_engines)

        stats["Pandas"][result["Pandas"]].append(fname)

        for eng in available_engines:
            stats[eng][result[eng]].append(fname)

    final = {"total_files": total}

    final["Pandas"] = {
        "correct": {
            "count": len(stats["Pandas"]["correct"]),
            "ratio": round(len(stats["Pandas"]["correct"]) / total, 4),
        },
        "error": {
            "count": len(stats["Pandas"]["error"]),
            "ratio": round(len(stats["Pandas"]["error"]) / total, 4),
            "files": stats["Pandas"]["error"],
        },
    }

    for eng in available_engines:
        correct = len(stats[eng]["correct"])
        mismatch = len(stats[eng]["mismatch"])
        error = len(stats[eng]["error"])

        final[eng] = {
            "correct": {"count": correct, "ratio": round(correct / total, 4)},
            "mismatch": {
                "count": mismatch,
                "ratio": round(mismatch / total, 4),
                "files": stats[eng]["mismatch"],
            },
            "error": {
                "count": error,
                "ratio": round(error / total, 4),
                "files": stats[eng]["error"],
            },
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=4, ensure_ascii=False)

    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="")
    parser.add_argument("--output", default="son")
    args = parser.parse_args()
    analyze(args.input_dir, args.output)
