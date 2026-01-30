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
    if v is None:
        return None
    if isinstance(v, float) and (v != v):  # NaN
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
            return sorted(
                normalized_list,
                key=lambda x: json.dumps(x, sort_keys=True)
            )
        except Exception:
            return normalized_list

    return obj


def normalize_result(res):
    if res is None:
        return None
    return normalize_structure(res)

def analyze_one(data):
    cpu_err = data.get("Pyspark_Pandas_CPU", {}).get("Error")
    gpu_err = data.get("Pyspark_Pandas_GPU", {}).get("Error")

    cpu_res = normalize_result(data.get("Pyspark_Pandas_CPU", {}).get("Result"))
    gpu_res = normalize_result(data.get("Pyspark_Pandas_GPU", {}).get("Result"))

    result = {"CPU": None, "GPU": None}

    if cpu_err not in [None, "None", ""]:
        result["CPU"] = "error"
    else:
        result["CPU"] = "correct"

    if gpu_err not in [None, "None", ""]:
        result["GPU"] = "error"
    else:
        result["GPU"] = "correct"

    if result["CPU"] == "correct" and result["GPU"] == "correct":
        if cpu_res != gpu_res:
            result["GPU"] = "mismatch"

    return result

def analyze(input_dir, output_path):
    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    total = len(files)

    stats = {
        "CPU": {"correct": [], "error": []},
        "GPU": {"correct": [], "mismatch": [], "error": []},
        "total_files": total,
    }

    for fname in files:
        path = os.path.join(input_dir, fname)
        data = load_json(path)

        r = analyze_one(data)

        # CPU
        stats["CPU"][r["CPU"]].append(fname)
        # GPU
        stats["GPU"][r["GPU"]].append(fname)

    final = {"total_files": total}

    # CPU (no mismatch)
    final["CPU"] = {
        "correct": {
            "count": len(stats["CPU"]["correct"]),
            "ratio": round(len(stats["CPU"]["correct"]) / total, 4)
        },
        "error": {
            "count": len(stats["CPU"]["error"]),
            "ratio": round(len(stats["CPU"]["error"]) / total, 4),
            "files": stats["CPU"]["error"]
        }
    }

    # GPU (has mismatch)
    final["GPU"] = {
        "correct": {
            "count": len(stats["GPU"]["correct"]),
            "ratio": round(len(stats["GPU"]["correct"]) / total, 4)
        },
        "mismatch": {
            "count": len(stats["GPU"]["mismatch"]),
            "ratio": round(len(stats["GPU"]["mismatch"]) / total, 4),
            "files": stats["GPU"]["mismatch"]
        },
        "error": {
            "count": len(stats["GPU"]["error"]),
            "ratio": round(len(stats["GPU"]["error"]) / total, 4),
            "files": stats["GPU"]["error"]
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=4, ensure_ascii=False)

    print(f"Saved report to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    analyze(args.input_dir, args.output)
