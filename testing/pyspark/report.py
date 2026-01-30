import os
import json
from typing import Any, Dict, List
import argparse

def _norm_error(err: Any) -> str:
    if err is None:
        return "None"
    if isinstance(err, str):
        s = err.strip()
        if s == "" or s.lower() in ("none", "null"):
            return "None"
        return s
    return str(err)

def _norm_result(res: Any) -> Any:
    return res if res is not None else []

def _ratio(n: int, total: int) -> float:
    return round(n / total, 4) if total > 0 else 0.0

def _normalize_result_for_comparison(result: Any) -> Any:
    if isinstance(result, list):
        try:
            return sorted(result, key=lambda x: json.dumps(x, sort_keys=True))
        except Exception:
            return sorted(map(str, result))
    return result

def analyze_json_files(directory: str, output_path: str):
    stats: Dict[str, List[str]] = {
        "mismatch": [],
        "cpu_error": [],
        "gpu_error": [],
        "correct": []
    }

    total_files = 0

    for root, _, files in os.walk(directory):
        for fname in files:
            if not fname.endswith(".json"):
                continue

            total_files += 1
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, directory)

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[JSON Error] {rel}: {e}")
                continue

            if "CPU" not in data or "GPU" not in data:
                print(f"[Skip] {rel}: missing CPU/GPU fields")
                continue

            cpu_res = _norm_result(data["CPU"].get("Result"))
            gpu_res = _norm_result(data["GPU"].get("Result"))
            cpu_err = _norm_error(data["CPU"].get("Error"))
            gpu_err = _norm_error(data["GPU"].get("Error"))

            if cpu_err != "None":
                stats["cpu_error"].append(rel)
                continue

            if gpu_err != "None":
                stats["gpu_error"].append(rel)
                continue

            cpu_res_n = _normalize_result_for_comparison(cpu_res)
            gpu_res_n = _normalize_result_for_comparison(gpu_res)

            if cpu_res_n != gpu_res_n:
                stats["mismatch"].append(rel)
            else:
                stats["correct"].append(rel)

    total = (
        len(stats["mismatch"])
        + len(stats["cpu_error"])
        + len(stats["gpu_error"])
        + len(stats["correct"])
    )

    report = {
        "total_files": total_files,
        "SQL": {
            "correct": {
                "count": len(stats["correct"]),
                "ratio": _ratio(len(stats["correct"]), total),
            },
            "mismatch": {
                "count": len(stats["mismatch"]),
                "ratio": _ratio(len(stats["mismatch"]), total),
                "files": stats["mismatch"],
            },
            "cpu_error": {
                "count": len(stats["cpu_error"]),
                "ratio": _ratio(len(stats["cpu_error"]), total),
                "files": stats["cpu_error"],
            },
            "gpu_error": {
                "count": len(stats["gpu_error"]),
                "ratio": _ratio(len(stats["gpu_error"]), total),
                "files": stats["gpu_error"],
            },
        }
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print(f"Analysis complete. Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    analyze_json_files(args.dir, args.out)
