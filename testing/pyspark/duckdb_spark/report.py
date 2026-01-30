import os
import json
from typing import Any, Optional, Dict, List
import argparse

def _norm_error(err: Any) -> Optional[str]:
    if err is None:
        return None
    if isinstance(err, str):
        s = err.strip()
        if s == "" or s.lower() in ("none", "null"):
            return None
        return s
    return str(err)

def _norm_result(res: Any) -> Any:
    return res

def _make_sorted_repr_if_list(res: Any) -> Any:
    """For lists of rows, create a sorted JSON string for comparison."""
    if isinstance(res, (list, tuple)):
        try:
            dumped = [json.dumps(x, sort_keys=True, ensure_ascii=False) for x in res]
        except TypeError:
            dumped = [str(x) for x in res]
        dumped.sort()
        return dumped
    return res

def _compare_results(left: Any, right: Any) -> bool:
    l = _make_sorted_repr_if_list(left)
    r = _make_sorted_repr_if_list(right)
    return l == r

def _init_stats_block() -> Dict[str, List[str]]:
    return {"correct": [], "mismatch": [], "spark_error": [], "duckdb_error": []}

def _empty_report_block(total: int, stats_block: Dict[str, List[str]]) -> Dict[str, Any]:
    def ratio(n: int) -> float:
        return round((n / total) if total else 0.0, 4)

    return {
        "correct": {"count": len(stats_block["correct"]), "ratio": ratio(len(stats_block["correct"]))},
        "mismatch": {
            "count": len(stats_block["mismatch"]),
            "ratio": ratio(len(stats_block["mismatch"])),
            "files": stats_block["mismatch"],
        },
        "spark_error": {
            "count": len(stats_block["spark_error"]),
            "ratio": ratio(len(stats_block["spark_error"])),
            "files": stats_block["spark_error"],
        },
        "duckdb_error": {
            "count": len(stats_block["duckdb_error"]),
            "ratio": ratio(len(stats_block["duckdb_error"])),
            "files": stats_block["duckdb_error"],
        },
    }

def analyze_json_dir(directory: str, output_path: str):
    total_files = 0
    stats = _init_stats_block()

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
                print(f"[JSON Error] {rel}: {type(e).__name__}: {e}")
                continue

            spark_err = _norm_error(data.get("Spark", {}).get("Error"))
            duckdb_err = _norm_error(data.get("DuckDB_Spark", {}).get("Error"))
            spark_res = _norm_result(data.get("Spark", {}).get("Result"))
            duckdb_res = _norm_result(data.get("DuckDB_Spark", {}).get("Result"))

            if spark_err:
                stats["spark_error"].append(rel)
            if duckdb_err:
                stats["duckdb_error"].append(rel)
            if not spark_err and not duckdb_err and not _compare_results(spark_res, duckdb_res):
                stats["mismatch"].append(rel)
            if not spark_err and not duckdb_err and _compare_results(spark_res, duckdb_res):
                stats["correct"].append(rel)

    report = {
        "total_files": total_files,
        "PythonCode": _empty_report_block(total_files, stats),
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    print(f"Analysis complete. Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default="./result/deepseek-reasoner/WHERE2/LLM+Err_Iter+RAG+DFG/result",
        help="Input directory with JSON files",
    )
    parser.add_argument(
        "--out",
        default="./result/deepseek-reasoner/WHERE2/LLM+Err_Iter+RAG+DFG/report.json",
        help="Output report JSON path",
    )
    args = parser.parse_args()

    analyze_json_dir(args.dir, args.out)
