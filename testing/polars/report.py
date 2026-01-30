import os
import json
import math
import argparse
from typing import Any, Optional, Dict, List

FALLBACK_TOKENS = [
    "Query execution with GPU not possible: unsupported operations"
]

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

def _has_warning(warn: Any) -> bool:
    if warn is None:
        return False
    if isinstance(warn, list):
        return any(_norm_error(x) for x in warn)
    return _norm_error(warn) is not None

def _init_baseline_block():
    return {"correct": [], "error": []}

def _init_stats_block(include_fallback: bool):
    b = {"correct": [], "mismatch": [], "error": []}
    if include_fallback:
        b["fallback"] = []
    return b

def _empty_report_block(total, blk, is_gpu):
    def ratio(n):
        return round(n/total, 4) if total else 0.0

    out = {
        "correct": {"count": len(blk["correct"]), "ratio": ratio(len(blk["correct"]))},
        "error": {
            "count": len(blk["error"]), "ratio": ratio(len(blk["error"])),
            "files": blk["error"]
        }
    }

    if "mismatch" in blk:
        out["mismatch"] = {
            "count": len(blk["mismatch"]),
            "ratio": ratio(len(blk["mismatch"])),
            "files": blk["mismatch"]
        }

    if is_gpu and "fallback" in blk:
        out["fallback"] = {
            "count": len(blk["fallback"]),
            "ratio": ratio(len(blk["fallback"])),
            "files": blk["fallback"]
        }

    return out

def _is_fallback(err, warn):
    if _has_warning(warn):
        return True
    if err is None:
        return False
    s = err.lower()
    return any(tok.lower() in s for tok in FALLBACK_TOKENS)

def _sortable_repr(obj):
    try:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except:
        return str(obj)

def _normalize(obj):
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, list):
        return [_normalize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _normalize(v) for k,v in obj.items()}
    return obj

def _compare_results(a, b, sort_before):
    a = _normalize(a)
    b = _normalize(b)

    # identical
    if a == b:
        return True

    # numeric tolerance
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)

    # dict comparison ignoring key order
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_compare_results(a[k], b[k], sort_before) for k in a)

    # list comparison ignoring order
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        sa = sorted(_sortable_repr(x) for x in a)
        sb = sorted(_sortable_repr(x) for x in b)
        return sa == sb

    return a == b

def _extract_blocks(d):
    # baseline Lazy
    lb = d.get("Lazy")
    if lb:
        lazy_res = _norm_result(lb.get("Result"))
        lazy_err = _norm_error(lb.get("Error"))
    else:
        lazy_res = None
        lazy_err = "NO_LAZY"

    # Eager
    eb = d.get("Eager")
    if eb:
        eager_res = _norm_result(eb.get("Result"))
        eager_err = _norm_error(eb.get("Error"))
        has_eager = True
    else:
        eager_res = None
        eager_err = None
        has_eager = False

    # Streaming
    sb = d.get("Lazy_Streaming")
    if sb:
        stream_res = _norm_result(sb.get("Result"))
        stream_err = _norm_error(sb.get("Error"))
        has_stream = True
    else:
        stream_res = None
        stream_err = None
        has_stream = False

    # GPU
    gb = d.get("Lazy_GPU")
    if gb:
        gpu_res = _norm_result(gb.get("Result"))
        gpu_err = _norm_error(gb.get("Error"))
        gpu_warn = gb.get("Warning")
        has_gpu = True
    else:
        gpu_res = None
        gpu_err = None
        gpu_warn = None
        has_gpu = False

    # GPU Streaming
    gsb = d.get("Lazy_GPU_Streaming")
    if gsb:
        gpu_s_res = _norm_result(gsb.get("Result"))
        gpu_s_err = _norm_error(gsb.get("Error"))
        gpu_s_warn = gsb.get("Warning")
        has_gpu_s = True
    else:
        gpu_s_res = None
        gpu_s_err = None
        gpu_s_warn = None
        has_gpu_s = False

    return (
        lazy_res, lazy_err,
        eager_res, eager_err, has_eager,
        stream_res, stream_err, has_stream,
        gpu_res, gpu_err, gpu_warn, has_gpu,
        gpu_s_res, gpu_s_err, gpu_s_warn, has_gpu_s
    )

def analyze_json_dir(directory, output_path):
    total_lazy = 0
    total_eager = 0
    total_stream = 0
    total_gpu = 0
    total_gpu_s = 0

    stats = {
        "Lazy": _init_baseline_block(),
        "Eager": _init_stats_block(False),
        "Streaming": _init_stats_block(False),
    }

    have_eager = False
    have_stream = False
    have_gpu = False
    have_gpu_s = False

    for root, _, files in os.walk(directory):
        for fname in files:
            if not fname.endswith(".json"):
                continue

            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, directory)

            try:
                d = json.load(open(fpath, "r", encoding="utf-8"))
            except:
                continue

            (
                lazy_res, lazy_err,
                eager_res, eager_err, has_eager_block,
                stream_res, stream_err, has_stream_block,
                gpu_res, gpu_err, gpu_warn, has_gpu_block,
                gpu_s_res, gpu_s_err, gpu_s_warn, has_gpu_s_block
            ) = _extract_blocks(d)

            sort_before = True

            # Lazy baseline
            total_lazy += 1
            if lazy_err is None:
                stats["Lazy"]["correct"].append(rel)
            else:
                stats["Lazy"]["error"].append(rel)

            # Eager vs Lazy
            if has_eager_block:
                have_eager = True
                total_eager += 1
                if eager_err or lazy_err:
                    stats["Eager"]["error"].append(rel)
                else:
                    if _compare_results(eager_res, lazy_res, sort_before):
                        stats["Eager"]["correct"].append(rel)
                    else:
                        stats["Eager"]["mismatch"].append(rel)

            # Streaming vs Lazy
            if has_stream_block:
                have_stream = True
                total_stream += 1
                if stream_err or lazy_err:
                    stats["Streaming"]["error"].append(rel)
                else:
                    if _compare_results(stream_res, lazy_res, sort_before):
                        stats["Streaming"]["correct"].append(rel)
                    else:
                        stats["Streaming"]["mismatch"].append(rel)

            # GPU vs Lazy
            if has_gpu_block:
                if not have_gpu:
                    stats["GPU"] = _init_stats_block(True)
                    have_gpu = True
                total_gpu += 1

                if gpu_err or lazy_err:
                    stats["GPU"]["error"].append(rel)
                    if _is_fallback(gpu_err, gpu_warn):
                        stats["GPU"]["fallback"].append(rel)
                else:
                    if _compare_results(gpu_res, lazy_res, sort_before):
                        stats["GPU"]["correct"].append(rel)
                    else:
                        stats["GPU"]["mismatch"].append(rel)

                    if _is_fallback(None, gpu_warn):
                        stats["GPU"]["fallback"].append(rel)

            # GPU Streaming vs GPU
            if has_gpu_s_block:
                if not have_gpu_s:
                    stats["GPU_Streaming"] = _init_stats_block(True)
                    have_gpu_s = True
                total_gpu_s += 1

                if gpu_s_err or gpu_err:
                    stats["GPU_Streaming"]["error"].append(rel)
                    if _is_fallback(gpu_s_err, gpu_s_warn):
                        stats["GPU_Streaming"]["fallback"].append(rel)
                else:
                    if _compare_results(gpu_s_res, gpu_res, sort_before):
                        stats["GPU_Streaming"]["correct"].append(rel)
                    else:
                        stats["GPU_Streaming"]["mismatch"].append(rel)

                    if _is_fallback(None, gpu_s_warn):
                        stats["GPU_Streaming"]["fallback"].append(rel)

    # Build final report
    report = {
        "total_files": total_lazy,
        "Lazy": _empty_report_block(total_lazy, stats["Lazy"], False)
    }

    if have_eager:
        report["Eager"] = _empty_report_block(total_eager, stats["Eager"], False)
    if have_stream:
        report["Streaming"] = _empty_report_block(total_stream, stats["Streaming"], False)
    if have_gpu:
        report["GPU"] = _empty_report_block(total_gpu, stats["GPU"], True)
    if have_gpu_s:
        report["GPU_Streaming"] = _empty_report_block(total_gpu_s, stats["GPU_Streaming"], True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    json.dump(report, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)

    print("Analysis complete:", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    analyze_json_dir(args.dir, args.out)
