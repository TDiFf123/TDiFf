import re
from collections import deque
from typing import List, Tuple, Dict, Optional, Set
import time
import requests

def dot_to_ascii(dot: str, dfg_host_port: str, fancy: bool = True, retries: int = 3, delay: int = 10):
    url = dfg_host_port
    boxart = 1 if fancy else 0

    params = {
        'boxart': boxart,
        'src': dot,
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()  # raise exception if HTTP status code is not 200
            text = response.text

            if text == '':
                raise SyntaxError('DOT string is not formatted correctly')

            return text

        except Exception as e:
            if attempt < retries:
                print(f"[Retry {attempt}/{retries}] Request failed: {e}, retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise RuntimeError(f"Request failed after maximum retry attempts ({retries})") from e

LINE_CHARS = set("│─┌┐└┘├┤┬┴┼")
BOX_CHARS_RE = re.compile(r"[│┌┐└┘├┤┬┴┼─═╞╡╪╫╣╠║╔╗╚╝╟╢]+")

def nows(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def clean_line(s: str) -> str:
    s = BOX_CHARS_RE.sub("", s)
    s = nows(s)
    if not s or set(s) <= {"─", " "}:
        return ""
    return s

def pad_grid(lines: List[str]) -> List[str]:
    w = max((len(x) for x in lines), default=0)
    return [ln.rstrip("\n").ljust(w) for ln in lines]


DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPP = {"U": "D", "D": "U", "L": "R", "R": "L"}

CHAR_DIRS = {
    "│": {"U", "D"},
    "─": {"L", "R"},
    "┌": {"R", "D"},
    "┐": {"L", "D"},
    "└": {"R", "U"},
    "┘": {"L", "U"},
    "├": {"U", "D", "R"},
    "┤": {"U", "D", "L"},
    "┬": {"L", "R", "D"},
    "┴": {"L", "R", "U"},
    "┼": {"U", "D", "L", "R"},
}

def neighbors_of(grid: List[str], r: int, c: int):
    ch = grid[r][c]
    if ch not in CHAR_DIRS:
        return
    for d, (dr, dc) in DIRS.items():
        if d not in CHAR_DIRS[ch]:
            continue
        r2, c2 = r + dr, c + dc
        if 0 <= r2 < len(grid) and 0 <= c2 < len(grid[0]):
            ch2 = grid[r2][c2]
            if ch2 in CHAR_DIRS and OPP[d] in CHAR_DIRS[ch2]:
                yield (r2, c2)



def _pairs(line: str, l: str, r: str) -> List[Tuple[int, int]]:
    spans, i, L = [], 0, len(line)
    while i < L:
        if line[i] == l:
            j = line.find(r, i + 1)
            if j != -1:
                spans.append((i, j))
                i = j + 1
                continue
        i += 1
    return spans

def _top_spans(line: str): return _pairs(line, "┌", "┐")
def _mid_spans(line: str): return _pairs(line, "│", "│")
def _bot_spans(line: str): return _pairs(line, "└", "┘")

def _span_dist(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _match_span(span: Tuple[int, int], candidates: List[Tuple[int, int]], tol: int) -> Optional[Tuple[int, int]]:
    if not candidates: return None
    best = min(candidates, key=lambda s: _span_dist(span, s))
    return best if _span_dist(span, best) <= tol else None

def split_boxes_with_positions(lines: List[str], tol_cols: int = 3):

    def _top_spans_inner(line: str) -> List[Tuple[int, int]]:
        spans, i, L = [], 0, len(line)
        while i < L:
            if line[i] == "┌":
                j = line.find("┐", i + 1)
                if j != -1:
                    spans.append((i, j))
                    i = j + 1
                    continue
            i += 1
        return spans

    def _find_char_near(line: str, target_idx: int, chars: Set[str], tol: int) -> Optional[int]:
        L = len(line)
        lo = max(0, target_idx - tol)
        hi = min(L - 1, target_idx + tol)
        for d in range(0, tol + 1):
            # 精确位
            if 0 <= target_idx < L and line[target_idx] in chars:
                return target_idx
            for sign in (-1, 1):
                idx = target_idx + sign * d
                if lo <= idx <= hi and line[idx] in chars:
                    return idx
        return None

    boxes_done: List[Dict] = []

    open_boxes: Dict[Tuple[int, int, int], Dict] = {}

    for r, line in enumerate(lines):

        for (l, rr) in _top_spans_inner(line):
            key = (l, rr, r)
            open_boxes[key] = {
                "left": l,
                "right": rr,
                "top_row": r,
                "bottom_row": r,
                "rows": [(r, l, rr, line[l:rr + 1])],
            }

        if not open_boxes:
            continue


        to_close = []
        for key, box in list(open_boxes.items()):
            l_box, r_box = box["left"], box["right"]

            l_idx = _find_char_near(line, l_box,
                                    {"│", "├", "┤", "┬", "┴", "┼", "┌", "└", "║", "╟", "╠"},
                                    tol_cols)
            r_idx = _find_char_near(line, r_box,
                                    {"│", "├", "┤", "┬", "┴", "┼", "┐", "┘", "║", "╢", "╣"},
                                    tol_cols)

            if l_idx is not None and r_idx is not None:
                seg = line[l_box:r_box + 1]
                if (r_box - l_box) >= 2 and len(seg) == (r_box - l_box + 1):
                    box["rows"].append((r, l_box, r_box, seg))
                    box["bottom_row"] = max(box["bottom_row"], r)

                l_bot = _find_char_near(line, l_box, {"└"}, tol_cols)
                r_bot = _find_char_near(line, r_box, {"┘"}, tol_cols)
                if l_bot is not None and r_bot is not None and l_bot < r_bot:
                    to_close.append(key)

        for key in to_close:
            boxes_done.append(open_boxes.pop(key))

    boxes_done.extend(list(open_boxes.values()))
    return boxes_done


KNOWN_KINDS = {
    "ORDER_BY", "PROJECTION", "COMPARISON_JOIN", "DEPENDENT_JOIN",
    "FILTER", "AGGREGATE", "SEQ_SCAN", "CTE", "CTE_SCAN", "PANDAS_SCAN",
    "HASH_JOIN", "HASH_GROUP_BY",
    "CROSS_PRODUCT", "ANY_JOIN",
    "LIMIT", "STREAMING_LIMIT",
    "UNION",  "DISTINCT"
}

SEMANTIC_HINTS = (
    "Table:", "Join Type:", "Conditions:", "Condition:", "Groups:", "Aggregates:",
    "Expressions:", "CTE Name:", "CTE Index:", "Distinct Targets:"
)

def extract_payload_from_box_rows(box, grid: List[str]) -> List[str]:
    payload = []
    for (r, l, rr, seg) in box["rows"]:
        m = re.match(r"^.*?│(.*?)│.*?$", seg)
        text = m.group(1) if m else seg
        text = clean_line(text)
        if text:
            payload.append(text)
    payload = [p for p in payload if p and set(p) != {"─"}]
    return payload

def parse_kind_payload(payload: List[str]) -> Tuple[str, List[str]]:
    kind = "UNKNOWN"
    if payload:
        k0_raw = payload[0]
        k0 = nows(k0_raw).upper()

        if k0.startswith("UNION"):
            return "UNION", payload

        k0 = k0.replace("COMPARISON JOIN", "COMPARISON_JOIN") \
               .replace("STREAMING LIMIT", "STREAMING_LIMIT")
        if k0 in KNOWN_KINDS:
            kind = k0
        else:
            for p in payload:
                pp = nows(p).upper()
                pp = pp.replace("COMPARISON JOIN", "COMPARISON_JOIN") \
                       .replace("STREAMING LIMIT", "STREAMING_LIMIT")
                if pp in KNOWN_KINDS:
                    kind = pp
                    break
    return kind, payload

def parse_info(kind: str, payload: List[str]) -> dict:
    info = {}
    if kind == "ORDER_BY":
        info["order_by"] = [p for p in payload[1:] if not p.lower().startswith("type:")]
    elif kind == "DISTINCT":
        cols, started = [], False
        for p in payload[1:]:
            if p.lower().startswith("distinct targets"):
                started = True
                if ":" in p:
                    expr = p.split(":", 1)[1].strip()
                    if expr:
                        cols.append(expr)
                continue
            if started:
                q = p.strip()
                if q:
                    cols.append(q)
        info["distinct targets"] = cols
    elif kind == "PROJECTION":
        cols, started = [], False
        for p in payload[1:]:
            if p.lower().startswith("expressions"):
                started = True
                if ":" in p:
                    expr = p.split(":", 1)[1].strip()
                    if expr:
                        cols.append(expr)
                continue
            if started:
                q = p.strip()
                if q:
                    cols.append(q)
        info["expressions"] = cols
    elif kind in {"COMPARISON_JOIN", "DEPENDENT_JOIN", "HASH_JOIN", "ANY_JOIN"}:
        jt, conds = None, []
        in_conds_block = False

        def looks_like_condition(s: str) -> bool:
            t = s.strip().lower()
            ops = ["=", "<", ">", "<=", ">=", "!=", " is ", " like ", " in ", " between ", " and ", " or "]
            return any(op in t for op in ops) or t in {"true", "false"}

        for p in payload[1:]:
            pl = p.lower()

            if pl.startswith("join type:"):
                jt = nows(p.split(":", 1)[1]).upper()
                in_conds_block = False
                continue

            if pl.startswith("conditions:") or pl.startswith("condition:"):
                rhs = p.split(":", 1)[1].strip()
                if rhs:
                    conds.append(nows(rhs))
                in_conds_block = True
                continue

            if pl.startswith(("groups:", "aggregates:", "expressions:", "table:",
                              "cte name:", "cte index:")):
                in_conds_block = False
                continue

            q = p.strip()
            if not q:
                continue
            if in_conds_block or looks_like_condition(q):
                conds.append(nows(q))

        info["join_type"] = jt or ("SINGLE" if kind == "DEPENDENT_JOIN" else ("ANY" if kind == "ANY_JOIN" else "?"))
        info["conditions"] = conds
    elif kind == "FILTER":
        cols, started = [], False
        for p in payload[1:]:
            if p.lower().startswith("expressions"):
                started = True
                if ":" in p:
                    expr = p.split(":", 1)[1].strip()
                    if expr:
                        cols.append(expr)
                continue
            if started:
                q = p.strip()
                if q:
                    cols.append(q)
        info["expressions"] = cols
    elif kind in {"AGGREGATE", "HASH_GROUP_BY"}:
        groups, expr = None, []
        for p in payload[1:]:
            pl = p.lower()
            if pl.startswith("groups:"):
                groups = nows(p.split(":", 1)[1])
            elif pl.startswith("expressions") or pl.startswith("aggregates"):
                continue
            else:
                q = p.strip()
                if q:
                    expr.append(q)
        if groups:
            info["groups"] = groups
        info["expressions"] = expr
    elif kind == "SEQ_SCAN":
        tbl = None
        for p in payload[1:]:
            if p.lower().startswith("table:"):
                tbl = nows(p.split(":", 1)[1])
        info["table"] = tbl or "?"
    elif kind == "PANDAS_SCAN":
        tbl = None
        for p in payload[1:]:
            if p.lower().startswith("table:"):
                tbl = nows(p.split(":", 1)[1])
        info["table"] = tbl or "?"
    elif kind == "CTE":
        name = None
        for p in payload[1:]:
            if p.lower().startswith("cte name:"):
                name = nows(p.split(":", 1)[1])
        info["cte_name"] = name or "?"
    elif kind == "CTE_SCAN":
        idx = None
        for p in payload[1:]:
            if p.lower().startswith("cte index:"):
                idx = nows(p.split(":", 1)[1])
        info["cte_index"] = idx or "?"
    elif kind == "UNION":
        pass
    return info

def label_of(kind: str, info: dict) -> str:
    if kind == "ORDER_BY":
        return "ORDER_BY [" + ", ".join(info.get("order_by", [])) + "]"
    if kind == "PROJECTION":
        return "PROJECTION [" + ", ".join(info.get("expressions", [])) + "]"
    if kind == "DISTINCT":
        return "DISTINCT [" + ", ".join(info.get("distinct targets", [])) + "]"
    if kind in {"COMPARISON_JOIN", "DEPENDENT_JOIN", "HASH_JOIN", "ANY_JOIN"}:
        jt = info.get("join_type", "?")
        cond = ", ".join(info.get("conditions", []))
        if jt == "?":
            jt = kind
        else:
            jt = jt + " JOIN"
        return f"{jt} [{cond}]"
    if kind == "FILTER":
        return "FILTER [" + ", ".join(info.get("expressions", [])) + "]"
    if kind in {"AGGREGATE", "HASH_GROUP_BY"}:
        groups = info.get("groups")
        expr = info.get("expressions", [])
        if groups and expr:
            return f"{kind} [Groups: {groups}; " + ", ".join(expr) + "]"
        if groups:
            return f"{kind} [Groups: {groups}]"
        return f"{kind} [" + ", ".join(expr) + "]"
    if kind == "SEQ_SCAN":
        return f"SEQ_SCAN [{info.get('table', '?')}]"
    if kind == "PANDAS_SCAN":
        return f"PANDAS_SCAN [{info.get('table', '?')}]"
    if kind == "CTE":
        return f"CTE [{info.get('cte_name', '?')}]"
    if kind == "CTE_SCAN":
        return f"CTE_SCAN [Index: {info.get('cte_index', '?')}]"
    if kind == "UNION":
        return "UNION"
    return kind

def is_semantic_box(payload: List[str], kind: str) -> bool:
    if kind in KNOWN_KINDS:
        return True
    for p in payload:
        for hint in SEMANTIC_HINTS:
            if hint.lower() in p.lower():
                return True
    return False


def mark_box_cells(box, grid: List[str]) -> Set[Tuple[int, int]]:
    cells = set()
    for (r, l, rr, seg) in box["rows"]:
        for off, ch in enumerate(seg):
            if ch in LINE_CHARS:
                c = l + off
                cells.add((r, c))
    return cells

def top_anchors(box, grid: List[str]) -> List[Tuple[int, int]]:
    r, l, rr = box["top_row"], box["left"], box["right"]
    return [(r, c) for c in range(l, rr + 1) if grid[r][c] == "┴"]

def compute_forbidden_cells(boxes: List[Dict], grid: List[str]) -> Set[Tuple[int, int]]:
    forbidden: Set[Tuple[int, int]] = set()
    if not grid:
        return forbidden
    max_rows = len(grid)
    max_cols = len(grid[0])

    for b in boxes:
        top = max(0, min(max_rows - 1, b["top_row"]))
        bot = max(0, min(max_rows - 1, b["bottom_row"]))
        left = max(0, min(max_cols - 1, b["left"]))
        right = max(0, min(max_cols - 1, b["right"]))
        for r in range(top, bot + 1):
            for c in range(left, right + 1):
                forbidden.add((r, c))
    return forbidden


def bfs_find_parent(start_anchor: Tuple[int, int],
                    grid: List[str],
                    owner_map: Dict[Tuple[int, int], int],
                    self_id: int,
                    node_tops: Dict[int, int],
                    forbidden: Optional[Set[Tuple[int, int]]] = None,
                    max_steps: int = 10000) -> Optional[int]:

    if not grid:
        return None

    if forbidden is None:
        forbidden = set()

    R, C = len(grid), len(grid[0])
    sr, sc = start_anchor
    q = deque()
    seen = set()

    if sr - 1 >= 0 and grid[sr - 1][sc] in CHAR_DIRS:
        q.append((sr - 1, sc))
        seen.add((sr - 1, sc))

    child_top = node_tops.get(self_id, None)

    steps = 0
    while q and steps < max_steps:
        r, c = q.popleft()
        steps += 1

        owner = owner_map.get((r, c))
        if owner is not None and owner != self_id:
            parent_top = node_tops.get(owner, None)
            if child_top is None or parent_top is None or parent_top < child_top:
                return owner

        for (nr, nc) in neighbors_of(grid, r, c):
            if (nr, nc) in seen:
                continue

            if (nr, nc) in forbidden:
                seen.add((nr, nc))
                owner2 = owner_map.get((nr, nc))
                if owner2 is not None and owner2 != self_id:
                    parent_top = node_tops.get(owner2, None)
                    if child_top is None or parent_top is None or parent_top < child_top:
                        return owner2

                continue


            seen.add((nr, nc))
            q.append((nr, nc))

    return None


def plan_to_edges_from_text(plan_text: str) -> List[Tuple[str, str]]:
    raw_lines = plan_text.splitlines()
    grid = pad_grid(raw_lines)

    boxes = split_boxes_with_positions(grid, tol_cols=3)

    nodes = []
    for i, box in enumerate(boxes):
        payload = extract_payload_from_box_rows(box, grid)
        kind, payload = parse_kind_payload(payload)
        if not is_semantic_box(payload, kind):
            continue
        info = parse_info(kind, payload)
        label = label_of(kind, info)
        nodes.append({
            "id": len(nodes),
            "kind": kind, "info": info, "label": label,
            "top": box["top_row"], "bot": box["bottom_row"],
            "left": box["left"], "right": box["right"],
            "box": box
        })

    owner_map: Dict[Tuple[int, int], int] = {}
    for n in nodes:
        for rc in mark_box_cells(n["box"], grid):
            owner_map[rc] = n["id"]

    node_tops = {n["id"]: n["top"] for n in nodes}

    forbidden = compute_forbidden_cells([n["box"] for n in nodes], grid)

    edges: Set[Tuple[int, int]] = set()
    for n in nodes:
        anchors = top_anchors(n["box"], grid)
        for a in anchors:
            parent_id = bfs_find_parent(a, grid, owner_map, n["id"], node_tops, forbidden=forbidden)
            if parent_id is not None:
                edges.add((n["id"], parent_id))

    def by_child_depth(e):
        child = nodes[e[0]]
        return (-child["bot"], child["left"])

    edges_sorted = sorted(edges, key=by_child_depth)

    id2label = {n["id"]: n["label"] for n in nodes}
    return [(id2label[c], id2label[p]) for (c, p) in edges_sorted]

def plan_to_ids_and_edges_from_text(plan_text: str, bottom_up: bool = False):
    raw_lines = plan_text.splitlines()
    grid = pad_grid(raw_lines)

    boxes = split_boxes_with_positions(grid, tol_cols=3)

    nodes = []
    for box in boxes:
        payload = extract_payload_from_box_rows(box, grid)
        kind, payload = parse_kind_payload(payload)
        if not is_semantic_box(payload, kind):
            continue
        info = parse_info(kind, payload)
        label = label_of(kind, info)
        nodes.append({
            "id": None,
            "kind": kind, "info": info, "label": label,
            "top": box["top_row"], "bot": box["bottom_row"],
            "left": box["left"], "right": box["right"],
            "box": box
        })

    if bottom_up:
        nodes_sorted = sorted(nodes, key=lambda n: (-n["bot"], n["left"]))[::-1]
    else:
        nodes_sorted = sorted(nodes, key=lambda n: (-n["bot"], n["left"]))

    for new_id, node in enumerate(nodes_sorted):
        node["id"] = new_id
    nodes = nodes_sorted

    owner_map: Dict[Tuple[int, int], int] = {}
    for n in nodes:
        for rc in mark_box_cells(n["box"], grid):
            owner_map[rc] = n["id"]

    node_tops = {n["id"]: n["top"] for n in nodes}

    forbidden = compute_forbidden_cells([n["box"] for n in nodes], grid)

    edges: Set[Tuple[int, int]] = set()
    for n in nodes:
        anchors = top_anchors(n["box"], grid)
        for a in anchors:
            parent_id = bfs_find_parent(a, grid, owner_map, n["id"], node_tops, forbidden=forbidden)
            if parent_id is not None:
                edges.add((n["id"], parent_id))

    def by_child_depth(e):
        child = [n for n in nodes if n["id"] == e[0]][0]
        return (-child["bot"], child["left"])

    edges_sorted = sorted(edges, key=by_child_depth)

    node_labels = [n["label"] for n in sorted(nodes, key=lambda n: n["id"])]
    return node_labels, edges_sorted

def generate_dfg(plan_text, dfg_host_port):
    node_labels, edges_by_id = plan_to_ids_and_edges_from_text(plan_text)

    mapping_lines = []
    for i, label in enumerate(node_labels):
        mapping_lines.append(f"{i}: {label}")

    mapping_txt = "\n".join(mapping_lines)

    edge_lines = []
    for (c, p) in edges_by_id:
        edge_lines.append(f"{c} -> {p}")

    edges_txt = "\n".join(edge_lines)

    # print(mapping_txt)
    # print(edges_txt)

    graph_dot = "graph {{\n{}\n}}".format(edges_txt)
    # print(graph_dot)
    graph_ascii = dot_to_ascii(graph_dot, dfg_host_port)
    DFG = graph_ascii + "\n--------Data Flow Node Mapping--------\n" + mapping_txt
    return DFG

def extract_first_plan(text):
    parts = text.strip().split("\n\n")
    first_plan = parts[0].strip()
    return first_plan

if __name__ == "__main__":
    plan_text = ""
    plan = extract_first_plan(plan_text)
    DFG = generate_dfg(plan, "http://localhost:8080/dot-to-ascii.php")
    print(DFG)
