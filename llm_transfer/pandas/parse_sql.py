import sqlglot
from sqlglot import parse_one, exp
from sqlglot.helper import AutoName
from enum import auto
import json
import re
from typing import Dict, Tuple, Any
import os

aggregate_functions = {
    'arg_max': 'arg_max_groupby',
    'arg_max_null': 'arg_max_null_groupby',
    'arg_min': 'arg_min_groupby',
    'arg_min_null': 'arg_min_null_groupby',
    'avg': 'avg_groupby',
    'count': 'count_groupby',
    'first': 'first_groupby',
    'histogram': 'histogram_groupby',
    'last': 'last_groupby',
    'max': 'max_groupby',
    'min': 'min_groupby',
    'product': 'product_groupby',
    'sum': 'sum_groupby',
    'corr': 'corr_groupby',
    'covar_samp': 'covar_samp_groupby',
    'median': 'median_groupby',
    'regr_r2': 'regr_r2_groupby',
    'skewness': 'skewness_groupby',
}

class TokenType_not_op(AutoName):
    L_PAREN = auto()
    R_PAREN = auto()
    L_BRACKET = auto()
    R_BRACKET = auto()
    L_BRACE = auto()
    R_BRACE = auto()
    COMMA = auto()
    DOT = auto()
    COLON = auto()
    DQMARK = auto()
    SEMICOLON = auto()
    BACKSLASH = auto()
    SLASH = auto()
    NULLSAFE_EQ = auto()
    COLON_EQ = auto()
    PIPE = auto()
    PIPE_SLASH = auto()
    DPIPE_SLASH = auto()
    ARROW = auto()
    DARROW = auto()
    FARROW = auto()
    HASH = auto()
    HASH_ARROW = auto()
    DHASH_ARROW = auto()
    LR_ARROW = auto()
    DAT = auto()
    LT_AT = auto()
    AT_GT = auto()
    DOLLAR = auto()
    PARAMETER = auto()
    SESSION_PARAMETER = auto()
    DAMP = auto()
    XOR = auto()
    DSTAR = auto()

    BLOCK_START = auto()
    BLOCK_END = auto()

    SPACE = auto()
    BREAK = auto()

    STRING = auto()
    NUMBER = auto()
    IDENTIFIER = auto()
    DATABASE = auto()
    COLUMN = auto()
    COLUMN_DEF = auto()
    SCHEMA = auto()
    TABLE = auto()
    WAREHOUSE = auto()
    STREAMLIT = auto()
    VAR = auto()
    BIT_STRING = auto()
    HEX_STRING = auto()
    BYTE_STRING = auto()
    NATIONAL_STRING = auto()
    RAW_STRING = auto()
    HEREDOC_STRING = auto()
    UNICODE_STRING = auto()

    # types
    BIT = auto()
    BOOLEAN = auto()
    TINYINT = auto()
    UTINYINT = auto()
    SMALLINT = auto()
    USMALLINT = auto()
    MEDIUMINT = auto()
    UMEDIUMINT = auto()
    INT = auto()
    UINT = auto()
    BIGINT = auto()
    UBIGINT = auto()
    INT128 = auto()
    UINT128 = auto()
    INT256 = auto()
    UINT256 = auto()
    FLOAT = auto()
    DOUBLE = auto()
    DECIMAL = auto()
    DECIMAL32 = auto()
    DECIMAL64 = auto()
    DECIMAL128 = auto()
    UDECIMAL = auto()
    BIGDECIMAL = auto()
    CHAR = auto()
    NCHAR = auto()
    VARCHAR = auto()
    NVARCHAR = auto()
    BPCHAR = auto()
    TEXT = auto()
    MEDIUMTEXT = auto()
    LONGTEXT = auto()
    MEDIUMBLOB = auto()
    LONGBLOB = auto()
    TINYBLOB = auto()
    TINYTEXT = auto()
    NAME = auto()
    BINARY = auto()
    VARBINARY = auto()
    JSON = auto()
    JSONB = auto()
    TIME = auto()
    TIMETZ = auto()
    TIMESTAMP = auto()
    TIMESTAMPTZ = auto()
    TIMESTAMPLTZ = auto()
    TIMESTAMPNTZ = auto()
    TIMESTAMP_S = auto()
    TIMESTAMP_MS = auto()
    TIMESTAMP_NS = auto()
    DATETIME = auto()
    DATETIME64 = auto()
    DATE = auto()
    DATE32 = auto()
    INT4RANGE = auto()
    INT4MULTIRANGE = auto()
    INT8RANGE = auto()
    INT8MULTIRANGE = auto()
    NUMRANGE = auto()
    NUMMULTIRANGE = auto()
    TSRANGE = auto()
    TSMULTIRANGE = auto()
    TSTZRANGE = auto()
    TSTZMULTIRANGE = auto()
    DATERANGE = auto()
    DATEMULTIRANGE = auto()
    UUID = auto()
    GEOGRAPHY = auto()
    NULLABLE = auto()
    GEOMETRY = auto()
    HLLSKETCH = auto()
    HSTORE = auto()
    SUPER = auto()
    SERIAL = auto()
    SMALLSERIAL = auto()
    BIGSERIAL = auto()
    XML = auto()
    YEAR = auto()
    UNIQUEIDENTIFIER = auto()
    USERDEFINED = auto()
    MONEY = auto()
    SMALLMONEY = auto()
    ROWVERSION = auto()
    IMAGE = auto()
    VARIANT = auto()
    OBJECT = auto()
    INET = auto()
    IPADDRESS = auto()
    IPPREFIX = auto()
    IPV4 = auto()
    IPV6 = auto()
    ENUM = auto()
    ENUM8 = auto()
    ENUM16 = auto()
    FIXEDSTRING = auto()
    LOWCARDINALITY = auto()
    NESTED = auto()
    AGGREGATEFUNCTION = auto()
    SIMPLEAGGREGATEFUNCTION = auto()
    TDIGEST = auto()
    UNKNOWN = auto()
    VECTOR = auto()
    NULL = auto()

    # keywords
    ALIAS = auto()
    ALTER = auto()
    ALWAYS = auto()
    ALL = auto()
    ANTI = auto()
    ANY = auto()
    APPLY = auto()
    ARRAY = auto()
    ASC = auto()
    ASOF = auto()
    AUTO_INCREMENT = auto()
    BEGIN = auto()
    CACHE = auto()
    CASE = auto()
    CHARACTER_SET = auto()
    CLUSTER_BY = auto()
    COLLATE = auto()
    COMMAND = auto()
    COMMENT = auto()
    COMMIT = auto()
    CONNECT_BY = auto()
    CONSTRAINT = auto()
    COPY = auto()
    CREATE = auto()
    CROSS = auto()
    CUBE = auto()
    CURRENT_DATE = auto()
    CURRENT_DATETIME = auto()
    CURRENT_TIME = auto()
    CURRENT_TIMESTAMP = auto()
    CURRENT_USER = auto()
    DECLARE = auto()
    DEFAULT = auto()
    DELETE = auto()
    DESC = auto()
    DESCRIBE = auto()
    DICTIONARY = auto()
    DISTINCT = auto()
    DISTRIBUTE_BY = auto()
    DROP = auto()
    ELSE = auto()
    END = auto()
    ESCAPE = auto()
    EXCEPT = auto()
    EXECUTE = auto()
    EXISTS = auto()
    FALSE = auto()
    FETCH = auto()
    FILTER = auto()
    FINAL = auto()
    FIRST = auto()
    FOR = auto()
    FORCE = auto()
    FOREIGN_KEY = auto()
    FORMAT = auto()
    FROM = auto()
    FULL = auto()
    FUNCTION = auto()
    GLOB = auto()
    GLOBAL = auto()
    GROUP_BY = auto()
    GROUPING_SETS = auto()
    HAVING = auto()
    HINT = auto()
    IGNORE = auto()
    INDEX = auto()
    INNER = auto()
    INSERT = auto()
    INTERSECT = auto()
    INTERVAL = auto()
    INTO = auto()
    INTRODUCER = auto()
    JOIN = auto()
    JOIN_MARKER = auto()
    KEEP = auto()
    KEY = auto()
    KILL = auto()
    LANGUAGE = auto()
    LATERAL = auto()
    LEFT = auto()
    LIMIT = auto()
    LIST = auto()
    LOAD = auto()
    LOCK = auto()
    MAP = auto()
    MATCH_CONDITION = auto()
    MATCH_RECOGNIZE = auto()
    MEMBER_OF = auto()
    MERGE = auto()
    MODEL = auto()
    NATURAL = auto()
    NEXT = auto()
    OBJECT_IDENTIFIER = auto()
    OFFSET = auto()
    ON = auto()
    ONLY = auto()
    OPERATOR = auto()
    ORDER_BY = auto()
    ORDER_SIBLINGS_BY = auto()
    ORDERED = auto()
    ORDINALITY = auto()
    OUTER = auto()
    OVER = auto()
    OVERLAPS = auto()
    OVERWRITE = auto()
    PARTITION = auto()
    PARTITION_BY = auto()
    PERCENT = auto()
    PIVOT = auto()
    PLACEHOLDER = auto()
    POSITIONAL = auto()
    PRAGMA = auto()
    PREWHERE = auto()
    PRIMARY_KEY = auto()
    PROCEDURE = auto()
    PROPERTIES = auto()
    PSEUDO_TYPE = auto()
    QUALIFY = auto()
    QUOTE = auto()
    RANGE = auto()
    RECURSIVE = auto()
    REFRESH = auto()
    RENAME = auto()
    REPLACE = auto()
    RETURNING = auto()
    REFERENCES = auto()
    RIGHT = auto()
    ROLLBACK = auto()
    ROLLUP = auto()
    ROW = auto()
    ROWS = auto()
    SELECT = auto()
    SEMI = auto()
    SEPARATOR = auto()
    SEQUENCE = auto()
    SERDE_PROPERTIES = auto()
    SET = auto()
    SETTINGS = auto()
    SHOW = auto()
    SOME = auto()
    SORT_BY = auto()
    START_WITH = auto()
    STORAGE_INTEGRATION = auto()
    STRAIGHT_JOIN = auto()
    STRUCT = auto()
    SUMMARIZE = auto()
    TABLE_SAMPLE = auto()
    TAG = auto()
    TEMPORARY = auto()
    TOP = auto()
    THEN = auto()
    TRUE = auto()
    TRUNCATE = auto()
    UNCACHE = auto()
    UNION = auto()
    UNNEST = auto()
    UNPIVOT = auto()
    UPDATE = auto()
    USE = auto()
    USING = auto()
    VALUES = auto()
    VIEW = auto()
    VOLATILE = auto()
    WHEN = auto()
    WHERE = auto()
    WINDOW = auto()
    WITH = auto()
    UNIQUE = auto()
    VERSION_SNAPSHOT = auto()
    TIMESTAMP_SNAPSHOT = auto()
    OPTION = auto()

class TokenType_clause(AutoName):
    DISTINCT = auto()
    GROUP_BY = auto()
    GROUPING_SETS = auto()
    HAVING = auto()
    JOIN = auto()
    LIMIT = auto()
    OFFSET = auto()
    ORDER_BY = auto()
    QUALIFY = auto()
    SELECT = auto()
    UNION = auto()
    WHERE = auto()

class TokenType_datatype(AutoName):
    BIT = auto()
    BOOLEAN = auto()
    TINYINT = auto()
    UTINYINT = auto()
    SMALLINT = auto()
    USMALLINT = auto()
    MEDIUMINT = auto()
    UMEDIUMINT = auto()
    INT = auto()
    UINT = auto()
    BIGINT = auto()
    UBIGINT = auto()
    INT128 = auto()
    UINT128 = auto()
    INT256 = auto()
    UINT256 = auto()
    FLOAT = auto()
    DOUBLE = auto()
    DECIMAL = auto()
    DECIMAL32 = auto()
    DECIMAL64 = auto()
    DECIMAL128 = auto()
    UDECIMAL = auto()
    BIGDECIMAL = auto()
    CHAR = auto()
    NCHAR = auto()
    VARCHAR = auto()
    NVARCHAR = auto()
    BPCHAR = auto()
    TEXT = auto()
    MEDIUMTEXT = auto()
    LONGTEXT = auto()
    MEDIUMBLOB = auto()
    LONGBLOB = auto()
    TINYBLOB = auto()
    TINYTEXT = auto()
    NAME = auto()
    BINARY = auto()
    VARBINARY = auto()
    JSON = auto()
    JSONB = auto()
    TIME = auto()
    TIMETZ = auto()
    TIMESTAMP = auto()
    TIMESTAMPTZ = auto()
    TIMESTAMPLTZ = auto()
    TIMESTAMPNTZ = auto()
    TIMESTAMP_S = auto()
    TIMESTAMP_MS = auto()
    TIMESTAMP_NS = auto()
    DATETIME = auto()
    DATETIME64 = auto()
    DATE = auto()
    DATE32 = auto()
    INT4RANGE = auto()
    INT4MULTIRANGE = auto()
    INT8RANGE = auto()
    INT8MULTIRANGE = auto()
    NUMRANGE = auto()
    NUMMULTIRANGE = auto()
    TSRANGE = auto()
    TSMULTIRANGE = auto()
    TSTZRANGE = auto()
    TSTZMULTIRANGE = auto()
    DATERANGE = auto()
    DATEMULTIRANGE = auto()
    UUID = auto()
    GEOGRAPHY = auto()
    NULLABLE = auto()
    GEOMETRY = auto()
    HLLSKETCH = auto()
    HSTORE = auto()
    SUPER = auto()
    SERIAL = auto()
    SMALLSERIAL = auto()
    BIGSERIAL = auto()
    XML = auto()
    YEAR = auto()
    UNIQUEIDENTIFIER = auto()
    USERDEFINED = auto()
    MONEY = auto()
    SMALLMONEY = auto()
    ROWVERSION = auto()
    IMAGE = auto()
    VARIANT = auto()
    OBJECT = auto()
    INET = auto()
    IPADDRESS = auto()
    IPPREFIX = auto()
    IPV4 = auto()
    IPV6 = auto()
    ENUM = auto()
    ENUM8 = auto()
    ENUM16 = auto()
    FIXEDSTRING = auto()
    LOWCARDINALITY = auto()
    NESTED = auto()
    AGGREGATEFUNCTION = auto()
    SIMPLEAGGREGATEFUNCTION = auto()
    TDIGEST = auto()
    UNKNOWN = auto()
    VECTOR = auto()
    NULL = auto()

def is_datatype_name(name):
    return name in TokenType_datatype.__members__

def is_clause_name(name):
    return name in TokenType_clause.__members__

def is_member_name(name):
    return name in TokenType_not_op.__members__

def parse_feature(tokens, sql_text: str):
    result = {
        "sql": sql_text,
        "datatype": set(),
        "clause": set(),
        "operator": set(),
        "function": set(),
    }

    val_indexes = []

    for index, tok in enumerate(tokens):
        if is_datatype_name(str(tok.token_type).split(".")[-1]):
            result["datatype"].add(tok.text)

    for index, tok in enumerate(tokens):
        if is_clause_name(str(tok.token_type).split(".")[-1]):
            result["clause"].add(tok.text)

    for index, tok in enumerate(tokens):
        if tok.token_type in [
            sqlglot.TokenType.VAR,
            sqlglot.TokenType.LEFT,
            sqlglot.TokenType.TIMESTAMP,
        ]:
            val_indexes.append(index)

        if not is_member_name(str(tok.token_type).split(".")[-1]):
            if (
                tok.token_type == sqlglot.TokenType.IS
                and index + 1 < len(tokens)
                and tokens[index + 1].token_type == sqlglot.TokenType.NULL
            ):
                result["operator"].add(tok.text + tokens[index + 1].text)
            else:
                result["operator"].add(tok.text)

    for VAL_index in val_indexes:
        if (
            VAL_index + 1 < len(tokens)
            and tokens[VAL_index + 1].token_type == sqlglot.TokenType.L_PAREN
            and tokens[VAL_index - 1].token_type != sqlglot.TokenType.UNKNOWN
            and tokens[VAL_index - 1].text.lower() != "table"
        ):
            result["function"].add(tokens[VAL_index].text)

    return {
        "sql": result["sql"],
        "datatype": sorted(result["datatype"]),
        "clause": sorted(result["clause"]),
        "operator": sorted(result["operator"]),
        "function": sorted(result["function"]),
    }

def aggregate_features(features_dict):
    all_clauses = set()
    all_funcs = set()

    for v in features_dict.values():
        all_clauses.update(v.get("clause", []))
        all_funcs.update(v.get("datatype", []))
        all_funcs.update(v.get("operator", []))
        all_funcs.update(v.get("function", []))

    return {
        "clause": sorted(all_clauses),
        "function": sorted(all_funcs),
    }

def _normalize_feature(name: str) -> str:
    if name is None:
        return ""
    s = str(name).strip()
    s = re.sub(r"\(.*$", "", s)  # 'substr(a,b,c)' -> 'substr'
    return s.strip().upper()

def build_normalized_lookup(raw_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, list]]:
    out: Dict[str, Any] = {}
    conflicts: Dict[str, list] = {}

    for k, v in (raw_dict or {}).items():
        nk = _normalize_feature(k)
        if nk in out and out[nk] is not v:
            conflicts.setdefault(nk, []).append({"original_key": k})
            continue
        out[nk] = v

    return out, conflicts
def get_norm(d_norm: Dict[str, Any], key_like: str):
    return d_norm.get(_normalize_feature(key_like))

def match_features(aggregated, clause_dict, func_dict, clause_map, func_map, expression_dict, node_dicts):
    matched = {"clause": [], "function": []}

    func_dict_norm, func_dict_conflicts = build_normalized_lookup(func_dict)
    func_map_norm, func_map_conflicts = build_normalized_lookup(func_map)

    for c in aggregated.get("clause", []):
        key = str(c).strip().upper()
        sql_obj = clause_dict.get(key)
        map_obj = clause_map.get(key)

        if sql_obj and map_obj:
            py_feature = map_obj.get("python_feature")
            py_doc_key = _normalize_feature(py_feature) if py_feature else ""
            py_doc = node_dicts.get(py_doc_key) if py_doc_key else None

            matched["clause"].append({
                "sql_doc": sql_obj,
                "python_map": {
                    "index": map_obj.get("python_index"),
                    "Feature": py_feature
                },
                "python_doc": py_doc
            })

    for f in aggregated.get("function", []):
        sql_obj = get_norm(func_dict_norm, f)
        map_obj = get_norm(func_map_norm, f)

        if sql_obj and map_obj:
            py_feature = map_obj.get("python_feature")
            py_doc_key = _normalize_feature(py_feature) if py_feature else ""
            py_doc = expression_dict.get(py_doc_key) if py_doc_key else None

            matched["function"].append({
                "sql_doc": sql_obj,
                "python_map": {
                    "index": map_obj.get("python_index"),
                    "Feature": py_feature
                },
                "python_doc": py_doc,
                "original_feature": f
            })

    return matched

def load_jsonl_by_feature(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                feature_name = obj.get("Feature", [None])[0]
                if feature_name:
                    data[feature_name.upper()] = obj
    return data

def load_function_map(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)

                sql_info = obj.get("sql", {})
                py_info = obj.get("python", {})

                sql_feature = sql_info.get("Feature", [None])[0]
                if sql_feature:
                    data[sql_feature.upper()] = {
                        "sql_feature": sql_feature,
                        "python_index": py_info.get("index"),
                        "python_feature": py_info.get("Feature", [None])[0]
                    }
    return data

def extract_features(matched: dict):
    results = []

    if "clause" in matched:
        for item in matched["clause"]:
            sql_feature = item.get("sql_doc", {}).get("Feature", [None])[0]
            py_feature = item.get("python_map", {}).get("Feature", None)
            results.append({"SQL": sql_feature, "Python": py_feature})

    if "function" in matched:
        for item in matched["function"]:
            sql_feature = item.get("sql_doc", {}).get("Feature", [None])[0]
            py_feature = item.get("python_map", {}).get("Feature", None)
            results.append({"SQL": sql_feature, "Python": py_feature})

    return results

def apply_groupby_mapping(aggregated, aggregate_functions):
    clauses = aggregated.get("clause", [])
    functions = aggregated.get("function", [])

    has_groupby = any(c.lower() == "group by" for c in clauses)

    if not has_groupby:
        return aggregated

    mapped_funcs = []
    for func in functions:
        key = func.lower()
        if key in aggregate_functions:
            mapped_funcs.append(aggregate_functions[key])
        else:
            mapped_funcs.append(func)
    aggregated["function"] = mapped_funcs
    return aggregated


def process_sql_features(sql: str, db: str):
    # if db == "polars":
    #     db = "duckdb"

    tokens = sqlglot.tokenize(sql)
    # print(tokens)

    features = {"__ALL__": parse_feature(tokens, sql)}
    # print(features)

    aggregated = aggregate_features(features)
    aggregated = apply_groupby_mapping(aggregated, aggregate_functions)
    # print(aggregated)

    clause_dict = load_jsonl_by_feature("feature_mapping/sql/clause.jsonl")
    func_dict = load_jsonl_by_feature("feature_mapping/sql/function.jsonl")

    expression_dict = load_jsonl_by_feature("feature_mapping/python/expression.jsonl")
    node_dicts = load_jsonl_by_feature("feature_mapping/python/node.jsonl")

    clause_map = load_function_map("./feature_mapping/duckdb_pandas_clause_map.jsonl")
    func_map = load_function_map("./feature_mapping/duckdb_pandas_function_map.jsonl")

    matched = match_features(
        aggregated,
        clause_dict,
        func_dict,
        clause_map,
        func_map,
        expression_dict,
        node_dicts
    )

    feature_map = extract_features(matched)
    return aggregated, matched, feature_map


def load_sqls_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    statements = [stmt.strip() + ';' for stmt in content.split(';') if stmt.strip()]
    return statements

# from your_module import process_sql_features

def process_sql_file(file_path, out_dir="results", dialect="duckdb"):
    os.makedirs(out_dir, exist_ok=True)

    sqls = load_sqls_from_file(file_path)

    for idx, sql in enumerate(sqls):
        record = {
            "file": os.path.basename(file_path),
            "stmt_index": idx,
            "sql": sql,
            "aggregated": None,
            "feature_map": None,
            "error": None
        }
        try:
            aggregated, matched, feature_map = process_sql_features(sql, dialect)
            record["aggregated"] = aggregated
            record["feature_map"] = feature_map
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {e}"

        out_path = os.path.join(out_dir, f"{os.path.basename(file_path)}_{idx}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sql = """SELECT DISTINCT t3.c0, t0.c0, t1.c0 FROM t3, t0, t2, t1 WHERE -6.94242785E8 GROUP BY t3.c0, t0.c0, t1.c0 UNION SELECT t3.c0, t0.c0, t1.c0 FROM t3, t0, t2, t1 WHERE (NOT -6.94242785E8) GROUP BY t3.c0, t0.c0, t1.c0 UNION SELECT t3.c0, t0.c0, t1.c0 FROM t3, t0, t2, t1 WHERE ((-6.94242785E8) IS NULL) GROUP BY t3.c0, t0.c0, t1.c0;"""
    aggregated, matched, feature_map = process_sql_features(sql, "duckdb")
    print(aggregated)
    print(matched)
    print(feature_map)