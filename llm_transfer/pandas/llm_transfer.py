import json
import os
import pandas as pd
import numpy as np
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import datetime
import duckdb
from parse_sql import process_sql_features
from generate_dfg import extract_first_plan, generate_dfg
import argparse
import time

EXCLUDE_KEYS = {"HTML", "Title", "Category", "Feature", "index"}

def init_duckdb_tables(csv_dir, con, include_csv_context):
    csv_context = ""
    tables = {}
    prompt_blocks = []
    for fname in os.listdir(csv_dir):
        if fname.lower().endswith(".csv"):
            table_name = os.path.splitext(fname)[0]
            path = os.path.join(csv_dir, fname)
            tables[table_name] = pd.read_csv(path)
            sql = f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{path}');"
            con.sql(sql)
            df = con.sql(f"SELECT * FROM {table_name}").df()
            df = df.astype("string").fillna("NULL")
            df = df.astype(str).replace({"True": "true", "False": "false"})
            block = f"## Table: `{table_name}`\n" + df.to_markdown(index=False, disable_numparse=True)
            prompt_blocks.append(block)
        if include_csv_context:
            csv_context = "\nTable schema and data information:\n" + "\n\n".join(prompt_blocks) + "\n"
            # print(f"Loaded {table_name} from {path}")
    return tables, csv_context


def exec_sql_statement(original_sql, con):
    try:
        sql_result = con.sql(original_sql).fetchdf()
        con.sql("PRAGMA explain_output = 'all';")
        sql_query_plan = con.sql(original_sql).explain()
        return sql_result, None, sql_query_plan
    except Exception as e:
        print("sql error message: ")
        print(e)
        return None, e, None


def exec_python_statement(transfer_code, tables):
    try:
        local_vars = {name: df for name, df in tables.items()}
        local_vars["pd"] = pd
        local_vars["np"] = np
        exec(transfer_code, local_vars, local_vars)
        transfer_result = local_vars["result"]
        return transfer_result, None
    except Exception as e:
        print("python error message: ")
        print(e)
        return None, e

def _norm(s):
    return (s or "").strip()

def _first_title(doc: dict) -> str:
    for k in ("Title", "Feature"):
        v = doc.get(k)
        if isinstance(v, list) and v:
            return _norm(v[0])
        if isinstance(v, str) and v.strip():
            return _norm(v)
    return ""

def _render_value(val) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, (int, float, bool)):
        return str(val)
    if isinstance(val, list):
        parts = []
        for x in val:
            if not x:
                continue
            if isinstance(x, list):
                head = str(x[0]) if len(x) > 0 else ""
                tail = " ".join(str(t) for t in x[1:]) if len(x) > 1 else ""
                if head or tail:
                    parts.append(f"- {head}: {tail}".rstrip(": "))
            else:
                s = str(x).strip()
                if s:
                    parts.append(s)
        return "\n".join(parts)
    if isinstance(val, dict):
        parts = []
        for k, v in val.items():
            s = _render_value(v)
            if s:
                parts.append(f"- {k}: {s}")
        return "\n".join(parts)
    return str(val)

def _render_doc_block(num: int, title: str, doc: dict) -> str:
    items = []
    for k, v in doc.items():
        if k in EXCLUDE_KEYS:
            continue
        rendered = _render_value(v)
        if rendered:
            items.append((k, rendered))
    if not items:
        return ""

    priority = ["Description", "Examples", "Parameters", "Returns", "Notes"]
    prioritized = [x for x in items if x[0] in priority]
    others = [x for x in items if x[0] not in priority]
    ordered = prioritized + others

    lines = [f"--------Knowledge {num}: {title}--------"]
    for k, v in ordered:
        lines.append(f"{k}:")
        lines.append(v)
    return "\n".join(lines) + "\n"

def _sql_key(name: str) -> str:
    return _norm(name).upper()

def _py_keys(name: str):
    full = _norm(name).lower()
    base = full.split(".")[-1]
    return {full, base}

def generate_feature_prompt(matched: dict, feature_map: list) -> str:
    allowed_sql = set()
    allowed_py_full_or_base = set()
    for mp in feature_map:
        sql_name = _norm(mp.get("SQL"))
        py_name = _norm(mp.get("Python"))
        if sql_name and py_name:
            allowed_sql.add(_sql_key(sql_name))
            allowed_py_full_or_base |= _py_keys(py_name)

    out = []

    # out.append("#Mapping from SQL syntax to Python DataFrame API:")
    # mcount = 0
    # for mp in feature_map:
    #     sql_name = _norm(mp.get("SQL"))
    #     py_name = _norm(mp.get("Python"))
    #     if not sql_name or not py_name:
    #         continue
    #     mcount += 1
    #     out.append(f"{mcount}.{sql_name} -> {py_name}")
    # out.append("")
    #
    out.append("# Knowledge of SQL and Python DataFrame:")

    seen_sql, sql_blocks, sql_num = set(), [], 0
    for group in ("clause", "function"):
        for item in matched.get(group, []):
            sql_doc = item.get("sql_doc") or {}
            title = _first_title(sql_doc)
            if not title:
                continue
            if _sql_key(title) not in allowed_sql:
                continue
            if title.lower() in seen_sql:
                continue
            block = _render_doc_block(sql_num + 1, title, sql_doc)
            if block:
                sql_num += 1
                seen_sql.add(title.lower())
                sql_blocks.append(block)
    if sql_blocks:
        out.append("## SQL:")
        out.extend(sql_blocks)

    seen_py, py_blocks, py_num = set(), [], 0
    for group in ("clause", "function"):
        for item in matched.get(group, []):
            py_doc = item.get("python_doc") or {}
            title = _first_title(py_doc)
            if not title:
                fmap = item.get("python_map") or {}
                title = _norm(fmap.get("Feature"))
            if not title:
                continue
            title_keys = _py_keys(title)
            if allowed_py_full_or_base.isdisjoint(title_keys):
                continue

            sql_title = _first_title(item.get("sql_doc") or {})
            combo_key = f"{(sql_title or '').lower()}->{title.lower()}"

            if combo_key in seen_py:
                continue

            block = _render_doc_block(py_num + 1, title, py_doc)
            if block:
                py_num += 1
                seen_py.add(combo_key)
                py_blocks.append(block)
    if py_blocks:
        out.append("## Python DataFrame:")
        out.extend(py_blocks)

    return "\n".join(out)

def save_prompt(prompt_dir, prompt_content, file_name):
    os.makedirs(prompt_dir, exist_ok=True)
    file_path = os.path.join(prompt_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(prompt_content)
    return file_path

def save_source_data(source_data, source_data_path):
    os.makedirs(os.path.dirname(source_data_path), exist_ok=True)
    with open(source_data_path, "w", encoding="utf-8") as f:
        json.dump(source_data, f, ensure_ascii=False, indent=4)

def load_sqls_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    statements = [stmt.strip() + ';' for stmt in content.split(';') if stmt.strip()]
    return statements

def llm_generate_transferred_sql(sql_statement, origin_db, conversation, iteration_num, tables, con, model, method, prompt_path, source_data_path, dfg_host_port, csv_context="", feature=False, DFG=False, Explanation=True):
    sql_result, sql_error_message, sql_query_plan = exec_sql_statement(sql_statement, con)
    aggregated, matched, feature_map = process_sql_features(sql_statement, origin_db)
    No = 9
    require = ""
    if feature:
        feature_prompt = generate_feature_prompt(matched, feature_map)
        require = require + f"{No}. Utilize the provided knowledge of SQL syntax (clause, function, etc.) and Python DataFrame API to assist in translation work.\n"
        No = No + 1
    else:
        feature_prompt = ""
        aggregated = None
        feature_map = None
    if DFG:
        plan = extract_first_plan(sql_query_plan)
        dfg = "\nData Flow Graph:\n" + generate_dfg(plan, dfg_host_port) + "\n"
        require = require + f"{No}. Based on the Data Flow Graph, utilize its node order and content to help generate DataFrame-style Python code.\n"
    else:
        dfg = ""
        sql_query_plan = None

    source_data = {
        "origin_sql": sql_statement,
        "parse_sql": aggregated,
        "feature_map": feature_map,
        "sql_query_plan": sql_query_plan
    }
    save_source_data(source_data, source_data_path)

    transfer_llm_string = """Let's think step by step. You are an expert in translating SQL-style language into Python DataFrame-style language.
With this expertise, translate the following Duckdb SQL statement into equivalent executable Python code using the Pandas DataFrame API while preserving the original semantics.

# Original Duckdb SQL statement:
{sql_statement}
{csv_context_block}
{data_flow_graph}
{feature}
# Requirements during translation:
1. Ensure that all column names and variables remain the same.
2. Ensure that no additional meaningless operations (e.g., NULL, current_time) are introduced.
3. Ensure that the translated statement maintains the logic and semantics of the original SQL statement.
4. Assume that all referenced SQL tables already exist in the form of DataFrames with their respective names.
5. Output only the translated code, excluding `import` statements or comments.
6. The final answer MUST assign the calculation result to a variable named `result` in a single statement (e.g., `result = <DataFrame expression>`). DO NOT return any unassigned or bare expression and always assign the final DataFrame to `result`.
7. Ensure that the translated Pandas Python code always produces a pandas DataFrame as the output.
8. When calling the Pandas or NumPy libraries, use their aliases `pd` or `np`.
{require}

Answer the following information: {format_instructions}
"""
    if Explanation:
        response_schemas = [
            ResponseSchema(name="TransferCode", description="The transferred DataFrame code"),
            ResponseSchema(name="Explanation", description="Explanation of the conversion")
        ]
    else:
        response_schemas = [
            ResponseSchema(name="TransferCode", description="The transferred DataFrame code")
        ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt_template = ChatPromptTemplate.from_template(transfer_llm_string)

    iterate_llm_string = """
        The corresponding executable DataFrame code that you provided in your most recent response resulted in an error when executed.\
        Please modify your most recent DataFrame code response based on the error message.\
        Ensure that all column names remain unchanged between the DataFrame code before and after the transfer.\
        error message:{error_message}
        Answer the following information: {format_instructions}
        """
    if Explanation:
        iterate_response_schemas = [
            ResponseSchema(type="string", name="TransferCode",
                           description='The new transferred DataFrame code result after modification.'),
            ResponseSchema(type="string", name="Explanation",
                           description='Explain the basis for the conversion and modification.')
        ]
    else:
        iterate_response_schemas = [
            ResponseSchema(type="string", name="TransferCode",
                           description='The new transferred DataFrame code result after modification.')
        ]

    iterate_output_parser = StructuredOutputParser.from_response_schemas(iterate_response_schemas)
    iterate_format_instructions = iterate_output_parser.get_format_instructions()

    iterate_prompt_template = ChatPromptTemplate.from_template(iterate_llm_string)
    python_error_messages = None
    conversation_cnt = 0
    iteration_history = []

    while conversation_cnt < iteration_num:
        if conversation_cnt == 0:
            prompt_messages = prompt_template.format_messages(
                origin_db=origin_db,
                sql_statement=sql_statement,
                data_flow_graph=dfg,
                require=require,
                csv_context_block=csv_context if csv_context else "",
                feature=feature_prompt,
                format_instructions=format_instructions
            )
            save_prompt(prompt_dir, prompt_messages[0].content, prompt_path)
            # return None
        else:
            if python_error_messages is None:
                print("NO error_messages")
                break

            print(python_error_messages)
            prompt_messages = iterate_prompt_template.format_messages(
                error_message=python_error_messages,
                format_instructions=iterate_format_instructions
            )
        # print(prompt_messages[0].content)
        response = conversation.predict(input=prompt_messages[0].content)
        # print(response)
        output_dict = output_parser.parse(response)
        python_statement = output_dict['TransferCode']

        print(f"Now Time : {time.perf_counter()}")

        print(f"Iteration #{conversation_cnt},TransferCode: {python_statement}")
        python_result, python_error_messages = exec_python_statement(python_statement, tables)
        if Explanation:
            explanation = output_dict['Explanation']
        else:
            explanation = None
        iteration_history.append({
            "iteration": conversation_cnt + 1,
            "TransferCode": python_statement,
            "Explanation": explanation,
            "ErrorMessage": str(python_error_messages),
            "ExecutionResult": python_result
        })

        final_output = python_statement
        final_python_result = python_result

        conversation_cnt += 1

        if not python_error_messages:
            break
    return {
        "Duckdb Version": duckdb.__version__,
        "Pandas": pd.__version__,
        "LLM": model,
        "Transfer Method": method,
        "OriginalSQL": sql_statement,
        "SQLExecutionResult": sql_result,
        "IterationHistory": iteration_history,
        "FinalPythonCode": final_output,
        "FinalPythonResult": final_python_result
    }

def default_serializer(obj):
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, Exception):
        return str(obj)
    return str(obj)

def save_transfer_result(result_dict, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False, default=default_serializer)

def main(db, sql, con, output_path, prompt_path, source_data_path, iteration_num, model, reasoning, method, base_url, csv_dir, api_key, tables, csv_context, dfg_host_port, feature=False, DFG=False, Explanation=True):
    chat_kwargs = dict(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0,
        model_kwargs={
            "response_format": {"type": "json_object"}
        },
    )

    if reasoning is not None:
        chat_kwargs["reasoning"] = {"effort": reasoning}

    chat = ChatOpenAI(**chat_kwargs)

    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=chat, memory=memory, verbose=False)
    output = llm_generate_transferred_sql(
        sql,
        origin_db=db,
        conversation=conversation,
        iteration_num=iteration_num,
        tables=tables,
        con=con,
        model=model,
        method=method,
        prompt_path=prompt_path,
        source_data_path=source_data_path,
        dfg_host_port=dfg_host_port,
        csv_context=csv_context,
        feature=feature,
        DFG=DFG,
        Explanation=Explanation
    )
    # if output == None:
    #     return
    save_transfer_result(output, output_path)
    print(f"[Completed] Saved result to {output_path}\n")

def str_to_bool(value):
    if isinstance(value, str):
        if value == 'True':
            return True
        elif value == 'False':
            return False
        else:
            raise ValueError(f"Invalid boolean value: {value}")
    return value

if __name__ == "__main__":

    model = "gpt-5-mini-2025-08-07"
    base_url = "https://api.openai.com/v1"
    api_key = "sk-xxxxxxxx"

    data = "WHERE2"
    parser = argparse.ArgumentParser()
    parser.add_argument("--sql_input_file", default="")
    parser.add_argument("--db", default='duckdb')
    parser.add_argument("--csv_dir", default="")
    parser.add_argument("--include_csv_context", type=str_to_bool, default=False)
    parser.add_argument("--feature", type=str_to_bool, default=False)
    parser.add_argument("--DFG", type=str_to_bool, default=False)
    parser.add_argument("--Explanation", type=str_to_bool, default=False)
    parser.add_argument("--model", default=model)
    parser.add_argument("--reasoning", default="medium")
    parser.add_argument("--base_url", default=base_url)
    parser.add_argument("--api_key", default=api_key)
    parser.add_argument("--method", default="")
    parser.add_argument("--iteration_num", type=int, default=1)
    parser.add_argument("--outer_result_dir", default="")
    parser.add_argument("--dfg_host_port", default='http://localhost:8080/dot-to-ascii.php')
    args = parser.parse_args()

    sql_statements = load_sqls_from_file(args.sql_input_file)
    # con = duckdb.connect(args.con)
    con = duckdb.connect()

    base_result_dir = os.path.join(os.getcwd(), "transfer_result")
    if args.reasoning is not None:
        model_result_dir = f"{args.model}_{args.reasoning}"
    else:
        model_result_dir = f"{args.model}"
    model_result_dir = os.path.join(base_result_dir, model_result_dir)
    outer_result_dir = os.path.join(model_result_dir, args.outer_result_dir)
    inner_result_dir = os.path.join(outer_result_dir, args.method)
    prompt_dir = os.path.join(inner_result_dir, "prompt")
    result_dir = os.path.join(inner_result_dir, "result")
    source_data_dir = os.path.join(inner_result_dir, "source")

    os.makedirs(prompt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(source_data_dir, exist_ok=True)
    tables, csv_context = init_duckdb_tables(args.csv_dir, con, args.include_csv_context)

    global_start_wall = datetime.datetime.now()
    global_start_perf = time.perf_counter()
    print(f"Start Time : {time.perf_counter()}")

    for i, sql in enumerate(sql_statements):
        prompt_path = os.path.join(prompt_dir, f"{i}.txt")
        result_path = os.path.join(result_dir, f"{i}.json")
        source_data_path = os.path.join(source_data_dir, f"{i}.json")
        if os.path.exists(result_path):
            print(f"Skipping SQL #{i}, result already exists: {result_path}")
            continue
        print(f"Processing SQL #{i}: {sql}")
        try:
            main(args.db, sql, con, result_path, prompt_path, source_data_path, args.iteration_num, model, args.reasoning, args.method, base_url, args.csv_dir, args.api_key, tables, csv_context, args.dfg_host_port, feature=args.feature, DFG=args.DFG, Explanation=args.Explanation)
        except Exception as e:
            print(e)
            continue

    con.close()

    global_end_wall = datetime.datetime.now()
    global_end_perf = time.perf_counter()
    global_timing = {
        "start_time": global_start_wall.isoformat(),
        "end_time": global_end_wall.isoformat(),
        "total_duration_seconds": round(global_end_perf - global_start_perf, 6),
        "total_sql_count": len(sql_statements),
    }

    print("\n========== Global Timing Summary ==========")
    print(f"Start Time : {global_timing['start_time']}")
    print(f"End Time   : {global_timing['end_time']}")
    print(f"Total Time : {global_timing['total_duration_seconds']} seconds")
    print("===========================================\n")

    summary_path = os.path.join(inner_result_dir, "timing_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(global_timing, f, indent=4, ensure_ascii=False)
    # if os.path.exists(args.con):
    #     os.remove(args.con)
    #     print(f"Deleted database file: {args.con}")