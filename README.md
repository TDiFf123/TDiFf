# TDiFf: Detecting Bugs in DataFrame Systems via Transferred DBMS Test Cases

## 1. Experimental Environment
Before this experiment, it is required to install the environments for each of the three DataFrame categories (Pandas-compatible, PySpark, and Polars). 
Their installation dependencies are provided in the respective `environment/` folders.

##### Example Usage
```shell
pip install -r environment/pandas/requirements.txt
```

## 2. Experimental Steps

### (1) LLM Transfer
TDiFf utilizes the provided `llm_transfer.py` script to transfer SQL query statements into DataFrame test cases. Its detailed configuration options are as follows:

##### Supported Options
| Option | Description |
|----------|----------|
| `--sql_input_file` | Path to the SQL query statement file for transfer.  | 
| `--csv_dir` | Path to the CSV table corresponding to the SQL query statement file for transfer. | 
| `--include_csv_context` | Enable CSV table content (default: True). | 
| `--outer_result_dir` | Path to save transfer results. | 
| `--feature` | Enable **Feature Knowledge** (default: True). | 
| `--DFG` | Enable **Data Flow Graph** (default: True). | 
| `--model` | LLM Model used for transfer (default: gpt-5-mini-2025-08-07). | 
| `--reasoning` | Reasoning effort parameter for LLM (default: medium). | 
| `--temperature` | Temperature parameter for LLM (default: 0). | 
| `--base_url` | Base URL path for LLM API (default: https://api.openai.com/v1). | 
| `--api_key` | API Key for LLM API. | 
| `--dfg_host_port` | Path to call the dot-to-ascii process. | 


##### Example Usage
```shell
cd llm_transfer
python pandas/llm_transfer.py \ 
    --sql_input_file "../data/test/duckdb.sql" \ 
    --csv_dir "../data/test/table" \
    --include_csv_context True \
    --outer_result_dir "pandas/result" \
    --feature True \
    --DFG True \
    --model "gpt-5-mini-2025-08-07" \
    --reasoning "medium" \
    --temperature 0 \
    --base_url "https://api.openai.com/v1" \
    --api_key "sk-xxxxxxxx" \
    --dfg_host_port "https://dot-to-ascii.ggerganov.com/dot-to-ascii.php"
```



### (2) Differential Testing
TDiFf leverages the provided `differential_testing.py` script to deploy differential testing using generated DataFrame test cases. Its detailed configuration options are as follows:

##### Supported Options

| Option | Description |
|----------|----------|
| `--json_dir` | Path to generated DataFrame test cases. | 
| `--csv_dir` | Path to the CSV file corresponding to the generated DataFrame test case. | 
| `--output_dir` | Path to save differential testing results. | 

##### Example Usage
```shell
cd testing
python pandas/differential_testing.py \
    --json_dir "../llm_transfer/pandas/result" \
    --csv_dir "../data/test/table" \
    --output_dir "pandas/diff_testing_result"
```


### (3) Bug Summury
After conducting differential testing, use the provided `report.py` script to compare and summarize results, facilitating duplicate removal and reporting of bugs.

##### Example Usage
```shell
cd testing
python pandas/report.py \
    --dir "pandas/diff_testing_result" \
    --out "pandas/"
```

## 3. Tools

The `tools/` folder contains crawler tools for building feature knowledge bases (`tools/feature_knowledge_crawler`) and scripts for extracting the feature mapping table between SQL features and DataFrame APIs (`tools/rag_feature_mapping/rag_based_feature_mapping.py`).