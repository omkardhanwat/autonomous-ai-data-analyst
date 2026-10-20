# Autonomous AI Data Analyst

A starter multi-agent data investigation app built with Python, LangGraph, Pandas, Streamlit, and optional Ollama.

## Features
- Upload CSV/XLSX datasets
- Dataset profiling and column/type inspection
- Stateful LangGraph workflow: Planner -> Analyst -> Validator -> Reporter
- Deterministic Pandas computations for common questions
- Optional local LLM planning/reporting through Ollama
- Charts and downloadable Markdown reports
- No Docker required

## Requirements
- Python 3.11 or 3.12
- Optional: Ollama installed locally with a model such as `mistral`

## Setup (Windows PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run without an LLM first
```powershell
streamlit run app.py
```
The app works in deterministic mode without Ollama. It answers supported questions using actual Pandas calculations.

## Optional local LLM
Install Ollama from https://ollama.com, then:
```powershell
ollama pull mistral
ollama serve
```
In a second terminal, set:
```powershell
$env:OLLAMA_MODEL="mistral"
```
Then run `streamlit run app.py`. If Ollama is unavailable, the app falls back to deterministic planning/reporting.

## Try these questions
- What are the columns and data types?
- Show summary statistics.
- Which category has the highest sales?
- Show sales by month.
- Are there missing values?
- Show the top 10 rows.
- Which products have the lowest revenue?

The automatic numeric analysis is intentionally conservative. It does not execute arbitrary LLM-generated Python or SQL.

## Project structure
```text
autonomous_ai_data_analyst/
├── app.py
├── requirements.txt
├── .env.example
├── src/
│   ├── __init__.py
│   ├── graph.py
│   ├── tools.py
│   ├── llm.py
│   └── reporting.py
└── sample_data/
    └── sales.csv
```

## Next development milestones
1. Add richer intent-to-operation mapping and robust column selection.
2. Add PostgreSQL read-only SQL tool.
3. Add PDF policy/report RAG with FAISS.
4. Add evaluation dataset and regression tests.
5. Add authentication, file limits, and deployment hardening.
