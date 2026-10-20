import io
import os
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from src.graph import run_analysis
from src.reporting import make_markdown_report

load_dotenv()

st.set_page_config(page_title="Autonomous AI Data Analyst", page_icon="📊", layout="wide")
st.title("Autonomous AI Data Analyst")
st.caption("Multi-agent data investigation using LangGraph, Pandas, and optional local Ollama.")

with st.sidebar:
    st.header("How it works")
    st.markdown("1. **Planner** chooses an investigation approach\n2. **Analyst** runs a predefined Pandas tool\n3. **Validator** checks the result structure\n4. **Reporter** explains the result")
    model = os.getenv("OLLAMA_MODEL", "").strip()
    st.write(f"Local LLM: **{model if model else 'Disabled (deterministic mode)'}**")
    st.info("No Docker required. Uploaded files are processed in this app session.")

uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
if uploaded:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as exc:
        st.error(f"Could not read the file: {exc}")
        st.stop()

    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    with st.expander("Preview dataset", expanded=True):
        st.dataframe(df.head(20), use_container_width=True)
    with st.expander("Column types and missing values"):
        metadata = pd.DataFrame({
            "column": df.columns.astype(str),
            "dtype": [str(x) for x in df.dtypes],
            "missing": [int(x) for x in df.isna().sum()],
            "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns],
        })
        st.dataframe(metadata, use_container_width=True)

    st.subheader("Ask a question")
    question = st.text_area(
        "What do you want to investigate?",
        placeholder="Example: Which product has the highest revenue?",
        value="Which category has the highest sales?",
        height=90,
    )
    if st.button("Run investigation", type="primary", disabled=not question.strip()):
        with st.spinner("Agents are investigating the dataset..."):
            try:
                output = run_analysis(df, question.strip())
                st.session_state["analysis_output"] = output
                st.session_state["analysis_question"] = question.strip()
            except Exception as exc:
                st.exception(exc)

    output = st.session_state.get("analysis_output")
    if output and st.session_state.get("analysis_question"):
        st.divider()
        st.subheader("Investigation results")
        st.markdown(output.get("answer", "No answer generated."))
        with st.expander("Agent plan"):
            st.write(output.get("plan", "No plan available."))
        with st.expander("Validation checks"):
            st.write(output.get("validation", "No validation information."))

    result = output.get("result", {})
    data = result.get("data")
    
    if result.get("kind") == "heatmap" and isinstance(data, pd.DataFrame):
        st.plotly_chart(
            px.imshow(
                data,
                text_auto=True,
                aspect="auto",
                title=result.get("title"),
            ),
            use_container_width=True,
        )
    
    elif isinstance(data, pd.DataFrame):
        st.dataframe(data, use_container_width=True)
    
        if result.get("kind") == "bar":
            st.plotly_chart(
                px.bar(data, x=result["x"], y=result["y"]),
                use_container_width=True,
            )
    
        elif result.get("kind") == "line":
            st.plotly_chart(
                px.line(
                    data,
                    x=result["x"],
                    y=result["y"],
                    markers=True,
                ),
                use_container_width=True,
            )
    
    elif isinstance(data, dict):
        st.json(data)

        report = make_markdown_report(
            st.session_state["analysis_question"],
            output.get("answer", ""),
            output.get("plan", ""),
            output.get("validation", ""),
        )
        st.download_button("Download Markdown report", report.encode("utf-8"), "analysis_report.md", "text/markdown")
else:
    st.markdown("### Start here")
    st.write("Upload a dataset to begin. You can use the included sample dataset:")
    sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "sales.csv")
    if os.path.exists(sample_path):
        sample = pd.read_csv(sample_path)
        st.dataframe(sample.head(8), use_container_width=True)
        st.caption("Upload this file from the sample_data folder to try the app.")
