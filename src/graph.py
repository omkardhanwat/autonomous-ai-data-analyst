from typing import TypedDict, Any
import json
import pandas as pd

from langgraph.graph import StateGraph, END
from .tools import profile_dataframe, route_question, run_tool
from .llm import ask_local_llm


class AnalystState(TypedDict, total=False):
    df: Any
    question: str
    profile: dict
    plan: str
    selected_tool: str
    result: dict
    validation: str
    answer: str
    retry_count: int


def planner_node(state: AnalystState) -> dict:
    df = state["df"]
    question = state["question"]

    profile = profile_dataframe(df)
    selected_tool = route_question(question)

    prompt = f"""
You are a data analyst planning agent.
Create a short analysis plan. Do not write or execute code.
Use only columns present in the schema.

Question: {question}
Schema: {json.dumps(profile, default=str)}
Suggested tool: {selected_tool}
"""

    plan = ask_local_llm(prompt) or (
        f"Inspect the schema, use the {selected_tool} tool, "
        "validate the computed result, and report the findings."
    )

    return {
        "profile": profile,
        "plan": plan,
        "selected_tool": selected_tool,
    }


def analysis_node(state: AnalystState) -> dict:
    result = run_tool(
        state["df"],
        state["question"],
        state["selected_tool"],
    )
    return {"result": result}


def validator_node(state: AnalystState) -> dict:
    result = state["result"]
    data = result.get("data")

    if result.get("kind") == "unsupported":
        validation = "Required columns could not be identified."
    elif isinstance(data, pd.DataFrame):
        validation = (
            f"Result contains {len(data)} rows "
            f"and {len(data.columns)} columns."
        )
        if data.empty:
            validation += " Warning: result is empty."
    elif isinstance(data, dict):
        validation = "Structured dataset profile returned."
    elif data is None:
        validation = "No tabular result was produced."
    else:
        validation = f"Result type: {type(data).__name__}."

    validation += " No generated code was executed."

    return {"validation": validation}


def retry_router(state: AnalystState) -> str:
    result = state.get("result", {})

    if (
        result.get("kind") == "unsupported"
        and state.get("retry_count", 0) < 1
    ):
        return "retry"

    return "report"


def retry_node(state: AnalystState) -> dict:
    return {
        "selected_tool": "profile",
        "retry_count": state.get("retry_count", 0) + 1,
        "plan": state.get("plan", "")
        + "\nFallback: required columns were unclear; "
          "the workflow will return a dataset profile.",
    }


def reporter_node(state: AnalystState) -> dict:
    result = state["result"]
    data = result.get("data")

    if isinstance(data, pd.DataFrame):
        data_text = data.head(15).to_string(index=False)
    elif isinstance(data, dict):
        data_text = json.dumps(data, indent=2, default=str)
    else:
        data_text = str(data)

    prompt = f"""
Answer the user's question concisely.
Use only the computed output. Do not invent numbers.
Separate observations from hypotheses.

Question: {state["question"]}
Tool output: {result.get("text", "")}
Computed data:
{data_text}
Validation: {state.get("validation", "")}
"""

    answer = ask_local_llm(prompt) or (
        f"**{result.get('title', 'Analysis')}**\n\n"
        f"{result.get('text', '')}\n\n"
        f"**Validation:** {state.get('validation', '')}\n\n"
        "These results describe the uploaded data and do not "
        "establish causation."
    )

    return {"answer": answer}


def build_graph():
    workflow = StateGraph(AnalystState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("retry", retry_node)
    workflow.add_node("reporter", reporter_node)

    workflow.set_entry_point("planner")

    # The selected tool is stored in state and used by the analyst.
    workflow.add_conditional_edges(
        "planner",
        lambda state: state["selected_tool"],
        {
            "profile": "analysis",
            "missing": "analysis",
            "preview": "analysis",
            "summary": "analysis",
            "correlation": "analysis",
            "time_series": "analysis",
            "ranking": "analysis",
            "general": "analysis",
        },
    )

    workflow.add_edge("analysis", "validator")

    workflow.add_conditional_edges(
        "validator",
        retry_router,
        {
            "retry": "retry",
            "report": "reporter",
        },
    )

    workflow.add_edge("retry", "analysis")
    workflow.add_edge("reporter", END)

    return workflow.compile()


ANALYST_GRAPH = build_graph()


def run_analysis(df: pd.DataFrame, question: str) -> dict:
    return ANALYST_GRAPH.invoke({
        "df": df,
        "question": question,
        "retry_count": 0,
    })
