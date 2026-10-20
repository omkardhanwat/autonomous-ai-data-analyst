from __future__ import annotations

import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(map(str, df.columns)),
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "missing": {
            str(c): int(n) for c, n in df.isna().sum().items()
        },
        "numeric_columns": list(
            map(str, df.select_dtypes(include="number").columns)
        ),
    }


def route_question(question: str) -> str:
    q = question.lower()

    if any(x in q for x in ["missing", "null", "empty value"]):
        return "missing"
    if any(x in q for x in ["correlation", "correlated"]):
        return "correlation"
    if any(x in q for x in ["by month", "monthly", "over time", "trend"]):
        return "time_series"
    if any(x in q for x in ["summary statistics", "describe", "distribution"]):
        return "summary"
    if any(x in q for x in ["preview", "first rows", "show rows"]):
        return "preview"
    if any(x in q for x in ["highest", "lowest", "top", "bottom", "most", "least"]):
        return "ranking"
    if any(x in q for x in ["column", "schema", "data type", "profile"]):
        return "profile"

    return "general"


def find_column(df, hints, numeric_only=False):
    candidates = [
        c for c in df.columns
        if not numeric_only or pd.api.types.is_numeric_dtype(df[c])
    ]

    for hint in hints:
        for col in candidates:
            if hint in str(col).lower():
                return col

    return None


def run_tool(df, question, tool_name):
    q = question.lower()

    if tool_name == "profile":
        return {
            "kind": "profile",
            "title": "Dataset profile",
            "text": f"Dataset contains {len(df)} rows and {len(df.columns)} columns.",
            "data": profile_dataframe(df),
        }

    if tool_name == "missing":
        result = pd.DataFrame({
            "column": df.columns.astype(str),
            "missing_values": df.isna().sum().values,
        })
        result = result[result["missing_values"] > 0]

        return {
            "kind": "table",
            "title": "Missing values",
            "text": f"Found {len(result)} columns containing missing values.",
            "data": result,
        }

    if tool_name == "preview":
        return {
            "kind": "table",
            "title": "Dataset preview",
            "text": "First 10 rows of the dataset.",
            "data": df.head(10),
        }

    if tool_name == "summary":
        numeric = df.select_dtypes(include="number")

        if numeric.empty:
            return {
                "kind": "text",
                "title": "Summary statistics",
                "text": "No numeric columns were found.",
                "data": None,
            }

        return {
            "kind": "table",
            "title": "Summary statistics",
            "text": "Descriptive statistics for numeric columns.",
            "data": numeric.describe().T.reset_index(names="column"),
        }

    if tool_name == "correlation":
        numeric = df.select_dtypes(include="number")

        if numeric.shape[1] < 2:
            return {
                "kind": "text",
                "title": "Correlation",
                "text": "At least two numeric columns are required.",
                "data": None,
            }

        return {
            "kind": "heatmap",
            "title": "Correlation matrix",
            "text": "Pearson correlations between numeric columns.",
            "data": numeric.corr(),
        }

    if tool_name == "time_series":
        date_col = find_column(
            df, ["date", "timestamp", "time"]
        )
        metric = find_column(
            df, ["sales", "revenue", "profit", "amount", "total"],
            numeric_only=True,
        )

        if date_col is None or metric is None:
            return {
                "kind": "unsupported",
                "title": "Time-series analysis",
                "text": "Could not identify a date column and numeric metric.",
                "data": None,
            }

        temp = df[[date_col, metric]].copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        temp = temp.dropna(subset=[date_col])

        if temp.empty:
            return {
                "kind": "unsupported",
                "title": "Time-series analysis",
                "text": "The detected date column contains no valid dates.",
                "data": None,
            }

        temp["period"] = temp[date_col].dt.to_period("M").astype(str)
        result = (
            temp.groupby("period", as_index=False)[metric]
            .sum()
            .sort_values("period")
        )

        return {
            "kind": "line",
            "title": f"{metric} by month",
            "text": f"Monthly totals calculated using {date_col}.",
            "data": result,
            "x": "period",
            "y": metric,
        }

    if tool_name == "ranking":
        metric = find_column(
            df, ["revenue", "sales", "profit", "amount", "total", "quantity"],
            numeric_only=True,
        )
        category = find_column(
            df, ["product", "category", "region", "customer", "segment"]
        )

        if metric is None or category is None:
            return {
                "kind": "unsupported",
                "title": "Ranking analysis",
                "text": "Could not identify a numeric metric and category column.",
                "data": None,
            }

        ascending = any(
            x in q for x in ["lowest", "bottom", "least"]
        )

        result = (
            df.groupby(category, dropna=False)[metric]
            .sum()
            .sort_values(ascending=ascending)
            .head(10)
            .reset_index()
        )

        result.columns = [str(category), f"sum_{metric}"]

        return {
            "kind": "bar",
            "title": f"{category} ranked by {metric}",
            "text": f"Ranked {category} using summed {metric}.",
            "data": result,
            "x": str(category),
            "y": f"sum_{metric}",
        }

    return {
        "kind": "profile",
        "title": "Dataset overview",
        "text": (
            "I could not map the question to a supported operation. "
            "Try asking about missing values, summary statistics, "
            "correlations, monthly sales, or top products."
        ),
        "data": profile_dataframe(df),
    }
