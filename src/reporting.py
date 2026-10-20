from datetime import datetime

def make_markdown_report(question: str, answer: str, plan: str, validation: str) -> str:
    return f"""# Autonomous AI Data Analyst Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Business question
{question}

## Investigation plan
{plan}

## Findings
{answer}

## Validation
{validation}

## Limitations
- Findings depend on the uploaded dataset's quality and coverage.
- Correlation or temporal association does not prove causation.
- Review results before using them for consequential business decisions.
"""
