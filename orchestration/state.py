from __future__ import annotations

from typing import Any, Dict


def make_initial_state(df, question: str, llm, target_col: str | None = None, run_ml: bool = False, task_type: str | None = None) -> Dict[str, Any]:
    return {
        'df': df,
        'question': question,
        'llm': llm,
        'target_col': target_col,
        'run_ml': run_ml,
        'task_type': task_type,
    }
