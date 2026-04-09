from __future__ import annotations

from typing import Any, Dict

from agents.base_agent import BaseAgent
from tools.data_summary_tool import summarize_dataframe


class InspectorAgent(BaseAgent):
    name = "inspector"

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        df = state["df"]
        summary = summarize_dataframe(df)
        state["summary"] = summary
        state["summary_markdown"] = summary.to_markdown()
        return state
