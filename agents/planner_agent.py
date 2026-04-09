from __future__ import annotations

from typing import Any, Dict

from agents.base_agent import BaseAgent


class PlannerAgent(BaseAgent):
    name = 'planner'

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state['question']
        summary = state['summary_markdown']
        target_col = state.get('target_col')
        llm = state['llm']

        system_prompt = (
            'You are a concise data science planning agent. Given a dataset summary, user question, and optional target column, '
            'produce a 4-step plan. Mention profiling, analysis or modeling, validation, and final communication.'
        )
        user_prompt = f"Question:\n{question}\n\nTarget column: {target_col or 'None'}\n\nDataset summary:\n{summary}"
        plan = llm.complete(system_prompt, user_prompt)
        if not plan:
            if target_col:
                plan = (
                    '1. Review schema, missingness, and class balance or target distribution.\n'
                    '2. Inspect the strongest descriptive patterns linked to the target.\n'
                    '3. Train a predictive baseline and compare performance against a simple benchmark.\n'
                    '4. Report the main drivers, limitations, and next actions.'
                )
            else:
                plan = (
                    '1. Profile the dataset shape, column types, missingness, and potential quality issues.\n'
                    '2. Identify the most relevant numeric and categorical variables for the question.\n'
                    '3. Generate focused statistics and a small set of clear visualizations.\n'
                    '4. Summarize the strongest patterns in plain business language.'
                )
        state['plan'] = plan
        return state
