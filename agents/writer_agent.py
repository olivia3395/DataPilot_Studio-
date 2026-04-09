from __future__ import annotations

from typing import Any, Dict

from agents.base_agent import BaseAgent


class WriterAgent(BaseAgent):
    name = 'writer'

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state['question']
        summary = state['summary_markdown']
        execution_result = state['execution_result']
        verification = state['verification']
        ml_result = state.get('ml_result')
        llm = state['llm']

        ml_block = ''
        if ml_result:
            ml_block = f"\nML result:\n{ml_result}\n"

        system_prompt = (
            'You are a data science reporting agent. Write a short, grounded summary using only the observed dataset summary, code outputs, and model metrics. '
            'Do not invent results. Keep it concise, specific, and business-friendly.'
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Dataset summary:\n{summary}\n\n"
            f"Execution stdout:\n{execution_result['stdout']}\n\n"
            f"Verification:\n{verification}\n"
            f"{ml_block}"
        )
        final_summary = llm.complete(system_prompt, user_prompt)
        if not final_summary:
            if execution_result['success']:
                stdout = execution_result['stdout'].strip() or 'The code ran successfully and produced at least one chart.'
                final_summary = f"The exploratory workflow completed successfully. Main observed output: {stdout[:700]}"
                if ml_result and ml_result.get('success'):
                    metrics = ', '.join(f"{k}={v:.3f}" for k, v in ml_result['metrics'].items())
                    final_summary += f" The modeling step used {ml_result['model_name']} with metrics {metrics}."
                final_summary += ' Treat the result as exploratory unless validated on a separate business objective.'
            else:
                final_summary = (
                    'The exploratory code failed to execute, so no reliable conclusion should be drawn yet. '
                    'Please inspect the traceback, simplify the question, or retry with a clearer target variable.'
                )
        state['final_summary'] = final_summary
        return state
