from __future__ import annotations

from typing import Any, Dict

from agents.base_agent import BaseAgent


class VerifierAgent(BaseAgent):
    name = 'verifier'

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        result = state['execution_result']
        ml_result = state.get('ml_result')
        notes = []
        if result['success']:
            notes.append('The exploratory analysis code executed successfully.')
            if result['chart_paths']:
                notes.append(f"It generated {len(result['chart_paths'])} chart(s).")
            if not result['stdout'].strip() and not result['chart_paths']:
                notes.append('The code ran, but the output is sparse, so the conclusion should remain modest.')
        else:
            notes.append('The exploratory code failed, so any interpretation should explicitly acknowledge the traceback.')

        if ml_result:
            if ml_result.get('success'):
                notes.append(f"A predictive modeling step also ran successfully using {ml_result['model_name']}.")
            else:
                notes.append(f"The modeling step did not finish cleanly: {ml_result.get('error', 'unknown error')}")

        state['verification'] = ' '.join(notes)
        return state
