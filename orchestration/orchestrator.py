from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from agents.analyst_agent import AnalystAgent
from agents.inspector_agent import InspectorAgent
from agents.planner_agent import PlannerAgent
from agents.verifier_agent import VerifierAgent
from agents.writer_agent import WriterAgent
from orchestration.state import make_initial_state
from tools.ml_tool import MLModelTool
from tools.python_tool import PythonExecutionTool


class DataScienceOrchestrator:
    def __init__(self, llm, output_dir: str = 'outputs') -> None:
        self.llm = llm
        self.inspector = InspectorAgent()
        self.planner = PlannerAgent()
        self.analyst = AnalystAgent()
        self.verifier = VerifierAgent()
        self.writer = WriterAgent()
        self.python_tool = PythonExecutionTool(output_dir=output_dir)
        self.ml_tool = MLModelTool()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, df, question: str, target_col: str | None = None, run_ml: bool = False, task_type: str | None = None) -> Dict[str, Any]:
        state = make_initial_state(df, question, self.llm, target_col=target_col, run_ml=run_ml, task_type=task_type)
        steps: List[Dict[str, str]] = []

        state = self.inspector.run(state)
        steps.append({'agent': 'Inspector', 'message': 'Generated dataset summary, profiling stats, and quick facts.'})

        state = self.planner.run(state)
        steps.append({'agent': 'Planner', 'message': state['plan']})

        state = self.analyst.run(state)
        steps.append({'agent': 'Analyst', 'message': 'Generated exploratory Python analysis code.'})

        execution_result = self.python_tool.run(state['generated_code'], state['df'])
        state['execution_result'] = execution_result
        if execution_result['success']:
            steps.append({'agent': 'Python Tool', 'message': 'Executed exploratory analysis successfully.'})
        else:
            last_line = execution_result['error'].splitlines()[-1] if execution_result['error'] else 'Unknown execution failure.'
            steps.append({'agent': 'Python Tool', 'message': f'Execution failed. {last_line}'})

        if run_ml and target_col:
            ml_result = self.ml_tool.run(df=state['df'], target_col=target_col, task_type=task_type)
            state['ml_result'] = ml_result
            if ml_result.get('success'):
                steps.append({'agent': 'Modeler', 'message': f"Trained {ml_result['model_name']} on target '{target_col}'."})
            else:
                steps.append({'agent': 'Modeler', 'message': ml_result.get('error', 'Modeling failed.')})
        else:
            state['ml_result'] = None

        state = self.verifier.run(state)
        steps.append({'agent': 'Verifier', 'message': state['verification']})

        state = self.writer.run(state)
        steps.append({'agent': 'Writer', 'message': 'Wrote the final grounded summary.'})

        state['steps'] = steps
        report_path = self.output_dir / 'latest_report.md'
        report_path.write_text(self._build_report(state), encoding='utf-8')
        state['report_markdown'] = report_path.read_text(encoding='utf-8')
        state['report_path'] = str(report_path)
        return state

    def _build_report(self, state: Dict[str, Any]) -> str:
        lines = ['# DataPilot Report', '']
        lines.append(f"## Question\n{state['question']}")
        if state.get('target_col'):
            lines.append(f"\n## Target\n{state['target_col']}")
        lines.append(f"\n## Plan\n{state['plan']}")
        lines.append(f"\n## Final Summary\n{state['final_summary']}")
        lines.append(f"\n## Verification\n{state['verification']}")
        if state.get('ml_result'):
            lines.append(f"\n## Model Result\n{state['ml_result']}")
        lines.append(f"\n## Execution Output\n```\n{state['execution_result']['stdout']}\n```")
        lines.append(f"\n## Generated Code\n```python\n{state['generated_code']}\n```")
        return '\n'.join(lines)
