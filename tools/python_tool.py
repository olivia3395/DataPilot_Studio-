from __future__ import annotations

import io
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DISALLOWED_TOKENS = [
    'import os',
    'import sys',
    'import subprocess',
    'import shutil',
    'import socket',
    '__import__',
    'open(',
    'eval(',
    'exec(',
    'requests.',
    'http',
    'https',
]


class PythonExecutionTool:
    def __init__(self, output_dir: str = 'outputs') -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, code: str, df: pd.DataFrame) -> Dict[str, Any]:
        for token in DISALLOWED_TOKENS:
            if token in code:
                return {
                    'success': False,
                    'stdout': '',
                    'error': f'Blocked unsafe token: {token}',
                    'chart_paths': [],
                    'locals': {},
                }

        for old_chart in self.output_dir.glob('chart_*.png'):
            old_chart.unlink(missing_ok=True)

        plt.close('all')
        stdout_buffer = io.StringIO()
        safe_globals: Dict[str, Any] = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'df': df.copy(),
            '__builtins__': {
                'len': len,
                'range': range,
                'min': min,
                'max': max,
                'sum': sum,
                'abs': abs,
                'float': float,
                'int': int,
                'str': str,
                'print': print,
                'sorted': sorted,
                'list': list,
                'dict': dict,
                'set': set,
                'enumerate': enumerate,
                'zip': zip,
                'round': round,
                'isinstance': isinstance,
            },
        }
        safe_locals: Dict[str, Any] = {}
        chart_paths = []

        try:
            with redirect_stdout(stdout_buffer):
                exec(code, safe_globals, safe_locals)
            figures = [plt.figure(num) for num in plt.get_fignums()]
            for idx, fig in enumerate(figures, start=1):
                path = self.output_dir / f'chart_{idx}.png'
                fig.savefig(path, bbox_inches='tight', dpi=140)
                chart_paths.append(str(path))
            return {
                'success': True,
                'stdout': stdout_buffer.getvalue(),
                'error': '',
                'chart_paths': chart_paths,
                'locals': safe_locals,
            }
        except Exception:
            return {
                'success': False,
                'stdout': stdout_buffer.getvalue(),
                'error': traceback.format_exc(),
                'chart_paths': chart_paths,
                'locals': safe_locals,
            }
        finally:
            plt.close('all')
