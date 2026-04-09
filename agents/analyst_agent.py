from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from agents.base_agent import BaseAgent


class AnalystAgent(BaseAgent):
    name = 'analyst'

    def _fallback_code(self, df: pd.DataFrame, question: str, target_col: str | None = None) -> str:
        numeric_cols: List[str] = df.select_dtypes(include=['number', 'bool']).columns.tolist()
        categorical_cols: List[str] = [c for c in df.columns if c not in numeric_cols]
        q = question.lower()

        if target_col and target_col in df.columns:
            if target_col in numeric_cols:
                top_feats = [c for c in numeric_cols if c != target_col][:4]
                if top_feats:
                    first = top_feats[0]
                    return f"""
print('Target distribution summary:')
print(df[{target_col!r}].describe().to_string())
correlations = df[{[target_col] + top_feats!r}].corr(numeric_only=True)[{target_col!r}].sort_values(ascending=False)
print('\nCorrelations with target:')
print(correlations.to_string())
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df[{target_col!r}].hist(bins=25, ax=axes[0])
axes[0].set_title('Target distribution: {target_col}')
df.plot.scatter(x={first!r}, y={target_col!r}, ax=axes[1], alpha=0.65)
axes[1].set_title('{first} vs {target_col}')
plt.tight_layout()
""".strip()
            else:
                focus_cat = categorical_cols[0] if categorical_cols else target_col
                focus_num = numeric_cols[0] if numeric_cols else None
                if focus_num:
                    return f"""
rate = df.groupby({target_col!r})[{focus_num!r}].agg(['mean', 'median', 'count'])
print('Numeric summary by target:')
print(rate.to_string())
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df[{target_col!r}].astype(str).value_counts().plot(kind='bar', ax=axes[0])
axes[0].set_title('Target balance: {target_col}')
df.boxplot(column={focus_num!r}, by={target_col!r}, ax=axes[1])
axes[1].set_title('{focus_num} by {target_col}')
plt.suptitle('')
plt.tight_layout()
""".strip()

        if 'missing' in q:
            return """
missing = df.isna().mean().sort_values(ascending=False)
print('Missing ratio by column:')
print(missing.to_string())
missing.head(12).plot(kind='bar')
plt.title('Top Missingness Ratios')
plt.ylabel('Missing Ratio')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
""".strip()

        if len(numeric_cols) >= 2 and ('correlation' in q or 'related' in q or 'relationship' in q):
            focus = numeric_cols[:6]
            x_col, y_col = focus[:2]
            return f"""
print('Correlation matrix:')
print(df[{focus!r}].corr(numeric_only=True).round(3).to_string())
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df[{focus!r}].corr(numeric_only=True).plot(kind='bar', ax=axes[0])
axes[0].set_title('Correlation profile')
df.plot.scatter(x={x_col!r}, y={y_col!r}, alpha=0.7, ax=axes[1])
axes[1].set_title('Scatter Plot: {x_col} vs {y_col}')
plt.tight_layout()
""".strip()

        if numeric_cols and categorical_cols:
            num_col = numeric_cols[0]
            cat_col = categorical_cols[0]
            return f"""
summary = df.groupby({cat_col!r})[{num_col!r}].agg(['mean','median','count']).sort_values('mean', ascending=False)
print('Grouped summary:')
print(summary.to_string())
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
summary['mean'].head(10).plot(kind='bar', ax=axes[0])
axes[0].set_title('Average {num_col} by {cat_col}')
axes[0].tick_params(axis='x', rotation=45)
df[{num_col!r}].hist(bins=20, ax=axes[1])
axes[1].set_title('Distribution of {num_col}')
plt.tight_layout()
""".strip()

        if numeric_cols:
            num_col = numeric_cols[0]
            return f"""
print(df[{num_col!r}].describe().to_string())
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df[{num_col!r}].hist(bins=20, ax=axes[0])
axes[0].set_title('Distribution of {num_col}')
df[{num_col!r}].rolling(20, min_periods=1).mean().plot(ax=axes[1])
axes[1].set_title('Rolling mean of {num_col}')
plt.tight_layout()
""".strip()

        cat_col = df.columns[0]
        return f"""
vc = df[{cat_col!r}].astype(str).value_counts().head(10)
print(vc.to_string())
vc.plot(kind='bar')
plt.title('Top values of {cat_col}')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
""".strip()

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        df = state['df']
        question = state['question']
        summary = state['summary_markdown']
        plan = state['plan']
        llm = state['llm']
        target_col = state.get('target_col')

        system_prompt = (
            'You are a Python data analysis agent. Write only executable Python code. The DataFrame is already available as df. '
            'Use only pandas, numpy, matplotlib.pyplot as plt, and seaborn as sns. Print compact results and generate at most two charts. '
            'Do not use files, network, subprocesses, or unsafe imports.'
        )
        user_prompt = f"Question:\n{question}\n\nTarget column: {target_col or 'None'}\n\nPlan:\n{plan}\n\nDataset summary:\n{summary}"
        code = llm.complete(system_prompt, user_prompt)
        if not code:
            code = self._fallback_code(df, question, target_col)
        clean = code.strip().removeprefix('```python').removeprefix('```').removesuffix('```').strip()
        state['generated_code'] = clean
        return state
