from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd


@dataclass
class DatasetSummary:
    shape: tuple[int, int]
    columns: List[str]
    dtypes: Dict[str, str]
    missing_ratio: Dict[str, float]
    numeric_summary: Dict[str, Dict[str, float]]
    categorical_top_values: Dict[str, Dict[str, int]]
    sample_rows: List[Dict[str, Any]]

    def to_markdown(self) -> str:
        lines = []
        lines.append(f"Shape: {self.shape[0]} rows x {self.shape[1]} columns")
        lines.append("Columns: " + ", ".join(self.columns))
        lines.append("\nDtypes:")
        for col, dtype in self.dtypes.items():
            lines.append(f"- {col}: {dtype}")
        lines.append("\nMissing ratio:")
        for col, ratio in self.missing_ratio.items():
            lines.append(f"- {col}: {ratio:.2%}")
        if self.numeric_summary:
            lines.append("\nNumeric summary:")
            for col, stats in self.numeric_summary.items():
                lines.append(
                    f"- {col}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, min={stats['min']:.3f}, max={stats['max']:.3f}"
                )
        if self.categorical_top_values:
            lines.append("\nCategorical top values:")
            for col, topv in self.categorical_top_values.items():
                pretty = ", ".join([f"{k} ({v})" for k, v in topv.items()])
                lines.append(f"- {col}: {pretty}")
        lines.append("\nSample rows:")
        for row in self.sample_rows[:5]:
            lines.append(f"- {row}")
        return "\n".join(lines)


def summarize_dataframe(df: pd.DataFrame) -> DatasetSummary:
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    numeric_summary: Dict[str, Dict[str, float]] = {}
    if numeric_cols:
        desc = df[numeric_cols].describe().transpose().fillna(0.0)
        for col in numeric_cols[:12]:
            numeric_summary[col] = {
                "mean": float(desc.loc[col, "mean"]) if "mean" in desc.columns else 0.0,
                "std": float(desc.loc[col, "std"]) if "std" in desc.columns else 0.0,
                "min": float(desc.loc[col, "min"]) if "min" in desc.columns else 0.0,
                "max": float(desc.loc[col, "max"]) if "max" in desc.columns else 0.0,
            }

    categorical_top_values: Dict[str, Dict[str, int]] = {}
    for col in categorical_cols[:8]:
        vc = df[col].astype(str).value_counts(dropna=False).head(5)
        categorical_top_values[col] = {str(k): int(v) for k, v in vc.items()}

    return DatasetSummary(
        shape=df.shape,
        columns=df.columns.tolist(),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        missing_ratio={col: float(df[col].isna().mean()) for col in df.columns},
        numeric_summary=numeric_summary,
        categorical_top_values=categorical_top_values,
        sample_rows=df.head(5).to_dict(orient="records"),
    )


def quick_dataset_facts(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include=['number', 'bool']).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    total_missing = int(df.isna().sum().sum())
    return {
        'rows': int(df.shape[0]),
        'columns': int(df.shape[1]),
        'numeric_columns': numeric_cols,
        'categorical_columns': categorical_cols,
        'missing_cells': total_missing,
        'duplicate_rows': int(df.duplicated().sum()),
    }
