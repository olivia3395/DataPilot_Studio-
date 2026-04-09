from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass(frozen=True)
class SampleDatasetInfo:
    key: str
    filename: str
    title: str
    description: str
    target_hint: str
    task_type: str


BASE_DIR = Path(__file__).resolve().parent.parent / 'data' / 'demo'

SAMPLE_DATASETS: Dict[str, SampleDatasetInfo] = {
    'customer_churn': SampleDatasetInfo(
        key='customer_churn',
        filename='customer_churn.csv',
        title='Customer Churn',
        description='Synthetic subscription dataset with tenure, billing, support, and churn outcome.',
        target_hint='churned',
        task_type='classification',
    ),
    'retail_sales': SampleDatasetInfo(
        key='retail_sales',
        filename='retail_sales_timeseries.csv',
        title='Retail Sales Time Series',
        description='Daily retail performance with promo activity, channel, region, visits, units, and revenue.',
        target_hint='sales_revenue',
        task_type='regression',
    ),
    'iris': SampleDatasetInfo(
        key='iris',
        filename='iris_classification.csv',
        title='Iris Classification',
        description='Classic flower measurement dataset for multi-class classification.',
        target_hint='species',
        task_type='classification',
    ),
    'wine': SampleDatasetInfo(
        key='wine',
        filename='wine_quality_classification.csv',
        title='Wine Classification',
        description='Chemical measurements for predicting wine class.',
        target_hint='wine_class',
        task_type='classification',
    ),
    'diabetes': SampleDatasetInfo(
        key='diabetes',
        filename='diabetes_regression.csv',
        title='Diabetes Progression',
        description='Standard regression dataset for predicting disease progression.',
        target_hint='disease_progression',
        task_type='regression',
    ),
}


def list_sample_datasets() -> List[SampleDatasetInfo]:
    return list(SAMPLE_DATASETS.values())


def get_sample_dataset(key: str) -> pd.DataFrame:
    info = SAMPLE_DATASETS[key]
    return pd.read_csv(BASE_DIR / info.filename)
