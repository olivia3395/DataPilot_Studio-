from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None
    XGBRegressor = None


class MLModelTool:
    def _infer_task_type(self, series: pd.Series) -> str:
        if pd.api.types.is_numeric_dtype(series):
            unique = series.nunique(dropna=True)
            return 'classification' if unique <= 10 else 'regression'
        return 'classification'

    def _split_features(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()
        numeric_cols = X.select_dtypes(include=['number', 'bool']).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]
        return X, y, numeric_cols, categorical_cols

    def _preprocessor(self, numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
        transformers = []
        if numeric_cols:
            transformers.append(('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), numeric_cols))
        if categorical_cols:
            transformers.append((
                'cat',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ]),
                categorical_cols,
            ))
        return ColumnTransformer(transformers=transformers)

    def _build_model(self, task_type: str):
        if task_type == 'classification':
            if XGBOOST_AVAILABLE:
                return XGBClassifier(
                    n_estimators=180,
                    max_depth=4,
                    learning_rate=0.07,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric='mlogloss',
                    random_state=42,
                ), 'XGBoost Classifier'
            return GradientBoostingClassifier(random_state=42), 'Gradient Boosting Classifier'
        if XGBOOST_AVAILABLE:
            return XGBRegressor(
                n_estimators=220,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective='reg:squarederror',
                random_state=42,
            ), 'XGBoost Regressor'
        return RandomForestRegressor(n_estimators=250, random_state=42), 'Random Forest Regressor'

    def _fallback_importance(self, X_train: pd.DataFrame, y_train: pd.Series, task_type: str, preprocessor: ColumnTransformer):
        model = RandomForestClassifier(n_estimators=180, random_state=42) if task_type == 'classification' else RandomForestRegressor(n_estimators=220, random_state=42)
        pipe = Pipeline([('prep', preprocessor), ('model', model)])
        pipe.fit(X_train, y_train)
        return pipe

    def _feature_importance(self, pipeline: Pipeline, fallback_pipeline: Optional[Pipeline], feature_names: List[str]) -> List[Tuple[str, float]]:
        model = pipeline.named_steps['model']
        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif fallback_pipeline is not None and hasattr(fallback_pipeline.named_steps['model'], 'feature_importances_'):
            importances = fallback_pipeline.named_steps['model'].feature_importances_
        if importances is None:
            return []
        ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        return [(name, float(score)) for name, score in ranked[:10]]

    def run(self, df: pd.DataFrame, target_col: str, task_type: Optional[str] = None) -> Dict[str, Any]:
        target_col = target_col.strip()
        if target_col not in df.columns:
            return {'success': False, 'error': f'Target column {target_col!r} was not found in the dataset.'}

        clean_df = df.dropna(subset=[target_col]).copy()
        if clean_df.shape[0] < 30:
            return {'success': False, 'error': 'The dataset is too small after dropping missing target rows.'}

        y = clean_df[target_col]
        inferred_task = task_type or self._infer_task_type(y)
        X, y, numeric_cols, categorical_cols = self._split_features(clean_df, target_col)
        if X.shape[1] == 0:
            return {'success': False, 'error': 'No feature columns remain after removing the target.'}

        test_size = 0.2 if len(clean_df) >= 80 else 0.25
        stratify = y if inferred_task == 'classification' and y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)
        preprocessor = self._preprocessor(numeric_cols, categorical_cols)
        model, model_name = self._build_model(inferred_task)
        pipeline = Pipeline([('prep', preprocessor), ('model', model)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        metrics: Dict[str, float] = {}
        if inferred_task == 'classification':
            metrics['accuracy'] = float(accuracy_score(y_test, preds))
            metrics['macro_f1'] = float(f1_score(y_test, preds, average='macro'))
            baseline = y_train.mode().iloc[0]
            metrics['baseline_accuracy'] = float((y_test == baseline).mean())
        else:
            metrics['rmse'] = float(root_mean_squared_error(y_test, preds))
            metrics['mae'] = float(mean_absolute_error(y_test, preds))
            metrics['r2'] = float(r2_score(y_test, preds))

        feature_names = pipeline.named_steps['prep'].get_feature_names_out().tolist()
        fallback_pipe = None
        if XGBOOST_AVAILABLE and feature_names and not hasattr(pipeline.named_steps['model'], 'feature_importances_'):
            fallback_pipe = self._fallback_importance(X_train, y_train, inferred_task, preprocessor)
        importances = self._feature_importance(pipeline, fallback_pipe, feature_names)

        preview_rows = pd.DataFrame({'actual': y_test}).reset_index(drop=True)
        preview_rows['predicted'] = pd.Series(preds).reset_index(drop=True)
        return {
            'success': True,
            'task_type': inferred_task,
            'model_name': model_name,
            'metrics': metrics,
            'feature_importance': importances,
            'predictions_preview': preview_rows.head(12).to_dict(orient='records'),
            'train_rows': int(len(X_train)),
            'test_rows': int(len(X_test)),
            'xgboost_used': bool(XGBOOST_AVAILABLE and model_name.startswith('XGBoost')),
        }
