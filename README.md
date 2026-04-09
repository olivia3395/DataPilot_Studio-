# DataPilot Studio

DataPilot Studio is a polished Streamlit-based data science agent app. It supports:

- exploratory data analysis with an agent workflow
- chart generation from LLM-written or fallback Python code
- optional baseline ML modeling with automatic preprocessing
- XGBoost support when the package is installed
- five built-in sample datasets so the app works immediately
- markdown report export and code download

## What is new in v2

- improved Streamlit UI with a cleaner dashboard layout
- five built-in sample datasets:
  - Customer Churn
  - Retail Sales Time Series
  - Iris Classification
  - Wine Classification
  - Diabetes Regression
- baseline ML modeling with target selection
- XGBoost classifier/regressor support with fallback tree models
- feature importance display and prediction preview
- markdown report export
- better chart generation and execution sandbox

## Project structure

```text
DataPilot/
├── streamlit_app.py
├── requirements.txt
├── .env.example
├── agents/
├── llm/
├── orchestration/
├── tools/
├── data/demo/
└── outputs/
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
python -m streamlit run streamlit_app.py
```

## Environment variables

If you want LLM-backed planning and code generation, put your key in `.env`:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini
```

Without an API key, the app still runs using a deterministic fallback pipeline.

## Suggested demo flows

### EDA-only
- choose Retail Sales Time Series
- leave modeling enabled or disabled
- ask: `Summarize the dataset and visualize the most important pattern.`

### Classification
- choose Customer Churn
- set target column to `churned`
- ask: `Find the main churn drivers and train a baseline model.`

### Regression
- choose Diabetes Regression
- set target column to `disease_progression`
- ask: `Train a regression baseline and explain the main predictive signals.`

## Notes

- The execution sandbox blocks unsafe imports such as `os`, `sys`, `subprocess`, and network access.
- Generated charts and the latest markdown report are saved under `outputs/`.
- When `xgboost` is unavailable, the app automatically falls back to a tree-based sklearn model.
