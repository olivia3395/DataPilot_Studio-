<div align="center">
  <h1>🧭 DataPilot Studio</h1>
  <p><em>A polished data science agent app — explore, visualize, model, and export<br/>with an LLM-powered workflow or a fully offline fallback pipeline</em></p>
  <br/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  &nbsp;
  <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white" />
  &nbsp;
  <img src="https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge&logoColor=white" />
  &nbsp;
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  &nbsp;
  <img src="https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
</div>

<br/>


## What It Does

**DataPilot Studio** turns a natural language question into a full data science workflow — EDA, chart generation, baseline ML modeling, and a downloadable report — all from a clean Streamlit dashboard.

Five sample datasets are built in, so the app runs immediately with no data prep required. Drop in an OpenAI key for LLM-backed planning and code generation, or run entirely offline using the deterministic fallback pipeline.

<br/>

## Features

| | |
|---|---|
| 🔍 **Exploratory data analysis** | Agent-driven EDA with automatic summarization |
| 📊 **Chart generation** | LLM-written or fallback Python, executed in a sandboxed environment |
| 🤖 **Baseline ML modeling** | Auto preprocessing, target selection, classification & regression |
| 🌲 **XGBoost support** | Automatic fallback to sklearn tree models if XGBoost is unavailable |
| 🗂️ **5 built-in datasets** | Works immediately — no data upload required |
| 📝 **Markdown report export** | Download a full analysis report in one click |
| 💾 **Code download** | Export generated chart and model code |

<br/>

## Built-in Datasets

| Dataset | Type | Suggested Target |
|---------|------|-----------------|
| 🧑‍💼 Customer Churn | Classification | `churned` |
| 🛒 Retail Sales Time Series | Time series | — |
| 🌸 Iris Classification | Classification | `species` |
| 🍷 Wine Classification | Classification | `wine_class` |
| 🏥 Diabetes Regression | Regression | `disease_progression` |

<br/>

## What's New in v2

- Cleaner dashboard layout with improved Streamlit UI
- Five built-in sample datasets (no file upload needed to get started)
- Baseline ML modeling with flexible target column selection
- XGBoost classifier/regressor with automatic sklearn fallback
- Feature importance display and prediction preview
- Markdown report export
- Improved chart generation and execution sandbox

<br/>

## Project Structure

```
DataPilot/
├── streamlit_app.py        # Main app entry point
├── requirements.txt
├── .env.example
├── agents/                 # Agent workflow logic
├── llm/                    # LLM client & prompt templates
├── orchestration/          # Pipeline orchestration
├── tools/                  # EDA, modeling, chart tools
├── data/demo/              # Built-in sample datasets
└── outputs/                # Generated charts & reports
```

<br/>

## Quickstart

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 2. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env

# 4. Launch the app
python -m streamlit run streamlit_app.py
```

<br/>

## Environment Variables

For LLM-backed planning and code generation, add your key to `.env`:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini
```

> Without a key, the app runs using a **deterministic fallback pipeline** — all features remain available.

<br/>

## Demo Flows

### 📈 EDA only
```
Dataset  →  Retail Sales Time Series
Prompt   →  "Summarize the dataset and visualize the most important pattern."
```

### 🔴 Classification
```
Dataset  →  Customer Churn
Target   →  churned
Prompt   →  "Find the main churn drivers and train a baseline model."
```

### 📉 Regression
```
Dataset  →  Diabetes Regression
Target   →  disease_progression
Prompt   →  "Train a regression baseline and explain the main predictive signals."
```

<br/>


<br/>

<div align="center">
  <img width="900" alt="DataPilot Studio dashboard" src="./images/screenshot.png" />
</div>

<br/>



## Security & Outputs

- The execution sandbox **blocks unsafe imports** — `os`, `sys`, `subprocess`, and network access are all restricted
- Generated charts and markdown reports are saved to `outputs/`
- No data leaves the app unless you explicitly configure an LLM API key

<br/>




<div align="center">
  <sub>Built with Streamlit · scikit-learn · XGBoost · OpenAI</sub>
</div>
