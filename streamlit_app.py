from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from llm.client import LLMClient
from orchestration.orchestrator import DataScienceOrchestrator
from tools.data_summary_tool import quick_dataset_facts
from tools.sample_datasets import SAMPLE_DATASETS, get_sample_dataset, list_sample_datasets

load_dotenv()

st.set_page_config(page_title='DataPilot Studio', page_icon='📊', layout='wide')

st.markdown(
    """
    <style>
    .hero-card {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #7c3aed 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .soft-card {
        border: 1px solid rgba(49, 51, 63, 0.12);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: rgba(248, 250, 252, 0.7);
    }
    .metric-chip {
        padding: 0.7rem 0.9rem;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid rgba(100, 116, 139, 0.18);
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='hero-card'>
        <h1 style='margin-bottom:0.2rem;'>DataPilot Studio</h1>
        <p style='margin:0;'>A polished AI data science agent for EDA, charting, and fast baseline modeling with optional XGBoost.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header('Workspace')
    data_mode = st.radio('Choose a data source', ['Built-in sample dataset', 'Upload your own CSV'], index=0)

    sample_lookup = {info.title: info.key for info in list_sample_datasets()}
    chosen_title = st.selectbox('Sample dataset', list(sample_lookup.keys()), disabled=data_mode != 'Built-in sample dataset')
    selected_key = sample_lookup[chosen_title]
    selected_info = SAMPLE_DATASETS[selected_key]

    uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'], disabled=data_mode != 'Upload your own CSV')

    st.divider()
    st.subheader('Modeling options')
    run_ml = st.checkbox('Run baseline ML modeling', value=True)
    task_type = st.selectbox('Task type', ['auto', 'classification', 'regression'])
    st.caption('If modeling is enabled, choose a target column after the dataset loads.')


def load_dataframe() -> pd.DataFrame:
    if data_mode == 'Upload your own CSV' and uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return get_sample_dataset(selected_key)


def default_question() -> str:
    if run_ml:
        return 'Summarize the dataset, highlight the strongest patterns, and train a predictive baseline.'
    return 'Summarize the dataset and visualize the most important pattern.'


df = load_dataframe()
question = st.text_area('Ask your data agent', value=default_question(), height=110)

facts = quick_dataset_facts(df)
all_columns = df.columns.tolist()
default_target = selected_info.target_hint if data_mode == 'Built-in sample dataset' else (all_columns[-1] if all_columns else None)
target_col = st.selectbox('Target column (optional)', options=['<none>'] + all_columns, index=(all_columns.index(default_target) + 1) if default_target in all_columns else 0)
target_col = None if target_col == '<none>' else target_col

c1, c2, c3, c4 = st.columns(4)
for col, label, value in [
    (c1, 'Rows', facts['rows']),
    (c2, 'Columns', facts['columns']),
    (c3, 'Missing Cells', facts['missing_cells']),
    (c4, 'Duplicate Rows', facts['duplicate_rows']),
]:
    with col:
        st.markdown(f"<div class='metric-chip'><div style='font-size:0.85rem;color:#475569;'>{label}</div><div style='font-size:1.25rem;font-weight:700;'>{value}</div></div>", unsafe_allow_html=True)

left, right = st.columns([1.4, 1])
with left:
    st.subheader('Dataset preview')
    st.dataframe(df.head(20), use_container_width=True, height=360)
with right:
    st.subheader('Dataset notes')
    if data_mode == 'Built-in sample dataset':
        st.markdown(f"<div class='soft-card'><strong>{selected_info.title}</strong><br>{selected_info.description}<br><br><strong>Suggested target:</strong> {selected_info.target_hint}<br><strong>Suggested task:</strong> {selected_info.task_type}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='soft-card'>Upload any CSV and point the app to a target column when you want a modeling run. Without a target, DataPilot stays in exploratory analysis mode.</div>", unsafe_allow_html=True)
    st.write('Numeric columns:', facts['numeric_columns'][:8])
    st.write('Categorical columns:', facts['categorical_columns'][:8])

if st.button('Run DataPilot', type='primary', use_container_width=True):
    if df is None:
        st.error('Please upload a CSV or select one of the built-in sample datasets.')
    elif not question.strip():
        st.error('Please enter an analysis question.')
    else:
        with st.spinner('Running the data science agent...'):
            llm = LLMClient()
            orchestrator = DataScienceOrchestrator(llm=llm, output_dir='outputs')
            result = orchestrator.run(
                df=df,
                question=question,
                target_col=target_col,
                run_ml=run_ml and bool(target_col),
                task_type=None if task_type == 'auto' else task_type,
            )

        st.success('Analysis complete.')
        tabs = st.tabs(['Executive Summary', 'Charts', 'Modeling', 'Agent Trace', 'Generated Code', 'Execution Output'])

        with tabs[0]:
            st.subheader('Executive summary')
            st.write(result['final_summary'])
            st.download_button('Download markdown report', data=result['report_markdown'], file_name='datapilot_report.md', mime='text/markdown')

        with tabs[1]:
            st.subheader('Charts')
            chart_paths = result['execution_result']['chart_paths']
            if not chart_paths:
                st.info('No chart was generated in this run.')
            else:
                for chart_path in chart_paths:
                    st.image(chart_path, use_container_width=True)

        with tabs[2]:
            st.subheader('Modeling results')
            ml_result = result.get('ml_result')
            if not ml_result:
                st.info('Modeling was not run. Enable the modeling checkbox and choose a target column.')
            elif not ml_result.get('success'):
                st.error(ml_result.get('error', 'Modeling failed.'))
            else:
                meta1, meta2, meta3 = st.columns(3)
                meta1.metric('Model', ml_result['model_name'])
                meta2.metric('Train Rows', ml_result['train_rows'])
                meta3.metric('Test Rows', ml_result['test_rows'])
                st.caption('XGBoost used' if ml_result['xgboost_used'] else 'XGBoost not available, so the app used a fallback tree model.')
                st.write('Metrics')
                st.json(ml_result['metrics'])
                if ml_result['feature_importance']:
                    fi_df = pd.DataFrame(ml_result['feature_importance'], columns=['feature', 'importance'])
                    st.write('Top feature importances')
                    st.bar_chart(fi_df.set_index('feature'))
                    st.dataframe(fi_df, use_container_width=True)
                st.write('Prediction preview')
                st.dataframe(pd.DataFrame(ml_result['predictions_preview']), use_container_width=True)

        with tabs[3]:
            st.subheader('Agent trace')
            for step in result['steps']:
                with st.expander(step['agent'], expanded=True):
                    st.write(step['message'])
            with st.expander('Dataset summary'):
                st.text(result['summary_markdown'])

        with tabs[4]:
            st.subheader('Generated Python code')
            st.code(result['generated_code'], language='python')
            st.download_button('Download generated code', data=result['generated_code'], file_name='generated_analysis.py', mime='text/x-python')

        with tabs[5]:
            st.subheader('Execution output')
            exec_result = result['execution_result']
            if exec_result['success']:
                st.success('Exploratory analysis code executed successfully.')
            else:
                st.error('Exploratory analysis code failed.')
            if exec_result['stdout']:
                st.text(exec_result['stdout'])
            if exec_result['error']:
                st.code(exec_result['error'], language='text')
