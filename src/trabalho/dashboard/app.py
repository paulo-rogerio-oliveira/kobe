import os
import streamlit as st
import pandas as pd
import plotly.express as px
import mlflow
from mlflow.tracking import MlflowClient


@st.cache_data(show_spinner=False)
def load_prediction_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Arquivo não encontrado: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return df

@st.cache_data(show_spinner=False)
def get_mlflow_metrics(experiment_name: str = "Default", max_results: int = 20) -> pd.DataFrame:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        st.error(f"Experimento {experiment_name} não encontrado no MLflow.")
        return pd.DataFrame()
    
    st.write("Experiment ID:", experiment.experiment_id)  # Debug: Mostra o id do experimento
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=max_results
    )

    # Extrai métricas de cada run
    metrics_list = []
    for run in runs:
        metrics = run.data.metrics
        metrics_list.append({
            "run_id": run.info.run_id,
            "logloss_prod": metrics.get("logloss_prod", None),
            "f1_prod": metrics.get("f1_prod", None),
            "start_time": run.info.start_time
        })
    
    df_metrics = pd.DataFrame(metrics_list)
    if not df_metrics.empty:
        df_metrics["start_time"] = pd.to_datetime(df_metrics["start_time"], unit="ms")
    return df_metrics

st.set_page_config(page_title="Monitoramento de Operação", layout="wide")
st.title("Dashboard de Monitoramento da Operação")

st.markdown("""
Este dashboard monitora a operação do modelo de predição de arremessos. 
Exibe as métricas registradas via MLflow e a distribuição dos dados de predição.
""")

st.header("Dados de Predição")
prediction_current_path = st.text_input("Caminho da base para predição", "./../../../data/01_raw/dataset_kobe_prod.parquet")
st.button("Carregar dados de predição", on_click=load_prediction_data, args=(prediction_current_path,))

prediction_path = st.text_input("Caminho do arquivo de resultado das predições", "./../../../data/08_reporting/resultado_pipeline.parquet", disabled=True)
st.write("Caminho absoluto:", os.path.abspath(prediction_path))
df_pred = load_prediction_data(prediction_path)
if df_pred.empty:
    st.warning("Nenhum dado disponível para visualização.")
else:
    st.write("Visualização das primeiras linhas dos dados:", df_pred.head())

    if "y_proba" in df_pred.columns:
        st.subheader("Distribuição das Probabilidades")
        fig_prob = px.histogram(df_pred, x="y_proba", nbins=20, title="Histograma das Probabilidades")
        st.plotly_chart(fig_prob, use_container_width=True)
    else:
        st.warning("Coluna 'y_proba' não encontrada no dataset.")

    if "y_pred" in df_pred.columns:
        st.subheader("Distribuição das Predições")
        fig_pred = px.histogram(df_pred, x="y_pred", title="Contagem das Classes Preditadas")
        st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.warning("Coluna 'y_pred' não encontrada no dataset.")
        
    if "shot_distance" in df_pred.columns:
        st.subheader("Distribuição da Feature: shot_distance")
        fig_feature = px.histogram(df_pred, x="shot_distance", nbins=20, title="Distribuição de shot_distance")
        st.plotly_chart(fig_feature, use_container_width=True)
    else:
        st.warning("Coluna 'shot_distance' não encontrada no dataset.")


