import os
import streamlit as st
import pandas as pd
import plotly.express as px
import mlflow
from mlflow.tracking import MlflowClient

# ----------------------------------------------------------------------
# Função para carregar os dados de predições
@st.cache_data
def load_prediction_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Arquivo não encontrado: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return df

# ----------------------------------------------------------------------
# Função para carregar métricas do MLflow
@st.cache_data
def get_mlflow_metrics(experiment_name: str = "Default", max_results: int = 20) -> pd.DataFrame:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        st.error(f"Experimento {experiment_name} não encontrado no MLflow.")
        return pd.DataFrame()
    
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
    return pd.DataFrame(metrics_list)

# ----------------------------------------------------------------------
# Layout do Dashboard com Streamlit
st.set_page_config(page_title="Monitoramento de Operação", layout="wide")
st.title("Dashboard de Monitoramento da Operação")

st.markdown("""
Este dashboard monitora a operação do modelo de predição de arremessos. 
Verifique as métricas registradas via MLflow e a distribuição dos dados de predição.
""")

# ----------------------------------------------------------------------
# Seção 1: Carregar e visualizar dados de predições
st.header("Dados de Predição")
prediction_path = st.text_input("Caminho do arquivo de predições", "./../../../data/08_reporting/resultado_pipeline.parquet")
print(os.path.abspath(prediction_path))
df_pred = load_prediction_data(prediction_path)
if df_pred.empty:
    st.warning("Nenhum dado disponível para visualização.")
else:
    st.write("Visualização das primeiras linhas dos dados:", df_pred.head())

    # Gráfico de distribuição de probabilidades
    if "y_proba" in df_pred.columns:
        st.subheader("Distribuição das Probabilidades")
        fig_prob = px.histogram(df_pred, x="y_proba", nbins=20, title="Histograma das Probabilidades")
        st.plotly_chart(fig_prob, use_container_width=True)
    else:
        st.warning("Coluna 'y_proba' não encontrada no dataset.")

    # Gráfico de distribuição das predições
    if "y_pred" in df_pred.columns:
        st.subheader("Distribuição das Predições")
        fig_pred = px.histogram(df_pred, x="y_pred", title="Contagem das Classes Preditadas")
        st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.warning("Coluna 'y_pred' não encontrada no dataset.")
        
    # Exemplo de análise de data drift para uma feature, como 'shot_distance'
    if "shot_distance" in df_pred.columns:
        st.subheader("Distribuição da Feature: shot_distance")
        fig_feature = px.histogram(df_pred, x="shot_distance", nbins=20, title="Distribuição de shot_distance")
        st.plotly_chart(fig_feature, use_container_width=True)
    else:
        st.warning("Coluna 'shot_distance' não encontrada no dataset.")

# ----------------------------------------------------------------------
# Seção 2: Métricas registradas pelo MLflow
st.header("Métricas do MLflow")
experiment_name = st.text_input("Nome do Experimento MLflow", "Default")
df_metrics = get_mlflow_metrics(experiment_name)
if df_metrics.empty:
    st.warning("Nenhuma métrica encontrada para o experimento informado.")
else:
    st.write("Métricas extraídas das últimas runs:", df_metrics)

    if "logloss_prod" in df_metrics.columns:
        st.subheader("Evolução do Log Loss")
        df_metrics = df_metrics.sort_values("start_time")
        fig_logloss = px.line(df_metrics, x="start_time", y="logloss_prod", title="Evolução do Log Loss")
        st.plotly_chart(fig_logloss, use_container_width=True)

    if "f1_prod" in df_metrics.columns:
        st.subheader("Evolução do F1 Score")
        df_metrics = df_metrics.sort_values("start_time")
        fig_f1 = px.line(df_metrics, x="start_time", y="f1_prod", title="Evolução do F1 Score")
        st.plotly_chart(fig_f1, use_container_width=True)

# ----------------------------------------------------------------------
# Observações e instruções adicionais
st.markdown("""
### Observações:
- **Métricas:** As métricas exibidas (Log Loss e F1 Score) ajudam a identificar se o modelo está se comportando conforme esperado em produção.
- **Data Drift:** A comparação de distribuições das features pode indicar se houve mudanças significativas na base de produção em comparação com a base de treinamento.
- **Alertas:** É possível estender este dashboard para incluir alertas visuais (ex.: mudança brusca nas métricas) e integração com sistemas de notificação.
""")
