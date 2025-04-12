import os
import mlflow
import pandas as pd
import pickle
from mlflow.models.signature import infer_signature
from sklearn.metrics import log_loss, f1_score
from sklearn.preprocessing import StandardScaler

def PipelineAplicacao(
    df_prod,
    modelo_final
):
    # 1. Carrega o modelo como sklearn para ter acesso ao predict_proba
    model_uri = "models:/trained_model/latest"
    model = mlflow.sklearn.load_model(model_uri)

    # 2. Carrega modelo
    model = modelo_final

    # 3. Inicia MLflow Run
    with mlflow.start_run(run_name="PipelineAplicacao"):

        X = df_prod.drop(columns="shot_made_flag")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        y_true = df_prod["shot_made_flag"].values

        y_proba = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)   

        # 5. Calcula métricas
        ll = log_loss(y_true, y_proba)
        f1 = f1_score(y_true, y_pred)
        
        mlflow.log_metric("logloss_prod", ll)
        mlflow.log_metric("f1_prod", f1)

        # 6. Salva tabela de resultados
        df_out = df_prod.copy()
        df_out["y_proba"] = y_proba
        df_out["y_pred"] = y_pred

        
        mlflow.log_text(df_out.to_csv(index=False), "resultado_pipeline.csv")

    print(f"Pipeline concluído. logloss={ll:.4f}, f1={f1:.4f}")
    return df_out