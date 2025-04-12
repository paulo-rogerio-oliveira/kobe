import pandas as pd
import mlflow
from kedro.framework.session import KedroSession

import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from pycaret.classification import setup, create_model, predict_model, save_model, compare_models
from sklearn.metrics import log_loss, f1_score
import os

def PreparacaoDados(
    df_dev: pd.DataFrame,
    df_prod: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[pd.DataFrame,  pd.DataFrame]:
    """Carrega, limpa, seleciona colunas, divide e retorna (filtered, train, test)."""
    # Inicia run MLflow
    with mlflow.start_run(run_name="PreparacaoDados", nested=True):
        # Concatena e limpa
        df = pd.concat([df_dev], ignore_index=True).dropna()

        # Seleciona colunas
        cols = [
            "lat", "lon", "minutes_remaining", 
            "period", "playoffs", "shot_distance", 
            "shot_made_flag"
        ]
        df = df[cols]

        df_processed_prod = pd.concat([df_prod], ignore_index=True).dropna()

        # Seleciona colunas
        cols = [
            "lat", "lon", "minutes_remaining", 
            "period", "playoffs", "shot_distance", 
            "shot_made_flag"
        ]
        df_processed_prod = df_processed_prod[cols]

        # Log dimensão
        mlflow.log_metric("filtered_rows", df.shape[0])
        mlflow.log_metric("filtered_cols", df.shape[1])

        # Divide estratificado
        train, test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["shot_made_flag"]
        )

        # Log params e métricas
        mlflow.log_param("test_size", test_size)
        mlflow.log_metric("train_rows", train.shape[0])
        mlflow.log_metric("test_rows", test.shape[0])

        return df, train, test, df_processed_prod



def Treinamento(
    df_train,
    df_test,
    target: str = "shot_made_flag",
    modelo_saida: str = "modelo_final"
):
    """Treina dois modelos com PyCaret, registra métricas no MLflow e retorna o melhor."""
    # Inicia a run no MLflow
    with mlflow.start_run(run_name="Treinamento", nested=True):
        # 1. Setup PyCaret
        exp = setup(
            data=df_train,
            target=target,
            session_id=42,
            normalize=True,            
            log_experiment=False,  # gerencia logs via mlflow diretamente
        )


        # Treinamento
        lr = create_model("lr")
        dt = create_model("dt")
        
        # 3. Avaliação na base de teste
        # Previsões
        pred_lr = predict_model(lr, data=df_test)
        pred_dt = predict_model(dt, data=df_test)
        
        # PyCaret adiciona coluna "Label" e "Score"
        y_true = df_test[target].values
        y_proba_lr = pred_lr["prediction_score"].values
        y_proba_dt = pred_dt["prediction_score"].values
        y_pred_dt = pred_dt["prediction_label"].values
        y_pred_lr = pred_lr["prediction_label"].values
        
        # Métricas
        ll_lr = log_loss(y_true, y_proba_lr)
        ll_dt = log_loss(y_true, y_proba_dt)
        f1_dt = f1_score(y_true, y_pred_dt)
        f1_lr = f1_score(y_true, y_pred_lr)
        
        # 4. Log das métricas
        mlflow.log_metric("logloss_lr", ll_lr)
        mlflow.log_metric("logloss_dt", ll_dt)
        mlflow.log_metric("f1_dt", f1_dt)
        mlflow.log_metric("f1_lr", f1_lr)
        
        # 5. Seleção do modelo final
        # Critério: menor log loss; em caso de empate, escolha aquele com maior F1
        if ll_lr < ll_dt:
            best_model, best_name = lr, "lr"
        elif ll_dt < ll_lr:
            best_model, best_name = dt, "dt"
        else:
            # empataram em logloss → escolher dt se F1 maior, senão lr
            best_model, best_name = (dt, "dt") if f1_dt > 0 else (lr, "lr")
        
        mlflow.log_param("modelo_escolhido", best_name)
        mlflow.log_artifact(os.path.abspath("./data/06_models/modelo_final.pickle"))
          # Log the sklearn model and register as version 1
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name="trained_model"
        )
        return best_model

