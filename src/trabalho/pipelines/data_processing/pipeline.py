from kedro.pipeline import Pipeline, node, pipeline

from .nodes import PreparacaoDados, Treinamento
from .aplicacao import PipelineAplicacao


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=PreparacaoDados,
                inputs=["dataset_kobe_dev", "dataset_kobe_prod"],
                outputs=["data_filtered", "base_train", "base_test", "data_processed_prod"],
                name="PreparacaoDados",
                tags=["mlflow"]
            ),
            node(
                func=Treinamento,
                inputs=["base_train", "base_test"],
                outputs="modelo_final",
                name="Treinamento",
                tags=["mlflow"]
            ),
            node(
                func=PipelineAplicacao,
                inputs=["data_processed_prod", "modelo_final"],
                outputs="resultado_pipeline",
                name="PipelineAplicacao",
                tags=["mlflow"]
            )
           
        ]
    )
