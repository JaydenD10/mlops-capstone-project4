"""
Prefect Workflow Implementation for MLOps Pipeline (Task 3).
Defines Prefect @task components and @flow pipeline graph:
Ingest -> Validate Data -> Feature Engineering -> Training -> Evaluation -> Gate -> Registration/Deployment -> Monitoring.
"""

from typing import Dict, Any
from prefect import task, flow

from config.schema import load_config
from src.data.ingestion import ingest_data
from src.data.validation import validate_data_gate
from src.features.preprocessor import fit_and_save_preprocessor
from src.models.train import train_and_evaluate_models
from src.models.gate import evaluate_model_gate, ModelGateValidationError
from src.pipeline.deploy import register_and_deploy_model
from src.monitoring.drift import run_monitoring_pipeline
import pandas as pd


@task(name="Ingestion Task", retries=2, retry_delay_seconds=3)
def prefect_ingest(config) -> Dict[str, Any]:
    return ingest_data(config)


@task(name="Data Validation Gate Task")
def prefect_validate_data(config, ingest_meta: Dict[str, Any]) -> bool:
    res = validate_data_gate(config.paths.train_data, strict=True)
    return res.passed


@task(name="Feature Engineering Task")
def prefect_build_features(config, data_valid: bool) -> Dict[str, Any]:
    df_train = pd.read_csv(config.paths.train_data)
    pipe, X_trans = fit_and_save_preprocessor(df_train, config)
    return {"features_shape": X_trans.shape}


@task(name="Model Training & Exploration Task")
def prefect_train(config, feature_meta: Dict[str, Any]) -> Dict[str, Any]:
    model, metrics, summary = train_and_evaluate_models(config)
    return summary


@task(name="Model Validation Gate Task")
def prefect_model_gate(config, training_summary: Dict[str, Any]) -> Dict[str, Any]:
    gate_res = evaluate_model_gate(training_summary["best_metrics"], config=config)
    if not gate_res.passed:
        raise ModelGateValidationError("Candidate model rejected by Model Gate.", gate_res.reasons)
    return gate_res.model_dump()


@task(name="Model Registration & Deployment Task")
def prefect_deploy(config, gate_result: Dict[str, Any]) -> Dict[str, Any]:
    return register_and_deploy_model(config)


@task(name="Operational Drift Monitoring Task")
def prefect_monitor(config, deploy_meta: Dict[str, Any]) -> Dict[str, Any]:
    return run_monitoring_pipeline(config)


@flow(name="MLOps Capstone End-to-End Property Price Pipeline")
def capstone_mlops_flow():
    config = load_config()

    # Stage 1: Ingest
    ingest_meta = prefect_ingest(config)

    # Stage 2: Data Gate
    data_valid = prefect_validate_data(config, ingest_meta)

    # Stage 3: Feature Engineering
    features_meta = prefect_build_features(config, data_valid)

    # Stage 4: Training & Tuning
    train_summary = prefect_train(config, features_meta)

    # Stage 5: Model Gate
    gate_result = prefect_model_gate(config, train_summary)

    # Stage 6: Registration & Deployment
    deploy_meta = prefect_deploy(config, gate_result)

    # Stage 7: Operational Drift Monitoring
    monitoring_summary = prefect_monitor(config, deploy_meta)

    return {
        "status": "COMPLETED",
        "deployed_version": deploy_meta.get("model_version"),
        "monitoring_batches": monitoring_summary.get("batches_analyzed")
    }


if __name__ == "__main__":
    result = capstone_mlops_flow()
    print("Prefect Flow Execution Result:", result)
