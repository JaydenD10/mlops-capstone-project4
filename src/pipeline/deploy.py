"""
Model Registration and Deployment Stage.
Integrates Model Validation Gate, MLflow Model Registry promotion,
and atomic deployment of active serving artifacts.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib

from config.schema import load_config, PipelineConfig
from config.logging_config import setup_logger
from src.models.gate import evaluate_model_gate, ModelGateValidationError
from src.registry.mlflow_manager import MLflowManager

logger = setup_logger("deploy")


def get_git_commit() -> str:
    """Attempts to retrieve current git commit hash."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        return commit
    except Exception:
        return "git_commit_local_dev"


def register_and_deploy_model(
    config: PipelineConfig = None,
    mlflow_mgr: MLflowManager = None,
    force_promote: bool = False
) -> Dict[str, Any]:
    """
    Evaluates candidate model through Model Gate.
    If gate passes, registers model in MLflow Registry, transitions to Production,
    and updates active serving artifacts on disk atomically.
    If gate fails, cleanly exits leaving active model unchanged.
    """
    if config is None:
        config = load_config()
    if mlflow_mgr is None:
        mlflow_mgr = MLflowManager(config)

    logger.info("Starting Model Registration & Deployment stage...")

    model_dir = Path(config.paths.model_dir)
    candidate_model_path = model_dir / "candidate_model.joblib"
    summary_path = model_dir / "training_summary.json"

    if not candidate_model_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"Candidate model or training summary missing at {model_dir}")

    with open(summary_path, "r", encoding="utf-8") as f:
        training_summary = json.load(f)

    candidate_metrics = training_summary["best_metrics"]
    candidate_name = training_summary["best_model_name"]

    # 1. Evaluate Model Gate
    gate_result = evaluate_model_gate(candidate_metrics, config=config, mlflow_mgr=mlflow_mgr)

    if not gate_result.passed and not force_promote:
        logger.error("MODEL GATE REJECTED CANDIDATE MODEL. ABORTING DEPLOYMENT.")
        raise ModelGateValidationError(
            f"Candidate model '{candidate_name}' failed Model Gate checks.",
            gate_result.reasons
        )

    # 2. Log official release run in MLflow
    git_commit = get_git_commit()
    with open(Path(config.paths.processed_dir) / "ingestion_metadata.json", "r") as f:
        ingestion_meta = json.load(f)

    with mlflow_mgr.start_run(run_name=f"release_{candidate_name}") as run:
        run_id = run.info.run_id
        # Log candidate model with MLflow sklearn flavor
        candidate_model = joblib.load(candidate_model_path)
        import mlflow.sklearn
        mlflow.sklearn.log_model(candidate_model, artifact_path="model")
        
        mlflow_mgr.log_artifact(str(Path(config.paths.active_preprocessor_path)), artifact_path="preprocessor")
        mlflow_mgr.log_artifact(str(summary_path), artifact_path="metadata")

        # Tags for complete lineage
        release_tags = {
            "model_type": candidate_name,
            "git_commit": git_commit,
            "data_train_hash": ingestion_meta.get("train_hash", ""),
            "data_raw_hash": ingestion_meta.get("raw_train_hash", ""),
            "config_version": config.project.version,
            "pipeline_stage": "production_release",
            "is_new_champion": str(gate_result.is_new_champion)
        }
        mlflow_mgr.log_metrics_and_params(
            params={"model_type": candidate_name},
            metrics={
                "rmse": candidate_metrics.get("test_rmse", candidate_metrics.get("rmse", 0.0)),
                "r2": candidate_metrics.get("test_r2", candidate_metrics.get("r2", 0.0)),
                "mae": candidate_metrics.get("test_mae", candidate_metrics.get("mae", 0.0)),
                "mape": candidate_metrics.get("test_mape", candidate_metrics.get("mape", 0.0)),
            },
            tags=release_tags
        )

    # 3. Register model and transition to Production
    model_version = mlflow_mgr.register_and_promote_model(
        run_id=run_id,
        model_artifact_path="model",
        stage=config.mlflow.registered_stage
    )

    # 4. Atomically promote to active serving artifact on disk
    active_model_path = Path(config.paths.active_model_path)
    shutil.copy2(candidate_model_path, active_model_path)

    # Save active deployment metadata
    deploy_metadata = {
        "model_name": config.mlflow.model_name,
        "model_type": candidate_name,
        "model_version": str(model_version.version),
        "run_id": run_id,
        "git_commit": git_commit,
        "data_hash": ingestion_meta.get("train_hash", ""),
        "test_rmse": candidate_metrics.get("test_rmse"),
        "test_r2": candidate_metrics.get("test_r2"),
        "test_mae": candidate_metrics.get("test_mae"),
        "founder_summary": candidate_metrics.get("test_founder_summary", ""),
        "deployed_artifact": str(active_model_path)
    }

    deploy_meta_path = model_dir / "active_model_metadata.json"
    with open(deploy_meta_path, "w", encoding="utf-8") as f:
        json.dump(deploy_metadata, f, indent=2)

    logger.info(f"MODEL DEPLOYMENT SUCCESSFUL. Active serving model is now Version {model_version.version}.")
    return deploy_metadata


if __name__ == "__main__":
    meta = register_and_deploy_model()
    print(f"Deployment Completed. Active Model Version: {meta['model_version']}")
