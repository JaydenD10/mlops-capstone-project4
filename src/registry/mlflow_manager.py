"""
MLflow Tracking and Model Registry Management Module.
Handles runs, metric/param/artifact logging, model registration, stage/alias transitions,
and full lineage tracking.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib
import mlflow
from mlflow.tracking import MlflowClient

from config.schema import load_config, PipelineConfig
from config.logging_config import setup_logger

logger = setup_logger("mlflow_manager")


class MLflowManager:
    """
    Central wrapper around MLflow Client and Tracking APIs.
    """
    def __init__(self, config: PipelineConfig = None):
        self.config = config or load_config()
        self.tracking_uri = self.config.mlflow.tracking_uri
        self.experiment_name = self.config.mlflow.experiment_name
        self.model_name = self.config.mlflow.model_name

        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

        # Ensure experiment exists
        exp = self.client.get_experiment_by_name(self.experiment_name)
        if exp is None:
            self.experiment_id = self.client.create_experiment(self.experiment_name)
        else:
            self.experiment_id = exp.experiment_id

        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name: str, tags: Dict[str, str] = None) -> mlflow.ActiveRun:
        """Starts an MLflow run under the configured experiment."""
        return mlflow.start_run(
            run_name=run_name,
            experiment_id=self.experiment_id,
            tags=tags or {}
        )

    def log_metrics_and_params(
        self,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        tags: Dict[str, str] = None
    ):
        """Logs parameters, metrics, and tags to active run."""
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        if tags:
            mlflow.set_tags(tags)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Logs a local artifact file to the active run."""
        mlflow.log_artifact(local_path, artifact_path)

    def register_and_promote_model(
        self,
        run_id: str,
        model_artifact_path: str,
        stage: str = "Production"
    ) -> Any:
        """
        Registers model from run artifact into Model Registry and transitions to stage.
        """
        model_uri = f"runs:/{run_id}/{model_artifact_path}"
        logger.info(f"Registering model from {model_uri} into registry as '{self.model_name}'...")

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=self.model_name
        )

        version_num = model_version.version
        logger.info(f"Registered model '{self.model_name}' Version {version_num}")

        # Transition stage or set alias
        try:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version_num,
                stage=stage,
                archive_existing_versions=True
            )
            logger.info(f"Model '{self.model_name}' Version {version_num} transitioned to stage: {stage}")
        except Exception as e:
            logger.warning(f"Stage transition notice: {e}. Setting alias '{stage.lower()}'...")
            try:
                self.client.set_registered_model_alias(
                    name=self.model_name,
                    alias=stage.lower(),
                    version=version_num
                )
            except Exception as ex:
                logger.warning(f"Could not set alias: {ex}")

        return model_version

    def get_current_production_model(self) -> Optional[Tuple[str, Dict[str, float]]]:
        """
        Retrieves currently deployed (champion) production model version and its metrics.
        Returns (version, metrics_dict) or None if no production model exists yet.
        """
        try:
            # Check for versions in Production stage
            latest_versions = self.client.get_latest_versions(
                name=self.model_name,
                stages=["Production"]
            )
            if not latest_versions:
                # Try alias 'production'
                try:
                    mv = self.client.get_model_version_by_alias(self.model_name, "production")
                    latest_versions = [mv]
                except Exception:
                    pass

            if not latest_versions:
                return None

            prod_version = latest_versions[0]
            run = self.client.get_run(prod_version.run_id)
            metrics = run.data.metrics
            return prod_version.version, metrics
        except Exception as e:
            logger.info(f"No existing production model found in registry: {e}")
            return None

    def get_model_lineage(self, model_version_number: Optional[str] = None) -> Dict[str, Any]:
        """
        Traces complete end-to-end lineage for a registered model version:
        Model Version -> Run ID -> Git Commit -> Data Hash -> Config -> Metrics -> Parameters.
        """
        if model_version_number is None:
            prod_info = self.get_current_production_model()
            if prod_info:
                model_version_number = prod_info[0]
            else:
                # Get latest version
                versions = self.client.search_model_versions(f"name='{self.model_name}'")
                if not versions:
                    return {"status": "no_models_registered"}
                model_version_number = versions[0].version

        mv = self.client.get_model_version(self.model_name, model_version_number)
        run = self.client.get_run(mv.run_id)

        lineage = {
            "registered_model_name": self.model_name,
            "version": mv.version,
            "current_stage": mv.current_stage,
            "creation_timestamp": mv.creation_timestamp,
            "run_id": mv.run_id,
            "run_name": run.info.run_name,
            "git_commit": run.data.tags.get("git_commit", "N/A"),
            "data_train_hash": run.data.tags.get("data_train_hash", "N/A"),
            "data_raw_hash": run.data.tags.get("data_raw_hash", "N/A"),
            "config_version": run.data.tags.get("config_version", "1.0.0"),
            "pipeline_stage": run.data.tags.get("pipeline_stage", "production"),
            "metrics": run.data.metrics,
            "parameters": run.data.params,
            "artifact_uri": run.info.artifact_uri
        }
        return lineage


if __name__ == "__main__":
    mgr = MLflowManager()
    print(f"MLflow Manager initialized. Tracking URI: {mgr.tracking_uri}, Exp: {mgr.experiment_name}")
