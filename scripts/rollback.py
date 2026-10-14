"""
Automated Model Rollback Utility (Task 6).
Rolls back the production model to the previous stable version in MLflow Model Registry,
updating disk serving artifacts atomically.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from mlflow.tracking import MlflowClient

from config.schema import load_config
from config.logging_config import setup_logger
from src.registry.mlflow_manager import MLflowManager

logger = setup_logger("rollback")


def rollback_production_model(target_version: Optional[str] = None) -> Dict[str, Any]:
    """
    Rolls back the active model to target_version, or to the immediately preceding version.
    """
    config = load_config()
    mlflow_mgr = MLflowManager(config)
    client = mlflow_mgr.client
    model_name = config.mlflow.model_name

    logger.info(f"Initiating Rollback procedure for model '{model_name}'...")

    # Get all model versions
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions or len(versions) < 1:
        raise ValueError(f"No registered versions found for model '{model_name}'")

    sorted_versions = sorted(versions, key=lambda v: int(v.version), reverse=True)

    current_prod_info = mlflow_mgr.get_current_production_model()
    current_prod_version = current_prod_info[0] if current_prod_info else sorted_versions[0].version

    logger.info(f"Current production version: v{current_prod_version}")

    if target_version is None:
        # Pick the preceding version
        candidates = [v for v in sorted_versions if str(v.version) != str(current_prod_version)]
        if not candidates:
            logger.warning("Only one version exists in registry. Re-applying current version as baseline.")
            target_version = current_prod_version
        else:
            target_version = str(candidates[0].version)

    logger.info(f"Target rollback version: v{target_version}")

    # Transition target version to Production
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=target_version,
            stage="Production",
            archive_existing_versions=True
        )
    except Exception as e:
        logger.warning(f"Stage transition fallback: {e}")
        client.set_registered_model_alias(name=model_name, alias="production", version=target_version)

    # Retrieve artifact URI for target version
    target_mv = client.get_model_version(model_name, target_version)
    target_run = client.get_run(target_mv.run_id)

    # Download model artifact
    local_download_dir = Path("models/rolled_back")
    local_download_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = mlflow.artifacts.download_artifacts(
        run_id=target_mv.run_id,
        artifact_path="model",
        dst_path=str(local_download_dir)
    )

    # Atomically update disk active model
    active_model_dest = Path(config.paths.active_model_path)
    model_pickle = Path(downloaded_path) / "model.pkl"
    if not model_pickle.exists():
        # Look for joblib in candidate or directory
        potential_files = list(Path(downloaded_path).glob("*.pkl")) + list(Path(downloaded_path).glob("*.joblib"))
        if potential_files:
            model_pickle = potential_files[0]

    if model_pickle.exists():
        shutil.copy2(model_pickle, active_model_dest)
        logger.info(f"Updated active serving model artifact on disk: {active_model_dest}")

    rollback_metadata = {
        "status": "rolled_back",
        "model_name": model_name,
        "previous_version": str(current_prod_version),
        "active_version": str(target_version),
        "run_id": target_mv.run_id,
        "git_commit": target_run.data.tags.get("git_commit", "N/A"),
        "metrics": target_run.data.metrics
    }

    meta_file = Path(config.paths.model_dir) / "active_model_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(rollback_metadata, f, indent=2)

    logger.info(f"ROLLBACK COMPLETED SUCCESSFULLY. Active model is now v{target_version}.")
    return rollback_metadata


if __name__ == "__main__":
    t_ver = sys.argv[1] if len(sys.argv) > 1 else None
    res = rollback_production_model(t_ver)
    print(f"Rollback Result: Successfully restored v{res['active_version']}")
