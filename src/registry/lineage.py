"""
Lineage Demonstration Module (Task 5).
Traces the provenance of the currently deployed production model back to
its MLflow Run ID, Git Commit, Data Hashes, Config, and Metrics.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

from config.schema import load_config
from config.logging_config import setup_logger
from src.registry.mlflow_manager import MLflowManager

logger = setup_logger("lineage")


def get_live_lineage(model_version: str = None) -> Dict[str, Any]:
    """
    Extracts live lineage for the deployed production model from MLflow registry.
    """
    config = load_config()
    mlflow_mgr = MLflowManager(config)

    # Read active deployment metadata from disk as well
    active_meta_file = Path(config.paths.model_dir) / "active_model_metadata.json"
    disk_meta = {}
    if active_meta_file.exists():
        with open(active_meta_file, "r", encoding="utf-8") as f:
            disk_meta = json.load(f)

    lineage = mlflow_mgr.get_model_lineage(model_version)
    lineage["disk_metadata"] = disk_meta

    return lineage


def print_lineage_tree(lineage: Dict[str, Any]):
    """Prints a clear, formatted lineage tree for founders and auditors."""
    print("=" * 80)
    print(" MLOPS PROJECT 4 - END-TO-END MODEL LINEAGE TRACE")
    print("=" * 80)
    print(f" Registered Model Name : {lineage.get('registered_model_name', 'HousePricePredictor')}")
    print(f" Deployed Model Version: v{lineage.get('version', '1')}")
    print(f" Stage Status          : {lineage.get('current_stage', 'Production')}")
    print(f" MLflow Run ID         : {lineage.get('run_id')}")
    print(f" MLflow Run Name       : {lineage.get('run_name')}")
    print("-" * 80)
    print(" PROVENANCE & ARTIFACT TRACEABILITY:")
    print(f" [1] Code Version (Git): {lineage.get('git_commit')}")
    print(f" [2] Pipeline Config   : v{lineage.get('config_version')}")
    print(f" [3] Train Data SHA-256: {lineage.get('data_train_hash')}")
    print(f" [4] Raw Data SHA-256  : {lineage.get('data_raw_hash')}")
    print("-" * 80)
    print(" VERIFIED EVALUATION METRICS:")
    metrics = lineage.get("metrics", {})
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"   - {k:<20}: {v:.4f}")
        else:
            print(f"   - {k:<20}: {v}")
    print("-" * 80)
    print(" MODEL ARTIFACT STORAGE:")
    print(f"   - MLflow Artifact URI: {lineage.get('artifact_uri')}")
    print(f"   - Active Disk Path   : {lineage.get('disk_metadata', {}).get('deployed_artifact', 'models/active_model.joblib')}")
    print("=" * 80)


if __name__ == "__main__":
    v_arg = sys.argv[1] if len(sys.argv) > 1 else None
    lin = get_live_lineage(v_arg)
    print_lineage_tree(lin)
