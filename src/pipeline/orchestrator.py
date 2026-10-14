"""
End-to-End Pipeline Orchestrator (Task 3).
Executes the full automated workflow graph:
Ingest -> Validate Data -> Build Features -> Train & Tune -> Model Gate -> Register & Deploy -> Monitor.
Maintains execution history, stage-level logging, and safe failure termination.
"""

import datetime
import json
import time
from pathlib import Path
from typing import Dict, Any, List

from config.schema import load_config, PipelineConfig
from config.logging_config import setup_logger
from src.data.ingestion import ingest_data
from src.data.validation import validate_data_gate, DataGateValidationError
from src.features.preprocessor import fit_and_save_preprocessor
from src.models.train import train_and_evaluate_models
from src.models.gate import evaluate_model_gate, ModelGateValidationError
from src.pipeline.deploy import register_and_deploy_model
from src.monitoring.drift import run_monitoring_pipeline
from src.registry.mlflow_manager import MLflowManager

logger = setup_logger("orchestrator")


class PipelineExecutionReport:
    """Tracks stage execution times and status across pipeline runs."""
    def __init__(self):
        self.start_time = time.time()
        self.stages: List[Dict[str, Any]] = []
        self.status = "INITIALIZING"

    def record_stage(self, stage_name: str, duration: float, success: bool, details: Any = None):
        self.stages.append({
            "stage": stage_name,
            "duration_seconds": round(duration, 3),
            "success": success,
            "timestamp": datetime.datetime.now().isoformat(),
            "details": details
        })

    def complete(self, status: str = "SUCCESS"):
        self.status = status
        self.total_duration = round(time.time() - self.start_time, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "total_duration_seconds": getattr(self, "total_duration", round(time.time() - self.start_time, 3)),
            "completed_at": datetime.datetime.now().isoformat(),
            "stages": self.stages
        }


def run_full_pipeline(
    config: PipelineConfig = None,
    stop_on_gate_failure: bool = True
) -> PipelineExecutionReport:
    """
    Executes the automated end-to-end MLOps pipeline.
    Stops cleanly and safely if any gate fails.
    """
    if config is None:
        config = load_config()

    report = PipelineExecutionReport()
    logger.info("================================================================================")
    logger.info(f"STARTING AUTOMATED END-TO-END PIPELINE: {config.project.name} v{config.project.version}")
    logger.info("================================================================================")

    # 1. Stage: INGEST
    t0 = time.time()
    try:
        logger.info("[STAGE 1/7] Ingesting official datasets and generating splits...")
        ingest_res = ingest_data(config)
        report.record_stage("ingest", time.time() - t0, True, {"train_rows": ingest_res["train_rows"]})
    except Exception as e:
        logger.error(f"[STAGE 1 FAILED] Ingestion failed: {e}")
        report.record_stage("ingest", time.time() - t0, False, str(e))
        report.complete("FAILED_INGESTION")
        return report

    # 2. Stage: DATA VALIDATION GATE
    t0 = time.time()
    try:
        logger.info("[STAGE 2/7] Running Data Validation Gate on training data...")
        gate_res = validate_data_gate(config.paths.train_data, strict=True)
        report.record_stage("validate_data", time.time() - t0, True, {"records": gate_res.total_records})
    except DataGateValidationError as e:
        logger.error(f"[STAGE 2 FAILED] DATA GATE FAILED: {e}")
        report.record_stage("validate_data", time.time() - t0, False, e.errors)
        report.complete("FAILED_DATA_GATE")
        if stop_on_gate_failure:
            return report

    # 3. Stage: BUILD FEATURES & PREPROCESS
    t0 = time.time()
    try:
        logger.info("[STAGE 3/7] Building features and fitting preprocessor...")
        import pandas as pd
        df_train = pd.read_csv(config.paths.train_data)
        pipe, X_trans = fit_and_save_preprocessor(df_train, config)
        report.record_stage("build_features", time.time() - t0, True, {"feature_count": X_trans.shape[1]})
    except Exception as e:
        logger.error(f"[STAGE 3 FAILED] Preprocessing failed: {e}")
        report.record_stage("build_features", time.time() - t0, False, str(e))
        report.complete("FAILED_PREPROCESSING")
        return report

    # 4. Stage: TRAIN & TUNE
    t0 = time.time()
    try:
        logger.info("[STAGE 4/7] Multi-model exploration, CV, and hyperparameter evaluation...")
        mlflow_mgr = MLflowManager(config)
        best_model, best_metrics, summary = train_and_evaluate_models(config, mlflow_mgr)
        report.record_stage("train", time.time() - t0, True, {
            "best_model": best_metrics["model_name"],
            "val_rmse": best_metrics["rmse"],
            "test_r2": best_metrics["test_r2"]
        })
    except Exception as e:
        logger.error(f"[STAGE 4 FAILED] Training failed: {e}")
        report.record_stage("train", time.time() - t0, False, str(e))
        report.complete("FAILED_TRAINING")
        return report

    # 5. Stage: MODEL VALIDATION GATE
    t0 = time.time()
    try:
        logger.info("[STAGE 5/7] Evaluating Candidate Model through Model Gate...")
        m_gate_res = evaluate_model_gate(best_metrics, config=config, mlflow_mgr=mlflow_mgr)
        if not m_gate_res.passed:
            raise ModelGateValidationError("Model Gate rejected candidate.", m_gate_res.reasons)
        report.record_stage("validate_model", time.time() - t0, True, {"passed": True})
    except ModelGateValidationError as e:
        logger.error(f"[STAGE 5 FAILED] MODEL GATE FAILED: {e}")
        report.record_stage("validate_model", time.time() - t0, False, e.reasons)
        report.complete("FAILED_MODEL_GATE")
        if stop_on_gate_failure:
            return report

    # 6. Stage: REGISTER & DEPLOY
    t0 = time.time()
    try:
        logger.info("[STAGE 6/7] Promoting candidate to Production in Model Registry...")
        deploy_meta = register_and_deploy_model(config, mlflow_mgr, force_promote=False)
        report.record_stage("deploy", time.time() - t0, True, {"version": deploy_meta["model_version"]})
    except Exception as e:
        logger.error(f"[STAGE 6 FAILED] Deployment failed: {e}")
        report.record_stage("deploy", time.time() - t0, False, str(e))
        report.complete("FAILED_DEPLOYMENT")
        return report

    # 7. Stage: MONITOR OPERATIONAL BATCHES
    t0 = time.time()
    try:
        logger.info("[STAGE 7/7] Ingesting operational batches and running drift monitoring...")
        monitoring_summary = run_monitoring_pipeline(config)
        report.record_stage("monitor", time.time() - t0, True, {"batch_statuses": monitoring_summary["batch_statuses"]})
    except Exception as e:
        logger.error(f"[STAGE 7 FAILED] Monitoring failed: {e}")
        report.record_stage("monitor", time.time() - t0, False, str(e))
        report.complete("FAILED_MONITORING")
        return report

    report.complete("SUCCESS")
    logger.info("================================================================================")
    logger.info(f"END-TO-END PIPELINE COMPLETED SUCCESSFULLY IN {report.total_duration}s")
    logger.info("================================================================================")

    # Save run execution history
    history_file = Path(config.paths.reports_dir) / "pipeline_run_history.json"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    history_records = []
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history_records = json.load(f)
        except Exception:
            history_records = []

    history_records.append(report.to_dict())
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history_records, f, indent=2)

    return report


if __name__ == "__main__":
    rep = run_full_pipeline()
    print("Pipeline Execution Summary:\n", json.dumps(rep.to_dict(), indent=2))
