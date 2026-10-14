"""
Model Validation Gate for MLOps Pipeline (Task 4).
Applies absolute quality thresholds and Champion vs Challenger regression comparison.
Fails safely, protecting existing production models from degradation.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from config.schema import load_config, PipelineConfig
from config.logging_config import setup_logger
from src.registry.mlflow_manager import MLflowManager

logger = setup_logger("model_gate")


class ModelGateValidationError(Exception):
    """Raised when candidate model fails validation gate thresholds."""
    def __init__(self, message: str, reasons: List[str]):
        super().__init__(message)
        self.reasons = reasons


class ModelGateResult(BaseModel):
    passed: bool
    candidate_metrics: Dict[str, Any]
    champion_metrics: Optional[Dict[str, Any]] = None
    champion_version: Optional[str] = None
    reasons: List[str]
    is_new_champion: bool


def evaluate_model_gate(
    candidate_metrics: Dict[str, float],
    config: PipelineConfig = None,
    mlflow_mgr: Optional[MLflowManager] = None
) -> ModelGateResult:
    """
    Evaluates candidate model metrics against:
    1. Absolute quality thresholds (min R2, max RMSE, max MAE, max MAPE).
    2. Champion vs Challenger comparison (candidate must equal or beat active production model).
    """
    if config is None:
        config = load_config()
    if mlflow_mgr is None:
        mlflow_mgr = MLflowManager(config)

    gate_cfg = config.model_gate
    reasons: List[str] = []

    c_r2 = float(candidate_metrics.get("r2", -999.0))
    c_rmse = float(candidate_metrics.get("rmse", 999.0))
    c_mae = float(candidate_metrics.get("mae", 999.0))
    c_mape = float(candidate_metrics.get("mape", 999.0))

    logger.info("Evaluating Candidate Model against Absolute Quality Thresholds:")
    logger.info(f"  Candidate R²:   {c_r2:.4f} (Required >= {gate_cfg.min_r2_score})")
    logger.info(f"  Candidate RMSE: {c_rmse:.2f} Lakhs (Required <= {gate_cfg.max_rmse_lakhs} Lakhs)")
    logger.info(f"  Candidate MAE:  {c_mae:.2f} Lakhs (Required <= {gate_cfg.max_mae_lakhs} Lakhs)")
    logger.info(f"  Candidate MAPE: {c_mape:.2%} (Required <= {gate_cfg.max_mape:.2%})")

    # 1. Absolute Threshold Checks
    if c_r2 < gate_cfg.min_r2_score:
        reasons.append(
            f"Candidate R² ({c_r2:.4f}) is below minimum quality threshold {gate_cfg.min_r2_score:.4f}"
        )
    if c_rmse > gate_cfg.max_rmse_lakhs:
        reasons.append(
            f"Candidate RMSE ({c_rmse:.2f} Lakhs) exceeds maximum allowed threshold {gate_cfg.max_rmse_lakhs:.2f} Lakhs"
        )
    if c_mae > gate_cfg.max_mae_lakhs:
        reasons.append(
            f"Candidate MAE ({c_mae:.2f} Lakhs) exceeds maximum allowed threshold {gate_cfg.max_mae_lakhs:.2f} Lakhs"
        )
    if c_mape > gate_cfg.max_mape:
        reasons.append(
            f"Candidate MAPE ({c_mape:.2%}) exceeds maximum allowed threshold {gate_cfg.max_mape:.2%}"
        )

    # 2. Champion vs Challenger Comparison
    champion_info = mlflow_mgr.get_current_production_model()
    champion_version = None
    champion_metrics = None
    is_new_champion = False

    if champion_info is not None:
        champion_version, champion_metrics = champion_info
        champ_rmse = float(champion_metrics.get("val_rmse", champion_metrics.get("rmse", 999.0)))
        champ_r2 = float(champion_metrics.get("val_r2", champion_metrics.get("r2", -999.0)))

        logger.info(f"Champion (v{champion_version}) metrics: RMSE={champ_rmse:.2f}, R²={champ_r2:.4f}")

        if gate_cfg.challenger_must_beat_champion:
            if c_rmse > champ_rmse:
                reasons.append(
                    f"Challenger RMSE ({c_rmse:.2f}) failed to beat Champion v{champion_version} RMSE ({champ_rmse:.2f})"
                )
            else:
                is_new_champion = True
    else:
        logger.info("No existing champion model in production registry. Candidate will establish baseline if passed.")
        is_new_champion = True

    passed = len(reasons) == 0
    if passed:
        logger.info("MODEL GATE PASSED: Candidate meets all quality and competitive thresholds.")
    else:
        logger.error(f"MODEL GATE FAILED: {len(reasons)} threshold violation(s):")
        for r in reasons:
            logger.error(f"  - {r}")

    return ModelGateResult(
        passed=passed,
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_metrics,
        champion_version=str(champion_version) if champion_version is not None else None,
        reasons=reasons,
        is_new_champion=is_new_champion
    )
