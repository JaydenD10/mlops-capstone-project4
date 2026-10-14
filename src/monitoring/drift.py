"""
Data and Performance Drift Detection Engine (Task 7).
Calculates Population Stability Index (PSI), Kolmogorov-Smirnov test,
Wasserstein distance, and ground-truth performance degradation across incoming batches.
Generates comprehensive JSON reports and alert status.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from config.schema import load_config, PipelineConfig
from config.logging_config import setup_logger
from src.features.preprocessor import transform_features
from src.models.evaluate import evaluate_predictions

logger = setup_logger("drift_monitor")


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4
) -> float:
    """
    Computes Population Stability Index (PSI) between reference and current distribution.
    PSI < 0.10: Stable, no drift
    0.10 <= PSI < 0.25: Moderate drift / Warning
    PSI >= 0.25: Significant drift / Critical Alert
    """
    ref = reference[~np.isnan(reference)]
    curr = current[~np.isnan(current)]

    if len(ref) == 0 or len(curr) == 0:
        return 0.0

    # Determine quantile bins based on reference
    quantiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(ref, quantiles)
    bins[0] -= 1e-5
    bins[-1] += 1e-5
    bins = np.unique(bins)

    if len(bins) < 2:
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=bins)
    curr_counts, _ = np.histogram(curr, bins=bins)

    ref_pct = (ref_counts + epsilon) / (len(ref) + epsilon * len(ref_counts))
    curr_pct = (curr_counts + epsilon) / (len(curr) + epsilon * len(curr_counts))

    psi_val = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(max(0.0, psi_val))


def calculate_categorical_psi(
    reference: pd.Series,
    current: pd.Series,
    epsilon: float = 1e-4
) -> float:
    """Computes PSI across categorical categories."""
    categories = list(set(reference.dropna().unique()).union(set(current.dropna().unique())))
    if not categories:
        return 0.0

    ref_counts = reference.value_counts()
    curr_counts = current.value_counts()

    ref_total = len(reference.dropna()) + epsilon * len(categories)
    curr_total = len(current.dropna()) + epsilon * len(categories)

    psi_val = 0.0
    for cat in categories:
        ref_p = (ref_counts.get(cat, 0) + epsilon) / ref_total
        curr_p = (curr_counts.get(cat, 0) + epsilon) / curr_total
        psi_val += (curr_p - ref_p) * np.log(curr_p / ref_p)

    return float(max(0.0, psi_val))


def evaluate_batch_drift(
    batch_df: pd.DataFrame,
    ref_train_df: pd.DataFrame,
    batch_id: int,
    config: PipelineConfig = None
) -> Dict[str, Any]:
    """
    Analyzes Data Drift and Performance Drift for a single operational batch.
    """
    if config is None:
        config = load_config()

    gate_cfg = config.monitoring
    feat_cfg = config.features

    # 1. Feature Data Drift
    feature_drift: Dict[str, Any] = {}
    total_drifted_features = 0

    # Numerical features
    for col in feat_cfg.numerical:
        if col in batch_df.columns and col in ref_train_df.columns:
            ref_vals = ref_train_df[col].dropna().values
            curr_vals = batch_df[col].dropna().values

            psi = calculate_psi(ref_vals, curr_vals)
            ks_stat, ks_pvalue = ks_2samp(ref_vals, curr_vals)
            w_dist = float(wasserstein_distance(ref_vals, curr_vals))

            is_drifted = (psi >= gate_cfg.psi_alert_threshold) or (ks_pvalue < gate_cfg.ks_p_value_threshold)
            if is_drifted:
                total_drifted_features += 1

            feature_drift[col] = {
                "type": "numerical",
                "psi": round(psi, 4),
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_pvalue), 6),
                "wasserstein_distance": round(w_dist, 4),
                "drift_detected": bool(is_drifted),
                "ref_mean": round(float(np.mean(ref_vals)), 2),
                "curr_mean": round(float(np.mean(curr_vals)), 2),
                "ref_std": round(float(np.std(ref_vals)), 2),
                "curr_std": round(float(np.std(curr_vals)), 2),
            }

    # Categorical features
    for col in feat_cfg.categorical:
        if col in batch_df.columns and col in ref_train_df.columns:
            psi = calculate_categorical_psi(ref_train_df[col], batch_df[col])
            is_drifted = psi >= gate_cfg.psi_alert_threshold
            if is_drifted:
                total_drifted_features += 1

            feature_drift[col] = {
                "type": "categorical",
                "psi": round(psi, 4),
                "drift_detected": bool(is_drifted),
                "ref_distribution": {k: round(v, 4) for k, v in ref_train_df[col].value_counts(normalize=True).items()},
                "curr_distribution": {k: round(v, 4) for k, v in batch_df[col].value_counts(normalize=True).items()}
            }

    # 2. Performance Drift (since ground truth Price_Lakhs is provided in batches)
    model_path = Path(config.paths.active_model_path)
    prep_path = Path(config.paths.active_preprocessor_path)

    perf_metrics = {}
    perf_degradation = 0.0

    if model_path.exists() and prep_path.exists() and feat_cfg.target in batch_df.columns:
        model = joblib.load(model_path)
        preprocessor = joblib.load(prep_path)

        X_batch = transform_features(batch_df, preprocessor, config)
        y_true = batch_df[feat_cfg.target].values
        y_pred = model.predict(X_batch)

        batch_eval = evaluate_predictions(y_true, y_pred)

        # Compare against baseline test metrics from active model metadata
        meta_file = Path(config.paths.model_dir) / "active_model_metadata.json"
        baseline_rmse = 7.74  # default reference
        if meta_file.exists():
            with open(meta_file, "r") as f:
                meta = json.load(f)
                baseline_rmse = float(meta.get("test_rmse", baseline_rmse))

        current_rmse = batch_eval["rmse"]
        perf_degradation = ((current_rmse - baseline_rmse) / baseline_rmse) * 100.0

        perf_metrics = {
            "baseline_rmse": round(baseline_rmse, 2),
            "current_rmse": round(current_rmse, 2),
            "current_mae": round(batch_eval["mae"], 2),
            "current_r2": round(batch_eval["r2"], 4),
            "current_mape": round(batch_eval["mape"], 4),
            "within_5_lakhs_pct": round(batch_eval["within_5_lakhs_pct"], 1),
            "within_10_percent_pct": round(batch_eval["within_10_percent_pct"], 1),
            "rmse_degradation_pct": round(perf_degradation, 2)
        }

    # 3. Alert Status Determination
    # Alert conditions:
    # Critical: Performance degradation > 15% OR >= 3 drifted features OR key feature PSI >= 0.25
    # Warning: 1-2 drifted features OR performance degradation between 5% and 15%
    # Stable: otherwise
    critical_triggers = []
    warning_triggers = []

    if perf_degradation > gate_cfg.max_performance_drop_pct:
        critical_triggers.append(
            f"Performance RMSE degradation {perf_degradation:.1f}% exceeds threshold {gate_cfg.max_performance_drop_pct}%"
        )
    elif perf_degradation > 5.0:
        warning_triggers.append(f"Minor performance degradation of {perf_degradation:.1f}% detected")

    for feat, data in feature_drift.items():
        psi = data["psi"]
        if psi >= gate_cfg.psi_alert_threshold:
            critical_triggers.append(f"Critical data drift on '{feat}' (PSI = {psi:.3f} >= {gate_cfg.psi_alert_threshold})")
        elif psi >= gate_cfg.psi_warning_threshold:
            warning_triggers.append(f"Moderate data drift warning on '{feat}' (PSI = {psi:.3f})")

    if len(critical_triggers) > 0:
        alert_level = "CRITICAL"
        recommended_action = "TRIGGER_AUTOMATED_RETRAINING_PIPELINE: Substantial concept/covariate drift detected."
    elif len(warning_triggers) > 0:
        alert_level = "WARNING"
        recommended_action = "MONITOR_CLOSELY: Moderate drift observed; queue for next scheduled retrain."
    else:
        alert_level = "HEALTHY"
        recommended_action = "NO_ACTION_REQUIRED: Distributions and model accuracy match historical baseline."

    report = {
        "batch_id": batch_id,
        "sample_count": len(batch_df),
        "alert_level": alert_level,
        "critical_triggers": critical_triggers,
        "warning_triggers": warning_triggers,
        "recommended_action": recommended_action,
        "total_drifted_features": total_drifted_features,
        "performance_metrics": perf_metrics,
        "feature_drift": feature_drift
    }

    return report


def run_monitoring_pipeline(config: PipelineConfig = None) -> Dict[str, Any]:
    """
    Processes all 3 incoming operational batches from capstone_incoming_batches.csv.
    Generates batch reports and aggregate monitoring summary.
    """
    if config is None:
        config = load_config()

    logger.info("Starting Operational Batch Monitoring & Drift Pipeline...")

    train_path = Path(config.paths.train_data)
    batches_path = Path(config.paths.raw_incoming_batches)
    drift_dir = Path(config.paths.drift_dir)
    drift_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        from src.data.ingestion import ingest_data
        ingest_data(config)

    df_train = pd.read_csv(train_path)
    df_batches = pd.read_csv(batches_path)

    batch_col = config.monitoring.batch_column
    unique_batches = sorted(df_batches[batch_col].unique())
    logger.info(f"Processing incoming batches: {unique_batches}")

    all_reports = {}
    for b_id in unique_batches:
        sub_batch = df_batches[df_batches[batch_col] == b_id].copy()
        report = evaluate_batch_drift(sub_batch, df_train, int(b_id), config)
        all_reports[f"batch_{b_id}"] = report

        # Save individual batch drift report
        b_file = drift_dir / f"drift_report_batch_{b_id}.json"
        with open(b_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Batch {b_id} Evaluation: [{report['alert_level']}] - Action: {report['recommended_action']}")
        if report["critical_triggers"]:
            for trig in report["critical_triggers"]:
                logger.warning(f"  [!] CRITICAL: {trig}")
        if report["warning_triggers"]:
            for trig in report["warning_triggers"]:
                logger.info(f"  [*] WARNING: {trig}")

    # Summary report
    summary = {
        "status": "completed",
        "batches_analyzed": [int(b) for b in unique_batches],
        "batch_statuses": {f"batch_{b_id}": all_reports[f"batch_{b_id}"]["alert_level"] for b_id in unique_batches},
        "reports": all_reports
    }
    summary_file = drift_dir / "drift_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"All monitoring reports successfully saved to: {drift_dir.resolve()}")
    return summary


if __name__ == "__main__":
    res = run_monitoring_pipeline()
    print("Monitoring Pipeline Run Complete. Batch Statuses:", res["batch_statuses"])
