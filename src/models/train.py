"""
Model Training and Hyperparameter Exploration Stage for MLOps Pipeline.
Trains multiple candidate algorithms (Ridge, RandomForest, GradientBoosting, LightGBM),
performs k-fold cross-validation, tracks runs in MLflow, and selects the optimal model.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
import lightgbm as lgb

from config.schema import load_config, PipelineConfig
from config.logging_config import setup_logger
from src.features.preprocessor import fit_and_save_preprocessor, transform_features
from src.models.evaluate import evaluate_predictions
from src.registry.mlflow_manager import MLflowManager

logger = setup_logger("train")


def build_models(config: PipelineConfig) -> Dict[str, Any]:
    """Instantiates candidate models using hyperparameters from config."""
    hp = config.training.hyperparameters
    models = {}

    if "ridge" in config.training.models_to_evaluate:
        models["ridge"] = Ridge(**hp.get("ridge", {"alpha": 10.0}))

    if "random_forest" in config.training.models_to_evaluate:
        rf_params = hp.get("random_forest", {
            "n_estimators": 150, "max_depth": 12, "min_samples_split": 5, "random_state": 42
        }).copy()
        models["random_forest"] = RandomForestRegressor(**rf_params)

    if "gradient_boosting" in config.training.models_to_evaluate:
        gb_params = hp.get("gradient_boosting", {
            "n_estimators": 150, "learning_rate": 0.08, "max_depth": 5, "random_state": 42
        }).copy()
        models["gradient_boosting"] = GradientBoostingRegressor(**gb_params)

    if "lightgbm" in config.training.models_to_evaluate:
        lgb_params = hp.get("lightgbm", {
            "n_estimators": 150, "learning_rate": 0.07, "max_depth": 6, "num_leaves": 31, "random_state": 42, "verbose": -1
        }).copy()
        models["lightgbm"] = lgb.LGBMRegressor(**lgb_params)

    return models


def train_and_evaluate_models(
    config: PipelineConfig = None,
    mlflow_mgr: MLflowManager = None
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Trains and cross-validates all candidate models.
    Selects the best performing model based on CV RMSE.
    Evaluates on validation set and logs experiments to MLflow.
    Returns (best_model, best_metrics, model_comparison_summary).
    """
    if config is None:
        config = load_config()
    if mlflow_mgr is None:
        mlflow_mgr = MLflowManager(config)

    logger.info("Starting Model Training & Exploration stage...")

    # Load train and validation data
    train_path = Path(config.paths.train_data)
    val_path = Path(config.paths.val_data)
    test_path = Path(config.paths.test_data)

    if not train_path.exists() or not val_path.exists():
        from src.data.ingestion import ingest_data
        ingest_data(config)

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    target_col = config.features.target
    y_train = df_train[target_col].values
    y_val = df_val[target_col].values
    y_test = df_test[target_col].values

    # Fit preprocessor on train, transform val and test
    preprocessor, X_train_trans = fit_and_save_preprocessor(df_train, config)
    X_val_trans = transform_features(df_val, preprocessor, config)
    X_test_trans = transform_features(df_test, preprocessor, config)

    models = build_models(config)
    cv = KFold(n_splits=config.training.cv_folds, shuffle=True, random_state=config.project.random_seed)

    best_model_name = None
    best_model = None
    best_val_rmse = float("inf")
    best_metrics = {}
    comparison_results = {}

    # Metadata for lineage
    with open(Path(config.paths.processed_dir) / "ingestion_metadata.json", "r") as f:
        ingestion_meta = json.load(f)

    for model_name, model in models.items():
        logger.info(f"--- Training & Cross-Validating: {model_name} ---")

        # 5-Fold Cross-Validation on Train
        neg_mse_scores = cross_val_score(
            model, X_train_trans, y_train, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1
        )
        cv_rmse = float(np.mean(np.sqrt(-neg_mse_scores)))
        cv_rmse_std = float(np.std(np.sqrt(-neg_mse_scores)))

        # Train on full train set and evaluate on validation set
        model.fit(X_train_trans, y_train)
        y_val_pred = model.predict(X_val_trans)
        val_eval = evaluate_predictions(y_val, y_val_pred)

        logger.info(f"  CV RMSE:   {cv_rmse:.2f} ± {cv_rmse_std:.2f}")
        logger.info(f"  Val RMSE:  {val_eval['rmse']:.2f} Lakhs")
        logger.info(f"  Val R²:    {val_eval['r2']:.4f}")
        logger.info(f"  Val MAE:   {val_eval['mae']:.2f} Lakhs")

        comparison_results[model_name] = {
            "cv_rmse": cv_rmse,
            "cv_rmse_std": cv_rmse_std,
            "val_rmse": val_eval["rmse"],
            "val_r2": val_eval["r2"],
            "val_mae": val_eval["mae"],
            "val_mape": val_eval["mape"]
        }

        # Track each model run in MLflow
        with mlflow_mgr.start_run(run_name=f"explore_{model_name}") as run:
            params = config.training.hyperparameters.get(model_name, {})
            flat_params = {f"{model_name}_{k}": str(v) for k, v in params.items()}
            metrics_to_log = {
                "cv_rmse": cv_rmse,
                "cv_rmse_std": cv_rmse_std,
                "val_rmse": val_eval["rmse"],
                "val_r2": val_eval["r2"],
                "val_mae": val_eval["mae"],
                "val_mape": val_eval["mape"],
                "val_within_5_lakhs_pct": val_eval["within_5_lakhs_pct"],
                "val_within_10_lakhs_pct": val_eval["within_10_lakhs_pct"]
            }
            tags_to_log = {
                "model_type": model_name,
                "data_train_hash": ingestion_meta.get("train_hash", ""),
                "data_raw_hash": ingestion_meta.get("raw_train_hash", ""),
                "pipeline_stage": "training_exploration"
            }
            mlflow_mgr.log_metrics_and_params(flat_params, metrics_to_log, tags_to_log)

        if val_eval["rmse"] < best_val_rmse:
            best_val_rmse = val_eval["rmse"]
            best_model_name = model_name
            best_model = model
            best_metrics = val_eval
            best_metrics["model_name"] = model_name
            best_metrics["cv_rmse"] = cv_rmse

    logger.info(f"=== Optimal Model Selected: {best_model_name} (Val RMSE: {best_val_rmse:.2f} Lakhs) ===")

    # Evaluate the selected model on the Test set
    y_test_pred = best_model.predict(X_test_trans)
    test_eval = evaluate_predictions(y_test, y_test_pred)
    best_metrics["test_rmse"] = test_eval["rmse"]
    best_metrics["test_r2"] = test_eval["r2"]
    best_metrics["test_mae"] = test_eval["mae"]
    best_metrics["test_mape"] = test_eval["mape"]
    best_metrics["test_founder_summary"] = test_eval["founder_summary"]

    logger.info(f"Held-Out Test Performance for {best_model_name}:")
    logger.info(f"  Test RMSE: {test_eval['rmse']:.2f} Lakhs")
    logger.info(f"  Test R²:   {test_eval['r2']:.4f}")
    logger.info(f"  Test MAE:  {test_eval['mae']:.2f} Lakhs")
    logger.info(f"  Founder:   {test_eval['founder_summary']}")

    # Save candidate model to disk
    model_dir = Path(config.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    candidate_model_path = model_dir / "candidate_model.joblib"
    joblib.dump(best_model, candidate_model_path)

    # Save summary report
    summary = {
        "best_model_name": best_model_name,
        "best_metrics": best_metrics,
        "all_models_comparison": comparison_results
    }
    with open(model_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved Candidate Model to: {candidate_model_path}")
    return best_model, best_metrics, summary


if __name__ == "__main__":
    b_model, b_metrics, summary = train_and_evaluate_models()
    print(f"Training Stage Finished. Selected: {b_metrics['model_name']} with Test R²: {b_metrics['test_r2']:.4f}")
