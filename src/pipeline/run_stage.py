"""
Single-Stage Execution CLI (Task 3).
Allows running or re-running any individual stage of the MLOps pipeline independently.
Available stages: ingest, validate_data, build_features, train, evaluate, validate_model, deploy, monitor.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

from config.schema import load_config
from config.logging_config import setup_logger

logger = setup_logger("run_stage")


def execute_stage(stage_name: str) -> Any:
    """Dispatches execution to the requested stage."""
    config = load_config()
    stage = stage_name.lower().strip()
    logger.info(f"Executing standalone stage: '{stage}'...")

    if stage in ["ingest", "data_ingest"]:
        from src.data.ingestion import ingest_data
        return ingest_data(config)

    elif stage in ["validate_data", "data_gate", "validate"]:
        from src.data.validation import validate_data_gate
        return validate_data_gate(strict=True)

    elif stage in ["build_features", "preprocess", "features"]:
        import pandas as pd
        from src.features.preprocessor import fit_and_save_preprocessor
        train_path = Path(config.paths.train_data)
        if not train_path.exists():
            from src.data.ingestion import ingest_data
            ingest_data(config)
        df_train = pd.read_csv(train_path)
        pipe, X_trans = fit_and_save_preprocessor(df_train, config)
        return {"features_shape": X_trans.shape}

    elif stage in ["train", "model_train"]:
        from src.models.train import train_and_evaluate_models
        model, metrics, summary = train_and_evaluate_models(config)
        return summary

    elif stage in ["evaluate", "model_eval"]:
        import json
        import pandas as pd
        import joblib
        from src.features.preprocessor import transform_features
        from src.models.evaluate import evaluate_predictions

        test_path = Path(config.paths.test_data)
        model_path = Path(config.paths.active_model_path)
        prep_path = Path(config.paths.active_preprocessor_path)

        if not model_path.exists() or not prep_path.exists():
            raise FileNotFoundError("Active model or preprocessor artifact not found.")

        df_test = pd.read_csv(test_path)
        preprocessor = joblib.load(prep_path)
        model = joblib.load(model_path)

        X_test = transform_features(df_test, preprocessor, config)
        y_test = df_test[config.features.target].values
        preds = model.predict(X_test)
        eval_res = evaluate_predictions(y_test, preds)
        logger.info(f"Evaluation Results: RMSE={eval_res['rmse']:.2f}, R2={eval_res['r2']:.4f}")
        return eval_res

    elif stage in ["validate_model", "model_gate"]:
        import json
        from src.models.gate import evaluate_model_gate, ModelGateValidationError
        summary_path = Path(config.paths.model_dir) / "training_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Training summary missing at {summary_path}")
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        gate_res = evaluate_model_gate(summary["best_metrics"], config=config)
        if not gate_res.passed:
            raise ModelGateValidationError("Model Gate failed.", gate_res.reasons)
        return gate_res.model_dump()

    elif stage in ["deploy", "register"]:
        from src.pipeline.deploy import register_and_deploy_model
        return register_and_deploy_model(config)

    elif stage in ["monitor", "drift"]:
        from src.monitoring.drift import run_monitoring_pipeline
        return run_monitoring_pipeline(config)

    else:
        valid_stages = ["ingest", "validate_data", "build_features", "train", "evaluate", "validate_model", "deploy", "monitor"]
        raise ValueError(f"Unknown stage '{stage}'. Choose from: {valid_stages}")


def main():
    parser = argparse.ArgumentParser(description="Run an individual stage of the MLOps pipeline.")
    parser.add_argument("--stage", type=str, required=True, help="Name of the stage to execute")
    args = parser.parse_args()

    try:
        res = execute_stage(args.stage)
        logger.info(f"Stage '{args.stage}' completed successfully.")
        print(f"STAGE_SUCCESS: {args.stage}")
    except Exception as e:
        logger.error(f"Stage '{args.stage}' failed: {e}")
        print(f"STAGE_FAILURE: {args.stage} - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
