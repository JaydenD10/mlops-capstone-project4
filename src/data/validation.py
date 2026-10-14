"""
Data Validation Gate for MLOps Pipeline (Task 4).
Performs strict schema validation, range checks, categorical domain checks,
and missing value limits. Fails safely with diagnostic error reporting.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from pydantic import BaseModel

from config.schema import load_config, PipelineConfig
from config.logging_config import setup_logger

logger = setup_logger("data_validation")


class DataGateValidationError(Exception):
    """Raised when data gate validation fails."""
    def __init__(self, message: str, errors: List[str]):
        super().__init__(message)
        self.errors = errors


class ValidationResult(BaseModel):
    passed: bool
    total_records: int
    errors: List[str]
    warnings: List[str]
    null_summary: Dict[str, float]
    range_summary: Dict[str, Dict[str, float]]


def validate_dataframe(
    df: pd.DataFrame,
    config: PipelineConfig = None,
    is_training_data: bool = True
) -> ValidationResult:
    """
    Validates a DataFrame against Data Gate rules defined in config.yaml.
    Returns ValidationResult object.
    """
    if config is None:
        config = load_config()

    gate_cfg = config.data_gate
    feat_cfg = config.features

    errors: List[str] = []
    warnings: List[str] = []

    # 1. Schema Check: Required Columns
    expected_cols = (
        feat_cfg.numerical + feat_cfg.categorical + feat_cfg.id_columns
    )
    if is_training_data:
        expected_cols.append(feat_cfg.target)

    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns in dataset: {missing_cols}")

    if errors:
        return ValidationResult(
            passed=False,
            total_records=len(df),
            errors=errors,
            warnings=warnings,
            null_summary={},
            range_summary={}
        )

    # 2. Missing Value Checks
    null_summary: Dict[str, float] = {}
    total_cells = df.shape[0] * df.shape[1]
    total_nulls = df.isnull().sum().sum()
    overall_null_pct = total_nulls / total_cells if total_cells > 0 else 0.0

    if overall_null_pct > gate_cfg.max_null_pct_overall:
        errors.append(
            f"Overall missing value percentage {overall_null_pct:.2%} exceeds maximum allowed {gate_cfg.max_null_pct_overall:.2%}"
        )

    for col in expected_cols:
        col_null_pct = df[col].isnull().mean()
        null_summary[col] = float(col_null_pct)
        if col_null_pct > gate_cfg.max_null_pct_per_column:
            errors.append(
                f"Column '{col}' has {col_null_pct:.2%} missing values, exceeding limit {gate_cfg.max_null_pct_per_column:.2%}"
            )

    # In training data, target must NEVER be missing
    if is_training_data and feat_cfg.target in df.columns:
        if df[feat_cfg.target].isnull().any():
            errors.append(f"Target column '{feat_cfg.target}' contains {df[feat_cfg.target].isnull().sum()} null values")

    # 3. Categorical Domain Checks
    if "Property_Condition" in df.columns:
        invalid_conds = df[~df["Property_Condition"].isin(gate_cfg.allowed_property_conditions) & df["Property_Condition"].notnull()]["Property_Condition"].unique()
        if len(invalid_conds) > 0:
            errors.append(f"Invalid Property_Condition values found: {list(invalid_conds)}. Allowed: {gate_cfg.allowed_property_conditions}")

    if "City_Zone" in df.columns:
        invalid_zones = df[~df["City_Zone"].isin(gate_cfg.allowed_city_zones) & df["City_Zone"].notnull()]["City_Zone"].unique()
        if len(invalid_zones) > 0:
            errors.append(f"Invalid City_Zone values found: {list(invalid_zones)}. Allowed: {gate_cfg.allowed_city_zones}")

    # 4. Numerical Range Checks
    range_summary: Dict[str, Dict[str, float]] = {}
    for col, bounds in gate_cfg.numerical_ranges.items():
        if col in df.columns:
            valid_vals = df[col].dropna()
            if len(valid_vals) > 0:
                c_min = float(valid_vals.min())
                c_max = float(valid_vals.max())
                range_summary[col] = {"min": c_min, "max": c_max}

                out_of_bounds = ((valid_vals < bounds.min) | (valid_vals > bounds.max)).sum()
                if out_of_bounds > 0:
                    errors.append(
                        f"Column '{col}' has {out_of_bounds} values outside allowed range [{bounds.min}, {bounds.max}]. Found min={c_min}, max={c_max}"
                    )

    passed = len(errors) == 0
    return ValidationResult(
        passed=passed,
        total_records=len(df),
        errors=errors,
        warnings=warnings,
        null_summary=null_summary,
        range_summary=range_summary
    )


def validate_data_gate(data_path: Optional[str] = None, strict: bool = True) -> ValidationResult:
    """
    Validates training data file or specified file through the Data Gate.
    If strict=True and validation fails, logs error and raises DataGateValidationError.
    """
    config = load_config()
    target_path = Path(data_path if data_path else config.paths.train_data)

    if not target_path.exists():
        fallback = Path("capstone_house_prices_train.csv")
        if fallback.exists():
            target_path = fallback
        else:
            raise FileNotFoundError(f"Data file to validate not found at {target_path}")

    logger.info(f"Running Data Gate Validation on: {target_path}")
    df = pd.read_csv(target_path)
    result = validate_dataframe(df, config=config, is_training_data=True)

    if result.passed:
        logger.info(f"DATA GATE PASSED: {result.total_records} records verified successfully.")
    else:
        logger.error(f"DATA GATE FAILED: {len(result.errors)} validation violations encountered:")
        for err in result.errors:
            logger.error(f"  - {err}")
        if strict:
            raise DataGateValidationError(
                f"Data Gate rejected dataset {target_path} with {len(result.errors)} violations.",
                result.errors
            )

    return result


if __name__ == "__main__":
    file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        res = validate_data_gate(file_arg, strict=True)
        print("Data Gate Validation: SUCCESS")
    except DataGateValidationError as exc:
        print(f"Data Gate Validation: FAILED - {exc}")
        sys.exit(1)
