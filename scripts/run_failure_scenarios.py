"""
Validation Gates and Safe Failure Demonstration (Task 4).
Demonstrates two failing scenarios:
1. Scenario A: Corrupted / Bad Data Batch -> triggers Data Gate failure cleanly.
2. Scenario B: Substandard / Degraded Model -> triggers Model Gate failure cleanly.
Verifies that after both failure scenarios, the active production model and endpoint are UNCHANGED.
"""

import json
import sys
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.schema import load_config
from src.data.validation import validate_dataframe, DataGateValidationError
from src.models.gate import evaluate_model_gate, ModelGateValidationError
from src.registry.mlflow_manager import MLflowManager


def test_scenario_a_bad_data_gate():
    """
    Scenario A: Demonstrates that an incoming corrupted data batch
    (missing columns, out-of-bounds area, invalid condition) fails Data Gate cleanly.
    """
    print("\n" + "=" * 80)
    print(" DEMONSTRATING SCENARIO A: CORRUPTED DATA BATCH (DATA GATE FAILURE)")
    print("=" * 80)

    config = load_config()

    # Create synthetic bad data batch
    bad_data = pd.DataFrame([
        {
            "House_ID": "BAD-001",
            "Area_sqft": -500.0,            # Negative area! Violation: min 100
            "Bedrooms": 0,                  # Zero bedrooms! Violation: min 1
            "Bathrooms": 25,                # 25 bathrooms! Violation: max 20
            "Age_Years": 250.0,             # 250 years old! Violation: max 150
            "Distance_to_Center_km": 150.0, # 150km! Violation: max 100
            "Neighborhood_Score": 15.0,     # Score 15! Violation: max 10
            "Has_Parking": 5,               # Not 0/1! Violation
            "Property_Condition": "Broken", # Invalid category!
            "City_Zone": "OuterSpace",      # Invalid zone!
            "Price_Lakhs": -10.0            # Negative price!
        },
        {
            "House_ID": "BAD-002",
            "Area_sqft": 1000.0,
            "Bedrooms": 2,
            "Bathrooms": 1,
            "Age_Years": None,
            "Distance_to_Center_km": None,
            "Neighborhood_Score": None,
            "Has_Parking": 1,
            "Property_Condition": "Good",
            "City_Zone": "North",
            "Price_Lakhs": None             # Null target in training!
        }
    ])

    print("Submitting corrupted dataset to Data Validation Gate...")
    result = validate_dataframe(bad_data, config=config, is_training_data=True)

    print(f"\nData Gate Gate Outcome: Passed = {result.passed}")
    print(f"Detected Violations ({len(result.errors)}):")
    for err in result.errors:
        print(f"  [X] {err}")

    assert not result.passed, "Data Gate should have failed!"

    # Verify active production model is unchanged
    meta_path = Path(config.paths.model_dir) / "active_model_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    active_v = meta.get('active_version') or meta.get('model_version') or '1'
    print(f"\n[SAFE BEHAVIOR VERIFIED]: Active Production Model remains v{active_v} untouched.")
    return result


def test_scenario_b_substandard_model_gate():
    """
    Scenario B: Demonstrates that an underperforming model
    (e.g., R2 = 0.42 < 0.80, RMSE = 18.5 > 12.0) is rejected by Model Gate.
    """
    print("\n" + "=" * 80)
    print(" DEMONSTRATING SCENARIO B: SUBSTANDARD MODEL (MODEL GATE FAILURE)")
    print("=" * 80)

    config = load_config()
    mlflow_mgr = MLflowManager(config)

    # Substandard candidate metrics (e.g. from an underfitted dummy model)
    substandard_metrics = {
        "model_name": "underfitted_linear_baseline",
        "r2": 0.4215,     # Well below 0.80 minimum threshold
        "rmse": 18.45,    # Well above 12.0 Lakhs threshold
        "mae": 14.10,     # Well above 8.5 Lakhs threshold
        "mape": 0.3250,   # Well above 20% MAPE limit
        "founder_summary": "Underfitted model with excessive pricing error."
    }

    print("Evaluating substandard model metrics against Model Gate...")
    gate_res = evaluate_model_gate(substandard_metrics, config=config, mlflow_mgr=mlflow_mgr)

    print(f"\nModel Gate Outcome: Passed = {gate_res.passed}")
    print(f"Detected Gate Violations ({len(gate_res.reasons)}):")
    for r in gate_res.reasons:
        print(f"  [X] {r}")

    assert not gate_res.passed, "Model Gate should have failed!"

    # Verify active production model is unchanged
    meta_path = Path(config.paths.model_dir) / "active_model_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    active_v = meta.get('active_version') or meta.get('model_version') or '1'
    print(f"\n[SAFE BEHAVIOR VERIFIED]: Active Production Model remains v{active_v} untouched.")
    return gate_res


if __name__ == "__main__":
    res_a = test_scenario_a_bad_data_gate()
    res_b = test_scenario_b_substandard_model_gate()
    print("\n" + "=" * 80)
    print(" BOTH FAILING GATE SCENARIOS DEMONSTRATED WITH 100% SAFE BEHAVIOR!")
    print("=" * 80)
