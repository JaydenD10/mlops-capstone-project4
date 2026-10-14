"""
Unit tests for Data Gate Validation (Task 4).
"""

import pandas as pd
import pytest
from config.schema import load_config
from src.data.validation import validate_dataframe


@pytest.fixture
def valid_dataframe():
    return pd.DataFrame([
        {
            "House_ID": f"HS-{i:06d}",
            "Area_sqft": 1000.0 + i * 100,
            "Bedrooms": 2,
            "Bathrooms": 2,
            "Age_Years": 5.0,
            "Distance_to_Center_km": 10.0,
            "Neighborhood_Score": 6.5,
            "Has_Parking": 1,
            "Property_Condition": "Good",
            "City_Zone": "Central",
            "Price_Lakhs": 50.0 + i * 5
        } for i in range(10)
    ])


def test_data_gate_passes_on_valid_data(valid_dataframe):
    cfg = load_config()
    result = validate_dataframe(valid_dataframe, config=cfg, is_training_data=True)
    assert result.passed
    assert len(result.errors) == 0


def test_data_gate_fails_on_missing_required_column(valid_dataframe):
    cfg = load_config()
    bad_df = valid_dataframe.drop(columns=["Area_sqft"])
    result = validate_dataframe(bad_df, config=cfg, is_training_data=True)
    assert not result.passed
    assert any("Missing required columns" in e for e in result.errors)


def test_data_gate_fails_on_invalid_category(valid_dataframe):
    cfg = load_config()
    bad_df = valid_dataframe.copy()
    bad_df.loc[0, "Property_Condition"] = "UltraLuxurious"
    result = validate_dataframe(bad_df, config=cfg, is_training_data=True)
    assert not result.passed
    assert any("Invalid Property_Condition" in e for e in result.errors)


def test_data_gate_fails_on_range_violation(valid_dataframe):
    cfg = load_config()
    bad_df = valid_dataframe.copy()
    bad_df.loc[0, "Neighborhood_Score"] = 25.0  # Allowed: [1, 10]
    result = validate_dataframe(bad_df, config=cfg, is_training_data=True)
    assert not result.passed
    assert any("Neighborhood_Score" in e for e in result.errors)
