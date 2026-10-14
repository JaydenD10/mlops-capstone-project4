"""
Unit tests for Preprocessing and Feature Engineering (Task 3).
"""

import numpy as np
import pandas as pd
import pytest
from config.schema import load_config
from src.features.preprocessor import build_preprocessor_pipeline, FeatureEngineer


def test_feature_engineer_adds_domain_features():
    df = pd.DataFrame([{
        "Area_sqft": 1000.0,
        "Bedrooms": 2,
        "Bathrooms": 2,
        "Age_Years": 10.0,
        "Distance_to_Center_km": 5.0,
        "Neighborhood_Score": 8.0,
        "Has_Parking": 1,
        "Property_Condition": "Good",
        "City_Zone": "Central"
    }])

    fe = FeatureEngineer()
    out = fe.transform(df)

    assert "Area_Per_Room" in out.columns
    assert "Bath_To_Bed_Ratio" in out.columns
    assert "Overall_Desirability" in out.columns
    assert "Age_Factor" in out.columns
    assert out["Area_Per_Room"].iloc[0] == 1000.0 / 4.0


def test_preprocessor_pipeline_transforms_cleanly():
    cfg = load_config()
    pipeline = build_preprocessor_pipeline(cfg)

    df = pd.DataFrame([{
        "Area_sqft": 1200.0,
        "Bedrooms": 3,
        "Bathrooms": 2,
        "Age_Years": np.nan,  # tests imputer
        "Distance_to_Center_km": 10.0,
        "Neighborhood_Score": np.nan,  # tests imputer
        "Has_Parking": 1,
        "Property_Condition": "Good",
        "City_Zone": "West"
    }, {
        "Area_sqft": 800.0,
        "Bedrooms": 2,
        "Bathrooms": 1,
        "Age_Years": 20.0,
        "Distance_to_Center_km": 15.0,
        "Neighborhood_Score": 6.0,
        "Has_Parking": 0,
        "Property_Condition": "Fair",
        "City_Zone": "South"
    }])

    transformed = pipeline.fit_transform(df)
    assert transformed.shape[0] == 2
    assert transformed.shape[1] > 0
    assert not np.isnan(transformed).any()
