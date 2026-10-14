"""
Feature Engineering and Preprocessing Pipeline.
Implements domain features, handles missing values (imputation),
encodes categoricals, scales numericals, and exports a reusable Scikit-Learn transformer.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

from config.schema import load_config, PipelineConfig
from config.logging_config import setup_logger

logger = setup_logger("preprocessor")


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-specific feature engineering transformer for real-estate pricing.
    Computes ratio and interaction features without target leakage.
    """
    def __init__(self):
        self.engineered_feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        # 1. Area Per Room
        rooms = X_out["Bedrooms"].fillna(1) + X_out["Bathrooms"].fillna(1)
        rooms = np.where(rooms == 0, 1, rooms)
        X_out["Area_Per_Room"] = X_out["Area_sqft"] / rooms

        # 2. Bathroom to Bedroom Ratio
        beds = X_out["Bedrooms"].fillna(1)
        beds = np.where(beds == 0, 1, beds)
        X_out["Bath_To_Bed_Ratio"] = X_out["Bathrooms"].fillna(1) / beds

        # 3. Overall Desirability: Neighborhood score vs Distance to center
        # Higher score closer to center = prime value
        dist = X_out["Distance_to_Center_km"].fillna(12.0)
        score = X_out["Neighborhood_Score"].fillna(6.0)
        X_out["Overall_Desirability"] = score / (dist + 1.0)

        # 4. Property Age Factor
        age = X_out["Age_Years"].fillna(20.0)
        X_out["Age_Factor"] = np.exp(-age / 50.0)

        self.engineered_feature_names = [
            "Area_Per_Room", "Bath_To_Bed_Ratio", "Overall_Desirability", "Age_Factor"
        ]
        return X_out


def build_preprocessor_pipeline(config: PipelineConfig) -> Pipeline:
    """
    Constructs a complete Scikit-Learn preprocessing pipeline based on config.
    """
    numerical_base = config.features.numerical
    engineered_cols = ["Area_Per_Room", "Bath_To_Bed_Ratio", "Overall_Desirability", "Age_Factor"]
    all_numeric = numerical_base + engineered_cols

    # Numeric Pipeline: Median Imputer + RobustScaler
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=config.preprocessing.numerical_imputer)),
        ("scaler", RobustScaler())
    ])

    # Categorical Ordinal: Property_Condition
    cond_order = [config.preprocessing.condition_order]
    cond_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=cond_order, handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    # Categorical Nominal: City_Zone
    zone_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, all_numeric),
            ("cond", cond_pipeline, ["Property_Condition"]),
            ("zone", zone_pipeline, ["City_Zone"])
        ],
        remainder="drop"
    )

    full_pipeline = Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("col_transform", column_transformer)
    ])

    return full_pipeline


def fit_and_save_preprocessor(
    train_df: pd.DataFrame,
    config: PipelineConfig = None,
    save_path: Optional[str] = None
) -> Tuple[Pipeline, np.ndarray]:
    """
    Fits the preprocessing pipeline on training features and saves the fitted artifact.
    Returns (fitted_pipeline, transformed_X_train).
    """
    if config is None:
        config = load_config()

    target_col = config.features.target
    drop_cols = [target_col] + config.features.id_columns + config.features.metadata_columns
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols].copy()

    logger.info(f"Fitting preprocessor on {len(X_train)} samples across {len(feature_cols)} features...")
    pipeline = build_preprocessor_pipeline(config)
    X_train_transformed = pipeline.fit_transform(X_train)

    out_path = Path(save_path if save_path else config.paths.active_preprocessor_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    logger.info(f"Fitted preprocessor successfully saved to: {out_path} (Output shape: {X_train_transformed.shape})")

    return pipeline, X_train_transformed


def transform_features(
    df: pd.DataFrame,
    preprocessor: Pipeline,
    config: PipelineConfig = None
) -> np.ndarray:
    """
    Transforms arbitrary feature dataframe using a fitted preprocessor pipeline.
    """
    if config is None:
        config = load_config()

    drop_cols = [config.features.target] + config.features.id_columns + config.features.metadata_columns
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return preprocessor.transform(df[feature_cols].copy())


if __name__ == "__main__":
    cfg = load_config()
    train_path = Path(cfg.paths.train_data)
    if not train_path.exists():
        from src.data.ingestion import ingest_data
        ingest_data(cfg)

    df_train = pd.read_csv(train_path)
    pipe, X_trans = fit_and_save_preprocessor(df_train, cfg)
    print(f"Preprocessing completed. Transformed shape: {X_trans.shape}")
