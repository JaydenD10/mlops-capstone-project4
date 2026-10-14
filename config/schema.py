"""
Configuration schema and validation module for MLOps Project 4.
Validates config.yaml on load using Pydantic.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml
from pydantic import BaseModel, Field, field_validator


class ProjectConfig(BaseModel):
    name: str
    version: str
    author: str
    description: str
    random_seed: int = 42


class PathsConfig(BaseModel):
    raw_train_data: str
    raw_incoming_batches: str
    processed_dir: str
    train_data: str
    val_data: str
    test_data: str
    batches_dir: str
    model_dir: str
    active_model_path: str
    active_preprocessor_path: str
    reports_dir: str
    figures_dir: str
    drift_dir: str


class FeaturesConfig(BaseModel):
    target: str
    id_columns: List[str]
    metadata_columns: List[str]
    numerical: List[str]
    categorical: List[str]
    engineered: List[str] = Field(default_factory=list)


class SplitConfig(BaseModel):
    train_ratio: float = Field(gt=0, lt=1)
    val_ratio: float = Field(gt=0, lt=1)
    test_ratio: float = Field(gt=0, lt=1)
    stratify_by: Optional[str] = None

    @field_validator("test_ratio")
    @classmethod
    def validate_ratios(cls, v, info):
        # Validate that split ratios sum to approximately 1.0
        data = info.data
        train_ratio = data.get("train_ratio", 0.0)
        val_ratio = data.get("val_ratio", 0.0)
        total = train_ratio + val_ratio + v
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        return v


class NumericRange(BaseModel):
    min: float
    max: float


class DataGateConfig(BaseModel):
    max_null_pct_per_column: float = Field(ge=0.0, le=1.0)
    max_null_pct_overall: float = Field(ge=0.0, le=1.0)
    allowed_property_conditions: List[str]
    allowed_city_zones: List[str]
    numerical_ranges: Dict[str, NumericRange]


class PreprocessingConfig(BaseModel):
    numerical_imputer: str = "median"
    categorical_imputer: str = "most_frequent"
    scaler: str = "robust"
    condition_order: List[str]


class TrainingConfig(BaseModel):
    cv_folds: int = Field(ge=2, le=10)
    models_to_evaluate: List[str]
    hyperparameters: Dict[str, Dict[str, Union[int, float, str, List[int], None]]]


class ModelGateConfig(BaseModel):
    min_r2_score: float = Field(ge=0.0, le=1.0)
    max_rmse_lakhs: float = Field(gt=0.0)
    max_mae_lakhs: float = Field(gt=0.0)
    max_mape: float = Field(gt=0.0, le=1.0)
    challenger_must_beat_champion: bool = True
    min_champion_improvement_pct: float = 0.0


class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str
    model_name: str
    registered_stage: str = "Production"


class ServingConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    title: str
    version: str
    latency_p95_threshold_ms: float = 50.0
    latency_p99_threshold_ms: float = 100.0


class MonitoringConfig(BaseModel):
    drift_significance_level: float = 0.05
    psi_alert_threshold: float = 0.25
    psi_warning_threshold: float = 0.10
    ks_p_value_threshold: float = 0.01
    max_performance_drop_pct: float = 15.0
    batch_column: str = "Batch"


class PipelineConfig(BaseModel):
    project: ProjectConfig
    paths: PathsConfig
    features: FeaturesConfig
    split: SplitConfig
    data_gate: DataGateConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    model_gate: ModelGateConfig
    mlflow: MLflowConfig
    serving: ServingConfig
    monitoring: MonitoringConfig


def load_config(config_path: Union[str, Path] = "config/config.yaml") -> PipelineConfig:
    """
    Load and strictly validate configuration against PipelineConfig schema.
    Raises ValidationError if invalid.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f)

    if not isinstance(raw_dict, dict):
        raise ValueError(f"Config YAML must define a dictionary, got {type(raw_dict)}")

    # Validates with Pydantic
    return PipelineConfig(**raw_dict)


if __name__ == "__main__":
    cfg = load_config()
    print(f"Configuration successfully validated: {cfg.project.name} v{cfg.project.version}")
