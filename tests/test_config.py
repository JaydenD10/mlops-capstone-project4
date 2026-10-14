"""
Unit tests for configuration loading and validation schema (Task 2).
"""

import pytest
from pydantic import ValidationError
from config.schema import load_config, PipelineConfig


def test_load_valid_config():
    """Verify that the official config.yaml loads and validates successfully."""
    cfg = load_config("config/config.yaml")
    assert isinstance(cfg, PipelineConfig)
    assert cfg.project.name == "property-price-predictor"
    assert cfg.features.target == "Price_Lakhs"
    assert len(cfg.features.numerical) > 0
    assert len(cfg.features.categorical) > 0


def test_config_ratio_validation():
    """Verify that split ratios must sum to 1.0."""
    cfg_dict = load_config("config/config.yaml").model_dump()
    cfg_dict["split"]["train_ratio"] = 0.50
    cfg_dict["split"]["val_ratio"] = 0.20
    cfg_dict["split"]["test_ratio"] = 0.20  # Sums to 0.90 -> invalid!

    with pytest.raises(ValidationError):
        PipelineConfig(**cfg_dict)


def test_config_missing_file_raises():
    """Verify that a non-existent config path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config.yaml")
