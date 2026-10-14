"""
Unit tests for Model Validation Gate (Task 4).
"""

import pytest
from config.schema import load_config
from src.models.gate import evaluate_model_gate


def test_model_gate_passes_for_quality_model():
    cfg = load_config()
    metrics = {
        "rmse": 7.50,
        "mae": 5.40,
        "r2": 0.88,
        "mape": 0.09
    }
    result = evaluate_model_gate(metrics, config=cfg)
    assert result.passed
    assert len(result.reasons) == 0


def test_model_gate_rejects_low_r2():
    cfg = load_config()
    metrics = {
        "rmse": 7.50,
        "mae": 5.40,
        "r2": 0.65,  # Threshold is 0.80
        "mape": 0.09
    }
    result = evaluate_model_gate(metrics, config=cfg)
    assert not result.passed
    assert any("Candidate R²" in r for r in result.reasons)


def test_model_gate_rejects_excessive_rmse():
    cfg = load_config()
    metrics = {
        "rmse": 16.50,  # Threshold is 12.0
        "mae": 5.40,
        "r2": 0.85,
        "mape": 0.09
    }
    result = evaluate_model_gate(metrics, config=cfg)
    assert not result.passed
    assert any("Candidate RMSE" in r for r in result.reasons)
