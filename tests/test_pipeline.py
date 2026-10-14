"""
Integration tests for Pipeline Stages and Full Execution (Task 3).
"""

from pathlib import Path
import pytest
from config.schema import load_config
from src.pipeline.run_stage import execute_stage


def test_individual_stages_execute_independently():
    """Verify that individual pipeline stages can execute cleanly on their own."""
    cfg = load_config()

    # Stage: Ingest
    res_ingest = execute_stage("ingest")
    assert "train_rows" in res_ingest

    # Stage: Validate Data Gate
    res_val = execute_stage("validate_data")
    assert res_val.passed is True

    # Stage: Build Features
    res_feat = execute_stage("build_features")
    assert res_feat["features_shape"][0] > 0

    # Stage: Validate Model Gate
    res_gate = execute_stage("validate_model")
    assert res_gate["passed"] is True
