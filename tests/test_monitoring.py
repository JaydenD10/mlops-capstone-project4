"""
Unit tests for Drift Monitoring and Metrics (Task 7).
"""

import numpy as np
import pytest
from src.monitoring.drift import calculate_psi, calculate_categorical_psi
import pandas as pd


def test_psi_identical_distributions():
    """Identical distributions must have near-zero PSI."""
    np.random.seed(42)
    dist1 = np.random.normal(50, 10, 1000)
    dist2 = dist1.copy()
    psi = calculate_psi(dist1, dist2)
    assert psi < 0.05


def test_psi_shifted_distribution():
    """Significantly shifted distribution must trigger high PSI."""
    np.random.seed(42)
    dist_ref = np.random.normal(50, 5, 1000)
    dist_shifted = np.random.normal(85, 5, 1000)
    psi = calculate_psi(dist_ref, dist_shifted)
    assert psi > 0.25  # Significant drift threshold


def test_categorical_psi():
    s_ref = pd.Series(["A"] * 50 + ["B"] * 50)
    s_curr = pd.Series(["A"] * 95 + ["B"] * 5)
    psi = calculate_categorical_psi(s_ref, s_curr)
    assert psi > 0.10
