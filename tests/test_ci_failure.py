"""
Controlled Unit Test Failure for CI Gate Validation.
This file is used exclusively on a test branch to demonstrate
safe failure detection by GitHub Actions CI checks.
"""


def test_controlled_ci_gate_failure():
    """Deliberate assertion failure demonstrating automated CI gating."""
    assert False, "Controlled assertion failure: CI validation gate rejected defective commit."
