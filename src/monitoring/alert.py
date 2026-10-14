"""
Alert Evaluation and Automated Mitigation Playbook (Task 7).
Formalizes alert levels, threshold justifications, and automated response actions.
"""

from typing import Dict, Any, List
from config.logging_config import setup_logger

logger = setup_logger("monitoring_alert")


class MonitoringAlertPolicy:
    """
    Evaluates drift signals and specifies the automated mitigation response.
    """
    THRESHOLD_JUSTIFICATIONS = {
        "psi_warning_0_10": (
            "PSI between 0.10 and 0.25 indicates moderate distributional change. "
            "Statistical convention established by credit scoring and financial risk models."
        ),
        "psi_critical_0_25": (
            "PSI >= 0.25 represents significant population shift where reference model "
            "assumptions no longer hold, risking biased predictions."
        ),
        "ks_test_pvalue_0_01": (
            "Two-sample Kolmogorov-Smirnov test with p < 0.01 rejects the null hypothesis "
            "that batch feature sample was drawn from training distribution at 99% confidence."
        ),
        "perf_degradation_15pct": (
            "A 15% increase in RMSE directly impacts marketplace margins for the startup, "
            "leading to unacceptable valuation discrepancies for property buyers and sellers."
        )
    }

    ACTION_PLAYBOOKS = {
        "HEALTHY": {
            "severity": "INFO",
            "action": "NO_ACTION",
            "description": "System operating nominally within historical confidence bounds.",
            "next_step": "Continue real-time operational serving and standard weekly monitoring."
        },
        "WARNING": {
            "severity": "WARNING",
            "action": "FLAG_AND_ACCELERATE_MONITORING",
            "description": "Moderate covariate drift observed without catastrophic performance drop.",
            "next_step": "Double telemetry sampling rate, inspect feature logs, queue data for scheduled retraining."
        },
        "CRITICAL": {
            "severity": "ERROR",
            "action": "TRIGGER_PIPELINE_OR_ALERT_ONCALL",
            "description": "Severe performance degradation or major feature drift detected.",
            "next_step": (
                "1. Trigger automated retraining pipeline with newly accumulated batch data included.\n"
                "2. Candidate model must pass Data Gate and Model Gate before replacing champion.\n"
                "3. If candidate model passes Model Gate, promote to Production with zero-downtime hot-swap.\n"
                "4. If candidate model fails gate, keep current production model serving and page MLOps on-call."
            )
        }
    }


def get_playbook_summary(alert_level: str) -> Dict[str, Any]:
    """Returns playbook actions and explanations for the given alert level."""
    return MonitoringAlertPolicy.ACTION_PLAYBOOKS.get(alert_level, MonitoringAlertPolicy.ACTION_PLAYBOOKS["HEALTHY"])
