"""
Monitoring Pipeline Runner Script.
Executes drift monitoring over incoming batches 1, 2, and 3.
"""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.drift import run_monitoring_pipeline

if __name__ == "__main__":
    print("=" * 80)
    print(" EXECUTING OPERATIONAL DRIFT MONITORING (BATCHES 1, 2, 3)")
    print("=" * 80)
    summary = run_monitoring_pipeline()

    print("\n" + "-" * 80)
    print(" BATCH MONITORING SUMMARY:")
    for b_key, status in summary["batch_statuses"].items():
        rep = summary["reports"][b_key]
        perf = rep.get("performance_metrics", {})
        print(f" * {b_key.upper():<10} : [{status:<8}] - RMSE: {perf.get('current_rmse')} Lakhs (Degradation: {perf.get('rmse_degradation_pct', 0.0):+.1f}%)")
        if rep["critical_triggers"]:
            for t in rep["critical_triggers"]:
                print(f"     [!] {t}")
    print("-" * 80)
    print("Reports stored at: reports/drift/")
