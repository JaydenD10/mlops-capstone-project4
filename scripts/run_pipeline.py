"""
Pipeline Execution Entrypoint.
Runs the complete automated pipeline end-to-end.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.orchestrator import run_full_pipeline

if __name__ == "__main__":
    report = run_full_pipeline()
    if report.status == "SUCCESS":
        print(f"\n[OK] Pipeline completed successfully in {report.total_duration}s.")
        sys.exit(0)
    else:
        print(f"\n[ERROR] Pipeline halted with status: {report.status}")
        sys.exit(1)
