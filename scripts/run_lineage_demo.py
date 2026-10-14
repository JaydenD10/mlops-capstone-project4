"""
Lineage Demonstration CLI Script.
Traces and visualizes the complete provenance of the currently deployed model.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.registry.lineage import get_live_lineage, print_lineage_tree

if __name__ == "__main__":
    v_arg = sys.argv[1] if len(sys.argv) > 1 else None
    lineage = get_live_lineage(v_arg)
    print_lineage_tree(lineage)
