"""
API Client Simulation Utility.
Sends real-time sample requests to the FastAPI serving endpoint to generate
telemetry and verify operational latency.
"""

import sys
import time
from pathlib import Path
import requests

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_client_simulation(base_url: str = "http://localhost:8000"):
    print("=" * 80)
    print(f" TESTING API SERVING ENDPOINT AT {base_url}")
    print("=" * 80)

    try:
        # 1. Health
        h_resp = requests.get(f"{base_url}/health", timeout=5)
        print(f"GET /health -> Status {h_resp.status_code}")
        print(f"  Response: {h_resp.json()}")

        # 2. Predict
        payload = {
            "Area_sqft": 1450.0,
            "Bedrooms": 3,
            "Bathrooms": 2,
            "Age_Years": 4.0,
            "Distance_to_Center_km": 5.2,
            "Neighborhood_Score": 8.5,
            "Has_Parking": 1,
            "Property_Condition": "Good",
            "City_Zone": "Central"
        }
        p_resp = requests.post(f"{base_url}/predict", json=payload, timeout=5)
        print(f"\nPOST /predict -> Status {p_resp.status_code}")
        print(f"  Response: {p_resp.json()}")

        # 3. Lineage
        l_resp = requests.get(f"{base_url}/lineage", timeout=5)
        print(f"\nGET /lineage -> Status {l_resp.status_code}")
        print(f"  Model Version: {l_resp.json().get('version')}")

        # 4. Metrics
        m_resp = requests.get(f"{base_url}/metrics", timeout=5)
        print(f"\nGET /metrics -> Status {m_resp.status_code}")
        print(f"  Payload Size: {len(m_resp.text)} bytes")

    except requests.exceptions.ConnectionError:
        print(f"[!] Server not running at {base_url}. Start with: uvicorn src.api.app:app --port 8000")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    run_client_simulation(url)
