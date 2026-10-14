"""
Prometheus Metrics Instrumentation for FastAPI Serving Endpoint (Task 7).
Tracks requests, error counters, latency histograms, prediction price distributions,
and active model version gauges.
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Operational Metrics
REQUEST_COUNT = Counter(
    "model_api_requests_total",
    "Total HTTP requests to the model serving API",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "model_api_request_duration_seconds",
    "HTTP request latency duration in seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5]
)

PREDICTION_PRICE_HISTOGRAM = Histogram(
    "model_prediction_price_lakhs",
    "Distribution of predicted real-estate asking prices in INR Lakhs",
    buckets=[10, 25, 40, 55, 70, 85, 100, 120, 150, 200, 300]
)

ERROR_COUNT = Counter(
    "model_api_errors_total",
    "Total count of exceptions and 4xx/5xx errors",
    ["error_type", "endpoint"]
)

ACTIVE_MODEL_GAUGE = Gauge(
    "model_active_version",
    "Currently deployed active model version",
    ["model_name", "model_version"]
)


def get_prometheus_metrics_payload() -> bytes:
    """Generates the latest Prometheus scrape payload."""
    return generate_latest()
