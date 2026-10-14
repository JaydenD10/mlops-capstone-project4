"""
Production FastAPI Application for Real-Estate Price Prediction (Task 7).
Serves real-time inference, batch prediction, Prometheus telemetry, health checks,
and active model lineage.
"""

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.schema import load_config, PipelineConfig
from config.logging_config import setup_logger
from src.api.schemas import (
    PropertyInput, BatchPropertyInput, PredictionOutput,
    BatchPredictionOutput, HealthResponse
)
from src.api.metrics import (
    REQUEST_COUNT, REQUEST_LATENCY, PREDICTION_PRICE_HISTOGRAM,
    ERROR_COUNT, ACTIVE_MODEL_GAUGE, get_prometheus_metrics_payload,
    CONTENT_TYPE_LATEST
)
from src.features.preprocessor import transform_features
from src.registry.lineage import get_live_lineage

logger = setup_logger("serving_api")

# Global state
app_state: Dict[str, Any] = {
    "model": None,
    "preprocessor": None,
    "metadata": {},
    "start_time": time.time(),
    "config": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: loads artifacts on startup."""
    cfg = load_config()
    app_state["config"] = cfg
    app_state["start_time"] = time.time()

    model_path = Path(cfg.paths.active_model_path)
    prep_path = Path(cfg.paths.active_preprocessor_path)
    meta_path = Path(cfg.paths.model_dir) / "active_model_metadata.json"

    # Fallback to candidate model if active model is not yet placed
    if not model_path.exists():
        candidate_path = Path(cfg.paths.model_dir) / "candidate_model.joblib"
        if candidate_path.exists():
            model_path = candidate_path

    if model_path.exists() and prep_path.exists():
        app_state["model"] = joblib.load(model_path)
        app_state["preprocessor"] = joblib.load(prep_path)
        logger.info(f"Loaded active model from: {model_path}")
        logger.info(f"Loaded active preprocessor from: {prep_path}")
    else:
        logger.warning(f"Model artifacts not found on startup ({model_path}). Retrying from pipeline...")
        try:
            from src.pipeline.orchestrator import run_full_pipeline
            run_full_pipeline(cfg)
            app_state["model"] = joblib.load(cfg.paths.active_model_path)
            app_state["preprocessor"] = joblib.load(cfg.paths.active_preprocessor_path)
        except Exception as e:
            logger.error(f"Could not initialize model: {e}")

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            app_state["metadata"] = json.load(f)
    else:
        app_state["metadata"] = {
            "model_name": cfg.mlflow.model_name,
            "model_version": "1",
            "model_type": "ridge"
        }

    # Record active version in Prometheus gauge
    ACTIVE_MODEL_GAUGE.labels(
        model_name=app_state["metadata"].get("model_name", "HousePricePredictor"),
        model_version=str(app_state["metadata"].get("model_version", "1"))
    ).set(1)

    logger.info("Serving API ready to accept inference requests.")
    yield
    logger.info("Serving API shutting down.")


cfg = load_config()
app = FastAPI(
    title=cfg.serving.title,
    version=cfg.serving.version,
    description="Production MLOps End-to-End Model Serving API with Prometheus Monitoring",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Middleware for recording Prometheus request telemetry."""
    endpoint = request.url.path
    method = request.method
    start = time.time()

    try:
        response = await call_next(request)
        duration = time.time() - start

        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=response.status_code).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        return response
    except Exception as exc:
        duration = time.time() - start
        ERROR_COUNT.labels(error_type=exc.__class__.__name__, endpoint=endpoint).inc()
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=500).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        raise exc


@app.get("/", tags=["General"])
async def root():
    """Root endpoint returning API metadata and documentation links."""
    return {
        "app": cfg.serving.title,
        "version": cfg.serving.version,
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health",
        "metrics_url": "/metrics",
        "lineage_url": "/lineage"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Operational health check returning model version and uptime."""
    model_loaded = app_state["model"] is not None
    prep_loaded = app_state["preprocessor"] is not None
    status_str = "healthy" if (model_loaded and prep_loaded) else "degraded"

    return HealthResponse(
        status=status_str,
        app_name=cfg.serving.title,
        version=cfg.serving.version,
        model_name=app_state["metadata"].get("model_name", cfg.mlflow.model_name),
        model_version=str(app_state["metadata"].get("model_version", "1")),
        model_loaded=model_loaded,
        preprocessor_loaded=prep_loaded,
        uptime_seconds=round(time.time() - app_state["start_time"], 2)
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics_endpoint():
    """Prometheus metrics scrape endpoint."""
    return Response(content=get_prometheus_metrics_payload(), media_type=CONTENT_TYPE_LATEST)


@app.get("/lineage", tags=["Lineage"])
async def model_lineage():
    """Returns the end-to-end provenance lineage of the deployed active model."""
    try:
        current_version = str(app_state["metadata"].get("model_version", "1"))
        lin = get_live_lineage(current_version)
        return lin
    except Exception as e:
        logger.error(f"Error retrieving lineage: {e}")
        return {"status": "error", "message": str(e), "metadata": app_state["metadata"]}


@app.post("/predict", response_model=PredictionOutput, tags=["Inference"])
async def predict_single(property_in: PropertyInput):
    """
    Real-time asking price prediction for a single property listing.
    Enforces validation, logs price distribution, and returns latency metrics.
    """
    if app_state["model"] is None or app_state["preprocessor"] is None:
        ERROR_COUNT.labels(error_type="ModelNotReady", endpoint="/predict").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not yet ready to serve predictions."
        )

    t0 = time.time()
    try:
        # Convert Pydantic input to single-row DataFrame
        input_dict = property_in.model_dump()
        input_df = pd.DataFrame([input_dict])

        # Preprocess features
        X_trans = transform_features(input_df, app_state["preprocessor"], app_state["config"])

        # Predict asking price in Lakhs
        pred_price = float(app_state["model"].predict(X_trans)[0])
        pred_price = max(1.0, round(pred_price, 2))  # Sanity clamp to positive

        latency_ms = round((time.time() - t0) * 1000.0, 2)

        # Record distribution in Prometheus
        PREDICTION_PRICE_HISTOGRAM.observe(pred_price)

        # 95% Error Margin based on test MAE (~5.9 Lakhs)
        margin = 5.90
        low = max(0.0, round(pred_price - margin, 2))
        high = round(pred_price + margin, 2)

        m_name = app_state["metadata"].get("model_type", "ridge")
        m_version = str(app_state["metadata"].get("model_version", "1"))

        return PredictionOutput(
            predicted_price_lakhs=pred_price,
            confidence_interval_low_lakhs=low,
            confidence_interval_high_lakhs=high,
            model_name=m_name,
            model_version=m_version,
            latency_ms=latency_ms,
            founder_note=f"Estimated asking price is INR {pred_price} Lakhs (Expected Range: INR {low} - INR {high} Lakhs)."
        )
    except Exception as exc:
        logger.error(f"Inference error: {exc}")
        ERROR_COUNT.labels(error_type=exc.__class__.__name__, endpoint="/predict").inc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.post("/batch-predict", response_model=BatchPredictionOutput, tags=["Inference"])
async def predict_batch(batch_in: BatchPropertyInput):
    """
    High-throughput batch prediction for bulk property listings.
    """
    if app_state["model"] is None or app_state["preprocessor"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not yet ready to serve predictions."
        )

    t0 = time.time()
    try:
        records = [p.model_dump() for p in batch_in.properties]
        input_df = pd.DataFrame(records)

        X_trans = transform_features(input_df, app_state["preprocessor"], app_state["config"])
        raw_preds = app_state["model"].predict(X_trans)

        m_name = app_state["metadata"].get("model_type", "ridge")
        m_version = str(app_state["metadata"].get("model_version", "1"))
        margin = 5.90

        outputs: List[PredictionOutput] = []
        for p in raw_preds:
            val = max(1.0, round(float(p), 2))
            PREDICTION_PRICE_HISTOGRAM.observe(val)
            outputs.append(
                PredictionOutput(
                    predicted_price_lakhs=val,
                    confidence_interval_low_lakhs=max(0.0, round(val - margin, 2)),
                    confidence_interval_high_lakhs=round(val + margin, 2),
                    model_name=m_name,
                    model_version=m_version,
                    latency_ms=0.0,
                    founder_note=f"INR {val} Lakhs"
                )
            )

        total_latency_ms = round((time.time() - t0) * 1000.0, 2)
        mean_price = float(np.mean([o.predicted_price_lakhs for o in outputs]))

        return BatchPredictionOutput(
            predictions=outputs,
            total_records=len(outputs),
            mean_predicted_price_lakhs=round(mean_price, 2),
            batch_latency_ms=total_latency_ms
        )
    except Exception as exc:
        logger.error(f"Batch inference error: {exc}")
        ERROR_COUNT.labels(error_type=exc.__class__.__name__, endpoint="/batch-predict").inc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
