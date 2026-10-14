# MLOps Lifecycle Mapping Document

This document establishes the bidirectional traceability between the standard MLOps Lifecycle (as outlined in Topic 1 & Topic 15 of the course syllabus) and the codebase components.

---

## MLOps Lifecycle Mapping Table

```
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                       MLOPS CONTINUOUS LIFECYCLE                            │
   │                                                                             │
   │   [1. Framing] ──► [2. Ingestion] ──► [3. Validation Gate] ──► [4. Features]│
   │         ▲                                                            │      │
   │         │                                                            ▼      │
   │   [8. Retrain] ◄── [7. Monitor] ◄── [6. Deploy] ◄── [5. Model Gate] ◄─      │
   └─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Business Problem Framing & Requirements
- **Syllabus Topic:** Topic 1 (MLOps fundamentals & lifecycle)
- **Objective:** Establish quantitative goals, failure tolerance, and service level agreements (SLAs).
- **Codebase Artifact:** `docs/design_note.md`, `config/config.yaml`
- **Output:** Defined numeric bounds: $R^2 \ge 0.80$, $\text{RMSE} \le 12.0\text{ Lakhs}$, P95 latency $< 50\text{ ms}$.

### Stage 2: Data Ingestion, Hashing & Versioning
- **Syllabus Topic:** Topics 4, 10 (Versioning, Data Management & Pipeline Artifacts)
- **Objective:** Automated ingestion of official datasets (`capstone_house_prices_train.csv`) without manual copying.
- **Codebase Artifact:** `src/data/ingestion.py`
- **Output:** Deterministic 70/15/15 train/val/test splits, cryptographic SHA-256 hashes recorded in `data/processed/ingestion_metadata.json`.

### Stage 3: Data Quality Gate & Safe Failure
- **Syllabus Topic:** Topic 15 (End-to-End MLOps Pipeline & Quality Gates)
- **Objective:** Automated schema enforcement, domain validation, and null limit checks before compute resources are spent.
- **Codebase Artifact:** `src/data/validation.py`
- **Output:** Validation report; raises `DataGateValidationError` on corrupted inputs without altering production state.

### Stage 4: Feature Engineering & Preprocessor Pipeline
- **Syllabus Topic:** Topics 13, 15 (Feature Stores & Preprocessing Pipelines)
- **Objective:** Impute missing values (`Age_Years`, `Distance_to_Center_km`, `Neighborhood_Score`), calculate ratio interactions without target leakage, and save transformation pipeline.
- **Codebase Artifact:** `src/features/preprocessor.py`
- **Output:** Serialized Scikit-Learn transformer saved to `models/preprocessor.joblib`.

### Stage 5: Multi-Model Exploration, Tuning & Tracking
- **Syllabus Topic:** Topics 12, 15 (Training, Hyperparameter Tuning & Experiment Tracking)
- **Objective:** Cross-validate candidate models (Ridge, Random Forest, Gradient Boosting, LightGBM) with 5-fold CV and track in MLflow.
- **Codebase Artifact:** `src/models/train.py`, `src/registry/mlflow_manager.py`
- **Output:** Candidate model serialized to `models/candidate_model.joblib`; all runs, metrics, parameters, and tags logged to `mlflow.db`.

### Stage 6: Model Validation Gate (Champion vs Challenger)
- **Syllabus Topic:** Topic 15 (Automated Deployment Gating)
- **Objective:** Compare candidate performance against absolute thresholds and against the currently deployed Champion.
- **Codebase Artifact:** `src/models/gate.py`
- **Output:** `ModelGateResult(passed=True/False)`; rejects substandard models and keeps the previous model serving cleanly.

### Stage 7: Registry Promotion & Serving Deployment
- **Syllabus Topic:** Topics 2, 6, 15 (Registry Management & Containerized Serving)
- **Objective:** Promote validated model in MLflow Model Registry to `Production`, hot-swap serving artifact `models/active_model.joblib`, and serve through FastAPI.
- **Codebase Artifact:** `src/pipeline/deploy.py`, `src/api/app.py`, `Dockerfile`
- **Output:** Live REST microservice with endpoints `/predict`, `/batch-predict`, `/health`, `/metrics`, and `/lineage`.

### Stage 8: Operational Telemetry & Drift Monitoring
- **Syllabus Topic:** Topics 6, 7 (Endpoint Performance Monitoring & Drift Detection)
- **Objective:** Ingest incoming operational batches (`Batch 1, 2, 3`), detect covariate shift (PSI, KS-test) and performance degradation, trigger automated alert playbook.
- **Codebase Artifact:** `src/monitoring/drift.py`, `src/monitoring/alert.py`, `src/monitoring/dashboard.py`
- **Output:** Detailed JSON drift reports in `reports/drift/`, Prometheus telemetry, and Streamlit operational dashboard.
