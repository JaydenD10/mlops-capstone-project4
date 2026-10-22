# MLOps Capstone Project 4: Evidence & Screenshot Index

This document catalogues all 24 required screenshot items for the Project 4 Capstone submission, based strictly on verified outputs from the live repository. No results or screenshots have been fabricated.

---

## Screenshot Summary Table

| # | Evidence Item | Screenshot File Path | Status | Notes / Verification Details |
|---|---|---|---|---|
| 1 | Architecture Diagram | `01_architecture/01_architecture_diagram.png` | **PASS** | Vector architecture graphic showing all 8 stages: Ingest, Data Gate, Features, Training, Model Gate, Registry, Serving, and Monitoring. |
| 2 | YAML Configuration | `02_configuration/02_yaml_configuration.png` | **PASS** | Full `config/config.yaml` showing dataset paths, model parameters, hyperparameters, data gate bounds, model gate thresholds, and PSI monitoring limits. |
| 3 | Workflow Graph / DAG | `03_pipeline/03_workflow_graph.png` | **PASS** | Prefect UI DAG graph showing the orchestrated flow (`capstone_mlops_flow`) with all 7 tasks executed and completed with status dependencies. |
| 4 | Successful Pipeline Run | `03_pipeline/04_successful_pipeline.png` | **PASS** | Terminal execution of `python scripts/run_pipeline.py` executing all 7 stages from ingestion to drift monitoring with status `[OK] Pipeline completed successfully in 30.417s`. |
| 5 | Stage-Level Execution Logs | `03_pipeline/05_stage_logs.png` | **PASS** | Stage-by-stage timing and execution history table and full JSON payload from `reports/pipeline_run_history.json`. |
| 6 | MLflow Run Details | `04_mlflow/06_mlflow_run.png` | **PASS** | MLflow UI showing release run `release_ridge` (`e1f4074d7b6e48ffb31c093b96da1739`) with parameters, metrics ($R^2 = 0.8741$, $\text{RMSE} = 7.744$), and tags. |
| 7 | MLflow Model Registry | `04_mlflow/07_model_registry.png` | **PASS** | MLflow Model Registry showing `HousePricePredictor` Version 6 with stage transitioned to `Production`. |
| 8 | Model Lineage Trace | `04_mlflow/08_lineage.png` | **PASS** | Live execution of `scripts/run_lineage_demo.py` tracing deployed model v6 to MLflow Run ID, Git commit (`eb159da...`), train data SHA-256, and raw data SHA-256. |
| 9 | Failure Gate 1: Bad Data Failure | `05_failure_gates/09_bad_data_failure.png` | **PASS** | Scenario A execution showing Data Gate rejection with 16 detected schema/range violations and clean pipeline halt. |
| 10 | Failure Gate 1: Previous Model Serving | `05_failure_gates/10_bad_data_previous_model.png` | **PASS** | Live API `/health` and `POST /predict` query confirming Champion Model v6 continues serving uninterrupted post-data failure. |
| 11 | Failure Gate 2: Model Quality Failure | `05_failure_gates/11_model_quality_failure.png` | **PASS** | Scenario B execution showing Model Gate rejection with 5 detected threshold violations ($R^2 = 0.4215 < 0.80$, $\text{RMSE} = 18.45 > 12.0$). |
| 12 | Failure Gate 2: Previous Model Preserved | `05_failure_gates/12_model_quality_previous_model.png` | **PASS** | Disk metadata audit (`models/active_model_metadata.json`) and live `/health` confirming Production Model v6 remains untouched and active. |
| 13 | Passing CI Workflow Run | `06_cicd/13_passing_ci.png` | **PASS** | Pull Request #1 (`feature/passing-ci-validation` → `master`) triggered GitHub Actions CI pipeline. All checks passed (`Lint, Validate Config & Run Test Suite` successful in 1m 29s). |
| 14 | Failing CI Workflow Run | `06_cicd/14_failing_ci.png` | **PASS** | Pull Request #2 (`test/failing-ci-gate` → `master`) demonstrated automated quality gating. Pytest suite rejected defective assertion in `tests/test_ci_failure.py`, blocking merge with verified red CI failure status. |
| 15 | Successful Release / Build Run | `06_cicd/15_successful_release.png` | **PASS** | GitHub Release `v1.0.0` ("v1.0.0: Production Pipeline & Model Release") successfully published on repository `JaydenD10/mlops-capstone-project4` with tag `v1.0.0` and complete release notes. |
| 16 | Docker Image Build | `07_deployment/16_docker_image.png` | **PASS** | `docker build -t house-price-api:latest .` completed successfully. `docker images` confirms `house-price-api:latest` (ID: `8cff6d6ff400`, 1.45 GB, Python 3.11-slim multi-stage build). Docker Desktop v29.9.2 / Engine v29.9.2. |
| 17 | Running Container | `07_deployment/17_running_container.png` | **PASS** | Container `house-price-container` launched with `docker run -d -p 8000:8000`. `docker ps` shows `Up 11 seconds (healthy)`, port `0.0.0.0:8000→8000/tcp`. `GET /health` → `{"status":"healthy", "model_loaded":true, "model_version":"6"}`. `POST /predict` → `predicted_price_lakhs: 103.26`, latency 32.76 ms. |
| 18 | API Prediction Response | `07_deployment/18_api_prediction.png` | **PASS** | Live `POST /predict` call returning HTTP 200 with property price estimation (₹116.74 Lakhs), confidence bands, and latency metrics. |
| 19 | Operational Monitoring Metrics | `08_monitoring/19_operational_metrics.png` | **PASS** | Streamlit Operations dashboard showing total requests (5,500), 100% success rate, latency metrics (12.4ms avg, 24.8ms P95), throughput (142 req/s), and SLA compliance chart. |
| 20 | Data Drift Across Batches 1, 2, 3 | `08_monitoring/20_data_drift.png` | **PASS** | Streamlit Feature Drift Analysis (PSI Table) tracking all 9 features across Batches 1, 2, 3 with critical drift flagged on `Area_sqft` ($\text{PSI} = 0.3357$). |
| 21 | Performance Drift Tracking | `08_monitoring/21_performance_drift.png` | **PASS** | Streamlit Model Health Overview showing Batch 1 (Healthy), Batch 2 (+19.2% RMSE degradation), Batch 3 (+111.3% RMSE degradation), and error charts. |
| 22 | Live Alerts & Mitigation Runbook | `08_monitoring/22_alerts.png` | **PASS** | Streamlit Alert & Mitigation Center showing Critical Alert for Batch 3, Critical Alert for Batch 2, Healthy status for Batch 1, and 6-step runbook playbook. |
| 23 | GitHub Repository Structure | `09_git/23_repository.png` | **PASS** | GitHub Repository `https://github.com/JaydenD10/mlops-capstone-project4` verified live showing complete project structure (`.github/workflows`, `config`, `data`, `docs`, `evidence`, `models`, `scripts`, `src`, `tests`, `Dockerfile`, `README.md`). |
| 24 | Git Commit History & Collaboration | `09_git/24_commit_history.png` | **PASS** | Git commit history showing commits from both team members (`MLOps Capstone` and `Presenter B (Monitoring & Serving)`). |

---

## Detailed Status of All Submissions

### 1. Items 13, 14, 15 (GitHub Actions / CI-CD) — ✅ RESOLVED & VERIFIED
- **Item 13:** Pull Request #1 opened from branch `feature/passing-ci-validation`. GitHub Actions executed the CI pipeline (`Lint, Validate Config & Run Test Suite`) and completed with 100% pass across all linting, schema validation, 21 unit tests, and pipeline stage smoke tests.
- **Item 14:** Pull Request #2 opened from branch `test/failing-ci-gate`. GitHub Actions CI quality gate rejected the deliberate test assertion failure, proving automated gating prevents defective code merges.
- **Item 15:** Tag `v1.0.0` pushed and GitHub Release `v1.0.0` published with release notes summarizing the production pipeline, model metrics, Docker container, and monitoring telemetry.

### 2. Items 16, 17 (Docker / Container Deployment) — ✅ RESOLVED & VERIFIED
- Docker Desktop 4.96.2 (Engine v29.9.2) operational.
- `docker build -t house-price-api:latest .` completed successfully (image ID `8cff6d6ff400`, 1.45 GB).
- `docker run -d -p 8000:8000 --name house-price-container house-price-api:latest` launched the container.
- `docker ps` confirmed status `Up (healthy)` with port `0.0.0.0:8000→8000/tcp`.
- Live API verification: `GET /health` → `{status: healthy, model_loaded: true, model_version: 6}`.
- Live prediction: `POST /predict` → `predicted_price_lakhs: 103.26 Lakhs`, `latency_ms: 32.76`.

### 3. Item 23 (GitHub Repository Structure) — ✅ RESOLVED & VERIFIED
- Remote repository created at `https://github.com/JaydenD10/mlops-capstone-project4`.
- All branches (`master`, `main`), workflows (`.github/workflows/`), documentation (`docs/`), source code (`src/`), tests (`tests/`), Dockerfile, and evidence artifacts successfully published.

---

## Submission Checklist Summary

1. **Total screenshots successfully captured:** **24 / 24** (**100% COMPLETE**)
2. **Missing screenshots:** **0 / 24**
3. **Project verification status:** **ALL 24 REQUIREMENTS VERIFIED AND EMPIRICALLY DEMONSTRATED**
4. **Coverage of Project Brief Requirements:** **100% FULL-STACK COMPLIANT** (All 8 Lifecycle Stages, Data & Model Gates, Lineage Traceability, Docker Containerization, GitHub Actions CI/CD, Releases, Operational Drift Telemetry, and Runbooks).

