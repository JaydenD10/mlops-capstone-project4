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
| 13 | Passing CI Workflow Run | `06_cicd/13_passing_ci.png` | **MISSING** | **Reason:** No remote GitHub repository is linked (`git remote -v` is empty). GitHub Actions workflow runs have not been executed on GitHub. |
| 14 | Failing CI Workflow Run | `06_cicd/14_failing_ci.png` | **MISSING** | **Reason:** No remote GitHub repository is linked. Pull request checks on GitHub Actions have not been executed on GitHub. |
| 15 | Successful Release / Build Run | `06_cicd/15_successful_release.png` | **MISSING** | **Reason:** No remote GitHub repository is linked. CD release workflows on GitHub Actions have not been executed on GitHub. |
| 16 | Docker Image Build | `07_deployment/16_docker_image.png` | **PASS** | `docker build -t house-price-api:latest .` completed successfully. `docker images` confirms `house-price-api:latest` (ID: `8cff6d6ff400`, 1.45 GB, Python 3.11-slim multi-stage build). Docker Desktop v29.9.2 / Engine v29.9.2. |
| 17 | Running Container | `07_deployment/17_running_container.png` | **PASS** | Container `house-price-container` launched with `docker run -d -p 8000:8000`. `docker ps` shows `Up 11 seconds (healthy)`, port `0.0.0.0:8000→8000/tcp`. `GET /health` → `{"status":"healthy", "model_loaded":true, "model_version":"6"}`. `POST /predict` → `predicted_price_lakhs: 103.26`, latency 32.76 ms. |
| 18 | API Prediction Response | `07_deployment/18_api_prediction.png` | **PASS** | Live `POST /predict` call returning HTTP 200 with property price estimation (₹116.74 Lakhs), confidence bands, and latency metrics. |
| 19 | Operational Monitoring Metrics | `08_monitoring/19_operational_metrics.png` | **PASS** | Streamlit Operations dashboard showing total requests (5,500), 100% success rate, latency metrics (12.4ms avg, 24.8ms P95), throughput (142 req/s), and SLA compliance chart. |
| 20 | Data Drift Across Batches 1, 2, 3 | `08_monitoring/20_data_drift.png` | **PASS** | Streamlit Feature Drift Analysis (PSI Table) tracking all 9 features across Batches 1, 2, 3 with critical drift flagged on `Area_sqft` ($\text{PSI} = 0.3357$). |
| 21 | Performance Drift Tracking | `08_monitoring/21_performance_drift.png` | **PASS** | Streamlit Model Health Overview showing Batch 1 (Healthy), Batch 2 (+19.2% RMSE degradation), Batch 3 (+111.3% RMSE degradation), and error charts. |
| 22 | Live Alerts & Mitigation Runbook | `08_monitoring/22_alerts.png` | **PASS** | Streamlit Alert & Mitigation Center showing Critical Alert for Batch 3, Critical Alert for Batch 2, Healthy status for Batch 1, and 6-step runbook playbook. |
| 23 | GitHub Repository Structure | `09_git/23_repository.png` | **MISSING** | **Reason:** No remote GitHub repository is configured (`git remote` is empty; local project has not been published or pushed to a remote GitHub URL). |
| 24 | Git Commit History & Collaboration | `09_git/24_commit_history.png` | **PASS** | Git commit history showing commits from both team members (`MLOps Capstone` and `Presenter B (Monitoring & Serving)`). |

---

## Detailed Explanation of Missing Items

### 1. Items 13, 14, 15 (GitHub Actions / CI-CD)
- **What is missing:** Web screenshot evidence of GitHub Actions workflow runs (passing PR test run, failing PR test run, and successful post-merge release build).
- **Exact technical cause:** Running `git remote -v` in the repository outputs no remote URL. The repository exists purely on the local filesystem. Without a live repository on GitHub, GitHub Actions cannot execute workflows or generate run history pages.
- **Workflow files present in repo:** The repository contains complete workflow specifications in `.github/workflows/ci.yml` and `.github/workflows/cd.yml`.

### 2. Items 16, 17 (Docker / Container Deployment) — ✅ RESOLVED
- **Resolved:** Docker Desktop 4.96.2 (Engine v29.9.2) was located at `%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin\docker.exe` and added to PATH.
- `docker build -t house-price-api:latest .` completed successfully (multi-stage Python 3.11-slim build, 1.45 GB image, ID `8cff6d6ff400`).
- `docker run -d -p 8000:8000 --name house-price-container house-price-api:latest` launched the container.
- `docker ps` confirmed status `Up (healthy)` with port `0.0.0.0:8000→8000/tcp`.
- Live API verification: `GET /health` → `{status: healthy, model_loaded: true, model_version: 6}`.
- Live prediction: `POST /predict` → `predicted_price_lakhs: 103.26 Lakhs`, `latency_ms: 32.76`.

### 3. Item 23 (GitHub Repository Structure)
- **What is missing:** Web screenshot of the GitHub repository homepage showing file tree, README, and commit badges.
- **Exact technical cause:** No GitHub remote is configured (`git config --get remote.origin.url` is empty). The repository has not yet been pushed to GitHub.

---

## Submission Checklist Summary

1. **Total screenshots successfully captured:** **20 / 24**
2. **Missing screenshots:** **4 / 24** (Items 13, 14, 15, 23)
3. **Project functionality preventing collection:**
   - ~~Absence of Docker Desktop~~ — **RESOLVED**: Docker Desktop 4.96.2 confirmed installed and operational. Items 16 and 17 are now PASS.
   - Absence of a configured remote GitHub repository prevents executing GitHub Actions workflows and viewing GitHub web pages.
4. **Commands you need to run manually (to obtain the 4 remaining missing screenshots):**
   - **For GitHub (Screenshots 13, 14, 15, 23):**
     1. Create a GitHub repository (e.g., `https://github.com/<your-username>/mlops-capstone-project4`).
     2. Add the remote and push:
        ```bash
        git remote add origin https://github.com/<your-username>/mlops-capstone-project4.git
        git branch -M main
        git push -u origin main
        ```
     3. Open a Pull Request from a feature branch to trigger the CI workflow (`.github/workflows/ci.yml`).
     4. Create a branch with an intentional lint/test error to trigger a failing PR workflow for Screenshot 14.
     5. Merge the passing PR to trigger the CD release workflow (`.github/workflows/cd.yml`) for Screenshot 15.
   - **For Docker (Screenshots 16, 17):** ✅ **COMPLETED on 2026-10-22**
     - `docker build -t house-price-api:latest .` — **DONE** (image ID `8cff6d6ff400`, 1.45 GB)
     - `docker run -d -p 8000:8000 --name house-price-container house-price-api:latest` — **DONE** (container `5860b54f9d7b`, status `healthy`)
     - `GET /health` → `{status: healthy, model_version: 6}` — **VERIFIED**
     - `POST /predict` → `predicted_price_lakhs: 103.26`, latency 32.76 ms — **VERIFIED**
5. **Browser pages you need to open manually:**
   - GitHub Actions tab: `https://github.com/<your-username>/mlops-capstone-project4/actions`
   - GitHub Repository home: `https://github.com/<your-username>/mlops-capstone-project4`
6. **Coverage of Project Brief Requirements:**
   - **Core MLOps Pipeline & Governance:** **100% COVERED** (Architecture, Configuration, DAG Orchestration, Pipeline Execution, Stage Logs, MLflow Experiment Tracking, MLflow Model Registry, Provenance Lineage, Data Gate Safe Failure, Model Gate Safe Failure, Real-time API Inference, Operational Telemetry, Feature Drift Analysis across Batches 1–3, Performance Drift Tracking, Alert Policies & Runbook, Git Commit Multi-Author Collaboration).
   - **External Cloud / Virtualization Services:** **Requires manual cloud/host setup** (Remote GitHub hosting & Docker Desktop engine).
