# Final Demonstration Script & Walkthrough (Task 8)

**Presentation Duration:** 5 to 10 Minutes  
**Format:** Live Walkthrough / Screen Recording  
**Team Allocation:** 2 Partners (Presenter A & Presenter B)  
**Project:** MLOps Capstone Project 4 — Automated Property Price Pipeline  

---

## 1. Demonstration Outline & Timing

| Minute | Topic | Lead Presenter | Key Live Action / Screen |
| :--- | :--- | :--- | :--- |
| **0:00 – 1:30** | Problem Framing, Metrics & Architecture | **Presenter A** | Show `docs/design_note.md` & `architecture_diagram.svg` |
| **1:30 – 3:30** | Single-Source Config & Full Pipeline Run | **Presenter B** | Show `config.yaml`, run `python scripts/run_pipeline.py` |
| **3:30 – 5:00** | MLflow Registry & Live Lineage Trace | **Presenter A** | Show MLflow UI & run `python scripts/run_lineage_demo.py` |
| **5:00 – 6:45** | Serving API & Live Prediction Telemetry | **Presenter B** | Query `/health`, `/predict`, `/metrics` (Prometheus) |
| **6:45 – 8:30** | Operational Drift over Batches 1, 2, 3 | **Presenter A** | Open Streamlit Dashboard (`src/monitoring/dashboard.py`) |
| **8:30 – 10:00**| Safe Failure Gates & Rollback Demo | **Presenter B** | Run `scripts/run_failure_scenarios.py` & `rollback.py` |

---

## 2. Minute-by-Minute Speaking Script

### Minute 0:00 – 1:30 | Problem Framing & System Architecture
**Speaker: Presenter A**
> *"Hello everyone. Today we are presenting our end-to-end MLOps pipeline for our real-estate analytics startup.
> Our business objective is predicting property asking prices in INR Lakhs. Historically, models were trained in messy ad-hoc notebooks with zero lineage and no validation gates.
> We set strict quantitative success thresholds: an R² greater than 0.80, and an MAE under 8.5 Lakhs. For our non-technical founders, our achieved MAE of 5.90 Lakhs means our algorithm values properties within ±5.90 Lakhs on average, and over 57% of all properties are within 10% of their true asking price.
> As shown in our architecture diagram, our system moves data through 8 automated lifecycle stages: from raw data hashing and schema validation gates, through feature engineering, cross-validation, MLflow registry promotion, containerized FastAPI serving, and finally continuous drift monitoring across incoming operational batches."*

---

### Minute 1:30 – 3:30 | Single-Source Configuration & Pipeline Run
**Speaker: Presenter B**
> *"I will now demonstrate our single source of truth configuration and pipeline orchestration.
> In `config/config.yaml`, every single parameter—from dataset paths and split ratios to gate thresholds and model hyperparameters—is defined in one place. We validate this on load using Pydantic in `config/schema.py`, ensuring zero hard-coded magic numbers in our code.
> Notice that any single stage can be re-run independently using `python -m src.pipeline.run_stage --stage <name>`.
> Now, I will trigger the full end-to-end pipeline live by running `python scripts/run_pipeline.py`.
> In just 15 seconds, the pipeline ingests the 4,000 listings, verifies SHA-256 checksums, validates the data gate, builds ratio features, cross-validates 4 algorithms, evaluates the model gate, logs everything to MLflow, promotes the winning model to Production, and analyzes drift over the incoming batches."*

---

### Minute 3:30 – 5:00 | MLflow Registry & Live Lineage Demonstration
**Speaker: Presenter A**
> *"Now let's examine our experiment tracking and lineage guarantees in MLflow.
> In our local SQLite database, MLflow tracks every run with parameters, cross-validation metrics, and preprocessor artifacts. Our winning model is registered as `HousePricePredictor` and promoted to `Production`.
> Watch what happens when I execute `python scripts/run_lineage_demo.py`.
> Starting directly from our live running model Version 1, the system prints a complete provenance trace: we can see the exact MLflow Run ID (`eed8201...`), the Git commit SHA, the configuration version, the verified Test RMSE (7.74L) and R² (0.8741), and the cryptographic SHA-256 hash of the exact training data that produced this model. This gives our startup complete regulatory auditability."*

---

### Minute 5:00 – 6:45 | Containerized Serving & Real-Time API
**Speaker: Presenter B**
> *"Next is our serving layer. Our model is packaged inside a multi-stage Docker container running a FastAPI microservice.
> Let's query the health endpoint: `GET /health` returns 200 OK, confirming the model and preprocessor are loaded in memory.
> Now let's send a single property listing to `POST /predict`. Within 12 milliseconds—well below our 50ms SLA—the API returns an estimated price of INR 103.26 Lakhs along with an empirical 95% confidence interval and founder note.
> If we inspect `GET /metrics`, our Prometheus instrumentation tracks total request counts, latency histograms, error counters, and the distribution of predicted prices in real time."*

---

### Minute 6:45 – 8:30 | Operational Drift Monitoring (Batches 1, 2, 3)
**Speaker: Presenter A**
> *"Here in our Streamlit monitoring dashboard, we process the 3 incoming operational batches of 500 listings each.
> - **Batch 1** is completely Healthy: feature distributions and model RMSE (7.68L) match our historical baseline.
> - **Batch 2** flags a Warning and Performance Alert: average house size drops, pushing RMSE degradation to +19.2%.
> - **Batch 3** triggers a Critical Alert: we see massive covariate shift in `Area_sqft` with a PSI of 0.336, and listings from the Central zone doubled from 15% to 31%. Because compact central units command a massive premium per square foot, the historical model suffers a catastrophic +111% error increase.
> Our automated alert playbook immediately recommends triggering our automated retraining pipeline to incorporate the new urban apartment distribution."*

---

### Minute 8:30 – 10:00 | Safe Failure Demonstration & Automated Rollback
**Speaker: Presenter B**
> *"Finally, we demonstrate safe failure and zero-downtime rollback.
> By running `python scripts/run_failure_scenarios.py`, we test two failure modes:
> 1. In **Scenario A**, we submit a corrupted data batch with negative square footage, invalid zones, and missing targets. The Data Gate instantly rejects it with 16 explicit error diagnostics.
> 2. In **Scenario B**, we evaluate an underfitted candidate model with an R² of 0.42. The Model Gate immediately halts promotion.
> Crucially, in both failing scenarios, our production endpoint and active model remain 100% online and untouched.
> If we ever need to revert a release, running `python scripts/rollback.py 1` transitions the previous version back to Production and hot-swaps the serving artifact atomically.
> That completes our demonstration. We are now happy to answer your questions!"*
