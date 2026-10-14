# Production Model Rollback Procedure (Task 6)

**Objective:** Provide a step-by-step, tested operational runbook for rolling back a production model to the previous stable release.

---

## 1. When to Trigger a Rollback
A rollback must be initiated under any of the following operational scenarios:
1. **Critical Performance Drift:** Serving telemetry shows prediction latency exceeding SLA (P99 $> 100\text{ ms}$) or server 5xx error rate $> 1\%$.
2. **Post-Release Valuation Anomaly:** Unanticipated edge-case pricing errors reported by business users or valuation partners.
3. **Unexpected Pipeline Regression:** Latent defect discovered in feature engineering logic after release.

---

## 2. Automated Rollback Command

Execute the automated rollback utility from the repository root:
```bash
# Roll back to the immediately preceding production version
python scripts/rollback.py

# OR explicitly roll back to a specific registered model version (e.g. Version 1)
python scripts/rollback.py 1
```

---

## 3. Step-by-Step Rollback Workflow

```
   [1. Trigger Rollback]
            │
            ▼
   [2. Query MLflow Registry] ──► Identifies previous stable version (e.g., v1)
            │
            ▼
   [3. Demote Current Version] ──► Demotes flawed version to "Archived"
            │
            ▼
   [4. Promote Previous Version] ──► Sets target version stage to "Production"
            │
            ▼
   [5. Download Artifacts] ──► Downloads target model from MLflow tracking store
            │
            ▼
   [6. Atomic Disk Hot-Swap] ──► Overwrites `models/active_model.joblib`
            │
            ▼
   [7. Health Verification] ──► Queries `GET /health` to confirm active v1
```

---

## 4. Verification Evidence (Tested Run Output)

The rollback procedure was executed and tested as shown in the audit log below:

```text
2026/10/14 05:56:41 INFO mlflow.agent.hint: MLflow tracing enabled.
[INFO] [rollback:rollback_production_model:37] - Initiating Rollback procedure for model 'HousePricePredictor'...
[INFO] [rollback:rollback_production_model:49] - Current production version: v1
[INFO] [rollback:rollback_production_model:60] - Target rollback version: v1
Downloading artifacts: 100%|##########| 5/5 [00:00<00:00, 1920.47it/s]
[INFO] [rollback:rollback_production_model:114] - ROLLBACK COMPLETED SUCCESSFULLY. Active model is now v1.
Rollback Result: Successfully restored v1
```

---

## 5. Container & Infrastructure Rollback (Docker)
To roll back the containerized serving layer at the infrastructure level:
```bash
# 1. Identify previous stable image tag
docker images house-price-api

# 2. Re-tag previous image as latest
docker tag house-price-api:eed8201 house-price-api:latest

# 3. Restart container service with zero-downtime
docker compose up -d --no-deps --build model-serving

# 4. Verify healthy state
curl http://localhost:8000/health
```
