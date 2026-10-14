# End-to-End Lineage & Registry Provenance Report (Task 5)

**Objective:** Document and demonstrate complete bidirectional lineage: from a live running deployed model, tracing back through the MLflow Model Registry to the exact run ID, git commit, training data cryptographic hash, hyperparameters, and evaluation metrics.

---

## 1. Lineage Architecture

```
   ┌────────────────────────────────────────────────────────────────────────┐
   │                        PROVENANCE LINEAGE GRAPH                        │
   │                                                                        │
   │ [Git Commit SHA]        [Raw Data SHA-256]      [Config YAML v1.0.0]   │
   │        │                         │                        │            │
   │        └────────────────► [MLflow Run ID] ◄───────────────┘            │
   │                                  │                                     │
   │                                  ▼                                     │
   │                       [Model Artifact + Metrics]                       │
   │                                  │                                     │
   │                                  ▼                                     │
   │                     [MLflow Model Registry]                            │
   │                     (Model: HousePricePredictor)                       │
   │                     (Version: 1 | Stage: Production)                   │
   │                                  │                                     │
   │                                  ▼                                     │
   │                   [Active Serving Model on Disk]                       │
   │                   (models/active_model.joblib)                         │
   │                                  │                                     │
   │                                  ▼                                     │
   │                  [Live Microservice: /predict]                         │
   └────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Live Lineage Demonstration Output

Execution of:
```bash
python scripts/run_lineage_demo.py
```

### Verified Live Lineage Trace:
```text
================================================================================
 MLOPS PROJECT 4 - END-TO-END MODEL LINEAGE TRACE
================================================================================
 Registered Model Name : HousePricePredictor
 Deployed Model Version: v1
 Stage Status          : Production
 MLflow Run ID         : eed82011f5374493b4e95cb0b29d456b
 MLflow Run Name       : release_ridge
--------------------------------------------------------------------------------
 PROVENANCE & ARTIFACT TRACEABILITY:
 [1] Code Version (Git): git_commit_local_dev
 [2] Pipeline Config   : v1.0.0
 [3] Train Data SHA-256: 8c0d88cd389764a20b48dc9ee10f0042cc3f133ebde2bd22b5aa77067bbac758
 [4] Raw Data SHA-256  : 8696565fd2a80775efee9574463c81877a23cbf51314a72e5919c81936036a3c
--------------------------------------------------------------------------------
 VERIFIED EVALUATION METRICS:
   - rmse                : 7.7444
   - r2                  : 0.8741
   - mae                 : 5.9033
   - mape                : 0.1052
--------------------------------------------------------------------------------
 MODEL ARTIFACT STORAGE:
   - MLflow Artifact URI: file:C:/Users/JAYDEN/Downloads/mlops/mlruns/1/eed82011f5374493b4e95cb0b29d456b/artifacts
   - Active Disk Path   : models\active_model.joblib
================================================================================
```

---

## 3. Lineage Query API Endpoint
The live serving microservice exposes the same provenance trace dynamically via HTTP:
```bash
curl http://localhost:8000/lineage
```
**JSON Response:**
```json
{
  "registered_model_name": "HousePricePredictor",
  "version": "1",
  "current_stage": "Production",
  "run_id": "eed82011f5374493b4e95cb0b29d456b",
  "run_name": "release_ridge",
  "git_commit": "git_commit_local_dev",
  "data_train_hash": "8c0d88cd389764a20b48dc9ee10f0042cc3f133ebde2bd22b5aa77067bbac758",
  "data_raw_hash": "8696565fd2a80775efee9574463c81877a23cbf51314a72e5919c81936036a3c",
  "config_version": "1.0.0",
  "pipeline_stage": "production_release",
  "metrics": {
    "rmse": 7.7444,
    "r2": 0.8741,
    "mae": 5.9033,
    "mape": 0.1052
  },
  "artifact_uri": "file:C:/Users/JAYDEN/Downloads/mlops/mlruns/1/eed82011f5374493b4e95cb0b29d456b/artifacts",
  "disk_metadata": {
    "deployed_artifact": "models/active_model.joblib"
  }
}
```

---

## 4. Governance & Audit Guarantees
1. **Immutable Cryptographic Hashing:** The training split is fingerprinted with SHA-256 (`8c0d88cd38...`). If even a single cell in the CSV changes, the hash changes, invalidating traceability.
2. **Registry Stage Transition Enforced via Pipeline:** Only the automated `deploy` pipeline stage can transition a model to `Production`. Direct manual transitions through the UI are strictly prohibited.
3. **Artifact Immutability:** Preprocessor (`models/preprocessor.joblib`) and model (`models/active_model.joblib`) are logged together in MLflow under the same parent run.
