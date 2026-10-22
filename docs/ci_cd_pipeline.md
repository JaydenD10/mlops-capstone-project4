# CI/CD Automated Build & Release Pipeline (Task 6)

**Objective:** Document the GitHub Actions CI/CD workflows, pull request gating, image tagging scheme, container builds, and release verification.

---

## 1. CI/CD Architecture & Flow

```
   ┌────────────────────────────────────────────────────────────────────────┐
   │                         PULL REQUEST WORKFLOW                          │
   │                                                                        │
   │ [Developer PR] ──► [1. Ruff Linting] ──► [2. Config Validation]        │
   │                           │                                            │
   │                           ▼                                            │
   │                  [3. Pytest Suite (21 tests)]                          │
   │                           │                                            │
   │                           ▼                                            │
   │             [4. Pipeline Smoke Test (All 7 Stages)]                    │
   │                           │                                            │
   │                           ▼                                            │
   │             [5. Safe Failure Gates Demonstration]                      │
   │                           │                                            │
   │                           ▼                                            │
   │                 [PR Checks Pass: Ready to Merge]                       │
   └────────────────────────────────────────────────────────────────────────┘

   ┌────────────────────────────────────────────────────────────────────────┐
   │                        MERGE TO MAIN (CD RELEASE)                      │
   │                                                                        │
   │ [Merge to Main] ──► [Full Pytest & Pipeline Execution]                 │
   │                               │                                        │
   │                               ▼                                        │
   │                  [Extract Git Short SHA Tag]                           │
   │                               │                                        │
   │                               ▼                                        │
   │              [Build Multi-Stage Docker Container]                      │
   │           Tagged: `house-price-api:${SHORT_SHA}` & `latest`            │
   │                               │                                        │
   │                               ▼                                        │
   │              [Validate Container Runtime Healthcheck]                  │
   │                               │                                        │
   │                               ▼                                        │
   │                 [Zero-Downtime Production Release]                     │
   └────────────────────────────────────────────────────────────────────────┘
```

---

## 2. GitHub Actions Workflows

### 2.1 Continuous Integration (`.github/workflows/ci.yml`)
Triggered on: Pull requests to `main` and pushes to feature branches.
- **Steps:**
  1. **Checkout Code:** Uses `actions/checkout@v4`.
  2. **Python Environment:** Sets up Python 3.11 with cached pip dependencies.
  3. **Linter & Code Quality:** Executes `ruff check .` to enforce PEP8 standards.
  4. **Config Validation:** Executes `python config/schema.py` to ensure zero hardcoded values and schema conformity.
  5. **Automated Test Suite:** Runs `pytest -v` across all 21 unit and integration tests.
  6. **Stage-by-Stage Smoke Test:** Executes individual stages via `python -m src.pipeline.run_stage --stage <name>`.
  7. **Safe Failure Gate Verification:** Executes `python scripts/run_failure_scenarios.py` to confirm that bad data and substandard models are cleanly halted without breaking production.

### 2.2 Continuous Deployment & Release (`.github/workflows/cd.yml`)
Triggered on: Merges to `main` or semantic release tags (`v*.*.*`).
- **Steps:**
  1. Full regression testing.
  2. Executes complete orchestrated pipeline via `python scripts/run_pipeline.py`.
  3. Generates release image tag linked directly to the commit SHA:
     - Tag: `house-price-api:${{ env.SHORT_SHA }}`
     - Tag: `house-price-api:latest`
  4. Builds the container using multi-stage build in `Dockerfile`.
  5. Runs the container locally in the runner and asserts `curl -f http://localhost:8000/health` returns 200 OK.

---

## 3. Container Image Tagging & Traceability
Every production container is tagged with the exact Git commit SHA:
```text
docker build -t house-price-api:eed8201 -t house-price-api:latest .
```
When inspecting a running container in production:
```bash
docker inspect mlops-house-price-api | grep IMAGE
```
The operator immediately discovers the exact commit that produced the running code and model artifact.

<!-- Automated CI Check Verification Note -->
