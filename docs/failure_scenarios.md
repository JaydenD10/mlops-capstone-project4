# Safe Failure & Validation Gate Evidence Report (Task 4)

**Objective:** Demonstrate that the MLOps pipeline implements strict validation gates that fail safely when abnormal conditions occur, ensuring zero downtime and zero contamination of the active production serving environment.

---

## 1. Validation Gate Architecture & Safe Failure Design

```
   ┌────────────────────────────────────────────────────────────────────────┐
   │                          DATA VALIDATION GATE                          │
   │                                                                        │
   │ [Incoming Batch] ──► [Schema / Range / Null Checks]                    │
   │                                  │                                     │
   │              ┌───────────────────┴───────────────────┐                 │
   │              ▼                                       ▼                 │
   │       [Passed Gate]                            [Failed Gate]           │
   │              │                                       │                 │
   │              ▼                                       ▼                 │
   │    Proceed to Features                       HALT PIPELINE!            │
   │                                         Explanatory Log Raised         │
   │                                     Production Model v1 Unchanged      │
   └────────────────────────────────────────────────────────────────────────┘

   ┌────────────────────────────────────────────────────────────────────────┐
   │                         MODEL VALIDATION GATE                          │
   │                                                                        │
   │ [Candidate Model] ──► [Quality Thresholds & Champion vs Challenger]    │
   │                                  │                                     │
   │              ┌───────────────────┴───────────────────┐                 │
   │              ▼                                       ▼                 │
   │       [Passed Gate]                            [Failed Gate]           │
   │              │                                       │                 │
   │              ▼                                       ▼                 │
   │   Promote in MLflow                        ABORT DEPLOYMENT!           │
   │  Hot-Swap to Active                 Candidate Discarded/Archived       │
   │                                     Production Model v1 Unchanged      │
   └────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Failing Scenario A: Corrupted / Out-of-Bounds Data Batch

### 2.1 Test Execution Command
```bash
python scripts/run_failure_scenarios.py
```

### 2.2 Injected Anomalies in Batch
A synthetic corrupted batch was submitted with multiple simultaneous schema, type, and distribution corruptions:
1. `Area_sqft = -500.0` (Negative area; violation of physical lower bound of $100\text{ sqft}$).
2. `Bedrooms = 0` (Violation of integer lower bound of $1$).
3. `Bathrooms = 25` (Excessive value; violation of upper bound of $20$).
4. `Age_Years = 250.0` (Violation of maximum building age of $150\text{ years}$).
5. `Distance_to_Center_km = 150.0` (Violation of maximum zone distance of $100\text{ km}$).
6. `Neighborhood_Score = 15.0` (Out of scale; bound is $[1.0, 10.0]$).
7. `Has_Parking = 5` (Invalid non-binary integer).
8. `Property_Condition = "Broken"` (Invalid category; not in `{Poor, Fair, Good, Excellent}`).
9. `City_Zone = "OuterSpace"` (Invalid zone; not in `{North, South, East, West, Central}`).
10. `Price_Lakhs = -10.0` and `Price_Lakhs = None` (Negative and missing training targets).
11. Overall null rate of $18.18\%$ exceeding the strict $10.0\%$ gate ceiling.

### 2.3 Diagnostic Output & Gate Rejection
```text
================================================================================
 DEMONSTRATING SCENARIO A: CORRUPTED DATA BATCH (DATA GATE FAILURE)
================================================================================
Submitting corrupted dataset to Data Validation Gate...

Data Gate Gate Outcome: Passed = False
Detected Violations (16):
  [X] Overall missing value percentage 18.18% exceeds maximum allowed 10.00%
  [X] Column 'Age_Years' has 50.00% missing values, exceeding limit 15.00%
  [X] Column 'Distance_to_Center_km' has 50.00% missing values, exceeding limit 15.00%
  [X] Column 'Neighborhood_Score' has 50.00% missing values, exceeding limit 15.00%
  [X] Column 'Price_Lakhs' has 50.00% missing values, exceeding limit 15.00%
  [X] Target column 'Price_Lakhs' contains 1 null values
  [X] Invalid Property_Condition values found: ['Broken']. Allowed: ['Poor', 'Fair', 'Good', 'Excellent']
  [X] Invalid City_Zone values found: ['OuterSpace']. Allowed: ['North', 'South', 'East', 'West', 'Central']
  [X] Column 'Area_sqft' has 1 values outside allowed range [100.0, 10000.0]. Found min=-500.0, max=1000.0
  [X] Column 'Bedrooms' has 1 values outside allowed range [1.0, 20.0]. Found min=0.0, max=2.0
  [X] Column 'Bathrooms' has 1 values outside allowed range [1.0, 20.0]. Found min=1.0, max=25.0
  [X] Column 'Age_Years' has 1 values outside allowed range [0.0, 150.0]. Found min=250.0, max=250.0
  [X] Column 'Distance_to_Center_km' has 1 values outside allowed range [0.0, 100.0]. Found min=150.0, max=150.0
  [X] Column 'Neighborhood_Score' has 1 values outside allowed range [1.0, 10.0]. Found min=15.0, max=15.0
  [X] Column 'Has_Parking' has 1 values outside allowed range [0.0, 1.0]. Found min=1.0, max=5.0
  [X] Column 'Price_Lakhs' has 1 values outside allowed range [1.0, 2000.0]. Found min=-10.0, max=-10.0

[SAFE BEHAVIOR VERIFIED]: Active Production Model remains v1 untouched.
```

### 2.4 Safe Failure Guarantee
- The pipeline stopped immediately with `DataGateValidationError`.
- No features were built, no model training compute was wasted.
- Production endpoint `/predict` remained completely online, serving with the existing Champion Model (v1).

---

## 3. Failing Scenario B: Substandard / Degraded Candidate Model

### 3.1 Injected Model Quality Defect
A candidate model was evaluated that suffered from extreme underfitting:
- Candidate $R^2 = 0.4215$ (Violates configured minimum threshold of $0.80$).
- Candidate $\text{RMSE} = 18.45\text{ Lakhs}$ (Violates configured maximum limit of $12.0\text{ Lakhs}$).
- Candidate $\text{MAE} = 14.10\text{ Lakhs}$ (Violates configured maximum limit of $8.50\text{ Lakhs}$).
- Candidate $\text{MAPE} = 32.50\%$ (Violates configured maximum limit of $20.0\%$).
- Champion Comparison: Candidate failed to beat the active Champion Model v1 ($\text{RMSE} = 7.74\text{ Lakhs}$).

### 3.2 Diagnostic Output & Gate Rejection
```text
================================================================================
 DEMONSTRATING SCENARIO B: SUBSTANDARD MODEL (MODEL GATE FAILURE)
================================================================================
Evaluating substandard model metrics against Model Gate...
[INFO] Evaluating Candidate Model against Absolute Quality Thresholds:
  Candidate R²:   0.4215 (Required >= 0.8)
  Candidate RMSE: 18.45 Lakhs (Required <= 12.0 Lakhs)
  Candidate MAE:  14.10 Lakhs (Required <= 8.5 Lakhs)
  Candidate MAPE: 32.50% (Required <= 20.00%)
[INFO] Champion (v1) metrics: RMSE=7.74, R²=0.8741
[ERROR] MODEL GATE FAILED: 5 threshold violation(s):
  - Candidate R² (0.4215) is below minimum quality threshold 0.8000
  - Candidate RMSE (18.45 Lakhs) exceeds maximum allowed threshold 12.00 Lakhs
  - Candidate MAE (14.10 Lakhs) exceeds maximum allowed threshold 8.50 Lakhs
  - Candidate MAPE (32.50%) exceeds maximum allowed threshold 20.00%
  - Challenger RMSE (18.45) failed to beat Champion v1 RMSE (7.74)

Model Gate Outcome: Passed = False
Detected Gate Violations (5):
  [X] Candidate R² (0.4215) is below minimum quality threshold 0.8000
  [X] Candidate RMSE (18.45 Lakhs) exceeds maximum allowed threshold 12.00 Lakhs
  [X] Candidate MAE (14.10 Lakhs) exceeds maximum allowed threshold 8.50 Lakhs
  [X] Candidate MAPE (32.50%) exceeds maximum allowed threshold 20.00%
  [X] Challenger RMSE (18.45) failed to beat Champion v1 RMSE (7.74)

[SAFE BEHAVIOR VERIFIED]: Active Production Model remains v1 untouched.
```

### 3.3 Safe Failure Guarantee
- The candidate model was rejected and discarded.
- No promotion occurred in the MLflow Model Registry.
- Active serving file `models/active_model.joblib` was preserved.
- The live endpoint `/predict` continued serving without interruption using Champion Model v1.

---

## 4. Summary Table of Validation Gate Results

| Scenario | Injected Condition | Gate Triggered | Outcome | Production Endpoint Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Scenario A** | Corrupted Data Batch | `DataGateValidationError` | **REJECTED (16 errors)** | **Zero Impact:** Pipeline halted; Model v1 remained live. |
| **Scenario B** | Underfitted Model ($R^2=0.42$) | `ModelGateValidationError` | **REJECTED (5 errors)** | **Zero Impact:** Promotion blocked; Model v1 remained live. |
| **Nominal Run** | Clean Training Dataset | Data & Model Gates | **PASSED** | Promoted to Production Model v1. |
