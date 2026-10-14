# Operational Telemetry & Drift Monitoring Report (Task 7)

**Dataset Analyzed:** `capstone_incoming_batches.csv` (1,500 total listings across 3 batches of 500 each)  
**Reference Baseline:** `capstone_house_prices_train.csv` (4,000 listings)  
**Active Production Model:** `HousePricePredictor` v1 (Ridge Regression with Feature Engineering)  
**Author:** Lead MLOps Engineer  

---

## 1. Executive Summary

Three simulated operational batches of incoming real-estate listings (500 properties each) were processed through the drift monitoring engine. Because asking prices were provided with the batches, both **Covariate Data Drift** (distributional shifts in input features) and **Performance Drift** (ground-truth predictive error degradation) were rigorously quantified:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             BATCH DRIFT SUMMARY                                 │
├─────────┬──────────────┬─────────────┬─────────────┬───────────┬────────────────┤
│  Batch  │ Status Level │ Current RMSE│ Baseline RMSE│ Degradation│ Top Drift Risk │
├─────────┼──────────────┼─────────────┼─────────────┼───────────┼────────────────┤
│ Batch 1 │ 🟢 HEALTHY   │  7.68 Lakhs │  7.74 Lakhs │   -0.8%   │ None (Stable)  │
│ Batch 2 │ 🔴 CRITICAL  │  9.23 Lakhs │  7.74 Lakhs │  +19.2%   │ Perf Degradation│
│ Batch 3 │ 🔴 CRITICAL  │ 16.36 Lakhs │  7.74 Lakhs │ +111.3%   │ Area_sqft + Zone│
└─────────┴──────────────┴─────────────┴─────────────┴───────────┴────────────────┘
```

---

## 2. Statistical Methodology & Alert Threshold Justification

### 2.1 Population Stability Index (PSI)
The primary metric used to quantify covariate feature shift is the Population Stability Index:
$$\text{PSI} = \sum_{i=1}^{k} (P_i - Q_i) \times \ln\left(\frac{P_i}{Q_i}\right)$$
Where $P_i$ is the actual frequency in the incoming batch and $Q_i$ is the reference frequency from the training dataset.

- **$\text{PSI} < 0.10$ (Healthy / Green):** Statistical convention from risk modeling indicating the population has not changed meaningfully.
- **$0.10 \le \text{PSI} < 0.25$ (Warning / Yellow):** Moderate shift. The model can still be served, but telemetry polling should accelerate and data queued for retraining.
- **$\text{PSI} \ge 0.25$ (Critical / Red):** Significant structural shift in the feature population. Model predictions are no longer statistically reliable without adaptation.

### 2.2 Kolmogorov-Smirnov (KS) Two-Sample Test
A non-parametric hypothesis test evaluating whether continuous batch features originate from the historical distribution. Rejection threshold: $p < 0.01$.

### 2.3 Performance Degradation Metric
$$\text{Degradation} \% = \frac{\text{Batch RMSE} - \text{Baseline RMSE}}{\text{Baseline RMSE}} \times 100\%$$
- **Alert Condition:** A degradation $> 15.0\%$ represents unacceptable variance that directly impairs real-estate valuation accuracy.

---

## 3. In-Depth Batch Analysis & Evidence

### 3.1 Batch 1: Stable Operations (Healthy)
- **Sample Size:** 500 listings
- **Mean Price:** ₹ 61.96 Lakhs (vs Training ₹ 61.90 Lakhs)
- **Mean Area:** 1152.29 sqft (vs Training 1150.84 sqft)
- **Evaluated Performance:**
  - **RMSE:** 7.68 Lakhs (Degradation: **-0.8%**, slightly better than baseline)
  - **MAE:** 5.76 Lakhs
  - **$R^2$ Score:** 0.8712
  - **Within $\pm 10\%$:** 58.6%
- **Drift Diagnosis:**
  - All numerical features exhibited $\text{PSI} < 0.05$.
  - Categorical distribution for `City_Zone` and `Property_Condition` matched training data within $\pm 1.2\%$.
- **Action:** `NO_ACTION_REQUIRED`. Standard operations maintained.

---

### 3.2 Batch 2: Emerging Performance Drift (Critical Alert)
- **Sample Size:** 500 listings
- **Mean Price:** ₹ 63.82 Lakhs
- **Mean Area:** 1064.39 sqft (Noticeable drop in average unit size)
- **Evaluated Performance:**
  - **RMSE:** 9.23 Lakhs (Degradation: **+19.2%**, exceeds 15% threshold!)
  - **MAE:** 6.94 Lakhs
  - **$R^2$ Score:** 0.8145
  - **Within $\pm 10\%$:** 48.2%
- **Drift Diagnosis:**
  - Moderate drift in property size and distance distribution.
  - While individual feature PSIs remained below 0.25, the compounding shift produced a **19.2% surge in predictive error**, triggering the operational performance alert.
- **Action:** `TRIGGER_AUTOMATED_RETRAINING_PIPELINE`.

---

### 3.3 Batch 3: Catastrophic Covariate & Concept Drift (Critical Alert)
- **Sample Size:** 500 listings
- **Mean Price:** ₹ 69.61 Lakhs (Surge of +12.5% despite smaller houses!)
- **Mean Area:** 936.32 sqft (Massive -18.6% decrease in property size!)
- **Evaluated Performance:**
  - **RMSE:** 16.36 Lakhs (Catastrophic Degradation: **+111.3%**!)
  - **MAE:** 12.85 Lakhs
  - **$R^2$ Score:** 0.4610
  - **Within $\pm 10\%$:** 26.4%
- **Root Cause Evidence:**
  1. **`Area_sqft` Covariate Shift:** $\text{PSI} = 0.336$ (Critical, $\ge 0.25$). The incoming stock consists predominantly of compact urban apartments rather than standard suburban homes.
  2. **`City_Zone` Geographical Shift:** $\text{PSI} = 0.206$ (Warning). Listings from the premium `Central` zone doubled from **15.6%** in historical data to **30.8%** in Batch 3.
  3. **Concept Drift:** Smaller properties commanding significantly higher asking prices per square foot because of central location concentration. The historical model under-weights central location premiums for compact units.
- **Action:** `TRIGGER_AUTOMATED_RETRAINING_PIPELINE`.

---

## 4. Comprehensive Feature Drift Table

| Feature Name | Feature Type | Batch 1 PSI | Batch 2 PSI | Batch 3 PSI | Maximum Drift Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Area_sqft** | Numerical | 0.021 | 0.084 | **0.336** | 🔴 **CRITICAL** |
| **Bedrooms** | Numerical | 0.015 | 0.048 | **0.197** | 🟡 **WARNING** |
| **Bathrooms** | Numerical | 0.018 | 0.035 | **0.110** | 🟡 **WARNING** |
| **Age_Years** | Numerical | 0.012 | 0.024 | 0.045 | 🟢 **STABLE** |
| **Distance_to_Center_km** | Numerical | 0.019 | 0.062 | 0.081 | 🟢 **STABLE** |
| **Neighborhood_Score** | Numerical | 0.014 | 0.028 | 0.039 | 🟢 **STABLE** |
| **Has_Parking** | Numerical | 0.008 | 0.015 | 0.022 | 🟢 **STABLE** |
| **Property_Condition** | Categorical | 0.016 | 0.042 | 0.038 | 🟢 **STABLE** |
| **City_Zone** | Categorical | 0.025 | 0.071 | **0.206** | 🟡 **WARNING** |

---

## 5. Automated Mitigation Playbook (Action Protocol)

When a Critical Alert fires (as observed in Batch 2 and Batch 3):

```
       [CRITICAL ALERT FIRED]
                  │
                  ▼
   1. DO NOT BLINDLY RE-DEPLOY!
      Keep Champion Model v1 online.
                  │
                  ▼
   2. AUTOMATED RETRAINING TRIGGER
      Re-ingest historical listings + newly accumulated batch listings.
                  │
                  ▼
   3. STRICT QUALITY GATES
      Candidate retrained model must pass Data Gate and Model Gate.
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
   [PASSED]              [FAILED]
       │                     │
       ▼                     ▼
   4. HOT-SWAP           5. ESCALATION
   Promote in MLflow     Keep Model v1 online.
   Deploy Version 2      Page MLOps On-Call Engineer.
```

1. **Active Model Protection:** The current production model remains online. Automatic replacement without validation is prohibited.
2. **Automated Retraining Job:** A background retraining run is initiated where historical training data is merged with the drifted batch data to recalibrate location interaction coefficients.
3. **Model Gate Evaluation:** The retrained model is evaluated against the current Champion. It must meet $R^2 \ge 0.80$ and outperform Champion v1 on the updated distribution.
4. **Promotion or Escalation:**
   - If passed: Automated hot-swap to Version 2 in MLflow Model Registry.
   - If failed: Dispatch an urgent incident notification to the engineering team.
