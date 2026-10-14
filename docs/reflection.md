# Engineering Reflection & Future Improvements (Deliverable 8)

**Authors:** MLOps Capstone Project Team  
**Context:** Capstone Project 4 — Reflection Memo on Production Hardening  

---

## 1. What Went Well

1. **Robust Gate Architecture:** The separation of the Data Validation Gate and the Model Validation Gate was the most impactful architectural decision. When running failure simulations, the ability of the pipeline to halt cleanly and leave the production endpoint completely untouched proved that safety does not require complex orchestrators—just strict contracts.
2. **Deterministic Reproducibility:** Leveraging a single configuration file (`config/config.yaml`) validated by Pydantic completely eliminated hardcoded parameters. The random seed isolation in the 70/15/15 stratified split ensured that training metrics (Val RMSE 7.52L, Test $R^2$ 0.8741) were exactly reproduced on every run.
3. **Actionable Drift Findings:** The drift engine uncovered clear, actionable insights in the provided operational datasets: Batch 3 was not merely random noise; it represented a genuine structural market shift toward compact, high-value Central urban apartments (Central zone proportion doubling from 15% to 31%, Area PSI = 0.336).

---

## 2. What We Would Improve with More Time

### 2.1 Feature Store Integration (Feast / Hopsworks)
- **Current Limitation:** In our current pipeline, feature engineering happens inside the training and inference pipeline using Scikit-Learn transformers.
- **Future Enhancement:** With more time, we would implement an open-source Feature Store (such as **Feast**). This would enable centralized feature definitions, historical point-in-time joins to completely eradicate training-serving skew, and real-time online feature serving via Redis.

### 2.2 Automated Retraining Loop via Message Queues
- **Current Limitation:** When Batch 2 or Batch 3 triggers a Critical Drift alert, our playbook recommends executing retraining, but triggering the retrain requires either a webhook invocation or CLI execution.
- **Future Enhancement:** Implement an event-driven architecture using Kafka or Celery. Upon detecting $\text{PSI} \ge 0.25$ or performance degradation $> 15\%$, a message would automatically enqueue an isolated background retraining container, run the challenger model through the Model Gate, and hot-swap to Version 2 without human intervention.

### 2.3 Shadow Deployments & Canary Traffic Splitting
- **Current Limitation:** Model deployment currently operates on an atomic hot-swap pattern (Version 1 is immediately replaced by Version 2).
- **Future Enhancement:** Deploying through an Envoy/Traefik reverse proxy or Kubernetes service mesh to support **Shadow Deployments** (where new candidate models receive 100% of live traffic in parallel without returning predictions to users) and **Canary Deployments** (routing 10% of user traffic to the challenger model to observe real-world latency and error distributions before full rollout).

### 2.4 Deep Learning & Geolocation Embeddings
- **Current Limitation:** The `City_Zone` feature is currently represented using one-hot encoding across 5 zones.
- **Future Enhancement:** If latitude and longitude coordinates were captured, we would introduce spatial geospatial embeddings (e.g., H3 hex-bins or Graph Neural Networks over neighborhood amenity nodes) to model micro-market pricing dynamics with greater granularity.

### 2.5 Explainability & SHAP Values in Live API
- **Current Limitation:** The API currently returns predicted prices and confidence intervals, but does not return local feature attributions.
- **Future Enhancement:** Incorporate TreeSHAP / LinearSHAP into the FastAPI `/predict` payload so that every prediction returns a waterfall breakdown of top factors driving that specific listing's valuation (e.g. `+12.4 Lakhs for Central Zone`, `-3.2 Lakhs for Age`).
