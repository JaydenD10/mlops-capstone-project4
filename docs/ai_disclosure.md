# Generative AI Disclosure Statement

**Course:** Elective II — Machine Learning Operations (MLOps)  
**Project:** Capstone Project 4 — End-to-End MLflow Pipeline from Raw Data to Monitored Deployment  
**Faculty / Evaluator:** Dr. Santosh Kumar  
**Project Team:** Lead MLOps Engineers (Presenter A & Presenter B)  
**Date of Submission:** October 2026  

---

## 1. Overview & Policy Adherence

This document discloses the use of Generative Artificial Intelligence (GenAI) and AI-assisted development tools in the design, implementation, and documentation of MLOps Project 4, in strict compliance with the academic integrity policy and evaluation guidelines set forth for this course.

---

## 2. Tools & Models Utilized

| Tool / Platform | Model Family | Primary Purpose |
| :--- | :--- | :--- |
| **Google Antigravity IDE** | Gemini 3.8 / Flash / Pro | Code assistance, repository navigation, terminal automation, test orchestration |
| **GitHub Copilot / Claude** | Claude 3.5 / GPT-4o | Boilerplate generation, typing annotations, docstring drafting |

---

## 3. Scope of Generative AI Assistance

Generative AI tools were employed strictly as assistive utilities for developer productivity in the following bounded areas:

1. **Boilerplate & Syntactic Scaffolding:**
   - Drafting initial Pydantic schema structures in `config/schema.py`.
   - Generating standard Prometheus client exposition boilerplate in `src/api/metrics.py`.
   - Constructing multi-stage Dockerfile and Docker Compose syntax templates.

2. **UI & CSS Styling:**
   - Writing customized CSS stylesheets for the Streamlit monitoring dashboard (`src/monitoring/dashboard.py`), including enterprise cards, color-coded status badges, and typography resets.

3. **Documentation Structuring & Formatting:**
   - Structuring markdown tables, GitHub-style callout alerts, and mermaid diagram syntax within the `docs/` deliverables.

4. **Test Case Fixtures:**
   - Generating synthetic mock payloads for corrupt data and low-performing model metrics used to test validation gates in `scripts/run_failure_scenarios.py`.

---

## 4. Human Engineering, Verification & Independent Ownership

All core engineering logic, machine learning decisions, statistical analyses, and architectural implementations were independently designed, evaluated, and verified by the team:

| Engineering Component | Human Design & Verification Responsibility |
| :--- | :--- |
| **Problem Formulation & Metric Bounds** | Selected business-appropriate thresholds ($R^2 \ge 0.80$, $\text{RMSE} \le 12.0\text{L}$, $\text{MAE} \le 8.5\text{L}$) grounded in underwriting risk tolerances on median listing values. |
| **Feature Engineering & Modeling** | Conceived domain features (`Area_Per_Room`, `Overall_Desirability`); ran 5-fold cross-validation across Ridge, Random Forest, GBDT, and LightGBM to select the champion model. |
| **Data & Model Gates** | Designed the fail-safe gating logic ensuring that failed data batches or sub-standard models halt promotion while leaving the live production model untouched. |
| **Statistical Drift Analysis** | Derived and verified the Population Stability Index (PSI) calculations, 10-bin quantile discretization, and two-sample Kolmogorov-Smirnov statistical tests. Identified root-cause covariate shift in `Area_sqft` ($\text{PSI} = 0.336$) and geographic concentration doubling in `Central` zone (15.7% to 30.8%). |
| **Rollback & Provenance** | Engineered the automated zero-downtime rollback protocol and verified end-to-end cryptographic lineage back to SHA-256 data hashes and Git commit SHAs. |
| **Empirical Verification** | Personally executed the 21-test pytest suite, live FastAPI test queries, MLflow runs, and the complete 8-stage automated pipeline. |

---

## 5. Affirmation of Academic Integrity

We confirm that:
1. No synthetic, fake, or unverified test passes were fabricated.
2. All official CSV datasets remain strictly unmodified (`SHA-256` verified).
3. The team possesses complete comprehension of every line of code, configuration parameter, and mathematical metric within this repository and is fully prepared to defend the implementation during the live oral demonstration.
