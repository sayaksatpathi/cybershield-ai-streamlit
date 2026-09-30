# 🛡️ CyberShield AI — Fraud Detection System

An ML-powered web app that flags fraudulent transactions from uploaded datasets, with an interactive analytics dashboard.

**🔗 Live:** https://cybershield-ai-app-vmbevtd5fcdrjfexcthga5.streamlit.app/

---

## Overview

Upload a transaction dataset (CSV) and CyberShield runs it through an ensemble of classic ML models to surface likely fraud, then presents results and analytics in a web dashboard. Built to make fraud screening approachable for non-technical users.

## Features

- **5-model ensemble** — Random Forest, Gradient Boosting, Logistic Regression, SVM, Isolation Forest.
- **CSV upload & scoring** — drag-and-drop datasets (large-file support).
- **Analytics dashboard** — result breakdowns and reporting.
- **Web interface** — Streamlit front end, deployed on Streamlit Community Cloud.

## Tech Stack

| Layer | Tools |
|------|-------|
| ML | scikit-learn (Random Forest, Gradient Boosting, Logistic Regression, SVM, Isolation Forest), pandas, NumPy |
| App | Streamlit (primary), Flask (API backend) |
| Deploy | Streamlit Community Cloud |

## Quick Start

```bash
git clone https://github.com/sayaksatpathi/cybershield-ai-streamlit.git
cd cybershield-ai-streamlit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## How it works

1. Upload a transaction CSV.
2. Each model scores the rows; the ensemble aggregates the signals.
3. The dashboard shows flagged transactions and summary analytics.

## Roadmap / Honest Notes

- Add labelled benchmark results (precision/recall on a held-out set) so accuracy claims are reproducible.
- Add model-explainability (feature importance / SHAP) for flagged transactions.
- Containerize for one-command deployment.

---

**Developed by Sayak Satpathi.**
