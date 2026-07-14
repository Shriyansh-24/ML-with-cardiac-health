# CardioGenome

Predicts genetic cardiac risk from personal/family health data using a real
ML model, then contextualises the result with gene-level data, gene editing
research, and an equity layer showing who actually has access to these
advances.

Educational project — **not a medical diagnosis tool.**

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then visit `http://127.0.0.1:5000`.

## Build status

- [x] Step 1 — Project skeleton (folder structure, `app.py`, `base.html`)
- [x] Step 2 — Input form + Module 1 rules-based risk profiler
      (`services/risk_profiler.py` — scores HCM, LQTS, FH out of 3 criteria each)
- [x] Step 3 — ML training pipeline trained on Framingham Heart Study
      (`ml/train_model.py` — Random Forest, GridSearchCV, 10-fold CV, 0.715 ROC-AUC)
- [x] Step 4 — ML inference + hybrid predictor wired into results route
      (`services/predictor.py` — combines ML output with family history / variant boosts)

**What's next:**
- [ ] Step 5 — Module 2: ClinVar API (MYH7 only)
- [ ] Step 6 — Module 2: GWAS Catalog + remaining genes
- [ ] Step 7 — Module 3: gene editing research dataset + display
- [ ] Step 8 — Module 4: equity dataset + Plotly visualisation
- [ ] Step 9 — Combined results page

## Project structure

```
cardiogenome/
├── app.py                  # Flask routes only — no business logic
├── requirements.txt
├── ml/                     # Training pipeline + saved model (cardiac_model.pkl) + model card
│   ├── train_model.py      # Full pipeline: load, clean, train, tune, save
│   ├── cardiac_model.pkl   # Trained Random Forest (5 features, 0.715 ROC-AUC)
│   └── README.md           # Detailed model card with metrics & limitations
├── services/               # Business logic: risk rules, ML inference, API calls
│   ├── risk_profiler.py    # Module 1: rules-based scoring (0-3 per condition)
│   └── predictor.py        # Step 4: hybrid ML + rules predictor (0-100 per condition)
├── templates/              # Jinja2 templates (base/index/results)
│   ├── base.html           # Shared HTML shell
│   ├── index.html          # Full intake form (15+ fields across 5 fieldsets)
│   └── results.html        # Combined report: ML overview + condition cards
└── static/                 # CSS
```
