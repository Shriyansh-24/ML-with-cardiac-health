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
- [ ] Step 2 — Input form + Module 1 rules-based risk profiler
- [ ] Step 3 — ML training pipeline (`train_model.py`)
- [ ] Step 4 — ML inference (`predictor.py`) wired into results route
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
├── data/                   # Static JSON datasets (Modules 3 & 4)
├── ml/                     # Training pipeline + saved model + model card
├── services/                # Business logic: risk rules, API calls, inference
├── templates/               # Jinja2 templates (base/index/results)
└── static/                  # CSS
```
