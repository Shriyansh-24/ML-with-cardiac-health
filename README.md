# CardioGenome

Predicts genetic cardiac risk from personal/family health data using a real
ML model (XGBoost, **0.818 ROC-AUC**), then contextualises the result with
gene-level data from ClinVar and GWAS Catalog, gene editing research, and
an equity layer showing who actually has access to these advances.

Educational project — **not a medical diagnosis tool.**

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then visit `http://127.0.0.1:5000`.

## Accounts & saved reports (Phase 2 — Supabase)

The app works fully **without** an account (as in Phase 1). Account features
are opt-in and turn on automatically once you connect Supabase:

1. Create a project at [supabase.com](https://supabase.com) and grab the
   project URL + keys from **Project Settings → API**.
2. Run `sql/schema.sql` in the Supabase **SQL Editor** — it creates the
   `reports` table (with row-level security: users can only see their own
   rows) and the anonymised `research_data` table.
3. Set these env vars (Render Dashboard → Environment, or your shell):

   ```
   SUPABASE_URL=https://<project-ref>.supabase.co
   SUPABASE_ANON_KEY=<anon public key>
   SUPABASE_SERVICE_ROLE_KEY=<service role key>   # server-side research inserts only
   SECRET_KEY=<long random string>                # signs session cookies
   ```

   > ⚠️ The service-role key **bypasses row-level security** — it is used
   > only on the server for anonymised research inserts and never exposed
   > to the browser.

4. Deploy/restart, and the nav bar gains **Log in / Sign up**. During
   development you may want to turn **off email confirmation** in Supabase
   (Authentication → Providers → Email) so signups log in instantly.

### What accounts unlock

- **Save reports** — a "💾 Save this report" bar on the results page; a
  one-time nudge asks "are you sure you don't want to save?" if you skip it.
- **My reports** (`/history`) — list, view, and delete your saved reports.
- **Anonymised research opt-in** — a checkbox at signup; consented reports
  are stored in `research_data` with no name, email, or account link.

## Build status

| Step | Module | What | Status |
|---|---|---|---|
| 1 | Skeleton | Flask app, `base.html`, folder structure | ✅ done |
| 2 | Module 1 | Intake form (17+ fields across 5 fieldsets) + rules-based risk profiler (`services/risk_profiler.py`) | ✅ done |
| 3 | ML model | **NHANES XGBoost** training pipeline (`ml/train_nhanes.py`, 16,842 samples, SMOTE, GridSearchCV) — 0.818 ROC-AUC, a **+0.10 improvement** over the retired Framingham model. The old Framingham Random Forest (`ml/train_model.py`) and its model file have been removed from the repository. | ✅ done |
| 4 | ML pipeline | Hybrid ML + rules predictor (`services/predictor.py`) — combines XGBoost probability with condition-specific family history / variant boosts. Single model (NHANES only), no fallback complexity. | ✅ done |
| 5 | Module 2 | **ClinVar API** (`services/clinvar_api.py`) — fetches real MYH7 variant data from NCBI E-utilities with 1-hour in-memory caching. Shows total variant count, clinical significance breakdown, and notable variants on the HCM card. | ✅ done |
| 6 | Module 2 | **GWAS Catalog API** (`services/gwas_api.py`) — fetches real gene-disease associations from EMBL-EBI GWAS Catalog API v2 for LQTS (KCNQ1, KCNH2, SCN5A) and FH (LDLR, APOB) genes. Shows trait summary pills and expandable notable-variant list with rsID, p-value, risk allele, and effect size. | ✅ done |
| 7 | Module 3 | **Gene editing research** (`data/gene_editing.json`) — static dataset with CRISPR, base editing, and gene therapy approaches for HCM, LQTS, and FH, displayed in the "Gene Editing Research" tab. | ✅ done |
| 8 | Module 4 | **Equity dashboard** (`data/equity.json` + `services/equity.py`) — 4 interactive Plotly bar charts examining disparities in genetic testing access by race, income, clinical trial representation, and global access to cardiac gene therapies. Displayed in the "Equity & Access" tab. | ✅ done |
| 9 | Integration | Final polish — responsive breakpoints, GWAS trait chips limited to top 8 with "+X more" pill, gene chip tags, combined page reviewed for cohesive display across all 4 modules. | ✅ done |

### Additional UI/UX enhancements

- **Loading screen** — full-screen dark overlay with animated beating heart SVG, pulse rings, bouncing dots, cycling status text, and progress bars that appear on form submission.
- **Score bar animations** — each condition card has a smooth animated gradient score bar (green → yellow → orange → red) that fills in on page load.
- **Card hover effects** — cards lift slightly with a subtle shadow on hover.
- **Risk label fix** — replaced ambiguous Low/Moderate/High labels on condition cards with clear "Additional factors detected" / "No additional factors" tags that accurately reflect whether condition-specific evidence was found.
- **Flag badge** — when cardiac risk probability exceeds 20%, a pulsing "⚠ Flagged" badge lights up in the overview card.
- **Model source badge** — overview card shows a blue "NHANES" badge (or red "Framingham" for fallback) with tooltip indicating the dataset sample size.
- **BMI from form data** — optional height (cm) and weight (kg) fields in the intake form let the predictor calculate real BMI instead of using the imputed median. The overview note dynamically acknowledges this.
- **Tabbed results layout** — the combined results page is split into 3 tabs (Risk Results, Equity & Access, Gene Editing Research) with a sliding-indicator navigation bar, making the page far less overwhelming while keeping all advanced data accessible. The overview card remains visible above all tabs.
- **Responsive design** — the 2-column card grid collapses to single-column on mobile; the tab bar stacks vertically; all sections have appropriate breakpoints.

## Project structure

```
cardiogenome/
├── app.py                  # Flask routes only — no business logic
├── requirements.txt
├── sql/
│   └── schema.sql          # Phase 2: Supabase tables + row-level security
├── ml/                     # Training pipeline + saved model + model card
│   ├── train_nhanes.py     # NHANES XGBoost pipeline: load, clean, train,
│   │                       #   tune (GridSearchCV), save (0.818 ROC-AUC)
│   ├── cardiac_model_nhanes.pkl  # Trained XGBoost (7 features, 0.818 ROC-AUC)
│   └── README.md           # Detailed model card with metrics & limitations
├── services/               # Business logic: risk rules, ML inference, API calls
│   ├── risk_profiler.py    # Module 1: rules-based scoring (0-3 per condition)
│   ├── predictor.py        # Hybrid ML + rules predictor (0-100 per condition)
│   ├── clinvar_api.py      # Module 2: ClinVar E-utilities fetcher (MYH7 variants)
│   ├── gwas_api.py         # Step 6: GWAS Catalog v2 associations (LQTS & FH)
│   ├── equity.py           # Module 4: equity dashboard — Plotly chart HTML
│   └── auth_service.py     # Phase 2: Supabase auth + saved-report persistence
├── templates/              # Jinja2 templates
│   ├── base.html           # Shared HTML shell (header, nav, footer, flashes)
│   ├── index.html          # Full intake form (17+ fields across 5 fieldsets)
│   ├── results.html        # Tabbed report: ML overview + condition cards
│   ├── signup.html         # Phase 2: account creation + research opt-in
│   ├── login.html          # Phase 2: email/password sign-in
│   ├── history.html        # Phase 2: saved reports list
│   └── report_detail.html  # Phase 2: single saved report view
├── data/                   # Static datasets
│   ├── gene_editing.json   # Module 3: CRISPR, base editing, gene therapy
│   └── equity.json         # Module 4: disparities datasets (4 charts)
└── static/
    └── style.css           # All styles (form, results, tabs, loading, equity)
```
