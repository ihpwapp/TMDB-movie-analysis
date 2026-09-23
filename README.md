# TMDB Hollywood ROI — Movie Revenue Prediction

Predicting how much a movie will earn at the box office using only information known
**before release**, so the model can support a greenlight decision ("should this film be made?").
Data comes from a PostgreSQL database populated by a separate TMDB ingestion repo.

---

## The problem

**Questions**
- Is this movie going to be commercially successful?
- Which features correlate most with movie success?
- How much can movie metadata predict revenue?

**Beneficiaries**: studios, producers, and cast/agents deciding whether a project gets greenlit.

**Success definition** (business metric, used for analysis — not as a model feature):
a movie makes **2.5× its budget** in revenue.

**Prediction point**: before release / at greenlight. Everything used must be known then.

---

## The data

- `queries/extract_movie_features.sql` → the **ML dataset** `extract_movie_features.csv` (8,040 films).
  Filters: `budget > $500k & revenue >= budget/10 & runtime >= 30min`. This is the dataset the
  whole pipeline was built on and is deliberately kept as-is (it conditions on revenue, see
  *Limitations*).
- `queries/extract_history_features.sql` → an **unfiltered** history extract
  (14,895 films → `data/history_features.csv`) used *only* to compute leakage-safe historical
  features from prior releases.
- Sources span ~1925–2027; revenue/budget are heavily skewed and modelled via `log1p`.

**Reserved columns** (`movie_popularity`, `vote_average`, `vote_count`) are **not** used — they
are measured after release, i.e. leak the future.

---

## What was built (this session)

### 1. SQL extracts (extended)
`extract_movie_features.sql` gained: `genre_count`, `production_company_count`,
`production_country_count`, `cast_size`, `crew_size`, `producer_count`, `writer_count`,
`composer_count`, `keyword_count`, and keyword flags `is_superhero`, `is_sequel`, `is_remake`,
`is_based_on_novel`, `is_based_on_true_story`.

### 2. Scripts (`scripts/`, matching the repo's existing style)
| file | purpose |
|---|---|
| `load_data.py` | read the extract + history CSVs (or re-run the SQL) |
| `validate_data.py` | duplicate / null / target sanity checks |
| `feature_engineering.py` | release features (quarter/decade), `log1p` transforms, frequency encoding helpers |
| `historical_features.py` | **leakage-safe** prior-release aggregates + a brute-force leak test |
| `prepare_dataset.py` | merges extract + historical features into the modelling table |
| `evaluation.py` | time split, feature-set configs, experiment runnner, experiment log, algorithm comparison |

### 3. Feature engineering
- **Basic**: budget/runtime (log), release year/month/quarter/day-of-week, language, genre, country (one-hot, min frequency 40).
- **Production**: production company (frequency-encoded, fit on train only), company/country counts.
- **People**: cast/crew sizes, producer/writer/composer counts, director (frequency-encoded).
- **Historical** (leakage-safe): for director/company/cast/franchise — prior movie count, avg
  revenue / budget / ROI, max revenue, computed **only from films released strictly earlier**
  (`release_date < current`; same-day releases treated as concurrent and excluded).
- **Star power**: `lead_actor_previous_avg_revenue` — prior revenue of the **first-billed** actor.
- **Advanced**: keyword count and keyword flags.

Every encoding is fit on the **train split only**; val/test just transform.

### 4. Validation & leakage control
- `scripts/historical_features.py` has a **leak test** that recomputes each historical feature
  brute-force for a random sample and asserts equality (currently passes for all 4 feature families).
- Manual sanity checks (e.g. *Avengers: Infinity War*'s historical features use only pre-2018 films).

### 5. Baselines & experiments (reports/experiments/experiment_log.csv)
Out-of-time split: **train ≤ 2016 (6,262) · val 2017–2021 (916) · test ≥ 2022 (862)**.

Fixed methodology across experiments: XGBoost, same seed (42), same split, same metrics
(MAE/RMSE/R² on `log1p(revenue)`, plus raw-$ RMSE). One feature group changes at a time.

| experiment | test R² | test log-RMSE | vs previous |
|---|---|---|---|
| naive (median) | −0.001 | 1.638 | — |
| linear (numbers only) | 0.475 | 1.187 | — |
| XGBoost · `basic` | 0.505 | 1.152 | — |
| `production` | 0.563 | 1.083 | +0.058 |
| `people` | 0.567 | 1.078 | +0.004 |
| `historical` | 0.589 | 1.049 | +0.022 |
| `historical_lead` (**best**) | **0.592** | 1.045 | +0.003 |
| `advanced` | 0.587 | 1.052 | −0.002 |

**Algorithm comparison** on the frozen `historical_lead` set:
**XGBoost 0.592** > LightGBM 0.571 ≈ Linear 0.578 > RandomForest 0.567.

**Round 2 — model expansion** (notebook `06`), same split/protocol, no improvement:
CatBoost 0.589 is the best challenger; lasso/elasticnet/ridge ≈ the 0.578 linear ceiling;
HistGB 0.571; SVR 0.453; kNN 0.435; MLP 0.209; a val-R²-weighted blend of the 5 tree models
= 0.589; a val-selected XGBoost tuning scan = 0.591 — all below the robust round-1 default.
XGBoost stays the production model (`models/` untouched).

Verdict: with pre-release information only, XGBoost on `historical_lead` is the best model —
it cuts test error roughly in half vs the naive guess (log-RMSE 1.045 vs 1.638).

**Classification validation — thresholding the existing model (notebook `07`, no retraining):**
- **4 revenue tiers** (`< $25M / $25-100M / $100-500M / ≥ $500M`): usable — test accuracy **0.607**
  (vs 0.486 majority), macro-F1 **0.564** (vs 0.164).
- **Greenlight hurdle (revenue ≥ 2.5× budget): effectively a coin flip** — test ROC-AUC **0.511**.
  Root cause: predicted revenue lands at ~1.03× the hurdle on average, so nearly every film sits
  right on the decision boundary and residual noise decides the class. A dedicated classifier
  (Route B) is needed for a trustworthy go/no-go — this result is the evidence for that.

### 6. Notebooks (numbered, executed top-to-bottom)
- `01.EDA.ipynb` — distributions, correlations, missing values (pre-existing).
- `02.feature_engineering.ipynb` — builds + validates every feature, leak demo, saves `data/processed/ml_features.csv`.
- `03.baseline.ipynb` — naive → linear → XGBoost baselines.
- `04.model_comparison.ipynb` — the six feature-stage experiments + algorithm comparison.
- `05.error_analysis.ipynb` — why the model is wrong: residual scale, worst over/under-predictions, error by budget tier / genre / era / franchise / history coverage.
- `06.model_expansion.ipynb` — round-2 model zoo (regularized linear, SVR, kNN, MLP, HistGB, CatBoost), val-R²-weighted tree ensemble, and an XGBoost tuning scan; XGBoost still wins.
- `07.classification_eval.ipynb` — validates the regressor as a classifier by thresholding (greenlight + 4 revenue tiers); confusion matrices saved to `reports/figures/`, metrics to `reports/classification_metrics.csv`.

### 7. Final prediction pipeline
Reusable package `scripts/models/` (reuses the same functions as the notebooks — no duplicated preprocessing):
- `scripts/models/train.py` — fits XGBoost on the frozen `historical_lead` set (train+val, test ≥ 2022 held out),
  saves `models/xgboost_historical_lead.json`, `models/encoders.joblib` and `models/meta.json` (test metrics: **R² 0.592, log-RMSE 1.045**).
- `scripts/models/predict.py` — `load_pipeline()` reloads the artifacts and `predict_revenue()` turns a prepared
  feature row back into dollars; verified to reproduce the notebook test metrics exactly.
```bash
python scripts/models/train.py    # recreate artifacts
python scripts/models/predict.py  # sanity check: reload + score the test set
```

### 8. Streamlit app (`app/streamlit_app.py`)
Two tabs that **call the same pipeline functions** (`scripts/models/predict.py` + `request.py` — no
preprocessing is duplicated inside the app):
- **Movie Lookup & Edit** — pick any movie in the dataset (exact id-based features), tweak
  budget/runtime/release/genre/company/director, and see the re-prediction. A **Clear results**
  button empties the prediction/SHAP panel while keeping the form edits intact.
- **Greenlight Form** — describe a not-yet-released film (budget, genre, director, stars,
  franchise, keywords…) → revenue estimate with the 2.5× greenlight verdict. A **template picker**
  auto-fills the form from this dataset's typical stats for **Superhero blockbuster / Indie film /
  Comedy / Animation** (numbers = category medians; director/company/star/franchise default to the
  most common names in the data; everything stays editable — blank = no track record).
  > ⚠️ The 2.5× verdict is this model thresholded — notebook `07` shows it ranks greenlights at
  > ~AUC 0.51 (coin flip); treat it as a stretch/goal sanity-check, not a go/no-go system.
  > A dedicated classifier is the documented next step.
- Every prediction is explained with **SHAP** (waterfall + top-driver bar + contribution
  table with approximate $-moves). Track-record lookups are name-based and tolerant of
  typos/suffixes ("warner bros" matches *Warner Bros. Pictures*), approximate by nature
  (from the reference filmography) and flagged as such.
```bash
streamlit run app/streamlit_app.py
```

### 9. Tests (`tests/`)
Targeted unit tests for the reusable logic (fast, no dataset load):
`test_features.py` (release/log/frequency features), `test_historical.py` (leakage-safe
prior semantics + brute-force verification), `test_preprocessing.py` (train-only encoders,
unseen→0), `test_request.py` (free-form/`edit` feature builder + labels),
`test_presets.py` (greenlight-form templates) and `test_classification.py` (thresholded
greenlight + tier helpers).
```bash
python -m pytest tests/ -q
```

### 10. Reproducing
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # fill in DATABASE_URL (needed only to re-generate CSVs)
jupyter notebook notebooks/
```
The CSVs are already generated; notebooks run against them without the DB.
`scripts/` can also be run directly: `python scripts/load_data.py`, `python scripts/historical_features.py`.

---

## What I can improve (recommended next steps)

**Data / methodology**
- Widen the ML dataset: the current extract drops flops via `revenue >= budget/10`, which
  biases the target. A release-point dataset that includes flops (and handles zero/missing
  revenue) is the single biggest improvement to the story.
- Add gross-inflation adjustment for older films (budget/revenue in today's dollars).
- Marketing spend is a strong predictor that the extract lacks.
- **Train a dedicated binary classifier** for the greenlight/go-no-go decision
  (Route B, same features as `historical_lead`, `objective=binary:logistic`) — notebook `07`
  shows thresholded regression is a coin flip (AUC 0.51), so a purpose-trained classifier is
  the clear next build.

**Process**
- Unit tests for `feature_engineering`, `historical_features`, and the `models` request builder.
- Polish the final README with the Streamlit demo and error-analysis findings.

---

## Current status
- ✅ Data layer, feature engineering, baseline, incremental feature experiments, algorithm comparison
- ✅ Error analysis notebook (`05.error_analysis.ipynb`)
- ✅ Final prediction pipeline (`scripts/models/train.py` + `predict.py`, artifacts in `models/`)
- ✅ Streamlit app (`app/streamlit_app.py`, two tabs, reuses the pipeline)
- ✅ Tests (`tests/`, 14 passing)
- ⬜ Final docs polish (Streamlit demo section, limitations, future work)

## Known limitations
- Dataset is revenue-conditioned (flops underrepresented) → predictions are optimistic.
- No inflation adjustment; no marketing spend; no WOM/post-release signals (by design).
- Franchise/collection coverage is sparse; historical features are `NaN` for debut directors/actors.
- The DB holds some future/filming releases — excluded by the revenue filter, but the release-year span matters for the time split.