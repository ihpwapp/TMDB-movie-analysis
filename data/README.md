# data/

This folder holds derived data used by the modelling notebooks.

- `extract_movie_features.csv` — the ML dataset produced by `queries/extract_movie_features.sql`
  (kept at the repo root, not here). Filter: budget > 500k, revenue >= budget/10, runtime >= 30 min.
- `history_features.csv` — unfiltered per-movie extract from `queries/extract_history_features.sql`;
  used exclusively to compute leakage-safe historical features. Regenerated, not committed.
- `processed/ml_features.csv` — full feature table (ML extract + historical features) produced by
  `notebooks/02.feature_engineering.ipynb`. Regenerated, not committed.
- `raw/` — reserved for any future snapshot data.

Never edit the CSVs by hand; regenerate them from the SQL queries.