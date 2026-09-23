from pathlib import Path

import nbformat

NB = Path(__file__).resolve().parent.parent / "notebooks" / "06.model_expansion.ipynb"

cells = []


def md(source):
    cells.append(nbformat.v4.new_markdown_cell(source))


def code(source):
    cells.append(nbformat.v4.new_code_cell(source))


md(
    "# 06 · Model expansion\n"
    "\n"
    "Round 1 compared 4 algorithms (`linear_regression`, `random_forest`, `xgboost`, `lightgbm`)\n"
    "on the frozen `historical_lead` feature set — XGBoost won (test R² **0.5924**).\n"
    "\n"
    "This notebook tests a **second round of model families** under the **exact same protocol**\n"
    "(same time split ≤2016 / 2017–2021 / ≥2022, encoders on train only, `log1p` target,\n"
    "same metrics, same logging):\n"
    "\n"
    "| model | family | notes |\n"
    "|---|---|---|\n"
    "| `ridge` / `lasso` / `elasticnet` | regularized linear | imputed + scaled |\n"
    "| `svr` | kernel (RBF) | imputed + scaled |\n"
    "| `knn` | instance-based | imputed + scaled |\n"
    "| `mlp` | neural net | imputed + scaled |\n"
    "| `histgb` | sklearn gradient boosting | — |\n"
    "| `catboost` | gradient boosting (ordered) | val early-stopping |\n"
    "\n"
    "Plus a validation-weighted **ensemble** of the tree boosters and a small **XGBoost tuning**\n"
    "scan. Every result is appended to `reports/experiments/experiment_log.csv`."
)

code(
    "import numpy as np\n"
    "import pandas as pd\n"
    "from sklearn.metrics import r2_score\n"
    "from xgboost import XGBRegressor\n"
    "\n"
    "from scripts.prepare_dataset import prepare_ml_dataset\n"
    "from scripts.evaluation import (\n"
    "    EXTRA_ALGORITHMS,\n"
    "    FEATURE_SETS,\n"
    "    TRAIN_END_YEAR,\n"
    "    VAL_END_YEAR,\n"
    "    _quiet_fit,\n"
    "    _split_matrices,\n"
    "    blend_algorithms,\n"
    "    compare_extra_algorithms,\n"
    "    evaluate_predictions,\n"
    "    log_experiment,\n"
    ")\n"
    "\n"
    "ds = prepare_ml_dataset()\n"
    "print('rows:', len(ds), '| release_years:', ds['release_year'].min(), '-', ds['release_year'].max())"
)

md(
    "### Round-2 model comparison\n"
    "\n"
    "`compare_extra_algorithms` mirrors `compare_algorithms` from notebook 04 — same split,\n"
    "same matrices, same evaluation, and each run is logged as `alg_<model>`."
)

code(
    "extra = compare_extra_algorithms(ds, 'historical_lead', notes='round-2 model expansion')\n"
    "extra.round(4).sort_values('r2', ascending=False)"
)

md("### Validation-weighted ensemble\n"
   "\n"
   "Average held-out predictions of all the tree models, weighting each member by its\n"
   "**validation** R² — the test split is untouched during weight selection.")

code(
    "ens = blend_algorithms(\n"
    "    ds, 'historical_lead',\n"
    "    ['xgboost', 'lightgbm', 'catboost', 'histgb', 'random_forest'],\n"
    "    notes='tree-ensemble blend',\n"
    ")\n"
    "print('weights:', ens['weights'])\n"
    "pd.Series(ens['test_metrics']).round(4)"
)

md("### XGBoost tuning scan\n"
   "\n"
   "Four hyperparameter combinations, selected on validation R², then re-evaluated on test.")

code(
    "X_train, X_val, X_test, y_train, y_val, y_test = _split_matrices(ds, FEATURE_SETS['historical_lead'])\n"
    "\n"
    "grid = [\n"
    "    dict(learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8),\n"
    "    dict(learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.8),  # round-1 default\n"
    "    dict(learning_rate=0.08, max_depth=8, subsample=0.8, colsample_bytree=0.7),\n"
    "    dict(learning_rate=0.03, max_depth=8, subsample=0.9, colsample_bytree=0.7),\n"
    "]\n"
    "\n"
    "rows = []\n"
    "fits = {}\n"
    "for g in grid:\n"
    "    m = XGBRegressor(\n"
    "        n_estimators=500, random_state=42, verbosity=0, early_stopping_rounds=50, **g\n"
    "    )\n"
    "    _quiet_fit(m, X_train, y_train, X_val, y_val)\n"
    "    val_r2 = r2_score(y_val, m.predict(X_val))\n"
    "    test_r2 = r2_score(y_test, m.predict(X_test))\n"
    "    fits[tuple(sorted(g.items()))] = m\n"
    "    rows.append((g, round(val_r2, 4), round(test_r2, 4)))\n"
    "\n"
    "scan = pd.DataFrame(rows, columns=['params', 'val_r2', 'test_r2'])\n"
    "scan['params'] = scan['params'].astype(str)\n"
    "scan.sort_values('val_r2', ascending=False)"
)

code(
    "best_params = rows[np.argmax([r[1] for r in rows])][0]\n"
    "print('best on val:', best_params)\n"
    "best_model = fits[tuple(sorted(best_params.items()))]\n"
    "\n"
    "mets = evaluate_predictions(y_test, best_model.predict(X_test))\n"
    "log_experiment({\n"
    "    'experiment_id': 'alg_xgboost_tuned',\n"
    "    'feature_set': 'historical_lead',\n"
    "    'model': 'xgboost_tuned',\n"
    "    'hyperparameters': str(best_params),\n"
    "    'train_period': f'<={TRAIN_END_YEAR}',\n"
    "    'validation_period': f'{TRAIN_END_YEAR + 1}-{VAL_END_YEAR}',\n"
    "    'test_period': f'>={VAL_END_YEAR + 1}',\n"
    "    **{k: round(v, 4) if k in ('mae', 'rmse', 'r2') else round(v, 0) for k, v in mets.items()},\n"
    "    'notes': 'selected on val R2 from 4-combo scan',\n"
    "})\n"
    "print(pd.Series(mets).round(4))"
)

md("### Final ranking\n"
   "\n"
   "All round-1 + round-2 + tuned + ensemble results, plus the round-1 champion from notebook 04.")

code(
    "from scripts.evaluation import read_experiment_log\n"
    "log = read_experiment_log()\n"
    "log.index = [f'{r}\u200a·\u200a{log.loc[r,\"model\"]}' for r in log.index]\n"
    "tab = (log[log['feature_set'] == 'historical_lead']\n"
    "       .drop(columns=['feature_set', 'train_period', 'validation_period', 'test_period'])\n"
    "       .sort_values('r2', ascending=False))\n"
    "tab.drop_duplicates('experiment_id').round(4)"
)

md(
    "### Verdict\n"
    "\n"
    "Which round-2 approach improves on round-1's XGBoost (test R² 0.5924)? The table above gives\n"
    "the numbers; the narrative goes in the next markdown cell after inspecting the results."
)

nbformat.write(nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.14"},
}), NB)
print("wrote", NB)