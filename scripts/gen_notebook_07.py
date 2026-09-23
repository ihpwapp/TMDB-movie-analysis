from pathlib import Path

import nbformat

NB = Path(__file__).resolve().parent.parent / "notebooks" / "07.classification_eval.ipynb"

cells = []


def md(source):
    cells.append(nbformat.v4.new_markdown_cell(source))


def code(source):
    cells.append(nbformat.v4.new_code_cell(source))


md(
    "# 07 · Classification validation (no retraining)\n"
    "\n"
    "The production model is a **regressor** (XGBoost, `historical_lead`, test R² 0.5924). This\n"
    "notebook validates it as a **classifier by thresholding** — Route A, exactly the greenlight\n"
    "verdict the Streamlit app renders today. No new models are trained and `models/` is untouched.\n"
    "\n"
    "Two tasks:\n"
    "\n"
    "| task | label | positive share (test) |\n"
    "|---|---|---|\n"
    "| **binary greenlight** | `revenue >= 2.5 * budget` (i.e. `roi >= 1.5`) | 38% |\n"
    "| **4 revenue tiers** | `< $25M / $25-100M / $100-500M / >= $500M` | 49/25/20/6% |\n"
    "\n"
    "Same frozen protocol: time split ≤2016 / 2017–2021 / ≥2022 and encoders fit on train only."
)

code(
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "\n"
    "from scripts.prepare_dataset import prepare_ml_dataset\n"
    "from scripts.evaluation import (\n"
    "    FEATURE_SETS, _split_matrices, time_split, greenlight_label,\n"
    "    binary_threshold_metrics, tune_greenlight_offset,\n"
    "    revenue_tier, tier_metrics, REVENUE_TIERS,\n"
    ")\n"
    "from scripts.models.predict import load_pipeline\n"
    "\n"
    "ds = prepare_ml_dataset()\n"
    "X_train, X_val, X_test, y_train, y_val, y_test = _split_matrices(ds, FEATURE_SETS['historical_lead'])\n"
    "_, val0, test0 = time_split(ds)\n"
    "rev_val, bud_val = val0['revenue'].to_numpy(), val0['budget'].to_numpy()\n"
    "rev_test, bud_test = test0['revenue'].to_numpy(), test0['budget'].to_numpy()\n"
    "\n"
    "model = load_pipeline()['model']\n"
    "pred_val = model.predict(X_val)   # log1p revenue\n"
    "pred_test = model.predict(X_test)\n"
    "\n"
    "print('test rows:', len(rev_test), '| greenlight share: %.3f' % greenlight_label(rev_test, bud_test).mean())"
)

md(
    "### Binary: greenlight (revenue ≥ 2.5× budget)\n"
    "\n"
    "The model's predicted revenue is compared to the hurdle directly (the **natural cutoff**), then\n"
    "an additive log-space **offset is tuned on validation** (max F1) and applied on test."
)

code(
    "natural_val = binary_threshold_metrics(y_val, pred_val, rev_val, bud_val)\n"
    "natural_test = binary_threshold_metrics(y_test, pred_test, rev_test, bud_test)\n"
    "\n"
    "best_offset, tuned_val = tune_greenlight_offset(y_val, pred_val, rev_val, bud_val)\n"
    "hurdle = np.log1p(bud_test * 2.5)\n"
    "tuned_test = binary_threshold_metrics(y_test, pred_test + best_offset, rev_test, bud_test)\n"
    "\n"
    "share_neg = 1 - greenlight_label(rev_test, bud_test).mean()\n"
    "bl = dict(accuracy=share_neg, precision=0.0, recall=0.0, f1=0.0, roc_auc=0.5, pr_auc=0.0)\n"
    "\n"
    "table = pd.DataFrame({\n"
    "    'val · natural': natural_val,\n"
    "    'val · tuned': tuned_val,\n"
    "    'test · natural': natural_test,\n"
    "    'test · tuned': tuned_test,\n"
    "    'test · majority baseline': bl,\n"
    "}).round(3)\n"
    "table"
)

code(
    "from sklearn.metrics import confusion_matrix\n"
    "y_true = greenlight_label(rev_test, bud_test).astype(int)\n"
    "pred = (pred_test + best_offset >= np.log1p(bud_test * 2.5)).astype(int)\n"
    "cm = pd.DataFrame(confusion_matrix(y_true, pred),\n"
    "                  index=['actual no', 'actual greenlight'],\n"
    "                  columns=['pred no', 'pred greenlight'])\n"
    "fig, ax = plt.subplots(figsize=(4, 3.2))\n"
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)\n"
    "ax.set_title('Greenlight confusion matrix (test, tuned cutoff)')\n"
    "plt.tight_layout()\n"
    "plt.savefig('../reports/figures/classification_binary_confusion.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "print('chosen val offset: %+.2f' % best_offset)"
)

md(
    "### Multiclass: four revenue tiers\n"
    "\n"
    "The regression's predicted revenue is mapped to the nearest band; no retraining, no per-class\n"
    "probability model. Compared against always predicting the test's most common tier."
)

code(
    "tier_test = tier_metrics(y_test, pred_test, rev_test)\n"
    "\n"
    "labels = [t[2] for t in REVENUE_TIERS]\n"
    "actual_tier = revenue_tier(rev_test)\n"
    "majority_tier = actual_tier.mode().iloc[0]\n"
    "pred_majority = pd.Series(majority_tier, index=actual_tier.index)\n"
    "\n"
    "from sklearn.metrics import f1_score\n"
    "b_acc = float((actual_tier == majority_tier).mean())\n"
    "b_f1 = f1_score(actual_tier, pred_majority, labels=labels, average='macro', zero_division=0)\n"
    "tier_rows = {\n"
    "    'accuracy': tier_test['accuracy'],\n"
    "    'balanced_accuracy': tier_test['balanced_accuracy'],\n"
    "    'macro_f1': tier_test['macro_f1'],\n"
    "    'weighted_f1': tier_test['weighted_f1'],\n"
    "    'majority_baseline_accuracy': b_acc,\n"
    "    'majority_baseline_macro_f1': b_f1,\n"
    "}\n"
    "pd.Series(tier_rows).round(3)"
)

code(
    "fig, ax = plt.subplots(figsize=(5.2, 4))\n"
    "sns.heatmap(tier_test['confusion'].astype(int), annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)\n"
    "ax.set_title('Revenue-tier confusion matrix (test)')\n"
    "plt.tight_layout()\n"
    "plt.savefig('../reports/figures/classification_tiers_confusion.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
)

code(
    "from pathlib import Path\n"
    "summary = pd.DataFrame({\n"
    "    'task': ['greenlight'] * 5 + ['tiers'] * 4,\n"
    "    'metric': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc',\n"
    "               'accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1'],\n"
    "    'test_value': [\n"
    "        tuned_test['accuracy'], tuned_test['precision'], tuned_test['recall'],\n"
    "        tuned_test['f1'], tuned_test['roc_auc'],\n"
    "        tier_test['accuracy'], tier_test['balanced_accuracy'],\n"
    "        tier_test['macro_f1'], tier_test['weighted_f1'],\n"
    "    ],\n"
    "})\n"
    "out = Path('../reports') / 'classification_metrics.csv'\n"
    "out.parent.mkdir(exist_ok=True)\n"
    "summary.to_csv(out, index=False)\n"
    "print('wrote', out.resolve())"
)

md(
    "### Verdict\n"
    "\n"
    "Full numbers above; the summary answers 'can the existing model be used for classification?':\n"
    "yes — as a threshold classifier on the greenlight hurdle (ROC-AUC / confusion matrix above) and\n"
    "as a coarse four-tier sorter. The honest caveat: the `>= $500M` tier is rare and the regressor\n"
    "under-calls the biggest outliers, so rare-tier accuracy is weak by construction."
)

nbformat.write(nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.14"},
}), NB)
print("wrote", NB)