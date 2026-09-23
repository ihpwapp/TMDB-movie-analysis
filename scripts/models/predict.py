import json
import sys
from pathlib import Path

# allow `python scripts/models/predict.py` to resolve the scripts package
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from scripts.evaluation import FEATURE_SETS, build_matrix

ARTIFACT_DIR = Path(__file__).resolve().parent.parent.parent / "models"


def load_pipeline(artifact_dir=ARTIFACT_DIR):
    # Load the model, the encoders (fit on train only), the name-based history
    # stats used by the app forms, and the meta record so a caller can reproduce
    # predictions exactly as the experiments did
    model = XGBRegressor()
    model.load_model(str(artifact_dir / "xgboost_historical_lead.json"))
    encoders = joblib.load(artifact_dir / "encoders.joblib")
    name_stats = joblib.load(artifact_dir / "name_history_stats.joblib")
    meta = json.loads((artifact_dir / "meta.json").read_text())
    return {"model": model, "encoders": encoders, "name_stats": name_stats, "meta": meta}


def build_request_matrix(feature_df, pipeline):
    # Apply the saved encoders to a prepared feature frame (no re-fitting)
    if isinstance(feature_df, pd.Series):
        feature_df = pd.DataFrame([feature_df])
    config = FEATURE_SETS[pipeline["meta"]["feature_set"]]
    X, _ = build_matrix(feature_df, config, encoders=pipeline["encoders"], fit_encoders=False)
    return X


def predict_revenue(feature_df, pipeline):
    # feature_df must be a fully prepared row (all numeric/log/flag columns from
    # prepare_ml_dataset). Returns revenue in dollars (back-transformed).
    X = build_request_matrix(feature_df, pipeline)
    pred_log = pipeline["model"].predict(X)
    return np.expm1(pred_log)


if __name__ == "__main__":
    from scripts.evaluation import evaluate_predictions, time_split
    from scripts.prepare_dataset import prepare_ml_dataset

    pipeline = load_pipeline()
    ds = prepare_ml_dataset()
    _, _, test = time_split(ds)

    preds = predict_revenue(test, pipeline)
    metrics = evaluate_predictions(np.log1p(test["revenue"]), np.log1p(preds))
    print("reloaded model test metrics:", {k: round(v, 4) for k, v in metrics.items()})
    print("saved  model test metrics:", pipeline["meta"]["test_metrics"])