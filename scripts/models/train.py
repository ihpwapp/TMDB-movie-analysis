import json
import sys
from datetime import date
from pathlib import Path

# allow `python scripts/models/train.py` to resolve the scripts package
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import joblib
import numpy as np

from scripts.evaluation import (
    ALGORITHMS,
    FEATURE_SETS,
    TRAIN_END_YEAR,
    VAL_END_YEAR,
    XGB_PARAMS,
    _quiet_fit,
    build_matrix,
    evaluate_predictions,
    time_split,
)
from scripts.models.request import build_name_history_stats
from scripts.prepare_dataset import prepare_ml_dataset

# Trained artifacts live in models/ at the repository root
ARTIFACT_DIR = Path(__file__).resolve().parent.parent.parent / "models"

FEATURE_SET = "historical_lead"


def train_model(feature_set_name=FEATURE_SET, artifact_dir=ARTIFACT_DIR):
    # Fit the frozen best config (XGBoost on historical_lead) on train + val,
    # hold out the newest years as the test period, and save every artifact that
    # predict.py needs to reproduce a prediction later.
    ds = prepare_ml_dataset()
    config = FEATURE_SETS[feature_set_name]

    train, val, test = time_split(ds)

    X_train, encoders = build_matrix(train, config, fit_encoders=True)
    X_val, _ = build_matrix(val, config, encoders=encoders)
    X_test, _ = build_matrix(test, config, encoders=encoders)

    y_train = np.log1p(train["revenue"])
    y_val = np.log1p(val["revenue"])
    y_test = np.log1p(test["revenue"])

    model = ALGORITHMS["xgboost"]["model"]()
    _quiet_fit(model, X_train, y_train, X_val, y_val)

    metrics = evaluate_predictions(y_test, model.predict(X_test))

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(artifact_dir / "xgboost_historical_lead.json"))
    joblib.dump(encoders, artifact_dir / "encoders.joblib")
    joblib.dump(build_name_history_stats(ds), artifact_dir / "name_history_stats.joblib")

    meta = {
        "feature_set": feature_set_name,
        "model_type": "xgboost",
        "params": XGB_PARAMS,
        "columns": list(X_train.columns),
        "train_period": f"<={TRAIN_END_YEAR}",
        "validation_period": f"{TRAIN_END_YEAR + 1}-{VAL_END_YEAR}",
        "test_period": f">={VAL_END_YEAR + 1}",
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "test_metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "created": str(date.today()),
    }
    (artifact_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return model, encoders, meta


if __name__ == "__main__":
    model, encoders, meta = train_model()
    print("saved artifacts to", ARTIFACT_DIR)
    print("test metrics:", meta["test_metrics"])