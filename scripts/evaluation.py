import io
import os
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBClassifier, XGBRegressor

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    CatBoostRegressor = None
    HAS_CATBOOST = False

from scripts.feature_engineering import apply_frequency_map, create_frequency_map

TRAIN_END_YEAR = 2016
VAL_END_YEAR = 2021
RANDOM_STATE = 42

EXPERIMENT_LOG = Path(__file__).resolve().parent.parent / "reports" / "experiments" / "experiment_log.csv"

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
    "early_stopping_rounds": 20,
}

# Cumulative feature sets: each stage adds to the previous one
FEATURE_SETS = {
    "basic": {
        "numeric": [
            "log_budget", "log_runtime", "release_year", "release_month",
            "release_quarter", "release_day_of_week",
        ],
        "freq": [],
        "ohe": ["original_language", "primary_genre", "primary_country"],
    },
    "production": {
        "numeric": [
            "log_budget", "log_runtime", "release_year", "release_month",
            "release_quarter", "release_day_of_week",
            "production_company_count", "production_country_count",
        ],
        "freq": ["primary_production_company"],
        "ohe": ["original_language", "primary_genre", "primary_country"],
    },
    "people": {
        "numeric": [
            "log_budget", "log_runtime", "release_year", "release_month",
            "release_quarter", "release_day_of_week",
            "production_company_count", "production_country_count",
            "cast_size", "crew_size", "producer_count", "writer_count", "composer_count",
        ],
        "freq": ["primary_production_company", "director_name"],
        "ohe": ["original_language", "primary_genre", "primary_country"],
    },
    "historical": {
        "numeric": [
            "log_budget", "log_runtime", "release_year", "release_month",
            "release_quarter", "release_day_of_week",
            "production_company_count", "production_country_count",
            "cast_size", "crew_size", "producer_count", "writer_count", "composer_count",
            "director_previous_movie_count", "log_director_previous_avg_revenue",
            "log_director_previous_avg_budget", "director_previous_avg_roi",
            "log_director_previous_max_revenue", "log_production_company_previous_avg_revenue",
            "log_cast_previous_avg_revenue", "log_franchise_previous_avg_revenue",
            "franchise_previous_avg_roi", "franchise_movie_count",
        ],
        "freq": ["primary_production_company", "director_name"],
        "ohe": ["original_language", "primary_genre", "primary_country"],
    },
    "historical_lead": {
        "numeric": [
            "log_budget", "log_runtime", "release_year", "release_month",
            "release_quarter", "release_day_of_week",
            "production_company_count", "production_country_count",
            "cast_size", "crew_size", "producer_count", "writer_count", "composer_count",
            "director_previous_movie_count", "log_director_previous_avg_revenue",
            "log_director_previous_avg_budget", "director_previous_avg_roi",
            "log_director_previous_max_revenue", "log_production_company_previous_avg_revenue",
            "log_cast_previous_avg_revenue", "log_lead_actor_previous_avg_revenue",
            "log_franchise_previous_avg_revenue",
            "franchise_previous_avg_roi", "franchise_movie_count",
        ],
        "freq": ["primary_production_company", "director_name"],
        "ohe": ["original_language", "primary_genre", "primary_country"],
    },
    "advanced": {
        "numeric": [
            "log_budget", "log_runtime", "release_year", "release_month",
            "release_quarter", "release_day_of_week",
            "production_company_count", "production_country_count",
            "cast_size", "crew_size", "producer_count", "writer_count", "composer_count",
            "keyword_count",
            "director_previous_movie_count", "log_director_previous_avg_revenue",
            "log_director_previous_avg_budget", "director_previous_avg_roi",
            "log_director_previous_max_revenue", "log_production_company_previous_avg_revenue",
            "log_cast_previous_avg_revenue", "log_franchise_previous_avg_revenue",
            "franchise_previous_avg_roi", "franchise_movie_count",
        ],
        "freq": ["primary_production_company", "director_name"],
        "ohe": ["original_language", "primary_genre", "primary_country"],
        "flags": ["is_superhero", "is_sequel", "is_remake", "is_based_on_novel", "is_based_on_true_story"],
    },
}

ALGORITHMS = {
    "linear_regression": {
        "model": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]),
        "params": "SimpleImputer(median)+StandardScaler+LinearRegression",
        "uses_eval": False,
    },
    "random_forest": {
        "model": lambda: RandomForestRegressor(
            n_estimators=500, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "params": "n_estimators=500,min_samples_leaf=5,seed=42",
        "uses_eval": False,
    },
    "xgboost": {
        "model": lambda: XGBRegressor(**XGB_PARAMS),
        "params": str(XGB_PARAMS),
        "uses_eval": True,
    },
    "lightgbm": {
        "model": lambda: LGBMRegressor(
            n_estimators=500, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1,
        ),
        "params": "n_estimators=500,lr=0.05,subsample=0.9,colsample=0.8,seed=42",
        "uses_eval": True,
    },
}

# Second round of modelling approaches: regularized linear, kernel, instance-based,
# neural, sklearn/distributed gradients. Fitted under the exact same protocol.
EXTRA_ALGORITHMS = {
    "ridge": {
        "model": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=10.0)),
        ]),
        "params": "Ridge(alpha=10), imputed+scaled",
        "uses_eval": False,
    },
    "lasso": {
        "model": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Lasso(alpha=0.001, max_iter=10000)),
        ]),
        "params": "Lasso(alpha=1e-3), imputed+scaled",
        "uses_eval": False,
    },
    "elasticnet": {
        "model": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)),
        ]),
        "params": "ElasticNet(alpha=1e-3,l1_ratio=0.5), imputed+scaled",
        "uses_eval": False,
    },
    "svr": {
        "model": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", SVR(C=10.0, epsilon=0.1, gamma="scale")),
        ]),
        "params": "SVR(RBF,C=10,eps=0.1), imputed+scaled",
        "uses_eval": False,
    },
    "knn": {
        "model": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1)),
        ]),
        "params": "kNN(k=15,dist-weight), imputed+scaled",
        "uses_eval": False,
    },
    "mlp": {
        "model": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=800,
                learning_rate_init=1e-3, random_state=RANDOM_STATE,
            )),
        ]),
        "params": "MLP(64-32,adam,lr=1e-3,iters=800), imputed+scaled",
        "uses_eval": False,
    },
    "histgb": {
        "model": lambda: HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
            random_state=RANDOM_STATE,
        ),
        "params": "HistGB(max_iter=500,lr=0.05,leaf_nodes=31)",
        "uses_eval": False,
    },
}

if HAS_CATBOOST:
    EXTRA_ALGORITHMS["catboost"] = {
        "model": lambda: CatBoostRegressor(
            iterations=1000, learning_rate=0.05, depth=6,
            loss_function="RMSE", random_seed=RANDOM_STATE,
            early_stopping_rounds=20, verbose=0, thread_count=-1,
        ),
        "params": "CatBoost(iters=1000,lr=0.05,depth=6), early-stop=20",
        "uses_eval": True,
    }


def _all_algorithm_specs():
    return {**ALGORITHMS, **EXTRA_ALGORITHMS}


def time_split(df):
    # Out-of-time validation: train = oldest, val = middle, test = newest
    train = df[df["release_year"] <= TRAIN_END_YEAR].copy()
    val = df[(df["release_year"] > TRAIN_END_YEAR) & (df["release_year"] <= VAL_END_YEAR)].copy()
    test = df[df["release_year"] > VAL_END_YEAR].copy()
    return train, val, test


def build_matrix(feature_df, config, encoders=None, fit_encoders=False):
    # Encoders are fit on the train split only; val/test transform reuses them
    flags = config.get("flags", [])
    numeric = config["numeric"] + flags

    X = pd.DataFrame(index=feature_df.index)
    for col in numeric:
        X[col] = feature_df[col]

    if fit_encoders:
        encoders = {"freq": {}, "ohe": None}
        for col in config["freq"]:
            encoders["freq"][col] = create_frequency_map(feature_df[col])
        if config["ohe"]:
            encoders["ohe"] = OneHotEncoder(
                min_frequency=40, handle_unknown="infrequent_if_exist", sparse_output=False
            )
            encoders["ohe"].fit(feature_df[config["ohe"]])

    for col in config["freq"]:
        X[col] = apply_frequency_map(feature_df[col], encoders["freq"][col])

    if config["ohe"] and encoders["ohe"] is not None:
        X_ohe = encoders["ohe"].transform(feature_df[config["ohe"]])
        X_ohe = pd.DataFrame(
            X_ohe, columns=encoders["ohe"].get_feature_names_out(config["ohe"]), index=X.index
        )
        X = pd.concat([X, X_ohe], axis=1)

    return X, encoders


def evaluate_predictions(y_true, y_pred):
    raw_true = np.expm1(y_true)
    raw_pred = np.expm1(y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "raw_rmse": np.sqrt(mean_squared_error(raw_true, raw_pred)),
        "raw_mae": mean_absolute_error(raw_true, raw_pred),
    }


def _quiet_fit(model, X_train, y_train, X_val=None, y_val=None):
    # tree boosters print per-iteration eval logs; keep notebooks/console clean
    with redirect_stdout(io.StringIO()):
        if X_val is not None:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        else:
            model.fit(X_train, y_train)
    return model


def fit_predict(model, X_train, y_train, X_val, y_val, X_test):
    _quiet_fit(model, X_train, y_train, X_val, y_val)
    return model.predict(X_val), model.predict(X_test)


def log_experiment(row):
    # Append one experiment to reports/experiments/experiment_log.csv
    EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "experiment_id", "feature_set", "model", "hyperparameters",
        "train_period", "validation_period", "test_period",
        "mae", "rmse", "r2", "raw_rmse", "raw_mae", "notes",
    ]
    if not EXPERIMENT_LOG.exists():
        pd.DataFrame(columns=columns).to_csv(EXPERIMENT_LOG, index=False)
    pd.DataFrame([row]).to_csv(EXPERIMENT_LOG, mode="a", header=False, index=False)
    return pd.read_csv(EXPERIMENT_LOG)


def read_experiment_log():
    if not EXPERIMENT_LOG.exists():
        return pd.DataFrame()
    return pd.read_csv(EXPERIMENT_LOG)


def run_experiment(experiment_id, feature_set_name, df_feat, model, params, notes=""):
    # One feature-stage experiment: same algorithm/methodology, only features change
    config = FEATURE_SETS[feature_set_name]
    train, val, test = time_split(df_feat)

    X_train, encoders = build_matrix(train, config, fit_encoders=True)
    X_val, _ = build_matrix(val, config, encoders=encoders)
    X_test, _ = build_matrix(test, config, encoders=encoders)

    y_train = np.log1p(train["revenue"])
    y_val = np.log1p(val["revenue"])
    y_test = np.log1p(test["revenue"])

    _quiet_fit(model, X_train, y_train, X_val, y_val)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    metrics = evaluate_predictions(y_test, pred_test)

    row = {
        "experiment_id": experiment_id,
        "feature_set": feature_set_name,
        "model": model.__class__.__name__,
        "hyperparameters": str(params),
        "train_period": f"<={TRAIN_END_YEAR}",
        "validation_period": f"{TRAIN_END_YEAR + 1}-{VAL_END_YEAR}",
        "test_period": f">={VAL_END_YEAR + 1}",
        "mae": round(metrics["mae"], 4),
        "rmse": round(metrics["rmse"], 4),
        "r2": round(metrics["r2"], 4),
        "raw_rmse": round(metrics["raw_rmse"], 0),
        "raw_mae": round(metrics["raw_mae"], 0),
        "notes": notes,
    }
    log_experiment(row)
    return {
        "experiment_id": experiment_id,
        "feature_set": feature_set_name,
        "val_metrics": evaluate_predictions(y_val, pred_val),
        "test_metrics": metrics,
        "model": model,
        "X_train": X_train,
    }


def train_predict(df_feat, feature_set_name):
    # Fit the frozen (best) config on train and return test-frame predictions for error analysis
    config = FEATURE_SETS[feature_set_name]
    train, val, test = time_split(df_feat)

    X_train, encoders = build_matrix(train, config, fit_encoders=True)
    X_val, _ = build_matrix(val, config, encoders=encoders)
    X_test, _ = build_matrix(test, config, encoders=encoders)

    y_train = np.log1p(train["revenue"])
    y_val = np.log1p(val["revenue"])

    model = ALGORITHMS["xgboost"]["model"]()
    _quiet_fit(model, X_train, y_train, X_val, y_val)

    preds = model.predict(X_test)
    out = test.copy()
    out["pred_log_revenue"] = preds
    out["actual_log_revenue"] = np.log1p(test["revenue"])
    out["pred_revenue"] = np.expm1(preds)
    out["residual_log"] = out["actual_log_revenue"] - out["pred_log_revenue"]
    out["error_dollars"] = out["revenue"] - out["pred_revenue"]
    return model, out


def _fit_one(spec, X_train, y_train, X_val, y_val):
    model = spec["model"]()
    if spec["uses_eval"]:
        _quiet_fit(model, X_train, y_train, X_val, y_val)
    else:
        _quiet_fit(model, X_train, y_train)
    return model


def _split_matrices(df_feat, config):
    train, val, test = time_split(df_feat)
    X_train, encoders = build_matrix(train, config, fit_encoders=True)
    X_val, _ = build_matrix(val, config, encoders=encoders)
    X_test, _ = build_matrix(test, config, encoders=encoders)
    return (
        X_train, X_val, X_test,
        np.log1p(train["revenue"]), np.log1p(val["revenue"]), np.log1p(test["revenue"]),
    )


def compare_algorithms(df_feat, feature_set_name, notes=""):
    # Freeze the best feature set and compare modelling approaches
    config = FEATURE_SETS[feature_set_name]
    X_train, X_val, X_test, y_train, y_val, y_test = _split_matrices(df_feat, config)

    results = []
    for name, spec in ALGORITHMS.items():
        model = _fit_one(spec, X_train, y_train, X_val, y_val)
        pred_test = model.predict(X_test)
        metrics = evaluate_predictions(y_test, pred_test)
        results.append({
            "model": name,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "raw_rmse": metrics["raw_rmse"],
            "raw_mae": metrics["raw_mae"],
        })
        log_experiment({
            "experiment_id": f"alg_{name}",
            "feature_set": feature_set_name,
            "model": name,
            "hyperparameters": spec["params"],
            "train_period": f"<={TRAIN_END_YEAR}",
            "validation_period": f"{TRAIN_END_YEAR + 1}-{VAL_END_YEAR}",
            "test_period": f">={VAL_END_YEAR + 1}",
            "mae": round(metrics["mae"], 4),
            "rmse": round(metrics["rmse"], 4),
            "r2": round(metrics["r2"], 4),
            "raw_rmse": round(metrics["raw_rmse"], 0),
            "raw_mae": round(metrics["raw_mae"], 0),
            "notes": notes,
        })
    return pd.DataFrame(results)


def compare_extra_algorithms(df_feat, feature_set_name, notes=""):
    # Same protocol as compare_algorithms, over the second-round algorithm zoo
    config = FEATURE_SETS[feature_set_name]
    X_train, X_val, X_test, y_train, y_val, y_test = _split_matrices(df_feat, config)

    results = []
    for name, spec in EXTRA_ALGORITHMS.items():
        model = _fit_one(spec, X_train, y_train, X_val, y_val)
        pred_test = model.predict(X_test)
        metrics = evaluate_predictions(y_test, pred_test)
        results.append({
            "model": name,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "raw_rmse": metrics["raw_rmse"],
            "raw_mae": metrics["raw_mae"],
        })
        log_experiment({
            "experiment_id": f"alg_{name}",
            "feature_set": feature_set_name,
            "model": name,
            "hyperparameters": spec["params"],
            "train_period": f"<={TRAIN_END_YEAR}",
            "validation_period": f"{TRAIN_END_YEAR + 1}-{VAL_END_YEAR}",
            "test_period": f">={VAL_END_YEAR + 1}",
            "mae": round(metrics["mae"], 4),
            "rmse": round(metrics["rmse"], 4),
            "r2": round(metrics["r2"], 4),
            "raw_rmse": round(metrics["raw_rmse"], 0),
            "raw_mae": round(metrics["raw_mae"], 0),
            "notes": notes,
        })
    return pd.DataFrame(results)


def blend_algorithms(df_feat, feature_set_name, model_keys, notes=""):
    # Average the held-out predictions of several fitted models, weighting each by
    # its validation-period R^2 (>=0). Weights come only from the val split, so the
    # test split stays untouched for evaluation.
    config = FEATURE_SETS[feature_set_name]
    X_train, X_val, X_test, y_train, y_val, y_test = _split_matrices(df_feat, config)
    specs = _all_algorithm_specs()

    model_keys = [k for k in model_keys if k in specs]
    if not model_keys:
        raise ValueError("no known model keys")

    pred_val, pred_test = {}, {}
    for key in model_keys:
        model = _fit_one(specs[key], X_train, y_train, X_val, y_val)
        pred_val[key] = model.predict(X_val)
        pred_test[key] = model.predict(X_test)

    val_r2 = {k: r2_score(y_val, pred_val[k]) for k in model_keys}
    weights = np.clip([val_r2[k] for k in model_keys], 0, None)
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(model_keys)) / len(model_keys)

    blend_val = sum(w * pred_val[k] for w, k in zip(weights, model_keys))
    blend_test = sum(w * pred_test[k] for w, k in zip(weights, model_keys))

    val_metrics = evaluate_predictions(y_val, blend_val)
    metrics = evaluate_predictions(y_test, blend_test)

    weight_note = ", ".join(f"{k}:{w:.2f}" for w, k in zip(weights, model_keys))
    log_experiment({
        "experiment_id": "alg_ensemble",
        "feature_set": feature_set_name,
        "model": "weighted_blend",
        "hyperparameters": f"val-R2-weighted blend of [{', '.join(model_keys)}]",
        "train_period": f"<={TRAIN_END_YEAR}",
        "validation_period": f"{TRAIN_END_YEAR + 1}-{VAL_END_YEAR}",
        "test_period": f">={VAL_END_YEAR + 1}",
        "mae": round(metrics["mae"], 4),
        "rmse": round(metrics["rmse"], 4),
        "r2": round(metrics["r2"], 4),
        "raw_rmse": round(metrics["raw_rmse"], 0),
        "raw_mae": round(metrics["raw_mae"], 0),
        "notes": f"{notes} weights: {weight_note}",
    })
    return {
        "weights": dict(zip(model_keys, weights.round(3))),
        "val_metrics": val_metrics,
        "test_metrics": metrics,
    }


GREENLIGHT_MULTIPLE = 2.5

# Fixed-dollar revenue bands (leakage-free labels): flop / mid / hit / blockbuster
REVENUE_TIERS = [
    (0.0, 25e6, "< $25M"),
    (25e6, 100e6, "$25-100M"),
    (100e6, 500e6, "$100-500M"),
    (500e6, np.inf, ">= $500M"),
]


def greenlight_label(revenue, budget, multiple=GREENLIGHT_MULTIPLE):
    return revenue >= multiple * budget


def binary_threshold_metrics(y_true_log, pred_log, revenue, budget, multiple=GREENLIGHT_MULTIPLE):
    # Route-A classification: threshold the existing regression on predicted raw revenue.
    y_true = (np.asarray(revenue) >= multiple * np.asarray(budget)).astype(int)
    pred_rev = np.expm1(pred_log)
    pred = (pred_rev >= multiple * np.asarray(budget)).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, pred_log),
        "pr_auc": average_precision_score(y_true, pred_log),
    }


def tune_greenlight_offset(y_true_log, pred_log, revenue, budget, multiple=GREENLIGHT_MULTIPLE):
    # Choose an additive log-space offset on the validation split (max F1). Returns the
    # offset and the metrics summary that produced it.
    y_true = (np.asarray(revenue) >= multiple * np.asarray(budget)).astype(int)
    hurdle = np.log1p(np.asarray(budget) * multiple)
    best = (0.0, -np.inf)
    for offset in np.arange(-0.6, 0.6, 0.02):
        pred = (pred_log + offset >= hurdle).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best[1]:
            best = (float(round(offset, 2)), float(f1))
    offset, _ = best
    pred = (pred_log + offset >= hurdle).astype(int)
    rows = {
        "offset": offset,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }
    return offset, rows


def revenue_tier(revenue):
    arr = np.asarray(revenue, dtype=float)
    out = np.empty(len(arr), dtype=object)
    for i, row in enumerate(arr):
        for lo, hi, label in REVENUE_TIERS:
            if lo <= row < hi:
                out[i] = label
                break
    return pd.Series(out, index=getattr(revenue, "index", None))


def tier_metrics(y_true_log, y_pred_log, revenue_true):
    # Discrete prediction: nearest band from the regression's predicted revenue.
    y_true_tier = revenue_tier(revenue_true)
    y_pred_tier = revenue_tier(np.expm1(y_pred_log))
    labels = [t[2] for t in REVENUE_TIERS]
    cm = confusion_matrix(y_true_tier, y_pred_tier, labels=labels)
    return {
        "accuracy": accuracy_score(y_true_tier, y_pred_tier),
        "balanced_accuracy": balanced_accuracy_score(y_true_tier, y_pred_tier),
        "macro_f1": f1_score(y_true_tier, y_pred_tier, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true_tier, y_pred_tier, labels=labels, average="weighted", zero_division=0),
        "confusion": pd.DataFrame(cm, index=[f"actual {l}" for l in labels], columns=[f"pred {l}" for l in labels]),
    }


# Dedicated classifiers (Route B) for the greenlight target, same protocol as above.
CLASSIFIERS = {
    "logistic": {
        "model": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "params": "LogisticRegression(imputed+scaled)",
        "uses_eval": False,
    },
    "xgboost_binary": {
        "model": lambda: XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8, random_state=RANDOM_STATE,
            verbosity=0, early_stopping_rounds=20, eval_metric="auc",
        ),
        "params": "XGBClassifier(binary:logistic, n_est=500, lr=0.05, depth=6)",
        "uses_eval": True,
    },
}


def binary_class_metrics(y_true, y_proba, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),
    }


def best_prob_threshold(y_true, y_proba):
    # Max-F1 probability threshold from the validation split.
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    best = (0.50, -np.inf)
    for t in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        if f1 > best[1]:
            best = (float(round(t, 2)), float(f1))
    threshold, _ = best
    return threshold, binary_class_metrics(y_true, y_proba, threshold=threshold)


def compare_binary_classifiers(df_feat, feature_set_name, target, notes=""):
    # Dedicated classifiers on a binary target (e.g. greenlight). Same matrices/split.
    config = FEATURE_SETS[feature_set_name]
    X_train, X_val, X_test, _, _, _ = _split_matrices(df_feat, config)
    y_train = np.asarray(target.loc[X_train.index], dtype=int)
    y_val = np.asarray(target.loc[X_val.index], dtype=int)
    y_test = np.asarray(target.loc[X_test.index], dtype=int)

    results = []
    for name, spec in CLASSIFIERS.items():
        model = spec["model"]()
        if spec["uses_eval"]:
            _quiet_fit(model, X_train, y_train, X_val, y_val)
        else:
            _quiet_fit(model, X_train, y_train)
        proba_val = model.predict_proba(X_val)[:, 1]
        proba_test = model.predict_proba(X_test)[:, 1]
        threshold, val_rows = best_prob_threshold(y_val, proba_val)
        row = binary_class_metrics(y_test, proba_test, threshold=threshold)
        row["model"] = name
        row["threshold"] = threshold
        row["val_f1_at_threshold"] = round(val_rows["f1"], 4)
        results.append(row)

    results = pd.DataFrame(results).set_index("model")
    results = results[["threshold", "val_f1_at_threshold", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]]
    share_pos = float(np.mean(y_test))
    log_experiment({
        "experiment_id": "clf_greenlight",
        "feature_set": feature_set_name,
        "model": "logistic+xgboost_binary",
        "hyperparameters": f"test positive share {share_pos:.3f}; {notes.strip()}",
        "train_period": f"<={TRAIN_END_YEAR}",
        "validation_period": f"{TRAIN_END_YEAR + 1}-{VAL_END_YEAR}",
        "test_period": f">={VAL_END_YEAR + 1}",
        "mae": round(results["f1"].max(), 4),
        "rmse": round(results["roc_auc"].max(), 4),
        "r2": float("nan"),
        "raw_rmse": 0,
        "raw_mae": 0,
        "notes": f"dedicated binary classifiers on greenlight; best f1 {results['f1'].idxmax()}",
    })
    return results