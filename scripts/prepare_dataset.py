from scripts.feature_engineering import (
    create_historical_log_features,
    create_log_features,
    create_release_features,
)
from scripts.historical_features import compute_historical_features
from scripts.load_data import load_feature_dataset, load_history_data

OHE_COLS = ["original_language", "primary_genre", "primary_country"]
FREQ_COLS = ["primary_production_company", "director_name"]


def prepare_ml_dataset(extract_path=None, history_path=None):
    # Merges the ML extract with leakage-safe historical features and builds
    # the derived (log/quarter/etc) columns used by the experiments
    df = load_feature_dataset(extract_path) if extract_path else load_feature_dataset()
    hist = load_history_data(history_path) if history_path else load_history_data()

    hist_features = compute_historical_features(hist)
    ds = df.merge(hist_features, on="movie_id", how="left")

    ds = create_release_features(ds)
    ds = create_log_features(ds)
    ds = create_historical_log_features(ds)

    for col in OHE_COLS:
        ds[col] = ds[col].fillna("Unknown")
    for col in FREQ_COLS:
        ds[col] = ds[col].fillna("Unknown")
    for col in ["is_superhero", "is_sequel", "is_remake", "is_based_on_novel", "is_based_on_true_story"]:
        ds[col] = ds[col].fillna(0).astype(int)

    return ds