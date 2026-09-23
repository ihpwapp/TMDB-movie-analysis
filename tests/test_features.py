import numpy as np
import pandas as pd

from scripts.feature_engineering import (
    apply_frequency_map,
    create_frequency_map,
    create_historical_log_features,
    create_log_features,
    create_release_features,
)


def test_release_features_quarter():
    df = pd.DataFrame({"release_year": [2024], "release_month": [11]})
    out = create_release_features(df)
    assert out["release_quarter"][0] == 4
    assert out["release_decade"][0] == 2020


def test_log_features_are_log1p():
    df = pd.DataFrame({"budget": [0, 100], "runtime": [120, 90], "revenue": [999, 50]})
    out = create_log_features(df)
    assert out["log_budget"].tolist() == [0.0, np.log1p(100)]
    assert out["log_revenue"].tolist() == [np.log1p(999), np.log1p(50)]


def test_frequency_map_fit_and_apply():
    s = pd.Series(["a", "a", "b", "c", "c", "c"])
    freq = create_frequency_map(s)
    assert abs(freq["a"] - 2 / 6) < 1e-9
    out = apply_frequency_map(pd.Series(["a", "zzz"]), freq)
    assert out[0] == freq["a"]
    assert out[1] == 0.0


def test_historical_log_features_nan_safe():
    df = pd.DataFrame({
        "director_previous_avg_revenue": [100.0, np.nan],
        "lead_actor_previous_avg_revenue": [np.nan, 1.0],
    })
    out = create_historical_log_features(df)
    assert out["log_director_previous_avg_revenue"][0] == np.log1p(100)
    assert np.isnan(out["log_director_previous_avg_revenue"][1])
    assert out["log_lead_actor_previous_avg_revenue"][1] == np.log1p(1.0)