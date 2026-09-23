import numpy as np
import pandas as pd

from scripts.historical_features import (
    compute_historical_features,
    verify_historical_features,
)


def _history():
    # three films: same director for 1&2, shared collection for 2&3, cast overlap
    return pd.DataFrame({
        "movie_id": [1, 2, 3],
        "release_date": pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01"]),
        "budget": [100, 200, 400],
        "revenue": [300, 600, 2000],
        "director_ids": [[11], [11], [12]],
        "company_ids": [[21], [22], [21]],
        "cast_ids": [[31, 32], [31, 33], [33, 31]],
        "belongs_to_collection_id": [0, 41, 41],
    })


def test_prior_stats_use_strictly_earlier_releases():
    out = compute_historical_features(_history())
    m2 = out[out["movie_id"] == 2]
    assert m2["director_previous_movie_count"].iloc[0] == 1
    assert abs(m2["director_previous_avg_revenue"].iloc[0] - 300) < 1e-6
    m3 = out[out["movie_id"] == 3]
    # franchise prior = film 2 only (film 3 itself is concurrent release-date-wise)
    assert abs(m3["franchise_previous_avg_revenue"].iloc[0] - 600) < 1e-6


def test_concurrent_releases_are_excluded():
    h = pd.DataFrame({
        "movie_id": [1, 2],
        "release_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        "budget": [100, 100],
        "revenue": [300, 400],
        "director_ids": [[11], [11]],
        "company_ids": [[21], [21]],
        "cast_ids": [[31], [31]],
        "belongs_to_collection_id": [0, 0],
    })
    out = compute_historical_features(h)
    m2 = out[out["movie_id"] == 2]
    assert m2["director_previous_movie_count"].iloc[0] == 0
    assert np.isnan(m2["director_previous_avg_revenue"].iloc[0])


def test_lead_actor_uses_first_billed_semantics():
    out = compute_historical_features(_history())
    m3 = out[out["movie_id"] == 3]
    # lead of film 3 = actor 33, who was never first-billed before -> no prior
    assert np.isnan(m3["lead_actor_previous_avg_revenue"].iloc[0])


def test_brute_force_verification_passes_on_synthetic():
    h = _history()
    features = compute_historical_features(h)
    verify_historical_features(h, features, n=3)