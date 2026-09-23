import numpy as np
import pandas as pd

from scripts.evaluation import FEATURE_SETS, build_matrix


def _frame():
    # minimal row containing every column the basic config reads
    return pd.DataFrame({
        "log_budget": [1.0, 2.0], "log_runtime": [3.0, 4.0],
        "release_year": [2010, 2011], "release_month": [1, 2],
        "release_quarter": [1, 1], "release_day_of_week": [3, 4],
        "original_language": ["en", "es"], "primary_genre": ["Action", "Drama"],
        "primary_country": ["United States", "France"],
    })


def test_encoders_fit_on_train_only_and_reuse_on_test():
    config = FEATURE_SETS["basic"]
    X_train, enc = build_matrix(_frame(), config, fit_encoders=True)
    assert X_train.shape[0] == 2

    novel = pd.DataFrame({
        "log_budget": [9.0], "log_runtime": [8.0],
        "release_year": [2024], "release_month": [6], "release_quarter": [2],
        "release_day_of_week": [1],
        "original_language": ["xx"], "primary_genre": ["WeirdNeverSeen"],
        "primary_country": ["NeverLand"],
    })
    X_test, _ = build_matrix(novel, config, encoders=enc, fit_encoders=False)
    assert list(X_test.columns) == list(X_train.columns)


def test_unseen_frequency_category_falls_back_to_zero():
    config = FEATURE_SETS["production"]
    train = pd.DataFrame({
        "log_budget": [1.0, 2.0], "log_runtime": [3.0, 4.0],
        "release_year": [2010, 2011], "release_month": [1, 2],
        "release_quarter": [1, 1], "release_day_of_week": [3, 4],
        "production_company_count": [1, 2], "production_country_count": [1, 1],
        "primary_production_company": ["A", "A"],
        "original_language": ["en", "en"], "primary_genre": ["Action", "Action"],
        "primary_country": ["United States", "United States"],
    })
    X_train, enc = build_matrix(train, config, fit_encoders=True)
    test_df = train.iloc[[0]].copy()
    test_df["primary_production_company"] = "NeverSeenCo"
    X_test, _ = build_matrix(test_df, config, encoders=enc, fit_encoders=False)
    assert X_test["primary_production_company"].iloc[0] == 0.0

    assert X_train.columns[0] == "log_budget"