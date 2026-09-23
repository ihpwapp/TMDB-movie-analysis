import numpy as np
import pandas as pd

from scripts.models.request import (
    build_input_features,
    format_feature_value,
    humanize,
    normalize_entity_name,
)


def _stats():
    def table(pairs):
        df = pd.DataFrame(
            pairs,
            columns=["name", "count", "avg_revenue", "avg_budget", "avg_roi", "max_revenue"],
        )
        df["name_lower"] = df["name"].str.lower()
        df["name_norm"] = df["name"].map(normalize_entity_name)
        return df

    return {
        "director": table([("Nolan", 8, 500e6, 150e6, 3.3, 900e6), ("Nobody", 1, 10e6, 5e6, 2.0, 10e6)]),
        "company": table([("WB", 5, 400e6, 100e6, 4.0, 700e6)]),
        "lead": table([("DiCaprio", 9, 250e6, 80e6, 3.1, 500e6)]),
        "franchise": table([("Batman Collection", 4, 600e6, 150e6, 4.0, 900e6)]),
    }


def _inputs(**over):
    base = {
        "budget": 100e6, "runtime": 120, "release_date": "2027-05-21",
        "original_language": "en", "primary_genre": "Action", "primary_country": "United States",
        "company": "WB", "director": "Nolan", "cast": "DiCaprio, Pitt", "franchise": "Batman Collection",
        "genre_count": 1, "production_company_count": 1, "production_country_count": 1,
        "cast_size": 2, "crew_size": 100, "producer_count": 2, "writer_count": 2,
        "composer_count": 1, "keyword_count": 4,
        "is_superhero": True, "is_sequel": True, "is_remake": False,
        "is_based_on_novel": False, "is_based_on_true_story": False,
    }
    base.update(over)
    return base


def test_free_form_looks_up_names_case_insensitively():
    row = build_input_features(_inputs(director="nolan"), _stats())
    assert row["log_director_previous_avg_revenue"] == np.log1p(500e6)
    assert row["lead_actor_previous_avg_revenue"] == 250e6
    assert row["log_lead_actor_previous_avg_revenue"] == np.log1p(250e6)
    assert row["franchise_previous_avg_roi"] == 4.0
    assert row["release_quarter"] == 2
    assert row["log_budget"] == np.log1p(100e6)
    assert row["is_superhero"] == 1


def test_unknown_names_give_nan_history():
    row = build_input_features(
        _inputs(director="Nobody Real", company="No Company", cast="Unknown Star", franchise=""),
        _stats(),
    )
    assert np.isnan(row["director_previous_avg_revenue"])
    assert np.isnan(row["production_company_previous_avg_revenue"])
    assert np.isnan(row["lead_actor_previous_avg_revenue"])


def _orig_row():
    return pd.Series({
        "log_budget": np.log1p(150e6), "log_runtime": np.log1p(120),
        "release_year": 2020, "release_month": 1, "release_quarter": 1, "release_day_of_week": 4,
        "production_company_count": 2, "production_country_count": 1, "cast_size": 5,
        "crew_size": 150, "producer_count": 3, "writer_count": 2, "composer_count": 1,
        "keyword_count": 6, "director_previous_movie_count": 4, "director_previous_avg_revenue": 100e6,
        "director_previous_avg_budget": 40e6, "director_previous_avg_roi": 2.5,
        "director_previous_max_revenue": 200e6, "production_company_previous_avg_revenue": 90e6,
        "cast_previous_avg_revenue": 80e6, "lead_actor_previous_avg_revenue": 75e6,
        "franchise_previous_avg_revenue": 60e6, "franchise_previous_avg_roi": 3.0,
        "franchise_movie_count": 2, "log_director_previous_avg_revenue": np.log1p(100e6),
        "log_director_previous_avg_budget": np.log1p(40e6), "log_director_previous_max_revenue": np.log1p(200e6),
        "log_production_company_previous_avg_revenue": np.log1p(90e6),
        "log_cast_previous_avg_revenue": np.log1p(80e6), "log_lead_actor_previous_avg_revenue": np.log1p(75e6),
        "log_franchise_previous_avg_revenue": np.log1p(60e6),
        "original_language": "en", "primary_genre": "Action", "primary_country": "United States",
        "primary_production_company": "Some Co", "director_name": "Some Dir", "cast": "Lead A, Lead B",
        "collection_name": "Some Coll", "release_date": pd.Timestamp("2020-06-01"),
        "is_superhero": 0, "is_sequel": 0, "is_remake": 0, "is_based_on_novel": 0,
        "is_based_on_true_story": 0, "budget": 150e6, "runtime": 120, "revenue": 300e6,
    })


def test_edit_keeps_unchanged_history_exactly():
    orig = _orig_row()
    row = build_input_features(
        _inputs(budget=150e6, release_date="2020-06-01", company="Some Co", director="Some Dir",
                cast="Lead A, Lead B", franchise="Some Coll"),
        _stats(), orig_row=orig,
    )
    assert row["director_previous_avg_revenue"] == 100e6
    assert row["franchise_movie_count"] == 2
    assert row["log_budget"] == np.log1p(150e6)


def test_humanize_labels():
    assert humanize("log_budget") == "Budget (log)"
    assert humanize("original_language_en") == "Language: en"
    assert humanize("primary_genre_Adventure") == "Genre: Adventure"


def test_normalize_entity_name():
    assert normalize_entity_name("Warner Bros. Pictures") == "warner bros"
    assert normalize_entity_name("Marvel Studios") == "marvel"
    assert normalize_entity_name("Walt Disney") == "walt disney"
    assert normalize_entity_name(None) == ""


def test_suffix_tolerant_name_matching():
    # "warner bros" must match stored "Warner Bros. Pictures" after normalization
    def company_table(pairs):
        df = pd.DataFrame(pairs, columns=["name", "count", "avg_revenue", "avg_budget", "avg_roi", "max_revenue"])
        return df.assign(name_lower=df["name"].str.lower(), name_norm=df["name"].map(normalize_entity_name))

    stats = {
        "director": company_table([]),
        "company": company_table([["Warner Bros. Pictures", 5, 400e6, 100e6, 4.0, 700e6]]),
        "lead": company_table([]),
        "franchise": company_table([]),
    }
    row = build_input_features(_inputs(company="warner bros"), stats)
    assert row["production_company_previous_avg_revenue"] == 400e6


def test_fuzzy_fallback_for_typos():
    stats = _stats()
    stats["director"] = pd.DataFrame(
        [["Christopher Nolan", 8, 500e6, 150e6, 3.3, 900e6]],
        columns=["name", "count", "avg_revenue", "avg_budget", "avg_roi", "max_revenue"],
    )
    stats["director"]["name_lower"] = stats["director"]["name"].str.lower()
    stats["director"]["name_norm"] = stats["director"]["name"].map(normalize_entity_name)

    row = build_input_features(_inputs(director="Christoper Nolen"), stats)
    assert row["director_previous_avg_revenue"] == 500e6

    row = build_input_features(_inputs(director="Totally Different Name"), stats)
    assert np.isnan(row["director_previous_avg_revenue"])


def test_format_feature_value():
    assert format_feature_value("log_budget", np.log1p(150e6)) == "$150M"
    assert format_feature_value("original_language_en", 1.0) == "Yes"
    assert format_feature_value("original_language_es", 0.0) == "No"
    assert format_feature_value("primary_genre_infrequent_sklearn", 1.0) == "Other"
    assert format_feature_value("primary_production_company", "Marvel Studios") == "Marvel Studios"
    assert format_feature_value("release_year", float("nan")) == ""