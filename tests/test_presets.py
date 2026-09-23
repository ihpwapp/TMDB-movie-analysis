import numpy as np
import pandas as pd
import pytest

from app.presets import FORM_FIELDS, estimate_preset_defaults


def _ds():
    columns = [
        "budget", "runtime", "release_month", "original_language", "primary_genre",
        "primary_country", "primary_production_company", "director_name",
        "collection_name", "cast", "cast_size", "crew_size", "producer_count",
        "writer_count", "composer_count", "keyword_count",
        "is_superhero", "is_sequel", "is_remake", "is_based_on_novel",
        "is_based_on_true_story",
    ]
    rows = [
        # four low-budget dramas -> indie median budget 3.5M
        [2e6, 90, 8, "en", "Drama", "United States", "Paramount", "A", "F1", "Drama star, one", 25, 40, 2, 0, 1, 9, 0, 0, 0, 0, 0],
        [5e6, 95, 8, "en", "Drama", "United States", "Paramount", "A", "F1", "Drama star, two", 25, 40, 2, 0, 1, 9, 0, 0, 0, 0, 0],
        [3e6, 92, 8, "en", "Drama", "United States", "Fox", "B", "", "Other star", 20, 30, 1, 0, 1, 5, 0, 0, 0, 0, 0],
        [8e6, 88, 8, "en", "Drama", "United States", "Paramount", "A", "", "Drama star, three", 25, 40, 2, 0, 1, 9, 0, 0, 0, 0, 0],
        # three superhero blockbusters, two are sequels
        [100e6, 130, 7, "en", "Adventure", "United States", "Columbia", "Zack", "X-Men Collection", "Wolverine, Wilson", 60, 140, 3, 2, 1, 15, 1, 1, 0, 0, 0],
        [120e6, 135, 7, "en", "Adventure", "United States", "Columbia", "Zack", "X-Men Collection", "Wolverine, Lane", 60, 140, 3, 2, 1, 15, 1, 1, 0, 0, 0],
        [130e6, 110, 7, "en", "Adventure", "United States", "Marvel", "Other", "", "Stark, Tony", 55, 120, 3, 2, 1, 12, 1, 0, 0, 0, 0],
        # two comedies
        [10e6, 95, 12, "en", "Comedy", "United States", "Universal", "C", "", "Comic, Guy", 40, 44, 2, 1, 1, 8, 0, 0, 0, 0, 0],
        [20e6, 100, 12, "en", "Comedy", "United States", "Universal", "C", "", "Comic, Gal", 40, 44, 2, 1, 1, 8, 0, 0, 0, 0, 0],
        # two animation
        [25e6, 88, 6, "en", "Animation", "United States", "Disney", "Hayao", "", "Totoro, Big", 24, 68, 2, 0, 1, 11, 0, 0, 0, 0, 0],
        [45e6, 92, 6, "en", "Animation", "United States", "Disney", "Hayao", "", "Totoro, Small", 24, 68, 2, 0, 1, 11, 0, 0, 0, 0, 0],
    ]
    return pd.DataFrame(rows, columns=columns)


def test_preset_returns_every_form_field():
    defaults = estimate_preset_defaults(_ds(), "superhero_blockbuster")
    assert set(FORM_FIELDS) == set(defaults)


def test_superhero_blockbuster_defaults():
    defaults = estimate_preset_defaults(_ds(), "superhero_blockbuster")
    assert defaults["is_superhero"] is True
    assert defaults["budget"] == 120e6
    assert defaults["primary_genre"] == "Adventure"
    assert defaults["director"] == "Zack"
    assert defaults["company"] == "Columbia"
    assert defaults["franchise"] == "X-Men Collection"
    assert defaults["cast"] == "Wolverine"
    assert defaults["is_sequel"] is True
    assert defaults["release_date"] == pd.Timestamp("2027-07-15")


def test_indie_defaults():
    defaults = estimate_preset_defaults(_ds(), "indie")
    assert defaults["is_superhero"] is False
    assert defaults["budget"] == 4e6
    assert defaults["primary_genre"] == "Drama"


def test_comedy_and_animation_genre():
    assert estimate_preset_defaults(_ds(), "comedy")["primary_genre"] == "Comedy"
    assert estimate_preset_defaults(_ds(), "animation")["primary_genre"] == "Animation"
    assert estimate_preset_defaults(_ds(), "animation")["budget"] == 35e6


def test_blank_uses_overall_medians_and_modes():
    defaults = estimate_preset_defaults(_ds(), None)
    budgets = [2e6, 5e6, 3e6, 8e6, 100e6, 120e6, 130e6, 10e6, 20e6, 25e6, 45e6]
    assert defaults["budget"] == float(np.median(budgets))
    assert defaults["primary_genre"] == "Drama"
    assert defaults["is_superhero"] is False


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        estimate_preset_defaults(_ds(), "nonsense")