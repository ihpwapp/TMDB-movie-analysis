import numpy as np
import pandas as pd

FORM_FIELDS = [
    "budget", "runtime", "release_date",
    "original_language", "primary_genre", "primary_country",
    "company", "director", "franchise", "cast",
    "genre_count", "production_company_count", "production_country_count",
    "cast_size", "crew_size", "producer_count", "writer_count", "composer_count",
    "keyword_count",
    "is_superhero", "is_sequel", "is_remake",
    "is_based_on_novel", "is_based_on_true_story",
]

# label (shown in the UI) -> preset key passed to estimate_preset_defaults
PRESETS = [
    ("Blank (dataset median)", None),
    ("Superhero blockbuster", "superhero_blockbuster"),
    ("Indie film", "indie"),
    ("Comedy", "comedy"),
    ("Animation", "animation"),
]

_FLAG_COLUMNS = ["is_sequel", "is_remake", "is_based_on_novel", "is_based_on_true_story"]
_TEXT_COLUMNS = ["company", "director", "franchise", "cast"]


def _mask(ds, key):
    if key is None:
        return pd.Series(True, index=ds.index)
    if key == "superhero_blockbuster":
        return ds["is_superhero"] == 1
    if key == "indie":
        return ds["budget"] < 10e6
    if key in ("comedy", "animation"):
        return ds["primary_genre"] == key.capitalize()
    raise ValueError(f"unknown preset: {key}")


def _median(df, col):
    return float(df[col].median()) if len(df) else float("nan")


def _mode(df, col):
    values = df[col].dropna()
    if values.empty:
        return ""
    top = values.mode()
    return top.iloc[0] if not top.empty else ""


def _top_name(df, col):
    if col == "cast":
        if df[col].isna().all():
            return ""
        series = df[col].dropna().str.split(",").str[0].str.strip()
    else:
        series = df[col].dropna()
    if series.empty:
        return ""
    counts = series.value_counts()
    return counts.index[0] if not counts.empty else ""


def _top_flag(df, col):
    return bool(df[col].mean() >= 0.5) if len(df) else False


def estimate_preset_defaults(ds, key):
    sub = ds[_mask(ds, key)]
    month = _mode(sub, "release_month")
    try:
        month = int(float(month))
    except (TypeError, ValueError):
        month = 12

    if key in ("comedy", "animation"):
        genre = key.capitalize()
    else:
        genre = _mode(sub, "primary_genre") or "Drama"

    defaults = {
        "budget": _median(sub, "budget"),
        "runtime": _median(sub, "runtime"),
        "release_date": pd.Timestamp(year=2027, month=month, day=15),
        "original_language": _mode(sub, "original_language") or "en",
        "primary_genre": genre,
        "primary_country": _mode(sub, "primary_country") or "United States",
        "company": _top_name(sub, "primary_production_company"),
        "director": _top_name(sub, "director_name"),
        "franchise": _top_name(sub, "collection_name"),
        "cast": _top_name(sub, "cast"),
        "genre_count": 1,
        "production_company_count": 1,
        "production_country_count": 1,
        "cast_size": _median(sub, "cast_size"),
        "crew_size": _median(sub, "crew_size"),
        "producer_count": _median(sub, "producer_count"),
        "writer_count": _median(sub, "writer_count"),
        "composer_count": _median(sub, "composer_count"),
        "keyword_count": _median(sub, "keyword_count"),
        "is_superhero": key == "superhero_blockbuster",
        "is_sequel": _top_flag(sub, "is_sequel"),
        "is_remake": _top_flag(sub, "is_remake"),
        "is_based_on_novel": _top_flag(sub, "is_based_on_novel"),
        "is_based_on_true_story": _top_flag(sub, "is_based_on_true_story"),
    }

    for name in FORM_FIELDS:
        if name not in defaults:
            defaults[name] = _median(sub, name)
    return defaults