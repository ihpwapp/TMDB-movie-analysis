import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from scripts.feature_engineering import (
    create_historical_log_features,
    create_log_features,
    create_release_features,
)

# Raw inputs accepted by the free-form / edit forms. These are the human fields;
# every numeric/log/historical column a feature set needs is derived from them
# by build_input_features below, so nothing is duplicated in Streamlit.
INPUT_KEYS = [
    "budget", "runtime", "release_date", "original_language", "primary_genre",
    "primary_country", "company", "director", "cast", "franchise",
    "genre_count", "production_company_count", "production_country_count",
    "cast_size", "crew_size", "producer_count", "writer_count", "composer_count",
    "keyword_count",
    "is_superhero", "is_sequel", "is_remake", "is_based_on_novel", "is_based_on_true_story",
]

# Words like "Pictures"/"Studios"/"The" add no signal when matching a typed name
# against the filmography, so they are dropped during normalization.
_IGNORED_TOKENS = {
    "pictures", "picture", "studios", "studio", "entertainment", "films", "film",
    "productions", "production", "productioncompany", "company", "corporation",
    "corp", "incorporated", "llc", "inc", "ltd", "limited", "group",
    "international", "media", "the", "co",
}


def normalize_entity_name(name):
    # lower-cases and strips legal/descriptor suffixes so "Warner Bros. Pictures"
    # typed as "warner bros" still matches the filmography
    if name is None:
        return ""
    s = re.sub(r"[^a-z0-9 ]+", " ", str(name).lower().replace("'", ""))
    tokens = [t for t in s.split() if t]
    if not tokens:
        return ""
    kept = [t for t in tokens if t not in _IGNORED_TOKENS]
    return " ".join(kept) if kept else " ".join(tokens)


def format_feature_value(feature_name, value):
    # human-readable presentation of a matrix cell for the SHAP contribution table
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) and np.isnan(value):
        return ""
    if feature_name.startswith("log_"):
        raw = float(np.expm1(value))
        return f"${raw/1e6:,.0f}M" if raw >= 1e6 else f"{raw:,.0f}"
    if "infrequent" in feature_name:
        return "Other"
    if feature_name.startswith(("original_language_", "primary_genre_", "primary_country_")):
        return "Yes" if value == 1.0 else "No"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _fuzzy_history_hit(table, norm):
    # fallback when the normalized name does not match exactly: pick the closest
    # filmography name above a similarity threshold
    best, best_ratio = None, 0.0
    for cand in table["name_norm"].dropna().tolist():
        if not cand or len(cand) < len(norm) * 0.6 or len(cand) > len(norm) * 1.6:
            continue
        ratio = SequenceMatcher(None, norm, cand).ratio()
        if ratio > best_ratio:
            best, best_ratio = cand, ratio
    if best is not None and best_ratio >= 0.85:
        return table[table["name_norm"] == best]
    return None


def _entity_history(table, name):
    # case-insensitive lookup of a single entity's prior-film stats, with a
    # suffix-tolerant exact match first and a fuzzy fallback for typos
    norm = normalize_entity_name(name)
    if not norm:
        return None
    if "name_norm" not in table.columns:
        table = table.assign(name_norm=table["name"].map(normalize_entity_name))
    hit = table[table["name_norm"] == norm]
    if hit.empty:
        hit = _fuzzy_history_hit(table, norm)
    if hit is None or hit.empty:
        return None
    s = hit.iloc[0]
    return {
        "count": int(s["count"]),
        "avg_revenue": s["avg_revenue"],
        "avg_budget": s["avg_budget"],
        "avg_roi": s["avg_roi"],
        "max_revenue": s["max_revenue"],
    }


def build_name_history_stats(ds):
    # Name-based prior-film stats, used ONLY for free-form/edited input where no
    # TMDB id is known. Prior = films released before the newest release date in
    # the dataset. Approximate by nature: names cover the extracted filmography
    # only, and cast history is keyed on the first-billed actor.
    base = ds[["movie_id", "release_date", "budget", "revenue", "roi",
               "director_name", "primary_production_company", "collection_name", "cast"]]
    base = base.drop_duplicates(subset="movie_id").copy()
    reference = base["release_date"].max()
    base = base[base["release_date"] < reference].copy()

    def agg_table(frame):
        frame = frame.dropna(subset=["name"])
        frame = frame[frame["name"].str.strip() != ""]
        g = frame.groupby("name", as_index=False).agg(
            count=("revenue", "size"),
            avg_revenue=("revenue", "mean"),
            avg_budget=("budget", "mean"),
            avg_roi=("roi", "mean"),
            max_revenue=("revenue", "max"),
        )
        g["name_lower"] = g["name"].str.lower()
        g["name_norm"] = g["name"].map(normalize_entity_name)
        return g

    stats = {
        "director": agg_table(base.rename(columns={"director_name": "name"})
                              [["name", "budget", "revenue", "roi"]]),
        "company": agg_table(base.rename(columns={"primary_production_company": "name"})
                             [["name", "budget", "revenue", "roi"]]),
        "lead": agg_table(base.assign(name=base["cast"].str.split(", ").str[0])
                          [["name", "budget", "revenue", "roi"]]),
        "franchise": agg_table(base.rename(columns={"collection_name": "name"})
                               [["name", "budget", "revenue", "roi"]]),
    }
    return stats


def _fill_history(row, hist, prefix):
    if hist:
        row[f"{prefix}_previous_movie_count"] = hist["count"]
        row[f"{prefix}_previous_avg_revenue"] = hist["avg_revenue"]
        row[f"{prefix}_previous_avg_budget"] = hist["avg_budget"]
        row[f"{prefix}_previous_avg_roi"] = hist["avg_roi"]
        row[f"{prefix}_previous_max_revenue"] = hist["max_revenue"]
    return row


def build_input_features(inputs, name_stats, orig_row=None):
    # Shared feature builder for a single movie. When orig_row (a prepared row
    # from the dataset) is given, it is reused as-is and only the fields the user
    # edited are recomputed, so unchanged movies keep their exact id-based track
    # records. Without orig_row a full row is built from the inputs and the
    # name-based history stats.
    inputs = {k: ("" if v is None else v) for k, v in inputs.items()}

    def first_or(clist, original):
        parts = [p.strip() for p in str(clist).split(",") if p.strip()]
        return parts[0] if parts else original

    if orig_row is not None:
        row = orig_row.copy()
        lead_input = first_or(inputs.get("cast", ""), None)

        row["log_budget"] = np.log1p(inputs["budget"])
        row["log_runtime"] = np.log1p(inputs["runtime"])
        row["release_date"] = pd.Timestamp(inputs["release_date"])
        row["release_year"] = row["release_date"].year
        row["release_month"] = row["release_date"].month
        row["release_day_of_week"] = row["release_date"].weekday() + 1
        row["release_quarter"] = np.ceil(row["release_month"] / 3).astype(int)
        row["original_language"] = inputs["original_language"]
        row["primary_genre"] = inputs["primary_genre"]
        row["primary_country"] = inputs["primary_country"]
        row["primary_production_company"] = inputs["company"]
        row["director_name"] = inputs["director"]

        if inputs["director"] != orig_row["director_name"]:
            row = _fill_history(row, _entity_history(name_stats["director"], inputs["director"]), "director")
        if inputs["company"] != orig_row["primary_production_company"]:
            hist = _entity_history(name_stats["company"], inputs["company"])
            row["production_company_previous_avg_revenue"] = hist["avg_revenue"] if hist else np.nan
        if lead_input and lead_input != orig_row["cast"].split(", ")[0]:
            hist = _entity_history(name_stats["lead"], lead_input)
            if hist:
                row["lead_actor_previous_avg_revenue"] = hist["avg_revenue"]
                row["cast_previous_avg_revenue"] = hist["avg_revenue"]
        if inputs.get("franchise") and inputs["franchise"] != orig_row.get("collection_name", ""):
            hist = _entity_history(name_stats["franchise"], inputs["franchise"])
            if hist:
                row["franchise_previous_avg_revenue"] = hist["avg_revenue"]
                row["franchise_previous_avg_roi"] = hist["avg_roi"]
                row["franchise_movie_count"] = hist["count"]
    else:
        d = {c: np.nan for c in [
            "log_budget", "log_runtime", "release_year", "release_month",
            "release_quarter", "release_day_of_week",
            "production_company_count", "production_country_count",
            "cast_size", "crew_size", "producer_count", "writer_count", "composer_count",
            "keyword_count",
            "director_previous_movie_count", "director_previous_avg_revenue",
            "director_previous_avg_budget", "director_previous_avg_roi",
            "director_previous_max_revenue", "production_company_previous_avg_revenue",
            "cast_previous_avg_revenue", "lead_actor_previous_avg_revenue",
            "franchise_previous_avg_revenue", "franchise_previous_avg_roi", "franchise_movie_count",
        ]}
        lead = first_or(inputs.get("cast", ""), "")

        release = pd.Timestamp(inputs["release_date"])
        d["release_date"] = release
        d["release_year"] = release.year
        d["release_month"] = release.month
        d["release_day_of_week"] = release.weekday() + 1
        d["release_quarter"] = int(np.ceil(release.month / 3))

        for col, default in [
            ("budget", 0.0), ("runtime", 0.0),
            ("genre_count", 1), ("production_company_count", 1),
            ("production_country_count", 1), ("cast_size", 1), ("crew_size", 1),
            ("producer_count", 1), ("writer_count", 1), ("composer_count", 1),
            ("keyword_count", 0),
        ]:
            d[col] = inputs.get(col, default)
        d["log_budget"] = np.log1p(d["budget"])
        d["log_runtime"] = np.log1p(d["runtime"])

        d = _fill_history(d, _entity_history(name_stats["director"], inputs["director"]), "director")
        comp = _entity_history(name_stats["company"], inputs["company"])
        d["production_company_previous_avg_revenue"] = comp["avg_revenue"] if comp else np.nan
        lead_hist = _entity_history(name_stats["lead"], lead)
        if lead_hist:
            d["lead_actor_previous_avg_revenue"] = lead_hist["avg_revenue"]
            d["cast_previous_avg_revenue"] = lead_hist["avg_revenue"]
        fran = _entity_history(name_stats["franchise"], inputs.get("franchise", ""))
        if fran:
            d["franchise_previous_avg_revenue"] = fran["avg_revenue"]
            d["franchise_previous_avg_roi"] = fran["avg_roi"]
            d["franchise_movie_count"] = fran["count"]

        d["original_language"] = inputs["original_language"]
        d["primary_genre"] = inputs["primary_genre"]
        d["primary_country"] = inputs["primary_country"]
        d["primary_production_company"] = inputs["company"]
        d["director_name"] = inputs["director"]
        for col in ["is_superhero", "is_sequel", "is_remake", "is_based_on_novel", "is_based_on_true_story"]:
            d[col] = int(inputs.get(col, False))

        row = pd.Series(d)

    return create_historical_log_features(pd.DataFrame([row])).iloc[0]


def humanize(feature_name):
    # turn raw matrix columns into readable labels for the SHAP plot
    labels = {
        "log_budget": "Budget (log)", "log_runtime": "Runtime (log)",
        "release_year": "Release year", "release_month": "Release month",
        "release_quarter": "Release quarter", "release_day_of_week": "Release weekday",
        "production_company_count": "# production companies",
        "production_country_count": "# production countries",
        "cast_size": "Cast size", "crew_size": "Crew size",
        "producer_count": "# producers", "writer_count": "# writers",
        "composer_count": "# composers", "keyword_count": "# keywords",
        "director_previous_movie_count": "Director prior films",
        "log_director_previous_avg_revenue": "Director prior avg revenue (log)",
        "log_director_previous_avg_budget": "Director prior avg budget (log)",
        "director_previous_avg_roi": "Director prior avg ROI",
        "log_director_previous_max_revenue": "Director prior max revenue (log)",
        "log_production_company_previous_avg_revenue": "Company prior avg revenue (log)",
        "log_cast_previous_avg_revenue": "Cast prior avg revenue (log)",
        "log_lead_actor_previous_avg_revenue": "Lead actor prior avg revenue (log)",
        "log_franchise_previous_avg_revenue": "Franchise prior avg revenue (log)",
        "franchise_previous_avg_roi": "Franchise prior ROI",
        "franchise_movie_count": "Franchise prior films",
        "is_superhero": "Superhero", "is_sequel": "Sequel", "is_remake": "Remake",
        "is_based_on_novel": "Based on novel", "is_based_on_true_story": "Based on true story",
        "primary_production_company": "Production company", "director_name": "Director",
    }
    if feature_name in labels:
        return labels[feature_name]
    for prefix, label in [
        ("original_language_", "Language: "), ("primary_genre_", "Genre: "),
        ("primary_country_", "Country: "),
    ]:
        if feature_name.startswith(prefix):
            return label + feature_name[len(prefix):].replace("infrequent_sklearn", "Other").replace("Unknown", "(missing)")
    if "infrequent" in feature_name:
        return "Other (infrequent category)"
    return feature_name.replace("_", " ") if feature_name else feature_name