import sys
from pathlib import Path

import numpy as np
import pandas as pd

# allow `python scripts/historical_features.py` to resolve the scripts package
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def prepare_history_table(history_df):
    # Restrict historical aggregates to meaningfully reported films
    t = history_df.copy()
    t["budget"] = pd.to_numeric(t["budget"], errors="coerce")
    t["revenue"] = pd.to_numeric(t["revenue"], errors="coerce")
    t = t[(t["budget"] > 0) & (t["revenue"] > 0)].copy()
    t["roi"] = np.where(t["budget"] > 0, t["revenue"] / t["budget"], np.nan)
    return t


def _prior_stats_by_date(rows, group_col):
    # For each (group, release_date) compute cumulative stats over STRICTLY earlier
    # release dates for that group. Movies on the same release date are treated as
    # concurrent and excluded (conservative, leak-safe).
    rows = rows.sort_values([group_col, "release_date"])
    per_date = rows.groupby([group_col, "release_date"]).agg(
        n=("revenue", "size"),
        sum_budget=("budget", "sum"),
        sum_revenue=("revenue", "sum"),
        sum_roi=("roi", "sum"),
        n_roi=("roi", lambda s: s.notna().sum()),
        max_revenue=("revenue", "max"),
    ).reset_index()
    per_date = per_date.sort_values([group_col, "release_date"])

    value_cols = ["n", "sum_budget", "sum_revenue", "sum_roi", "n_roi", "max_revenue"]
    cum = per_date.groupby(group_col)[value_cols].cumsum()
    prior_cum = cum.groupby(per_date[group_col]).shift(1)

    per_date["prior_n"] = prior_cum["n"]
    per_date["prior_avg_budget"] = prior_cum["sum_budget"] / prior_cum["n"]
    per_date["prior_avg_revenue"] = prior_cum["sum_revenue"] / prior_cum["n"]
    per_date["prior_avg_roi"] = prior_cum["sum_roi"] / prior_cum["n_roi"]
    per_date["prior_max_revenue"] = prior_cum["max_revenue"]

    keep = [
        "release_date", "prior_n", "prior_avg_budget", "prior_avg_revenue",
        "prior_avg_roi", "prior_max_revenue",
    ]
    res = per_date[keep + [group_col]].merge(
        rows[[group_col, "release_date", "movie_id"]],
        on=[group_col, "release_date"], how="right",
    )
    return res


def _explode_members(rows, id_col, member_col, member_name):
    # one row per (movie, member) for the given entity id array column
    exploded = rows.explode(member_col)
    exploded = exploded[exploded[member_col].notna()].copy()
    exploded[member_name] = exploded[member_col]
    return exploded.drop(columns=[member_col])


def _rollup(per_movie_member, prefix):
    g = per_movie_member.groupby("movie_id").agg(
        previous_movie_count=("prior_n", "sum"),
        previous_avg_revenue=("prior_avg_revenue", "mean"),
        previous_avg_budget=("prior_avg_budget", "mean"),
        previous_avg_roi=("prior_avg_roi", "mean"),
        previous_max_revenue=("prior_max_revenue", "max"),
    ).reset_index()
    return g.rename(columns={c: f"{prefix}_{c}" for c in g.columns if c != "movie_id"})


def compute_historical_features(history_df):
    # Leakage-safe prior-release aggregates for director, production company,
    # top-3 cast and franchise (collection). Prior = movies released strictly
    # before the current movie's release_date.
    t = prepare_history_table(history_df)

    prior_cols = [
        "director_previous_movie_count",
        "director_previous_avg_revenue",
        "director_previous_avg_budget",
        "director_previous_avg_roi",
        "director_previous_max_revenue",
        "production_company_previous_avg_revenue",
        "cast_previous_avg_revenue",
        "lead_actor_previous_avg_revenue",
        "franchise_previous_avg_revenue",
        "franchise_previous_avg_roi",
        "franchise_movie_count",
    ]
    out = pd.DataFrame({"movie_id": t["movie_id"].unique()})

    # Director - explode person ids
    dir_rows = _explode_members(t, "movie_id", "director_ids", "person_id")
    if len(dir_rows):
        dir_stats = _prior_stats_by_date(dir_rows, "person_id")
        out = out.merge(_rollup(dir_stats, "director"), on="movie_id", how="outer")

    # Production company - explode company ids
    comp_rows = _explode_members(t, "movie_id", "company_ids", "company_id")
    if len(comp_rows):
        comp_stats = _prior_stats_by_date(comp_rows, "company_id")
        comp_rollup = comp_stats.groupby("movie_id")["prior_avg_revenue"].mean().reset_index()
        out = out.merge(
            comp_rollup.rename(columns={"prior_avg_revenue": "production_company_previous_avg_revenue"}),
            on="movie_id", how="outer",
        )

    # Top-3 cast - explode person ids
    cast_rows = _explode_members(t, "movie_id", "cast_ids", "person_id")
    if len(cast_rows):
        cast_stats = _prior_stats_by_date(cast_rows, "person_id")
        cast_rollup = cast_stats.groupby("movie_id")["prior_avg_revenue"].mean().reset_index()
        out = out.merge(
            cast_rollup.rename(columns={"prior_avg_revenue": "cast_previous_avg_revenue"}),
            on="movie_id", how="outer",
        )

    # Lead actor - first-billed only (cast_ids is ordered by billing)
    lead = t[t["cast_ids"].apply(lambda x: bool(x))].copy()
    if len(lead):
        lead["person_id"] = lead["cast_ids"].apply(lambda x: x[0])
        lead = lead.drop(columns=["cast_ids"])
        lead_stats = _prior_stats_by_date(lead, "person_id")
        lead_rollup = lead_stats.groupby("movie_id")["prior_avg_revenue"].mean().reset_index()
        out = out.merge(
            lead_rollup.rename(columns={"prior_avg_revenue": "lead_actor_previous_avg_revenue"}),
            on="movie_id", how="outer",
        )

    # Franchise - single collection id per movie (exclude standalone rows)
    coll = t[t["belongs_to_collection_id"] > 0].copy()
    if len(coll):
        coll_stats = _prior_stats_by_date(coll, "belongs_to_collection_id")
        coll_stats = coll_stats.rename(columns={
            "prior_avg_revenue": "franchise_previous_avg_revenue",
            "prior_avg_roi": "franchise_previous_avg_roi",
            "prior_n": "franchise_movie_count",
        })[["movie_id", "franchise_previous_avg_revenue", "franchise_previous_avg_roi", "franchise_movie_count"]]
        out = out.merge(coll_stats, on="movie_id", how="outer")

    out = out[["movie_id"] + [c for c in prior_cols if c in out.columns]]
    return out


def brute_force_prior_revenue(history_df, movie_id, member_col, first_only=False):
    # Straightforward per-movie reference implementation used by the leak test.
    # Matches the rollup used in compute_historical_features: for each member of
    # the movie, prior avg revenue over that member's earlier movies, averaged
    # across the movie's members. first_only=True checks the first-billed (lead)
    # member only.
    t = prepare_history_table(history_df)
    row = t[t["movie_id"] == movie_id].iloc[0]
    members = row[member_col]
    if not members:
        return np.nan
    if first_only:
        members = members[:1]
    member_avgs = []
    for member in members:
        prior = t[(t["release_date"] < row["release_date"])]
        if first_only:
            # lead semantics: only prior films where this person was first-billed
            prior = prior[prior[member_col].apply(lambda x: bool(x or []) and x[0] == member)]
        else:
            prior = prior[prior[member_col].apply(lambda x: member in (x or []))]
        member_avgs.append(np.nan if len(prior) == 0 else prior["revenue"].mean())
    if not member_avgs or all(np.isnan(member_avgs)):
        return np.nan
    return np.nanmean(member_avgs)


def verify_historical_features(history_df, hist_features, n=25, seed=123):
    # Check sampled movies: prior avg revenue must match a brute-force recomputation
    rng = np.random.RandomState(seed)
    sample = rng.choice(hist_features["movie_id"], size=n, replace=False)
    checks = [
        ("director_previous_avg_revenue", "director_ids", False),
        ("production_company_previous_avg_revenue", "company_ids", False),
        ("cast_previous_avg_revenue", "cast_ids", False),
        ("lead_actor_previous_avg_revenue", "cast_ids", True),
    ]
    for feat, col, first_only in checks:
        diffs = []
        for mid in sample:
            expected = brute_force_prior_revenue(history_df, mid, col, first_only=first_only)
            got = hist_features.loc[hist_features["movie_id"] == mid, feat].iloc[0]
            if expected != expected:  # NaN both ways
                diffs.append(0.0 if got != got else 1.0)
            else:
                diffs.append(abs(got - expected) / expected if expected != 0 else 0.0)
        max_rel = max(diffs)
        print(f"{feat}: max relative difference vs brute force = {max_rel:.4f}")
        assert np.allclose(diffs, 0, atol=1e-6), f"leak check failed for {feat}"
    print("verify_historical_features: passed")


if __name__ == "__main__":
    from scripts.load_data import load_history_data

    hist = load_history_data()
    features = compute_historical_features(hist)
    print(features.shape)
    verify_historical_features(hist, features)
    print(features.describe().T)