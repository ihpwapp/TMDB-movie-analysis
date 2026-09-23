import pandas as pd


def validate_dataset(df):
    # Lightweight checks for the ML dataset before modelling
    required = [
        "movie_id", "release_date", "budget", "revenue", "runtime",
        "primary_genre", "primary_production_company", "primary_country",
        "director_name", "cast",
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"rows: {len(df)}")
    print(f"columns: {len(df.columns)}")

    dupes = df["movie_id"].duplicated().sum()
    print(f"duplicate movie_ids: {dupes}")
    if dupes:
        df = df.drop_duplicates(subset="movie_id")

    n_null_target = df["revenue"].isna().sum()
    n_zero_target = (df["revenue"] == 0).sum()
    print(f"revenue nulls: {n_null_target}, revenue == 0: {n_zero_target}")

    n_bad_budget = ((df["budget"].isna()) | (df["budget"] <= 0)).sum()
    n_bad_runtime = ((df["runtime"].isna()) | (df["runtime"] <= 0)).sum()
    print(f"budget <= 0 or null: {n_bad_budget}, runtime <= 0 or null: {n_bad_runtime}")

    unexpected = [c for c in df.columns if c.startswith("Unnamed")]
    if unexpected:
        print(f"warning: unexpected columns {unexpected} - file may be corrupted")

    # target should be strictly positive after the extract filter
    valid_target = df["revenue"].notna() & (df["revenue"] > 0)
    print(f"rows with valid target (revenue > 0): {valid_target.sum()}")

    return df