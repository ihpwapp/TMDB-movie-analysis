import ast
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# allow `python scripts/load_data.py` to resolve the scripts package
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.validate_data import validate_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACT_CSV = PROJECT_ROOT / "extract_movie_features.csv"
HISTORY_CSV = PROJECT_ROOT / "data" / "history_features.csv"


def load_feature_dataset(path=EXTRACT_CSV):
    # ML dataset produced by queries/extract_movie_features.sql
    df = pd.read_csv(path)
    df["release_date"] = pd.to_datetime(df["release_date"])
    return df


def load_history_data(path=HISTORY_CSV):
    # Unfiltered movie rows from queries/extract_history_features.sql
    # used only to compute leakage-safe historical features
    df = pd.read_csv(path)
    df["release_date"] = pd.to_datetime(df["release_date"])
    for col in ["director_ids", "company_ids", "cast_ids"]:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    df["belongs_to_collection_id"] = pd.to_numeric(
        df["belongs_to_collection_id"], errors="coerce"
    ).fillna(0).astype(int)
    return df


def load_feature_dataset_from_db(query_path=PROJECT_ROOT / "queries" / "extract_movie_features.sql"):
    # Re-run the extract query against PostgreSQL (used to (re)generate the CSV)
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    engine = create_engine(os.getenv("DATABASE_URL"))
    query = query_path.read_text(encoding="utf-8")
    return pd.read_sql(query, con=engine)


if __name__ == "__main__":
    df = load_feature_dataset()
    validate_dataset(df)