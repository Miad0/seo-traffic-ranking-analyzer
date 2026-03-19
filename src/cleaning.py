from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "date",
    "keyword",
    "url",
    "clicks",
    "impressions",
    "ctr",
    "position",
]


def load_data(path: str) -> pd.DataFrame:
    """Load CSV data and validate basic structure."""
    df = pd.read_csv(path)

    legacy_name_map = {"query": "keyword", "page": "url"}
    df = df.rename(columns={old: new for old, new in legacy_name_map.items() if old in df.columns and new not in df.columns})

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize SEO data."""
    cleaned = df.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["keyword"] = cleaned["keyword"].astype(str).str.strip().str.lower()
    cleaned["url"] = cleaned["url"].astype(str).str.strip().str.lower()

    numeric_cols = ["clicks", "impressions", "ctr", "position"]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned = cleaned.dropna(subset=["date", "keyword", "url", "impressions", "position"])
    cleaned["clicks"] = cleaned["clicks"].fillna(0)
    cleaned["ctr"] = cleaned["ctr"].fillna(cleaned["clicks"] / cleaned["impressions"].replace(0, pd.NA))
    cleaned["ctr"] = cleaned["ctr"].fillna(0)

    cleaned["clicks"] = cleaned["clicks"].clip(lower=0)
    cleaned["impressions"] = cleaned["impressions"].clip(lower=0)
    cleaned["ctr"] = cleaned["ctr"].clip(lower=0, upper=1)
    cleaned["position"] = cleaned["position"].clip(lower=1)

    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.sort_values(["date", "clicks", "impressions"], ascending=[True, False, False])
    cleaned.reset_index(drop=True, inplace=True)
    return cleaned
