from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def summarize_sitewide(df: pd.DataFrame) -> dict:
    total_clicks = int(df["clicks"].sum())
    total_impressions = int(df["impressions"].sum())
    weighted_ctr = (total_clicks / total_impressions) if total_impressions else 0
    avg_position = round(df["position"].mean(), 2)

    return {
        "total_clicks": total_clicks,
        "total_impressions": total_impressions,
        "weighted_ctr": round(weighted_ctr, 4),
        "avg_position": avg_position,
        "unique_keywords": int(df["keyword"].nunique()),
        "unique_urls": int(df["url"].nunique()),
    }


def top_pages(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    grouped = (
        df.groupby("url", as_index=False)
        .agg(
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
            avg_position=("position", "mean"),
        )
        .sort_values("clicks", ascending=False)
        .head(n)
    )
    grouped["ctr"] = grouped["clicks"] / grouped["impressions"].replace(0, pd.NA)
    return grouped


def keyword_opportunities(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    grouped = (
        df.groupby(["keyword", "url"], as_index=False)
        .agg(
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
            avg_position=("position", "mean"),
        )
    )
    grouped["ctr"] = grouped["clicks"] / grouped["impressions"].replace(0, pd.NA)

    mask = (grouped["avg_position"] >= 8) & (grouped["avg_position"] <= 15) & (grouped["impressions"] >= 200)
    opportunities = grouped.loc[mask].copy()
    opportunities["opportunity_score"] = opportunities["impressions"] / opportunities["avg_position"]
    return opportunities.sort_values("opportunity_score", ascending=False).head(n)


def low_ctr_pages(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    grouped = (
        df.groupby("url", as_index=False)
        .agg(clicks=("clicks", "sum"), impressions=("impressions", "sum"), avg_position=("position", "mean"))
    )
    grouped["ctr"] = grouped["clicks"] / grouped["impressions"].replace(0, pd.NA)
    mask = (grouped["impressions"] >= 500) & (grouped["avg_position"] <= 10)
    return grouped.loc[mask].sort_values(["ctr", "impressions"], ascending=[True, False]).head(n)


def cannibalization_report(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    grouped = (
        df.groupby("keyword")
        .agg(
            unique_urls=("url", "nunique"),
            total_clicks=("clicks", "sum"),
            total_impressions=("impressions", "sum"),
        )
        .reset_index()
    )
    cannibalized = grouped[grouped["unique_urls"] > 1].copy()
    return cannibalized.sort_values(["unique_urls", "total_impressions"], ascending=[False, False]).head(n)


def daily_performance(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby("date", as_index=False)
        .agg(clicks=("clicks", "sum"), impressions=("impressions", "sum"), avg_position=("position", "mean"))
        .sort_values("date")
    )
    daily["ctr"] = daily["clicks"] / daily["impressions"].replace(0, pd.NA)
    return daily


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0, pd.NA)


def build_prediction_dataset(df: pd.DataFrame) -> pd.DataFrame:
    ordered_dates = sorted(df["date"].dropna().unique())
    if len(ordered_dates) < 6:
        raise ValueError("Prediction module needs at least 6 distinct dates.")

    split_idx = max(3, math.floor(len(ordered_dates) * 0.7))
    split_idx = min(split_idx, len(ordered_dates) - 2)
    train_dates = ordered_dates[:split_idx]
    future_dates = ordered_dates[split_idx:]

    history = df[df["date"].isin(train_dates)].copy()
    future = df[df["date"].isin(future_dates)].copy()

    history_agg = (
        history.groupby(["keyword", "url"], as_index=False)
        .agg(
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
            avg_position=("position", "mean"),
            best_position=("position", "min"),
            days_active=("date", "nunique"),
        )
    )
    history_agg["ctr"] = _safe_ratio(history_agg["clicks"], history_agg["impressions"]).fillna(0)

    recent_cutoff = train_dates[-min(3, len(train_dates)):]
    recent = history[history["date"].isin(recent_cutoff)].copy()
    recent_agg = (
        recent.groupby(["keyword", "url"], as_index=False)
        .agg(
            recent_clicks=("clicks", "sum"),
            recent_impressions=("impressions", "sum"),
            recent_avg_position=("position", "mean"),
        )
    )
    recent_agg["recent_ctr"] = _safe_ratio(recent_agg["recent_clicks"], recent_agg["recent_impressions"]).fillna(0)

    future_agg = (
        future.groupby(["keyword", "url"], as_index=False)
        .agg(
            future_clicks=("clicks", "sum"),
            future_avg_position=("position", "mean"),
        )
    )

    dataset = history_agg.merge(recent_agg, on=["keyword", "url"], how="left").merge(
        future_agg, on=["keyword", "url"], how="inner"
    )
    dataset = dataset.fillna({
        "recent_clicks": 0,
        "recent_impressions": 0,
        "recent_avg_position": dataset["avg_position"],
        "recent_ctr": 0,
    })
    dataset["position_gap_to_top3"] = (dataset["avg_position"] - 3).clip(lower=0)
    dataset["recent_position_gain"] = dataset["avg_position"] - dataset["recent_avg_position"]
    dataset["impressions_per_day"] = dataset["impressions"] / dataset["days_active"].replace(0, pd.NA)
    dataset["clicks_per_day"] = dataset["clicks"] / dataset["days_active"].replace(0, pd.NA)
    dataset = dataset.fillna(0)
    return dataset


def train_prediction_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    dataset = build_prediction_dataset(df)
    feature_cols = [
        "clicks",
        "impressions",
        "avg_position",
        "best_position",
        "days_active",
        "ctr",
        "recent_clicks",
        "recent_impressions",
        "recent_avg_position",
        "recent_ctr",
        "position_gap_to_top3",
        "recent_position_gain",
        "impressions_per_day",
        "clicks_per_day",
    ]
    X = dataset[feature_cols]

    click_model = LinearRegression()
    click_model.fit(X, dataset["future_clicks"])
    position_model = LinearRegression()
    position_model.fit(X, dataset["future_avg_position"])

    dataset = dataset.copy()
    dataset["predicted_future_clicks"] = np.maximum(click_model.predict(X), 0)
    dataset["predicted_future_position"] = np.maximum(position_model.predict(X), 1)
    dataset["current_top3_probability"] = 1 / (1 + np.exp(dataset["predicted_future_position"] - 3))
    dataset["likely_to_reach_top3"] = dataset["predicted_future_position"] <= 3

    metrics = {
        "click_model_mae": float(mean_absolute_error(dataset["future_clicks"], dataset["predicted_future_clicks"])),
        "click_model_r2": float(r2_score(dataset["future_clicks"], dataset["predicted_future_clicks"])),
        "position_model_mae": float(mean_absolute_error(dataset["future_avg_position"], dataset["predicted_future_position"])),
        "position_model_r2": float(r2_score(dataset["future_avg_position"], dataset["predicted_future_position"])),
        "training_rows": int(len(dataset)),
        "feature_columns": feature_cols,
    }
    return dataset, metrics


def traffic_prediction_report(df: pd.DataFrame, n: int = 15) -> tuple[pd.DataFrame, dict]:
    dataset, metrics = train_prediction_models(df)
    report = dataset[[
        "keyword",
        "url",
        "clicks",
        "future_clicks",
        "predicted_future_clicks",
        "avg_position",
        "predicted_future_position",
        "current_top3_probability",
    ]].copy()
    report = report.rename(columns={
        "clicks": "historical_clicks",
        "future_clicks": "actual_future_clicks",
        "avg_position": "historical_avg_position",
    })
    report["predicted_click_growth"] = report["predicted_future_clicks"] - report["historical_clicks"]
    report = report.sort_values("predicted_future_clicks", ascending=False).head(n)
    return report, metrics


def keywords_likely_to_reach_top3_candidates(df: pd.DataFrame, n: int = 15) -> tuple[pd.DataFrame, dict]:
    dataset, metrics = train_prediction_models(df)
    candidates = dataset.loc[(dataset["avg_position"] > 3) & (dataset["avg_position"] <= 15)].copy()
    candidates = candidates[[
        "keyword",
        "url",
        "avg_position",
        "recent_avg_position",
        "predicted_future_position",
        "current_top3_probability",
        "impressions",
        "clicks",
        "predicted_future_clicks",
        "recent_position_gain",
    ]]
    candidates = candidates.rename(columns={
        "avg_position": "historical_avg_position",
        "clicks": "historical_clicks",
    })
    candidates = candidates.sort_values(
        ["current_top3_probability", "impressions", "predicted_future_clicks"],
        ascending=[False, False, False],
    ).head(n)
    return candidates, metrics


def create_text_summary(sitewide: dict, metrics: dict | None = None) -> str:
    summary = (
        f"The dataset contains {sitewide['unique_keywords']} unique keywords and {sitewide['unique_urls']} unique URLs. "
        f"Across the full period, the site generated {sitewide['total_clicks']:,} clicks from "
        f"{sitewide['total_impressions']:,} impressions, with a weighted CTR of "
        f"{sitewide['weighted_ctr']:.2%} and an average position of {sitewide['avg_position']}."
    )
    if metrics:
        summary += (
            f" The optional prediction module trained on {metrics['training_rows']} keyword-URL rows. "
            f"Its traffic model MAE is {metrics['click_model_mae']:.2f} clicks and its ranking model MAE is "
            f"{metrics['position_model_mae']:.2f} average positions."
        )
    return summary
