from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_daily_clicks(daily_df: pd.DataFrame, output_dir: str) -> str:
    output_path = Path(output_dir) / "daily_clicks.png"
    plt.figure(figsize=(10, 5))
    plt.plot(daily_df["date"], daily_df["clicks"])
    plt.title("Daily Clicks")
    plt.xlabel("Date")
    plt.ylabel("Clicks")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return str(output_path)


def plot_ctr_vs_position(df: pd.DataFrame, output_dir: str) -> str:
    output_path = Path(output_dir) / "ctr_vs_position.png"
    grouped = (
        df.groupby("keyword", as_index=False)
        .agg(clicks=("clicks", "sum"), impressions=("impressions", "sum"), avg_position=("position", "mean"))
    )
    grouped["ctr"] = grouped["clicks"] / grouped["impressions"].replace(0, pd.NA)

    plt.figure(figsize=(8, 5))
    plt.scatter(grouped["avg_position"], grouped["ctr"], alpha=0.6)
    plt.title("CTR vs Average Position")
    plt.xlabel("Average Position")
    plt.ylabel("CTR")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return str(output_path)


def plot_top_pages(top_pages_df: pd.DataFrame, output_dir: str) -> str:
    output_path = Path(output_dir) / "top_pages_clicks.png"
    data = top_pages_df.sort_values("clicks", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(data["url"], data["clicks"])
    plt.title("Top URLs by Clicks")
    plt.xlabel("Clicks")
    plt.ylabel("URL")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return str(output_path)


def plot_prediction_scatter(prediction_df: pd.DataFrame, output_dir: str) -> str:
    output_path = Path(output_dir) / "predicted_vs_actual_clicks.png"
    plt.figure(figsize=(8, 5))
    plt.scatter(prediction_df["actual_future_clicks"], prediction_df["predicted_future_clicks"], alpha=0.7)
    plt.title("Predicted vs Actual Future Clicks")
    plt.xlabel("Actual Future Clicks")
    plt.ylabel("Predicted Future Clicks")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return str(output_path)


def plot_keywords_likely_to_reach_top3_probability(candidates_df: pd.DataFrame, output_dir: str) -> str:
    output_path = Path(output_dir) / "keywords_likely_to_reach_top3_probability.png"
    data = candidates_df.sort_values("current_top3_probability", ascending=True).tail(10)
    plt.figure(figsize=(10, 6))
    plt.barh(data["keyword"], data["current_top3_probability"])
    plt.title("Keywords Likely to Reach Top 3")
    plt.xlabel("Estimated Probability")
    plt.ylabel("Keyword")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return str(output_path)
