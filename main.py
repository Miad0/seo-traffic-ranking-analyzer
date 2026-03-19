from __future__ import annotations

from pathlib import Path

from src.cleaning import clean_data, load_data
from src.analysis import (
    cannibalization_report,
    create_text_summary,
    daily_performance,
    keyword_opportunities,
    low_ctr_pages,
    summarize_sitewide,
    keywords_likely_to_reach_top3_candidates,
    top_pages,
    traffic_prediction_report,
)
from src.visualization import (
    plot_ctr_vs_position,
    plot_daily_clicks,
    plot_prediction_scatter,
    plot_keywords_likely_to_reach_top3_probability,
    plot_top_pages,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "seo_sample_data.csv"
OUTPUT_DIR = BASE_DIR / "output"


def save_dataframe(df, path: Path) -> None:
    df.to_csv(path, index=False)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    raw_df = load_data(str(DATA_PATH))
    df = clean_data(raw_df)

    sitewide = summarize_sitewide(df)
    top_pages_df = top_pages(df)
    opportunities_df = keyword_opportunities(df)
    low_ctr_df = low_ctr_pages(df)
    cannibalization_df = cannibalization_report(df)
    daily_df = daily_performance(df)
    prediction_df, prediction_metrics = traffic_prediction_report(df)
    candidates_df, _ = keywords_likely_to_reach_top3_candidates(df)

    save_dataframe(top_pages_df, OUTPUT_DIR / "top_urls.csv")
    save_dataframe(opportunities_df, OUTPUT_DIR / "keyword_opportunities.csv")
    save_dataframe(low_ctr_df, OUTPUT_DIR / "low_ctr_urls.csv")
    save_dataframe(cannibalization_df, OUTPUT_DIR / "cannibalization_report.csv")
    save_dataframe(daily_df, OUTPUT_DIR / "daily_performance.csv")
    save_dataframe(prediction_df, OUTPUT_DIR / "traffic_predictions.csv")
    save_dataframe(candidates_df, OUTPUT_DIR / "keywords_likely_to_reach_top3.csv")

    plot_daily_clicks(daily_df, str(OUTPUT_DIR))
    plot_ctr_vs_position(df, str(OUTPUT_DIR))
    plot_top_pages(top_pages_df, str(OUTPUT_DIR))
    plot_prediction_scatter(prediction_df, str(OUTPUT_DIR))
    plot_keywords_likely_to_reach_top3_probability(candidates_df, str(OUTPUT_DIR))

    summary = create_text_summary(sitewide, prediction_metrics)
    with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary + "\n\n")
        f.write("Main deliverables created in /output:\n")
        f.write("- top_urls.csv\n")
        f.write("- keyword_opportunities.csv\n")
        f.write("- low_ctr_urls.csv\n")
        f.write("- cannibalization_report.csv\n")
        f.write("- daily_performance.csv\n")
        f.write("- traffic_predictions.csv\n")
        f.write("- keywords_likely_to_reach_top3.csv\n")
        f.write("- daily_clicks.png\n")
        f.write("- ctr_vs_position.png\n")
        f.write("- top_pages_clicks.png\n")
        f.write("- predicted_vs_actual_clicks.png\n")
        f.write("- keywords_likely_to_reach_top3_probability.png\n\n")
        f.write("Prediction metrics:\n")
        f.write(f"- Traffic model MAE: {prediction_metrics['click_model_mae']:.2f}\n")
        f.write(f"- Traffic model R^2: {prediction_metrics['click_model_r2']:.3f}\n")
        f.write(f"- Ranking model MAE: {prediction_metrics['position_model_mae']:.2f}\n")
        f.write(f"- Ranking model R^2: {prediction_metrics['position_model_r2']:.3f}\n")

    print(summary)
    print("Project run completed. Check the output folder.")
