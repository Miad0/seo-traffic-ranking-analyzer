from __future__ import annotations

import streamlit as st

from src.cleaning import clean_data, load_data
from src.analysis import (
    cannibalization_report,
    daily_performance,
    keyword_opportunities,
    low_ctr_pages,
    summarize_sitewide,
    keywords_likely_to_reach_top3_candidates,
    top_pages,
    traffic_prediction_report,
)

st.set_page_config(page_title="SEO Traffic & Ranking Analyzer", layout="wide")
st.title("SEO Traffic & Ranking Analyzer")
st.caption("Python data analysis dashboard for SEO performance insights.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
file_path = uploaded_file if uploaded_file is not None else "data/seo_sample_data.csv"

df = clean_data(load_data(file_path))
sitewide = summarize_sitewide(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Clicks", f"{sitewide['total_clicks']:,}")
col2.metric("Impressions", f"{sitewide['total_impressions']:,}")
col3.metric("Weighted CTR", f"{sitewide['weighted_ctr']:.2%}")
col4.metric("Avg Position", sitewide["avg_position"])

st.subheader("Daily Performance")
daily_df = daily_performance(df).set_index("date")
st.line_chart(daily_df[["clicks", "impressions"]])

st.subheader("Top URLs")
st.dataframe(top_pages(df), use_container_width=True)

st.subheader("Keyword Opportunities")
st.dataframe(keyword_opportunities(df), use_container_width=True)

st.subheader("Low CTR URLs")
st.dataframe(low_ctr_pages(df), use_container_width=True)

st.subheader("Keyword Cannibalization")
st.dataframe(cannibalization_report(df), use_container_width=True)

st.subheader("Traffic Prediction")
prediction_df, prediction_metrics = traffic_prediction_report(df)
st.dataframe(prediction_df, use_container_width=True)
col5, col6 = st.columns(2)
col5.metric("Traffic Model MAE", f"{prediction_metrics['click_model_mae']:.2f}")
col6.metric("Ranking Model MAE", f"{prediction_metrics['position_model_mae']:.2f}")

st.subheader("Keywords Likely to Reach Top 3")
candidates_df, _ = keywords_likely_to_reach_top3_candidates(df)
st.dataframe(candidates_df, use_container_width=True)
