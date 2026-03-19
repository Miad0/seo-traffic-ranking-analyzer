# SEO Traffic & Ranking Analyzer

A Python project for analyzing SEO performance data and generating practical insights from search metrics.

## Features
- Data cleaning and preprocessing
- Keyword performance analysis
- CTR versus position analysis
- Detection of keyword cannibalization
- Traffic and ranking prediction using regression models
- Data visualization with charts
- Optional Streamlit dashboard

## Tech Stack
- Python
- pandas
- numpy
- matplotlib
- scikit-learn
- streamlit

## Project Structure
```text
seo_traffic_ranking_analyzer/
├── app.py
├── main.py
├── README.md
├── requirements.txt
├── data/
│   └── seo_sample_data.csv
├── output/
└── src/
    ├── analysis.py
    ├── cleaning.py
    └── visualization.py
```

## Usage

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the analysis:
```bash
python main.py
```

Run the dashboard:
```bash
streamlit run app.py
```

## Generated Output
The project creates CSV reports and chart images in the `output/` folder, including:
- `top_urls.csv`
- `keyword_opportunities.csv`
- `low_ctr_urls.csv`
- `cannibalization_report.csv`
- `daily_performance.csv`
- `traffic_predictions.csv`
- `keywords_likely_to_reach_top3.csv`
- `daily_clicks.png`
- `ctr_vs_position.png`
- `top_pages_clicks.png`
- `predicted_vs_actual_clicks.png`
- `keywords_likely_to_reach_top3_probability.png`

## Dataset Note
The included CSV is synthetic sample data for demonstration and testing.
