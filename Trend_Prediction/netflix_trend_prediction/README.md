# Auspify Technologies — Task 4: Trend Prediction Analysis

## Objective
Analyze historical Netflix content trends and forecast future content growth, following the internship task guide.

## Task workflow
1. Prepare release-year data.
2. Analyze yearly content trends.
3. Build forecasting models.
4. Visualize future predictions.
5. Interpret results.

## Project structure
```text
auspify_task4_netflix_trend_prediction/
├── data/
│   └── netflix_raw.csv
├── src/
│   └── trend_prediction.py
├── output/
│   ├── yearly_content_counts.csv
│   ├── future_trend_forecast.csv
│   ├── trend_analysis_report.txt
│   └── plots/
│       ├── historical_yearly_trend.png
│       └── future_trend_forecast.png
├── requirements.txt
└── README.md
```

## Run in Windows / VS Code
Open the extracted project folder in VS Code. In Terminal → New Terminal, run:
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python src/trend_prediction.py
```

To forecast a different number of future years:
```powershell
python src/trend_prediction.py --forecast-horizon 10
```

## Method
- Converts `release_year` to numeric and excludes rows with missing/invalid years.
- Counts titles for each release year and fills missing years in the observed range with zero counts.
- Trains a linear regression baseline using earlier years.
- Evaluates on a chronological holdout (latest 20% of years).
- Refits on all observed years and forecasts the next five years.
- Produces historical and forecast charts plus CSV reports.

## Interpretation caveat
The forecast is an illustrative extrapolation of title release-year counts in the supplied dataset. It is not a prediction of how many titles Netflix will add to its platform in future years. The linear regression baseline is simple and may not capture complex time-series patterns.
