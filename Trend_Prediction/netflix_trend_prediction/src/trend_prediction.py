"""
Auspify Technologies — Data Science Internship Task 4
Trend Prediction Analysis for Netflix content.

Workflow from the internship task guide:
1. Prepare release-year data.
2. Analyze yearly content trends.
3. Build forecasting models.
4. Visualize future predictions.
5. Interpret results.

Run from project root:
    python src/trend_prediction.py
"""
from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

def build_yearly_counts(df):
    if "release_year" not in df.columns:
        raise ValueError("The dataset must contain a 'release_year' column.")
    years = pd.to_numeric(df["release_year"], errors="coerce")
    years = years.dropna().astype(int)
    counts = years.value_counts().sort_index()
    yearly = counts.rename_axis("release_year").reset_index(name="title_count")
    # Add missing years between observed minimum and maximum so the time series is continuous.
    if not yearly.empty:
        full_years = pd.DataFrame({"release_year": range(int(yearly["release_year"].min()), int(yearly["release_year"].max()) + 1)})
        yearly = full_years.merge(yearly, on="release_year", how="left").fillna({"title_count": 0})
        yearly["title_count"] = yearly["title_count"].astype(int)
    return yearly

def run_analysis(input_path: Path, output_dir: Path, forecast_horizon: int = 5):
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    yearly = build_yearly_counts(df)
    if len(yearly) < 5:
        raise ValueError("Not enough release-year observations to fit a basic forecasting model.")

    yearly.to_csv(output_dir / "yearly_content_counts.csv", index=False)

    # Plot historical yearly content counts.
    plt.figure(figsize=(11, 5))
    plt.plot(yearly["release_year"], yearly["title_count"], marker="o", linewidth=1)
    plt.title("Netflix Dataset: Titles by Release Year")
    plt.xlabel("Release year")
    plt.ylabel("Number of titles in dataset")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "historical_yearly_trend.png", dpi=160)
    plt.close()

    # Hold out the latest 20% of observed years for a simple time-ordered test.
    n_test = max(1, int(np.ceil(len(yearly) * 0.2)))
    train = yearly.iloc[:-n_test]
    test = yearly.iloc[-n_test:]
    model = LinearRegression()
    model.fit(train[["release_year"]], train["title_count"])
    test_pred = np.clip(model.predict(test[["release_year"]]), 0, None)
    mae = mean_absolute_error(test["title_count"], test_pred)
    rmse = float(np.sqrt(mean_squared_error(test["title_count"], test_pred)))

    # Fit the model on all historical years to produce future trend estimates.
    final_model = LinearRegression()
    final_model.fit(yearly[["release_year"]], yearly["title_count"])
    last_year = int(yearly["release_year"].max())
    future_years = np.arange(last_year + 1, last_year + forecast_horizon + 1)
    future_pred = np.clip(final_model.predict(pd.DataFrame({"release_year": future_years})), 0, None)
    forecast = pd.DataFrame({
        "release_year": future_years,
        "predicted_title_count": np.rint(future_pred).astype(int),
        "predicted_title_count_raw": future_pred.round(2)
    })
    forecast.to_csv(output_dir / "future_trend_forecast.csv", index=False)

    # Plot historical series and simple forecast.
    plt.figure(figsize=(11, 5))
    plt.plot(yearly["release_year"], yearly["title_count"], label="Historical dataset counts", marker="o", linewidth=1)
    plt.plot(forecast["release_year"], forecast["predicted_title_count_raw"], label="Linear regression forecast", marker="o", linestyle="--")
    plt.axvline(last_year, linestyle=":", linewidth=1)
    plt.title("Historical Trend and Illustrative Future Forecast")
    plt.xlabel("Release year")
    plt.ylabel("Number of titles in supplied dataset")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "future_trend_forecast.png", dpi=160)
    plt.close()

    # Release year summary and interpretation.
    yearly_sum = int(yearly["title_count"].sum())
    peak_row = yearly.loc[yearly["title_count"].idxmax()]
    first_year = int(yearly["release_year"].min())
    peak_year = int(peak_row["release_year"])
    latest_year = int(yearly["release_year"].max())
    latest_count = int(yearly.loc[yearly["release_year"] == latest_year, "title_count"].iloc[0])
    report = [
        "AUSPIFY TECHNOLOGIES — TASK 4: TREND PREDICTION ANALYSIS",
        "=" * 59,
        f"Rows in source dataset: {len(df):,}",
        f"Records with valid release_year: {yearly_sum:,}",
        f"Release-year range: {first_year}–{latest_year}",
        f"Year with highest title count in this dataset: {peak_year} ({int(peak_row['title_count']):,} titles)",
        f"Title count at latest release year ({latest_year}): {latest_count:,}",
        "",
        "Forecast model:",
        "Linear Regression fitted to yearly counts. The latest 20% of years (minimum one year) were held out for a chronological test.",
        f"Holdout MAE: {mae:.2f} titles",
        f"Holdout RMSE: {rmse:.2f} titles",
        "",
        "Future trend estimates:",
    ]
    for _, row in forecast.iterrows():
        report.append(f"- {int(row['release_year'])}: approximately {int(row['predicted_title_count']):,} titles (model estimate)")
    report += [
        "",
        "Interpretation and limitations:",
        "- These counts represent titles in the supplied Netflix dataset grouped by release_year; they do not represent Netflix's actual yearly additions to its platform.",
        "- The dataset has a latest release year, so forecasts beyond that year are extrapolations and should be treated as illustrative.",
        "- Linear regression is a simple baseline and may not capture nonlinear patterns, catalog changes, or external factors.",
        "- Negative model predictions are clipped to zero before reporting.",
        "- The holdout metrics measure error on the final portion of the historical series and do not guarantee future accuracy."
    ]
    (output_dir / "trend_analysis_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nSaved outputs to: {output_dir.resolve()}")
    return yearly, forecast

def main():
    parser = argparse.ArgumentParser(description="Netflix trend prediction analysis for Auspify Task 4.")
    parser.add_argument("--input", default="data/netflix_raw.csv", help="Path to the source CSV")
    parser.add_argument("--output", default="output", help="Directory for reports and charts")
    parser.add_argument("--forecast-horizon", type=int, default=5, help="Number of future years to forecast (default: 5)")
    args = parser.parse_args()
    if args.forecast_horizon < 1:
        parser.error("--forecast-horizon must be at least 1")
    run_analysis(Path(args.input), Path(args.output), args.forecast_horizon)

if __name__ == "__main__":
    main()
