# Auspify Technologies — Task 6: Data Science Business Insights Dashboard

## Objective
Build an end-to-end Netflix catalogue analytics project combining data analysis, a predictive model, interactive visualizations, business recommendations, and a professional report.

## Project contents
- `app.py` — interactive Streamlit dashboard with filters, Plotly charts, ML classifier, confusion matrix, and recommendations.
- `data/netflix_titles.csv` — provided dataset copy.
- `analysis.py` — reproducible batch analysis that creates charts, tables, and a report.
- `outputs/` — pre-generated outputs from the supplied dataset.
- `requirements.txt` — Python dependencies.
- `run_windows.bat` — Windows setup and launch helper.

## Run on Windows
1. Extract the ZIP file.
2. Open the extracted project folder in VS Code.
3. Open Terminal in that folder.
4. Recommended: use Python 3.11 or 3.12 and create a virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
5. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. Start the dashboard:
   ```powershell
   python -m streamlit run app.py
   ```
7. Open the local URL printed in the terminal (normally `http://localhost:8501`).

You can also double-click `run_windows.bat` after extracting the project.

## Task 6 workflow coverage
1. Complete data analysis — dataset structure, missing values, duplicates, type mix, release-year trend, ratings, countries, and categories.
2. Predictive model — Logistic Regression classifier for `type` (Movie/TV Show), compared against a majority-class baseline.
3. Interactive visualizations — filter by type, country, release year, and title; Plotly charts; downloadable filtered data.
4. Business recommendations — practical, caveated recommendations in the dashboard.
5. Professional report — generated in `outputs/reports/`.

## Important interpretation notes
- The model predicts catalogue content type, not popularity, revenue, viewing hours, or user preferences.
- Country and category columns can contain multiple comma-separated values; counts are mention counts.
- Extremely high accuracy can occur because catalogue metadata may closely encode the content type. Check the confusion matrix and baseline, and do not present the score as proof of business value.
- The dataset does not include customer viewing, revenue, subscription, or cost data. Recommendations are descriptive hypotheses, not causal claims.
