# Auspify Technologies — Task 2: Exploratory Data Analysis (EDA)

## Objective
Explore the supplied Netflix dataset to uncover trends, patterns, and insights, following the Task 2 workflow in the Auspify Data Science Task List.

## Task guide workflow
1. Analyze dataset structure and statistics.
2. Study content distribution by type.
3. Identify top countries and categories.
4. Create visualizations for key metrics.
5. Summarize findings.

## Project structure
```text
auspify_task2_netflix_eda/
├── data/
│   └── netflix_raw.csv
├── src/
│   └── netflix_eda.py
├── output/
│   ├── dataset_profile.txt
│   ├── descriptive_statistics.csv
│   ├── content_type_distribution.csv
│   ├── top_countries.csv
│   ├── top_categories.csv
│   ├── titles_by_release_year.csv
│   ├── rating_distribution.csv
│   ├── eda_findings.txt
│   └── plots/
├── requirements.txt
└── README.md
```

## Run on Windows / VS Code
1. Extract the ZIP.
2. Open this project folder in VS Code.
3. In Terminal → New Terminal, run:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install -r requirements.txt
   python src/netflix_eda.py
   ```
4. Open the `output` folder to review the reports, CSV summaries, and PNG charts.

If PowerShell blocks virtual environment activation, use Command Prompt with `.venv\Scripts\activate.bat`, or run `python -m pip install -r requirements.txt` and `python src/netflix_eda.py` from the project folder.

## Outputs
- `dataset_profile.txt`: dimensions, data types, missing-value counts, duplicate count, and descriptive statistics.
- `descriptive_statistics.csv`: summary statistics by column.
- `content_type_distribution.csv`: Movie vs TV Show counts.
- `top_countries.csv`: top country mentions after splitting multi-country entries.
- `top_categories.csv`: top category mentions after splitting multi-category entries.
- `titles_by_release_year.csv`: title counts by release year.
- `rating_distribution.csv`: common content ratings.
- `eda_findings.txt`: automatically generated findings based on the supplied data.
- `plots/`: visualizations for content type, countries, categories, release years, and ratings.

## Notes
- The analysis uses the supplied CSV as-is; it does not replace or silently clean missing values.
- Country and category fields can contain comma-separated values, so the analysis splits them before counting.
- Findings are descriptive of this dataset snapshot only.
