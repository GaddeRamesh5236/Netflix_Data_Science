# Auspify Data Science Internship — Task 1
## Data Cleaning & Preprocessing (Netflix Dataset)

This project follows the Task 1 workflow in the Auspify Technologies task guide:
1. Import the dataset using Pandas.
2. Identify missing and duplicate records.
3. Handle null values and inconsistent formats.
4. Transform categorical and date-related columns.
5. Create a clean dataset for further analysis.

## Project structure
```text
auspify_task1_data_cleaning/
├── data/
│   └── netflix_raw.csv
├── output/
│   └── (generated files appear here after running)
├── src/
│   └── clean_netflix.py
├── requirements.txt
└── README.md
```

## Requirements
- Python 3.10 or newer recommended
- Pandas

## Run on Windows (PowerShell / VS Code terminal)
1. Extract the ZIP file.
2. Open the extracted `auspify_task1_data_cleaning` folder in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment:
   ```powershell
   python -m venv .venv
   ```
5. Activate it:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, run the script using `python -m pip` commands in the same terminal, or use Command Prompt and run `.venv\Scripts\activate.bat`.
6. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
7. Run the task:
   ```powershell
   python src/clean_netflix.py
   ```

## Output files
After the script runs, check the `output` folder:
- `netflix_cleaned.csv` — cleaned dataset for further analysis
- `missing_values_report.csv` — missing-value counts before and after cleaning
- `cleaning_report.txt` — summary of row counts, duplicates, missing values, and cleaning actions

## Cleaning decisions
- Text whitespace is trimmed and blank strings are converted to missing values.
- Exact duplicate rows are removed.
- Duplicate `show_id` records are reduced to the first occurrence.
- `type` labels are standardized for common casing differences.
- `date_added` is parsed and saved in `YYYY-MM-DD` format.
- `release_year` is converted to a nullable integer.
- Missing descriptive text (`director`, `country`, `rating`, `duration`, `listed_in`) is filled with `Unknown`.
- Missing identifiers and date/year values are not guessed; they are left missing and documented.

## Important
Always keep `data/netflix_raw.csv` unchanged. The script reads it and writes results to `output/`. The source dataset supplied for this project contains 8,790 rows and 10 columns before cleaning.
