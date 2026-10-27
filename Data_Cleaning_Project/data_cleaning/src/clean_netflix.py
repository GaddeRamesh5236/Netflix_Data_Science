
from pathlib import Path
import argparse
import pandas as pd

EXPECTED_COLUMNS = [
    "show_id", "type", "title", "director", "country", "date_added",
    "release_year", "rating", "duration", "listed_in"
]

def clean_dataset(input_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    original_rows, original_cols = df.shape

    # Normalize column names and text whitespace.
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].map(lambda value: value.strip() if isinstance(value, str) else value)
        df[col] = df[col].replace(r"^\s*$", pd.NA, regex=True)

    missing_before = df.isna().sum().rename("missing_before")
    duplicate_rows_before = int(df.duplicated().sum())
    duplicate_ids_before = int(df["show_id"].duplicated().sum()) if "show_id" in df.columns else 0

    # Remove exact duplicate rows, then duplicate show IDs (keep first occurrence).
    df = df.drop_duplicates()
    if "show_id" in df.columns:
        df = df.drop_duplicates(subset=["show_id"], keep="first")

    # Normalize selected categorical columns.
    for col in ["type", "rating"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    if "type" in df.columns:
        df["type"] = df["type"].replace({
            "movie": "Movie", "MOVIE": "Movie",
            "tv show": "TV Show", "TV SHOW": "TV Show"
        })

    # Parse dates consistently. Invalid or missing values remain NaT and are reported.
    if "date_added" in df.columns:
        df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
        # Store dates in a consistent YYYY-MM-DD format; missing dates remain blank in CSV.
        df["date_added"] = df["date_added"].dt.strftime("%Y-%m-%d")

    # Ensure release year is numeric; invalid values become missing.
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").astype("Int64")

    # Fill missing text values in descriptive columns with an explicit label.
    # Keep identifiers and date/year fields missing if they cannot be reliably inferred.
    for col in ["director", "country", "rating", "duration", "listed_in"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    missing_after = df.isna().sum().rename("missing_after")
    missing_report = pd.concat([missing_before, missing_after], axis=1).fillna(0).astype(int)
    missing_report.to_csv(output_dir / "missing_values_report.csv", index_label="column")

    df.to_csv(output_dir / "auspify_task1_data_cleaning/data/netflix_raw.csv", index=False)

    report = [
        "AUSPIFY TECHNOLOGIES — TASK 1 CLEANING REPORT",
        "=" * 48,
        f"Input file: {input_path}",
        f"Original rows: {original_rows}",
        f"Original columns: {original_cols}",
        f"Exact duplicate rows found before removal: {duplicate_rows_before}",
        f"Duplicate show_id values found before removal: {duplicate_ids_before}",
        f"Rows after duplicate removal: {len(df)}",
        f"Columns after cleaning: {len(df.columns)}",
        "",
        "Missing values by column (before -> after):",
    ]
    for col in missing_report.index:
        report.append(f"- {col}: {missing_report.loc[col, 'missing_before']} -> {missing_report.loc[col, 'missing_after']}")
    report += [
        "",
        "Cleaning actions:",
        "- Trimmed leading/trailing whitespace in text fields.",
        "- Converted blank strings to missing values.",
        "- Removed exact duplicate rows and duplicate show_id records (kept first).",
        "- Standardized type labels where needed.",
        "- Parsed date_added and formatted valid dates as YYYY-MM-DD.",
        "- Converted release_year to a nullable numeric integer.",
        "- Replaced missing director, country, rating, duration, and listed_in with 'Unknown'.",
        "- Saved cleaned dataset and missing-value report.",
        "",
        "Note: Missing show_id, date_added, or release_year values are not guessed; they remain missing and are documented."
    ]
    (output_dir / "cleaning_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report[:10]))
    print(f"\nSaved: {output_dir / 'netflix_cleaned.csv'}")
    print(f"Saved: {output_dir / 'missing_values_report.csv'}")
    print(f"Saved: {output_dir / 'cleaning_report.txt'}")
    return df

def main():
    parser = argparse.ArgumentParser(description="Clean the Netflix dataset for Auspify Task 1.")
    parser.add_argument("--input", default="data/netflix_raw.csv", help="Path to the raw CSV dataset")
    parser.add_argument("--output", default="output", help="Directory for cleaned outputs")
    args = parser.parse_args()
    clean_dataset(Path(args.input), Path(args.output))
 
if __name__ == "__main__":
    main()
