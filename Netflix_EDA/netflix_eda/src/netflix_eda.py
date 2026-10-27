"""
Auspify Technologies — Data Science Internship Task 2
Exploratory Data Analysis (EDA) on the Netflix dataset.

Workflow:
1. Analyze dataset structure and statistics.
2. Study content distribution by type.
3. Identify top countries and categories.
4. Create visualizations for key metrics.
5. Summarize findings.

Run from project root:
    python src/netflix_eda.py
"""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def split_multi_value(series):
    """Split comma-separated entries and return one value per row."""
    return (series.dropna().astype(str).str.split(", ").explode().str.strip()
            .replace("", pd.NA).dropna())

def run_eda(input_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    rows, cols = df.shape

    # Step 1: Structure and statistics
    structure_lines = [
        "AUSPIFY TECHNOLOGIES — TASK 2: NETFLIX EDA",
        "=" * 48,
        f"Dataset rows: {rows}",
        f"Dataset columns: {cols}",
        "",
        "Column names and data types:",
        df.dtypes.astype(str).to_string(),
        "",
        "Missing values by column:",
        df.isna().sum().to_string(),
        "",
        "Duplicate rows:",
        str(int(df.duplicated().sum())),
        "",
        "Numeric summary:",
        df.describe(include="number").to_string(),
        "",
        "Categorical summary:",
        df.describe(include="object").to_string(),
    ]
    (output_dir / "dataset_profile.txt").write_text("\n".join(structure_lines), encoding="utf-8")
    df.describe(include="all").transpose().to_csv(output_dir / "descriptive_statistics.csv")

    # Step 2: Content distribution by type
    type_counts = df["type"].fillna("Unknown").value_counts().rename_axis("type").reset_index(name="count") if "type" in df.columns else pd.DataFrame(columns=["type","count"])
    type_counts.to_csv(output_dir / "content_type_distribution.csv", index=False)

    plt.figure(figsize=(8, 5))
    if not type_counts.empty:
        plt.bar(type_counts["type"].astype(str), type_counts["count"])
        plt.title("Netflix Titles by Content Type")
        plt.xlabel("Content Type")
        plt.ylabel("Number of Titles")
        for i, val in enumerate(type_counts["count"]):
            plt.text(i, val, f"{val:,}", ha="center", va="bottom")
    else:
        plt.text(0.5, 0.5, "Column 'type' not found", ha="center")
    plt.tight_layout()
    plt.savefig(plots_dir / "content_type_distribution.png", dpi=160)
    plt.close()

    # Step 3: Top countries and categories; explode multi-country/category fields
    if "country" in df.columns:
        countries = split_multi_value(df["country"])
        country_counts = countries.value_counts().head(15).rename_axis("country").reset_index(name="title_mentions")
    else:
        country_counts = pd.DataFrame(columns=["country", "title_mentions"])
    country_counts.to_csv(output_dir / "top_countries.csv", index=False)

    category_col = "listed_in" if "listed_in" in df.columns else None
    if category_col:
        categories = split_multi_value(df[category_col])
        category_counts = categories.value_counts().head(15).rename_axis("category").reset_index(name="title_mentions")
    else:
        category_counts = pd.DataFrame(columns=["category", "title_mentions"])
    category_counts.to_csv(output_dir / "top_categories.csv", index=False)

    def horizontal_bar(data, label_col, value_col, title, filename):
        plt.figure(figsize=(10, 6))
        if not data.empty:
            plot_data = data.sort_values(value_col, ascending=True)
            plt.barh(plot_data[label_col].astype(str), plot_data[value_col])
            plt.title(title)
            plt.xlabel("Number of title mentions")
            plt.ylabel("")
        else:
            plt.text(0.5, 0.5, "Required column not found", ha="center")
        plt.tight_layout()
        plt.savefig(plots_dir / filename, dpi=160)
        plt.close()

    horizontal_bar(country_counts, "country", "title_mentions", "Top 15 Countries by Netflix Title Mentions", "top_countries.png")
    horizontal_bar(category_counts, "category", "title_mentions", "Top 15 Categories by Netflix Title Mentions", "top_categories.png")

    # Step 4: Additional key metrics and visualizations
    if "release_year" in df.columns:
        release_year = pd.to_numeric(df["release_year"], errors="coerce")
        year_counts = release_year.dropna().astype(int).value_counts().sort_index().rename_axis("release_year").reset_index(name="titles")
        year_counts.to_csv(output_dir / "titles_by_release_year.csv", index=False)
        plt.figure(figsize=(11, 5))
        plt.plot(year_counts["release_year"], year_counts["titles"], marker="o")
        plt.title("Netflix Titles by Release Year")
        plt.xlabel("Release Year")
        plt.ylabel("Number of Titles")
        plt.tight_layout()
        plt.savefig(plots_dir / "titles_by_release_year.png", dpi=160)
        plt.close()

    if "rating" in df.columns:
        rating_counts = df["rating"].fillna("Unknown").value_counts().head(12).rename_axis("rating").reset_index(name="count")
        rating_counts.to_csv(output_dir / "rating_distribution.csv", index=False)
        plt.figure(figsize=(10, 6))
        plt.barh(rating_counts["rating"].astype(str), rating_counts["count"])
        plt.title("Most Common Content Ratings")
        plt.xlabel("Number of Titles")
        plt.ylabel("Rating")
        plt.tight_layout()
        plt.savefig(plots_dir / "rating_distribution.png", dpi=160)
        plt.close()

    # Step 5: Summary based on actual data
    summary = [
        "AUSPIFY TECHNOLOGIES — TASK 2: EDA FINDINGS",
        "=" * 44,
        f"Dataset size: {rows:,} rows × {cols} columns.",
        f"Exact duplicate rows: {int(df.duplicated().sum()):,}.",
    ]
    if not type_counts.empty:
        total_types = int(type_counts["count"].sum())
        summary.append("Content type distribution:")
        for _, row in type_counts.iterrows():
            pct = row["count"] / total_types * 100 if total_types else 0
            summary.append(f"- {row['type']}: {int(row['count']):,} titles ({pct:.1f}%).")
    if not country_counts.empty:
        summary.append(f"Most frequently listed country: {country_counts.iloc[0]['country']} ({int(country_counts.iloc[0]['title_mentions']):,} title mentions).")
        summary.append("Top 5 countries:")
        for _, row in country_counts.head(5).iterrows():
            summary.append(f"- {row['country']}: {int(row['title_mentions']):,}")
    if not category_counts.empty:
        summary.append(f"Most frequent category: {category_counts.iloc[0]['category']} ({int(category_counts.iloc[0]['title_mentions']):,} title mentions).")
        summary.append("Top 5 categories:")
        for _, row in category_counts.head(5).iterrows():
            summary.append(f"- {row['category']}: {int(row['title_mentions']):,}")
    if "release_year" in df.columns:
        years = pd.to_numeric(df["release_year"], errors="coerce").dropna()
        if not years.empty:
            summary.append(f"Release year range: {int(years.min())}–{int(years.max())}.")
            summary.append(f"Most common release year: {int(years.mode().iloc[0])}.")
    summary += [
        "",
        "Interpretation notes:",
        "- Country and category fields may contain multiple comma-separated values; counts above split these into individual mentions.",
        "- Release year describes the year a title was released, not necessarily the date it was added to Netflix.",
        "- These are descriptive findings from the supplied dataset, not evidence of Netflix's current catalog or future performance."
    ]
    (output_dir / "eda_findings.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))
    print(f"\nEDA outputs saved in: {output_dir.resolve()}")
    return df

def main():
    parser = argparse.ArgumentParser(description="Run Netflix EDA for Auspify Task 2.")
    parser.add_argument("--input", default="data/netflix_raw.csv", help="Path to source CSV")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()
    run_eda(Path(args.input), Path(args.output))

if __name__ == "__main__":
    main()
