
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "netflix_titles.csv"
OUT = BASE / "outputs"
CHARTS = OUT / "charts"
REPORTS = OUT / "reports"
CHARTS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df.columns = [c.strip() for c in df.columns]
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
if "date_added" in df:
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
if "release_year" in df:
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")

df.describe(include="all").T.to_csv(OUT / "dataset_summary.csv")
df.isna().sum().rename("missing_count").to_csv(OUT / "missing_values.csv")
df["type"].value_counts(dropna=False).rename_axis("type").reset_index(name="count").to_csv(OUT / "content_type_counts.csv")

# Use matplotlib defaults and avoid treating multi-valued fields as exclusive categories.
fig, ax = plt.subplots(figsize=(8, 5))
df["type"].value_counts().plot(kind="bar", ax=ax)
ax.set_title("Netflix Titles by Content Type")
ax.set_xlabel("Content type"); ax.set_ylabel("Titles"); fig.tight_layout()
fig.savefig(CHARTS / "content_type_distribution.png", dpi=150); plt.close(fig)

year_counts = df.dropna(subset=["release_year"]).groupby("release_year").size().rename("titles").reset_index()
year_counts.to_csv(OUT / "release_year_counts.csv", index=False)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(year_counts["release_year"], year_counts["titles"])
ax.set_title("Netflix Catalogue Titles by Release Year")
ax.set_xlabel("Release year"); ax.set_ylabel("Titles"); fig.tight_layout()
fig.savefig(CHARTS / "release_year_trend.png", dpi=150); plt.close(fig)

if "country" in df:
    countries = df["country"].dropna().astype(str).str.split(",").explode().str.strip()
    countries = countries[countries.ne("")].value_counts().head(15).rename_axis("country").reset_index(name="mentions")
    countries.to_csv(OUT / "top_country_mentions.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    countries.sort_values("mentions").plot(kind="barh", x="country", y="mentions", legend=False, ax=ax)
    ax.set_title("Top Country Mentions"); ax.set_xlabel("Mentions"); ax.set_ylabel("Country"); fig.tight_layout()
    fig.savefig(CHARTS / "top_countries.png", dpi=150); plt.close(fig)

if "listed_in" in df:
    genres = df["listed_in"].dropna().astype(str).str.split(",").explode().str.strip()
    genres = genres[genres.ne("")].value_counts().head(15).rename_axis("category").reset_index(name="mentions")
    genres.to_csv(OUT / "top_category_mentions.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    genres.sort_values("mentions").plot(kind="barh", x="category", y="mentions", legend=False, ax=ax)
    ax.set_title("Top Category Mentions"); ax.set_xlabel("Mentions"); ax.set_ylabel("Category"); fig.tight_layout()
    fig.savefig(CHARTS / "top_categories.png", dpi=150); plt.close(fig)

type_counts = df["type"].value_counts()
peak = year_counts.loc[year_counts["titles"].idxmax()] if not year_counts.empty else None
top_country = countries.iloc[0] if "country" in df and not countries.empty else None
top_genre = genres.iloc[0] if "listed_in" in df and not genres.empty else None
lines = [
    "# Task 6 — Netflix Business Insights Report",
    "",
    "## Objective",
    "Combine catalogue analysis, an interpretable baseline comparison, visualizations, and business recommendations.",
    "",
    "## Dataset overview",
    f"- Rows: **{len(df):,}**",
    f"- Columns: **{df.shape[1]}**",
    f"- Exact duplicate rows: **{int(df.duplicated().sum()):,}**",
    f"- Total missing cells: **{int(df.isna().sum().sum()):,}**",
    f"- Content types: **{', '.join(f'{k}: {v:,}' for k,v in type_counts.items())}**",
]
if peak is not None:
    lines += [f"- Most represented release year in the data: **{int(peak['release_year'])}** ({int(peak['titles']):,} titles)."]
if top_country is not None:
    lines += [f"- Highest country mention: **{top_country['country']}** ({int(top_country['mentions']):,} mentions)."]
if top_genre is not None:
    lines += [f"- Highest category mention: **{top_genre['category']}** ({int(top_genre['mentions']):,} mentions)."]
lines += [
    "",
    "## Business recommendations",
    "1. Compare content mix within each market and category instead of applying one global catalogue strategy.",
    "2. Track country and category mentions as multi-label data; one title can contribute to multiple counts.",
    "3. Review content ratings by market to support audience-specific merchandising and catalogue curation.",
    "4. Separate release year from date added: the former describes original release timing; the latter describes catalogue acquisition timing.",
    "5. Improve data quality with consistent country/category normalization and clear handling of unknown values.",
    "6. For investment decisions, combine this catalogue with viewing hours, completion rates, subscriber retention, licensing costs, and revenue.",
    "",
    "## Limitations",
    "- This dataset describes catalogue metadata, not customer behavior or business performance.",
    "- Country/category totals are mention counts because values may contain multiple comma-separated labels.",
    "- A content-type classifier is a demonstration of machine-learning integration. It does not predict popularity, profitability, or demand.",
    "- Associations in the dataset should not be interpreted as causal effects.",
    "",
    "## Output files",
    "- `dataset_summary.csv`, `missing_values.csv`, `content_type_counts.csv`",
    "- `release_year_counts.csv`, `top_country_mentions.csv`, `top_category_mentions.csv`",
    "- Charts in `outputs/charts/`",
]
(REPORTS / "business_insights_report.md").write_text("\n".join(lines), encoding="utf-8")
print("Analysis complete. Outputs written to:", OUT)
