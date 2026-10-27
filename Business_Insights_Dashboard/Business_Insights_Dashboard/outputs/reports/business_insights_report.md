# Task 6 — Netflix Business Insights Report

## Objective
Combine catalogue analysis, an interpretable baseline comparison, visualizations, and business recommendations.

## Dataset overview
- Rows: **8,790**
- Columns: **10**
- Exact duplicate rows: **0**
- Total missing cells: **0**
- Content types: **Movie: 6,126, TV Show: 2,664**
- Most represented release year in the data: **2018** (1,146 titles).
- Highest country mention: **United States** (3,240 mentions).
- Highest category mention: **International Movies** (2,752 mentions).

## Business recommendations
1. Compare content mix within each market and category instead of applying one global catalogue strategy.
2. Track country and category mentions as multi-label data; one title can contribute to multiple counts.
3. Review content ratings by market to support audience-specific merchandising and catalogue curation.
4. Separate release year from date added: the former describes original release timing; the latter describes catalogue acquisition timing.
5. Improve data quality with consistent country/category normalization and clear handling of unknown values.
6. For investment decisions, combine this catalogue with viewing hours, completion rates, subscriber retention, licensing costs, and revenue.

## Limitations
- This dataset describes catalogue metadata, not customer behavior or business performance.
- Country/category totals are mention counts because values may contain multiple comma-separated labels.
- A content-type classifier is a demonstration of machine-learning integration. It does not predict popularity, profitability, or demand.
- Associations in the dataset should not be interpreted as causal effects.

## Output files
- `dataset_summary.csv`, `missing_values.csv`, `content_type_counts.csv`
- `release_year_counts.csv`, `top_country_mentions.csv`, `top_category_mentions.csv`
- Charts in `outputs/charts/`