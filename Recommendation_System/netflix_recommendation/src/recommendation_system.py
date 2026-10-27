"""
Auspify Technologies — Data Science Internship Task 3
Recommendation System Analysis for Netflix titles.

Workflow:
1. Extract relevant content features.
2. Perform text preprocessing.
3. Calculate content similarity.
4. Generate recommendations for selected titles.
5. Evaluate recommendation quality using a simple coverage/diversity check.

Run from project root:
    python src/recommendation_system.py
"""
from pathlib import Path
import argparse
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TEXT_FEATURES = ["title", "director", "cast", "listed_in", "description", "country"]

def normalize_text(value):
    if pd.isna(value):
        return ""
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value

def build_content_text(df):
    # Weight title/category/description features by including them in a single content profile.
    parts = []
    for col in TEXT_FEATURES:
        if col in df.columns:
            parts.append(df[col].fillna("").map(normalize_text))
    if not parts:
        return pd.Series([""] * len(df), index=df.index)
    combined = parts[0]
    for p in parts[1:]:
        combined = combined + " " + p
    return combined.str.strip()

def create_recommendations(input_path: Path, output_dir: Path, top_n: int = 10):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    required = {"show_id", "title"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df = df[df["title"] != ""].drop_duplicates(subset=["show_id"]).reset_index(drop=True)
    content_text = build_content_text(df)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000, ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(content_text)
    similarity = cosine_similarity(tfidf, dense_output=False)

    # Store recommendations for sample titles drawn from the dataset.
    sample_titles = df["title"].drop_duplicates().head(5).tolist()
    recommendation_rows = []
    for title in sample_titles:
        idxs = df.index[df["title"].str.casefold() == title.casefold()].tolist()
        if not idxs:
            continue
        idx = idxs[0]
        row = similarity[idx].toarray().ravel()
        row[idx] = -1
        best_idxs = row.argsort()[::-1][:top_n]
        for rank, rec_idx in enumerate(best_idxs, start=1):
            recommendation_rows.append({
                "input_title": title,
                "rank": rank,
                "recommended_title": df.iloc[rec_idx]["title"],
                "similarity_score": round(float(row[rec_idx]), 4),
                "recommended_type": df.iloc[rec_idx].get("type", ""),
                "recommended_release_year": df.iloc[rec_idx].get("release_year", ""),
                "recommended_categories": df.iloc[rec_idx].get("listed_in", ""),
            })
    recommendations = pd.DataFrame(recommendation_rows)
    recommendations.to_csv(output_dir / "sample_recommendations.csv", index=False)

    # Save a lightweight title index for quick lookup by the CLI demo.
    df[["show_id", "title"] + [c for c in ["type", "release_year", "listed_in", "description"] if c in df.columns]].to_csv(
        output_dir / "title_catalog.csv", index=False
    )

    # Evaluation summary: dataset coverage and category diversity among generated recommendations.
    if not recommendations.empty:
        unique_recommended = recommendations["recommended_title"].nunique()
        total_titles = df["title"].nunique()
        coverage = unique_recommended / total_titles * 100 if total_titles else 0
        diversity = recommendations.groupby("input_title")["recommended_title"].nunique().mean()
        mean_similarity = recommendations["similarity_score"].mean()
    else:
        unique_recommended, total_titles, coverage, diversity, mean_similarity = 0, df["title"].nunique(), 0, 0, 0

    report = [
        "AUSPIFY TECHNOLOGIES — TASK 3: RECOMMENDATION SYSTEM",
        "=" * 52,
        f"Titles processed: {len(df):,}",
        f"Unique titles processed: {df['title'].nunique():,}",
        f"Content features used: {', '.join([c for c in TEXT_FEATURES if c in df.columns])}",
        f"TF-IDF feature count: {len(vectorizer.vocabulary_):,}",
        f"Sample input titles evaluated: {len(sample_titles)}",
        f"Recommendations generated: {len(recommendations):,}",
        f"Unique recommended titles in sample output: {unique_recommended:,}",
        f"Unique-title recommendation coverage in sample output: {coverage:.2f}%",
        f"Average unique recommendations per sample title: {diversity:.2f}",
        f"Mean cosine similarity score in sample output: {mean_similarity:.4f}",
        "",
        "Method:",
        "- Text was lowercased, punctuation removed, and whitespace normalized.",
        "- Content fields were combined into a single text profile.",
        "- TF-IDF transformed the text into numeric feature vectors.",
        "- Cosine similarity ranked titles by content-profile similarity.",
        "",
        "Evaluation caveat:",
        "- This is a content-based prototype. No user ratings, click history, or relevance labels are present, so offline precision/recall cannot be measured reliably.",
        "- Coverage and mean similarity summarize only the generated sample recommendations; they are not proof of user satisfaction."
    ]
    (output_dir / "recommendation_evaluation.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"\nSaved recommendations: {output_dir / 'sample_recommendations.csv'}")
    return df, tfidf

def main():
    parser = argparse.ArgumentParser(description="Netflix content-based recommendation system for Auspify Task 3.")
    parser.add_argument("--input", default="data/netflix_raw.csv", help="Path to source CSV")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--top-n", type=int, default=10, help="Recommendations per sample title (default: 10)")
    args = parser.parse_args()
    create_recommendations(Path(args.input), Path(args.output), args.top_n)

if __name__ == "__main__":
    main()
