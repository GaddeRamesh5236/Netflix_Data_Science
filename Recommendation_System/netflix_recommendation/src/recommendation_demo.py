"""Interactive command-line demo for the content-based Netflix recommender."""
from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TEXT_FEATURES = ["title", "director", "cast", "listed_in", "description", "country"]

def norm(x):
    if pd.isna(x): return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", str(x).lower())).strip()

def main():
    path = Path("data/netflix_raw.csv")
    if not path.exists():
        raise FileNotFoundError("Run this demo from the project root so data/netflix_raw.csv can be found.")
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df = df[df["title"] != ""].drop_duplicates(subset=["show_id"]).reset_index(drop=True)
    content = pd.Series([""] * len(df))
    for col in TEXT_FEATURES:
        if col in df.columns:
            content = content + " " + df[col].fillna("").map(norm)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(content)
    titles = df["title"].tolist()
    print(f"Loaded {len(titles)} Netflix titles.")
    query = input("Enter a Netflix title (or part of a title): ").strip()
    matches = [i for i, t in enumerate(titles) if query.casefold() in t.casefold()]
    if not matches:
        print("No matching title found. Try another title.")
        return
    idx = matches[0]
    scores = cosine_similarity(matrix[idx], matrix).ravel()
    scores[idx] = -1
    best = scores.argsort()[::-1][:10]
    print(f"\nBecause you selected: {titles[idx]}\n")
    for rank, j in enumerate(best, start=1):
        category = df.iloc[j].get("listed_in", "")
        year = df.iloc[j].get("release_year", "")
        print(f"{rank:>2}. {titles[j]} | similarity={scores[j]:.3f} | year={year} | categories={category}")
if __name__ == "__main__":
    main()
