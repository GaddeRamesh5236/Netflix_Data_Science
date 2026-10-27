
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Auspify | Netflix Business Insights", page_icon="📊", layout="wide")
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "netflix_titles.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype("string").str.strip()
        df[c] = df[c].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    if "date_added" in df:
        df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
        df["added_year"] = df["date_added"].dt.year
        df["added_month"] = df["date_added"].dt.month_name()
    if "release_year" in df:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    if "duration" in df:
        extracted = df["duration"].astype("string").str.extract(r"(?P<duration_value>\d+)\s*(?P<duration_unit>min|Season|Seasons)", expand=True)
        df["duration_value"] = pd.to_numeric(extracted["duration_value"], errors="coerce")
        df["duration_unit"] = extracted["duration_unit"].str.lower().replace({"seasons":"season"})
    return df

@st.cache_data
def train_model(df):
    target = "type"
    features = [c for c in ["release_year", "rating", "country", "listed_in", "director"] if c in df.columns]
    model_df = df[features + [target]].dropna(subset=[target]).copy()
    # Keep high-cardinality text usable without allowing title/ID leakage.
    for col in features:
        model_df[col] = model_df[col].fillna("Unknown").astype(str)
    X, y = model_df[features], model_df[target].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    numeric = [c for c in features if c == "release_year"]
    categorical = [c for c in features if c not in numeric]
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2))]), categorical)
    ])
    model = Pipeline([("prep", prep), ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced"))])
    baseline = Pipeline([("prep", prep), ("classifier", DummyClassifier(strategy="most_frequent"))])
    model.fit(X_train, y_train)
    baseline.fit(X_train, y_train)
    pred = model.predict(X_test)
    base_pred = baseline.predict(X_test)
    metrics = {
        "model": {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision (weighted)": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall (weighted)": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1 (weighted)": f1_score(y_test, pred, average="weighted", zero_division=0),
        },
        "baseline": {"Accuracy": accuracy_score(y_test, base_pred)},
        "labels": sorted(y.unique().tolist()),
        "matrix": confusion_matrix(y_test, pred, labels=sorted(y.unique().tolist())).tolist(),
        "features": features,
        "test_size": len(y_test),
        "train_size": len(y_train),
        "target": y_test.tolist(),
        "predictions": pred.tolist(),
    }
    return model, metrics

df = load_data()
st.title("📊 Netflix Content — Business Insights Dashboard")
st.caption("Auspify Technologies | Data Science Internship · Task 6: Analytics + ML + Business Recommendations")
with st.sidebar:
    st.header("Dashboard filters")
    types = sorted(df["type"].dropna().astype(str).unique()) if "type" in df else []
    selected_types = st.multiselect("Content type", types, default=types)
    countries = sorted(df["country"].dropna().astype(str).unique()) if "country" in df else []
    selected_countries = st.multiselect("Country (first listed)", countries, default=[])
    years = df["release_year"].dropna()
    year_min = int(years.min()) if len(years) else 1900
    year_max = int(years.max()) if len(years) else 2025
    year_range = st.slider("Release year", year_min, year_max, (year_min, year_max))
    search = st.text_input("Search title", "")

filtered = df.copy()
if selected_types and "type" in filtered:
    filtered = filtered[filtered["type"].astype(str).isin(selected_types)]
if selected_countries and "country" in filtered:
    filtered = filtered[filtered["country"].astype(str).isin(selected_countries)]
if "release_year" in filtered:
    filtered = filtered[filtered["release_year"].between(year_range[0], year_range[1], inclusive="both")]
if search and "title" in filtered:
    filtered = filtered[filtered["title"].astype(str).str.contains(search, case=False, na=False)]

total = len(filtered)
movie_count = int((filtered["type"].astype(str).str.lower() == "movie").sum()) if "type" in filtered else 0
tv_count = int(filtered["type"].astype(str).str.lower().isin(["tv show", "tvshow"]).sum()) if "type" in filtered else 0
known_countries = int(filtered["country"].nunique()) if "country" in filtered else 0
c1,c2,c3,c4 = st.columns(4)
c1.metric("Titles in selection", f"{total:,}")
c2.metric("Movies", f"{movie_count:,}")
c3.metric("TV Shows", f"{tv_count:,}")
c4.metric("Countries represented", f"{known_countries:,}")

tab1, tab2, tab3, tab4 = st.tabs(["Overview & Trends", "Country & Genre", "ML Classifier", "Business Recommendations"])
with tab1:
    left,right = st.columns(2)
    if "type" in filtered:
        type_counts = filtered["type"].value_counts().rename_axis("Content type").reset_index(name="Titles")
        left.plotly_chart(px.pie(type_counts, names="Content type", values="Titles", hole=.45, title="Content mix"), use_container_width=True)
    if "release_year" in filtered:
        trend = filtered.dropna(subset=["release_year"]).groupby("release_year").size().reset_index(name="Titles")
        right.plotly_chart(px.line(trend, x="release_year", y="Titles", markers=True, title="Titles by release year"), use_container_width=True)
    if "rating" in filtered:
        ratings = filtered["rating"].fillna("Unknown").value_counts().head(12).rename_axis("Rating").reset_index(name="Titles")
        st.plotly_chart(px.bar(ratings, x="Rating", y="Titles", title="Most common content ratings"), use_container_width=True)
with tab2:
    left,right = st.columns(2)
    if "country" in filtered:
        country_counts = filtered["country"].dropna().astype(str).str.split(",").explode().str.strip()
        country_counts = country_counts[country_counts.ne("")].value_counts().head(15).rename_axis("Country").reset_index(name="Title mentions")
        left.plotly_chart(px.bar(country_counts.sort_values("Title mentions"), x="Title mentions", y="Country", orientation="h", title="Top country mentions"), use_container_width=True)
    if "listed_in" in filtered:
        genre_counts = filtered["listed_in"].dropna().astype(str).str.split(",").explode().str.strip()
        genre_counts = genre_counts[genre_counts.ne("")].value_counts().head(15).rename_axis("Category").reset_index(name="Title mentions")
        right.plotly_chart(px.bar(genre_counts.sort_values("Title mentions"), x="Title mentions", y="Category", orientation="h", title="Top categories"), use_container_width=True)
    st.caption("Country and category counts are mention counts: a title may list multiple countries or categories.")
with tab3:
    st.subheader("Predict whether a title is a Movie or TV Show")
    st.write("The model uses available metadata (release year, rating, country, categories, and director). It is a demonstration classifier, not a content-quality or commercial-success predictor.")
    with st.spinner("Training and evaluating the classifier..."):
        model, m = train_model(df)
    a,b,c,d = st.columns(4)
    a.metric("Model accuracy", f"{m['model']['Accuracy']:.1%}")
    b.metric("Weighted precision", f"{m['model']['Precision (weighted)']:.1%}")
    c.metric("Weighted recall", f"{m['model']['Recall (weighted)']:.1%}")
    d.metric("Weighted F1", f"{m['model']['F1 (weighted)']:.1%}")
    st.write(f"Baseline majority-class accuracy: **{m['baseline']['Accuracy']:.1%}** · Training rows: {m['train_size']:,} · Test rows: {m['test_size']:,}")
    st.write("**Input features:** " + ", ".join(m["features"]))
    cm = pd.DataFrame(m["matrix"], index=m["labels"], columns=m["labels"])
    st.plotly_chart(px.imshow(cm, text_auto=True, labels={"x":"Predicted","y":"Actual","color":"Count"}, title="Confusion matrix"), use_container_width=True)
    with st.expander("Try a single prediction"):
        c1,c2 = st.columns(2)
        release_year = c1.number_input("Release year", min_value=1900, max_value=2035, value=2020)
        rating_options = sorted(df["rating"].dropna().astype(str).unique()) if "rating" in df else ["Unknown"]
        rating = c2.selectbox("Rating", rating_options)
        country_options = sorted(df["country"].dropna().astype(str).unique()) if "country" in df else ["Unknown"]
        country = c1.selectbox("Country", country_options)
        genre_options = sorted(df["listed_in"].dropna().astype(str).unique()) if "listed_in" in df else ["Unknown"]
        genre = c2.selectbox("Category string", genre_options)
        director_options = sorted(df["director"].dropna().astype(str).unique()) if "director" in df else ["Unknown"]
        director = st.selectbox("Director", director_options)
        if st.button("Predict content type"):
            row = pd.DataFrame([{"release_year": release_year, "rating": rating, "country": country, "listed_in": genre, "director": director}])
            row = row[m["features"]]
            st.success(f"Predicted type: **{model.predict(row)[0]}**")
with tab4:
    st.subheader("Recommendations based on observed catalogue patterns")
    st.markdown("""
    1. **Balance format strategy:** monitor Movie vs TV Show share by country and category rather than using the overall mix alone.
    2. **Localize content decisions:** review top country mentions alongside genre demand before commissioning or licensing.
    3. **Use age-rating segmentation:** compare ratings within each market to support audience-specific merchandising.
    4. **Track recency carefully:** distinguish original release year from Netflix `date_added`; they answer different business questions.
    5. **Improve data governance:** standardize multi-country and multi-category fields before comparing market share.
    6. **Validate before action:** this public catalogue dataset does not contain viewing hours, subscriber conversion, costs, or revenue, so recommendations are descriptive—not causal ROI claims.
    """)
    st.info("These are data-informed hypotheses to investigate further, not proof of customer demand or profitability.")

st.subheader("Filtered data preview")
st.dataframe(filtered.head(100), use_container_width=True)
st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"), "netflix_filtered_selection.csv", "text/csv")
