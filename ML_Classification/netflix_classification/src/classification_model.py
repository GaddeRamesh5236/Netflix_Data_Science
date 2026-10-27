"""
Auspify Technologies — Data Science Internship Task 5
Machine Learning Classification Model using the Netflix dataset.

Task workflow:
1. Select and prepare features.
2. Split data into training and testing sets.
3. Train classification models.
4. Evaluate model performance.
5. Compare model accuracy.

Target: predict the Netflix title's content type (Movie vs TV Show).
This is a demonstration classification task using available metadata.
"""
from pathlib import Path
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.base import clone
import joblib

def run_classification(input_path: Path, output_dir: Path, random_state: int = 42):
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    required = {"type"}
    if not required.issubset(df.columns):
        raise ValueError("The dataset must include a 'type' column for the target label.")

    # Prepare target and predictors; type itself is excluded from features to avoid leakage.
    df["type"] = df["type"].astype("string").str.strip()
    df = df[df["type"].isin(["Movie", "TV Show"])].copy()
    y = df["type"].astype(str)

    candidate_numeric = [c for c in ["release_year"] if c in df.columns]
    candidate_categorical = [c for c in ["rating", "country"] if c in df.columns]
    candidate_text = [c for c in ["listed_in", "description", "director", "cast"] if c in df.columns]
    if not candidate_numeric and not candidate_categorical and not candidate_text:
        raise ValueError("No usable predictor columns were found.")

    X = df[candidate_numeric + candidate_categorical + candidate_text].copy()
    for c in candidate_numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in candidate_categorical + candidate_text:
        X[c] = X[c].fillna("Unknown").astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Numeric/categorical preprocessing and TF-IDF text features.
    transformers = []
    if candidate_numeric:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("numeric", numeric_pipe, candidate_numeric))
    if candidate_categorical:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("categorical", categorical_pipe, candidate_categorical))
    if candidate_text:
        # Merge available text fields into one text feature for TF-IDF.
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train["__combined_text"] = X_train[candidate_text].agg(" ".join, axis=1)
        X_test["__combined_text"] = X_test[candidate_text].agg(" ".join, axis=1)
        transformers.append(("text", TfidfVectorizer(max_features=12000, stop_words="english"), "__combined_text"))
        # ColumnTransformer can use a scalar column selector for a text transformer.
        X_train_model = X_train[candidate_numeric + candidate_categorical + ["__combined_text"]]
        X_test_model = X_test[candidate_numeric + candidate_categorical + ["__combined_text"]]
    else:
        X_train_model, X_test_model = X_train, X_test

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    models = {
        "Dummy baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=160, random_state=random_state, class_weight="balanced_subsample", n_jobs=-1)
    }

    results = []
    reports = []
    fitted_models = {}
    for name, classifier in models.items():
        pipeline = Pipeline([("preprocessor", clone(preprocessor)), ("classifier", classifier)])
        pipeline.fit(X_train_model, y_train)
        pred = pipeline.predict(X_test_model)
        accuracy = accuracy_score(y_test, pred)
        results.append({"model": name, "accuracy": round(float(accuracy), 4)})
        reports.append(f"MODEL: {name}\nAccuracy: {accuracy:.4f}\n\n{classification_report(y_test, pred, zero_division=0)}")
        fitted_models[name] = pipeline

        if name == "Logistic Regression":
            cm = confusion_matrix(y_test, pred, labels=["Movie", "TV Show"])
            plt.figure(figsize=(6, 5))
            plt.imshow(cm, interpolation="nearest")
            plt.title("Confusion Matrix — Logistic Regression")
            plt.xticks([0, 1], ["Movie", "TV Show"])
            plt.yticks([0, 1], ["Movie", "TV Show"])
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i, j]), ha="center", va="center")
            plt.tight_layout()
            plt.savefig(plots_dir / "confusion_matrix_logistic_regression.png", dpi=160)
            plt.close()

    result_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    result_df.to_csv(output_dir / "model_comparison.csv", index=False)
    plt.figure(figsize=(9, 5))
    plt.bar(result_df["model"], result_df["accuracy"])
    plt.title("Classification Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "model_accuracy_comparison.png", dpi=160)
    plt.close()

    # Save the best performing fitted pipeline and its expected input feature list.
    best_name = result_df.iloc[0]["model"]
    joblib.dump({"model_name": best_name, "pipeline": fitted_models[best_name],
                 "numeric_features": candidate_numeric, "categorical_features": candidate_categorical,
                 "text_features": candidate_text}, output_dir / "best_classifier.joblib")

    report_text = [
        "AUSPIFY TECHNOLOGIES — TASK 5: MACHINE LEARNING CLASSIFICATION",
        "=" * 64,
        f"Rows used: {len(df):,}",
        f"Target label: type (Movie vs TV Show)",
        f"Training rows: {len(X_train):,}",
        f"Testing rows: {len(X_test):,}",
        f"Numeric features: {candidate_numeric}",
        f"Categorical features: {candidate_categorical}",
        f"Text features: {candidate_text}",
        "",
        "Model comparison (accuracy):",
    ]
    report_text.extend([f"- {r['model']}: {r['accuracy']:.4f}" for r in results])
    report_text += [
        "",
        f"Best model by test accuracy: {best_name}",
        "",
        "Classification reports:",
        *reports,
        "",
        "Important limitation:",
        "- The target is content type (Movie vs TV Show). This is a practical demonstration using available metadata, not a business-approved production classifier.",
        "- Since the dataset is observational and has no external validation labels, results reflect only the supplied dataset and random train/test split."
    ]
    (output_dir / "classification_report.txt").write_text("\n".join(report_text), encoding="utf-8")
    print("\n".join(report_text[:20]))
    print(f"\nSaved model comparison: {output_dir / 'model_comparison.csv'}")
    print(f"Saved model artifact: {output_dir / 'best_classifier.joblib'}")
    return result_df

def main():
    parser = argparse.ArgumentParser(description="Netflix classification model for Auspify Task 5.")
    parser.add_argument("--input", default="data/netflix_raw.csv", help="Path to source CSV")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    run_classification(Path(args.input), Path(args.output), args.random_state)

if __name__ == "__main__":
    main()
