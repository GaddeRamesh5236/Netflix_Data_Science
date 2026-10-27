# Auspify Technologies — Task 3: Recommendation System Analysis

## Objective
Build a basic content-based recommendation model using Netflix titles and categories, following the internship task guide.

## Task workflow
1. Extract relevant content features.
2. Perform text preprocessing.
3. Calculate content similarity.
4. Generate recommendations for selected titles.
5. Evaluate recommendation quality.

## Project structure
```text
auspify_task3_netflix_recommendation/
├── data/
│   └── netflix_raw.csv
├── src/
│   ├── recommendation_system.py
│   └── recommendation_demo.py
├── output/
│   ├── sample_recommendations.csv
│   ├── title_catalog.csv
│   └── recommendation_evaluation.txt
├── requirements.txt
└── README.md
```

## Run in Windows / VS Code
Open this folder in VS Code, then run in the terminal:
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python src/recommendation_system.py
```

## Try a title interactively
After installing requirements:
```powershell
python src/recommendation_demo.py
```
Enter a title or part of a title from the Netflix dataset. The program returns up to 10 similar titles with similarity scores, year, and categories.

## How it works
- Uses title, director, cast, categories (`listed_in`), description, and country where available.
- Normalizes text to lowercase, removes punctuation, and collapses whitespace.
- Converts combined content text to TF-IDF vectors.
- Uses cosine similarity to rank related titles.
- Writes example recommendations and an evaluation summary into `output/`.

## Evaluation limitation
The dataset does not include user-level ratings, clicks, or relevance labels. Therefore, precision/recall or personalized user satisfaction cannot be measured from this file alone. The evaluation report instead provides sample coverage, unique recommendation counts, and mean similarity score.
