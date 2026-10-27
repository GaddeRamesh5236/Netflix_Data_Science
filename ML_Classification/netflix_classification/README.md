# Auspify Technologies — Task 5: Machine Learning Classification Model

## Objective
Build a machine learning model to classify Netflix content using available metadata, following the Task 5 workflow in the internship guide.

## Task workflow
1. Select and prepare features.
2. Split data into training and testing sets.
3. Train classification models.
4. Evaluate model performance.
5. Compare model accuracy.

## Target selection
This project predicts `type` (Movie vs TV Show) using other available metadata. The target column is excluded from the input features to prevent direct target leakage.

## Project structure
```text
auspify_task5_netflix_classification/
├── data/
│   └── netflix_raw.csv
├── src/
│   └── classification_model.py
├── output/
│   ├── model_comparison.csv
│   ├── classification_report.txt
│   ├── best_classifier.joblib
│   └── plots/
│       ├── model_accuracy_comparison.png
│       └── confusion_matrix_logistic_regression.png
├── requirements.txt
└── README.md
```

## Run in Windows / VS Code
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python src/classification_model.py
```

## Methods
- Numeric preprocessing: median imputation and standardization.
- Categorical preprocessing: most-frequent imputation and one-hot encoding.
- Text preprocessing: TF-IDF vectorization.
- Train/test split: 80% training, 20% testing with stratification.
- Models compared: Dummy baseline, Logistic Regression, Random Forest.
- Metrics: accuracy, precision, recall, F1-score, confusion matrix.

## Outputs
- `model_comparison.csv`: model accuracy comparison.
- `classification_report.txt`: dataset sizes, features, metrics, and classification reports.
- `best_classifier.joblib`: saved best-performing pipeline and feature metadata.
- `plots/`: model accuracy comparison and confusion matrix.

## Limitation
This is a training demonstration on the supplied dataset. It predicts content type using metadata and should not be interpreted as a production-grade model or as evidence of future catalog behavior.
