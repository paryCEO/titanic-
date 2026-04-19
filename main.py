"""
Titanic Survival Prediction Project
Dataset expected: train (2).csv

What this script does:
1. Loads the dataset
2. Cleans missing values
3. Treats all features as numeric
4. Splits data into train/validation sets
5. Trains 3 machine learning models
6. Compares them using Accuracy, Precision, Recall, F1-score, and Training Time
7. Prints confusion matrix and classification report
8. Saves the best model
9. Tests one sample prediction

Run:
    python main.py
"""

import time
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# ---------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------
DATA_PATH = Path("train (2).csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found. Put 'train (2).csv' in the same folder as this script."
    )

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("Dataset loaded successfully")
print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("=" * 70)


# ---------------------------------------------------
# 2. Basic checks
# ---------------------------------------------------
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isna().sum())

TARGET = "Survived"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' was not found in the dataset.")

DROP_COLUMNS = ["PassengerId"]

X = df.drop(columns=[TARGET] + [c for c in DROP_COLUMNS if c in df.columns])
y = df[TARGET]

print("\nTarget distribution:")
print(y.value_counts())


# ---------------------------------------------------
# 3. Treat all features as numeric
# ---------------------------------------------------
numeric_features = X.columns.tolist()

print("\nNumeric features:", numeric_features)


# ---------------------------------------------------
# 4. Preprocessing
# ---------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        )
    ],
    remainder="drop"
)


# ---------------------------------------------------
# 5. Split data
# ---------------------------------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain size:", X_train.shape)
print("Validation size:", X_valid.shape)


# ---------------------------------------------------
# 6. Build models
# ---------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}


# ---------------------------------------------------
# 7. Train and evaluate
# ---------------------------------------------------
results = []
trained_pipelines = {}

for model_name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_pred = pipeline.predict(X_valid)

    accuracy = accuracy_score(y_valid, y_pred)
    precision = precision_score(y_valid, y_pred, zero_division=0)
    recall = recall_score(y_valid, y_pred, zero_division=0)
    f1 = f1_score(y_valid, y_pred, zero_division=0)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Training Time (s)": training_time
    })

    trained_pipelines[model_name] = pipeline

    print("\n" + "-" * 70)
    print(f"Model: {model_name}")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision      : {precision:.4f}")
    print(f"Recall         : {recall:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print(f"Training Time  : {training_time:.4f} seconds")
    print("Confusion Matrix:")
    print(confusion_matrix(y_valid, y_pred))
    print("Classification Report:")
    print(classification_report(y_valid, y_pred, zero_division=0))


# ---------------------------------------------------
# 8. Comparison table
# ---------------------------------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

print("\n" + "=" * 70)
print("MODEL COMPARISON TABLE")
print(results_df.to_string(index=False))
print("=" * 70)


# ---------------------------------------------------
# 9. Select and save best model
# ---------------------------------------------------
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_pipelines[best_model_name]

print(f"\nBest model: {best_model_name}")

MODEL_PATH = Path("best_titanic_model.pkl")
joblib.dump(best_model, MODEL_PATH)

print(f"Best model saved to: {MODEL_PATH.resolve()}")


# ---------------------------------------------------
# 10. Predict on a sample passenger
# ---------------------------------------------------
sample = pd.DataFrame([
    {
        "Pclass": 3,
        "Sex": 1,
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "FamilySize": 1,
        "Embarked_C": 0,
        "Embarked_Q": 0,
        "Embarked_S": 1
    }
])

sample = sample.reindex(columns=X.columns, fill_value=0)

sample_prediction = best_model.predict(sample)[0]

if hasattr(best_model.named_steps["model"], "predict_proba"):
    sample_probability = best_model.predict_proba(sample)[0][1]
else:
    sample_probability = None

print("\n" + "=" * 70)
print("SAMPLE PASSENGER PREDICTION")
print("Predicted Survived =", int(sample_prediction))

if sample_probability is not None:
    print(f"Survival probability = {sample_probability:.4f}")
print("=" * 70)