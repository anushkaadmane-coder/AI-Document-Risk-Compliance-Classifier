import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "documents.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "risk_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)

CONFUSION_MATRIX_PATH = os.path.join(
    MODEL_DIR,
    "risk_confusion_matrix.png"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AI DOCUMENT RISK & COMPLIANCE CLASSIFIER")
print("MODEL TRAINING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH
)

print(
    f"Dataset loaded successfully."
)

print(
    f"Total documents: {len(df)}"
)


# ============================================================
# VALIDATE DATA
# ============================================================

required_columns = [
    "text",
    "risk_level"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column '{column}' not found."
        )


df = df.dropna(
    subset=[
        "text",
        "risk_level"
    ]
)

df["text"] = df["text"].astype(str)


print("\nRisk distribution:")

print(
    df["risk_level"].value_counts()
)


# ============================================================
# INPUT AND TARGET
# ============================================================

X = df["text"]

y = df["risk_level"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nData split:")
print(
    f"Training documents: {len(X_train)}"
)

print(
    f"Testing documents:  {len(X_test)}"
)


# ============================================================
# TF-IDF
# ============================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)

print(
    f"TF-IDF training shape: {X_train_tfidf.shape}"
)

print(
    f"TF-IDF testing shape: {X_test_tfidf.shape}"
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

model.fit(
    X_train_tfidf,
    y_train
)

print(
    "Model training completed."
)


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(
    X_test_tfidf
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"\nAccuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClass-wise performance:")
print("-" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "\nCreating confusion matrix..."
)

labels = [
    "Low",
    "Medium",
    "High",
    "Critical"
]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Document Risk Classification - Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    range(len(labels)),
    labels
)

plt.yticks(
    range(len(labels)),
    labels
)

plt.xlabel(
    "Predicted Risk"
)

plt.ylabel(
    "Actual Risk"
)

for i in range(len(labels)):

    for j in range(len(labels)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=150
)

plt.close()

print(
    f"Confusion matrix saved to:\n"
    f"{CONFUSION_MATRIX_PATH}"
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving trained model...")

with open(
    MODEL_PATH,
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


# ============================================================
# SAVE VECTORIZER
# ============================================================

print("Saving TF-IDF vectorizer...")

with open(
    VECTORIZER_PATH,
    "wb"
) as file:

    pickle.dump(
        vectorizer,
        file
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFiles created:")

print(
    f"\nModel:\n{MODEL_PATH}"
)

print(
    f"\nVectorizer:\n{VECTORIZER_PATH}"
)

print(
    f"\nConfusion Matrix:\n"
    f"{CONFUSION_MATRIX_PATH}"
)

print("\n" + "=" * 70)