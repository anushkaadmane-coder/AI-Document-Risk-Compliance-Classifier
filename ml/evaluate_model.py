import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
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

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "risk_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "tfidf_vectorizer.pkl"
)

EVALUATION_REPORT_PATH = os.path.join(
    BASE_DIR,
    "models",
    "evaluation_report.txt"
)

EVALUATION_MATRIX_PATH = os.path.join(
    BASE_DIR,
    "models",
    "evaluation_confusion_matrix.png"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("AI DOCUMENT RISK & COMPLIANCE CLASSIFIER")
print("MODEL EVALUATION")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking required files...")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found:\n{MODEL_PATH}"
    )

if not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(
        f"TF-IDF vectorizer not found:\n{VECTORIZER_PATH}"
    )

print("All required files found.")


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH
)

print(
    f"Total documents: {len(df)}"
)


# ============================================================
# VALIDATE DATASET
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


# ============================================================
# INPUT AND TARGET
# ============================================================

X = df["text"]

y = df["risk_level"]


# ============================================================
# SAME TEST SPLIT USED DURING TRAINING
# ============================================================

print("\nCreating evaluation split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    f"Training documents: {len(X_train)}"
)

print(
    f"Testing documents:  {len(X_test)}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

with open(
    MODEL_PATH,
    "rb"
) as file:

    model = pickle.load(
        file
    )

print(
    "Risk classification model loaded."
)


# ============================================================
# LOAD VECTORIZER
# ============================================================

print("\nLoading TF-IDF vectorizer...")

with open(
    VECTORIZER_PATH,
    "rb"
) as file:

    vectorizer = pickle.load(
        file
    )

print(
    "TF-IDF vectorizer loaded."
)


# ============================================================
# TRANSFORM TEST DATA
# ============================================================

print("\nTransforming test documents...")

X_test_tfidf = vectorizer.transform(
    X_test
)

print(
    f"Test feature shape: {X_test_tfidf.shape}"
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(
    X_test_tfidf
)

print(
    "Predictions generated successfully."
)


# ============================================================
# PROBABILITY PREDICTIONS
# ============================================================

if hasattr(model, "predict_proba"):

    y_probability = model.predict_proba(
        X_test_tfidf
    )

    confidence_scores = y_probability.max(
        axis=1
    )

else:

    confidence_scores = None


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


# ============================================================
# PRINT PERFORMANCE
# ============================================================

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

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    y_pred,
    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

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

print()

print(
    "Rows = Actual Risk"
)

print(
    "Columns = Predicted Risk"
)

print()

print(
    pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )
)


# ============================================================
# CREATE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Document Risk Classification - Evaluation Confusion Matrix"
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

for i in range(
    len(labels)
):

    for j in range(
        len(labels)
    ):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    EVALUATION_MATRIX_PATH,
    dpi=150
)

plt.close()

print(
    f"\nEvaluation confusion matrix saved to:"
)

print(
    EVALUATION_MATRIX_PATH
)


# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

if confidence_scores is not None:

    average_confidence = confidence_scores.mean()

    minimum_confidence = confidence_scores.min()

    maximum_confidence = confidence_scores.max()

    print("\n" + "=" * 70)
    print("CONFIDENCE ANALYSIS")
    print("=" * 70)

    print(
        f"\nAverage confidence : "
        f"{average_confidence * 100:.2f}%"
    )

    print(
        f"Minimum confidence: "
        f"{minimum_confidence * 100:.2f}%"
    )

    print(
        f"Maximum confidence: "
        f"{maximum_confidence * 100:.2f}%"
    )


# ============================================================
# SAVE EVALUATION REPORT
# ============================================================

print("\nSaving evaluation report...")

with open(
    EVALUATION_REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "AI DOCUMENT RISK & COMPLIANCE CLASSIFIER\n"
    )

    file.write(
        "MODEL EVALUATION REPORT\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Total documents: {len(df)}\n"
    )

    file.write(
        f"Training documents: {len(X_train)}\n"
    )

    file.write(
        f"Testing documents: {len(X_test)}\n\n"
    )

    file.write(
        "MODEL PERFORMANCE\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        f"Accuracy : {accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Precision: {precision * 100:.2f}%\n"
    )

    file.write(
        f"Recall   : {recall * 100:.2f}%\n"
    )

    file.write(
        f"F1 Score : {f1 * 100:.2f}%\n\n"
    )

    file.write(
        "CLASSIFICATION REPORT\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        report
    )

    file.write(
        "\n\nCONFUSION MATRIX\n"
    )

    file.write(
        "-" * 60 + "\n"
    )

    file.write(
        str(
            pd.DataFrame(
                cm,
                index=labels,
                columns=labels
            )
        )
    )

    if confidence_scores is not None:

        file.write(
            "\n\nCONFIDENCE ANALYSIS\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        file.write(
            f"Average confidence : "
            f"{average_confidence * 100:.2f}%\n"
        )

        file.write(
            f"Minimum confidence: "
            f"{minimum_confidence * 100:.2f}%\n"
        )

        file.write(
            f"Maximum confidence: "
            f"{maximum_confidence * 100:.2f}%\n"
        )


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFiles created:")

print(
    f"\nEvaluation report:"
)

print(
    EVALUATION_REPORT_PATH
)

print(
    f"\nEvaluation confusion matrix:"
)

print(
    EVALUATION_MATRIX_PATH
)

print("\n" + "=" * 70)