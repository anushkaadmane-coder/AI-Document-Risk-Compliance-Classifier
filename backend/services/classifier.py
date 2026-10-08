import os
import pickle


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "models",
    "risk_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "models",
    "tfidf_vectorizer.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

with open(
    MODEL_PATH,
    "rb"
) as file:

    model = pickle.load(
        file
    )


# ============================================================
# LOAD VECTORIZER
# ============================================================

with open(
    VECTORIZER_PATH,
    "rb"
) as file:

    vectorizer = pickle.load(
        file
    )


# ============================================================
# CONFIDENCE THRESHOLD
# ============================================================

CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# CLASSIFY DOCUMENT
# ============================================================

def classify_document(text):
    """
    Classify document risk and return
    prediction, confidence and review status.
    """

    if not text or not text.strip():

        raise ValueError(
            "Document text is empty."
        )

    # Convert text to TF-IDF
    text_vector = vectorizer.transform(
        [text]
    )

    # Predict risk
    prediction = model.predict(
        text_vector
    )[0]

    # Get probabilities
    probabilities = model.predict_proba(
        text_vector
    )[0]

    # Highest probability
    confidence = float(
        max(probabilities)
    )

    # Manual review decision
    review_required = (
        confidence < CONFIDENCE_THRESHOLD
    )

    if review_required:

        review_status = (
            "Manual Review Required"
        )

    else:

        review_status = (
            "Automatically Classified"
        )

    # Probability for each class
    class_probabilities = {}

    for class_name, probability in zip(
        model.classes_,
        probabilities
    ):

        class_probabilities[
            class_name
        ] = round(
            float(probability) * 100,
            2
        )

    return {
        "risk_level": prediction,
        "confidence": round(
            confidence * 100,
            2
        ),
        "review_required": review_required,
        "review_status": review_status,
        "class_probabilities": class_probabilities
    }