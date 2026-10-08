import re


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):
    """
    Clean extracted document text before classification.
    """

    if not text:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove excessive whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Remove unwanted special characters
    text = re.sub(
        r"[^a-z0-9\s.,!?%-]",
        " ",
        text
    )

    # Remove repeated spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()