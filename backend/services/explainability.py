import re


# ============================================================
# IMPORTANT RISK TERMS
# ============================================================

RISK_TERMS = {

    "Critical": [
        "regulatory",
        "penalties",
        "legal",
        "highly sensitive",
        "severe",
        "financial",
        "personal data",
        "privacy",
        "security",
        "incident",
        "protected",
        "compliance"
    ],

    "High": [
        "personal",
        "financial",
        "privacy",
        "security",
        "customer data",
        "encryption",
        "third-party",
        "regulatory",
        "sensitive"
    ],

    "Medium": [
        "confidential",
        "internal",
        "customer",
        "vendor",
        "access",
        "retention",
        "authorization",
        "business information"
    ],

    "Low": [
        "routine",
        "general",
        "administrative",
        "office",
        "meeting",
        "schedule",
        "workflow",
        "operations"
    ]
}


# ============================================================
# EXTRACT IMPORTANT TERMS
# ============================================================

def extract_important_terms(
    text,
    risk_level,
    limit=8
):
    """
    Find important risk-related terms
    from the document.
    """

    if not text:
        return []

    text_lower = text.lower()

    terms = RISK_TERMS.get(
        risk_level,
        []
    )

    found_terms = []

    for term in terms:

        if re.search(
            r"\b" + re.escape(term) + r"\b",
            text_lower
        ):

            found_terms.append(
                term
            )

    return found_terms[:limit]