from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import os
import shutil
import json
from pathlib import Path

from backend.database import (
    initialize_database,
    save_prediction,
    get_predictions,
    get_prediction,
    get_statistics,
    clear_predictions,
)

from backend.services.text_extractor import extract_text
from backend.services.preprocessor import preprocess_text
from backend.services.classifier import classify_document
from backend.services.explainability import extract_important_terms


# ============================================================
# APP CONFIGURATION
# ============================================================

app = FastAPI(
    title="AI Document Risk & Compliance Classifier",
    description="AI-powered document risk and compliance classification system",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INITIALIZE DATABASE
# ============================================================

initialize_database()


# ============================================================
# POLICY CATEGORY MAPPING
# ============================================================

POLICY_CATEGORY_MAP = {
    "Low": "General Business",
    "Medium": "Confidentiality",
    "High": "Privacy / Data Security",
    "Critical": "Regulatory Compliance",
}


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "AI Document Risk & Compliance Classifier API",
        "status": "running",
        "version": "1.0.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "AI Document Risk & Compliance Classifier"
    }


# ============================================================
# ANALYZE DOCUMENT
# ============================================================

@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in [".pdf", ".docx"]:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are supported."
        )

    # --------------------------------------------------------
    # Temporary file path
    # --------------------------------------------------------

    safe_filename = Path(file.filename).name

    file_path = UPLOAD_DIR / safe_filename

    try:

        # ----------------------------------------------------
        # Save uploaded file
        # ----------------------------------------------------

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

        # ----------------------------------------------------
        # Extract text
        # ----------------------------------------------------

        extracted_text = extract_text(
            str(file_path)
        )

        if not extracted_text or not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No readable text was found in the document."
            )

        # ----------------------------------------------------
        # Preprocess text
        # ----------------------------------------------------

        processed_text = preprocess_text(
            extracted_text
        )

        if not processed_text:
            raise HTTPException(
                status_code=400,
                detail="Document text could not be processed."
            )

        # ----------------------------------------------------
        # AI classification
        # ----------------------------------------------------

        classification = classify_document(
            processed_text
        )

        risk_level = classification["risk_level"]

        confidence = classification["confidence"]

        review_required = classification["review_required"]

        review_status = classification["review_status"]

        class_probabilities = classification[
            "class_probabilities"
        ]

        # ----------------------------------------------------
        # Policy category
        # ----------------------------------------------------

        policy_category = POLICY_CATEGORY_MAP.get(
            risk_level,
            "General Business"
        )

        # ----------------------------------------------------
        # Important terms
        # ----------------------------------------------------

        important_terms = extract_important_terms(
            processed_text,
            risk_level
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # SQLite cannot directly store a Python list.
        # Convert important_terms to JSON text.
        # ----------------------------------------------------

        important_terms_for_database = json.dumps(
            important_terms
        )

        # ----------------------------------------------------
        # Save prediction to database
        # ----------------------------------------------------

        prediction_id = save_prediction(

            filename=safe_filename,

            risk_level=risk_level,

            policy_category=policy_category,

            confidence=confidence,

            review_required=review_required,

            review_status=review_status,

            important_terms=important_terms_for_database,

            extracted_text=extracted_text
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {

            "success": True,

            "message": "Document analyzed successfully.",

            "prediction": {

                "id": prediction_id,

                "filename": safe_filename,

                "risk_level": risk_level,

                "policy_category": policy_category,

                "confidence": confidence,

                "review_required": review_required,

                "review_status": review_status,

                "important_terms": important_terms,

                "class_probabilities": class_probabilities,

                "extracted_text": extracted_text

            }

        }

    except HTTPException:

        raise

    except Exception as e:

        print(
            f"Document analysis error: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Document analysis failed: {str(e)}"
        )

    finally:

        # ----------------------------------------------------
        # Delete temporary uploaded file
        # ----------------------------------------------------

        try:

            if file_path.exists():
                file_path.unlink()

        except Exception as cleanup_error:

            print(
                f"File cleanup warning: {cleanup_error}"
            )


# ============================================================
# GET ALL PREDICTIONS
# ============================================================

@app.get("/predictions")
def predictions():

    try:

        return get_predictions()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve predictions: {str(e)}"
        )


# ============================================================
# GET SINGLE PREDICTION
# ============================================================

@app.get("/predictions/{prediction_id}")
def prediction_details(
    prediction_id: int
):

    try:

        prediction = get_prediction(
            prediction_id
        )

        if not prediction:

            raise HTTPException(
                status_code=404,
                detail="Prediction not found."
            )

        return prediction

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve prediction: {str(e)}"
        )


# ============================================================
# GET STATISTICS
# ============================================================

@app.get("/statistics")
def statistics():

    try:

        return get_statistics()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve statistics: {str(e)}"
        )


# ============================================================
# DELETE ALL PREDICTIONS
# ============================================================

@app.delete("/predictions")
def delete_predictions():

    try:

        clear_predictions()

        return {
            "success": True,
            "message": "All prediction history cleared."
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Could not clear predictions: {str(e)}"
        )


# ============================================================
# RUN INFORMATION
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )