import os
from io import BytesIO

import pytesseract
import pymupdf

from PIL import Image
from pypdf import PdfReader
from docx import Document


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]

for path in TESSERACT_PATHS:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        break


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(file_path):
    """
    First tries normal PDF text extraction.

    If the PDF is scanned/image-based and little or no
    text is found, automatically switches to OCR.
    """

    normal_text = ""

    try:
        reader = PdfReader(file_path)

        pages_text = []

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                pages_text.append(page_text)

        normal_text = "\n".join(pages_text).strip()

    except Exception as error:
        print(f"Normal PDF extraction failed: {error}")

    # Normal PDF
    if len(normal_text.strip()) >= 30:
        print("Normal PDF text detected.")
        return normal_text

    # Scanned PDF
    print("No usable PDF text detected.")
    print("Starting OCR...")

    return extract_scanned_pdf_text(file_path)


# ============================================================
# SCANNED PDF OCR
# ============================================================

def extract_scanned_pdf_text(file_path):
    """
    Converts each PDF page into an image and uses
    Tesseract OCR to extract the text.
    """

    # --------------------------------------------------------
    # Check Tesseract
    # --------------------------------------------------------

    try:
        version = pytesseract.get_tesseract_version()

        print(f"Tesseract version: {version}")

    except Exception as error:
        raise RuntimeError(
            "Tesseract OCR could not be found. "
            f"Error: {error}"
        )

    # --------------------------------------------------------
    # Open PDF
    # --------------------------------------------------------

    try:
        pdf = pymupdf.open(file_path)

    except Exception as error:
        raise RuntimeError(
            f"Could not open PDF for OCR: {error}"
        )

    extracted_pages = []

    try:
        total_pages = len(pdf)

        print(
            f"OCR: {total_pages} page(s) detected."
        )

        # ----------------------------------------------------
        # Process every page
        # ----------------------------------------------------

        for page_number, page in enumerate(
            pdf,
            start=1
        ):

            print(
                f"OCR processing page "
                f"{page_number}/{total_pages}..."
            )

            # ------------------------------------------------
            # Render page at high resolution
            # ------------------------------------------------

            matrix = pymupdf.Matrix(3, 3)

            pixmap = page.get_pixmap(
                matrix=matrix,
                alpha=False
            )

            # ------------------------------------------------
            # Convert to PIL image
            # ------------------------------------------------

            image_bytes = pixmap.tobytes("png")

            image = Image.open(
                BytesIO(image_bytes)
            )

            # ------------------------------------------------
            # OCR
            # ------------------------------------------------

            page_text = pytesseract.image_to_string(
                image,
                config="--psm 6"
            )

            page_text = page_text.strip()

            print(
                f"Page {page_number}: "
                f"{len(page_text)} characters detected."
            )

            if page_text:
                extracted_pages.append(page_text)

    finally:
        pdf.close()

    # --------------------------------------------------------
    # Combine OCR results
    # --------------------------------------------------------

    final_text = "\n\n".join(
        extracted_pages
    ).strip()

    # --------------------------------------------------------
    # Check OCR result
    # --------------------------------------------------------

    if not final_text:
        raise ValueError(
            "Could not extract readable text "
            "from the PDF even after OCR."
        )

    print(
        f"OCR completed successfully. "
        f"Total characters: {len(final_text)}"
    )

    return final_text


# ============================================================
# DOCX EXTRACTION
# ============================================================

def extract_docx_text(file_path):
    """
    Extract text from a DOCX document.
    """

    document = Document(file_path)

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs).strip()


# ============================================================
# MAIN TEXT EXTRACTION FUNCTION
# ============================================================

def extract_text(file_path):
    """
    Main document extraction function.

    Supported:
        PDF
        DOCX

    PDF automatically supports:
        Normal PDFs
        Scanned PDFs through OCR
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = os.path.splitext(
        file_path
    )[1].lower()

    if extension == ".pdf":

        return extract_pdf_text(
            file_path
        )

    elif extension == ".docx":

        return extract_docx_text(
            file_path
        )

    else:

        raise ValueError(
            "Unsupported file type. "
            "Only PDF and DOCX files are supported."
        )