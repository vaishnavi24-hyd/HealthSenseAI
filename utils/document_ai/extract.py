"""
Production-quality Medical PDF text extraction module for AegisHealth AI.
Leverages PyMuPDF (fitz) for rapid text scraping and pdfplumber as a highly
accurate fallback for complex tabular data commonly found in blood tests and lab reports.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import fitz  # PyMuPDF
import pdfplumber

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def validate_pdf_file(file_path: Union[str, Path]) -> bool:
    """
    Validates if the provided file path is a valid and accessible PDF file.
    
    Args:
        file_path (Union[str, Path]): Path to the PDF file.
        
    Returns:
        bool: True if valid, False otherwise.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"Validation failed: File does not exist -> {path}")
        return False
    if not path.is_file():
        logger.error(f"Validation failed: Path is not a file -> {path}")
        return False
    if path.suffix.lower() != '.pdf':
        logger.error(f"Validation failed: File is not a PDF -> {path}")
        return False
        
    return True

def clean_extracted_text(raw_text: str) -> str:
    """
    Cleans raw extracted text from medical PDFs by stripping out anomalous
    characters, excessive whitespace, and normalizing the text flow.
    
    Args:
        raw_text (str): The raw text extracted from the PDF.
        
    Returns:
        str: The cleaned, normalized text string.
    """
    if not raw_text:
        return ""
        
    # Remove null bytes and unusual control characters
    text = raw_text.replace('\x00', '')
    
    # Normalize varied newline combinations into a single standard newline
    text = re.sub(r'(\r\n|\r|\n)+', '\n', text)
    
    # Remove excessive horizontal whitespace (tabs, multiple spaces)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Strip leading/trailing whitespace from each line
    text = '\n'.join([line.strip() for line in text.split('\n') if line.strip()])
    
    return text.strip()

def extract_text_with_fitz(file_path: Path) -> str:
    """
    Rapidly extracts text using PyMuPDF (fitz). Highly efficient for standard
    text blocks in medical diagnostic reports.
    
    Args:
        file_path (Path): Validated path to the PDF file.
        
    Returns:
        str: Extracted raw text.
    """
    extracted_text = []
    try:
        with fitz.open(file_path) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # extract_text() usually handles standard text flows well
                text = page.get_text("text")
                if text:
                    extracted_text.append(text)
        logger.info("Successfully extracted text using PyMuPDF (fitz).")
        return "\n".join(extracted_text)
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {e}")
        raise e

def extract_text_with_pdfplumber(file_path: Path) -> str:
    """
    Extracts text using pdfplumber. Excellent fallback for complex
    tabular data like blood panels and lab assays.
    
    Args:
        file_path (Path): Validated path to the PDF file.
        
    Returns:
        str: Extracted raw text.
    """
    extracted_text = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text.append(text)
        logger.info("Successfully extracted text using pdfplumber.")
        return "\n".join(extracted_text)
    except Exception as e:
        logger.error(f"pdfplumber extraction failed: {e}")
        raise e

def extract_medical_pdf(file_path: Union[str, Path]) -> Dict[str, Optional[str]]:
    """
    Main orchestrator for extracting text from medical PDFs.
    Attempts extraction with PyMuPDF first for speed. If it yields poor results or fails,
    it falls back to pdfplumber which handles tables (lab reports) more robustly.
    
    Args:
        file_path (Union[str, Path]): Path to the medical PDF.
        
    Returns:
        Dict[str, Optional[str]]: Dictionary containing:
            - 'raw_text': The unedited text extracted from the document.
            - 'cleaned_text': The sanitized and normalized text.
            - 'error': Error message if extraction failed completely.
    """
    logger.info(f"Initiating extraction sequence for: {file_path}")
    
    result = {
        "raw_text": None,
        "cleaned_text": None,
        "error": None
    }
    
    if not validate_pdf_file(file_path):
        result["error"] = "Invalid or inaccessible PDF file path."
        return result
        
    path = Path(file_path)
    raw_text = ""
    
    try:
        # Strategy 1: Fast text extraction via PyMuPDF
        raw_text = extract_text_with_fitz(path)
        
        # If PyMuPDF extracts very little text, it might be heavily tabular or image-based
        if len(raw_text.strip()) < 50:
            logger.warning("PyMuPDF yielded low text volume. Falling back to pdfplumber...")
            raw_text = extract_text_with_pdfplumber(path)
            
    except Exception as e_fitz:
        logger.warning(f"Strategy 1 (fitz) encountered fatal error. Engaging Strategy 2 (pdfplumber)...")
        try:
            # Strategy 2: Robust fallback extraction via pdfplumber
            raw_text = extract_text_with_pdfplumber(path)
        except Exception as e_plumber:
            error_msg = f"Complete extraction failure. Fitz error: {e_fitz}. Plumber error: {e_plumber}"
            logger.error(error_msg)
            result["error"] = error_msg
            return result
            
    if not raw_text.strip():
        result["error"] = "Extraction completed but no text was found (document may be purely scanned images/OCR required)."
        return result
        
    # Populate successful results
    result["raw_text"] = raw_text
    result["cleaned_text"] = clean_extracted_text(raw_text)
    logger.info(f"Extraction complete. Yielded {len(result['cleaned_text'])} cleaned characters.")
    
    return result

if __name__ == "__main__":
    import tempfile
    
    print("--- Testing Medical PDF Extraction Module ---")
    
    # Create a mock PDF using reportlab for testing extraction logic
    from reportlab.pdfgen import canvas
    
    # 1. Setup mock file
    temp_pdf_path = Path(tempfile.gettempdir()) / "mock_lab_report.pdf"
    print(f"\n[1] Generating mock lab report at {temp_pdf_path}...")
    
    try:
        c = canvas.Canvas(str(temp_pdf_path))
        c.drawString(100, 800, "COMPREHENSIVE METABOLIC PANEL")
        c.drawString(100, 780, "Patient: John Doe | Age: 45 | Sex: Male")
        c.drawString(100, 750, "Glucose: 105 mg/dL (High)")
        c.drawString(100, 735, "Cholesterol: 240 mg/dL (High)")
        c.drawString(100, 720, "Insulin: 15 IU/mL (Normal)")
        c.showPage()
        c.save()
        print("✅ Mock PDF generated.")
    except ImportError:
        print("⚠️ ReportLab not found. Please provide an actual PDF for testing.")
        
    # 2. Test the extraction
    print("\n[2] Testing Extraction Pipeline...")
    if temp_pdf_path.exists():
        extracted_data = extract_medical_pdf(temp_pdf_path)
        
        if extracted_data["error"]:
            print(f"❌ Extraction failed: {extracted_data['error']}")
        else:
            print("✅ Extraction succeeded!")
            print("-" * 40)
            print("CLEANED TEXT FRAGMENT:")
            print(extracted_data["cleaned_text"][:200]) # Print first 200 chars
            print("-" * 40)
            
        # Cleanup
        temp_pdf_path.unlink()
    else:
        print("❌ Mock PDF was not created successfully.")
        
    # 3. Test validation failure
    print("\n[3] Testing Invalid File Handling...")
    bad_data = extract_medical_pdf("non_existent_file_123.pdf")
    if bad_data["error"]:
        print(f"✅ Correctly caught invalid file: {bad_data['error']}")
    else:
        print("❌ Failed to catch invalid file.")
