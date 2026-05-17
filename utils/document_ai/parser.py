"""
Production-quality Medical Document Parser for AegisHealth AI.
Utilizes advanced regex patterns to extract key clinical biomarkers from 
unstructured text extracted from medical PDFs.
"""

import logging
import re
import pandas as pd
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- REGEX PATTERNS ---
# These patterns handle variations like "Glucose: 182", "Cholesterol - 240", "BMI = 31.2", "Insulin 18.5"
BIOMARKER_PATTERNS = {
    "Glucose": r"(?i)(?:glucose|gluc|fasting plasma glucose|fpg)[\s:=\-]+([\d\.]+)",
    "Cholesterol": r"(?i)(?:cholesterol|chol|total cholesterol)[\s:=\-]+([\d\.]+)",
    "BloodPressure": r"(?i)(?:bp|blood pressure)[\s:=\-]+([\d\.]+)",  # Captures systolic
    "BMI": r"(?i)(?:bmi|body mass index)[\s:=\-]+([\d\.]+)",
    "Insulin": r"(?i)(?:insulin)[\s:=\-]+([\d\.]+)",
    "Age": r"(?i)(?:age|y/o|years old)[\s:=\-]+([\d\.]+)",
    "HeartRate": r"(?i)(?:hr|heart rate|pulse|thalach)[\s:=\-]+([\d\.]+)",
    "FastingBloodSugar": r"(?i)(?:fasting blood sugar|fbs)[\s:=\-]+([\d\.]+)"
}

def clean_extracted_value(val_str: str) -> str:
    """
    Cleans the regex matched string to remove any trailing non-numeric artifacts.
    
    Args:
        val_str (str): The raw matched string.
        
    Returns:
        str: Cleaned string containing only numeric characters and a decimal point.
    """
    # Strip out any trailing characters that aren't digits or decimals
    cleaned = re.sub(r'[^\d\.]', '', val_str)
    return cleaned

def convert_to_numeric(val_str: str) -> Optional[float]:
    """
    Safely converts a cleaned string to a float.
    
    Args:
        val_str (str): The cleaned string to convert.
        
    Returns:
        Optional[float]: The converted float, or None if conversion fails.
    """
    try:
        return float(val_str)
    except (ValueError, TypeError):
        return None

def validate_extracted_biomarker(name: str, value: float) -> Optional[float]:
    """
    Validates if the extracted value falls within logically acceptable clinical ranges.
    This prevents extreme outliers caused by parsing errors from corrupting the AI inference.
    
    Args:
        name (str): The biomarker name.
        value (float): The parsed numeric value.
        
    Returns:
        Optional[float]: The validated value, or None if it fails validation.
    """
    # Define broad but logical physiological limits
    RANGES = {
        "Glucose": (10, 1000),
        "Cholesterol": (50, 800),
        "BloodPressure": (30, 300),
        "BMI": (10, 100),
        "Insulin": (0, 1500),
        "Age": (0, 120),
        "HeartRate": (20, 300),
        "FastingBloodSugar": (10, 1000)
    }
    
    if name in RANGES:
        min_val, max_val = RANGES[name]
        if min_val <= value <= max_val:
            return value
        else:
            logger.warning(f"Validation failed for {name}: {value} is outside logical clinical range ({min_val}-{max_val}).")
            return None
            
    # Default fallback if no range specified
    return value

def parse_medical_values(text: str) -> Dict[str, Optional[float]]:
    """
    Parses unstructured medical text to extract structured clinical biomarkers.
    
    Args:
        text (str): Cleaned, unstructured text from a medical document.
        
    Returns:
        Dict[str, Optional[float]]: A dictionary of extracted clinical values.
    """
    logger.info("Initiating clinical text parsing sequence...")
    extracted_data = {}
    
    if not text or not isinstance(text, str):
        logger.error("Empty or invalid text provided for parsing.")
        return {key: None for key in BIOMARKER_PATTERNS.keys()}
        
    for biomarker, pattern in BIOMARKER_PATTERNS.items():
        match = re.search(pattern, text)
        
        if match:
            raw_val = match.group(1)
            cleaned_val = clean_extracted_value(raw_val)
            numeric_val = convert_to_numeric(cleaned_val)
            
            if numeric_val is not None:
                validated_val = validate_extracted_biomarker(biomarker, numeric_val)
                extracted_data[biomarker] = validated_val
                if validated_val is not None:
                    logger.debug(f"Successfully extracted {biomarker}: {validated_val}")
            else:
                logger.warning(f"Failed to convert {biomarker} value '{raw_val}' to numeric.")
                extracted_data[biomarker] = None
        else:
            logger.info(f"Biomarker '{biomarker}' not found in document text.")
            extracted_data[biomarker] = None
            
    logger.info(f"Parsing complete. Successfully extracted {sum(v is not None for v in extracted_data.values())}/{len(BIOMARKER_PATTERNS)} biomarkers.")
    return extracted_data

if __name__ == "__main__":
    print("--- Testing Medical Document Parser Module ---")
    
    sample_text = """
    COMPREHENSIVE METABOLIC PANEL
    Date: 2026-05-17
    Patient: John Doe | Age: 45 | Sex: Male
    
    Results:
    Glucose: 182 mg/dL (High)
    Cholesterol - 240
    BP: 145/90
    BMI = 31.2
    Insulin 18.5 IU/mL
    Pulse: 88 bpm
    Fasting Blood Sugar: 175
    
    Notes: Patient advised to follow up on elevated lipid profiles.
    """
    
    print("\n[1] Parsing Sample Medical Text...")
    results = parse_medical_values(sample_text)
    
    # Convert to pandas series for clean console output
    series_results = pd.Series(results).fillna("MISSING")
    
    print("\n✅ Extraction Results:")
    print("-" * 30)
    print(series_results)
    print("-" * 30)
    
    # Validate specific extractions
    assert results["Glucose"] == 182.0, "Glucose extraction failed"
    assert results["Cholesterol"] == 240.0, "Cholesterol extraction failed"
    assert results["BloodPressure"] == 145.0, "BloodPressure extraction failed"
    assert results["BMI"] == 31.2, "BMI extraction failed"
    assert results["Insulin"] == 18.5, "Insulin extraction failed"
    assert results["Age"] == 45.0, "Age extraction failed"
    assert results["HeartRate"] == 88.0, "HeartRate extraction failed"
    assert results["FastingBloodSugar"] == 175.0, "FastingBloodSugar extraction failed"
    
    print("\n✅ All unit tests passed! Parser successfully handles ':' , '-', '=', and space delimiters.")
