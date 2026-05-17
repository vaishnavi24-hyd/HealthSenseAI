"""
Heart Disease Prediction Engine
Handles loading trained artifacts, preparing inputs, and executing inference.
"""

import os
import sys
import logging
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
import joblib

# Configure logging
logger = logging.getLogger(__name__)

def load_artifact(filepath: str) -> Any:
    """
    Safely loads a serialized joblib artifact from disk.
    
    Args:
        filepath (str): Path to the artifact (e.g., model or scaler).
        
    Returns:
        Any: The deserialized Python object.
        
    Raises:
        FileNotFoundError: If the artifact does not exist.
        ValueError: If the file is corrupted.
    """
    if not os.path.exists(filepath):
        logger.error(f"Required artifact not found at {filepath}")
        raise FileNotFoundError(f"Missing essential backend artifact: {filepath}")
    try:
        obj = joblib.load(filepath)
        logger.info(f"Successfully loaded object from {filepath}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load artifact from {filepath}: {e}")
        raise ValueError(f"Corrupted or invalid artifact file: {e}")

def prepare_input_data(patient_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Converts a raw dictionary of patient metrics into a DataFrame.
    Ensures standard ordering based on the Cleveland dataset feature set.
    
    Args:
        patient_data (Dict[str, Any]): Dictionary of clinical values.
        
    Returns:
        pd.DataFrame: Formatted single-row DataFrame.
    """
    # Expected feature list for heart disease (13 features)
    features = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", 
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    
    try:
        # Convert dict to single-row dataframe ensuring exact column order
        df = pd.DataFrame([patient_data], columns=features)
        
        # Verify no missing inputs resulting in NaN
        if df.isnull().values.any():
            raise ValueError("Input dictionary contains missing clinical features.")
            
        return df
    except Exception as e:
        logger.error(f"Input formatting error: {e}")
        raise e

def scale_input_data(df: pd.DataFrame, scaler: Any) -> np.ndarray:
    """
    Applies the pre-fitted StandardScaler to the patient data.
    
    Args:
        df (pd.DataFrame): Patient features.
        scaler (Any): Fitted sklearn StandardScaler.
        
    Returns:
        np.ndarray: Scaled features array.
    """
    try:
        scaled_data = scaler.transform(df)
        return scaled_data
    except Exception as e:
        logger.error(f"Error during feature scaling: {e}")
        raise ValueError(f"Failed to scale input features: {e}")

def generate_risk_classification(probability: float) -> str:
    """
    Categorizes clinical risk based on probability percentage thresholds.
    
    Args:
        probability (float): Percentage probability (0-100).
        
    Returns:
        str: Categorical risk tier ('LOW', 'MEDIUM', 'HIGH').
    """
    if probability < 35.0:
        return "LOW"
    elif probability < 70.0:
        return "MEDIUM"
    else:
        return "HIGH"

def predict_heart_disease_risk(
    patient_data: Dict[str, Any],
    model_path: str = "models/heart_model.pkl",
    scaler_path: str = "models/heart_scaler.pkl"
) -> Dict[str, Any]:
    """
    Orchestrates the entire ML inference pipeline for a single patient.
    
    Args:
        patient_data (Dict[str, Any]): Dictionary containing patient clinical metrics.
        model_path (str): Relative path to the trained prediction model.
        scaler_path (str): Relative path to the trained StandardScaler.
        
    Returns:
        Dict[str, Any]: Standardized response dictionary containing label, probability, and risk tier.
    """
    logger.info("Initializing complete heart disease prediction engine workflow.")
    
    # Resolve absolute paths safely
    if not os.path.isabs(model_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(project_root, model_path)
        scaler_path = os.path.join(project_root, scaler_path)
        
    try:
        # Load core artifacts
        model = load_artifact(model_path)
        scaler = load_artifact(scaler_path)
        
        # Prepare & Scale Data
        logger.info("Preparing user input data for prediction.")
        input_df = prepare_input_data(patient_data)
        
        logger.info("Scaling user input data.")
        input_scaled = scale_input_data(input_df, scaler)
        
        # Inference
        logger.info("Generating prediction from the model.")
        label = int(model.predict(input_scaled)[0])
        
        # Calculate Probability
        if hasattr(model, "predict_proba"):
            # Some models expose probabilities
            prob = float(model.predict_proba(input_scaled)[0][1]) * 100
        else:
            # Fallback for models without native probability estimation
            # Note: The training pipeline currently builds LogisticRegression (which supports it)
            prob = 100.0 if label == 1 else 0.0
            
        prob_rounded = round(prob, 1)
        
        # Tier Assessment
        risk_level = generate_risk_classification(prob_rounded)
        
        result = {
            "prediction_label": label,
            "probability_percentage": prob_rounded,
            "risk_level": risk_level
        }
        
        logger.info(f"Prediction successful: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Prediction workflow failed: {e}")
        raise e

if __name__ == "__main__":
    # Configure simple testing logger
    logging.basicConfig(level=logging.INFO)
    
    # Sample patient metrics (Cleveland formatting)
    # Features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    sample_patient = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1
    }
    
    print("--- Testing Heart Disease Inference Engine ---")
    try:
        prediction_result = predict_heart_disease_risk(sample_patient)
        print("\n✅ Inference Successful!")
        print(f"Assessed Risk Level: {prediction_result['risk_level']}")
        print(f"Confidence Score:    {prediction_result['probability_percentage']}%")
        print(f"Raw Label Output:    {prediction_result['prediction_label']}")
    except Exception as ex:
        print(f"❌ Inference Test Failed: {ex}")
