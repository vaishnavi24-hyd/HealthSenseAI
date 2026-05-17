"""
Prediction engine for the Pima Indians Diabetes Dataset.
Handles loading models, scaling input, generating predictions, and calculating risk levels.
"""

import os
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants mapping to expected features in the model
EXPECTED_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

def load_object(filepath: str) -> Any:
    """
    Generic function to load a serialized object using joblib.

    Args:
        filepath (str): Path to the serialized file.

    Returns:
        Any: Loaded object.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For any other loading errors.
    """
    logger.debug(f"Attempting to load object from {filepath}")
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")
    try:
        obj = joblib.load(filepath)
        logger.info(f"Successfully loaded object from {filepath}")
        return obj
    except Exception as e:
        logger.error(f"Error loading object from {filepath}: {e}")
        raise e

def load_model(model_path: str) -> Any:
    """
    Load the trained machine learning model.

    Args:
        model_path (str): Path to the model .pkl file.

    Returns:
        Any: The loaded scikit-learn model.
    """
    return load_object(model_path)

def load_scaler(scaler_path: str) -> Any:
    """
    Load the trained StandardScaler.

    Args:
        scaler_path (str): Path to the scaler .pkl file.

    Returns:
        Any: The loaded scikit-learn scaler.
    """
    return load_object(scaler_path)

def prepare_user_input(input_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert a dictionary of user input into a properly formatted pandas DataFrame.

    Args:
        input_data (Dict[str, Any]): Dictionary containing patient data.

    Returns:
        pd.DataFrame: DataFrame containing a single row with correct column names.

    Raises:
        ValueError: If any expected features are missing from the input data.
    """
    logger.info("Preparing user input data for prediction.")
    # Check for missing features
    missing_features = [feat for feat in EXPECTED_FEATURES if feat not in input_data]
    if missing_features:
        logger.error(f"Missing expected features in input data: {missing_features}")
        raise ValueError(f"Missing expected features: {missing_features}")

    # Create DataFrame (ensure columns are in the correct order)
    df = pd.DataFrame([input_data])[EXPECTED_FEATURES]
    return df

def scale_user_input(df: pd.DataFrame, scaler: Any) -> np.ndarray:
    """
    Scale the prepared DataFrame using the loaded scaler.

    Args:
        df (pd.DataFrame): The prepared user input DataFrame.
        scaler (Any): The loaded StandardScaler.

    Returns:
        np.ndarray: The scaled features ready for the model.
    """
    logger.info("Scaling user input data.")
    try:
        scaled_data = scaler.transform(df)
        return scaled_data
    except Exception as e:
        logger.error(f"Error during feature scaling: {e}")
        raise e

def generate_prediction(model: Any, scaled_data: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Generate the prediction label and probability from the model.

    Args:
        model (Any): The trained machine learning model.
        scaled_data (np.ndarray): The scaled input features.

    Returns:
        Tuple[int, np.ndarray]: The predicted class (0 or 1) and the probability array.
    """
    logger.info("Generating prediction from the model.")
    try:
        prediction = int(model.predict(scaled_data)[0])
        # Some models might not support predict_proba, fallback safely if needed
        # but our training pipeline guarantees models with probability=True
        probabilities = model.predict_proba(scaled_data)[0]
        return prediction, probabilities
    except AttributeError:
        logger.error("Model does not support probability predictions.")
        raise
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        raise e

def calculate_probability(probabilities: np.ndarray, class_index: int = 1) -> float:
    """
    Calculate the percentage probability for a specific class.

    Args:
        probabilities (np.ndarray): Array of probabilities from the model.
        class_index (int): The index of the class to get probability for (default: 1 for positive class).

    Returns:
        float: Probability percentage (0.0 to 100.0).
    """
    prob_percentage = probabilities[class_index] * 100.0
    return round(prob_percentage, 2)

def determine_risk_level(probability_percentage: float) -> str:
    """
    Convert a probability percentage into a categorical risk level.

    Risk logic:
    0 - 35%  : LOW
    35 - 70% : MEDIUM
    70 - 100%: HIGH

    Args:
        probability_percentage (float): The probability percentage (0-100).

    Returns:
        str: The calculated risk level ("LOW", "MEDIUM", "HIGH").
    """
    if probability_percentage < 35.0:
        return "LOW"
    elif probability_percentage < 70.0:
        return "MEDIUM"
    else:
        return "HIGH"

def predict_diabetes_risk(
    input_data: Dict[str, Any], 
    model_path: str, 
    scaler_path: str
) -> Dict[str, Any]:
    """
    Complete end-to-end prediction engine.

    Args:
        input_data (Dict[str, Any]): Dictionary of patient parameters.
        model_path (str): Path to the trained model file.
        scaler_path (str): Path to the trained scaler file.

    Returns:
        Dict[str, Any]: Dictionary containing prediction results:
            - "prediction_label": int (0 or 1)
            - "probability_percentage": float
            - "risk_level": str
    """
    logger.info("Initializing complete prediction engine workflow.")
    try:
        # Step 1: Load artifacts
        model = load_model(model_path)
        scaler = load_scaler(scaler_path)

        # Step 2: Prepare and scale input
        df_input = prepare_user_input(input_data)
        scaled_input = scale_user_input(df_input, scaler)

        # Step 3: Generate prediction
        prediction, probabilities = generate_prediction(model, scaled_input)

        # Step 4: Calculate metrics
        prob_percentage = calculate_probability(probabilities, class_index=1)
        risk_level = determine_risk_level(prob_percentage)

        response = {
            "prediction_label": prediction,
            "probability_percentage": prob_percentage,
            "risk_level": risk_level
        }
        
        logger.info(f"Prediction successful: {response}")
        return response

    except Exception as e:
        logger.error(f"Prediction workflow failed: {e}")
        raise e

if __name__ == "__main__":
    # Test block to execute the prediction engine independently
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_model_path = os.path.join(project_root, "models", "diabetes_model.pkl")
    test_scaler_path = os.path.join(project_root, "models", "diabetes_scaler.pkl")

    # Sample patient data mapping to the EXPECTED_FEATURES
    sample_patient = {
        "Pregnancies": 2,
        "Glucose": 135,
        "BloodPressure": 75,
        "SkinThickness": 30,
        "Insulin": 120,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.45,
        "Age": 42
    }

    print("\n--- Running Prediction Engine on Sample Patient ---")
    print(f"Patient Data: {sample_patient}")
    print("-" * 50)

    try:
        # Attempt to run prediction
        result = predict_diabetes_risk(
            input_data=sample_patient,
            model_path=test_model_path,
            scaler_path=test_scaler_path
        )
        print("\n--- Prediction Results ---")
        print(f"Label:       {result['prediction_label']} (1=Positive, 0=Negative)")
        print(f"Probability: {result['probability_percentage']}%")
        print(f"Risk Level:  {result['risk_level']}")
        print("--------------------------\n")
    except FileNotFoundError:
        print("\n[!] Models not found. Please run the training pipeline first to generate them.")
    except Exception as e:
        print(f"\n[!] An error occurred during prediction: {e}")
