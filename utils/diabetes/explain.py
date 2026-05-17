"""
Model explainability module using SHAP for the Pima Indians Diabetes Dataset.
Handles model loading, scaling, SHAP value generation, and visualization.
"""

import os
import logging
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

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
    """Load the trained machine learning model."""
    return load_object(model_path)

def load_scaler(scaler_path: str) -> Any:
    """Load the trained StandardScaler."""
    return load_object(scaler_path)

def prepare_and_scale_input(input_data: Dict[str, Any], scaler: Any) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Convert a dictionary of user input into a pandas DataFrame and scale it.

    Args:
        input_data (Dict[str, Any]): Dictionary containing patient data.
        scaler (Any): The loaded StandardScaler.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: The unscaled DataFrame and the scaled numpy array.

    Raises:
        ValueError: If any expected features are missing.
    """
    logger.info("Preparing and scaling user input data for SHAP explanations.")
    missing_features = [feat for feat in EXPECTED_FEATURES if feat not in input_data]
    if missing_features:
        logger.error(f"Missing expected features: {missing_features}")
        raise ValueError(f"Missing expected features: {missing_features}")

    # Create DataFrame (ensure columns are in the correct order)
    df = pd.DataFrame([input_data])[EXPECTED_FEATURES]
    
    try:
        scaled_data = scaler.transform(df)
        return df, scaled_data
    except Exception as e:
        logger.error(f"Error during feature scaling: {e}")
        raise e

def generate_shap_values(model: Any, scaled_data: np.ndarray) -> Tuple[Any, np.ndarray]:
    """
    Initialize the explainer and generate SHAP values for the given scaled data.
    Uses TreeExplainer for RandomForestClassifier natively.

    Args:
        model (Any): The trained machine learning model.
        scaled_data (np.ndarray): Scaled input features.

    Returns:
        Tuple[Any, np.ndarray]: The initialized explainer object and the calculated SHAP values.
    """
    logger.info("Generating SHAP values.")
    from sklearn.ensemble import RandomForestClassifier
    
    try:
        if isinstance(model, RandomForestClassifier):
            logger.info("Using shap.TreeExplainer for RandomForestClassifier.")
            explainer = shap.TreeExplainer(model)
            shap_values_raw = explainer.shap_values(scaled_data)
            
            # TreeExplainer on a binary classifier usually returns a list of length 2
            # We want the SHAP values corresponding to the positive class (class 1)
            if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
                shap_values = shap_values_raw[1]
            elif len(shap_values_raw.shape) == 3:
                shap_values = shap_values_raw[:, :, 1]
            else:
                shap_values = shap_values_raw
        else:
            logger.info("Using generic shap.Explainer for non-RandomForest model.")
            explainer = shap.Explainer(model)
            explanation = explainer(scaled_data)
            shap_values = explanation.values
            # Handle multi-class / binary format from standard Explainer
            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]
                
        return explainer, shap_values
    except Exception as e:
        logger.error(f"Error generating SHAP values: {e}")
        raise e

def extract_feature_contributions(shap_values: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    """
    Extract the individual feature importance scores for a single prediction.

    Args:
        shap_values (np.ndarray): SHAP values array for the prediction (shape: 1, n_features).
        feature_names (List[str]): List of column names.

    Returns:
        Dict[str, float]: Dictionary mapping feature names to their SHAP contribution values.
    """
    logger.info("Extracting feature contributions.")
    # Ensure 1D array for single sample extraction
    if len(shap_values.shape) == 2:
        vals = shap_values[0]
    else:
        vals = shap_values
        
    importance_dict = {feature_names[i]: float(vals[i]) for i in range(len(feature_names))}
    return importance_dict

def extract_top_contributors(importance_dict: Dict[str, float], top_n: int = 3) -> Dict[str, float]:
    """
    Extract the top N features that have the most impact (positive or negative) on the prediction.

    Args:
        importance_dict (Dict[str, float]): The feature importance scores.
        top_n (int): The number of top features to extract.

    Returns:
        Dict[str, float]: Sorted dictionary of the top contributing features.
    """
    logger.info(f"Extracting top {top_n} contributors.")
    # Sort by absolute SHAP value
    sorted_items = sorted(importance_dict.items(), key=lambda item: abs(item[1]), reverse=True)
    return dict(sorted_items[:top_n])

def create_shap_plots(explainer: Any, shap_values: np.ndarray, df_input: pd.DataFrame) -> Tuple[plt.Figure, plt.Figure]:
    """
    Create standard SHAP bar and waterfall plots using Matplotlib figures.

    Args:
        explainer (Any): The initialized explainer object.
        shap_values (np.ndarray): The calculated SHAP values.
        df_input (pd.DataFrame): The unscaled patient data (used for displaying values).

    Returns:
        Tuple[plt.Figure, plt.Figure]: Matplotlib Figure objects for (bar_plot, waterfall_plot).
    """
    logger.info("Generating SHAP plots (Bar and Waterfall).")
    try:
        vals = shap_values[0] if len(shap_values.shape) == 2 else shap_values

        # Safely extract expected_value (base value)
        expected_value = 0.5
        if hasattr(explainer, "expected_value"):
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)) and len(ev) > 1:
                expected_value = float(ev[1])
            elif isinstance(ev, (float, int, np.float64)):
                expected_value = float(ev)
                
        # Build Explanation object for newer SHAP plotting API
        explanation = shap.Explanation(
            values=vals,
            base_values=expected_value,
            data=df_input.iloc[0].values,
            feature_names=EXPECTED_FEATURES
        )

        # SHAP relies heavily on the current matplotlib state.
        # Plot 1: Bar Plot
        fig_bar = plt.figure(figsize=(8, 4))
        shap.plots.bar(explanation, show=False)
        fig_bar = plt.gcf()
        
        # Plot 2: Waterfall Plot
        fig_waterfall = plt.figure(figsize=(8, 4))
        shap.plots.waterfall(explanation, show=False)
        fig_waterfall = plt.gcf()

        return fig_bar, fig_waterfall
    except Exception as e:
        logger.error(f"Error creating SHAP plots: {e}")
        # Return empty figures as a safe fallback
        return plt.figure(), plt.figure()

def generate_explanation(
    input_data: Dict[str, Any], 
    model_path: str, 
    scaler_path: str
) -> Dict[str, Any]:
    """
    Complete end-to-end SHAP explanation workflow for a single patient prediction.

    Args:
        input_data (Dict[str, Any]): Dictionary of patient parameters.
        model_path (str): Path to the trained model file.
        scaler_path (str): Path to the trained scaler file.

    Returns:
        Dict[str, Any]: Dictionary containing structured explanation results:
            - feature_importance: Detailed dictionary of all feature SHAP scores.
            - top_contributors: The highest-impact features.
            - shap_values: List of raw SHAP values.
            - plots: Dictionary containing Matplotlib figure objects.
    """
    logger.info("Initializing complete SHAP explanation workflow.")
    try:
        # Step 1: Load artifacts
        model = load_model(model_path)
        scaler = load_scaler(scaler_path)

        # Step 2: Prepare and scale input
        df_input, scaled_input = prepare_and_scale_input(input_data, scaler)

        # Step 3: Generate SHAP values
        explainer, shap_values = generate_shap_values(model, scaled_input)

        # Step 4: Extract metrics
        feature_importance = extract_feature_contributions(shap_values, EXPECTED_FEATURES)
        top_contributors = extract_top_contributors(feature_importance, top_n=3)

        # Step 5: Visualizations
        fig_bar, fig_waterfall = create_shap_plots(explainer, shap_values, df_input)

        response = {
            "feature_importance": feature_importance,
            "top_contributors": top_contributors,
            "shap_values": shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
            "plots": {
                "bar_plot": fig_bar,
                "waterfall_plot": fig_waterfall
            }
        }
        
        logger.info("SHAP explanation generated successfully.")
        return response

    except Exception as e:
        logger.error(f"SHAP explanation workflow failed: {e}")
        raise e

if __name__ == "__main__":
    # Test block to execute the explainability module independently
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_model_path = os.path.join(project_root, "models", "diabetes_model.pkl")
    test_scaler_path = os.path.join(project_root, "models", "diabetes_scaler.pkl")

    # Sample patient data mapping to EXPECTED_FEATURES
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

    print("\n--- Generating SHAP Explanations for Sample Patient ---")
    print(f"Patient Data: {sample_patient}")
    print("-" * 55)

    try:
        # Attempt to run explanation module
        result = generate_explanation(
            input_data=sample_patient,
            model_path=test_model_path,
            scaler_path=test_scaler_path
        )
        print("\n--- Explanation Results ---")
        print("\nTop 3 Contributing Features:")
        for feat, score in result["top_contributors"].items():
            impact_direction = "increased" if score > 0 else "decreased"
            print(f" - {feat}: {score:+.4f} (This {impact_direction} the risk probability)")
            
        print("\nPlot objects generated successfully:")
        print(f" - Bar Plot: {type(result['plots']['bar_plot'])}")
        print(f" - Waterfall Plot: {type(result['plots']['waterfall_plot'])}")
        print("-------------------------------\n")
    except FileNotFoundError:
        print("\n[!] Models not found. Please run the training pipeline first to generate them.")
    except Exception as e:
        print(f"\n[!] An error occurred during explanation: {e}")
