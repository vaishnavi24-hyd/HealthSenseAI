"""
Model explainability module using SHAP for the Heart Disease Dataset.
Handles model loading, scaling, SHAP value generation, and visualization.
"""

import os
import logging
from typing import Dict, Any, Tuple, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

# Configure logging
logger = logging.getLogger(__name__)

# Constants mapping to expected features in the Cleveland Heart Disease model
EXPECTED_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

def load_object(filepath: str) -> Any:
    """Safely loads a serialized joblib object."""
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

def load_model(model_path: str = "models/heart_model.pkl") -> Any:
    """Loads the trained machine learning model."""
    if not os.path.isabs(model_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(project_root, model_path)
    return load_object(model_path)

def load_scaler(scaler_path: str = "models/heart_scaler.pkl") -> Any:
    """Loads the trained StandardScaler."""
    if not os.path.isabs(scaler_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        scaler_path = os.path.join(project_root, scaler_path)
    return load_object(scaler_path)

def prepare_and_scale_input(input_data: Dict[str, Any], scaler: Any) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Converts a dictionary of user input into a pandas DataFrame and scales it.
    """
    logger.info("Preparing and scaling user input data for SHAP explanations.")
    missing_features = [feat for feat in EXPECTED_FEATURES if feat not in input_data]
    if missing_features:
        logger.error(f"Missing expected features: {missing_features}")
        raise ValueError(f"Missing expected features: {missing_features}")

    # Create DataFrame ensuring column order
    df = pd.DataFrame([input_data])[EXPECTED_FEATURES]
    
    try:
        scaled_data = scaler.transform(df)
        return df, scaled_data
    except Exception as e:
        logger.error(f"Error during feature scaling: {e}")
        raise e

def generate_shap_values(model: Any, scaled_data: np.ndarray) -> Tuple[Any, np.ndarray]:
    """
    Initializes the explainer and generates SHAP values.
    Uses TreeExplainer for RandomForestClassifier natively.
    """
    logger.info("Generating SHAP values.")
    from sklearn.ensemble import RandomForestClassifier
    
    try:
        if isinstance(model, RandomForestClassifier):
            logger.info("Using shap.TreeExplainer for RandomForestClassifier.")
            explainer = shap.TreeExplainer(model)
            shap_values_raw = explainer.shap_values(scaled_data)
            # Extract SHAP values for positive class
            if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2:
                shap_values = shap_values_raw[1]
            elif len(shap_values_raw.shape) == 3:
                shap_values = shap_values_raw[:, :, 1]
            else:
                shap_values = shap_values_raw
        else:
            from sklearn.linear_model import LogisticRegression
            if isinstance(model, LogisticRegression):
                logger.info("Using shap.LinearExplainer for LogisticRegression with mean background.")
                # Since data is standard scaled, np.zeros represents the exact training mean
                background = np.zeros((1, scaled_data.shape[1]))
                explainer = shap.LinearExplainer(model, background)
                shap_values_raw = explainer.shap_values(scaled_data)
                shap_values = shap_values_raw
            else:
                logger.info("Using generic shap.KernelExplainer as fallback.")
                background = np.zeros((1, scaled_data.shape[1]))
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values_raw = explainer.shap_values(scaled_data)
                if isinstance(shap_values_raw, list) and len(shap_values_raw) > 1:
                    shap_values = shap_values_raw[1]
                else:
                    shap_values = shap_values_raw
                    
        return explainer, shap_values
    except Exception as e:
        logger.error(f"Error generating SHAP values: {e}")
        raise e

def extract_feature_contributions(shap_values: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    """Extracts feature importance scores for a single prediction."""
    logger.info("Extracting feature contributions.")
    vals = shap_values[0] if len(shap_values.shape) == 2 else shap_values
    return {feature_names[i]: float(vals[i]) for i in range(len(feature_names))}

def extract_top_contributors(importance_dict: Dict[str, float], top_n: int = 3) -> Dict[str, float]:
    """Extracts the top N most impactful features."""
    logger.info(f"Extracting top {top_n} contributors.")
    sorted_items = sorted(importance_dict.items(), key=lambda item: abs(item[1]), reverse=True)
    return dict(sorted_items[:top_n])

def create_shap_plots(explainer: Any, shap_values: np.ndarray, df_input: pd.DataFrame) -> Tuple[plt.Figure, plt.Figure]:
    """
    Creates SHAP bar and waterfall plots using Matplotlib.
    Ensures safe compatibility with Streamlit by passing figures.
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

        # Set compatible matplotlib colormap to fix 'vlag' issue
        cmap_choice = plt.get_cmap("coolwarm")

        # 1. Bar Plot
        fig_bar = plt.figure(figsize=(8, 4))
        ax_bar = fig_bar.add_subplot(111)
        shap.plots.bar(explanation, show=False, ax=ax_bar)
        
        # 2. Waterfall Plot
        fig_waterfall = plt.figure(figsize=(8, 5))
        ax_waterfall = fig_waterfall.add_subplot(111)
        try:
            shap.plots.waterfall(explanation, show=False)
            # The waterfall plot often overtakes the current figure natively in SHAP
            fig_waterfall = plt.gcf()
        except Exception as plot_err:
            logger.warning(f"Waterfall plot failed, falling back to bar only: {plot_err}")
            fig_waterfall = None

        return fig_bar, fig_waterfall
    except Exception as e:
        logger.error(f"Failed to generate SHAP plots: {e}")
        return None, None

def generate_heart_explanation(
    patient_data: Dict[str, Any],
    model_path: str = "models/heart_model.pkl",
    scaler_path: str = "models/heart_scaler.pkl"
) -> Dict[str, Any]:
    """
    Orchestrates the entire SHAP explainability workflow for Heart Disease.
    
    Returns:
        Dict[str, Any]: Contains importance dict, top contributors, SHAP values, and plots.
    """
    logger.info("Initializing complete SHAP explanation workflow for Heart Disease.")
    
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)
    
    df_input, scaled_data = prepare_and_scale_input(patient_data, scaler)
    
    explainer, shap_values = generate_shap_values(model, scaled_data)
    
    importance_dict = extract_feature_contributions(shap_values, EXPECTED_FEATURES)
    top_contributors = extract_top_contributors(importance_dict, top_n=3)
    
    fig_bar, fig_waterfall = create_shap_plots(explainer, shap_values, df_input)
    
    logger.info("SHAP explanation generated successfully.")
    
    return {
        "feature_importance": importance_dict,
        "top_contributors": top_contributors,
        "shap_values": shap_values,
        "plots": {
            "bar_plot": fig_bar,
            "waterfall_plot": fig_waterfall
        }
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
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
    
    print("--- Testing Heart Disease SHAP Explainability Engine ---")
    try:
        results = generate_heart_explanation(sample_patient)
        print("\n✅ SHAP Explanation Successful!")
        print("Top 3 Contributors:")
        for feat, score in results["top_contributors"].items():
            direction = "Increased Risk" if score > 0 else "Decreased Risk"
            print(f"  - {feat}: {score:.4f} ({direction})")
    except Exception as ex:
        print(f"❌ SHAP Test Failed: {ex}")
