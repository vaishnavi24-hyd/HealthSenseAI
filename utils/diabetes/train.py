"""
Machine learning training pipeline for the Pima Indians Diabetes Dataset.
Handles model initialization, training, evaluation, selection, and serialization.
"""

import os
import sys
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Add project root to sys.path to allow imports when executed as a script
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.diabetes.preprocess import preprocess_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_models() -> Dict[str, Any]:
    """
    Initialize the machine learning models to be trained.

    Returns:
        Dict[str, Any]: Dictionary mapping model names to scikit-learn model instances.
    """
    return {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "Support Vector Classifier": SVC(probability=True, random_state=42)
    }

def train_models(
    models: Dict[str, Any], 
    X_train: np.ndarray, 
    y_train: pd.Series
) -> Dict[str, Any]:
    """
    Train a dictionary of machine learning models.

    Args:
        models (Dict[str, Any]): Dictionary of uninitialized models.
        X_train (np.ndarray): Scaled training features.
        y_train (pd.Series): Training targets.

    Returns:
        Dict[str, Any]: Dictionary of trained models.
    """
    trained_models = {}
    logger.info("Starting model training process...")
    for name, model in models.items():
        logger.info(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
            logger.info(f"{name} trained successfully.")
        except Exception as e:
            logger.error(f"Failed to train {name}: {e}")
    return trained_models

def evaluate_models(
    trained_models: Dict[str, Any], 
    X_test: np.ndarray, 
    y_test: pd.Series
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate trained models using standard classification metrics.

    Args:
        trained_models (Dict[str, Any]): Dictionary of trained models.
        X_test (np.ndarray): Scaled testing features.
        y_test (pd.Series): Testing targets.

    Returns:
        Dict[str, Dict[str, float]]: Nested dictionary containing evaluation metrics for each model.
    """
    results = {}
    logger.info("Evaluating models on the test set...")
    for name, model in trained_models.items():
        try:
            y_pred = model.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0)
            }
            results[name] = metrics
        except Exception as e:
            logger.error(f"Failed to evaluate {name}: {e}")
    return results

def print_evaluation_table(results: Dict[str, Dict[str, float]]) -> None:
    """
    Print a nicely formatted table comparing model evaluation metrics.

    Args:
        results (Dict[str, Dict[str, float]]): The evaluation results dictionary.
    """
    print("\n" + "="*75)
    print(f"{'Model Name':<28} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1 Score':<8}")
    print("-" * 75)
    for name, metrics in results.items():
        acc = f"{metrics.get('accuracy', 0.0):.4f}"
        prec = f"{metrics.get('precision', 0.0):.4f}"
        rec = f"{metrics.get('recall', 0.0):.4f}"
        f1 = f"{metrics.get('f1_score', 0.0):.4f}"
        print(f"{name:<28} | {acc:<8} | {prec:<9} | {rec:<8} | {f1:<8}")
    print("="*75 + "\n")

def select_best_model(
    trained_models: Dict[str, Any], 
    results: Dict[str, Dict[str, float]], 
    metric: str = "accuracy"
) -> Tuple[str, Any]:
    """
    Select the best model based on a specified evaluation metric.

    Args:
        trained_models (Dict[str, Any]): Dictionary of trained models.
        results (Dict[str, Dict[str, float]]): Dictionary of evaluation metrics.
        metric (str): The metric to use for selection (default: 'accuracy').

    Returns:
        Tuple[str, Any]: The name and instance of the best model.
    """
    logger.info(f"Selecting the best model based on highest {metric}.")
    best_model_name = max(results, key=lambda k: results[k].get(metric, 0.0))
    best_model = trained_models[best_model_name]
    best_score = results[best_model_name].get(metric, 0.0)
    logger.info(f"Best model selected: {best_model_name} with {metric} = {best_score:.4f}")
    return best_model_name, best_model

def save_model(model: Any, filepath: str) -> None:
    """
    Serialize and save the trained model to disk.

    Args:
        model (Any): The trained machine learning model.
        filepath (str): Path to save the model file.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        logger.info(f"Model successfully saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model to {filepath}: {e}")
        raise e

def run_training_pipeline(data_path: str, model_save_path: str) -> None:
    """
    Execute the complete end-to-end model training pipeline.

    Args:
        data_path (str): Path to the raw diabetes dataset CSV.
        model_save_path (str): Path to save the best model.
    """
    logger.info("Initializing the complete training pipeline.")
    
    try:
        # Step 1: Preprocess Data
        logger.info(f"Loading and preprocessing data from {data_path}...")
        X_train_scaled, X_test_scaled, y_train, y_test = preprocess_pipeline(
            file_path=data_path
        )
        
        # Step 2: Initialize Models
        models = get_models()
        
        # Step 3: Train Models
        trained_models = train_models(models, X_train_scaled, y_train)
        
        # Step 4: Evaluate Models
        results = evaluate_models(trained_models, X_test_scaled, y_test)
        
        # Step 5: Print Comparison Table
        print_evaluation_table(results)
        
        # Step 6: Select Best Model
        best_name, best_model = select_best_model(trained_models, results, metric="accuracy")
        
        # Step 7: Save the Best Model
        save_model(best_model, model_save_path)
        logger.info("Training pipeline completed successfully.")
        
    except Exception as e:
        logger.critical(f"Critical error occurred in the training pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Ensure correct behavior by building robust paths
    dataset_path = os.path.join(project_root, "data", "diabetes.csv")
    best_model_path = os.path.join(project_root, "models", "diabetes_model.pkl")
    
    # Execute the pipeline
    run_training_pipeline(
        data_path=dataset_path,
        model_save_path=best_model_path
    )
