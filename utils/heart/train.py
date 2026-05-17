"""
Heart Disease Model Training Pipeline
Trains, evaluates, compares, and saves the best machine learning model.
"""

import os
import sys
import logging
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure root directory is accessible for imports if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.heart.preprocess import preprocess_pipeline

# Configure logging
logger = logging.getLogger(__name__)

def train_models(X_train: np.ndarray, y_train: pd.Series) -> Dict[str, Any]:
    """
    Initializes and trains multiple classification models.
    
    Args:
        X_train (np.ndarray): Scaled training features.
        y_train (pd.Series): Training target labels.
        
    Returns:
        Dict[str, Any]: Dictionary of trained model objects.
    """
    logger.info("Initializing model training sequence.")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVC": SVC(probability=True, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
            logger.info(f"Successfully trained {name}.")
        except Exception as e:
            logger.error(f"Failed to train {name}: {e}")
            
    return trained_models

def evaluate_models(models: Dict[str, Any], X_test: np.ndarray, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Evaluates trained models against the test set using standard classification metrics.
    
    Args:
        models (Dict[str, Any]): Dictionary of trained models.
        X_test (np.ndarray): Scaled testing features.
        y_test (pd.Series): Testing target labels.
        
    Returns:
        Dict[str, Dict[str, float]]: Nested dictionary containing evaluation metrics per model.
    """
    logger.info("Evaluating models on test dataset.")
    results = {}
    
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            metrics = {
                "Accuracy": float(accuracy_score(y_test, y_pred)),
                "Precision": float(precision_score(y_test, y_pred, average='binary', zero_division=0)),
                "Recall": float(recall_score(y_test, y_pred, average='binary', zero_division=0)),
                "F1 Score": float(f1_score(y_test, y_pred, average='binary', zero_division=0))
            }
            results[name] = metrics
            logger.debug(f"{name} Evaluation: {metrics}")
        except Exception as e:
            logger.error(f"Error evaluating {name}: {e}")
            
    return results

def select_best_model(results: Dict[str, Dict[str, float]], models: Dict[str, Any]) -> Tuple[str, Any, Dict[str, float]]:
    """
    Selects the best performing model based on accuracy.
    
    Args:
        results (Dict[str, Dict[str, float]]): Evaluation metrics per model.
        models (Dict[str, Any]): Dictionary of trained models.
        
    Returns:
        Tuple[str, Any, Dict[str, float]]: Best model name, the model object, and its metrics.
    """
    logger.info("Selecting the best model based on highest Accuracy.")
    
    best_name = None
    best_score = -1.0
    
    for name, metrics in results.items():
        if metrics["Accuracy"] > best_score:
            best_score = metrics["Accuracy"]
            best_name = name
            
    if not best_name:
        raise ValueError("Failed to select a best model. Metric evaluation may have failed.")
        
    logger.info(f"Best model identified as {best_name} with Accuracy: {best_score:.4f}")
    return best_name, models[best_name], results[best_name]

def save_model(model: Any, filepath: str) -> None:
    """
    Serializes and saves the trained model to disk.
    
    Args:
        model (Any): The trained scikit-learn model.
        filepath (str): Destination path for the model.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        logger.info(f"Model successfully saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save model to {filepath}: {e}")
        raise e

def print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """Prints a formatted ASCII comparison table of model metrics."""
    print("\n" + "="*70)
    print(f"{'Model Performance Comparison':^70}")
    print("="*70)
    print(f"{'Model Name':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 70)
    
    for name, metrics in results.items():
        print(f"{name:<25} | {metrics['Accuracy']:.4f}     | {metrics['Precision']:.4f}     | {metrics['Recall']:.4f}     | {metrics['F1 Score']:.4f}")
    print("="*70 + "\n")

def run_training_pipeline(
    data_path: str = "data/heart.csv",
    model_save_path: str = "models/heart_model.pkl"
) -> Dict[str, Any]:
    """
    Orchestrates the entire ML training sequence from data ingestion to model serialization.
    
    Args:
        data_path (str): Path to raw CSV data.
        model_save_path (str): Destination path for the chosen model.
        
    Returns:
        Dict[str, Any]: Comprehensive results payload containing the best model and metrics.
    """
    logger.info("=== Starting Complete Training Pipeline ===")
    
    # Preprocess Data
    logger.info("Invoking Preprocessing Pipeline...")
    try:
        X_train, X_test, y_train, y_test = preprocess_pipeline(file_path=data_path)
    except Exception as e:
        logger.error(f"Pipeline halted during preprocessing: {e}")
        raise e
        
    # Train Models
    trained_models = train_models(X_train, y_train)
    if not trained_models:
        raise RuntimeError("No models were successfully trained.")
        
    # Evaluate Models
    eval_results = evaluate_models(trained_models, X_test, y_test)
    print_comparison_table(eval_results)
    
    # Select Best Model
    best_name, best_model, best_metrics = select_best_model(eval_results, trained_models)
    
    # Determine safe absolute path if executed remotely
    if not os.path.isabs(model_save_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_save_path = os.path.join(project_root, model_save_path)
        
    # Save Best Model
    save_model(best_model, model_save_path)
    
    logger.info("=== Training Pipeline Completed Successfully ===")
    
    return {
        "best_model_name": best_name,
        "metrics": best_metrics,
        "all_results": eval_results,
        "saved_path": model_save_path
    }

if __name__ == "__main__":
    # Configure root logger for script execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Calculate robust paths relative to this script's location
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_path = os.path.join(project_root, "data", "heart.csv")
    
    try:
        results = run_training_pipeline(data_path=dataset_path)
        print(f"🎉 Pipeline Execution Complete!")
        print(f"🏆 Best Model: {results['best_model_name']} (Accuracy: {results['metrics']['Accuracy']:.4f})")
        print(f"💾 Saved to:   {results['saved_path']}")
    except Exception as e:
        print(f"❌ Critical Failure in Training Pipeline: {e}")
