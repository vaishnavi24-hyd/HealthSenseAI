"""
Heart Disease Preprocessing Pipeline
Handles data loading, cleaning, splitting, and scaling for the heart disease dataset.
"""

import os
import logging
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
        
    Raises:
        FileNotFoundError: If the CSV file does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
    """
    logger.info(f"Loading dataset from {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"Dataset not found at {file_path}")
        raise FileNotFoundError(f"Dataset not found at {file_path}")
        
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded dataset with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise e

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset by handling missing values and duplicates.
    
    Args:
        df (pd.DataFrame): Raw dataset.
        
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    logger.info("Cleaning data...")
    df_clean = df.copy()
    
    # Drop pure duplicates if any
    initial_shape = df_clean.shape
    df_clean.drop_duplicates(inplace=True)
    if df_clean.shape[0] < initial_shape[0]:
        logger.info(f"Dropped {initial_shape[0] - df_clean.shape[0]} duplicate rows.")
        
    # Handle missing values (if any) with median imputation
    missing_cols = df_clean.columns[df_clean.isnull().any()].tolist()
    if missing_cols:
        logger.info(f"Found missing values in columns: {missing_cols}. Imputing with median.")
        for col in missing_cols:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    else:
        logger.info("No missing values detected.")
        
    return df_clean

def split_features_target(df: pd.DataFrame, target_column: str = 'target') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits the dataset into features (X) and target (y).
    
    Args:
        df (pd.DataFrame): The dataset.
        target_column (str): The name of the target column.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and Target (y).
    """
    logger.info(f"Splitting features and target. Target column: {target_column}")
    if target_column not in df.columns:
        logger.error(f"Target column '{target_column}' not found in dataset.")
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
        
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, scaler_save_path: str) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scales the features using StandardScaler and saves the scaler.
    
    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        scaler_save_path (str): Path to save the trained scaler.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, StandardScaler]: Scaled X_train, scaled X_test, and the scaler object.
    """
    logger.info("Scaling features using StandardScaler.")
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
    
    # Save scaler
    joblib.dump(scaler, scaler_save_path)
    logger.info(f"Scaler saved to {scaler_save_path}")
    
    return X_train_scaled, X_test_scaled, scaler

def preprocess_pipeline(
    file_path: str, 
    scaler_save_path: str = "models/heart_scaler.pkl",
    target_column: str = "target"
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Executes the complete data preprocessing pipeline.
    
    Args:
        file_path (str): Path to the raw CSV dataset.
        scaler_save_path (str): Path to save the fitted StandardScaler.
        target_column (str): Name of the target column.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]: Processed X_train, X_test, y_train, y_test.
    """
    logger.info("--- Starting Heart Disease Preprocessing Pipeline ---")
    
    # Step 1: Load Data
    df = load_data(file_path)
    
    # Step 2: Clean Data
    df = clean_data(df)
    
    # Step 3: Split Features and Target
    X, y = split_features_target(df, target_column)
    
    # Step 4: Train-Test Split (stratified)
    logger.info("Splitting data into train and test sets (test_size=0.2, random_state=42)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Step 5: Scale Features
    # Determine correct path relative to project root if executed from subfolder
    if not os.path.isabs(scaler_save_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        scaler_save_path = os.path.join(project_root, scaler_save_path)
        
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test, scaler_save_path)
    
    logger.info("--- Preprocessing Pipeline Completed Successfully ---")
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # Test block
    logging.basicConfig(level=logging.INFO)
    
    # Determine paths based on test execution
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(project_root, "data", "heart.csv")
    scaler_path = os.path.join(project_root, "models", "heart_scaler.pkl")
    
    try:
        X_train, X_test, y_train, y_test = preprocess_pipeline(
            file_path=data_path,
            scaler_save_path=scaler_path
        )
        print("Test Pipeline Successful.")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
    except Exception as ex:
        print(f"Pipeline Test Failed: {ex}")
