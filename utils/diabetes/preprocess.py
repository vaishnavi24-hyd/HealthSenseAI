"""
Preprocessing pipeline for the Pima Indians Diabetes Dataset.
Handles loading, cleaning, imputation, scaling, and splitting of data.
"""

import os
import logging
from typing import Tuple, Optional, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
TARGET_COLUMN = "Outcome"
COLUMNS_WITH_INVALID_ZEROS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the file is empty.
    """
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully with shape {df.shape}")
        return df
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise e
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e

def replace_invalid_zeros(
    df: pd.DataFrame, 
    columns: List[str], 
    strategy: str = "median"
) -> pd.DataFrame:
    """
    Replace invalid zero values in specified columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (List[str]): List of column names to process.
        strategy (str): Imputation strategy (currently supports 'median').

    Returns:
        pd.DataFrame: DataFrame with invalid zeros replaced.
    """
    df_processed = df.copy()
    logger.info(f"Replacing invalid zeros in columns: {columns} using {strategy} strategy.")
    
    for col in columns:
        if col not in df_processed.columns:
            logger.warning(f"Column '{col}' not found in DataFrame. Skipping.")
            continue
            
        # Temporarily replace 0 with NaN for calculation
        df_processed[col] = df_processed[col].replace(0, np.nan)
        
        if strategy == "median":
            fill_value = df_processed[col].median()
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
            
        df_processed[col] = df_processed[col].fillna(fill_value)
        logger.debug(f"Replaced zeros in '{col}' with {fill_value:.2f}")

    return df_processed

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle any remaining missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    logger.info("Checking for missing values.")
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        logger.info(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
        # Drop rows with missing values as a fallback
        df = df.dropna()
        logger.info(f"Dropped rows with missing values. New shape: {df.shape}")
    else:
        logger.info("No missing values found.")
    return df

def split_data(
    df: pd.DataFrame, 
    target_col: str = TARGET_COLUMN, 
    test_size: float = DEFAULT_TEST_SIZE, 
    random_state: int = DEFAULT_RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Name of the target column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    logger.info(f"Splitting data into train and test sets (test_size={test_size}).")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def scale_features(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale features using StandardScaler and optionally save the scaler.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        save_path (Optional[str]): Path to save the trained scaler.

    Returns:
        Tuple: X_train_scaled, X_test_scaled, scaler object
    """
    logger.info("Scaling features using StandardScaler.")
    scaler = StandardScaler()
    
    # Fit on training data and transform
    X_train_scaled = scaler.fit_transform(X_train)
    # Transform test data
    X_test_scaled = scaler.transform(X_test)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(scaler, save_path)
        logger.info(f"Scaler saved to {save_path}")
        
    return X_train_scaled, X_test_scaled, scaler

def preprocess_pipeline(
    file_path: str, 
    scaler_save_path: str = "models/diabetes_scaler.pkl",
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Complete end-to-end preprocessing pipeline for the Diabetes dataset.

    Args:
        file_path (str): Path to the raw CSV data.
        scaler_save_path (str): Path to save the trained StandardScaler.
        test_size (float): Proportion of data for testing.
        random_state (int): Seed for random operations.

    Returns:
        Tuple containing:
            - X_train_scaled (np.ndarray): Scaled training features.
            - X_test_scaled (np.ndarray): Scaled testing features.
            - y_train (pd.Series): Training targets.
            - y_test (pd.Series): Testing targets.
    """
    logger.info("Starting preprocessing pipeline.")
    
    try:
        # 1. Load Data
        df = load_data(file_path)
        
        # 2. Handle invalid zeros
        df_cleaned = replace_invalid_zeros(
            df, 
            columns=COLUMNS_WITH_INVALID_ZEROS, 
            strategy="median"
        )
        
        # 3. Handle missing values
        df_cleaned = handle_missing_values(df_cleaned)
        
        # 4. Split data
        X_train, X_test, y_train, y_test = split_data(
            df_cleaned, 
            target_col=TARGET_COLUMN,
            test_size=test_size,
            random_state=random_state
        )
        
        # 5. Scale features & save scaler
        X_train_scaled, X_test_scaled, scaler = scale_features(
            X_train, 
            X_test, 
            save_path=scaler_save_path
        )
        
        logger.info("Preprocessing pipeline completed successfully.")
        return X_train_scaled, X_test_scaled, y_train, y_test
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise e

if __name__ == "__main__":
    # Example usage
    sample_data_path = "data/diabetes.csv"
    if os.path.exists(sample_data_path):
        try:
            preprocess_pipeline(sample_data_path)
        except Exception as e:
            logger.warning(f"Pipeline execution encountered an error (likely due to empty data): {e}")
    else:
        logger.warning(f"Data not found at {sample_data_path}. Please place diabetes.csv there.")
