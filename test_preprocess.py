"""
Test script to run and verify the preprocessing pipeline.
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from utils.diabetes.preprocess import preprocess_pipeline

def main():
    data_path = "data/diabetes.csv"
    
    try:
        logger.info(f"Attempting to run preprocessing pipeline using '{data_path}'...")
        
        X_train_scaled, X_test_scaled, y_train, y_test = preprocess_pipeline(
            file_path=data_path
        )
        
        print("\n--- Preprocessing Results ---")
        print(f"X_train shape: {X_train_scaled.shape}")
        print(f"X_test shape: {X_test_scaled.shape}")
        print("✅ Preprocessing pipeline executed successfully!")
        print("-----------------------------\n")
        
    except FileNotFoundError:
        logger.error(f"Failed to find the dataset at {data_path}. Please ensure the file exists.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during preprocessing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
