"""
Production-quality history management module for AegisHealth AI Platform.
Handles secure saving and retrieving of patient screening records,
returning structured telemetry for downstream Pandas analytics.
"""

import logging
import sqlite3
import pandas as pd
from typing import Optional, List, Dict, Any

# Assuming db.py is located in the same directory
try:
    from database.db import execute_query
except ImportError:
    # Fallback if running directly from within the database directory
    from db import execute_query

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def save_screening(user_id: int, assessment_type: str, risk_level: str, probability: float, report_path: Optional[str] = None) -> bool:
    """
    Saves a new patient screening result into the database.
    
    Args:
        user_id (int): The ID of the authenticated user.
        assessment_type (str): Type of assessment (e.g., 'Diabetes Risk Assessment').
        risk_level (str): Calculated risk stratification (e.g., 'LOW', 'MEDIUM', 'HIGH').
        probability (float): Computed probability percentage.
        report_path (Optional[str]): Absolute or relative path to the generated PDF report.
        
    Returns:
        bool: True if the screening was saved successfully, False otherwise.
    """
    logger.info(f"Saving new '{assessment_type}' screening for user_id: {user_id}")
    
    query = """
    INSERT INTO screenings (user_id, assessment_type, risk_level, probability, report_path)
    VALUES (?, ?, ?, ?, ?);
    """
    
    try:
        execute_query(query, (user_id, assessment_type, risk_level, probability, report_path))
        logger.info(f"Screening saved successfully for user_id: {user_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to save screening for user_id {user_id}: {e}")
        return False

def _fetch_history_as_dataframe(query: str, parameters: tuple) -> pd.DataFrame:
    """
    Internal helper to fetch data and cleanly map it into a Pandas DataFrame.
    """
    try:
        results = execute_query(query, parameters)
        
        if not results:
            # Return empty DataFrame with expected columns if no results found
            return pd.DataFrame(columns=[
                "id", "user_id", "assessment_type", "risk_level", "probability", "report_path", "created_at"
            ])
            
        # Convert list of sqlite3.Row to list of dicts, then to DataFrame
        records = [dict(row) for row in results]
        df = pd.DataFrame(records)
        
        # Ensure chronological sorting (newest first by default)
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"])
            df = df.sort_values(by="created_at", ascending=False).reset_index(drop=True)
            
        return df
    except Exception as e:
        logger.error(f"Failed to fetch history dataframe: {e}")
        return pd.DataFrame()

def get_user_history(user_id: int, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Retrieves the complete screening history for a given user.
    
    Args:
        user_id (int): The ID of the authenticated user.
        limit (Optional[int]): Maximum number of recent records to return.
        
    Returns:
        pd.DataFrame: A DataFrame containing the user's chronological screening history.
    """
    logger.info(f"Retrieving screening history for user_id: {user_id}")
    
    query = """
    SELECT id, user_id, assessment_type, risk_level, probability, report_path, created_at
    FROM screenings
    WHERE user_id = ?
    ORDER BY created_at DESC
    """
    parameters = [user_id]
    
    if limit is not None:
        query += " LIMIT ?"
        parameters.append(limit)
        
    return _fetch_history_as_dataframe(query, tuple(parameters))

def get_history_by_type(user_id: int, assessment_type: str) -> pd.DataFrame:
    """
    Retrieves the screening history filtered by a specific clinical assessment type.
    Useful for visualizing isolated trends over time.
    
    Args:
        user_id (int): The ID of the authenticated user.
        assessment_type (str): Type of assessment (e.g., 'Diabetes Risk Assessment').
        
    Returns:
        pd.DataFrame: A DataFrame containing the filtered chronological screening history.
    """
    logger.info(f"Retrieving '{assessment_type}' history for user_id: {user_id}")
    
    query = """
    SELECT id, user_id, assessment_type, risk_level, probability, report_path, created_at
    FROM screenings
    WHERE user_id = ? AND assessment_type = ?
    ORDER BY created_at ASC
    """
    # Note: Using ASC sorting here inherently pre-formats the data for chronological trend plotting
    
    return _fetch_history_as_dataframe(query, (user_id, assessment_type))

def get_recent_screenings(user_id: int, days: int = 30) -> pd.DataFrame:
    """
    Retrieves screening records generated within a recent rolling window of days.
    
    Args:
        user_id (int): The ID of the authenticated user.
        days (int): The number of recent days to look back.
        
    Returns:
        pd.DataFrame: A DataFrame containing recent screening history.
    """
    logger.info(f"Retrieving screenings for user_id: {user_id} over the last {days} days")
    
    query = """
    SELECT id, user_id, assessment_type, risk_level, probability, report_path, created_at
    FROM screenings
    WHERE user_id = ? AND created_at >= date('now', ?)
    ORDER BY created_at DESC
    """
    date_modifier = f"-{days} days"
    
    return _fetch_history_as_dataframe(query, (user_id, date_modifier))

if __name__ == "__main__":
    import time
    
    print("--- Testing AegisHealth Screening History Module ---")
    
    TEST_USER_ID = 1  # Assumes user ID 1 exists from previous auth tests
    
    print("\n[1] Testing Save Screening (Diabetes)...")
    success_dia = save_screening(
        user_id=TEST_USER_ID,
        assessment_type="Diabetes Risk Assessment",
        risk_level="MEDIUM",
        probability=55.4,
        report_path="/reports/fake_diabetes_report.pdf"
    )
    if success_dia:
        print("✅ Diabetes screening saved successfully.")
        
    # Brief pause to ensure distinct timestamps
    time.sleep(1)
        
    print("\n[2] Testing Save Screening (Heart)...")
    success_hrt = save_screening(
        user_id=TEST_USER_ID,
        assessment_type="Heart Disease Risk Assessment",
        risk_level="LOW",
        probability=22.1,
        report_path="/reports/fake_heart_report.pdf"
    )
    if success_hrt:
        print("✅ Heart Disease screening saved successfully.")
        
    print("\n[3] Testing Retrieving User History (All)...")
    df_all = get_user_history(user_id=TEST_USER_ID)
    if not df_all.empty:
        print(f"✅ Retrieved {len(df_all)} records.")
        print(df_all[["assessment_type", "probability", "created_at"]].head())
    else:
        print("❌ Failed to retrieve full user history.")
        
    print("\n[4] Testing Filtering by Assessment Type...")
    df_dia = get_history_by_type(user_id=TEST_USER_ID, assessment_type="Diabetes Risk Assessment")
    if not df_dia.empty:
        print(f"✅ Retrieved {len(df_dia)} Diabetes records sorted for trend analysis.")
        print(df_dia[["probability", "created_at"]].head())
    else:
        print("❌ Failed to filter by assessment type.")
        
    print("\n[5] Testing Recent Screenings...")
    df_recent = get_recent_screenings(user_id=TEST_USER_ID, days=7)
    if not df_recent.empty:
        print(f"✅ Retrieved {len(df_recent)} records from the last 7 days.")
    else:
        print("❌ Failed to retrieve recent records.")
