"""
Production-quality SQLite database initialization module for AegisHealth AI Platform.
Handles secure connection management, table initialization, and robust query execution.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, Any, List, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Database configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "database"
DB_PATH = DB_DIR / "healthcare.db"

def get_connection() -> sqlite3.Connection:
    """
    Establishes and returns a secure connection to the SQLite database.
    Automatically creates the database directory if it does not exist.
    
    Returns:
        sqlite3.Connection: Active database connection object.
        
    Raises:
        sqlite3.Error: If the connection fails.
    """
    try:
        # Ensure the directory exists
        DB_DIR.mkdir(parents=True, exist_ok=True)
        
        # Establish connection
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Enables column access by name
        
        # Enforce foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON;")
        
        return conn
    except sqlite3.Error as e:
        logger.error(f"Failed to connect to database at {DB_PATH}: {e}")
        raise e

def initialize_database() -> None:
    """
    Initializes the database schema by creating required tables if they do not exist.
    """
    logger.info("Initializing healthcare database schema...")
    
    users_table_query = """
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    screenings_table_query = """
    CREATE TABLE IF NOT EXISTS screenings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        assessment_type TEXT NOT NULL,
        risk_level TEXT NOT NULL,
        probability REAL NOT NULL,
        report_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """
    
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(users_table_query)
        logger.info("Validated/Created 'users' table.")
        
        cursor.execute(screenings_table_query)
        logger.info("Validated/Created 'screenings' table.")
        
        conn.commit()
        logger.info("Database schema initialization completed successfully.")
        
    except sqlite3.Error as e:
        logger.error(f"Failed to initialize database schema: {e}")
        if conn:
            conn.rollback()
        raise e
    finally:
        close_connection(conn)

def execute_query(query: str, parameters: tuple = ()) -> List[sqlite3.Row]:
    """
    Executes a given SQL query safely with parameter binding.
    Handles both READ and WRITE operations.
    
    Args:
        query (str): The SQL query string.
        parameters (tuple, optional): Tuple of parameters to bind to the query to prevent SQL injection.
        
    Returns:
        List[sqlite3.Row]: List of fetched rows if applicable.
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(query, parameters)
        
        # Determine if it's a read or write operation
        if query.strip().upper().startswith(("SELECT", "PRAGMA")):
            results = cursor.fetchall()
            return results
        else:
            conn.commit()
            return []
            
    except sqlite3.Error as e:
        logger.error(f"Error executing query [{query}]: {e}")
        if conn:
            conn.rollback()
        raise e
    finally:
        close_connection(conn)

def close_connection(conn: Optional[sqlite3.Connection]) -> None:
    """
    Safely closes the database connection.
    
    Args:
        conn (Optional[sqlite3.Connection]): The connection object to close.
    """
    if conn:
        try:
            conn.close()
            logger.debug("Database connection closed cleanly.")
        except sqlite3.Error as e:
            logger.error(f"Error closing database connection: {e}")

if __name__ == "__main__":
    # Standalone execution to build the database schema
    print("--- AegisHealth Database Initialization ---")
    try:
        initialize_database()
        print(f"✅ Database successfully initialized at: {DB_PATH}")
    except Exception as ex:
        print(f"❌ Database initialization failed: {ex}")
