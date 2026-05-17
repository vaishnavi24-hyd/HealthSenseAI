"""
Production-quality authentication module for AegisHealth AI Platform.
Handles secure user registration, password hashing (bcrypt), and session validation.
"""

import logging
import bcrypt
import sqlite3
from typing import Optional, Dict, Any

# Assuming db.py is located in the same directory (database/db.py)
try:
    from database.db import execute_query
except ImportError:
    # Fallback if running directly from within the database directory
    from db import execute_query

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def hash_password(password: str) -> str:
    """
    Securely hashes a plaintext password using bcrypt.
    
    Args:
        password (str): The plaintext password.
        
    Returns:
        str: The bcrypt hashed password string (decoded to utf-8).
    """
    # Generate salt and hash the password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plaintext password against a stored bcrypt hash.
    
    Args:
        plain_password (str): The plaintext password attempt.
        hashed_password (str): The stored bcrypt hash.
        
    Returns:
        bool: True if password matches, False otherwise.
    """
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def user_exists(username: str, email: str) -> bool:
    """
    Checks if a user already exists with the given username or email.
    
    Args:
        username (str): The username to check.
        email (str): The email to check.
        
    Returns:
        bool: True if user exists, False otherwise.
    """
    query = "SELECT id FROM users WHERE username = ? OR email = ? LIMIT 1;"
    try:
        results = execute_query(query, (username, email))
        return len(results) > 0
    except Exception as e:
        logger.error(f"Error checking if user exists: {e}")
        # Default to True on error to prevent overwrites or leaks during failure
        return True

def signup(username: str, email: str, password: str) -> bool:
    """
    Registers a new user in the database.
    
    Args:
        username (str): Desired username.
        email (str): User's email address.
        password (str): Plaintext password.
        
    Returns:
        bool: True if registration was successful, False otherwise.
    """
    logger.info(f"Attempting to register user: {username}")
    
    if user_exists(username, email):
        logger.warning(f"Registration failed: Username '{username}' or email '{email}' already exists.")
        return False
        
    try:
        hashed_pw = hash_password(password)
        query = """
        INSERT INTO users (username, email, password_hash)
        VALUES (?, ?, ?);
        """
        execute_query(query, (username, email, hashed_pw))
        logger.info(f"User '{username}' registered successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to register user '{username}': {e}")
        return False

def login(identifier: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Authenticates a user via username or email.
    
    Args:
        identifier (str): Username or email address.
        password (str): Plaintext password attempt.
        
    Returns:
        Optional[Dict[str, Any]]: A dictionary containing user info if successful, None otherwise.
    """
    logger.info(f"Authentication attempt for identifier: {identifier}")
    
    query = """
    SELECT id, username, email, password_hash, created_at 
    FROM users 
    WHERE username = ? OR email = ? 
    LIMIT 1;
    """
    
    try:
        results = execute_query(query, (identifier, identifier))
        if not results:
            logger.warning(f"Login failed: No account found for '{identifier}'.")
            return None
            
        user_record = results[0]
        stored_hash = user_record["password_hash"]
        
        if verify_password(password, stored_hash):
            logger.info(f"Login successful for user '{user_record['username']}'.")
            return {
                "id": user_record["id"],
                "username": user_record["username"],
                "email": user_record["email"],
                "created_at": user_record["created_at"]
            }
        else:
            logger.warning(f"Login failed: Incorrect password for '{identifier}'.")
            return None
            
    except Exception as e:
        logger.error(f"Error during login process for '{identifier}': {e}")
        return None

if __name__ == "__main__":
    print("--- Testing AegisHealth Authentication Module ---")
    
    test_user = "dr_smith"
    test_email = "dr.smith@aegishealth.com"
    test_pass = "SecureMedPass123!"
    
    print("\n[1] Testing User Registration...")
    success = signup(test_user, test_email, test_pass)
    if success:
        print(f"✅ User '{test_user}' registered successfully.")
    else:
        print(f"⚠️ User '{test_user}' may already exist or registration failed.")
        
    print("\n[2] Testing Duplicate Registration Prevention...")
    duplicate_success = signup(test_user, test_email, test_pass)
    if not duplicate_success:
        print("✅ Duplicate prevention working correctly.")
    else:
        print("❌ Duplicate registration succeeded (THIS IS A BUG).")
        
    print("\n[3] Testing Valid Login...")
    user_data = login(test_user, test_pass)
    if user_data:
        print(f"✅ Login successful! Retrieved Data: {user_data}")
    else:
        print("❌ Valid login failed.")
        
    print("\n[4] Testing Invalid Login (Wrong Password)...")
    bad_login = login(test_user, "WrongPassword!")
    if not bad_login:
        print("✅ Invalid password correctly rejected.")
    else:
        print("❌ Invalid password incorrectly accepted.")
        
    print("\n[5] Testing Invalid Login (Unknown User)...")
    ghost_login = login("unknown_user", test_pass)
    if not ghost_login:
        print("✅ Unknown user correctly rejected.")
    else:
        print("❌ Unknown user incorrectly accepted.")
