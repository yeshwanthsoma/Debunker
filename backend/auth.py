"""
Authentication module for TruthLens API
Provides HTTP Basic Authentication for sensitive endpoints
"""

import secrets
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from config import get_settings

# Initialize HTTP Basic Auth
security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Verify HTTP Basic Auth credentials against environment variables
    
    Args:
        credentials: HTTP Basic Auth credentials from request header
        
    Returns:
        True if credentials are valid
        
    Raises:
        HTTPException: If credentials are invalid
    """
    settings = get_settings()
    
    # Get expected credentials from environment
    correct_username = settings.api_username
    correct_password = settings.api_password
    
    # Use secrets.compare_digest to prevent timing attacks
    is_correct_username = secrets.compare_digest(
        credentials.username.encode("utf8"), 
        correct_username.encode("utf8")
    )
    is_correct_password = secrets.compare_digest(
        credentials.password.encode("utf8"), 
        correct_password.encode("utf8")
    )
    
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return True

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Dependency for protected endpoints that require authentication
    
    Args:
        credentials: HTTP Basic Auth credentials
        
    Returns:
        Username if authenticated
    """
    verify_credentials(credentials)
    return credentials.username

# Optional: Create a dependency that returns user info
async def require_auth():
    """
    Simple dependency that just requires valid authentication
    Use this for endpoints that need auth but don't need user info
    """
    def auth_dependency(authenticated: bool = Depends(verify_credentials)):
        return authenticated
    
    return auth_dependency