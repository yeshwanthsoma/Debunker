"""
Authentication module for Debunker API
Provides HTTP Basic Authentication with two levels:
- Regular auth: rate limited access
- Admin auth: unlimited access, no rate limits
"""

import secrets
from typing import Optional, Tuple
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from config import get_settings

# Initialize HTTP Basic Auth
security = HTTPBasic()
optional_security = HTTPBasic(auto_error=False)


def _check_credentials(username: str, password: str) -> Tuple[bool, bool]:
    """
    Check credentials against both user and admin credentials.

    Returns:
        Tuple of (is_valid, is_admin)
    """
    settings = get_settings()

    # Check admin credentials first
    is_admin_username = secrets.compare_digest(
        username.encode("utf8"), settings.admin_username.encode("utf8")
    )
    is_admin_password = secrets.compare_digest(
        password.encode("utf8"), settings.admin_password.encode("utf8")
    )
    if is_admin_username and is_admin_password:
        return (True, True)

    # Check regular user credentials
    is_user_username = secrets.compare_digest(
        username.encode("utf8"), settings.api_username.encode("utf8")
    )
    is_user_password = secrets.compare_digest(
        password.encode("utf8"), settings.api_password.encode("utf8")
    )
    if is_user_username and is_user_password:
        return (True, False)

    return (False, False)


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> bool:
    """
    Verify HTTP Basic Auth credentials (accepts both user and admin).

    Returns:
        True if credentials are valid

    Raises:
        HTTPException: If credentials are invalid
    """
    is_valid, _ = _check_credentials(credentials.username, credentials.password)

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "x-basic"},
        )

    return True


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)) -> bool:
    """
    Verify admin credentials only. Rejects regular user credentials.

    Returns:
        True if admin credentials are valid

    Raises:
        HTTPException: If credentials are invalid or not admin
    """
    is_valid, is_admin = _check_credentials(credentials.username, credentials.password)

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "x-basic"},
        )

    if not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )

    return True


def get_user_type(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """
    Get user type from credentials.

    Returns:
        "admin" or "user"

    Raises:
        HTTPException: If credentials are invalid
    """
    is_valid, is_admin = _check_credentials(credentials.username, credentials.password)

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "x-basic"},
        )

    return "admin" if is_admin else "user"


def is_admin_request(request: Request) -> bool:
    """
    Check if request has valid admin credentials.
    Used for rate limit exemption.

    Returns:
        True if request has valid admin auth, False otherwise
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Basic "):
        return False

    try:
        import base64
        credentials = base64.b64decode(auth_header[6:]).decode("utf-8")
        username, password = credentials.split(":", 1)
        is_valid, is_admin = _check_credentials(username, password)
        return is_valid and is_admin
    except Exception:
        return False


def get_current_user(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """
    Dependency for protected endpoints that require authentication.

    Returns:
        Username if authenticated
    """
    verify_credentials(credentials)
    return credentials.username