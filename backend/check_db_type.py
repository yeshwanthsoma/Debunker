#!/usr/bin/env python3
"""
Check which database type is being used
"""

import os
from urllib.parse import urlparse
from config import get_settings


def mask_database_url(url: str) -> str:
    """Mask credentials in a database URL, showing only scheme and host."""
    try:
        parsed = urlparse(url)
        if parsed.password:
            return url.replace(f":{parsed.password}@", ":****@")
        return url
    except Exception:
        return "<unparseable URL>"


def get_database_type(url: str) -> str:
    """Determine database type from URL scheme."""
    if not url:
        return "Unknown"
    if url.startswith("postgresql://") or url.startswith("postgresql+"):
        return "PostgreSQL"
    if url.startswith("sqlite:"):
        return "SQLite"
    if url.startswith("mysql://") or url.startswith("mysql+"):
        return "MySQL"
    return "Unknown"

def check_database_config():
    """Check current database configuration"""
    print("🔍 Checking Database Configuration...")
    print("=" * 50)
    
    # Get settings
    settings = get_settings()
    
    # Check environment variables
    database_url = os.getenv('DATABASE_URL')
    railway_env = os.getenv('RAILWAY_ENVIRONMENT')
    
    print(f"📊 Environment Detection:")
    print(f"   DATABASE_URL exists: {'Yes' if database_url else 'No'}")
    print(f"   RAILWAY_ENVIRONMENT: {railway_env or 'Not set'}")
    
    if database_url:
        print(f"   DATABASE_URL type: {get_database_type(database_url)}")
        print(f"   DATABASE_URL (masked): {mask_database_url(database_url)}")

    print(f"\n🗄️ Current Database Configuration:")
    print(f"   Database URL: {mask_database_url(settings.database_url)}")
    print(f"   Database Type: {get_database_type(settings.database_url)}")
    print(f"   Pool Size: {settings.database_pool_size}")
    print(f"   Echo SQL: {settings.database_echo}")
    
    # Additional environment info
    print(f"\n🌐 Environment Info:")
    print(f"   Environment: {settings.environment}")
    print(f"   Debug Mode: {settings.debug}")
    
    return settings.database_url

if __name__ == "__main__":
    db_url = check_database_config()

    if not db_url:
        print("\n❌ No database URL configured")
    elif db_url.startswith('postgresql://') or db_url.startswith('postgresql+'):
        print("\n✅ Using PostgreSQL (Production Database)")
    elif db_url.startswith('sqlite:'):
        print("\n✅ Using SQLite (Local Development Database)")
    else:
        scheme = db_url.split('://')[0] if '://' in db_url else 'unknown'
        print(f"\n❓ Unknown database type: {scheme}")