#!/usr/bin/env python3
"""
Check which database type is being used
"""

import os
from config import get_settings

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
        print(f"   DATABASE_URL type: {database_url.split('://')[0] if '://' in database_url else 'Unknown'}")
        # Don't print full URL for security
        print(f"   DATABASE_URL (masked): {database_url[:20]}...{database_url[-10:] if len(database_url) > 30 else database_url}")
    
    print(f"\n🗄️ Current Database Configuration:")
    print(f"   Database URL: {settings.database_url}")
    print(f"   Database Type: {'PostgreSQL' if settings.database_url.startswith('postgresql://') else 'SQLite'}")
    print(f"   Pool Size: {settings.database_pool_size}")
    print(f"   Echo SQL: {settings.database_echo}")
    
    # Additional environment info
    print(f"\n🌐 Environment Info:")
    print(f"   Environment: {settings.environment}")
    print(f"   Debug Mode: {settings.debug}")
    
    return settings.database_url

if __name__ == "__main__":
    db_url = check_database_config()
    
    if db_url.startswith('postgresql://'):
        print(f"\n✅ Using PostgreSQL (Production Database)")
    elif db_url.startswith('sqlite:'):
        print(f"\n✅ Using SQLite (Local Development Database)")
    else:
        print(f"\n❓ Unknown database type: {db_url}")