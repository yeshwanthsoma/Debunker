"""
Configuration management for TruthLens Fact Checker
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Basic app settings
    environment: str = Field(default="development")
    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")
    secret_key: str = Field(default="dev-secret-key")
    
    # API Authentication
    api_username: str = Field(default="admin")
    api_password: str = Field(default="secure_password_change_in_production")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8080)
    api_timeout: int = Field(default=120)
    
    # Fact-Checking APIs
    google_fact_check_api_key: Optional[str] = Field(default=None)
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    news_api_key: Optional[str] = Field(default=None)
    
    # News Aggregation APIs
    reddit_client_id: Optional[str] = Field(default=None)
    reddit_secret: Optional[str] = Field(default=None)
    twitter_bearer_token: Optional[str] = Field(default=None)
    grok_api_key: Optional[str] = Field(default=None)
    
    # OpenAI Configuration
    openai_model: str = Field(default="gpt-4")
    openai_max_tokens: int = Field(default=2000)
    openai_temperature: float = Field(default=0.1)
    
    # Model Settings
    whisper_model: str = Field(default="openai/whisper-small")
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    sentiment_model: str = Field(default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    
    # File Upload Settings
    max_file_size: int = Field(default=52428800)  # 50MB
    allowed_audio_formats: list = Field(default=[
        "audio/mp3", "audio/mpeg", "audio/wav", 
        "audio/m4a", "audio/ogg", "audio/webm"
    ])
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100)
    rate_limit_window: int = Field(default=3600)
    
    # Cache Configuration
    cache_ttl: int = Field(default=3600)
    enable_cache: bool = Field(default=True)
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///./debunker.db")
    database_pool_size: int = Field(default=5)
    database_echo: bool = Field(default=False)
    
    # Railway Environment Detection
    railway_environment: Optional[str] = Field(default=None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Railway environment detection and PostgreSQL configuration
        if self.railway_environment or os.getenv('DATABASE_URL'):
            database_url = os.getenv('DATABASE_URL')
            if database_url:
                if database_url.startswith('postgres://'):
                    # Fix for newer PostgreSQL drivers
                    database_url = database_url.replace('postgres://', 'postgresql://', 1)
                self.database_url = database_url
                print(f"🚀 Railway environment detected - using PostgreSQL")
            else:
                print(f"💻 Local environment detected - using SQLite")
    
    # CORS Settings
    cors_origins: list = Field(default=["*"])
    cors_allow_credentials: bool = Field(default=True)
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"  # Ignore extra fields from .env
    }

@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()

def get_api_key(service: str) -> Optional[str]:
    """Get API key for a specific service"""
    settings = get_settings()
    
    key_mapping = {
        "google_fact_check": settings.google_fact_check_api_key,
        "openai": settings.openai_api_key,
        "anthropic": settings.anthropic_api_key,
        "news_api": settings.news_api_key,
        "reddit_client_id": settings.reddit_client_id,
        "reddit_secret": settings.reddit_secret,
        "twitter_bearer_token": settings.twitter_bearer_token,
        "grok": settings.grok_api_key
    }
    
    return key_mapping.get(service)

def is_api_available(service: str) -> bool:
    """Check if an API service is available (has API key)"""
    api_key = get_api_key(service)
    return api_key is not None and api_key.strip() != "" and not api_key.startswith("your_")
