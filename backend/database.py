"""
Database configuration and models for trending claims feature
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from datetime import datetime
from typing import Optional
import os
from config import get_settings

settings = get_settings()

# Database URL from environment or config
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./debunker.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=settings.database_echo if hasattr(settings, 'database_echo') else False,
    pool_size=settings.database_pool_size if hasattr(settings, 'database_pool_size') else 5
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

class TrendingClaim(Base):
    """Model for trending claims discovered from news/social media"""
    __tablename__ = "trending_claims"
    
    id = Column(Integer, primary_key=True, index=True)
    claim_text = Column(Text, nullable=False, index=True)
    title = Column(String(500), nullable=False)  # Headline/title for display
    category = Column(String(100), nullable=False, index=True)  # Politics, Health, Science, etc.
    
    # Fact-check results
    verdict = Column(String(50), nullable=True)  # True/False/Mixed/Unverifiable
    confidence = Column(Float, nullable=True)  # 0.0 to 1.0
    explanation = Column(Text, nullable=True)
    evidence_summary = Column(Text, nullable=True)
    
    # Source and discovery info
    source_type = Column(String(50), nullable=False)  # news, reddit, twitter, etc.
    source_url = Column(String(1000), nullable=True)
    discovered_at = Column(DateTime, default=datetime.utcnow, index=True)
    processed_at = Column(DateTime, nullable=True)
    
    # Engagement and trending metrics
    trending_score = Column(Float, default=0.0, index=True)
    view_count = Column(Integer, default=0)
    share_count = Column(Integer, default=0)
    controversy_level = Column(Float, default=0.0)  # How debatable/controversial
    
    # Processing status
    status = Column(String(50), default="discovered", index=True)  # discovered, processing, completed, failed
    processing_time = Column(Float, nullable=True)  # Time taken to fact-check
    
    # Metadata
    tags = Column(JSON, nullable=True)  # List of relevant tags
    keywords = Column(JSON, nullable=True)  # Extracted keywords
    related_entities = Column(JSON, nullable=True)  # People, organizations mentioned
    
    # Relationships
    sources = relationship("ClaimSource", back_populates="claim", cascade="all, delete-orphan")
    analytics = relationship("ClaimAnalytics", back_populates="claim", cascade="all, delete-orphan")

class ClaimSource(Base):
    """Model for storing source information about claims"""
    __tablename__ = "claim_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(Integer, ForeignKey("trending_claims.id"), nullable=False)
    
    # Source details
    source_name = Column(String(200), nullable=False)  # CNN, Reddit, Twitter, etc.
    source_url = Column(String(1000), nullable=True)
    source_type = Column(String(50), nullable=False)  # news, social_media, fact_check
    author = Column(String(200), nullable=True)
    published_at = Column(DateTime, nullable=True)
    
    # Content
    original_content = Column(Text, nullable=True)  # Original article/post content
    extracted_claim = Column(Text, nullable=True)  # Specific claim extracted
    
    # Social metrics (for social media sources)
    engagement_count = Column(Integer, default=0)  # likes, shares, retweets
    comment_count = Column(Integer, default=0)
    viral_score = Column(Float, default=0.0)
    
    # Reliability
    source_reliability = Column(Float, nullable=True)  # 0.0 to 1.0
    bias_score = Column(Float, nullable=True)  # -1.0 to 1.0 (left to right)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    claim = relationship("TrendingClaim", back_populates="sources")

class ClaimAnalytics(Base):
    """Model for tracking analytics and engagement for claims"""
    __tablename__ = "claim_analytics"
    
    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(Integer, ForeignKey("trending_claims.id"), nullable=False)
    
    # Daily metrics
    date = Column(DateTime, default=datetime.utcnow, index=True)
    daily_views = Column(Integer, default=0)
    daily_shares = Column(Integer, default=0)
    daily_clicks = Column(Integer, default=0)
    
    # Engagement tracking
    time_on_page = Column(Float, default=0.0)  # Average time spent reading
    bounce_rate = Column(Float, default=0.0)  # Percentage who leave immediately
    
    # Social media metrics
    social_mentions = Column(Integer, default=0)
    social_sentiment = Column(Float, default=0.0)  # -1.0 to 1.0
    
    # Search and discovery
    search_rank = Column(Integer, nullable=True)  # Position in trending list
    discovery_source = Column(String(100), nullable=True)  # How users found this claim
    
    # Relationships
    claim = relationship("TrendingClaim", back_populates="analytics")

class NewsSource(Base):
    """Model for managing news sources and their reliability"""
    __tablename__ = "news_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False, unique=True)
    domain = Column(String(200), nullable=False, unique=True)
    source_type = Column(String(50), nullable=False)  # news, blog, social_media
    
    # Reliability metrics
    reliability_score = Column(Float, default=0.5)  # 0.0 to 1.0
    bias_score = Column(Float, default=0.0)  # -1.0 to 1.0
    fact_check_rating = Column(String(50), nullable=True)  # High, Medium, Low
    
    # Configuration
    api_endpoint = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=True)
    update_frequency = Column(Integer, default=60)  # Minutes between updates
    last_updated = Column(DateTime, nullable=True)
    
    # Metadata
    description = Column(Text, nullable=True)
    country = Column(String(100), nullable=True)
    language = Column(String(10), default="en")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Database dependency for FastAPI
def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create tables
def create_tables():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)

# Initialize database
def init_db():
    """Initialize database with default data"""
    create_tables()
    
    # Add default news sources
    db = SessionLocal()
    try:
        # Check if sources already exist
        if db.query(NewsSource).count() == 0:
            default_sources = [
                NewsSource(
                    name="Reuters",
                    domain="reuters.com",
                    source_type="news",
                    reliability_score=0.9,
                    bias_score=0.0,
                    fact_check_rating="High"
                ),
                NewsSource(
                    name="BBC News",
                    domain="bbc.com",
                    source_type="news",
                    reliability_score=0.85,
                    bias_score=0.0,
                    fact_check_rating="High"
                ),
                NewsSource(
                    name="Associated Press",
                    domain="apnews.com",
                    source_type="news",
                    reliability_score=0.9,
                    bias_score=0.0,
                    fact_check_rating="High"
                ),
                NewsSource(
                    name="Reddit",
                    domain="reddit.com",
                    source_type="social_media",
                    reliability_score=0.3,
                    bias_score=0.0,
                    fact_check_rating="Low"
                )
            ]
            
            for source in default_sources:
                db.add(source)
            
            db.commit()
            print("✅ Default news sources added to database")
    
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_db()
    print("✅ Database initialized successfully")