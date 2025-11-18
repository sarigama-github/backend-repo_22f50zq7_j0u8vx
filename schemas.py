"""
Database Schemas for Stock Pulse Analyzer

We define collections for caching ticker analyses and news articles.
Each Pydantic model corresponds to a collection (lowercased name).
"""

from pydantic import BaseModel, Field
from typing import Optional, List

class Analysis(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, uppercase")
    price: float
    change_percent: float
    support: float
    resistance: float
    rsi: float
    rsi_signal: str
    macd: float
    macd_signal: float
    macd_hist: float
    ma50: float
    ma200: float
    ma_trend: str
    volume: float
    volume_trend: str
    last_updated: Optional[str] = None

class Article(BaseModel):
    ticker: str
    source: str
    title: str
    url: str
    published: Optional[str] = None
    summary: Optional[str] = None
    sentiment: Optional[str] = None
