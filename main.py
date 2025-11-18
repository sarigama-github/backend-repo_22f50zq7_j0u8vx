import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import feedparser
import requests
from bs4 import BeautifulSoup

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import db, create_document, get_documents

app = FastAPI(title="Stock Pulse Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------- Utility & Indicators ------------------------- #

def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

def compute_macd(series: pd.Series) -> Dict[str, float]:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return {"macd": float(macd.iloc[-1]), "signal": float(signal.iloc[-1]), "hist": float(hist.iloc[-1])}

def support_resistance(close: pd.Series) -> Dict[str, float]:
    # Simple heuristic: support = recent low, resistance = recent high over 30 periods
    recent = close.tail(30)
    return {"support": float(recent.min()), "resistance": float(recent.max())}

def trend_from_ma(ma_short: float, ma_long: float) -> str:
    if ma_short > ma_long * 1.01:
        return "Bullish"
    if ma_short < ma_long * 0.99:
        return "Bearish"
    return "Neutral"

def interpret_rsi(rsi: float) -> str:
    if rsi >= 70:
        return "Overbought"
    if rsi <= 30:
        return "Oversold"
    if rsi > 55:
        return "Bullish Momentum"
    if rsi < 45:
        return "Bearish Momentum"
    return "Neutral"

def volume_trend(vol_series: pd.Series) -> str:
    avg20 = vol_series.tail(20).mean()
    last = vol_series.iloc[-1]
    if last > 1.3 * avg20:
        return "High"
    if last < 0.7 * avg20:
        return "Low"
    return "Normal"


# ---------------------------- Models ---------------------------- #

class IndicatorResponse(BaseModel):
    ticker: str
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
    as_of: str
    chart: List[float]
    volume_series: List[float]

class NewsItem(BaseModel):
    source: str
    title: str
    url: str
    published: Optional[str] = None
    summary: Optional[str] = None

class AnalyzeResponse(BaseModel):
    indicators: IndicatorResponse
    news: List[NewsItem]


# ------------------------- Price & Indicators ------------------------- #

@app.get("/api/analyze", response_model=IndicatorResponse)
def analyze_ticker(ticker: str = Query(..., min_length=1, max_length=10)):
    t = ticker.upper().strip()
    try:
        # Pull sufficient history for indicators
        hist = yf.download(t, period="6mo", interval="1d", auto_adjust=True, progress=False)
        if hist is None or hist.empty:
            raise HTTPException(status_code=404, detail="No data for ticker")
        close = hist["Close"].dropna()
        volume = hist["Volume"].fillna(0)
        price = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) > 1 else price
        change_pct = ((price - prev) / prev) * 100 if prev else 0.0

        rsi_val = compute_rsi(close, 14)
        macd_vals = compute_macd(close)
        ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else float(close.mean())
        ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else float(close.mean())
        sr = support_resistance(close)
        vol_trend = volume_trend(volume)
        as_of = datetime.utcnow().isoformat() + "Z"

        chart_series = [float(x) for x in close.tail(90).to_list()]
        vol_series = [float(x) for x in volume.tail(90).to_list()]

        data = IndicatorResponse(
            ticker=t,
            price=price,
            change_percent=round(change_pct, 2),
            support=round(sr["support"], 2),
            resistance=round(sr["resistance"], 2),
            rsi=round(rsi_val, 2),
            rsi_signal=interpret_rsi(rsi_val),
            macd=round(macd_vals["macd"], 4),
            macd_signal=round(macd_vals["signal"], 4),
            macd_hist=round(macd_vals["hist"], 4),
            ma50=round(ma50, 2),
            ma200=round(ma200, 2),
            ma_trend=trend_from_ma(ma50, ma200),
            volume=float(volume.iloc[-1]),
            volume_trend=vol_trend,
            as_of=as_of,
            chart=chart_series,
            volume_series=vol_series,
        )

        try:
            create_document("analysis", {
                "ticker": t,
                **data.model_dump(),
                "last_updated": as_of
            })
        except Exception:
            pass

        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------- News ------------------------------- #

NEWS_SOURCES = [
    # Yahoo Finance ticker feed
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    # Google News search focused on finance (past 7 days)
    "https://news.google.com/rss/search?q={ticker}%20finance%20when:7d&hl=en-US&gl=US&ceid=US:en",
    # MarketWatch general feed (filter later by ticker mention)
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
]

USER_AGENT = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

def clean_text(text: str) -> str:
    return " ".join((text or "").replace("\n", " ").split())


def summarize_text(text: str, max_sentences: int = 3) -> str:
    # Very simple heuristic summarizer: take first N sentences after cleaning
    text = clean_text(text)
    sentences = []
    acc = []
    for ch in text:
        acc.append(ch)
        if ch in ".!?" and len("".join(acc).strip()) > 20:
            sentences.append("".join(acc).strip())
            acc = []
        if len(sentences) >= max_sentences:
            break
    if not sentences:
        return text[:300]
    return " ".join(sentences[:max_sentences])


def fetch_article_text(url: str) -> str:
    try:
        r = requests.get(url, headers=USER_AGENT, timeout=8)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        # Common containers
        selectors = [
            "article",
            "div.article-body",
            "div#article-body",
            "div.story-content",
            "div.c-article-body",
            "section.article-body",
        ]
        texts: List[str] = []
        for sel in selectors:
            for node in soup.select(sel):
                texts.append(node.get_text(" "))
        if not texts:
            texts = [soup.get_text(" ")]
        content = max(texts, key=len)
        return clean_text(content)
    except Exception:
        return ""


def parse_feeds(ticker: str) -> List[NewsItem]:
    t = ticker.upper()
    items: List[NewsItem] = []
    for src in NEWS_SOURCES:
        url = src.format(ticker=t)
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:12]:
                title = e.get("title", "").strip()
                link = e.get("link", "")
                published = e.get("published", "") or e.get("updated", "")
                summary_raw = e.get("summary", "")
                # If summary is too short, try fetching the article text and summarize
                summ_text = summary_raw
                if len(clean_text(summ_text)) < 60 and link:
                    art_text = fetch_article_text(link)
                    if art_text:
                        summ_text = summarize_text(art_text, 3)
                else:
                    summ_text = summarize_text(summ_text, 3)

                source = "Unknown"
                try:
                    source = e.get("source", {}).get("title") or feed.feed.get("title") or "Unknown"
                except Exception:
                    source = feed.feed.get("title") or "Unknown"

                # Filter: include if ticker is in title/summary or source is Yahoo/Google finance feed
                include = t in title.upper() or t in clean_text(summary_raw).upper() or "Yahoo" in (source or "") or "Google" in (source or "")
                if include and link:
                    items.append(NewsItem(
                        source=source or "",
                        title=title,
                        url=link,
                        published=published,
                        summary=summ_text,
                    ))
        except Exception:
            continue

    # Deduplicate by URL
    seen = set()
    unique_items: List[NewsItem] = []
    for it in items:
        if it.url in seen:
            continue
        seen.add(it.url)
        unique_items.append(it)
    # Limit to 10
    return unique_items[:10]


@app.get("/api/news", response_model=List[NewsItem])
def get_news(ticker: str = Query(..., min_length=1, max_length=10)):
    t = ticker.upper().strip()
    try:
        news = parse_feeds(t)
        # Store in DB (best-effort)
        try:
            if news:
                for n in news:
                    create_document("article", {**n.model_dump(), "ticker": t})
        except Exception:
            pass
        return news
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stock", response_model=AnalyzeResponse)
def stock_all(ticker: str = Query(..., min_length=1, max_length=10)):
    indicators = analyze_ticker(ticker)
    news = get_news(ticker)
    return AnalyzeResponse(indicators=indicators, news=news)


@app.get("/")
def root():
    return {"message": "Stock Pulse Analyzer API running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
