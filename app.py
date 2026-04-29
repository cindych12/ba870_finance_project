# ================================================================
# app.py  —  Stock Investment Decision Support
# ================================================================

import pickle
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(
    page_title="Stock Investment Decision Support",
    page_icon="📈",
    layout="wide",
)

# ── Global CSS (light-theme, classmate style) ────────────────────
st.markdown("""
<style>
.signal-card {
    background: #f0f4ff;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    border-left: 4px solid #94a3b8;
}
.signal-card .card-title {
    font-weight: 700;
    font-size: 15px;
    color: #1a1a1a;
    margin-bottom: 4px;
}
.signal-card .card-detail {
    font-size: 13px;
    color: #444;
    line-height: 1.5;
}
.info-box {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0 16px 0;
    color: #1e3a5f;
    font-size: 0.88rem;
}
.info-box b { color: #1d4ed8; }
.signal-plain-buy  { background:#f0fdf4; border-left:4px solid #22c55e; border-radius:6px; padding:14px 18px; margin:12px 0; color:#14532d; }
.signal-plain-hold { background:#fefce8; border-left:4px solid #eab308; border-radius:6px; padding:14px 18px; margin:12px 0; color:#713f12; }
.signal-plain-sell { background:#fef2f2; border-left:4px solid #ef4444;  border-radius:6px; padding:14px 18px; margin:12px 0; color:#7f1d1d; }
.home-section {
    background: #f8fafc;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 16px;
    border: 1px solid #e2e8f0;
}
.home-section h3 { margin-top: 0; }
.method-block {
    border-left: 4px solid #3b82f6;
    padding-left: 14px;
    margin: 12px 0;
}
.method-block.green { border-color: #22c55e; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────
FEATURE_COLS = [
    "Return_1d", "Return_5d",
    "MA_5", "MA_20", "MA_ratio", "MA_diff", "Price_vs_MA20",
    "Volatility_5", "Volume_MA5", "Volume_Ratio",
    "Momentum_3", "Momentum_10", "Momentum_20",
    "HL_Range", "Volatility_10", "Breakout_20", "Drawdown_20",
    "Volume_Spike",
]

FEATURE_DESCRIBE = {
    "Return_1d":     "1-day price return",
    "Return_5d":     "5-day price return",
    "MA_5":          "5-day moving average level",
    "MA_20":         "20-day moving average level",
    "MA_ratio":      "MA5/MA20 ratio (short vs long trend)",
    "MA_diff":       "MA5 minus MA20 gap",
    "Price_vs_MA20": "Price relative to 20-day MA",
    "Volatility_5":  "5-day return volatility",
    "Volume_MA5":    "5-day average volume",
    "Volume_Ratio":  "Today's volume vs 5-day avg",
    "Momentum_3":    "3-day price momentum",
    "Momentum_10":   "10-day price momentum",
    "Momentum_20":   "20-day price momentum",
    "HL_Range":      "Intraday High-Low range",
    "Volatility_10": "10-day return volatility",
    "Breakout_20":   "Price vs 20-day high (breakout)",
    "Drawdown_20":   "Price vs 20-day low (drawdown)",
    "Volume_Spike":  "Volume spike vs 20-day avg",
}

SIGNAL_LABEL     = { 1: "BUY",  0: "HOLD", -1: "SELL" }
SIGNAL_EMOJI     = { 1: "🟢",   0: "🟡",    -1: "🔴"  }
SIGNAL_COLOR     = { 1: "#22c55e", 0: "#eab308", -1: "#ef4444" }
SIGNAL_CSS_CLASS = { 1: "signal-plain-buy", 0: "signal-plain-hold", -1: "signal-plain-sell" }

SIGNAL_PLAIN = {
    1: (
        "📢 The model suggests this may be a good time to BUY.",
        "Based on technical indicators, this stock is showing strong recent momentum and "
        "an upward moving-average trend, suggesting it may be in the early-to-mid stage of "
        "an uptrend. This is not a guarantee of profit — always factor in your own risk tolerance."
    ),
    0: (
        "⏸️ The model suggests HOLDING — no clear action needed right now.",
        "Current technical indicators show no strong directional bias. "
        "If you already hold a position, it is reasonable to stay put. "
        "If you have not entered yet, consider waiting for a clearer signal."
    ),
    -1: (
        "⚠️ The model suggests considering a SELL or reducing exposure.",
        "Technical indicators show weakening momentum and possible bearish signals "
        "such as a death cross or support breakdown. Short-term downside risk appears elevated. "
        "If you hold a position, consider a stop-loss or trimming your size."
    ),
}

# Chart display periods
DISPLAY_PERIODS = ["1wk", "1mo", "3mo", "6mo", "1y"]
DISPLAY_PERIOD_LABEL = {
    "1wk": "1 Week",
    "1mo": "1 Month",
    "3mo": "3 Months",
    "6mo": "6 Months",
    "1y":  "1 Year",
}
DISPLAY_PERIOD_DAYS = {
    "1wk": 7, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365,
}

PERIODS = ["1mo", "3mo", "6mo", "1y"]
PERIOD_LABEL = {"1mo": "1 Month", "3mo": "3 Months", "6mo": "6 Months", "1y": "1 Year"}

MIN_FETCH_PERIOD = "6mo"

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "JPM", "JNJ", "XOM", "WMT",
    "META", "AMD", "BAC", "GS", "COST",
    "DIS", "CVX", "CAT", "BA", "PFE",
    "SPY", "QQQ",
]

# Company names for display
COMPANY_NAMES = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.", "NVDA": "NVIDIA Corp.", "TSLA": "Tesla Inc.",
    "JPM": "JPMorgan Chase & Co.", "JNJ": "Johnson & Johnson", "XOM": "Exxon Mobil Corp.",
    "WMT": "Walmart Inc.", "META": "Meta Platforms Inc.", "AMD": "Advanced Micro Devices",
    "BAC": "Bank of America Corp.", "GS": "Goldman Sachs Group", "COST": "Costco Wholesale Corp.",
    "DIS": "The Walt Disney Co.", "CVX": "Chevron Corp.", "CAT": "Caterpillar Inc.",
    "BA": "Boeing Co.", "PFE": "Pfizer Inc.", "SPY": "SPDR S&P 500 ETF",
    "QQQ": "Invesco QQQ Trust",
}

def ticker_label(t: str) -> str:
    name = COMPANY_NAMES.get(t, "")
    return f"{t} — {name}" if name else t

# ── Load model ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model = load_model()
HAS_PROBA      = model is not None and hasattr(model, "predict_proba")
HAS_IMPORTANCE = model is not None and hasattr(model, "feature_importances_")

# ── Feature engineering ──────────────────────────────────────────
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").copy()
    df["Return_1d"]     = df["Close"].pct_change()
    df["Return_5d"]     = df["Close"].pct_change(5)
    df["MA_5"]          = df["Close"].rolling(5).mean()
    df["MA_20"]         = df["Close"].rolling(20).mean()
    df["MA_ratio"]      = df["MA_5"] / df["MA_20"]
    df["MA_diff"]       = df["MA_5"] - df["MA_20"]
    df["Price_vs_MA20"] = df["Close"] / df["MA_20"]
    df["Volatility_5"]  = df["Return_1d"].rolling(5).std()
    df["Volatility_10"] = df["Return_1d"].rolling(10).std()
    df["Volume_MA5"]    = df["Volume"].rolling(5).mean()
    df["Volume_Ratio"]  = df["Volume"] / df["Volume_MA5"]
    df["Volume_MA20"]   = df["Volume"].rolling(20).mean()
    df["Volume_Spike"]  = df["Volume"] / df["Volume_MA20"]
    df["Momentum_3"]    = df["Close"] / df["Close"].shift(3) - 1
    df["Momentum_10"]   = df["Close"] / df["Close"].shift(10) - 1
    df["Momentum_20"]   = df["Close"] / df["Close"].shift(20) - 1
    df["HL_Range"]      = (df["High"] - df["Low"]) / df["Close"]
    df["High_20"]       = df["High"].rolling(20).max()
    df["Low_20"]        = df["Low"].rolling(20).min()
    df["Breakout_20"]   = df["Close"] / df["High_20"]
    df["Drawdown_20"]   = df["Close"] / df["Low_20"]
    return df

# ── Data download ────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_stock(ticker: str, period: str = "6mo") -> pd.DataFrame:
    # For 1y display we need a full year; otherwise 6mo is enough for all rolling features
    fetch_period = period if period in ("1y", "2y") else MIN_FETCH_PERIOD
    df = yf.download(ticker, period=fetch_period, progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()
    df = df[needed].reset_index()
    date_col = next((c for c in df.columns if c.lower() in {"date", "datetime", "timestamp"}), None)
    if date_col is None:
        return pd.DataFrame()
    if date_col != "Date":
        df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df.sort_values("Date").reset_index(drop=True)

def slice_for_display(df: pd.DataFrame, display_period: str) -> pd.DataFrame:
    days   = DISPLAY_PERIOD_DAYS.get(display_period, 180)
    cutoff = df["Date"].max() - pd.Timedelta(days=days)
    return df[df["Date"] >= cutoff].copy()

# ── Confidence ───────────────────────────────────────────────────
def get_confidence(X: np.ndarray, signal: int) -> float:
    if not HAS_PROBA:
        return None
    proba   = model.predict_proba(X)[0]
    classes = list(model.classes_)
    if signal in classes:
        return round(proba[classes.index(signal)] * 100, 1)
    return None

# ── Per-feature plain-English explanation ────────────────────────
def explain_feature(feat: str, val: float) -> str:
    pct = val * 100
    if feat == "MA_ratio":
        direction = "above" if val > 1.0 else "below"
        sentiment = "bullish — short-term trend is rising. 📈" if val > 1.0 else "bearish — short-term trend is falling. 📉"
        return (f"We compare the 5-day average price to the 20-day average price. "
                f"A ratio of {val:.3f} means the short-term average is {direction} the lon
