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

@st.cache_data(show_spinner=False)
def get_company_name(ticker: str) -> str:
    """Lookup company name: local dict first, then yFinance."""
    if ticker in COMPANY_NAMES:
        return COMPANY_NAMES[ticker]
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker

def ticker_label(t: str) -> str:
    name = get_company_name(t)
    return f"{t} — {name}" if name and name != t else t

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
                f"A ratio of {val:.3f} means the short-term average is {direction} the long-term average — {sentiment}")
    if feat == "MA_diff":
        direction = "above" if val > 0 else "below"
        return (f"The 5-day moving average is ${abs(val):.2f} {direction} the 20-day moving average. "
                f"{'Positive gap suggests upward momentum.' if val > 0 else 'Negative gap suggests downward pressure.'}")
    if feat == "Price_vs_MA20":
        rel      = "above" if val > 1 else "below"
        pct_diff = abs(val - 1) * 100
        return (f"Think of the 20-day average as the stock's one-month fair value. "
                f"The current price is {pct_diff:.1f}% {rel} this average. "
                f"{'Running above average — momentum is strong. 🔥' if val > 1 else 'Trading below average — may be weak or undervalued. 💡'}")
    if feat == "Return_1d":
        d = "UP" if val > 0 else "DOWN"
        return f"The stock moved {d} {abs(pct):.2f}% yesterday. {'Positive momentum.' if val > 0 else 'Negative momentum.'}"
    if feat == "Return_5d":
        d = "UP" if val > 0 else "DOWN"
        return (f"Over the past 5 trading days (~1 week), the stock is {d} {abs(pct):.2f}%. "
                f"{'Short-term strength.' if val > 0 else 'Short-term weakness.'}")
    if feat == "Momentum_3":
        d = "gained" if val > 0 else "lost"
        return f"The stock has {d} {abs(pct):.2f}% over the last 3 days. {'Recent buying pressure.' if val > 0 else 'Recent selling pressure.'}"
    if feat == "Momentum_10":
        d = "up" if val > 0 else "down"
        return (f"The stock is {d} {abs(pct):.2f}% over the last 10 days. "
                f"{'Uptrend has staying power.' if val > 0 else 'Downtrend has staying power.'}")
    if feat == "Momentum_20":
        d = "up" if val > 0 else "down"
        return f"The stock is {d} {abs(pct):.2f}% over ~1 month. {'Sustained upward trend.' if val > 0 else 'Sustained downward trend.'}"
    if feat == "MA_5":
        return f"The average closing price over the last 5 trading days is ${val:.2f}. Used with MA_20 to detect trend direction."
    if feat == "MA_20":
        return f"The average closing price over the last 20 trading days (~1 month) is ${val:.2f}. Acts as a key support/resistance reference."
    if feat == "Volatility_5":
        level = "High" if val > 0.02 else "Low"
        mood  = "moving a lot this week — higher risk and reward." if val > 0.02 else "calm and stable this week."
        return f"{level} volatility ({abs(pct):.2f}%) — the stock has been {mood}"
    if feat == "Volatility_10":
        level = "High" if val > 0.02 else "Low"
        mood  = "moving a lot over 2 weeks — uncertain market." if val > 0.02 else "stable over 2 weeks."
        return f"{level} volatility ({abs(pct):.2f}%) — the stock has been {mood}"
    if feat == "Volume_MA5":
        return f"Average daily shares traded over the last 5 days: {val:,.0f}. Used as a baseline for volume spikes."
    if feat == "Volume_Ratio":
        level = "higher" if val > 1 else "lower"
        return (f"Today's volume is {val:.2f}x the 5-day average — {level} than usual. "
                f"{'Elevated volume can confirm a price move.' if val > 1.2 else 'Normal or low volume.'}")
    if feat == "Volume_Spike":
        return (f"Today's volume is {val:.2f}x the 20-day average. "
                f"{'Significant spike — strong interest. 🔊' if val > 1.5 else 'Normal range.' if val > 0.8 else 'Low volume — weak participation. 🔇'}")
    if feat == "HL_Range":
        width = "wide" if val > 0.02 else "narrow"
        return (f"Today's price moved {abs(pct):.2f}% from low to high — a {width} intraday range. "
                f"{'Wide = active, volatile session.' if val > 0.02 else 'Narrow = calm, steady day.'}")
    if feat == "Breakout_20":
        return (f"Price is at {val*100:.1f}% of the 20-day high. "
                f"{'Close to 20-day high — potential breakout. 🚀' if val > 0.97 else 'Well below 20-day high — momentum is weak.'}")
    if feat == "Drawdown_20":
        return (f"Price is at {val*100:.1f}% of the 20-day low. "
                f"{'Well above recent low — good support cushion. ✅' if val > 1.05 else 'Close to 20-day low — watch carefully.'}")
    return f"Value: {val:.4f}"

FEATURE_TITLE = {
    "Return_1d":     lambda v: f"⚡ Yesterday's price change: {v*100:.2f}%",
    "Return_5d":     lambda v: f"📅 Last 5 days price change: {v*100:.2f}%",
    "MA_5":          lambda v: f"📐 5-day average price: ${v:.2f}",
    "MA_20":         lambda v: f"📐 20-day average price: ${v:.2f}",
    "MA_ratio":      lambda v: f"📊 Short vs Long-term Trend: {v:.3f}",
    "MA_diff":       lambda v: f"📏 MA gap (short minus long): ${v:.2f}",
    "Price_vs_MA20": lambda v: f"📍 Price vs 20-day average: {v:.3f}",
    "Volatility_5":  lambda v: f"🌊 5-day price swings: {v*100:.2f}%",
    "Volatility_10": lambda v: f"🌊 10-day price swings: {v*100:.2f}%",
    "Volume_MA5":    lambda v: f"📦 5-day avg volume: {v:,.0f}",
    "Volume_Ratio":  lambda v: f"📦 Volume ratio (today vs 5d avg): {v:.2f}x",
    "Volume_Spike":  lambda v: f"🔊 Volume spike (today vs 20d avg): {v:.2f}x",
    "Momentum_3":    lambda v: f"🚀 3-day momentum: {v*100:.2f}%",
    "Momentum_10":   lambda v: f"🏃 10-day momentum: {v*100:.2f}%",
    "Momentum_20":   lambda v: f"🗓️ 1-month momentum: {v*100:.2f}%",
    "HL_Range":      lambda v: f"📏 Today's High-Low range: {v*100:.2f}%",
    "Breakout_20":   lambda v: f"🚀 Price vs 20-day high: {v*100:.1f}%",
    "Drawdown_20":   lambda v: f"🛡️ Price vs 20-day low: {v*100:.1f}%",
}

# ── Model helpers ────────────────────────────────────────────────
def get_top_features(n: int = 3):
    if not HAS_IMPORTANCE:
        return []
    imp     = model.feature_importances_
    indices = np.argsort(imp)[::-1][:n]
    return [(FEATURE_COLS[i], imp[i]) for i in indices]

def predict_ticker(ticker: str, fetch_period: str = "6mo"):
    """
    fetch_period controls how much data is downloaded.
    For display windows > 6mo (e.g. 1y) we fetch more so the chart shows the full range.
    Features and the signal are always computed from the full fetched dataset,
    and the prediction uses the most recent row only.
    """
    raw = fetch_stock(ticker, period=fetch_period)
    if raw.empty or len(raw) < 25:
        return None, None, None, None
    df     = compute_features(raw)
    latest = df.dropna(subset=FEATURE_COLS).iloc[-1]
    X      = latest[FEATURE_COLS].values.reshape(1, -1)
    signal = int(model.predict(X)[0])
    conf   = get_confidence(X, signal)
    return signal, conf, latest, df

# ── Reusable UI helpers ──────────────────────────────────────────
def render_signal_plain(signal: int):
    title, body = SIGNAL_PLAIN[signal]
    css = SIGNAL_CSS_CLASS[signal]
    st.markdown(
        f"<div class='{css}'><b>{title}</b><br><span>{body}</span></div>",
        unsafe_allow_html=True,
    )

def render_confidence(confidence: float, color: str):
    if confidence is None:
        return
    if confidence >= 65:
        note, icon, nc = "High confidence — the model is fairly certain about this signal.", "✅", "#166534"
    elif confidence >= 45:
        note, icon, nc = "Moderate confidence — signal is plausible but not strong.", "🟡", "#854d0e"
    else:
        note, icon, nc = "Low confidence — treat with caution — the model is not very sure about this signal.", "🔺", "#991b1b"
    st.markdown("**Model Confidence**")
    st.markdown(
        f"<div style='background:#e5e7eb;border-radius:8px;height:22px;width:100%;overflow:hidden;margin-bottom:4px'>"
        f"<div style='width:{confidence}%;height:22px;background:{color};border-radius:8px;"
        f"display:flex;align-items:center;padding-left:10px'>"
        f"<span style='color:white;font-weight:700;font-size:13px'>{confidence:.1f}%</span></div></div>"
        f"<div style='font-size:13px;color:{nc};margin-top:2px'>{icon} {note}</div>",
        unsafe_allow_html=True,
    )

def render_top_factors(latest: pd.Series):
    top = get_top_features(3)
    if not top:
        return
    st.markdown("#### 📌 Why this signal? (Top 3 most important factors)")
    for feat_col, _ in top:
        val   = float(latest.get(feat_col, 0))
        title = FEATURE_TITLE.get(feat_col, lambda v: f"{feat_col}: {v:.4f}")(val)
        body  = explain_feature(feat_col, val)
        st.markdown(
            f"<div class='signal-card'>"
            f"<div class='card-title'>{title}</div>"
            f"<div class='card-detail'>{body}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Settings")

    # ── Ticker search ──────────────────────────────────────────────
    st.markdown("**Search & add a stock**")
    search_input = st.text_input(
        "search_ticker",
        placeholder="Type ticker or company name (e.g. NFLX, Netflix)",
        label_visibility="collapsed",
    )

    # If user typed something, try to resolve it via yFinance
    if search_input.strip():
        candidate = search_input.strip().upper()
        name = get_company_name(candidate)
        if name != candidate:
            # Valid ticker with a real company name
            if st.button(f"➕ Add  {candidate} — {name}", use_container_width=True):
                if "extra_tickers" not in st.session_state:
                    st.session_state["extra_tickers"] = []
                if candidate not in st.session_state["extra_tickers"]:
                    st.session_state["extra_tickers"].append(candidate)
                    st.rerun()
        else:
            st.caption(f"⚠️ Could not find company for '{search_input.strip()}'. Check the ticker symbol.")

    extra_tickers = st.session_state.get("extra_tickers", [])
    all_tickers   = list(dict.fromkeys(DEFAULT_TICKERS + extra_tickers))

    selected = st.multiselect(
        "Stocks to analyse",
        options=all_tickers,
        default=["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"],
        format_func=ticker_label,
    )

    run_btn = st.button("🔍 Get Recommendations", type="primary")

# ════════════════════════════════════════════════════════════════
# Model status
# ════════════════════════════════════════════════════════════════
if model is None:
    st.error("⚠️ `model.pkl` not found. Please run `python train_model.py` first, then restart the app.")
    st.stop()

# ════════════════════════════════════════════════════════════════
# Tabs
# ════════════════════════════════════════════════════════════════
tab_home, tab_rec, tab_detail, tab_chart = st.tabs([
    "🏠 Home",
    "🔮 Recommendations",
    "🔍 Single Stock Detail",
    "📉 Price Chart",
])

# ════════════════════════════════════════════════════════════════
# Tab 0 — Home
# ════════════════════════════════════════════════════════════════
with tab_home:
    st.markdown("## 📈 Stock Investment Decision Support")
    st.markdown(
        "A machine-learning powered tool that fetches real-time market data and gives you "
        "a clear **Buy / Hold / Sell** signal — with confidence scores and plain-English explanations "
        "so you always understand *why* the model made its call."
    )

    st.markdown("---")

    # Mission
    st.markdown(
        "<div class='home-section'>"
        "<h3>🎯 Our Mission</h3>"
        "<p>Investing can feel overwhelming — especially if you're just getting started. "
        "We built this tool to <b>democratize access to data-driven investment insights</b>, "
        "making the kind of technical analysis that professional traders use every day "
        "accessible to <b>everyone</b>, from seasoned investors to curious beginners.</p>"
        "<p>We don't tell you what to do with your money. We give you a clear, transparent signal "
        "backed by real data — and explain every step — so you can make <b>informed decisions</b> with confidence.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        # Dataset
        st.markdown(
            "<div class='home-section'>"
            "<h3>📦 Dataset & Features</h3>"
            "<p>Our model was trained on historical market data sourced via <b>yFinance</b>, "
            "covering <b>20 large-cap S&P 500 companies</b> with thousands of daily price and volume observations.</p>"
            "<div style='background:#fefce8;border-left:4px solid #eab308;border-radius:6px;padding:10px 14px;margin:10px 0;font-size:0.88rem;color:#713f12'>"
            "<b>⚠️ Accuracy Note</b><br>"
            "Because the model was trained on large-cap U.S. stocks, predictions are most reliable for:<br>"
            "✅ S&P 500 components &nbsp; ✅ Large-cap U.S. equities &nbsp; ✅ U.S. ETFs (SPY, QQQ)<br>"
            "Predictions for small-cap stocks, international equities, or crypto ETFs may be less accurate — use with caution."
            "</div>"
            "<p>We engineer <b>18 technical features</b> from raw OHLCV data:</p>",
            unsafe_allow_html=True,
        )
        feature_table = pd.DataFrame({
            "Category": [
                "Returns", "Returns",
                "Trend (Moving Avg)", "Trend (Moving Avg)", "Trend (Moving Avg)",
                "Momentum", "Momentum", "Momentum",
                "Volatility", "Volatility",
                "Volume", "Volume", "Volume",
                "Range & Breakout", "Range & Breakout", "Range & Breakout",
            ],
            "Feature": [
                "Return_1d", "Return_5d",
                "MA_5, MA_20", "MA_ratio", "MA_diff, Price_vs_MA20",
                "Momentum_3", "Momentum_10", "Momentum_20",
                "Volatility_5", "Volatility_10",
                "Volume_MA5", "Volume_Ratio", "Volume_Spike",
                "HL_Range", "Breakout_20", "Drawdown_20",
            ],
        })
        st.dataframe(feature_table, use_container_width=True, hide_index=True, height=320)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        # Model
        st.markdown(
            "<div class='home-section'>"
            "<h3>🧠 Model & Method</h3>"
            "<p>This app uses a <b>Random Forest Classifier</b> trained offline on labelled historical data. "
            "Each trading day is labelled as a Buy, Hold, or Sell signal based on subsequent price movement.</p>"
            "<div class='method-block'>"
            "<b>Random Forest Classifier</b><br>"
            "An ensemble of decision trees that votes on the most likely outcome. "
            "It outputs a probability distribution across BUY / HOLD / SELL — "
            "which we use to compute the <b>Confidence %</b> shown in the app."
            "</div>"
            "<div class='method-block green'>"
            "<b>Feature Importance</b><br>"
            "The model ranks which indicators matter most. "
            "The app surfaces the <b>Top 3 factors</b> driving each prediction so you always know why."
            "</div>"
            "<p style='margin-top:14px;font-size:0.9rem;color:#555'>"
            "⚠️ <b>Disclaimer:</b> This tool is for educational and informational purposes only. "
            "It is not financial advice. Always do your own research and consult a licensed advisor before investing."
            "</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        # How to use
        st.markdown(
            "<div class='home-section'>"
            "<h3>🗺️ How to Use</h3>"
            "<ol>"
            "<li>Select your <b>stocks</b> in the sidebar (or add custom tickers)</li>"
            "<li>Click <b>🔍 Get Recommendations</b> for a batch overview of all selected stocks</li>"
            "<li>Go to <b>Single Stock Detail</b> for a deep-dive — confidence score, signal explanation, and top factors</li>"
            "<li>Head to <b>Price Chart</b> and pick a time period (1 Week to 1 Year) to view price and moving averages</li>"
            "</ol>"
            "</div>",
            unsafe_allow_html=True,
        )

# ════════════════════════════════════════════════════════════════
# Tab 1 — Recommendations
# ════════════════════════════════════════════════════════════════
with tab_rec:
    st.title("📈 Stock Investment Decision Support")
    st.markdown("Pre-trained Random Forest → live market data → **Buy / Hold / Sell** signal.")
    st.success("✅ Model loaded from `model.pkl`" + (" &nbsp; 🎯 Confidence scoring enabled" if HAS_PROBA else ""))

    if not selected:
        st.info("Select stocks in the sidebar and click **Get Recommendations**.")
    elif run_btn or "results" not in st.session_state:
        results  = []
        progress = st.progress(0, text="Fetching data…")
        for i, ticker in enumerate(selected):
            progress.progress((i + 1) / len(selected), text=f"Processing {ticker}…")
            signal, confidence, latest, _ = predict_ticker(ticker)
            if signal is None:
                results.append({
                    "Ticker": ticker,
                    "Company": COMPANY_NAMES.get(ticker, ""),
                    "Close ($)": "—", "Signal": "⚠️ No data", "Confidence": "—",
                })
                continue
            results.append({
                "Ticker":     ticker,
                "Company":    COMPANY_NAMES.get(ticker, ""),
                "Close ($)":  round(float(latest["Close"]), 2),
                "Signal":     f"{SIGNAL_EMOJI[signal]} {SIGNAL_LABEL[signal]}",
                "Confidence": f"{confidence:.1f}%" if confidence is not None else "N/A",
                "_sig":       signal,
                "_conf":      confidence or 0,
            })
        progress.empty()
        st.session_state["results"] = results

    if "results" in st.session_state:
        res   = st.session_state["results"]
        valid = [r for r in res if "_sig" in r]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 BUY",  sum(1 for r in valid if r["_sig"] ==  1))
        c2.metric("🟡 HOLD", sum(1 for r in valid if r["_sig"] ==  0))
        c3.metric("🔴 SELL", sum(1 for r in valid if r["_sig"] == -1))
        avg_conf = np.mean([r["_conf"] for r in valid]) if valid else 0
        c4.metric("🎯 Avg Confidence", f"{avg_conf:.1f}%")

        st.markdown("---")
        display = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in res])
        st.dataframe(display, use_container_width=True, hide_index=True)

        if HAS_PROBA and valid:
            st.markdown("#### 🎯 Confidence by Stock")
            fig_c, ax_c = plt.subplots(figsize=(10, 2.5))
            tickers_ = [r["Ticker"] for r in valid]
            confs_   = [r["_conf"]  for r in valid]
            colors_  = [SIGNAL_COLOR[r["_sig"]] for r in valid]
            bars = ax_c.barh(tickers_, confs_, color=colors_, alpha=0.85)
            ax_c.set_xlim(0, 100)
            ax_c.set_xlabel("Confidence (%)")
            ax_c.axvline(50, color="gray", linestyle="--", alpha=0.5, linewidth=1)
            for bar, v in zip(bars, confs_):
                ax_c.text(v + 1, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9)
            ax_c.grid(axis="x", alpha=0.2)
            fig_c.tight_layout()
            st.pyplot(fig_c)
            plt.close(fig_c)

# ════════════════════════════════════════════════════════════════
# Tab 2 — Period Comparison
# ════════════════════════════════════════════════════════════════
# Tab 3 — Single Stock Detail
# ════════════════════════════════════════════════════════════════
with tab_detail:
    pick = st.selectbox(
        "Choose a stock",
        options=selected if selected else DEFAULT_TICKERS[:5],
        format_func=ticker_label,
    )

    if st.button("Analyse", key="analyse_btn"):
        with st.spinner(f"Fetching {pick}…"):
            signal, confidence, latest, df_feat = predict_ticker(pick)

        if signal is None:
            st.error(f"Could not fetch data for **{ticker_label(pick)}**. Check the ticker symbol.")
        else:
            color    = SIGNAL_COLOR[signal]
            co_name  = COMPANY_NAMES.get(pick, pick)
            conf_text = (f"<div style='font-size:1.1rem;color:{color}99;margin-top:6px'>"
                         f"Confidence: {confidence:.1f}%</div>") if confidence is not None else ""

            st.markdown(
                f"<div style='text-align:center;padding:20px;border-radius:12px;"
                f"background:{color}15;border:2px solid {color}'>"
                f"<span style='font-size:3rem'>{SIGNAL_EMOJI[signal]}</span><br>"
                f"<span style='font-size:1rem;color:#888'>{co_name}</span><br>"
                f"<span style='font-size:2rem;font-weight:700;color:{color}'>{pick}: {SIGNAL_LABEL[signal]}</span>"
                + conf_text + "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("")

            render_confidence(confidence, color)
            st.markdown("")
            render_signal_plain(signal)
            st.markdown("")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Latest Close",    f"${float(latest['Close']):.2f}")
            m2.metric("1-Day Return",    f"{float(latest['Return_1d'])*100:.2f}%")
            m3.metric("5-Day Return",    f"{float(latest['Return_5d'])*100:.2f}%")
            m4.metric("MA Ratio (5/20)", f"{float(latest['MA_ratio']):.4f}")

            st.markdown("---")
            render_top_factors(latest)

            st.markdown("---")
            st.markdown("#### 📊 Feature Values Used for Prediction")
            feat_df = pd.DataFrame({
                "Feature":     FEATURE_COLS,
                "Value":       [round(float(latest[f]), 6) for f in FEATURE_COLS],
                "Description": [FEATURE_DESCRIBE.get(f, "") for f in FEATURE_COLS],
            })
            st.dataframe(feat_df, use_container_width=True, hide_index=True, height=320)

# ════════════════════════════════════════════════════════════════
# Tab 4 — Price Chart
# ════════════════════════════════════════════════════════════════
with tab_chart:
    chart_ticker = st.selectbox(
        "Select stock",
        options=selected if selected else DEFAULT_TICKERS[:5],
        format_func=ticker_label,
        key="chart_sel",
    )

    chart_display_period = st.radio(
        "Display period",
        options=DISPLAY_PERIODS,
        index=2,
        format_func=lambda p: DISPLAY_PERIOD_LABEL[p],
        horizontal=True,
        key="chart_period_radio",
    )

    if st.button("Show Chart", key="chart_btn"):
        with st.spinner(f"Loading {chart_ticker}…"):
            # Pass the display period so 1-year charts actually fetch 1 year of data
            fetch_p = chart_display_period if chart_display_period == "1y" else "6mo"
            sig, conf, latest, df_feat = predict_ticker(chart_ticker, fetch_period=fetch_p)

        if df_feat is None:
            st.error(f"Could not fetch data for **{ticker_label(chart_ticker)}**.")
        else:
            df_plot = slice_for_display(
                df_feat.dropna(subset=["MA_5", "MA_20"]),
                chart_display_period,
            )
            if df_plot.empty:
                st.error("Not enough data for the selected display period. Try a longer window.")
            else:
                period_label = DISPLAY_PERIOD_LABEL[chart_display_period]
                co_name      = COMPANY_NAMES.get(chart_ticker, chart_ticker)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df_plot["Date"], df_plot["Close"], label="Close", color="#3b82f6", linewidth=1.8)
                ax.plot(df_plot["Date"], df_plot["MA_5"],  label="MA 5",  color="#f97316", linewidth=1.2, linestyle="--")
                ax.plot(df_plot["Date"], df_plot["MA_20"], label="MA 20", color="#a855f7", linewidth=1.2, linestyle="--")

                if chart_display_period == "1wk":
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %b %d"))
                    ax.xaxis.set_major_locator(mdates.DayLocator())
                elif chart_display_period == "1mo":
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
                else:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

                plt.xticks(rotation=30, ha="right")
                ax.set_title(f"{chart_ticker} ({co_name}) — Close & Moving Averages ({period_label})")
                ax.set_ylabel("Price (USD)")
                ax.legend()
                ax.grid(alpha=0.25)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                if sig is not None:
                    color    = SIGNAL_COLOR[sig]
                    conf_str = f" &nbsp; Confidence: **{conf:.1f}%**" if conf is not None else ""
                    st.markdown(
                        f"**Signal:** "
                        f"<span style='color:{color};font-size:1.2rem;font-weight:bold'>"
                        f"{SIGNAL_EMOJI[sig]} {SIGNAL_LABEL[sig]}</span>" + conf_str,
                        unsafe_allow_html=True,
                    )
                    render_signal_plain(sig)
                    st.markdown("")
                    render_top_factors(latest)
