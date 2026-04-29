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

# ── Global CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
.signal-card {
    background: #f0f4ff;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    border-left: 4px solid #94a3b8;
}
.signal-card .card-title { font-weight:700; font-size:15px; color:#1a1a1a; margin-bottom:4px; }
.signal-card .card-detail { font-size:13px; color:#444; line-height:1.5; }
.info-box { background:#eff6ff; border-left:4px solid #3b82f6; border-radius:6px; padding:12px 16px; margin:8px 0 16px 0; color:#1e3a5f; font-size:0.88rem; }
.info-box b { color:#1d4ed8; }
.signal-plain-buy  { background:#f0fdf4; border-left:4px solid #22c55e; border-radius:6px; padding:14px 18px; margin:12px 0; color:#14532d; }
.signal-plain-hold { background:#fefce8; border-left:4px solid #eab308; border-radius:6px; padding:14px 18px; margin:12px 0; color:#713f12; }
.signal-plain-sell { background:#fef2f2; border-left:4px solid #ef4444;  border-radius:6px; padding:14px 18px; margin:12px 0; color:#7f1d1d; }
.home-section { background:#f8fafc; border-radius:12px; padding:24px 28px; margin-bottom:16px; border:1px solid #e2e8f0; }
.home-section h3 { margin-top:0; }
.method-block { border-left:4px solid #3b82f6; padding-left:14px; margin:12px 0; }
.method-block.green { border-color:#22c55e; }
.disclaimer { font-size:0.78rem; color:#94a3b8; margin-top:10px; font-style:italic; }
</style>
""", unsafe_allow_html=True)

DISCLAIMER = "*⚠️ This tool provides financial information for educational purposes. While the signals may be used as a reference, please exercise caution and conduct your own research. Past signals do not guarantee future returns.*"

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

DISPLAY_PERIODS = ["1wk", "1mo", "3mo", "6mo", "1y"]
DISPLAY_PERIOD_LABEL = {
    "1wk": "1 Week", "1mo": "1 Month", "3mo": "3 Months",
    "6mo": "6 Months", "1y": "1 Year",
}
DISPLAY_PERIOD_DAYS = {
    "1wk": 7, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365,
}

MIN_FETCH_PERIOD = "6mo"

# Training metadata — update if you retrain the model
MODEL_TRAIN_INFO = {
    "companies": 20,
    "index":     "S&P 500",
    "period":    "2021 – 2026 (current)",
    "source":    "yFinance (daily OHLCV)",
}

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "JPM", "JNJ", "XOM", "WMT",
    "META", "AMD", "BAC", "GS", "COST",
    "DIS", "CVX", "CAT", "BA", "PFE",
    "SPY", "QQQ",
]

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

# ── Company name lookup (yFinance fallback) ──────────────────────
@st.cache_data(show_spinner=False)
def get_company_name(ticker: str) -> str:
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

# ── yFinance ticker search by name ──────────────────────────────
@st.cache_data(show_spinner=False)
def search_ticker(query: str) -> list:
    """Search yFinance for tickers matching a company name or ticker string.
    Returns list of (ticker, name) tuples."""
    try:
        results = yf.Search(query, max_results=5).quotes
        out = []
        for r in results:
            sym  = r.get("symbol", "")
            name = r.get("longname") or r.get("shortname") or ""
            if sym and name:
                out.append((sym, name))
        return out
    except Exception:
        # Fallback: treat input as direct ticker
        name = get_company_name(query.upper())
        if name != query.upper():
            return [(query.upper(), name)]
        return []

# ── Load model ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model          = load_model()
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

# ── Feature explanations ─────────────────────────────────────────
def explain_feature(feat: str, val: float) -> str:
    pct = val * 100
    if feat == "MA_ratio":
        d = "above" if val > 1.0 else "below"
        s = "bullish — short-term trend is rising. 📈" if val > 1.0 else "bearish — short-term trend is falling. 📉"
        return f"5-day vs 20-day average: ratio of {val:.3f} means short-term is {d} long-term — {s}"
    if feat == "MA_diff":
        d = "above" if val > 0 else "below"
        return f"MA5 is ${abs(val):.2f} {d} MA20. {'Positive gap = upward momentum.' if val > 0 else 'Negative gap = downward pressure.'}"
    if feat == "Price_vs_MA20":
        rel = "above" if val > 1 else "below"
        return f"Price is {abs(val-1)*100:.1f}% {rel} the 20-day average (one-month fair value). {'Strong momentum. 🔥' if val > 1 else 'May be weak or undervalued. 💡'}"
    if feat == "Return_1d":
        return f"Stock moved {'UP' if val>0 else 'DOWN'} {abs(pct):.2f}% yesterday. {'Positive momentum.' if val>0 else 'Negative momentum.'}"
    if feat == "Return_5d":
        return f"Past 5 days: {'UP' if val>0 else 'DOWN'} {abs(pct):.2f}%. {'Short-term strength.' if val>0 else 'Short-term weakness.'}"
    if feat == "Momentum_3":
        return f"{'Gained' if val>0 else 'Lost'} {abs(pct):.2f}% over 3 days. {'Recent buying pressure.' if val>0 else 'Recent selling pressure.'}"
    if feat == "Momentum_10":
        return f"{'Up' if val>0 else 'Down'} {abs(pct):.2f}% over 10 days. {'Uptrend has staying power.' if val>0 else 'Downtrend has staying power.'}"
    if feat == "Momentum_20":
        return f"{'Up' if val>0 else 'Down'} {abs(pct):.2f}% over ~1 month. {'Sustained upward trend.' if val>0 else 'Sustained downward trend.'}"
    if feat == "MA_5":
        return f"5-day average price: ${val:.2f}. Used with MA_20 to detect trend direction."
    if feat == "MA_20":
        return f"20-day average price: ${val:.2f}. Key support/resistance reference (~1 month)."
    if feat == "Volatility_5":
        return f"{'High' if val>0.02 else 'Low'} 5-day volatility ({abs(pct):.2f}%) — stock has been {'moving a lot this week.' if val>0.02 else 'calm this week.'}"
    if feat == "Volatility_10":
        return f"{'High' if val>0.02 else 'Low'} 10-day volatility ({abs(pct):.2f}%) — {'uncertain market over 2 weeks.' if val>0.02 else 'stable over 2 weeks.'}"
    if feat == "Volume_MA5":
        return f"Avg daily volume over 5 days: {val:,.0f} shares. Used as a volume baseline."
    if feat == "Volume_Ratio":
        return f"Today's volume is {val:.2f}x the 5-day average. {'Elevated — can confirm a move.' if val>1.2 else 'Normal or low.'}"
    if feat == "Volume_Spike":
        return f"Volume is {val:.2f}x the 20-day average. {'Significant spike — strong interest. 🔊' if val>1.5 else 'Normal range.' if val>0.8 else 'Low — weak participation. 🔇'}"
    if feat == "HL_Range":
        return f"Price moved {abs(pct):.2f}% from low to high today — {'wide (volatile session).' if val>0.02 else 'narrow (calm day).'}"
    if feat == "Breakout_20":
        return f"Price at {val*100:.1f}% of 20-day high. {'Near breakout. 🚀' if val>0.97 else 'Below recent highs — momentum weak.'}"
    if feat == "Drawdown_20":
        return f"Price at {val*100:.1f}% of 20-day low. {'Good support cushion. ✅' if val>1.05 else 'Near support — watch carefully.'}"
    return f"Value: {val:.4f}"

FEATURE_TITLE = {
    "Return_1d":     lambda v: f"⚡ Yesterday's change: {v*100:.2f}%",
    "Return_5d":     lambda v: f"📅 5-day change: {v*100:.2f}%",
    "MA_5":          lambda v: f"📐 5-day avg price: ${v:.2f}",
    "MA_20":         lambda v: f"📐 20-day avg price: ${v:.2f}",
    "MA_ratio":      lambda v: f"📊 Short vs Long Trend: {v:.3f}",
    "MA_diff":       lambda v: f"📏 MA gap: ${v:.2f}",
    "Price_vs_MA20": lambda v: f"📍 Price vs 20-day avg: {v:.3f}",
    "Volatility_5":  lambda v: f"🌊 5-day volatility: {v*100:.2f}%",
    "Volatility_10": lambda v: f"🌊 10-day volatility: {v*100:.2f}%",
    "Volume_MA5":    lambda v: f"📦 5-day avg volume: {v:,.0f}",
    "Volume_Ratio":  lambda v: f"📦 Volume vs 5d avg: {v:.2f}x",
    "Volume_Spike":  lambda v: f"🔊 Volume vs 20d avg: {v:.2f}x",
    "Momentum_3":    lambda v: f"🚀 3-day momentum: {v*100:.2f}%",
    "Momentum_10":   lambda v: f"🏃 10-day momentum: {v*100:.2f}%",
    "Momentum_20":   lambda v: f"🗓️ 1-month momentum: {v*100:.2f}%",
    "HL_Range":      lambda v: f"📏 High-Low range: {v*100:.2f}%",
    "Breakout_20":   lambda v: f"🚀 vs 20-day high: {v*100:.1f}%",
    "Drawdown_20":   lambda v: f"🛡️ vs 20-day low: {v*100:.1f}%",
}

# ── Model helpers ────────────────────────────────────────────────
def get_top_features(n: int = 3):
    if not HAS_IMPORTANCE:
        return []
    imp     = model.feature_importances_
    indices = np.argsort(imp)[::-1][:n]
    return [(FEATURE_COLS[i], imp[i]) for i in indices]

def predict_ticker(ticker: str, fetch_period: str = "6mo"):
    raw = fetch_stock(ticker, period=fetch_period)
    if raw.empty or len(raw) < 25:
        return None, None, None, None
    df     = compute_features(raw)
    latest = df.dropna(subset=FEATURE_COLS).iloc[-1]
    X      = latest[FEATURE_COLS].values.reshape(1, -1)
    signal = int(model.predict(X)[0])
    conf   = get_confidence(X, signal)
    return signal, conf, latest, df

# ── Backtest: past 3-month signal accuracy ───────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def backtest_accuracy(ticker: str, forward_days: int = 5) -> dict:
    """
    For each day in the past 3 months, compute features and predict a signal.
    Then check if the stock actually moved in the predicted direction forward_days later.
    Returns accuracy stats.
    """
    raw = fetch_stock(ticker, period="1y")
    if raw.empty or len(raw) < 60:
        return {}
    df = compute_features(raw)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    # Use last 3 months only for backtest evaluation
    cutoff = df["Date"].max() - pd.Timedelta(days=90)
    df_bt  = df[df["Date"] <= df["Date"].max() - pd.Timedelta(days=forward_days)]
    df_bt  = df_bt[df_bt["Date"] >= cutoff].reset_index(drop=True)

    if len(df_bt) < 10:
        return {}

    correct = 0
    total   = 0
    by_signal = {1: {"correct": 0, "total": 0}, 0: {"correct": 0, "total": 0}, -1: {"correct": 0, "total": 0}}

    for idx, row in df_bt.iterrows():
        X      = row[FEATURE_COLS].values.reshape(1, -1)
        signal = int(model.predict(X)[0])

        # Find actual return forward_days later
        future_idx = df[df["Date"] > row["Date"]].head(forward_days)
        if len(future_idx) < forward_days:
            continue
        future_ret = float(future_idx.iloc[-1]["Close"]) / float(row["Close"]) - 1

        # Check if signal was correct
        was_correct = (
            (signal == 1  and future_ret > 0.01) or
            (signal == -1 and future_ret < -0.01) or
            (signal == 0  and abs(future_ret) <= 0.01)
        )
        correct += int(was_correct)
        total   += 1
        by_signal[signal]["total"]   += 1
        by_signal[signal]["correct"] += int(was_correct)

    if total == 0:
        return {}
    return {
        "overall": round(correct / total * 100, 1),
        "total":   total,
        "by_signal": by_signal,
    }

# ── Reusable UI helpers ──────────────────────────────────────────
def render_signal_plain(signal: int):
    title, body = SIGNAL_PLAIN[signal]
    css = SIGNAL_CSS_CLASS[signal]
    st.markdown(
        f"<div class='{css}'><b>{title}</b><br><span>{body}</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='disclaimer'>{DISCLAIMER}</div>", unsafe_allow_html=True)

def render_confidence(confidence: float, color: str):
    if confidence is None:
        return
    # Context: random baseline for 3-class = 33%
    if confidence >= 60:
        note, icon, nc = "High confidence — model is fairly certain.", "✅", "#166534"
    elif confidence >= 45:
        note, icon, nc = "Moderate confidence — signal is plausible but not strong.", "🟡", "#854d0e"
    else:
        note, icon, nc = "Low confidence — close to random guessing (baseline ~33%). Treat with caution.", "🔺", "#991b1b"

    st.markdown("**Model Confidence**")
    st.markdown(
        f"<div style='background:#e5e7eb;border-radius:8px;height:22px;width:100%;overflow:hidden;margin-bottom:4px'>"
        f"<div style='width:{confidence}%;height:22px;background:{color};border-radius:8px;"
        f"display:flex;align-items:center;padding-left:10px'>"
        f"<span style='color:white;font-weight:700;font-size:13px'>{confidence:.1f}%</span></div></div>"
        f"<div style='font-size:13px;color:{nc};margin-top:2px'>{icon} {note}</div>"
        f"<div style='font-size:11px;color:#94a3b8;margin-top:3px'>"
        f"ℹ️ In a 3-class model (BUY / HOLD / SELL), random guessing scores ~33%. "
        f"Confidence above 50% indicates the model has a meaningful edge over chance."
        f"</div>",
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

    if "extra_tickers" not in st.session_state:
        st.session_state["extra_tickers"] = []
    if "selected_tickers" not in st.session_state:
        st.session_state["selected_tickers"] = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]

    st.markdown("**Search & add a stock**")
    search_input = st.text_input(
        "search_ticker",
        placeholder="Ticker or company name (e.g. Netflix, UBER)",
        label_visibility="collapsed",
    )

    if search_input.strip():
        with st.spinner("Searching…"):
            results = search_ticker(search_input.strip())
        if results:
            for sym, name in results[:3]:
                label = f"➕ {sym} — {name}"
                if st.button(label, key=f"add_{sym}", use_container_width=True):
                    if sym not in st.session_state["extra_tickers"]:
                        st.session_state["extra_tickers"].append(sym)
                    if sym not in st.session_state["selected_tickers"]:
                        st.session_state["selected_tickers"].append(sym)
                    st.rerun()
        else:
            st.caption(f"⚠️ No results for '{search_input.strip()}'. Try the exact ticker symbol.")

    all_tickers    = list(dict.fromkeys(DEFAULT_TICKERS + st.session_state["extra_tickers"]))
    valid_selected = [t for t in st.session_state["selected_tickers"] if t in all_tickers]

    selected = st.multiselect(
        "Stocks to analyse",
        options=all_tickers,
        default=valid_selected,
        format_func=ticker_label,
        key="stocks_multiselect",
    )
    st.session_state["selected_tickers"] = selected

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
        st.markdown(
            "<div class='home-section'>"
            "<h3>📦 Dataset & Training</h3>"
            f"<p>The model was trained on historical market data sourced via <b>yFinance</b>, "
            f"covering <b>{MODEL_TRAIN_INFO['companies']} large-cap {MODEL_TRAIN_INFO['index']} companies</b> "
            f"over the period <b>{MODEL_TRAIN_INFO['period']}</b> ({MODEL_TRAIN_INFO['source']}).</p>"
            "<div style='background:#fefce8;border-left:4px solid #eab308;border-radius:6px;padding:10px 14px;margin:10px 0;font-size:0.88rem;color:#713f12'>"
            "<b>⚠️ Accuracy Note</b><br>"
            "Predictions are most reliable for:<br>"
            "✅ S&P 500 components &nbsp; ✅ Large-cap U.S. equities &nbsp; ✅ U.S. ETFs (SPY, QQQ)<br>"
            "Small-cap stocks, international equities, or crypto ETFs may be less accurate — use with caution.<br><br>"
            "<b>Training period:</b> Data runs from 2021 through to the current date (2026), sourced live via yFinance."
            "</div>"
            "<p>We engineer <b>18 technical features</b> from raw OHLCV data:</p>",
            unsafe_allow_html=True,
        )
        feature_table = pd.DataFrame({
            "Category": [
                "Returns", "Returns",
                "Trend", "Trend", "Trend",
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
        st.dataframe(feature_table, use_container_width=True, hide_index=True, height=310)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            "<div class='home-section'>"
            "<h3>🧠 Model & Method</h3>"
            "<p>This app uses a <b>Random Forest Classifier</b> trained offline on labelled historical data. "
            "Each trading day is labelled as BUY, HOLD, or SELL based on subsequent price movement.</p>"
            "<div class='method-block'>"
            "<b>Random Forest Classifier</b><br>"
            "An ensemble of decision trees that votes on the most likely outcome. "
            "Outputs a probability distribution across BUY / HOLD / SELL — used to compute the <b>Confidence %</b>."
            "</div>"
            "<div class='method-block green'>"
            "<b>Confidence Score</b><br>"
            "In a 3-class problem, random guessing scores ~33%. "
            "Scores above 50% indicate the model has a meaningful signal. "
            "Scores near 33% are essentially noise — treat them with caution."
            "</div>"
            "<div class='method-block'>"
            "<b>Feature Importance</b><br>"
            "The model ranks which indicators matter most. "
            "The app surfaces the <b>Top 3 factors</b> driving each prediction."
            "</div>"
            "<p style='margin-top:14px;font-size:0.88rem;color:#7f1d1d;background:#fef2f2;"
            "border-left:4px solid #ef4444;border-radius:6px;padding:10px 14px'>"
            "⚠️ <b>Disclaimer:</b> This tool is for educational and informational purposes only. "
            "It is <b>not financial advice</b>. Always do your own research and consult a licensed "
            "financial advisor before making any investment decisions."
            "</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='home-section'>"
            "<h3>🗺️ How to Use</h3>"
            "<ol>"
            "<li>Search for a stock in the sidebar by ticker or company name</li>"
            "<li>Select stocks to analyse from the list</li>"
            "<li>Click <b>🔍 Get Recommendations</b> for a batch overview — sorted by signal strength</li>"
            "<li>Go to <b>Single Stock Detail</b> for a deep-dive: confidence, explanation, and top factors</li>"
            "<li>Use <b>Price Chart</b> to view price and moving averages across any time period</li>"
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
    st.success("✅ Model loaded" + (" &nbsp; 🎯 Confidence scoring enabled" if HAS_PROBA else ""))
    st.markdown(f"<div class='disclaimer'>{DISCLAIMER}</div>", unsafe_allow_html=True)

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
                    "Ticker": ticker, "Company": get_company_name(ticker),
                    "Close ($)": "—", "Signal": "⚠️ No data", "Confidence": "—",
                    "_sig": None, "_conf": -1,
                })
                continue
            results.append({
                "Ticker":     ticker,
                "Company":    get_company_name(ticker),
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
        valid = [r for r in res if r.get("_sig") is not None]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 BUY",  sum(1 for r in valid if r["_sig"] ==  1))
        c2.metric("🟡 HOLD", sum(1 for r in valid if r["_sig"] ==  0))
        c3.metric("🔴 SELL", sum(1 for r in valid if r["_sig"] == -1))
        avg_conf = np.mean([r["_conf"] for r in valid]) if valid else 0
        c4.metric("🎯 Avg Confidence", f"{avg_conf:.1f}%")

        st.markdown("---")

        # Sort options
        sort_by = st.radio(
            "Sort by",
            ["Signal (BUY first)", "Confidence (highest first)", "Default order"],
            horizontal=True,
            index=0,
        )

        def sort_key(r):
            if sort_by == "Signal (BUY first)":
                # BUY=1 first, then HOLD=0, then SELL=-1, then No data
                order = {1: 0, 0: 1, -1: 2, None: 3}
                return (order.get(r.get("_sig"), 3), -r.get("_conf", -1))
            elif sort_by == "Confidence (highest first)":
                return -r.get("_conf", -1)
            return 0  # default

        sorted_res = sorted(res, key=sort_key)
        display = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in sorted_res])
        st.dataframe(display, use_container_width=True, hide_index=True)

        if HAS_PROBA and valid:
            st.markdown("#### 🎯 Confidence by Stock")
            # Sort chart same way
            sorted_valid = sorted(valid, key=sort_key)
            fig_c, ax_c  = plt.subplots(figsize=(10, max(2.5, len(sorted_valid) * 0.4)))
            tickers_ = [r["Ticker"] for r in sorted_valid]
            confs_   = [r["_conf"]  for r in sorted_valid]
            colors_  = [SIGNAL_COLOR[r["_sig"]] for r in sorted_valid]
            bars = ax_c.barh(tickers_, confs_, color=colors_, alpha=0.85)
            ax_c.set_xlim(0, 100)
            ax_c.set_xlabel("Confidence (%)")
            ax_c.axvline(33, color="#94a3b8", linestyle=":", linewidth=1, label="Random baseline (33%)")
            ax_c.axvline(50, color="gray",    linestyle="--", alpha=0.5, linewidth=1, label="50% threshold")
            for bar, v in zip(bars, confs_):
                ax_c.text(v + 1, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9)
            ax_c.legend(fontsize=8)
            ax_c.grid(axis="x", alpha=0.2)
            fig_c.tight_layout()
            st.pyplot(fig_c)
            plt.close(fig_c)

# ════════════════════════════════════════════════════════════════
# Tab 2 — Single Stock Detail  (auto-analyse on ticker change)
# ════════════════════════════════════════════════════════════════
with tab_detail:
    pick = st.selectbox(
        "Choose a stock",
        options=selected if selected else DEFAULT_TICKERS[:5],
        format_func=ticker_label,
        key="detail_pick",
    )

    # Auto-analyse when ticker changes OR on first load
    if st.session_state.get("detail_last_pick") != pick:
        st.session_state["detail_last_pick"]   = pick
        st.session_state["detail_result"]      = None  # clear stale result
        st.session_state["detail_needs_fetch"] = True

    if st.session_state.get("detail_needs_fetch", True):
        with st.spinner(f"Analysing {ticker_label(pick)}…"):
            signal, confidence, latest, df_feat = predict_ticker(pick)
        st.session_state["detail_result"]      = (signal, confidence, latest, df_feat)
        st.session_state["detail_needs_fetch"] = False
    else:
        signal, confidence, latest, df_feat = st.session_state.get("detail_result", (None, None, None, None))

    # Refresh button
    if st.button("🔄 Refresh", key="analyse_btn"):
        with st.spinner(f"Refreshing {ticker_label(pick)}…"):
            signal, confidence, latest, df_feat = predict_ticker(pick)
        st.session_state["detail_result"] = (signal, confidence, latest, df_feat)

    if signal is None:
        st.error(f"Could not fetch data for **{ticker_label(pick)}**. Check the ticker symbol.")
    else:
        color   = SIGNAL_COLOR[signal]
        co_name = get_company_name(pick)
        conf_text = (
            f"<div style='font-size:1.1rem;color:{color}99;margin-top:6px'>"
            f"Confidence: {confidence:.1f}%</div>"
        ) if confidence is not None else ""

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

        # Backtest accuracy
        st.markdown("---")
        st.markdown("#### 📊 Historical Signal Accuracy (past 3 months)")
        with st.spinner("Running backtest…"):
            bt = backtest_accuracy(pick)
        if bt:
            ba1, ba2, ba3, ba4 = st.columns(4)
            ba1.metric("Overall Accuracy", f"{bt['overall']}%")
            ba2.metric("🟢 BUY  accuracy",
                       f"{round(bt['by_signal'][1]['correct']/bt['by_signal'][1]['total']*100,1)}%" if bt['by_signal'][1]['total'] else "—")
            ba3.metric("🟡 HOLD accuracy",
                       f"{round(bt['by_signal'][0]['correct']/bt['by_signal'][0]['total']*100,1)}%" if bt['by_signal'][0]['total'] else "—")
            ba4.metric("🔴 SELL accuracy",
                       f"{round(bt['by_signal'][-1]['correct']/bt['by_signal'][-1]['total']*100,1)}%" if bt['by_signal'][-1]['total'] else "—")
            st.caption(f"Based on {bt['total']} trading days. A signal is 'correct' if the stock moved >1% in the predicted direction within 5 trading days.")
        else:
            st.caption("Not enough data to compute backtest accuracy.")

        st.markdown("---")
        st.markdown("#### 📋 Feature Values Used for Prediction")
        feat_df = pd.DataFrame({
            "Feature":     FEATURE_COLS,
            "Value":       [round(float(latest[f]), 6) for f in FEATURE_COLS],
            "Description": [FEATURE_DESCRIBE.get(f, "") for f in FEATURE_COLS],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True, height=320)

# ════════════════════════════════════════════════════════════════
# Tab 3 — Price Chart
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
                st.error("Not enough data for the selected period. Try a longer window.")
            else:
                period_label = DISPLAY_PERIOD_LABEL[chart_display_period]
                co_name      = get_company_name(chart_ticker)

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df_plot["Date"], df_plot["Close"], label="Close", color="#3b82f6", linewidth=1.8)
                ax.plot(df_plot["Date"], df_plot["MA_5"],  label="MA 5",  color="#f97316", linewidth=1.2, linestyle="--")
                ax.plot(df_plot["Date"], df_plot["MA_20"], label="MA 20", color="#a855f7", linewidth=1.2, linestyle="--")

                # ── Mark today's signal on the chart ──────────────────
                if sig is not None:
                    last_date  = df_plot["Date"].iloc[-1]
                    last_close = float(df_plot["Close"].iloc[-1])
                    sig_color  = SIGNAL_COLOR[sig]
                    sig_marker = {1: "^", 0: "o", -1: "v"}[sig]
                    sig_offset = {1: 1.01, 0: 1.0, -1: 0.99}[sig]
                    ax.scatter(last_date, last_close * sig_offset,
                               marker=sig_marker, color=sig_color, s=150, zorder=5,
                               label=f"Signal: {SIGNAL_LABEL[sig]}")
                    ax.axvline(last_date, color=sig_color, linestyle=":", linewidth=1.2, alpha=0.6)

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
