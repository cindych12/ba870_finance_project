# ================================================================
# app.py  —  Stock Investment Decision Support
# Loads model.pkl (trained offline), fetches live data via yfinance,
# computes features, and shows Buy / Hold / Sell recommendation.
# ================================================================

import pickle
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Investment Decision Support",
    page_icon="📈",
    layout="wide",
)

# ── Constants ────────────────────────────────────────────────────
FEATURE_COLS = [
    "Return_1d", "Return_5d",
    "MA_5", "MA_20", "MA_ratio", "MA_diff", "Price_vs_MA20",
    "Volatility_5", "Volume_MA5", "Volume_Ratio",
    "Momentum_3", "Momentum_10", "Momentum_20",
    "HL_Range", "Volatility_10", "Breakout_20", "Drawdown_20",
    "Volume_Spike",
]

SIGNAL_LABEL = { 1: "BUY",  0: "HOLD", -1: "SELL" }
SIGNAL_EMOJI = { 1: "🟢",   0: "🟡",    -1: "🔴"  }
SIGNAL_COLOR = { 1: "#22c55e", 0: "#eab308", -1: "#ef4444" }

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "JPM", "JNJ", "XOM", "WMT",
    "META", "AMD", "BAC", "GS", "COST",
    "DIS", "CVX", "CAT", "BA", "PFE",
    "SPY", "QQQ",
]

# ── Load model ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model = load_model()

# ── Feature engineering (mirrors notebook exactly) ───────────────
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
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[["Open", "High", "Low", "Close", "Volume"]].reset_index()
    date_col = next((c for c in df.columns if c.lower() in {"date", "datetime"}), None)
    if date_col and date_col != "Date":
        df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df.sort_values("Date").reset_index(drop=True)

# ── Predict for one ticker ───────────────────────────────────────
def predict_ticker(ticker: str):
    """Returns (signal_int, latest_row_series, full_df_with_features)"""
    raw = fetch_stock(ticker, period="6mo")
    if raw.empty or len(raw) < 25:
        return None, None, None
    df = compute_features(raw)
    latest = df.dropna(subset=FEATURE_COLS).iloc[-1]
    X = latest[FEATURE_COLS].values.reshape(1, -1)
    signal = int(model.predict(X)[0])
    return signal, latest, df

# ════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════

st.title("📈 Stock Investment Decision Support")
st.markdown("Loads a pre-trained Random Forest → fetches latest market data → gives **Buy / Hold / Sell** signal.")

# ── Model status banner ──────────────────────────────────────────
if model is None:
    st.error(
        "⚠️ model.pkl not found. "
        "Please run python train_model.py first, then restart the app."
    )
    st.stop()
else:
    st.success("✅ Model loaded from `model.pkl`")

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    custom_raw = st.text_input("Add custom tickers (comma-separated)", placeholder="e.g. NFLX, UBER")
    extra = [t.strip().upper() for t in custom_raw.split(",") if t.strip()]
    all_tickers = list(dict.fromkeys(DEFAULT_TICKERS + extra))

    selected = st.multiselect(
        "Stocks to analyse",
        options=all_tickers,
        default=["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"],
    )
    run_btn = st.button("🔍 Get Recommendations", type="primary")

# ── Tabs ─────────────────────────────────────────────────────────
tab_rec, tab_detail, tab_chart = st.tabs(
    ["🔮 Recommendations", "🔍 Single Stock Detail", "📉 Price Chart"]
)

# ════════════════════════════════════════════════════════════════
# Tab 1 — Recommendations (batch)
# ════════════════════════════════════════════════════════════════
with tab_rec:
    if not selected:
        st.info("Select stocks in the sidebar and click **Get Recommendations**.")
    elif run_btn or "results" not in st.session_state:
        if not selected:
            st.stop()
        results = []
        progress = st.progress(0, text="Fetching data…")
        for i, ticker in enumerate(selected):
            progress.progress((i + 1) / len(selected), text=f"Processing {ticker}…")
            signal, latest, _ = predict_ticker(ticker)
            if signal is None:
                results.append({"Ticker": ticker, "Close ($)": "—", "Signal": "⚠️ No data"})
                continue
            results.append({
                "Ticker":    ticker,
                "Close ($)": round(float(latest["Close"]), 2),
                "Signal":    f"{SIGNAL_EMOJI[signal]} {SIGNAL_LABEL[signal]}",
                "_sig":      signal,
            })
        progress.empty()
        st.session_state["results"] = results
        st.session_state["selected"] = selected

    if "results" in st.session_state:
        res = st.session_state["results"]
        valid = [r for r in res if "_sig" in r]

        # Summary cards
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 BUY",  sum(1 for r in valid if r["_sig"] ==  1))
        c2.metric("🟡 HOLD", sum(1 for r in valid if r["_sig"] ==  0))
        c3.metric("🔴 SELL", sum(1 for r in valid if r["_sig"] == -1))

        st.markdown("---")

        display = pd.DataFrame([{k: v for k, v in r.items() if k != "_sig"} for r in res])
        st.dataframe(display, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════
# Tab 2 — Single Stock Detail
# ════════════════════════════════════════════════════════════════
with tab_detail:
    pick = st.selectbox("Choose a stock", options=selected if selected else DEFAULT_TICKERS[:5])

    if st.button("Analyse", key="analyse_btn"):
        with st.spinner(f"Fetching {pick}…"):
            signal, latest, df_feat = predict_ticker(pick)

        if signal is None:
            st.error(f"Could not fetch data for **{pick}**. Check the ticker symbol.")
        else:
            color = SIGNAL_COLOR[signal]
            label = SIGNAL_LABEL[signal]
            emoji = SIGNAL_EMOJI[signal]
            st.markdown(
                f"<div style='text-align:center;padding:20px;border-radius:12px;"
                f"background:{color}22;border:2px solid {color}'>"
                f"<span style='font-size:3rem'>{emoji}</span><br>"
                f"<span style='font-size:2rem;font-weight:700;color:{color}'>{pick}: {label}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown("")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Latest Close",     f"${latest['Close']:.2f}")
            m2.metric("1-Day Return",     f"{latest['Return_1d']*100:.2f}%")
            m3.metric("5-Day Return",     f"{latest['Return_5d']*100:.2f}%")
            m4.metric("MA Ratio (5/20)",  f"{latest['MA_ratio']:.4f}")

            st.markdown("#### Feature Values Used for Prediction")
            feat_df = pd.DataFrame({
                "Feature": FEATURE_COLS,
                "Value":   [round(float(latest[f]), 6) for f in FEATURE_COLS],
            })
            st.dataframe(feat_df, use_container_width=True, hide_index=True, height=320)

# ════════════════════════════════════════════════════════════════
# Tab 3 — Price Chart
# ════════════════════════════════════════════════════════════════
with tab_chart:
    chart_ticker = st.selectbox(
        "Select stock", options=selected if selected else DEFAULT_TICKERS[:5], key="chart_sel"
    )
    days = st.slider("Days to display", 30, 180, 90)

    if st.button("Show Chart", key="chart_btn"):
        with st.spinner(f"Loading {chart_ticker}…"):
            sig, latest, df_feat = predict_ticker(chart_ticker)

        if df_feat is None:
            st.error(f"Could not fetch data for **{chart_ticker}**.")
        else:
            df_plot = df_feat.dropna(subset=["MA_5", "MA_20"]).tail(days)
