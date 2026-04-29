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

# ── Constants ────────────────────────────────────────────────────
FEATURE_COLS = [
    "Return_1d", "Return_5d",
    "MA_5", "MA_20", "MA_ratio", "MA_diff", "Price_vs_MA20",
    "Volatility_5", "Volume_MA5", "Volume_Ratio",
    "Momentum_3", "Momentum_10", "Momentum_20",
    "HL_Range", "Volatility_10", "Breakout_20", "Drawdown_20",
    "Volume_Spike",
]

FEATURE_EXPLAIN = {
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

SIGNAL_LABEL = { 1: "BUY",  0: "HOLD", -1: "SELL" }
SIGNAL_EMOJI = { 1: "🟢",   0: "🟡",    -1: "🔴"  }
SIGNAL_COLOR = { 1: "#22c55e", 0: "#eab308", -1: "#ef4444" }

PERIOD_OPTIONS = {
    "1 Week":   "5d",
    "1 Month":  "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year":   "1y",
}

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
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
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

# ── Explanation helper ───────────────────────────────────────────
def _explain_feature(feat: str, val: float, signal: int) -> str:
    """Return explanation string if this feature supports the signal, else empty string."""
    if signal == 1:  # BUY
        if feat == "MA_ratio"      and val > 1.02:  return f"MA5/MA20 = {val:.3f} → short-term trend above long-term"
        if feat == "Momentum_10"   and val > 0.05:  return f"10-day momentum = {val*100:.1f}% → strong upward momentum"
        if feat == "Momentum_20"   and val > 0.05:  return f"20-day momentum = {val*100:.1f}% → sustained upward trend"
        if feat == "Return_5d"     and val > 0.03:  return f"5-day return = {val*100:.1f}% → recent price strength"
        if feat == "Volume_Spike"  and val > 1.3:   return f"Volume spike = {val:.2f}x avg → strong buying interest"
        if feat == "Breakout_20"   and val > 0.97:  return f"Near 20-day high ({val:.3f}) → potential breakout"
        if feat == "Price_vs_MA20" and val > 1.02:  return f"Price {val:.3f}x above 20-day MA → bullish positioning"

    elif signal == -1:  # SELL
        if feat == "MA_ratio"      and val < 0.98:  return f"MA5/MA20 = {val:.3f} → short-term trend below long-term"
        if feat == "Momentum_10"   and val < -0.05: return f"10-day momentum = {val*100:.1f}% → strong downward momentum"
        if feat == "Momentum_20"   and val < -0.05: return f"20-day momentum = {val*100:.1f}% → sustained downward trend"
        if feat == "Return_5d"     and val < -0.03: return f"5-day return = {val*100:.1f}% → recent price weakness"
        if feat == "Drawdown_20"   and val < 1.03:  return f"Near 20-day low ({val:.3f}) → bearish pressure"
        if feat == "Price_vs_MA20" and val < 0.98:  return f"Price {val:.3f}x below 20-day MA → bearish positioning"

    else:  # HOLD
        if feat == "MA_ratio"      and 0.99 < val < 1.01: return f"MA5/MA20 = {val:.3f} → no clear trend direction"
        if feat == "Volatility_10" and val > 0.02:         return f"10-day volatility = {val*100:.2f}% → uncertain market"
        if feat == "Volume_Ratio"  and 0.8 < val < 1.2:   return f"Volume ratio = {val:.2f} → no volume confirmation"

    return ""

# ── Predict + confidence + explanation ──────────────────────────
def predict_ticker(ticker: str):
    """Returns (signal, confidence, top_reasons, latest_row, full_df)"""
    # Always fetch 6mo for enough warmup rows (20-day rolling)
    raw = fetch_stock(ticker, period="6mo")
    if raw.empty or len(raw) < 25:
        return None, None, None, None, None

    df     = compute_features(raw)
    latest = df.dropna(subset=FEATURE_COLS).iloc[-1]
    X      = latest[FEATURE_COLS].values.reshape(1, -1)

    signal = int(model.predict(X)[0])

    # Confidence from predict_proba
    proba      = model.predict_proba(X)[0]
    classes    = list(model.classes_)          # e.g. [-1, 0, 1]
    sig_idx    = classes.index(signal)
    confidence = round(float(proba[sig_idx]) * 100, 1)

    # Top 3 explanations ranked by feature importance
    importances = model.feature_importances_
    feat_vals   = latest[FEATURE_COLS].values
    reasons = []
    for feat, imp, val in zip(FEATURE_COLS, importances, feat_vals):
        expl = _explain_feature(feat, float(val), signal)
        if expl:
            reasons.append((feat, imp, expl))
    reasons.sort(key=lambda x: x[1], reverse=True)
    top_reasons = [(f, e) for f, _, e in reasons[:3]]

    # Fallback: if no directional reasons matched, show top features by importance
    if not top_reasons:
        top_idx = np.argsort(importances)[::-1][:3]
        top_reasons = [
            (FEATURE_COLS[i], f"{FEATURE_EXPLAIN[FEATURE_COLS[i]]}: {feat_vals[i]:.4f}")
            for i in top_idx
        ]

    return signal, confidence, top_reasons, latest, df


# ════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════
st.title("📈 Stock Investment Decision Support")
st.markdown("Loads a pre-trained Random Forest → fetches latest market data → gives **Buy / Hold / Sell** signal.")

if model is None:
    st.error("⚠️ `model.pkl` not found. Please run `python train_model.py` first.")
    st.stop()
else:
    st.success("✅ Model loaded from `model.pkl`")

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # ★ Period selector
    period_label = st.selectbox(
        "📅 Chart Period",
        options=list(PERIOD_OPTIONS.keys()),
        index=3,   # default: 6 Months
    )
    chart_period = PERIOD_OPTIONS[period_label]

    custom_raw = st.text_input("Add custom tickers (comma-separated)", placeholder="e.g. NFLX, UBER")
    extra      = [t.strip().upper() for t in custom_raw.split(",") if t.strip()]
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
# Tab 1 — Recommendations
# ════════════════════════════════════════════════════════════════
with tab_rec:
    if not selected:
        st.info("Select stocks in the sidebar and click **Get Recommendations**.")
    elif run_btn or "results" not in st.session_state:
        results  = []
        progress = st.progress(0, text="Fetching data…")
        for i, ticker in enumerate(selected):
            progress.progress((i + 1) / len(selected), text=f"Processing {ticker}…")
            signal, confidence, top_reasons, latest, _ = predict_ticker(ticker)
            if signal is None:
                results.append({"Ticker": ticker, "Close ($)": "—", "Signal": "⚠️ No data",
                                 "Confidence": "—", "_sig": None})
                continue
            results.append({
                "Ticker":     ticker,
                "Close ($)":  round(float(latest["Close"]), 2),
                "Signal":     f"{SIGNAL_EMOJI[signal]} {SIGNAL_LABEL[signal]}",
                "Confidence": f"{confidence}%",   # ★ NEW
                "_sig":       signal,
                "_conf":      confidence,
                "_reasons":   top_reasons,
            })
        progress.empty()
        st.session_state["results"] = results

    if "results" in st.session_state:
        res   = st.session_state["results"]
        valid = [r for r in res if r.get("_sig") is not None]

        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 BUY",  sum(1 for r in valid if r["_sig"] ==  1))
        c2.metric("🟡 HOLD", sum(1 for r in valid if r["_sig"] ==  0))
        c3.metric("🔴 SELL", sum(1 for r in valid if r["_sig"] == -1))
        st.markdown("---")

        # Results table
        display = pd.DataFrame([
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in res
        ])
        st.dataframe(display, use_container_width=True, hide_index=True)

        # ★ NEW: Expandable signal explanations
        st.markdown("### 📋 Signal Explanations")
        for r in valid:
            color     = SIGNAL_COLOR[r["_sig"]]
            conf      = r["_conf"]
            bar_color = "#22c55e" if conf >= 60 else "#eab308" if conf >= 40 else "#ef4444"
            with st.expander(f"{r['Ticker']}  —  {SIGNAL_EMOJI[r['_sig']]} {SIGNAL_LABEL[r['_sig']]}  |  Confidence: {conf}%"):
                st.markdown(
                    f"<div style='background:#333;border-radius:8px;height:18px;width:100%'>"
                    f"<div style='background:{bar_color};width:{conf}%;height:18px;border-radius:8px;"
                    f"display:flex;align-items:center;padding-left:8px'>"
                    f"<span style='color:white;font-size:12px;font-weight:bold'>{conf}%</span></div></div>",
                    unsafe_allow_html=True,
                )
                st.markdown("**Top reasons for this signal:**")
                for feat, explanation in r["_reasons"]:
                    st.markdown(f"- **{feat}**: {explanation}")

# ════════════════════════════════════════════════════════════════
# Tab 2 — Single Stock Detail
# ════════════════════════════════════════════════════════════════
with tab_detail:
    pick = st.selectbox("Choose a stock", options=selected if selected else DEFAULT_TICKERS[:5])

    if st.button("Analyse", key="analyse_btn"):
        with st.spinner(f"Fetching {pick}…"):
            signal, confidence, top_reasons, latest, df_feat = predict_ticker(pick)

        if signal is None:
            st.error(f"Could not fetch data for **{pick}**.")
        else:
            color = SIGNAL_COLOR[signal]
            label = SIGNAL_LABEL[signal]
            emoji = SIGNAL_EMOJI[signal]

            # Signal badge
            st.markdown(
                f"<div style='text-align:center;padding:20px;border-radius:12px;"
                f"background:{color}22;border:2px solid {color}'>"
                f"<span style='font-size:3rem'>{emoji}</span><br>"
                f"<span style='font-size:2rem;font-weight:700;color:{color}'>{pick}: {label}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown("")

            # ★ NEW: Confidence bar
            bar_color = "#22c55e" if confidence >= 60 else "#eab308" if confidence >= 40 else "#ef4444"
            st.markdown("**Model Confidence**")
            st.markdown(
                f"<div style='background:#333;border-radius:8px;height:24px;width:100%'>"
                f"<div style='background:{bar_color};width:{confidence}%;height:24px;border-radius:8px;"
                f"display:flex;align-items:center;padding-left:10px'>"
                f"<span style='color:white;font-weight:bold'>{confidence}%</span></div></div>",
                unsafe_allow_html=True,
            )
            st.markdown("")

            # ★ NEW: Explanation
            st.markdown("**📌 Why this signal?**")
            for feat, explanation in top_reasons:
                st.markdown(f"- **{feat}**: {explanation}")
            st.markdown("")

            # Key metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Latest Close",    f"${latest['Close']:.2f}")
            m2.metric("1-Day Return",    f"{latest['Return_1d']*100:.2f}%")
            m3.metric("5-Day Return",    f"{latest['Return_5d']*100:.2f}%")
            m4.metric("MA Ratio (5/20)", f"{latest['MA_ratio']:.4f}")

            # Feature table with descriptions
            st.markdown("#### Feature Values Used for Prediction")
            feat_df = pd.DataFrame({
                "Feature":     FEATURE_COLS,
                "Value":       [round(float(latest[f]), 6) for f in FEATURE_COLS],
                "Description": [FEATURE_EXPLAIN[f] for f in FEATURE_COLS],
            })
            st.dataframe(feat_df, use_container_width=True, hide_index=True, height=340)

# ════════════════════════════════════════════════════════════════
# Tab 3 — Price Chart  (uses sidebar period selector)
# ════════════════════════════════════════════════════════════════
with tab_chart:
    chart_ticker = st.selectbox(
        "Select stock", options=selected if selected else DEFAULT_TICKERS[:5], key="chart_sel"
    )
    st.caption(f"Displaying: **{period_label}**  (change period in the sidebar ←)")

    if st.button("Show Chart", key="chart_btn"):
        with st.spinner(f"Loading {chart_ticker}…"):
            raw_chart = fetch_stock(chart_ticker, period=chart_period)
            sig, confidence, top_reasons, latest, _ = predict_ticker(chart_ticker)

        if raw_chart.empty:
            st.error(f"Could not fetch data for **{chart_ticker}**.")
        else:
            df_plot = compute_features(raw_chart).dropna(subset=["MA_5", "MA_20"])

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_plot["Date"], df_plot["Close"], label="Close", color="#3b82f6", linewidth=1.8)
            ax.plot(df_plot["Date"], df_plot["MA_5"],  label="MA 5",  color="#f97316", linewidth=1.2, linestyle="--")
            ax.plot(df_plot["Date"], df_plot["MA_20"], label="MA 20", color="#a855f7", linewidth=1.2, linestyle="--")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            plt.xticks(rotation=30, ha="right")
            ax.set_title(f"{chart_ticker} — {period_label}")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Signal + confidence + explanation below chart
            if sig is not None:
                color     = SIGNAL_COLOR[sig]
                bar_color = "#22c55e" if confidence >= 60 else "#eab308" if confidence >= 40 else "#ef4444"
                st.markdown(
                    f"**Signal:** <span style='color:{color};font-size:1.2rem;font-weight:bold'>"
                    f"{SIGNAL_EMOJI[sig]} {SIGNAL_LABEL[sig]}</span> &nbsp;|&nbsp; "
                    f"**Confidence:** <span style='color:{bar_color};font-weight:bold'>{confidence}%</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("**📌 Why this signal?**")
                for feat, explanation in top_reasons:
                    st.markdown(f"- **{feat}**: {explanation}")
