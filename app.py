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
    """
    Return a beginner-friendly (emoji + plain English) explanation for this feature.
    Every feature always returns a string — no empty fallback.
    """
    pct = val * 100  # convert ratio to percentage where relevant

    explanations = {
        # ── Trend indicators ──────────────────────────────────────────
        "MA_ratio": (
            f"📊 Short-term vs Long-term Trend: {val:.3f}\n"
            f"We compare the 5-day average price to the 20-day average price. "
            f"{'A value above 1.0 means the stock has been rising recently — a bullish sign. 📈' if val > 1.0 else 'A value below 1.0 means the stock has been falling recently — a bearish sign. 📉'}"
        ),
        "MA_diff": (
            f"📏 Gap between short & long trend: ${val:.2f}\n"
            f"{'The short-term average is ABOVE the long-term average by ${:.2f}, suggesting upward momentum.'.format(abs(val)) if val > 0 else 'The short-term average is BELOW the long-term average by ${:.2f}, suggesting downward pressure.'.format(abs(val))}"
        ),
        "Price_vs_MA20": (
            f"📍 Price vs 20-day average: {val:.3f}\n"
            f"Think of the 20-day average as the stock's 'fair value' over the past month. "
            f"{'The current price is {:.1f}% ABOVE this average — the stock is running hot. 🔥'.format((val-1)*100) if val > 1 else 'The current price is {:.1f}% BELOW this average — the stock may be undervalued. 💡'.format((1-val)*100)}"
        ),
        # ── Momentum indicators ───────────────────────────────────────
        "Return_1d": (
            f"⚡ Yesterday's price change: {pct:.2f}%\n"
            f"{'The stock went UP {:.2f}% yesterday — recent positive momentum.'.format(abs(pct)) if val > 0 else 'The stock went DOWN {:.2f}% yesterday — recent negative momentum.'.format(abs(pct))}"
        ),
        "Return_5d": (
            f"📅 Last 5 days price change: {pct:.2f}%\n"
            f"{'Over the past week, the stock is UP {:.2f}% — short-term strength.'.format(abs(pct)) if val > 0 else 'Over the past week, the stock is DOWN {:.2f}% — short-term weakness.'.format(abs(pct))}"
        ),
        "Momentum_3": (
            f"🚀 3-day momentum: {pct:.2f}%\n"
            f"{'The stock gained {:.2f}% in 3 days — very recent buying pressure.'.format(abs(pct)) if val > 0 else 'The stock lost {:.2f}% in 3 days — very recent selling pressure.'.format(abs(pct))}"
        ),
        "Momentum_10": (
            f"🏃 10-day momentum: {pct:.2f}%\n"
            f"{'Up {:.2f}% over 2 weeks — the uptrend has some staying power.'.format(abs(pct)) if val > 0 else 'Down {:.2f}% over 2 weeks — the downtrend has some staying power.'.format(abs(pct))}"
        ),
        "Momentum_20": (
            f"🗓️ 1-month momentum: {pct:.2f}%\n"
            f"{'Up {:.2f}% over the past month — a sustained upward trend.'.format(abs(pct)) if val > 0 else 'Down {:.2f}% over the past month — a sustained downward trend.'.format(abs(pct))}"
        ),
        # ── Moving averages (absolute levels, less informative alone) ─
        "MA_5": (
            f"📐 5-day average price: ${val:.2f}\n"
            f"This is the average closing price over the last 5 trading days. Used together with MA_20 to detect trend direction."
        ),
        "MA_20": (
            f"📐 20-day average price: ${val:.2f}\n"
            f"This is the average closing price over the last 20 trading days (~1 month). Acts as a support/resistance reference."
        ),
        # ── Volatility indicators ─────────────────────────────────────
        "Volatility_5": (
            f"🌊 5-day price swings: {pct:.2f}%\n"
            f"{'High volatility ({:.2f}%) — the stock has been moving a lot this week. Higher risk, higher reward potential.'.format(abs(pct)) if val > 0.02 else 'Low volatility ({:.2f}%) — the stock has been relatively calm and stable this week.'.format(abs(pct))}"
        ),
        "Volatility_10": (
            f"🌊 10-day price swings: {pct:.2f}%\n"
            f"{'High volatility ({:.2f}%) — the stock has been moving a lot over 2 weeks. Signals an uncertain market.'.format(abs(pct)) if val > 0.02 else 'Low volatility ({:.2f}%) — the stock has been relatively stable over 2 weeks.'.format(abs(pct))}"
        ),
        "HL_Range": (
            f"📏 Today's High-Low range: {pct:.2f}%\n"
            f"This measures how wide the price moved today (from lowest to highest). "
            f"{'A wide range ({:.2f}%) means investors are uncertain or reacting to news — more risk today.'.format(abs(pct)) if val > 0.02 else 'A narrow range ({:.2f}%) means calm, steady trading today.'.format(abs(pct))}"
        ),
        # ── Volume indicators ─────────────────────────────────────────
        "Volume_MA5": (
            f"📦 Average daily trading volume (5-day): {val:,.0f} shares\n"
            f"This is how many shares are being traded per day on average. Higher volume = more investor interest."
        ),
        "Volume_Ratio": (
            f"🔊 Today's volume vs recent average: {val:.2f}x\n"
            f"{'Today {:.2f}x MORE shares were traded than usual — strong investor activity. 🔥'.format(val) if val > 1.2 else 'Today {:.2f}x FEWER shares were traded than usual — quiet market, weak conviction.'.format(val) if val < 0.8 else 'Volume is normal today ({:.2f}x average) — no unusual buying or selling pressure.'.format(val)}"
        ),
        "Volume_Spike": (
            f"💥 Volume spike vs 20-day average: {val:.2f}x\n"
            f"{'Volume is {:.2f}x the monthly average — something significant may be happening (news, earnings, etc.).'.format(val) if val > 1.5 else 'Volume is slightly above average ({:.2f}x) — mild increase in investor interest.'.format(val) if val > 1.1 else 'Volume is normal or below average ({:.2f}x) — no major unusual activity.'.format(val)}"
        ),
        # ── Breakout / Drawdown ───────────────────────────────────────
        "Breakout_20": (
            f"🏔️ How close to 20-day HIGH: {val:.3f}\n"
            f"A value of 1.0 means the stock is at its highest price in 20 days. "
            f"{'At {:.1f}% of its 20-day high — very close to breaking out to new highs! 🚀'.format(val*100) if val > 0.97 else 'At {:.1f}% of its 20-day high — still {:.1f}% away from a new high.'.format(val*100, (1-val)*100)}"
        ),
        "Drawdown_20": (
            f"🕳️ How far from 20-day LOW: {val:.3f}\n"
            f"A value of 1.0 means the stock is at its lowest price in 20 days. "
            f"{'At {:.1f}x the 20-day low — the stock has recovered well from its recent bottom. ✅'.format(val) if val > 1.1 else 'Very close to the 20-day low ({:.3f}) — the stock may be under continued selling pressure. ⚠️'.format(val)}"
        ),
    }

    return explanations.get(feat, f"{feat}: {val:.4f}")

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

    # Top 3 features by importance — every feature always has an explanation now
    importances = model.feature_importances_
    feat_vals   = latest[FEATURE_COLS].values
    top_idx     = np.argsort(importances)[::-1][:3]
    top_reasons = [
        (FEATURE_COLS[i], _explain_feature(FEATURE_COLS[i], float(feat_vals[i]), signal))
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

        # Expandable signal explanations
        st.markdown("### 📋 Signal Explanations")
        for r in valid:
            color     = SIGNAL_COLOR[r["_sig"]]
            conf      = r["_conf"]
            bar_color = "#22c55e" if conf >= 60 else "#eab308" if conf >= 40 else "#ef4444"
            conf_text = "High confidence" if conf >= 60 else "Medium confidence" if conf >= 40 else "Low confidence — treat with caution"
            with st.expander(f"{r['Ticker']}  —  {SIGNAL_EMOJI[r['_sig']]} {SIGNAL_LABEL[r['_sig']]}  |  Confidence: {conf}%"):
                # Confidence bar + label
                st.markdown(
                    f"<div style='background:#333;border-radius:8px;height:22px;width:100%'>"
                    f"<div style='background:{bar_color};width:{conf}%;height:22px;border-radius:8px;"
                    f"display:flex;align-items:center;padding-left:10px'>"
                    f"<span style='color:white;font-size:13px;font-weight:bold'>{conf}%</span></div></div>"
                    f"<p style='color:{bar_color};margin-top:4px;font-size:13px'>⬆ {conf_text}</p>",
                    unsafe_allow_html=True,
                )
                st.markdown("**🔍 Top 3 reasons the model gave this signal:**")
                for feat, explanation in r["_reasons"]:
                    lines = explanation.split("\n")
                    title_line = lines[0]
                    detail_line = lines[1] if len(lines) > 1 else ""
                    st.markdown(
                        f"<div style='background:#1e1e1e;border-left:4px solid {color};"
                        f"border-radius:6px;padding:10px 14px;margin:6px 0'>"
                        f"<div style='font-weight:bold;font-size:14px'>{title_line}</div>"
                        f"<div style='color:#aaa;font-size:13px;margin-top:4px'>{detail_line}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

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

            # Confidence bar
            bar_color = "#22c55e" if confidence >= 60 else "#eab308" if confidence >= 40 else "#ef4444"
            conf_text = "High confidence" if confidence >= 60 else "Medium confidence" if confidence >= 40 else "Low confidence — treat with caution"
            st.markdown("**Model Confidence**")
            st.markdown(
                f"<div style='background:#333;border-radius:8px;height:24px;width:100%'>"
                f"<div style='background:{bar_color};width:{confidence}%;height:24px;border-radius:8px;"
                f"display:flex;align-items:center;padding-left:10px'>"
                f"<span style='color:white;font-weight:bold'>{confidence}%</span></div></div>"
                f"<p style='color:{bar_color};font-size:13px;margin-top:4px'>⬆ {conf_text} — the model is {'quite sure' if confidence>=60 else 'somewhat sure' if confidence>=40 else 'not very sure'} about this signal.</p>",
                unsafe_allow_html=True,
            )
            st.markdown("")

            # Explanation cards
            st.markdown("**📌 Why this signal? (Top 3 most important factors)**")
            for feat, explanation in top_reasons:
                lines = explanation.split("\n")
                title_line  = lines[0]
                detail_line = lines[1] if len(lines) > 1 else ""
                st.markdown(
                    f"<div style='background:#1e1e1e;border-left:4px solid {color};"
                    f"border-radius:6px;padding:10px 14px;margin:6px 0'>"
                    f"<div style='font-weight:bold;font-size:14px'>{title_line}</div>"
                    f"<div style='color:#aaa;font-size:13px;margin-top:4px'>{detail_line}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
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
                conf_text = "High confidence" if confidence >= 60 else "Medium confidence" if confidence >= 40 else "Low confidence — treat with caution"
                st.markdown(
                    f"**Signal:** <span style='color:{color};font-size:1.2rem;font-weight:bold'>"
                    f"{SIGNAL_EMOJI[sig]} {SIGNAL_LABEL[sig]}</span> &nbsp;|&nbsp; "
                    f"**Confidence:** <span style='color:{bar_color};font-weight:bold'>{confidence}% — {conf_text}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("**📌 Why this signal? (Top 3 most important factors)**")
                for feat, explanation in top_reasons:
                    lines = explanation.split("\n")
                    title_line  = lines[0]
                    detail_line = lines[1] if len(lines) > 1 else ""
                    st.markdown(
                        f"<div style='background:#1e1e1e;border-left:4px solid {color};"
                        f"border-radius:6px;padding:10px 14px;margin:6px 0'>"
                        f"<div style='font-weight:bold;font-size:14px'>{title_line}</div>"
                        f"<div style='color:#aaa;font-size:13px;margin-top:4px'>{detail_line}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
