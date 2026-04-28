import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Stock Investment Decision Support",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Stock Investment Decision Support App")
st.markdown("Get **Buy / Hold / Sell** recommendations powered by machine learning and technical indicators.")

# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "JPM", "JNJ", "XOM", "WMT",
    "META", "AMD", "BAC", "GS", "COST",
    "DIS", "CVX", "CAT", "BA", "PFE",
    "SPY", "QQQ",
]

FEATURE_COLS = [
    "Return_1d", "Return_5d",
    "MA_5", "MA_20", "MA_ratio", "MA_diff", "Price_vs_MA20",
    "Volatility_5", "Volatility_10",
    "Volume_MA5", "Volume_Ratio", "Volume_Spike",
    "Momentum_3", "Momentum_10", "Momentum_20",
    "HL_Range", "Breakout_20", "Drawdown_20",
]

SIGNAL_MAP = {1: "🟢 BUY", 0: "🟡 HOLD", -1: "🔴 SELL"}
SIGNAL_COLOR = {1: "#22c55e", 0: "#eab308", -1: "#ef4444"}

# ─────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────
def add_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("Date").copy()
    group["Return_1d"]    = group["Close"].pct_change()
    group["Return_5d"]    = group["Close"].pct_change(5)
    group["MA_5"]         = group["Close"].rolling(5).mean()
    group["MA_20"]        = group["Close"].rolling(20).mean()
    group["MA_ratio"]     = group["MA_5"] / group["MA_20"]
    group["MA_diff"]      = group["MA_5"] - group["MA_20"]
    group["Price_vs_MA20"]= group["Close"] / group["MA_20"]
    group["Volatility_5"] = group["Return_1d"].rolling(5).std()
    group["Volatility_10"]= group["Return_1d"].rolling(10).std()
    group["Volume_MA5"]   = group["Volume"].rolling(5).mean()
    group["Volume_Ratio"] = group["Volume"] / group["Volume_MA5"]
    group["Volume_MA20"]  = group["Volume"].rolling(20).mean()
    group["Volume_Spike"] = group["Volume"] / group["Volume_MA20"]
    group["Momentum_3"]   = group["Close"] / group["Close"].shift(3) - 1
    group["Momentum_10"]  = group["Close"] / group["Close"].shift(10) - 1
    group["Momentum_20"]  = group["Close"] / group["Close"].shift(20) - 1
    group["HL_Range"]     = (group["High"] - group["Low"]) / group["Close"]
    group["High_20"]      = group["High"].rolling(20).max()
    group["Low_20"]       = group["Low"].rolling(20).min()
    group["Breakout_20"]  = group["Close"] / group["High_20"]
    group["Drawdown_20"]  = group["Close"] / group["Low_20"]
    group["Future_Return_5d"] = group["Close"].shift(-5) / group["Close"] - 1
    return group


@st.cache_data(show_spinner="Downloading historical data…")
def download_data(tickers: list, period: str = "5y") -> pd.DataFrame:
    all_data = []
    for ticker in tickers:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].reset_index()
        df["Ticker"] = ticker
        df = df.sort_values("Date")
        all_data.append(df)
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True).dropna()


@st.cache_resource(show_spinner="Training models… this may take a minute…")
def train_models(tickers_key: str, period: str):
    tickers = tickers_key.split(",")
    raw_df  = download_data(tickers, period)
    if raw_df.empty:
        return None, None, None, None, None

    df = raw_df.groupby("Ticker", group_keys=False).apply(add_features)
    df = df.dropna().reset_index(drop=True)

    df_model = df.dropna(subset=FEATURE_COLS + ["Future_Return_5d"]).copy()
    df_model  = df_model.sort_values(["Date", "Ticker"]).copy()

    cutoff_date = df_model["Date"].quantile(0.8)
    train_df    = df_model[df_model["Date"] < cutoff_date].copy()
    test_df     = df_model[df_model["Date"] >= cutoff_date].copy()

    buy_cutoff  = train_df["Future_Return_5d"].quantile(0.7)
    sell_cutoff = train_df["Future_Return_5d"].quantile(0.3)

    def label_signal(x):
        if x > buy_cutoff and x > 0:   return  1
        if x < sell_cutoff and x < 0:  return -1
        return 0

    train_df["Signal"] = train_df["Future_Return_5d"].apply(label_signal)
    test_df["Signal"]  = test_df["Future_Return_5d"].apply(label_signal)

    X_train, y_train = train_df[FEATURE_COLS], train_df["Signal"]
    X_test,  y_test  = test_df[FEATURE_COLS],  test_df["Signal"]

    # Logistic Regression
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    preprocessor = ColumnTransformer([("num", numeric_transformer, FEATURE_COLS)], remainder="drop")
    log_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   LogisticRegression(solver="lbfgs", max_iter=1000, C=0.8, class_weight="balanced")),
    ])
    log_model.fit(X_train, y_train)

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=8,
        min_samples_split=10, min_samples_leaf=5,
        random_state=42, class_weight="balanced",
    )
    rf_model.fit(X_train, y_train)

    return log_model, rf_model, X_test, y_test, df_model


# ─────────────────────────────────────────
# Sidebar — Settings
# ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    period = st.selectbox("Historical Period", ["2y", "3y", "5y"], index=2)
    custom_input = st.text_input(
        "Add custom tickers (comma-separated)",
        placeholder="e.g. NFLX, UBER",
    )
    extra = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
    all_tickers = list(dict.fromkeys(DEFAULT_TICKERS + extra))  # dedup, preserve order

    selected_tickers = st.multiselect(
        "Stocks to include in training",
        options=all_tickers,
        default=DEFAULT_TICKERS,
    )
    model_choice = st.radio("Model", ["Random Forest", "Logistic Regression"])
    train_btn    = st.button("🚀 Train / Refresh Model", type="primary")

# ─────────────────────────────────────────
# Main Tabs
# ─────────────────────────────────────────
tab_pred, tab_perf, tab_fi, tab_chart = st.tabs([
    "🔮 Predictions", "📊 Model Performance", "📌 Feature Importance", "📉 Price Chart"
])

# ─────────────────────────────────────────
# Train
# ─────────────────────────────────────────
if not selected_tickers:
    st.warning("Please select at least one ticker in the sidebar.")
    st.stop()

tickers_key = ",".join(sorted(selected_tickers))
log_model, rf_model, X_test, y_test, df_model = train_models(tickers_key, period)

if log_model is None:
    st.error("Failed to download data. Check ticker symbols and try again.")
    st.stop()

active_model = rf_model if model_choice == "Random Forest" else log_model

# ─────────────────────────────────────────
# Tab 1: Predictions
# ─────────────────────────────────────────
with tab_pred:
    st.subheader("Latest Signal for Each Stock")

    latest_rows = (
        df_model.sort_values("Date")
        .groupby("Ticker")
        .tail(1)
        .copy()
    )
    latest_rows["RF_Prediction"]       = rf_model.predict(latest_rows[FEATURE_COLS])
    latest_rows["Logistic_Prediction"] = log_model.predict(latest_rows[FEATURE_COLS])

    pred_col = "RF_Prediction" if model_choice == "Random Forest" else "Logistic_Prediction"
    display  = latest_rows[["Date", "Ticker", "Close", pred_col]].copy()
    display["Signal"] = display[pred_col].map(SIGNAL_MAP)
    display["Close"]  = display["Close"].round(2)
    display = display.rename(columns={"Close": "Latest Close ($)", pred_col: "_num"})
    display = display[["Date", "Ticker", "Latest Close ($)", "Signal"]]

    # Summary cards
    buy_count  = (latest_rows[pred_col] ==  1).sum()
    hold_count = (latest_rows[pred_col] ==  0).sum()
    sell_count = (latest_rows[pred_col] == -1).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 BUY",  buy_count)
    c2.metric("🟡 HOLD", hold_count)
    c3.metric("🔴 SELL", sell_count)

    st.dataframe(display.reset_index(drop=True), use_container_width=True, height=420)

    # Single-stock deep dive
    st.markdown("---")
    st.subheader("🔍 Single Stock Detail")
    pick = st.selectbox("Choose a stock", options=sorted(latest_rows["Ticker"].tolist()))
    row  = latest_rows[latest_rows["Ticker"] == pick].iloc[0]
    sig_num = int(row[pred_col])
    sig_label = SIGNAL_MAP[sig_num]
    sig_color = SIGNAL_COLOR[sig_num]

    st.markdown(
        f"<h2 style='color:{sig_color};text-align:center'>{pick}: {sig_label}</h2>",
        unsafe_allow_html=True,
    )

    feat_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Value":   [round(float(row[f]), 5) for f in FEATURE_COLS],
    })
    st.dataframe(feat_df, use_container_width=True, height=300)

# ─────────────────────────────────────────
# Tab 2: Model Performance
# ─────────────────────────────────────────
with tab_perf:
    st.subheader(f"Test-Set Performance — {model_choice}")

    y_pred = active_model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        labels=[-1, 0, 1],
        target_names=["Sell", "Hold", "Buy"],
        output_dict=True,
    )

    st.metric("Overall Accuracy", f"{acc:.2%}")

    report_df = pd.DataFrame(report).T.round(3)
    st.dataframe(report_df, use_container_width=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
    ax.set_xticklabels(["Sell","Hold","Buy"])
    ax.set_yticklabels(["Sell","Hold","Buy"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ─────────────────────────────────────────
# Tab 3: Feature Importance (RF only)
# ─────────────────────────────────────────
with tab_fi:
    st.subheader("Feature Importance — Random Forest")
    fi_df = pd.DataFrame({
        "Feature":    FEATURE_COLS,
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(fi_df)))
    ax2.barh(fi_df["Feature"], fi_df["Importance"], color=colors)
    ax2.invert_yaxis()
    ax2.set_xlabel("Importance")
    ax2.set_title("Random Forest Feature Importance")
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.dataframe(fi_df, use_container_width=True)

# ─────────────────────────────────────────
# Tab 4: Price Chart with MA
# ─────────────────────────────────────────
with tab_chart:
    st.subheader("📉 Price Chart with Moving Averages")
    chart_ticker = st.selectbox("Select ticker", options=sorted(selected_tickers), key="chart_ticker")
    days_back    = st.slider("Days to display", 60, 365, 180)

    stock_df = df_model[df_model["Ticker"] == chart_ticker].sort_values("Date").tail(days_back).copy()

    if stock_df.empty:
        st.warning("No data available for this ticker.")
    else:
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(stock_df["Date"], stock_df["Close"], label="Close",   color="#3b82f6", linewidth=1.5)
        ax3.plot(stock_df["Date"], stock_df["MA_5"],  label="MA 5",    color="#f97316", linewidth=1,   linestyle="--")
        ax3.plot(stock_df["Date"], stock_df["MA_20"], label="MA 20",   color="#a855f7", linewidth=1,   linestyle="--")
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=30)
        ax3.set_title(f"{chart_ticker} — Close Price & Moving Averages")
        ax3.set_ylabel("Price ($)")
        ax3.legend()
        ax3.grid(alpha=0.3)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        # Overlay buy/sell signals on chart
        st.markdown("#### Signal overlay (last predictions)")
        pred_col_chart = "RF_Prediction" if model_choice == "Random Forest" else "Logistic_Prediction"
        latest_signal  = latest_rows[latest_rows["Ticker"] == chart_ticker]
        if not latest_signal.empty:
            sig = int(latest_signal[pred_col_chart].values[0])
            st.markdown(
                f"**Latest {model_choice} signal for {chart_ticker}:** "
                f"<span style='color:{SIGNAL_COLOR[sig]};font-weight:bold'>{SIGNAL_MAP[sig]}</span>",
                unsafe_allow_html=True,
            )
