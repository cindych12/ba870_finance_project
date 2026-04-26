import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pickle

# Title
st.title("Stock Recommendation App")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# User input
ticker = st.text_input("Enter stock ticker:", "AAPL")

if st.button("Run Analysis"):
    df = yf.download(ticker, period="1y")

    if df.empty:
        st.error("No data found")
    else:
        st.subheader("Stock Price")
        st.line_chart(df["Close"])

        # ===== Feature Engineering (簡化版) =====
        df["Return_1d"] = df["Close"].pct_change()
        df["MA_5"] = df["Close"].rolling(5).mean()
        df["MA_20"] = df["Close"].rolling(20).mean()
        df = df.dropna()

        X = df[["Return_1d", "MA_5", "MA_20"]].tail(1)

        # Prediction
        prediction = model.predict(X)[0]

        st.subheader("Prediction")
        st.write(prediction)

