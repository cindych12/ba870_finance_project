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



def create_features(df):
    df = df.copy()
    df["Return_1d"] = df["Close"].pct_change()
    
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)

    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_ratio"] = df["MA_5"] / df["MA_20"]
    df["MA_diff"] = df["MA_5"] - df["MA_20"]
    df["Price_vs_MA20"] = df["Close"] / df["MA_20"]

    df["Volatility_5"] = df["Return_1d"].rolling(5).std()
    df["Volatility_10"] = df["Return_1d"].rolling(10).std()

    df["Volume_MA5"] = df["Volume"].rolling(5).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA5"]

    df["Momentum_3"] = df["Close"] / df["Close"].shift(3) - 1
    df["Momentum_10"] = df["Close"] / df["Close"].shift(10) - 1
    df["Momentum_20"] = df["Close"] / df["Close"].shift(20) - 1

    df["HL_Range"] = (df["High"] - df["Low"]) / df["Close"]
    df["Breakout_20"] = df["Close"] / df["Close"].rolling(20).max() - 1
    df["Drawdown_20"] = df["Close"] / df["Close"].rolling(20).max() - 1

    df["Volume_Spike"] = df["Volume"] / df["Volume"].rolling(20).mean()

    df = df.dropna()
    return df
        
feature_cols = [
    "Return_1d", "Return_5d",
    "MA_5", "MA_20",
    "MA_ratio", "MA_diff", "Price_vs_MA20",
    "Volatility_5", "Volume_MA5", "Volume_Ratio",
    "Momentum_3", "Momentum_10", "Momentum_20",
    "HL_Range",
    "Volatility_10",
    "Breakout_20",
    "Drawdown_20",
    "Volume_Spike"]

# User input
ticker = st.text_input("Enter stock ticker:", "AAPL")

if st.button("Run Analysis"):
    df = yf.download(ticker, period="1y")

    if df.empty:
        st.error("No data found")
    else:
        st.subheader("Stock Price")
        st.line_chart(df["Close"])

        df = create_features(df)

        X = df[feature_cols].tail(1)

        prediction = model.predict(X)[0]

        st.subheader("Prediction")
        st.write(prediction)






