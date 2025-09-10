# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import ta  # technical analysis helpers

st.set_page_config(layout="wide", page_title="ETH Price Predictor (proto)")

st.title("ETH Price Predictor — Prototype")
st.caption("Prototype: historische data -> features -> model -> korte termijnvoorspelling. Geen financieel advies.")

# ---- 1) Haal historische data op (CoinGecko public API voorbeeld) ----
@st.cache_data(ttl=3600)
def fetch_eth_daily(days=365):
    # CoinGecko KISS: market_chart?days=xxx returns last N days
    url = f"https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
    params = {"vs_currency":"usd", "days": days, "interval": "daily"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    prices = js["prices"]            # [ [ts, price], ... ]
    market_caps = js.get("market_caps", [])
    total_volumes = js.get("total_volumes", [])
    df = pd.DataFrame(prices, columns=["ts","close"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.set_index("ts").sort_index()
    # approximate open/high/low using available daily price only (simple)
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open","close"]].max(axis=1)
    df["low"]  = df[["open","close"]].min(axis=1)
    df["volume"] = pd.Series([v[1] for v in total_volumes], index=df.index) if total_volumes else np.nan
    return df

days = st.sidebar.slider("Aantal dagen historisch", min_value=365, max_value=4000, value=365, step=30)
df = fetch_eth_daily(days=days)
st.sidebar.write(f"Data van {df.index.min().date()} t/m {df.index.max().date()} ({len(df)} rijen)")

# ---- 2) Feature engineering ----
def add_features(df):
    df = df.copy()
    # returns
    df["ret_1d"] = df["close"].pct_change()
    # moving averages
    for w in [3,7,14,30,90]:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
        df[f"std_{w}"] = df["close"].rolling(w).std()
    # RSI (ta library)
    try:
        df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)
    except Exception:
        df["rsi_14"] = np.nan
    # lag features
    for lag in range(1,8):
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
    df = df.dropna()
    return df

df_feat = add_features(df)
st.write("Voorbeeld data (laatste 6 rijen):")
st.dataframe(df_feat.tail(6))

# ---- 3) Train/test split (time-aware) ----
horizon = st.sidebar.number_input("Voorspellingshorizon (dagen)", min_value=1, max_value=30, value=1)
# Target = close price horizon days ahead
df_feat["target"] = df_feat["close"].shift(-horizon)
df_feat = df_feat.dropna()

split_pct = float(st.sidebar.slider("Train% (rest validation/test)", 50, 90, 80))
n = len(df_feat)
n_train = int(n * (split_pct/100.0))
train = df_feat.iloc[:n_train].copy()
test = df_feat.iloc[n_train:].copy()

features = [c for c in df_feat.columns if c not in ["target"]]
st.sidebar.write(f"Features used: {len(features)}")

# ---- 4) Train a model (RandomForest example) ----
if st.button("Train model"):
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    X_train = train[features]
    y_train = train["target"]
    model.fit(X_train, y_train)

    # predict on test
    X_test = test[features]
    y_test = test["target"]
    preds = model.predict(X_test)

    # metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds)
    mape = (np.mean(np.abs((y_test - preds)/y_test))) * 100

    st.metric("MAE", f"{mae:.2f} USD")
    st.metric("RMSE", f"{rmse:.2f} USD")
    st.metric("MAPE", f"{mape:.2f} %")

    # plot last N days: actual vs preds
    n_plot = 120
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(test.index[-n_plot:], test["target"].values[-n_plot:], label="Actual")
    ax.plot(test.index[-n_plot:], preds[-n_plot:], label="Predicted")
    ax.set_title("Actual vs Predicted (test set)")
    ax.legend()
    st.pyplot(fig)

    # show feature importances
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(15)
    st.write("Top feature importances:")
    st.bar_chart(imp)

    # single-step future forecast: take last row of full features and predict horizon
    last_row = df_feat.iloc[[-1]][features]
    next_pred = model.predict(last_row)[0]
    st.success(f"Model voorspelt voor {horizon} dag(en) vooruit: {next_pred:.2f} USD (op {df_feat.index[-1].date()} + {horizon} dagen)")

    # simple backtest signal example: if pred > current -> long else flat (very naive)
    current_price = df_feat["close"].iloc[-1]
    st.write(f"Huidige prijs (laatste): {current_price:.2f} USD")
    signal = "LONG" if next_pred > current_price else "FLAT/SHORT"
    st.info(f"Voorbeeld signaal (super simpel): {signal}")

else:
    st.info("Klik op 'Train model' om te trainen en een voorspelling te zien.")

st.markdown("---")
st.write("Limitations & notes:")
st.write("""
- Dit is een *prototype*: model- en data-keuzes zijn bewust eenvoudig gehouden.
- Crypto-prijzen zijn zeer volatiel; korte termijn voorspellingen hebben doorgaans lage betrouwbaarheid.
- Verbeteringen: hyperparameter tuning, cross-validation (time series split), meer features (on-chain, sentiment), en sequentiële modellen (LSTM/Transformer).
- Data-bron en endpoint limieten: API calls kunnen rate-limited zijn. Voor productie gebruik een betrouwbare bron of gekochte feed.
""")