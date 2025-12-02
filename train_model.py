# train_model.py
import os
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

DATA_DIR = "data/raw"
MODEL_PATH = "models/stock_model.pkl"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

def download_data(ticker="AAPL", period="5y"):
    print(f"[INFO] Downloading data for {ticker} ...")
    df = yf.download(ticker, period=period)
    df = df[["Close"]].dropna()
    csv_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df.to_csv(csv_path)
    print(f"[INFO] Saved raw data to {csv_path}")
    return df

def create_features(df, n_lags=5):
    """
    Creates lag features:
    X: [Close(t-1), Close(t-2), ..., Close(t-5)]
    y: Close(t)
    """
    for i in range(1, n_lags + 1):
        df[f"lag_{i}"] = df["Close"].shift(i)
    df = df.dropna()
    X = df[[f"lag_{i}" for i in range(1, n_lags + 1)]]
    y = df["Close"]
    return X, y

def train_and_save_model(ticker="AAPL"):
    df = download_data(ticker)
    X, y = create_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
    print("[INFO] Training model ...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"[INFO] Test RMSE: {rmse:.2f}")

    joblib.dump(model, MODEL_PATH)
    print(f"[INFO] Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    # change ticker if you want: e.g. "TCS.NS"
    train_and_save_model(ticker="AAPL")
