import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def max_drawdown(close: pd.Series) -> float:
    peak = close.cummax()
    dd = (close / peak) - 1
    return float(dd.min())


def clamp(x, low=0, high=100) -> float:
    return max(low, min(high, float(x)))


@st.cache_data(ttl=3600)
def load_prices(tickers: list[str], start: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = yf.download(t, start=start, group_by="column", progress=False)
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df["ret_1d"] = df["Close"].pct_change()
        df["ma20"] = df["Close"].rolling(20).mean()
        df["ma50"] = df["Close"].rolling(50).mean()
        df["ma200"] = df["Close"].rolling(200).mean()
        df["vol_20d"] = df["ret_1d"].rolling(20).std() * np.sqrt(252)
        df["vol_avg20"] = df["Volume"].rolling(20).mean()
        df["rsi14"] = rsi(df["Close"], 14)
        out[t] = df
    return out


def health_from_df(df: pd.DataFrame) -> dict:
    d = df.dropna()
    if len(d) < 220:
        return {
            "health": np.nan, "trend": np.nan, "risk": np.nan, "liq": np.nan,
            "mdd": np.nan, "close": np.nan, "rsi": np.nan, "vol20": np.nan, "vol_ratio": np.nan
        }
    latest = d.iloc[-1]
    close = float(latest["Close"])
    ma50 = float(latest["ma50"])
    ma200 = float(latest["ma200"])
    vol20 = float(latest["vol_20d"])
    rsi14 = float(latest["rsi14"])
    vol_ratio = float(latest["Volume"] / latest["vol_avg20"])
    mdd = max_drawdown(d["Close"])

    trend_score = (close > ma50) * 50 + (close > ma200) * 50
    risk_score = clamp(100 - (vol20 * 200) - (abs(mdd) * 200))
    liq_score = clamp(60 + (vol_ratio - 1) * 20)
    health = clamp(0.40 * trend_score + 0.25 * risk_score + 0.15 * liq_score)

    return {
        "health": float(health),
        "trend": float(trend_score),
        "risk": float(risk_score),
        "liq": float(liq_score),
        "mdd": float(mdd),
        "close": float(close),
        "rsi": float(rsi14),
        "vol20": float(vol20),
        "vol_ratio": float(vol_ratio),
    }
