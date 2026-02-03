# src/technical.py
from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd
import streamlit as st

# Optional dependency
try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


# =========================
# Helpers
# =========================
def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _score_0_100_from_01(x01: float) -> float:
    return float(_clip01(x01) * 100.0)


def _sma(s: pd.Series, window: int) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").rolling(window).mean()


def _rsi14(close: pd.Series) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce")
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _max_drawdown(close: pd.Series) -> Optional[float]:
    c = pd.to_numeric(close, errors="coerce").dropna()
    if len(c) < 2:
        return None
    peak = c.cummax()
    dd = (c / peak) - 1.0
    return float(dd.min())  # negative


def _annualized_vol(returns: pd.Series, window: int = 20) -> Optional[float]:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < window + 5:
        return None
    v = r.rolling(window).std().iloc[-1]
    if pd.isna(v):
        return None
    return float(v * np.sqrt(252.0))


def _weighted_mean(values: Dict[str, float], weights: Dict[str, float]) -> float:
    num = 0.0
    den = 0.0
    for k, w in weights.items():
        v = values.get(k, np.nan)
        if pd.notna(v):
            num += float(v) * float(w)
            den += float(w)
    return float(num / den) if den > 0 else np.nan


def _extract_ticker_frame(big: pd.DataFrame, t: str) -> Optional[pd.DataFrame]:
    """
    Robustly extract per-ticker OHLCV frame from yfinance batch download output.
    Handles MultiIndex columns in either (Ticker, Field) or (Field, Ticker).
    """
    if big is None or big.empty:
        return None

    # MultiIndex case (typical when downloading multiple tickers)
    if isinstance(big.columns, pd.MultiIndex):
        lvl0 = set(big.columns.get_level_values(0))
        lvl1 = set(big.columns.get_level_values(1))

        if t in lvl0:
            # (Ticker, Field)
            df = big[t].copy()
        elif t in lvl1:
            # (Field, Ticker)
            df = big.xs(t, level=1, axis=1).copy()
        else:
            return None
    else:
        # Single ticker fallback
        df = big.copy()

    if df is None or df.empty:
        return None

    # Normalize columns (make sure standard OHLCV present)
    df = df.dropna(how="all")
    if df.empty:
        return None

    df = df.reset_index()
    # Some yfinance outputs have "Adj Close" etc; we only need Close/Volume at minimum
    if "Close" not in df.columns:
        return None
    return df


# =========================
# Public API (dipakai Streamlit)
# =========================
@st.cache_data(ttl=3600)
def load_prices(tickers: list[str], start: str, debug: bool = False) -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}

    # 1) yfinance tidak ada
    if yf is None:
        return {"_ERROR_": "yfinance tidak tersedia (import gagal). Pastikan requirements.txt ada yfinance."}

    out: dict[str, pd.DataFrame] = {}
    CHUNK = 10
    any_download_attempt = False

    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        try:
            any_download_attempt = True
            big = yf.download(
                tickers=chunk,
                start=start,
                group_by="column",
                progress=False,
                threads=True,
            )
        except Exception as e:
            if debug:
                out["_ERROR_"] = f"yfinance download error: {e}"
            continue

        # 2) request sukses tapi kosong
        if big is None or getattr(big, "empty", True):
            continue

        for t in chunk:
            try:
                if isinstance(big.columns, pd.MultiIndex):
                    lvl0 = set(big.columns.get_level_values(0))
                    lvl1 = set(big.columns.get_level_values(1))

                    if t in lvl0:
                        df = big[t].dropna().reset_index()
                    elif t in lvl1:
                        df = big.xs(t, level=1, axis=1).dropna().reset_index()
                    else:
                        continue
                else:
                    df = big.dropna().reset_index()

                if df.empty or "Close" not in df.columns:
                    continue

                df["ret_1d"] = df["Close"].pct_change()
                df["ma20"] = df["Close"].rolling(20).mean()
                df["ma50"] = df["Close"].rolling(50).mean()
                df["ma200"] = df["Close"].rolling(200).mean()
                df["vol_20d"] = df["ret_1d"].rolling(20).std() * np.sqrt(252)
                if "Volume" in df.columns:
                    df["vol_avg20"] = df["Volume"].rolling(20).mean()
                df["rsi14"] = _rsi14(df["Close"])  # <— penting: jangan rsi(...)

                out[t] = df
            except Exception as e:
                if debug:
                    out["_ERROR_"] = f"processing error for {t}: {e}"
                continue

    # 3) kalau semuanya kosong, kasih error yang berguna
    only_debug_keys = [k for k in out.keys() if str(k).startswith("_")]
    if (len(out) == 0) or (len(out) == len(only_debug_keys)):
        if debug and "_ERROR_" not in out:
            out["_ERROR_"] = (
                "yfinance mengembalikan data kosong untuk semua ticker. "
                "Kemungkinan: koneksi ke Yahoo diblok/rate limit, atau ticker format salah."
            )

    return out



def health_from_df(dfp: pd.DataFrame) -> Dict[str, float]:
    """
    Output:
      health: 0-100
      trend:  0-100
      risk:   0-100 (lebih tinggi = lebih aman)
      liq:    0-100
      close, rsi, vol20, mdd, vol_ratio
    """
    empty = {
        "health": np.nan,
        "trend": np.nan,
        "risk": np.nan,
        "liq": np.nan,
        "close": np.nan,
        "rsi": np.nan,
        "vol20": np.nan,
        "mdd": np.nan,
        "vol_ratio": np.nan,
    }

    if dfp is None or dfp.empty or "Close" not in dfp.columns:
        return empty

    df = dfp.copy()

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close) < 220:
        out = empty.copy()
        out["close"] = float(close.iloc[-1]) if len(close) else np.nan
        return out

    out = empty.copy()
    out["close"] = float(close.iloc[-1])

    # ===== Trend score =====
    ma50 = _sma(close, 50)
    ma200 = _sma(close, 200)

    ma50_last = ma50.iloc[-1]
    ma200_last = ma200.iloc[-1]

    if pd.isna(ma50_last) or pd.isna(ma200_last) or ma200_last == 0:
        trend_score = np.nan
    else:
        regime01 = 1.0 if ma50_last > ma200_last else 0.0

        ratio = (ma50_last / ma200_last) - 1.0
        dist01 = _clip01((ratio + 0.15) / 0.30)

        ma50_20 = ma50.iloc[-21] if len(ma50) >= 21 else np.nan
        if pd.isna(ma50_20) or ma50_20 == 0:
            slope01 = 0.5
        else:
            slope = (ma50_last / ma50_20) - 1.0
            slope01 = _clip01((slope + 0.05) / 0.10)

        trend01 = (0.55 * regime01) + (0.25 * dist01) + (0.20 * slope01)
        trend_score = _score_0_100_from_01(trend01)

    out["trend"] = float(trend_score) if pd.notna(trend_score) else np.nan

    # ===== Risk score =====
    ret = close.pct_change()
    vol20 = _annualized_vol(ret, window=20)
    mdd = _max_drawdown(close)

    out["vol20"] = float(vol20) if vol20 is not None else np.nan
    out["mdd"] = float(mdd) if mdd is not None else np.nan

    if vol20 is None or mdd is None:
        risk_score = np.nan
    else:
        vol_bad01 = _clip01(vol20 / 0.80)
        vol_good01 = 1.0 - vol_bad01

        dd_abs = abs(mdd)
        dd_bad01 = _clip01(dd_abs / 0.80)
        dd_good01 = 1.0 - dd_bad01

        risk01 = (0.55 * vol_good01) + (0.45 * dd_good01)
        risk_score = _score_0_100_from_01(risk01)

    out["risk"] = float(risk_score) if pd.notna(risk_score) else np.nan

    # ===== Liquidity score =====
    liq_score = np.nan
    if "Volume" in df.columns:
        vol = pd.to_numeric(df["Volume"], errors="coerce")
        vol_avg20 = vol.rolling(20).mean()

        v_last = vol.iloc[-1] if len(vol) else np.nan
        v_avg = vol_avg20.iloc[-1] if len(vol_avg20) else np.nan

        if pd.notna(v_last) and pd.notna(v_avg) and v_avg != 0:
            out["vol_ratio"] = float(v_last / v_avg)

        turnover = out["close"] * float(v_avg) if pd.notna(out["close"]) and pd.notna(v_avg) else np.nan
        if pd.notna(turnover) and turnover > 0:
            logt = float(np.log10(turnover))
            liq01 = _clip01((logt - 8.0) / 3.0)
            liq_score = _score_0_100_from_01(liq01)

    out["liq"] = float(liq_score) if pd.notna(liq_score) else np.nan

    # ===== RSI =====
    rsi = _rsi14(close)
    out["rsi"] = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else np.nan

    # ===== Final Health =====
    out["health"] = _weighted_mean(
        {"trend": out["trend"], "risk": out["risk"], "liq": out["liq"]},
        {"trend": 0.45, "risk": 0.40, "liq": 0.15},
    )

    return out
