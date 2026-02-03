import numpy as np
import pandas as pd


def normalize_0_100(s: pd.Series) -> pd.Series:
    if s is None or len(s) == 0:
        return s
    out = pd.to_numeric(s, errors="coerce")
    if out.notna().sum() == 0:
        return pd.Series([np.nan] * len(out), index=out.index)
    mn = out.min(skipna=True)
    mx = out.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx):
        return pd.Series([np.nan] * len(out), index=out.index)
    if mx == mn:
        return pd.Series(np.where(out.notna(), 50.0, np.nan), index=out.index, dtype=float)
    return ((out - mn) / (mx - mn) * 100.0).astype(float)


def label_bucket(x):
    if x is None:
        return "—"
    x = pd.to_numeric(x, errors="coerce")
    if pd.isna(x):
        return "—"
    if x >= 80:
        return "Strong"
    if x >= 60:
        return "Watch"
    return "Risky"
