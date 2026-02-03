import numpy as np
import pandas as pd


def normalize_0_100(s: pd.Series) -> pd.Series:
    if s is None or len(s) == 0:
        return s
    out = s.astype(float)
    if out.notna().sum() == 0:
        return pd.Series([np.nan] * len(out), index=out.index)
    mn = float(out.min())
    mx = float(out.max())
    if mx > mn:
        return (out - mn) / (mx - mn) * 100.0
    return pd.Series([np.nan] * len(out), index=out.index)


def label_bucket(x):
    if pd.isna(x):
        return "N/A"
    if x >= 75:
        return "Strong"
    if x >= 60:
        return "Watch"
    return "Risky"
