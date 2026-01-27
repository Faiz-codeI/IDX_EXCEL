import os
import pandas as pd


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def load_fundamentals(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    df = pd.read_csv(path)
    return normalize_columns(df)


def to_base_ticker(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    s = s.replace(".JK", "")
    return s


def to_ticker_jk(x: str) -> str:
    b = to_base_ticker(x)
    return f"{b}.JK" if b else ""


def infer_ticker_column(df: pd.DataFrame):
    candidates = ["ticker", "symbol", "kode", "emiten", "stock", "saham", "code"]
    lower_map = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in lower_map:
            return lower_map[k]
    for c in df.columns:
        if df[c].dtype == object:
            sample = df[c].dropna().astype(str).head(100)
            if len(sample) == 0:
                continue
            ratio = (
                sample.str.upper()
                .str.replace(".JK", "", regex=False)
                .str.match(r"^[A-Z]{3,5}$")
                .mean()
            )
            if ratio > 0.6:
                return c
    return None


def infer_score_column(df: pd.DataFrame):
    candidates = ["score", "total_score", "final_score", "fundamental_score", "fund_score"]
    lower_map = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in lower_map:
            return lower_map[k]
    for c in df.columns:
        if "score" in c.lower() and pd.api.types.is_numeric_dtype(df[c]):
            return c
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return num_cols[0] if num_cols else None


def load_universe_from_txt(path: str) -> list[str]:
    if not path or not os.path.exists(path):
        return []
    out: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out
