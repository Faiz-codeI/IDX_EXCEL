from __future__ import annotations

from PIL import Image
import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

bucket_order = ["Strong", "Watch", "Risky"]

BUCKET_COLORS = {
    "Strong": "#00c896",
    "Watch":  "#f5c542",
    "Risky":  "#ff6b57",
}




from src.dataio import (
    load_fundamentals,
    infer_ticker_column,
    to_ticker_jk,
    to_base_ticker,
    load_universe_from_txt,
)
from src.technical import load_prices, health_from_df
from src.scoring import label_bucket


# ======================================================
# PLOTLY GLOBAL STYLE (CONSISTENT & PREMIUM)
# ======================================================
px.defaults.template = "plotly_dark"
px.defaults.width = None


def style_fig(fig, height: int = 560, title: str | None = None):
    if title:
        fig.update_layout(title=title)
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)
    return fig


# ======================================================
# UI
# ======================================================
def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* Layout */
        .block-container { padding-top: 1.05rem; padding-bottom: 2rem; max-width: 1400px; }
        div[data-testid="stVerticalBlock"] { gap: 0.85rem; }
        h1, h2, h3 { letter-spacing: -0.2px; }

        /* Sidebar */
        section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

        /* Badges */
        .badge {
          display:inline-block;
          padding: 0.22rem 0.55rem;
          border-radius: 999px;
          font-size: 0.84rem;
          font-weight: 650;
          border: 1px solid rgba(255,255,255,0.12);
          line-height: 1.1;
          background: rgba(160,160,160,0.16);
        }
        .badge-strong  { background: rgba(0, 184, 148, 0.18); }
        .badge-neutral { background: rgba(160,160,160,0.18); }
        .badge-risky   { background: rgba(255,159,67,0.18); }
        .badge-avoid   { background: rgba(255, 99,132,0.18); }

        /* Cards */
        .card {
          border: 1px solid rgba(255,255,255,0.10);
          border-radius: 14px;
          padding: 0.95rem 1.0rem;
          background: rgba(255,255,255,0.03);
        }
        .card h4 { margin: 0 0 0.25rem 0; font-size: 1.05rem; }
        .muted { color: rgba(255,255,255,0.65); }

        /* Table polish */
        div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid rgba(255,255,255,0.10); }

        /* Pills (radio) */
        div[role="radiogroup"] label { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.10);
          padding: 0.25rem 0.65rem; border-radius: 999px; margin-right: 0.35rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def score_to_label_kind(score: float) -> tuple[str, str]:
    if score >= 80:
        return ("Strong", "strong")
    if score >= 60:
        return ("Neutral", "neutral")
    if score >= 40:
        return ("Risky", "risky")
    return ("Avoid", "avoid")


def badge_html(label: str, kind: str) -> str:
    cls = {
        "strong": "badge badge-strong",
        "neutral": "badge badge-neutral",
        "risky": "badge badge-risky",
        "avoid": "badge badge-avoid",
    }.get(kind, "badge badge-neutral")
    return f'<span class="{cls}">{label}</span>'


def safe_topn_slider(
    label: str,
    n_rows: int,
    default: int = 20,
    min_floor: int = 5,
    cap: int = 300,
    key: str | None = None,
) -> int:
    max_n = int(min(cap, n_rows))
    if max_n <= 0:
        return 0
    if max_n == 1:
        st.caption(f"{label}: hanya 1 baris data tersedia.")
        return 1

    min_n = int(min(min_floor, max_n))
    default_n = int(min(default, max_n))
    if min_n >= max_n:
        min_n = 1

    return st.slider(label, min_n, max_n, default_n, key=key)


def round_cols(df: pd.DataFrame, cols: list[str], nd: int) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(nd)
    return out


def add_label_from_score(df: pd.DataFrame, score_col: str, out_col: str = "label") -> pd.DataFrame:
    d = df.copy()
    labels = []
    for v in pd.to_numeric(d[score_col], errors="coerce").tolist():
        if pd.isna(v):
            labels.append("—")
        else:
            lbl, _ = score_to_label_kind(float(v))
            labels.append(lbl)
    d[out_col] = labels
    return d


def _fmt_pct(n: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{(n / total) * 100:.1f}%"


def hard_flag_panel(title: str, tickers: list[str], total_universe: int, expanded: bool = False):
    tickers = sorted([t for t in tickers if t])
    header = f"{title} — HARD FLAG — {len(tickers)} ({_fmt_pct(len(tickers), total_universe)})"
    with st.expander(header, expanded=expanded):
        st.write(", ".join(tickers) if tickers else "—")


def clean_price_dict(d: dict) -> dict:
    """Remove debug keys like _ERROR_ from load_prices result."""
    return {k: v for k, v in d.items() if not str(k).startswith("_")}


def show_prices_debug(d: dict, picked: list[str], debug_mode: bool):
    """Show error + counts (optional)."""
    if "_ERROR_" in d:
        st.error(d["_ERROR_"])
    d2 = clean_price_dict(d)
    if debug_mode:
        st.write("Picked sample:", picked[:5])
        st.write("Loaded tickers:", list(d2.keys())[:10])
        st.write("Count loaded:", len(d2))
    return d2


# ======================================================
# SECTOR MAP FROM EXCELS
# ======================================================
def _sector_from_filename(path: str) -> str:
    base = os.path.basename(path)
    name = base.replace(".xlsx", "").strip()
    parts = [p.strip() for p in name.split("-")]
    if len(parts) >= 3:
        return parts[1]
    return "Unknown"


@st.cache_data(show_spinner=False)
def load_sector_map_from_excels(glob_pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(glob_pattern)) if glob_pattern else []
    if not paths:
        return pd.DataFrame(columns=["ticker_base", "sector"])

    rows = []
    for p in paths:
        try:
            df = pd.read_excel(p, sheet_name=0, dtype=str).fillna("")
        except Exception:
            continue

        if "Kode" not in df.columns:
            continue

        sector = _sector_from_filename(p)

        tickers = (
            df["Kode"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map(to_base_ticker)
            .replace("", np.nan)
            .dropna()
            .unique()
            .tolist()
        )

        for t in tickers:
            rows.append({"ticker_base": t, "sector": sector})

    if not rows:
        return pd.DataFrame(columns=["ticker_base", "sector"])

    return pd.DataFrame(rows).drop_duplicates(subset=["ticker_base"], keep="last")


# ======================================================
# FUNDAMENTAL CORE SCORING (ENGINE TETAP)
# ======================================================
def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([np.nan] * len(df), index=df.index)


def _winsor_minmax_0_100(
    series: pd.Series,
    higher_is_better: bool = True,
    p_low: float = 0.05,
    p_high: float = 0.95,
) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series([np.nan] * len(s), index=s.index)

    lo = s.quantile(p_low)
    hi = s.quantile(p_high)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        out = pd.Series(np.nan, index=s.index, dtype=float)
        out.loc[s.notna()] = 50.0
        return out

    clipped = s.clip(lower=lo, upper=hi)
    x = (clipped - lo) / (hi - lo) * 100.0
    if not higher_is_better:
        x = 100.0 - x
    return x.astype(float)


def _norm_0_100_grouped(
    df: pd.DataFrame,
    colname: str,
    group_cols: list[str],
    higher_is_better: bool = True,
    p_low: float = 0.05,
    p_high: float = 0.95,
) -> pd.Series:
    s = pd.to_numeric(_col(df, colname), errors="coerce")
    if not group_cols:
        return _winsor_minmax_0_100(s, higher_is_better=higher_is_better, p_low=p_low, p_high=p_high)

    tmp = df.copy()
    tmp["_tmp_val"] = s

    def _scale(g: pd.Series) -> pd.Series:
        return _winsor_minmax_0_100(g, higher_is_better=higher_is_better, p_low=p_low, p_high=p_high)

    return tmp.groupby(group_cols, dropna=False)["_tmp_val"].transform(_scale)


def _weighted_mean_rowwise(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    w = pd.Series(weights, dtype=float)
    cols = [c for c in w.index if c in df.columns]
    if not cols:
        return pd.Series([np.nan] * len(df), index=df.index)

    sub = df[cols]
    ww = w[cols]

    avail = sub.notna().astype(float).mul(ww, axis=1)
    denom = avail.sum(axis=1).replace(0, np.nan)
    num = sub.fillna(0).mul(ww, axis=1).sum(axis=1)
    return (num / denom).astype(float)


@st.cache_data(show_spinner=False)
def compute_fundamental_core(
    fund: pd.DataFrame,
    ticker_col: str,
    p_low: float = 0.05,
    p_high: float = 0.95,
) -> pd.DataFrame:
    f = fund.copy()
    f["ticker_base"] = _col(f, ticker_col).astype(str).map(to_base_ticker)
    f["ticker"] = f["ticker_base"].map(to_ticker_jk)

    if "year" not in f.columns:
        f["year"] = np.nan
    if "sector" not in f.columns:
        f["sector"] = np.nan

    group_cols = ["year", "sector"]

    revenue = pd.to_numeric(_col(f, "revenue"), errors="coerce")
    cfo = pd.to_numeric(_col(f, "cfo"), errors="coerce")
    avg_assets = pd.to_numeric(_col(f, "avg_assets"), errors="coerce")
    net_income = pd.to_numeric(_col(f, "net_income"), errors="coerce")

    f["CFO_Margin_calc"] = np.where((revenue > 0) & cfo.notna(), cfo / revenue, np.nan)
    f["Accrual_Ratio"] = np.where(
        cfo.notna() & avg_assets.notna() & (avg_assets != 0),
        (net_income - cfo) / avg_assets,
        np.nan,
    )

    f["S_ROE"] = _norm_0_100_grouped(f, "ROE", group_cols, True, p_low, p_high)
    f["S_ROA"] = _norm_0_100_grouped(f, "ROA", group_cols, True, p_low, p_high)
    f["S_Profit"] = _weighted_mean_rowwise(f[["S_ROE", "S_ROA"]], {"S_ROE": 0.6, "S_ROA": 0.4})

    f["S_DER"] = _norm_0_100_grouped(f, "Debt_to_Equity", group_cols, False, p_low, p_high)
    eq_neg = pd.to_numeric(_col(f, "equity_negative"), errors="coerce")
    f["S_EquitySafe"] = np.where(eq_neg == 1, 0.0, np.where(eq_neg == 0, 100.0, np.nan))
    f["S_Leverage"] = _weighted_mean_rowwise(
        f[["S_DER", "S_EquitySafe"]],
        {"S_DER": 0.7, "S_EquitySafe": 0.3},
    )

    f["S_RevYoY"] = _norm_0_100_grouped(f, "revenue_yoy", group_cols, True, p_low, p_high)
    f["S_NIYoY"] = _norm_0_100_grouped(f, "net_income_yoy", group_cols, True, p_low, p_high)
    f["S_Growth"] = _weighted_mean_rowwise(f[["S_RevYoY", "S_NIYoY"]], {"S_RevYoY": 0.5, "S_NIYoY": 0.5})

    f["S_CFOmargin"] = _norm_0_100_grouped(f, "CFO_Margin_calc", group_cols, True, p_low, p_high)
    f["S_Accrual"] = _norm_0_100_grouped(f, "Accrual_Ratio", group_cols, False, p_low, p_high)
    f["S_CashQ"] = _weighted_mean_rowwise(f[["S_CFOmargin", "S_Accrual"]], {"S_CFOmargin": 0.6, "S_Accrual": 0.4})

    f["fund_health_0_100"] = _weighted_mean_rowwise(
        f[["S_Profit", "S_Leverage", "S_Growth", "S_CashQ"]],
        {"S_Profit": 0.35, "S_Leverage": 0.25, "S_Growth": 0.25, "S_CashQ": 0.15},
    )
    f["fund_bucket"] = f["fund_health_0_100"].apply(label_bucket)

    out = f[
        [
            "ticker",
            "ticker_base",
            "year",
            "sector",
            "fund_health_0_100",
            "fund_bucket",
            "S_Profit",
            "S_Leverage",
            "S_Growth",
            "S_CashQ",
        ]
    ].copy()

    out["sector"] = out["sector"].replace("", np.nan)
    return out.sort_values(["fund_health_0_100"], ascending=False)


# ======================================================
# LOGO HELPERS (optional)
# ======================================================
@st.cache_data(show_spinner=False)
def load_logo(base_ticker: str):
    for ext in ("png", "jpg", "jpeg", "webp"):
        path = os.path.join("assets", "logos", f"{base_ticker}.{ext}")
        if os.path.exists(path):
            return Image.open(path)
    return None


def render_symbol_header(ticker_jk: str, base: str, subtitle: str = ""):
    left, right = st.columns([0.15, 0.85], vertical_alignment="center")
    with left:
        logo = load_logo(base)
        if logo:
            st.image(logo, width=56)
        else:
            st.markdown(f"### {base}")
    with right:
        st.markdown(f"## {ticker_jk}")
        if subtitle:
            st.caption(subtitle)


def render_kpis(options_universe: list[str], start, picked: list[str], sector_list: list[str]):
    c1, c2, c3, c4 = st.columns(4, gap="large")
    c1.metric("Universe", f"{len(options_universe):,} ticker")
    c2.metric("Start Date", str(start))
    c3.metric("Selected", len(picked))
    c4.metric("Sectors", f"{len(sector_list):,}")


def _mode_selector():
    st.subheader("Explore — Insight Dashboard")
    st.caption("Pilih mode dulu. Setelah itu lihat ringkasan (KPI) → distribusi → top ranking.")
    mode = st.radio(
        "Mode",
        ["Teknikal", "Fundamental", "Gabungan"],
        horizontal=True,
        key="explore_mode",
    )
    return mode


# ======================================================
# PAGE
# ======================================================
st.set_page_config(page_title="IDX Dashboard — Teknikal + Fundamental", page_icon="📈", layout="wide")
inject_css()

st.title("IDX Stock Dashboard")
st.caption("Overview → Explore → Analyze → Compare • Teknikal + Fundamental (per sektor) + Skor Gabungan")

DEFAULT_FUNDAMENTALS_CSV = "data/fundamentals_table.csv"
DEFAULT_UNIVERSE_TXT = "data/universe_tickers.txt"
DEFAULT_SECTOR_GLOB = "data/sectors/*.xlsx"


# ======================================================
# SIDEBAR (Controls)
# ======================================================
with st.sidebar:
    st.header("Controls")
    debug_mode = st.toggle("Debug mode", value=False)

    st.subheader("📦 Data Universe")
    fundamentals_path = st.text_input("Path fundamentals_table.csv", value=DEFAULT_FUNDAMENTALS_CSV)
    use_universe_txt = st.toggle("Batasi universe pakai universe_tickers.txt", value=False)
    universe_txt_path = st.text_input("Path universe_tickers.txt (optional)", value=DEFAULT_UNIVERSE_TXT)

    st.divider()
    st.subheader("🏷️ Mapping Sektor IDX")
    sector_glob = st.text_input("Glob file sektor", value=DEFAULT_SECTOR_GLOB)
    st.caption("Contoh: data/sectors/*.xlsx (kolom wajib: 'Kode')")

    st.divider()
    st.subheader("⚙️ Teknikal")
    start = st.date_input("Mulai data harga dari", value=pd.to_datetime("2020-01-01"))

    st.divider()
    st.subheader("🧾 Fundamental")
    p_low = st.slider("Winsor p_low", 0.00, 0.20, 0.05, 0.01)
    p_high = st.slider("Winsor p_high", 0.80, 1.00, 0.95, 0.01)

    st.divider()
    st.subheader("🔗 Sinkronisasi")
    ref_year = st.number_input("Tahun referensi fundamental", min_value=2000, max_value=2100, value=2024, step=1)
    sync_tech_to_fund = st.toggle("Teknikal hanya ticker yang ada fundamental", value=True)
    only_has_sector = st.toggle("Hanya ticker yang sudah punya sector", value=True)

    st.divider()
    st.subheader("🧩 Gabungan")
    w_tech = st.slider("Bobot Teknikal", 0.0, 1.0, 0.5, 0.05)
    w_fund = st.slider("Bobot Fundamental", 0.0, 1.0, 0.5, 0.05)

    st.divider()
    if st.button("Clear cache (harga & fundamental)"):
        st.cache_data.clear()
        st.rerun()


# ======================================================
# LOAD FUNDAMENTALS + SECTOR MAP
# ======================================================
with st.status("Loading fundamentals & sector mapping...", expanded=False) as status:
    try:
        fund = load_fundamentals(fundamentals_path)
    except Exception as e:
        st.error("Gagal membaca fundamentals_table.csv")
        st.code(str(e))
        st.stop()

    ticker_col_guess = infer_ticker_column(fund)
    if ticker_col_guess is None:
        st.error("Kolom ticker tidak terdeteksi. Pastikan ada kolom seperti: ticker/symbol/kode/emiten.")
        st.stop()

    fund = fund.copy()
    fund["ticker_base"] = fund[ticker_col_guess].astype(str).map(to_base_ticker)

    sector_map = load_sector_map_from_excels(sector_glob)
    if sector_map.empty:
        fund["sector"] = np.nan
    else:
        fund = fund.merge(sector_map, on="ticker_base", how="left")
        fund["sector"] = fund["sector"].replace("", np.nan)

    status.update(label="Data loaded.", state="complete")

if debug_mode:
    with st.sidebar:
        st.subheader("🧪 Debug sektor glob")
        paths = sorted(glob.glob(sector_glob)) if sector_glob else []
        st.write("Glob:", sector_glob)
        st.write("Matched files:", len(paths))
        if paths:
            st.write([os.path.basename(p) for p in paths[:5]])
        st.subheader("🧪 Sector sanity check")
        st.write(fund[["ticker_base", "sector"]].dropna().head(10))


sector_list = sorted(fund["sector"].dropna().astype(str).unique().tolist())

with st.sidebar:
    st.subheader("🏭 Filter Sektor")
    picked_sectors = st.multiselect("Pilih sektor (opsional)", options=sector_list, default=[])


# ======================================================
# BUILD FUNDAMENTAL REF UNIVERSE
# ======================================================
core_ref_all = compute_fundamental_core(fund, ticker_col=ticker_col_guess, p_low=float(p_low), p_high=float(p_high))
core_ref = core_ref_all.copy()

if "year" in core_ref.columns:
    core_ref = core_ref[core_ref["year"] == int(ref_year)].copy()
if picked_sectors:
    core_ref = core_ref[core_ref["sector"].astype(str).isin(picked_sectors)].copy()
if only_has_sector:
    core_ref = core_ref.dropna(subset=["sector"])

universe_ref_base = sorted(set(core_ref["ticker_base"].dropna().astype(str).tolist()))
universe_ref = [to_ticker_jk(x) for x in universe_ref_base]
universe_ref = [u for u in universe_ref if u]

universe_base = sorted(set(fund["ticker_base"].replace("", np.nan).dropna().unique().tolist()))
universe = [to_ticker_jk(x) for x in universe_base]
universe = [u for u in universe if u]

if use_universe_txt:
    allowed = load_universe_from_txt(universe_txt_path)
    if allowed:
        allowed_base = {to_base_ticker(x) for x in allowed}
        universe = [u for u in universe if to_base_ticker(u) in allowed_base]
        universe_ref = [u for u in universe_ref if to_base_ticker(u) in allowed_base]

options_universe = universe_ref if sync_tech_to_fund else universe
with st.sidebar:
    st.subheader("✅ Ticker")
    if not options_universe:
        st.error("Universe kosong. Cek fundamental/year/sector atau mapping sektor.")
        st.stop()

    if st.button("Reset picked (pilih semua)"):
        st.session_state["picked"] = options_universe
        st.rerun()

    default_pick = st.session_state.get("picked", options_universe)
    default_pick = [x for x in default_pick if x in options_universe]

    picked = st.multiselect(
        "Pilih saham (untuk Explore/Compare)",
        options=options_universe,
        default=default_pick,
        max_selections=len(options_universe),
    )
    st.session_state["picked"] = picked

    st.caption(f"Terpilih: {len(picked)} / {len(options_universe)}")

    if not picked:
        st.warning("Pilih minimal 1 saham.")
        st.stop()

# ======================================================
# MAIN NAV
# ======================================================
tab_overview, tab_explore, tab_analyze, tab_compare, tab_method = st.tabs(
    ["Overview", "Explore", "Analyze", "Compare", "Methodology"]
)

with tab_overview:
    render_kpis(options_universe, start, picked, sector_list)
    st.divider()
    c1, c2 = st.columns([0.58, 0.42], gap="large")
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Alur pemakaian (pemula)")
        st.write("1) **Explore**: pilih mode → lihat top kandidat & distribusi skor.")
        st.write("2) **Analyze**: pilih 1 saham → lihat chart + ringkasan skor.")
        st.write("3) **Compare**: pilih 2–5 saham → lihat peta perbandingan & tabel ringkas.")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Catatan penting")
        st.write("• **Red flags** ditampilkan terpisah agar tidak mengganggu ranking.")
        st.write("• Engine scoring kamu **tidak diubah** (rumus/normalisasi/bobot tetap).")
        st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# EXPLORE (UI/UX DASHBOARD: Teknikal / Fundamental / Gabungan)
# ======================================================
with tab_explore:
    render_kpis(options_universe, start, picked, sector_list)
    st.divider()

    mode = _mode_selector()

    # -------------------------
    # TEKNIKAL
    # -------------------------
    if mode == "Teknikal":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📈 Technical Market Snapshot")
        st.caption("Tujuan: screening cepat berdasarkan trend/risk/liquidity + skor teknikal (0–100).")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.spinner("Loading price data..."):
            raw_data = load_prices(picked, start=str(start), debug=True)

        data = show_prices_debug(raw_data, picked, debug_mode)

        if not data:
            st.warning("Tidak ada data harga yang berhasil diambil. Cek ticker / koneksi.")
            st.stop()

        rows = []
        for t, dfp in data.items():
            s = health_from_df(dfp)
            rows.append(
                {
                    "ticker": t,
                    "ticker_base": to_base_ticker(t),
                    "health_tech": s.get("health", np.nan),
                    "trend": s.get("trend", np.nan),
                    "risk": s.get("risk", np.nan),
                    "liquidity": s.get("liq", np.nan),
                    "close": s.get("close", np.nan),
                    "rsi14": s.get("rsi", np.nan),
                    "max_drawdown": s.get("mdd", np.nan),
                }
            )

        tech_df = pd.DataFrame(rows).dropna(subset=["health_tech"]).copy()
        if tech_df.empty:
            st.warning("Skor teknikal belum bisa dihitung (data kurang panjang / banyak NaN).")
            st.stop()

        tech_df["bucket"] = tech_df["health_tech"].apply(label_bucket)

        avg_score = float(tech_df["health_tech"].mean())
        pct_strong = float((tech_df["bucket"] == "Strong").mean() * 100)
        med_risk = float(pd.to_numeric(tech_df["risk"], errors="coerce").median())
        n_ticker = int(len(tech_df))

        c1, c2, c3, c4 = st.columns(4, gap="large")
        c1.metric("Avg Technical Score", f"{avg_score:.1f}")
        c2.metric("% Strong", f"{pct_strong:.0f}%")
        c3.metric("Median Risk", "—" if np.isnan(med_risk) else f"{med_risk:.2f}")
        c4.metric("Tickers", f"{n_ticker:,}")

        left, right = st.columns([0.42, 0.58], gap="large")
        with left:
            donut = tech_df["bucket"].value_counts().reset_index()
            donut.columns = ["Bucket", "Count"]
            fig = px.pie(donut, names="Bucket", values="Count", hole=0.62)
            fig = style_fig(fig, height=360, title="Distribusi Bucket (Teknikal)")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            hist = tech_df[["health_tech"]].copy()
            fig = px.histogram(hist, x="health_tech", nbins=20)
            fig = style_fig(fig, height=360, title="Distribusi Skor Teknikal (0–100)")
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("🏁 Top Kandidat (Teknikal)")
        top_n = safe_topn_slider("Top N", len(tech_df), default=20, min_floor=5, cap=200, key="topn_tech")
        view = tech_df.sort_values("health_tech", ascending=False).head(top_n).copy()
        view = round_cols(view, ["health_tech", "trend", "risk", "liquidity", "rsi14", "max_drawdown", "close"], 2)
        view = add_label_from_score(view, "health_tech", out_col="label")

        show_cols = ["ticker", "health_tech", "label", "trend", "risk", "liquidity", "rsi14", "max_drawdown"]
        st.dataframe(view[show_cols], use_container_width=True, hide_index=True)

        bar_df = view.sort_values("health_tech", ascending=True).copy()
        fig = px.bar(bar_df, x="health_tech", y="ticker", orientation="h", color="label")
        fig = style_fig(fig, height=min(780, 140 + 22 * len(bar_df)), title=f"Top {len(bar_df)} Technical Score")
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # FUNDAMENTAL
    # -------------------------
    elif mode == "Fundamental":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧾 Fundamental Snapshot (Per Sektor)")
        st.caption("Tujuan: ranking fair antar peer (tahun & sektor sama). Fokus di Profit / Leverage / Growth.")
        st.markdown("</div>", unsafe_allow_html=True)

        core_all = compute_fundamental_core(
            fund,
            ticker_col=ticker_col_guess,
            p_low=float(p_low),
            p_high=float(p_high),
        )

        years = sorted([int(x) for x in core_all["year"].dropna().unique().tolist()]) if "year" in core_all.columns else []
        default_year = int(ref_year) if int(ref_year) in years else (years[-1] if years else None)

        if default_year is not None:
            year_pick = st.selectbox("Tahun", options=years, index=years.index(default_year), key="fund_year_pick")
            core = core_all[core_all["year"] == year_pick].copy()
        else:
            core = core_all.copy()
            year_pick = None

        if picked_sectors:
            core = core[core["sector"].astype(str).isin(picked_sectors)].copy()

        if only_has_sector:
            core = core.dropna(subset=["sector"])

        with st.container(border=True):
            st.subheader("⚠️ Red Flags (HARD FLAG)")
            raw = fund.copy()

            if year_pick is not None and "year" in raw.columns:
                raw = raw[raw["year"] == year_pick].copy()
            if picked_sectors:
                raw = raw[raw["sector"].astype(str).isin(picked_sectors)].copy()

            visible = set(core["ticker_base"].dropna().astype(str).tolist())
            raw = raw[raw["ticker_base"].isin(visible)].copy()
            total_universe = int(raw["ticker_base"].nunique())

            if "equity_negative" in raw.columns:
                eq = pd.to_numeric(raw["equity_negative"], errors="coerce")
                eq_neg = raw.loc[eq.eq(1), "ticker_base"].dropna().astype(str).unique().tolist()
                hard_flag_panel("Equity negatif (eq==1)", eq_neg, total_universe, expanded=True)

            if "net_income" in raw.columns:
                ni = pd.to_numeric(raw["net_income"], errors="coerce")
                ni_neg = raw.loc[ni.lt(0), "ticker_base"].dropna().astype(str).unique().tolist()
                hard_flag_panel("Net income (negatif)", ni_neg, total_universe, expanded=False)

            st.caption("Hanya HARD FLAG ditampilkan di dashboard agar ranking tetap fokus.")

        if core.empty:
            st.info("Tidak ada data fundamental untuk filter ini.")
            st.stop()

        # ======================================================
        # ✅ FIX UTAMA: NORMALISASI fund_bucket biar donut gak kosong
        # ======================================================
        core["fund_bucket"] = (
            core["fund_bucket"]
            .fillna("")
            .astype(str)
            .str.replace("\u00a0", " ", regex=False)  # NBSP (sering bikin mismatch)
            .str.strip()
            .str.lower()
        )

        core["fund_bucket"] = core["fund_bucket"].map({
            "strong": "Strong",
            "watch":  "Watch",
            "risky":  "Risky",
        }).fillna("Other")

        avg_f = float(pd.to_numeric(core["fund_health_0_100"], errors="coerce").mean())
        pct_strong = float((core["fund_bucket"] == "Strong").mean() * 100)
        n_sector = int(core["sector"].dropna().nunique())
        n_rows = int(core["ticker_base"].nunique())

        c1, c2, c3, c4 = st.columns(4, gap="large")
        c1.metric("Avg Fundamental Score", f"{avg_f:.1f}")
        c2.metric("% Strong", f"{pct_strong:.0f}%")
        c3.metric("Sectors", f"{n_sector:,}")
        c4.metric("Emitens", f"{n_rows:,}")

        left, right = st.columns([0.42, 0.50], gap="large")

        with left:
            bucket_order = ["Strong", "Watch", "Risky", "Other"]

            donut = (
                core["fund_bucket"]
                .value_counts()
                .reindex(bucket_order, fill_value=0)
                .reset_index()
            )
            donut.columns = ["Bucket", "Count"]
            donut = donut[donut["Count"] > 0]

            # sekarang harusnya tidak akan kosong
            fig = px.pie(
                donut,
                names="Bucket",
                values="Count",
                hole=0.62,
                color="Bucket",
                category_orders={"Bucket": bucket_order},
                color_discrete_map={**BUCKET_COLORS, "Other": "#9aa0a6"},
            )
            fig.update_traces(sort=False, textposition="inside", textinfo="percent")
            fig.update_layout(
                height=360,
                title="Distribusi Bucket (Fundamental)",
                margin=dict(t=110, b=20, l=20, r=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.18,
                    xanchor="center",
                    x=0.5,
                    title=None,
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            sec_avg = (
                core.groupby("sector", dropna=False)["fund_health_0_100"]
                .mean()
                .reset_index()
                .sort_values("fund_health_0_100", ascending=False)
                .head(12)
            )
            fig = px.bar(sec_avg, x="fund_health_0_100", y="sector", orientation="h")
            fig = style_fig(fig, height=360, title="Rata-rata Fundamental Score per Sektor (Top 12)")
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("🏁 Top Kandidat (Fundamental)")
        only_complete = st.toggle("Hanya yang lengkap (Profit+Leverage+Growth)", value=False, key="fund_complete_only")
        core_view = core.copy()
        if only_complete:
            core_view = core_view.dropna(subset=["S_Profit", "S_Leverage", "S_Growth", "fund_health_0_100"])

        show_per_sector = st.toggle("Tampilkan ranking per sektor", value=True, key="show_rank_per_sector")
        if show_per_sector and not core_view["sector"].dropna().empty:
            top_each = st.slider("Top N per sektor", 5, 100, 15, 5, key="topn_per_sector")
            sectors_sorted = sorted(core_view["sector"].dropna().astype(str).unique().tolist())
            for sec in sectors_sorted:
                sec_df = core_view[core_view["sector"].astype(str) == sec].sort_values("fund_health_0_100", ascending=False)
                with st.expander(f"{sec} — {len(sec_df):,} emiten", expanded=False):
                    cols = ["fund_bucket", "ticker_base", "year", "fund_health_0_100", "S_Profit", "S_Leverage", "S_Growth"]
                    df_show = round_cols(sec_df[cols].head(top_each), ["fund_health_0_100", "S_Profit", "S_Leverage", "S_Growth"], 1)
                    st.dataframe(df_show, use_container_width=True, hide_index=True)

        topn = safe_topn_slider("Top N (overall)", len(core_view), default=30, min_floor=5, cap=300, key="topn_fund")
        view = core_view.sort_values("fund_health_0_100", ascending=False).head(topn).copy()
        show_cols = ["fund_bucket", "ticker_base", "year", "sector", "fund_health_0_100", "S_Profit", "S_Leverage", "S_Growth"]
        view = round_cols(view[show_cols], ["fund_health_0_100", "S_Profit", "S_Leverage", "S_Growth"], 1)

        st.dataframe(view, use_container_width=True, hide_index=True)

        bar_df = view.sort_values("fund_health_0_100", ascending=True).copy()
        fig = px.bar(
            bar_df,
            x="fund_health_0_100",
            y="ticker_base",
            orientation="h",
            color="fund_bucket",
            color_discrete_map={**BUCKET_COLORS, "Other": "#9aa0a6"},
            category_orders={"fund_bucket": bucket_order},
        )

        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.12,
                xanchor="center",
                x=0.5,
                title=None,
            ),
            margin=dict(t=100, b=20, l=80, r=30),
        )

        fig = style_fig(
            fig,
            height=min(820, 160 + 24 * len(bar_df)),
            title=f"Top {len(bar_df)} Fundamental Score (0–100)",
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Kenapa ada nilai 0 atau 100 di sub-score?", expanded=False):
            st.write(
                "Ini normal karena sub-score memakai normalisasi 0–100 per **tahun & sektor** (winsor + minmax). "
                "Yang terbaik di peer-group bisa mendekati 100, yang terburuk bisa mendekati 0. "
                "Tujuannya supaya perbandingan antar emiten dalam sektor yang sama lebih fair."
            )

    # -------------------------
    # GABUNGAN
    # -------------------------
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧩 Combined Snapshot (Teknikal + Fundamental)")
        st.caption("Tujuan: shortlist kandidat yang sehat secara fundamental dan bagus secara price action.")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.spinner("Loading price data..."):
            raw_data = load_prices(picked, start=str(start), debug=True)

        data = show_prices_debug(raw_data, picked, debug_mode)

        if not data:
            st.warning("Tidak ada data harga yang berhasil diambil. Cek ticker / koneksi.")
            st.stop()

        tech_rows = []
        for t, dfp in data.items():
            s = health_from_df(dfp)
            tech_rows.append(
                {
                    "ticker": t,
                    "ticker_base": to_base_ticker(t),
                    "health_tech": s.get("health", np.nan),
                    "trend": s.get("trend", np.nan),
                    "risk": s.get("risk", np.nan),
                    "liquidity": s.get("liq", np.nan),
                }
            )

        tech_df = pd.DataFrame(tech_rows).dropna(subset=["health_tech"])
        if tech_df.empty:
            st.warning("Tidak ada skor teknikal yang valid.")
            st.stop()

        fund2 = core_ref[["ticker_base", "sector", "fund_health_0_100", "fund_bucket"]].rename(
            columns={"fund_health_0_100": "fund_norm_0_100"}
        )
        combined = tech_df.merge(fund2, on="ticker_base", how="left")

        if sync_tech_to_fund:
            combined = combined.dropna(subset=["fund_norm_0_100"])

        if combined.empty:
            st.warning("Tidak ada ticker yang match setelah sinkron fundamental.")
            st.stop()

        denom = (w_tech + w_fund) if (w_tech + w_fund) > 0 else 1.0
        combined["score_combined"] = ((combined["health_tech"] * w_tech) + (combined["fund_norm_0_100"] * w_fund)) / denom
        combined["bucket"] = combined["score_combined"].apply(label_bucket)
        combined = round_cols(combined, ["health_tech", "fund_norm_0_100", "score_combined"], 2)
        combined = add_label_from_score(combined, "score_combined", out_col="label")
        combined = combined.sort_values("score_combined", ascending=False)

        avg_c = float(pd.to_numeric(combined["score_combined"], errors="coerce").mean())
        pct_strong = float((combined["bucket"] == "Strong").mean() * 100)
        avg_t = float(pd.to_numeric(combined["health_tech"], errors="coerce").mean())
        avg_f = float(pd.to_numeric(combined["fund_norm_0_100"], errors="coerce").mean())

        c1, c2, c3, c4 = st.columns(4, gap="large")
        c1.metric("Avg Combined", f"{avg_c:.1f}")
        c2.metric("% Strong", f"{pct_strong:.0f}%")
        c3.metric("Avg Tech", f"{avg_t:.1f}")
        c4.metric("Avg Fund", f"{avg_f:.1f}")

        left, right = st.columns([0.42, 0.58], gap="large")
        with left:
            donut = combined["bucket"].value_counts().reset_index()
            donut.columns = ["Bucket", "Count"]
            fig = px.pie(donut, names="Bucket", values="Count", hole=0.62)
            fig = style_fig(fig, height=360, title="Distribusi Bucket (Combined)")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            w_df = pd.DataFrame({"Component": ["Tech", "Fund"], "Weight": [float(w_tech), float(w_fund)]})
            fig = px.bar(w_df, x="Component", y="Weight", text="Weight")
            fig = style_fig(fig, height=360, title="Bobot Gabungan (User-controlled)")
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("🏁 Top Kandidat (Combined)")
        top_n = safe_topn_slider("Top N", len(combined), default=30, min_floor=5, cap=200, key="topn_combo")
        view = combined.head(top_n).copy()
        show_cols = ["ticker", "sector", "score_combined", "label", "health_tech", "fund_norm_0_100", "trend", "risk", "liquidity"]
        st.dataframe(view[show_cols], use_container_width=True, hide_index=True)

        plot_df = combined.copy()
        fig = px.scatter(
            plot_df,
            x="health_tech",
            y="fund_norm_0_100",
            size="score_combined",
            color="bucket",
            hover_name="ticker",
            hover_data={
                "score_combined": ":.2f",
                "health_tech": ":.2f",
                "fund_norm_0_100": ":.2f",
                "bucket": True,
                "label": True,
                "sector": True,
                "trend": ":.2f",
                "risk": ":.2f",
                "liquidity": ":.2f",
            },
        )

        fig.add_shape(type="rect", x0=0, x1=60, y0=0, y1=60, fillcolor="rgba(255,99,132,0.10)", line_width=0)
        fig.add_shape(type="rect", x0=60, x1=100, y0=0, y1=60, fillcolor="rgba(255,159,67,0.10)", line_width=0)
        fig.add_shape(type="rect", x0=0, x1=60, y0=60, y1=100, fillcolor="rgba(160,160,160,0.08)", line_width=0)
        fig.add_shape(type="rect", x0=60, x1=100, y0=60, y1=100, fillcolor="rgba(0,184,148,0.10)", line_width=0)
        fig.add_vline(x=60, line_width=1)
        fig.add_hline(y=60, line_width=1)
        fig.add_annotation(x=80, y=92, text="Sweet Spot", showarrow=False)
        fig.add_annotation(x=25, y=92, text="Fund OK, Tech Weak", showarrow=False)
        fig.add_annotation(x=80, y=25, text="Tech OK, Fund Weak", showarrow=False)
        fig.add_annotation(x=25, y=25, text="Avoid Zone", showarrow=False)

        fig = style_fig(fig, height=640, title="Teknikal vs Fundamental (Bubble • 0–100)")
        st.plotly_chart(fig, use_container_width=True)


# ======================================================
# ANALYZE (detail 1 saham)
# ======================================================
with tab_analyze:
    render_kpis(options_universe, start, picked, sector_list)
    st.divider()
    st.subheader("Analyze — Detail 1 saham")
    st.caption("Pilih 1 saham untuk lihat chart harga + snapshot skor (Tech/Fund/Combined).")

    analyze_pick = st.selectbox("Pilih ticker", options=picked, key="analyze_pick")
    base = to_base_ticker(analyze_pick)

    with st.spinner("Loading detail..."):
        raw_1 = load_prices([analyze_pick], start=str(start), debug=True)
        if "_ERROR_" in raw_1:
            st.error(raw_1["_ERROR_"])
        data_1 = clean_price_dict(raw_1)

        dfp = data_1.get(analyze_pick)
        tech = health_from_df(dfp) if dfp is not None else {}

        fund_row = core_ref[core_ref["ticker_base"] == base].head(1)
        fund_score = float(fund_row["fund_health_0_100"].iloc[0]) if not fund_row.empty else np.nan
        tech_score = float(tech.get("health", np.nan)) if tech else np.nan

        denom = (w_tech + w_fund) if (w_tech + w_fund) > 0 else 1.0
        combo_score = (
            (tech_score * w_tech + fund_score * w_fund) / denom
            if (not np.isnan(tech_score) and not np.isnan(fund_score))
            else np.nan
        )

    if not np.isnan(combo_score):
        lbl, kind = score_to_label_kind(float(combo_score))
    else:
        lbl, kind = ("—", "neutral")

    st.markdown(f"### {analyze_pick}  {badge_html(lbl, kind)}", unsafe_allow_html=True)

    sec = fund_row["sector"].iloc[0] if (not fund_row.empty and "sector" in fund_row.columns) else "—"
    fb = fund_row["fund_bucket"].iloc[0] if (not fund_row.empty and "fund_bucket" in fund_row.columns) else "—"
    st.caption(f"Sektor: {sec} • Fundamental bucket: {fb} • Base: {base}")

    left, right = st.columns([0.68, 0.32], gap="large")

    with left:
        if dfp is None or dfp.empty or "Close" not in dfp.columns:
            st.warning("Tidak ada data harga untuk ticker ini.")
        else:
            dfp = dfp.dropna(subset=["Close"]).copy()
            render_symbol_header(analyze_pick, base, "Candlestick + MA50/MA200")

            fig = go.Figure()
            fig.add_candlestick(
                x=dfp["Date"],
                open=dfp["Open"],
                high=dfp["High"],
                low=dfp["Low"],
                close=dfp["Close"],
                name=analyze_pick,
            )
            if "ma50" in dfp.columns:
                fig.add_scatter(x=dfp["Date"], y=dfp["ma50"], name="MA50")
            if "ma200" in dfp.columns:
                fig.add_scatter(x=dfp["Date"], y=dfp["ma200"], name="MA200")

            fig.update_layout(xaxis_rangeslider_visible=False)
            fig = style_fig(fig, height=560, title="Price (Candlestick)")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        with st.container(border=True):
            st.subheader("Scores")
            c1, c2, c3 = st.columns(3)
            c1.metric("Combined", "—" if np.isnan(combo_score) else f"{combo_score:.1f}")
            c2.metric("Tech", "—" if np.isnan(tech_score) else f"{tech_score:.1f}")
            c3.metric("Fund", "—" if np.isnan(fund_score) else f"{fund_score:.1f}")

        with st.container(border=True):
            st.subheader("Technical snapshot")
            if not tech:
                st.info("Snapshot teknikal tidak tersedia.")
            else:
                t_trend = tech.get("trend", np.nan)
                t_risk = tech.get("risk", np.nan)
                t_liq = tech.get("liq", np.nan)
                t_rsi = tech.get("rsi", np.nan)
                t_mdd = tech.get("mdd", np.nan)
                t_close = tech.get("close", np.nan)

                st.write(f"Close: {t_close:.2f}" if pd.notna(t_close) else "Close: —")
                st.write(f"Trend: {t_trend:.2f}" if pd.notna(t_trend) else "Trend: —")
                st.write(f"Risk: {t_risk:.2f}" if pd.notna(t_risk) else "Risk: —")
                st.write(f"Liquidity: {t_liq:.2f}" if pd.notna(t_liq) else "Liquidity: —")
                st.write(f"RSI14: {t_rsi:.2f}" if pd.notna(t_rsi) else "RSI14: —")
                st.write(f"Max Drawdown: {t_mdd:.2f}" if pd.notna(t_mdd) else "Max Drawdown: —")

        with st.container(border=True):
            st.subheader("Fundamental snapshot")
            if fund_row.empty:
                st.info("Tidak ada data fundamental untuk ticker ini pada filter ref_year/sector sekarang.")
            else:
                fr = fund_row.iloc[0]
                items = [
                    ("Profit", fr.get("S_Profit", np.nan)),
                    ("Leverage", fr.get("S_Leverage", np.nan)),
                    ("Growth", fr.get("S_Growth", np.nan)),
                ]
                for name, val in items:
                    st.write(f"{name}: {float(val):.1f}" if pd.notna(val) else f"{name}: —")


# ======================================================
# COMPARE (2–5 saham)
# ======================================================
with tab_compare:
    render_kpis(options_universe, start, picked, sector_list)
    st.divider()
    st.subheader("Compare — 2–5 saham")
    st.caption("Bandingkan 2–5 saham dengan tabel ringkas + peta Tech vs Fund.")

    picks = st.multiselect(
        "Pilih 2–5 ticker",
        options=options_universe,
        default=picked[:2],
        max_selections=5,
        key="compare_picks",
    )

    if len(picks) < 2:
        st.info("Pilih minimal 2 ticker untuk compare.")
    else:
        with st.spinner("Loading compare data..."):
            raw_cmp = load_prices(picks, start=str(start), debug=True)

        if "_ERROR_" in raw_cmp:
            st.error(raw_cmp["_ERROR_"])
        data = clean_price_dict(raw_cmp)

        rows = []
        for t in picks:
            dfp = data.get(t)
            tech = health_from_df(dfp) if dfp is not None else {}
            base = to_base_ticker(t)

            fund_row = core_ref[core_ref["ticker_base"] == base].head(1)
            fund_score = float(fund_row["fund_health_0_100"].iloc[0]) if not fund_row.empty else np.nan
            tech_score = float(tech.get("health", np.nan)) if tech else np.nan

            denom = (w_tech + w_fund) if (w_tech + w_fund) > 0 else 1.0
            combo_score = (
                (tech_score * w_tech + fund_score * w_fund) / denom
                if (not np.isnan(tech_score) and not np.isnan(fund_score))
                else np.nan
            )

            rows.append(
                {
                    "ticker": t,
                    "sector": fund_row["sector"].iloc[0] if (not fund_row.empty and "sector" in fund_row.columns) else np.nan,
                    "bucket": (label_bucket(combo_score) if pd.notna(combo_score) else "—"),
                    "score_combined": combo_score,
                    "health_tech": tech_score,
                    "fund_0_100": fund_score,
                }
            )

        comp = pd.DataFrame(rows)
        comp = round_cols(comp, ["score_combined", "health_tech", "fund_0_100"], 2).sort_values("score_combined", ascending=False)

        st.dataframe(comp, use_container_width=True, hide_index=True)

        fig = px.scatter(
            comp,
            x="health_tech",
            y="fund_0_100",
            size="score_combined",
            color="bucket",
            hover_name="ticker",
            hover_data={"sector": True, "score_combined": ":.2f", "health_tech": ":.2f", "fund_0_100": ":.2f"},
        )
        fig.add_shape(type="rect", x0=60, x1=100, y0=60, y1=100, fillcolor="rgba(0,184,148,0.10)", line_width=0)
        fig.add_vline(x=60, line_width=1)
        fig.add_hline(y=60, line_width=1)
        fig = style_fig(fig, height=560, title="Compare: Technical vs Fundamental (Bubble)")
        st.plotly_chart(fig, use_container_width=True)


# ======================================================
# METHODOLOGY
# ======================================================
with tab_method:
    st.subheader("Methodology (non-teknis)")
    with st.container(border=True):
        st.write("• Skor 0–100: makin tinggi makin menarik menurut model scoring kamu.")
        st.write("• Fundamental dinormalisasi per **tahun & sektor** agar fair dibandingkan peer group.")
        st.write("• Red flags ditampilkan terpisah agar tidak mengganggu ranking utama.")
        st.write("• Skor gabungan = bobot(Tech) + bobot(Fund) (dibagi total bobot).")
