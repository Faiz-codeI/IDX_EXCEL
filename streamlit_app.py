from PIL import Image
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.dataio import (
    load_fundamentals,
    infer_ticker_column,
    infer_score_column,
    to_ticker_jk,
    to_base_ticker,
    load_universe_from_txt,
)
from src.technical import load_prices, health_from_df
from src.scoring import normalize_0_100, label_bucket


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


def style_fig(fig: go.Figure, height: int, title: str):
    fig.update_layout(height=height, title=title, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def round_cols(df: pd.DataFrame, cols, digits: int):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].round(digits)
    return df


def render_kpis(universe, start, picked):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Universe", f"{len(universe):,} ticker")
    c2.metric("Start Date", str(start))
    c3.metric("Selected", len(picked))
    c4.metric("Mode", "Teknikal + Fundamental")


st.set_page_config(page_title="IDX Dashboard — Teknikal + Fundamental", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
div[data-testid="stVerticalBlock"] {gap: 0.6rem;}
h1, h2, h3 {letter-spacing: -0.3px;}
section[data-testid="stSidebar"] .block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

st.title("📊 IDX Stock Dashboard")
st.caption("Teknikal • Fundamental • Skor Gabungan (Universe dinamis dari fundamentals_table.csv)")


DEFAULT_FUNDAMENTALS_CSV = "data/fundamentals_table.csv"
DEFAULT_UNIVERSE_TXT = "data/universe_tickers.txt"


with st.sidebar:
    st.header("📦 Data Universe")
    fundamentals_path = st.text_input("Path fundamentals_table.csv", value=DEFAULT_FUNDAMENTALS_CSV)
    use_universe_txt = st.toggle("Batasi universe pakai universe_tickers.txt", value=False)
    universe_txt_path = st.text_input("Path universe_tickers.txt (optional)", value=DEFAULT_UNIVERSE_TXT)

    st.divider()
    st.header("⚙️ Pengaturan Teknikal")
    start = st.date_input("Mulai data harga dari", value=pd.to_datetime("2020-01-01"))

    st.divider()
    st.header("🧩 Pengaturan Gabungan")
    w_tech = st.slider("Bobot Teknikal", 0.0, 1.0, 0.5, 0.05)
    w_fund = st.slider("Bobot Fundamental", 0.0, 1.0, 0.5, 0.05)


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

universe_base = (
    fund[ticker_col_guess]
    .astype(str)
    .map(to_base_ticker)
    .dropna()
    .unique()
    .tolist()
)
universe_base = sorted({x for x in universe_base if x})

if use_universe_txt:
    allowed = load_universe_from_txt(universe_txt_path)
    if allowed:
        allowed_base = {to_base_ticker(x) for x in allowed}
        universe_base = [t for t in universe_base if t in allowed_base]

universe = [to_ticker_jk(x) for x in universe_base]


with st.sidebar:
    st.header("✅ Pilih Ticker")
    default_pick = st.session_state.get("picked", universe[:1] if universe else [])
    picked = st.multiselect("Pilih saham (dinamis dari CSV)", options=universe, default=default_pick)
    st.session_state["picked"] = picked
    if not picked:
        st.warning("Pilih minimal 1 saham.")
        st.stop()


tab_explore, tab_tech, tab_fund, tab_combo = st.tabs(
    ["🧭 Explore", "📈 Teknikal", "🧾 Fundamental", "🧩 Gabungan"]
)


with tab_explore:
    render_kpis(universe, start, picked)
    st.divider()

    st.subheader("Explore — Insight Dashboard")

    mode = st.radio(
        "Mode Analisis",
        ["Teknikal", "Fundamental", "Gabungan"],
        horizontal=True,
        key="explore_mode"
    )

    if mode == "Teknikal":
        st.subheader("📈 Technical Market Snapshot")

        with st.spinner("Loading price data..."):
            data = load_prices(picked, start=str(start))

        if not data:
            st.warning("Tidak ada data harga yang berhasil diambil.")
            st.stop()

        rows = []
        for t, dfp in data.items():
            s = health_from_df(dfp)
            rows.append({
                "ticker": t,
                "ticker_base": to_base_ticker(t),
                "health_tech": s.get("health"),
                "trend": s.get("trend"),
                "risk": s.get("risk"),
                "liquidity": s.get("liq"),
            })

        tech_df = pd.DataFrame(rows).dropna(subset=["health_tech"])
        if tech_df.empty:
            st.warning("Skor teknikal belum bisa dihitung (data kurang panjang / banyak NaN).")
            st.stop()

        tech_df["bucket"] = tech_df["health_tech"].apply(label_bucket)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Technical Score", f"{tech_df['health_tech'].mean():.1f}")
        c2.metric("% Strong", f"{(tech_df['bucket']=='Strong').mean()*100:.0f}%")
        c3.metric("Median Risk", f"{tech_df['risk'].median():.2f}")
        c4.metric("Saham Dianalisis", len(tech_df))

        st.divider()

        left, right = st.columns([0.45, 0.55])

        with left:
            donut = tech_df["bucket"].value_counts().reset_index()
            donut.columns = ["Bucket", "Count"]
            fig = px.pie(donut, names="Bucket", values="Count", hole=0.6)
            fig = style_fig(fig, height=360, title="Distribusi Kualitas Teknikal")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            bar_df = tech_df[["trend", "risk"]].mean().reset_index()
            bar_df.columns = ["Metric", "Average"]
            fig = px.bar(bar_df, x="Metric", y="Average", text="Average")
            fig = style_fig(fig, height=360, title="Rata-rata Trend vs Risk")
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        top = tech_df.sort_values("health_tech", ascending=False).head(15)
        top = round_cols(top, ["health_tech", "trend", "risk", "liquidity"], 2)
        st.dataframe(
            top[["ticker", "health_tech", "bucket", "trend", "risk", "liquidity"]],
            use_container_width=True,
            hide_index=True
        )
    elif mode == "Fundamental":
        st.info("Fundamental dashboard akan dirapikan menyusul.")
    elif mode == "Gabungan":
        st.info("Gabungan dashboard akan dirapikan menyusul.")


with tab_tech:
    data = load_prices(picked, start=str(start))
    if not data:
        st.warning("Tidak ada data harga yang berhasil diambil. Cek ticker / koneksi.")
        st.stop()

    rows = []
    for t, dfp in data.items():
        s = health_from_df(dfp)
        rows.append({
            "ticker": t,
            "ticker_base": to_base_ticker(t),
            "health_tech": s["health"],
            "trend": s["trend"],
            "risk": s["risk"],
            "liquidity": s["liq"],
            "close": s["close"],
            "rsi14": s["rsi"],
            "vol_20d": s["vol20"],
            "max_drawdown": s["mdd"],
            "vol_ratio": s["vol_ratio"],
        })

    summary = pd.DataFrame(rows).dropna(subset=["health_tech"]).sort_values("health_tech", ascending=False)
    if summary.empty:
        st.warning("Skor teknikal belum bisa dihitung (data kurang panjang / banyak NaN).")
        st.stop()

    st.subheader("🏁 Ranking Kesehatan Saham (Teknikal)")
    top_n = st.slider("Tampilkan Top N (Teknikal)", 5, min(300, len(summary)), min(20, len(summary)))
    summary_view = summary.head(top_n)

    st.dataframe(
        summary_view.style
          .format({
              "health_tech": "{:.1f}",
              "trend": "{:.0f}",
              "risk": "{:.1f}",
              "liquidity": "{:.1f}",
              "close": "{:,.0f}",
              "rsi14": "{:.1f}",
              "vol_20d": "{:.2f}",
              "max_drawdown": "{:.1%}",
              "vol_ratio": "{:.2f}",
          })
          .background_gradient(subset=["health_tech"]),
        use_container_width=True
    )

    st.caption("Skor teknikal: Trend (MA50/MA200), Risk (volatilitas & drawdown), Liquidity (volume).")
    st.divider()

    pick = st.selectbox("Detail chart saham:", options=summary["ticker"].tolist())
    if pick in data:
        dfp = data[pick].dropna()
        base = to_base_ticker(pick)
        render_symbol_header(pick, base, "Teknikal — MA / RSI / Volume")

        col1, col2, col3, col4 = st.columns(4)
        srow = summary[summary["ticker"] == pick].iloc[0].to_dict()
        col1.metric("Health", f"{srow['health_tech']:.1f}/100")
        col2.metric("Close", f"{srow['close']:,.0f}")
        col3.metric("RSI(14)", f"{srow['rsi14']:.1f}")
        col4.metric("Max Drawdown", f"{srow['max_drawdown']:.1%}")

        fig = go.Figure()
        fig.add_candlestick(
            x=dfp["Date"],
            open=dfp["Open"], high=dfp["High"], low=dfp["Low"], close=dfp["Close"],
            name=pick
        )
        fig.add_scatter(x=dfp["Date"], y=dfp["ma50"], name="MA50")
        fig.add_scatter(x=dfp["Date"], y=dfp["ma200"], name="MA200")
        fig.update_layout(height=550, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("RSI & Volume", expanded=True):
            rsi_fig = go.Figure()
            rsi_fig.add_scatter(x=dfp["Date"], y=dfp["rsi14"], name="RSI14")
            rsi_fig.update_layout(height=250)
            st.plotly_chart(rsi_fig, use_container_width=True)

            vol_fig = go.Figure()
            vol_fig.add_bar(x=dfp["Date"], y=dfp["Volume"], name="Volume")
            vol_fig.add_scatter(x=dfp["Date"], y=dfp["vol_avg20"], name="Avg20 Volume")
            vol_fig.update_layout(height=250)
            st.plotly_chart(vol_fig, use_container_width=True)


with tab_fund:
    st.subheader("🧾 Fundamental (fundamentals_table.csv)")
    cols = list(fund.columns)
    score_guess = infer_score_column(fund)

    with st.expander("🔧 Mapping kolom Fundamental", expanded=True):
        cc1, cc2 = st.columns(2)
        with cc1:
            ticker_col = st.selectbox(
                "Kolom ticker/kode emiten",
                cols,
                index=cols.index(ticker_col_guess) if ticker_col_guess in cols else 0
            )
        with cc2:
            num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(fund[c])]
            if not num_cols:
                st.error("Tidak ada kolom numerik di fundamentals_table.csv")
                st.stop()
            score_col = st.selectbox(
                "Kolom fundamental score (atau metrik numerik)",
                num_cols,
                index=num_cols.index(score_guess) if score_guess in num_cols else 0
            )

    fund_view = fund.copy()
    fund_view["ticker_base"] = fund_view[ticker_col].astype(str).map(to_base_ticker)
    fund_view["ticker"] = fund_view["ticker_base"].map(to_ticker_jk)

    only_picked = st.toggle("Tampilkan hanya ticker yang dipilih di sidebar", value=True)
    if only_picked:
        fund_view = fund_view[fund_view["ticker"].isin(picked)]

    fund_view = fund_view.sort_values(score_col, ascending=False)
    topn = st.slider("Tampilkan Top N (Fundamental)", 5, min(300, len(fund_view)), min(30, len(fund_view)))
    fv = fund_view.head(topn)

    st.dataframe(fv, use_container_width=True, height=420)
    if len(fv):
        fig = px.bar(
            fv.sort_values(score_col),
            x=score_col, y="ticker_base", orientation="h",
            title=f"Top {len(fv)} Fundamental"
        )
        st.plotly_chart(fig, use_container_width=True)


with tab_combo:
    st.subheader("🧩 Gabungan (Teknikal + Fundamental)")

    data = load_prices(picked, start=str(start))
    tech_rows = []
    for t, dfp in data.items():
        s = health_from_df(dfp)
        tech_rows.append({
            "ticker": t,
            "ticker_base": to_base_ticker(t),
            "health_tech": s["health"],
            "trend": s["trend"],
            "risk": s["risk"],
            "liquidity": s["liq"],
        })
    tech_df = pd.DataFrame(tech_rows).dropna(subset=["health_tech"])

    score_guess = infer_score_column(fund)
    if score_guess is None:
        st.warning("Kolom fundamental numerik belum terdeteksi. Pilih di tab Fundamental.")
        st.stop()

    fund2 = fund.copy()
    fund2["ticker_base"] = fund2[ticker_col_guess].astype(str).map(to_base_ticker)
    fund2 = fund2.rename(columns={score_guess: "score_fund"})

    combined = tech_df.merge(fund2[["ticker_base", "score_fund"]], on="ticker_base", how="left")
    combined["fund_norm_0_100"] = normalize_0_100(combined["score_fund"])

    denom = (w_tech + w_fund) if (w_tech + w_fund) > 0 else 1.0
    combined["score_combined"] = (
        (combined["health_tech"] * w_tech) +
        (combined["fund_norm_0_100"] * w_fund)
    ) / denom

    combined["bucket"] = combined["score_combined"].apply(label_bucket)
    view = combined.sort_values("score_combined", ascending=False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah ticker", f"{len(view):,}")
    c2.metric("Ada fundamental", f"{view['fund_norm_0_100'].notna().sum():,}")
    c3.metric("Bobot T/F", f"{w_tech:.2f} / {w_fund:.2f}")
    c4.metric("Avg combined", f"{view['score_combined'].mean():.2f}" if len(view) else "-")
    st.divider()

    left, right = st.columns([1.1, 2.2], gap="large")
    with left:
        st.subheader("📌 Watchlist")
        top_c = st.slider("Top N (Gabungan)", 5, min(300, len(view)), min(20, len(view)))
        view_top = view.head(top_c).copy()

        q = st.text_input("Search ticker (contoh: BBCA / UNVR)", value="")
        if q.strip():
            ql = q.strip().lower()
            view_top = view_top[
                view_top["ticker"].str.lower().str.contains(ql) |
                view_top["ticker_base"].str.lower().str.contains(ql)
            ]

        options = view_top["ticker"].tolist()
        if not options:
            st.info("Tidak ada hasil untuk filter/search tersebut.")
            st.stop()

        chosen = st.radio("Pilih emiten", options, label_visibility="collapsed")
        st.caption("Ringkas Top list")
        st.dataframe(
            view_top[["bucket", "ticker_base", "score_combined"]].rename(columns={
                "ticker_base": "Ticker",
                "score_combined": "Combined"
            }),
            use_container_width=True,
            height=320
        )

    with right:
        row = view[view["ticker"] == chosen].iloc[0].to_dict()
        render_symbol_header(chosen, row["ticker_base"], f"Status: {row['bucket']}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Combined", f"{row['score_combined']:.1f}")
        m2.metric("Teknikal", f"{row['health_tech']:.1f}")
        m3.metric("Fund (0–100)", f"{row['fund_norm_0_100']:.1f}" if pd.notna(row["fund_norm_0_100"]) else "-")
        m4.metric("Trend", f"{row['trend']:.0f}")

        st.divider()
        if view["fund_norm_0_100"].notna().any():
            fig = px.scatter(
                view,
                x="health_tech",
                y="fund_norm_0_100",
                hover_data=["ticker", "bucket"],
                title="Teknikal vs Fundamental (normalized 0–100)"
            )
            fig.add_hline(y=60)
            fig.add_vline(x=60)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Kanan-atas = fundamental bagus + teknikal sehat (ideal).")
        else:
            st.info("Belum ada fundamental yang match ticker (cek format ticker: UNVR vs UNVR.JK).")

        with st.expander("Detail skor (row)"):
            st.write(pd.Series(row))
