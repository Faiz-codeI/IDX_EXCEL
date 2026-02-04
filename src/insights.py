from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


KIND_EMOJI = {
    "good": "✅",
    "neutral": "➖",
    "bad": "⚠️",
    "info": "ℹ️",
}


def _format_pct(value: float) -> str:
    return f"{value:.1f}%"


def _format_number(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}"


def _has_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    return all(col in df.columns for col in columns)


def build_market_pulse(df: pd.DataFrame) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    if df.empty:
        return items

    if _has_columns(df, ["score_combined"]):
        score = pd.to_numeric(df["score_combined"], errors="coerce")
        breadth_pct = float((score >= 60).mean() * 100)
        if breadth_pct >= 60:
            kind = "good"
            note = "Breadth is strong"
        elif breadth_pct >= 40:
            kind = "neutral"
            note = "Breadth is neutral"
        else:
            kind = "bad"
            note = "Breadth is weak"
        items.append(
            {
                "title": "Market Breadth",
                "value": _format_pct(breadth_pct),
                "note": note,
                "kind": kind,
            }
        )

    if _has_columns(df, ["risk"]):
        risk = pd.to_numeric(df["risk"], errors="coerce")
        med_risk = float(risk.median())
        if pd.isna(med_risk):
            med_risk_value = "—"
            kind = "info"
            note = "Median risk unavailable"
        else:
            if med_risk < 0.35:
                kind = "good"
                note = "Risk is low"
            elif med_risk <= 0.55:
                kind = "neutral"
                note = "Risk is moderate"
            else:
                kind = "bad"
                note = "Risk is elevated"
            med_risk_value = _format_number(med_risk)
        items.append(
            {
                "title": "Median Risk",
                "value": med_risk_value,
                "note": note,
                "kind": kind,
            }
        )

    if _has_columns(df, ["sector", "score_combined"]):
        sector_scores = (
            df[["sector", "score_combined"]]
            .dropna(subset=["sector", "score_combined"])
            .groupby("sector", dropna=False)["score_combined"]
            .median()
            .sort_values(ascending=False)
        )
        if not sector_scores.empty:
            leaders = ", ".join(sector_scores.head(2).index.astype(str))
            items.append(
                {
                    "title": "Sector Leaders",
                    "value": leaders,
                    "note": "Top 2 by median combined score",
                    "kind": "info",
                }
            )

    if _has_columns(df, ["health_tech", "fund_norm_0_100"]):
        tech = pd.to_numeric(df["health_tech"], errors="coerce")
        fund = pd.to_numeric(df["fund_norm_0_100"], errors="coerce")
        mismatch = (tech - fund) >= 20
        mismatch_count = int(mismatch.sum())
        items.append(
            {
                "title": "Mismatch",
                "value": f"{mismatch_count:,}",
                "note": "Tech ≥ Fund by 20+ points",
                "kind": "neutral" if mismatch_count else "good",
            }
        )

    return items


def build_bucket_summary(df: pd.DataFrame, bucket_col: str = "bucket") -> dict[str, dict[str, float]]:
    if bucket_col not in df.columns or df.empty:
        return {}

    counts = df[bucket_col].value_counts(dropna=False)
    total = float(counts.sum())
    if total <= 0:
        return {}

    summary: dict[str, dict[str, float]] = {}
    for bucket in ["Strong", "Watch", "Risky"]:
        count = int(counts.get(bucket, 0))
        pct = round((count / total) * 100, 1)
        summary[bucket] = {"count": count, "pct": pct}

    return summary


def render_insights(items: list[dict[str, Any]], title: str = "🧠 Market Pulse") -> None:
    with st.container(border=True):
        st.subheader(title)
        if not items:
            st.caption("No insights available.")
            return

        for item in items:
            prefix = KIND_EMOJI.get(item.get("kind", "info"), "ℹ️")
            title_text = item.get("title", "Insight")
            value = item.get("value", "—")
            note = item.get("note", "")
            note_suffix = f" — {note}" if note else ""
            st.markdown(f"• {prefix} **{title_text}**: {value}{note_suffix}")
