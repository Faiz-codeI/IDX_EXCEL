*** Begin Patch
*** Add File: src/ui_kit.py
+from __future__ import annotations
+
+import streamlit as st
+
+
+def inject_css() -> None:
+    st.markdown(
+        """
+        <style>
+        .block-container { padding-top: 1.15rem; padding-bottom: 2rem; }
+        .muted { color: rgba(250,250,250,0.68); }
+        .small-note { color: rgba(250,250,250,0.70); font-size: 0.90rem; }
+
+        .badge {
+          display:inline-block;
+          padding: 0.22rem 0.55rem;
+          border-radius: 999px;
+          font-size: 0.84rem;
+          font-weight: 650;
+          border: 1px solid rgba(255,255,255,0.12);
+          line-height: 1.1;
+          background: rgba(160,160,160,0.16);
+        }
+        .badge-strong { background: rgba(0, 184, 148, 0.18); }
+        .badge-neutral { background: rgba(160, 160, 160, 0.18); }
+        .badge-risky { background: rgba(255, 159, 67, 0.18); }
+        .badge-avoid { background: rgba(255, 99, 132, 0.18); }
+
+        .card {
+          padding: 0.85rem 0.9rem;
+          border-radius: 16px;
+          border: 1px solid rgba(255,255,255,0.10);
+          background: rgba(255,255,255,0.02);
+        }
+        </style>
+        """,
+        unsafe_allow_html=True,
+    )
+
+
+def badge_html(label: str, kind: str) -> str:
+    cls = {
+        "strong": "badge badge-strong",
+        "neutral": "badge badge-neutral",
+        "risky": "badge badge-risky",
+        "avoid": "badge badge-avoid",
+    }.get(kind, "badge badge-neutral")
+    return f'<span class="{cls}">{label}</span>'
+
+
+def score_to_label_kind(score: float) -> tuple[str, str]:
+    # mapping generik; aman untuk 0–100
+    if score >= 80:
+        return ("Strong", "strong")
+    if score >= 60:
+        return ("Neutral", "neutral")
+    if score >= 40:
+        return ("Risky", "risky")
+    return ("Avoid", "avoid")
+
+
+def page_header(title: str, subtitle: str) -> None:
+    st.title(title)
+    st.caption(subtitle)
+
+
+def safe_topn_slider(
+    label: str,
+    n_rows: int,
+    default: int = 20,
+    min_floor: int = 5,
+    cap: int = 300,
+    key: str | None = None,
+) -> int:
+    max_n = int(min(cap, n_rows))
+    if max_n <= 0:
+        return 0
+    if max_n == 1:
+        st.caption(f"{label}: hanya 1 baris data tersedia.")
+        return 1
+
+    min_n = int(min(min_floor, max_n))
+    default_n = int(min(default, max_n))
+    if min_n >= max_n:
+        min_n = 1
+
+    return st.slider(label, min_n, max_n, default_n, key=key)
+
*** End Patch
