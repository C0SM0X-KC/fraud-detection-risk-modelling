"""Analyst console."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from theme import CSS  # noqa: E402

st.set_page_config(page_title="Fraud Risk Console", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

views = Path(__file__).resolve().parent / "views"
page = st.navigation([
    st.Page(views / "check.py", title="Check a transaction", default=True),
    st.Page(views / "ledger.py", title="Live ledger"),
    st.Page(views / "analytics.py", title="Fraud analytics"),
    st.Page(views / "monitor.py", title="Model and monitoring"),
])
page.run()
