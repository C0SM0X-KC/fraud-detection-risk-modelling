"""Render the console's analytics page headlessly and exercise its filters.

    python scripts/check_analytics_page.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT / "serving"), str(REPO_ROOT / "src")]

page = AppTest.from_file(str(REPO_ROOT / "serving" / "views" / "analytics.py"),
                         default_timeout=180)
page.run()
assert not page.exception, page.exception
print("default view: ok |", len(page.dataframe), "tables |",
      sum(len(page.get(k)) for k in ("arrow_vega_lite_chart", "vega_lite_chart")), "charts")

page.sidebar.multiselect[0].set_value(["test"])
page.sidebar.multiselect[1].set_value(["C"])
page.run()
assert not page.exception, page.exception
print("test span, product C: ok |", page.markdown[2].value[:90])

page.selectbox[0].set_value("Account history")
page.run()
assert not page.exception, page.exception
print("breakdown by account history: ok")
