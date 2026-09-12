"""Visual identity for the analyst console.

The console is a queue, not a dashboard, and the design follows from that.

Colour carries meaning and nothing else. Two accents do two different jobs:
brass is what the model judged, oxide is what turned out to be true, and the gap
between them is the thing an analyst is actually assessing. Traffic that cleared
gets no colour at all, because clearing is the absence of an action rather than
a positive one. The consequence is that the screen is roughly ninety-seven per
cent quiet and three per cent marked, which is exactly the alert rate the system
produces - the palette states the result before any number does.

Every figure is set in a monospaced face with tabular lining numerals. That is
not a stylistic preference: a column of amounts and scores has to align on the
decimal to be scannable, and proportional figures make a queue unreadable.
"""

from __future__ import annotations

TOKENS = {
    "ink": "#12151C",        # page
    "stock": "#171B24",      # panels
    "rule": "#262C38",       # hairlines
    "type": "#C6CAD2",       # body
    "quiet": "#737C8B",      # secondary
    "faint": "#4A525F",      # tertiary, structural labels
    "brass": "#C9922E",      # the model's judgement
    "brass_dim": "#3A2F14",  # bar track
    "oxide": "#B8453A",      # confirmed fraud (ground truth)
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --ink:#12151C; --stock:#171B24; --rule:#262C38;
  --type:#C6CAD2; --quiet:#737C8B; --faint:#4A525F;
  --brass:#C9922E; --brass-dim:#3A2F14; --oxide:#B8453A;
  --mono:'IBM Plex Mono',ui-monospace,monospace;
  --sans:'IBM Plex Sans',system-ui,sans-serif;
}

/* Streamlit chrome: remove the parts that belong to the framework, not the product. */
#MainMenu, footer, header[data-testid="stHeader"] { display:none !important; }
[data-testid="stAppViewContainer"] { background:var(--ink); }
[data-testid="stSidebar"] { background:var(--stock); border-right:1px solid var(--rule); }
[data-testid="stAppViewBlockContainer"] { padding:2.2rem 2.6rem 4rem; max-width:1500px; }
[data-testid="stVerticalBlock"] { gap:0.9rem; }

html, body, [class*="css"] { font-family:var(--sans); color:var(--type); }

/* -- structural labels ------------------------------------------------ */
.eyebrow {
  font-family:var(--mono); font-size:10.5px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--faint); margin:0 0 .35rem;
}
.masthead {
  font-family:var(--sans); font-size:23px; font-weight:600; letter-spacing:-.01em;
  color:#E7E9ED; margin:0 0 .2rem;
}
.masthead-sub {
  font-family:var(--mono); font-size:11.5px; color:var(--quiet); margin:0 0 1.1rem;
}
.section {
  font-family:var(--mono); font-size:10.5px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--quiet);
  display:flex; align-items:center; gap:.8rem; margin:1.4rem 0 .7rem;
}
.section::after { content:""; flex:1; height:1px; background:var(--rule); }

/* -- replay position: a hairline, not a chunky bar -------------------- */
.tape { height:2px; background:var(--rule); margin:.2rem 0 1.5rem; position:relative; }
.tape > span { position:absolute; inset:0 auto 0 0; background:var(--brass); }

/* -- the figure line --------------------------------------------------- */
.figures { display:flex; gap:0; border-top:1px solid var(--rule); border-bottom:1px solid var(--rule); }
.fig { flex:1; padding:.85rem 1.1rem; border-right:1px solid var(--rule); }
.fig:last-child { border-right:none; }
.fig-k {
  font-family:var(--mono); font-size:10px; letter-spacing:.13em;
  text-transform:uppercase; color:var(--quiet); margin-bottom:.35rem;
}
.fig-v {
  font-family:var(--mono); font-size:27px; font-weight:500;
  font-variant-numeric:tabular-nums lining-nums; color:#E7E9ED; line-height:1;
}
.fig-v.mark { color:var(--brass); }
.fig-n { font-family:var(--mono); font-size:11px; color:var(--faint); margin-top:.3rem; }

/* -- reason codes: magnitude is visible, not just readable ------------- */
.reason { margin-bottom:.65rem; }
.reason-head {
  display:flex; justify-content:space-between; align-items:baseline;
  font-family:var(--mono); font-size:11.5px; margin-bottom:.25rem;
}
.reason-f { color:var(--type); }
.reason-v { color:var(--faint); }
.reason-c { color:var(--brass); font-variant-numeric:tabular-nums; }
.reason-bar { height:3px; background:var(--brass-dim); }
.reason-bar > span { display:block; height:100%; background:var(--brass); }

.verdict {
  font-family:var(--mono); font-size:11.5px; color:var(--quiet);
  border-left:2px solid var(--brass); padding:.5rem .9rem; margin-bottom:1rem;
  background:var(--stock);
}
.verdict b { color:#E7E9ED; font-weight:600; }
.verdict .held { color:var(--brass); }
.verdict .true { color:var(--oxide); }

/* -- where the fraud actually sits ------------------------------------ */
.band { display:flex; align-items:center; gap:.7rem; margin-bottom:.3rem;
        font-family:var(--mono); font-size:10.5px; }
.band-k { width:52px; color:var(--faint); text-align:right;
          font-variant-numeric:tabular-nums; }
.band-track { flex:1; height:9px; background:var(--stock); position:relative; }
.band-track > i { position:absolute; left:0; top:0; height:4px; background:var(--faint); }
.band-track > b { position:absolute; left:0; bottom:0; height:4px; background:var(--brass); }
.legend { font-family:var(--mono); font-size:10px; color:var(--faint);
          letter-spacing:.08em; margin-top:.5rem; }
.legend s { text-decoration:none; color:var(--faint); }
.legend em { font-style:normal; color:var(--brass); }

/* -- tables ------------------------------------------------------------ */
[data-testid="stDataFrame"] { border:1px solid var(--rule); }
[data-testid="stDataFrame"] * { font-family:var(--mono) !important;
                                font-variant-numeric:tabular-nums lining-nums; }

/* -- controls ---------------------------------------------------------- */
[data-testid="stSidebar"] .stButton button {
  font-family:var(--mono); font-size:11px; letter-spacing:.1em; text-transform:uppercase;
  border:1px solid var(--rule); background:transparent; color:var(--type);
  border-radius:2px; transition:border-color .12s ease, color .12s ease;
}
[data-testid="stSidebar"] .stButton button:hover { border-color:var(--brass); color:var(--brass); }
[data-testid="stSidebar"] .stButton button:focus-visible { outline:2px solid var(--brass); outline-offset:2px; }
label, .stSelectbox label, [data-testid="stWidgetLabel"] p {
  font-family:var(--mono) !important; font-size:10.5px !important;
  letter-spacing:.12em !important; text-transform:uppercase !important;
  color:var(--quiet) !important;
}
.note { font-family:var(--mono); font-size:10.5px; color:var(--faint); line-height:1.55; }

/* -- empty state: an invitation, not an apology ------------------------ */
.empty { border:1px dashed var(--rule); padding:2.4rem; text-align:center; }
.empty p { font-family:var(--mono); font-size:12px; color:var(--quiet); margin:0; }
.empty p + p { margin-top:.4rem; color:var(--faint); font-size:11px; }

@media (prefers-reduced-motion: reduce) {
  * { transition:none !important; animation:none !important; }
}
@media (max-width:900px) {
  .figures { flex-wrap:wrap; }
  .fig { flex:1 1 33%; border-bottom:1px solid var(--rule); }
  [data-testid="stAppViewBlockContainer"] { padding:1.4rem 1rem 3rem; }
}
</style>
"""


def figure_line(items: list[tuple[str, str, str | None, bool]]) -> str:
    """One row of figures: (label, value, note, mark-as-risk)."""
    cells = []
    for label, value, note, mark in items:
        note_html = f'<div class="fig-n">{note}</div>' if note else ""
        cells.append(
            f'<div class="fig"><div class="fig-k">{label}</div>'
            f'<div class="fig-v{" mark" if mark else ""}">{value}</div>{note_html}</div>'
        )
    return f'<div class="figures">{"".join(cells)}</div>'


def reason_rows(reasons: list[tuple[str, str, float]]) -> str:
    """Reason codes with a bar each, so contribution size is seen not read."""
    if not reasons:
        return '<p class="note">No positive contribution recorded for this alert.</p>'
    largest = max(abs(c) for _, _, c in reasons) or 1.0
    out = []
    for feature, value, contribution in reasons:
        width = max(2.0, 100.0 * abs(contribution) / largest)
        out.append(
            f'<div class="reason"><div class="reason-head">'
            f'<span class="reason-f">{feature}</span>'
            f'<span><span class="reason-v">{value}</span> &nbsp; '
            f'<span class="reason-c">+{contribution:.3f}</span></span></div>'
            f'<div class="reason-bar"><span style="width:{width:.1f}%"></span></div></div>'
        )
    return "".join(out)


def band_rows(bands: list[tuple[str, float, float]]) -> str:
    """Share of volume against share of fraud, per score band.

    The evidence for where the tier cuts belong: a thin slice of volume holds
    most of the fraud, and this is the picture of that.
    """
    rows = []
    for label, volume_share, fraud_share in bands:
        rows.append(
            f'<div class="band"><span class="band-k">{label}</span>'
            f'<span class="band-track">'
            f'<i style="width:{min(100.0, volume_share * 100):.1f}%"></i>'
            f'<b style="width:{min(100.0, fraud_share * 100):.1f}%"></b>'
            f"</span></div>"
        )
    rows.append(
        '<div class="legend"><s>&#9644; share of volume</s> &nbsp;&nbsp; '
        "<em>&#9644; share of all fraud</em></div>"
    )
    return "".join(rows)
