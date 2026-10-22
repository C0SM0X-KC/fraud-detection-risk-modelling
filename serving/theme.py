"""Visual identity for the analyst console."""

from __future__ import annotations

TOKENS = {
    "stock": "#0F141C",   # page - deep slate, not black
    "band": "#151C27",    # the alternating band
    "panel": "#131A24",   # slips and panels
    "rule": "#253040",    # ruled lines
    "ink": "#A8B2C1",     # body text
    "faint": "#5F6B7C",   # secondary
    "pen": "#4F9CF9",     # the model marked it
    "stamp": "#EDF3FA",   # confirmed fraud - the brightest thing on screen
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=Azeret+Mono:wght@400;500;600&display=swap');

:root {
  --stock:#0F141C; --band:#151C27; --panel:#131A24; --rule:#253040;
  --ink:#A8B2C1; --faint:#5F6B7C; --pen:#4F9CF9; --stamp:#EDF3FA;
  --mono:'Azeret Mono',ui-monospace,monospace;
  --sans:'Archivo',system-ui,sans-serif;
}

/* Hide the framework furniture, but NOT the toolbar itself: the control that
   reopens a collapsed sidebar lives inside it, and hiding the whole bar strands
   anyone who collapses one. Only the deploy and menu actions go. */
#MainMenu, footer,
[data-testid="stToolbarActions"],
[data-testid="stHeaderActionElements"],
[data-testid="stMainMenuButton"] { display:none !important; }
header[data-testid="stHeader"] { background:transparent !important; }

/* The sidebar controls are real controls, so they get the accent and a focus
   ring rather than being left in the framework's default grey. */
button[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"] button {
  color:var(--pen) !important; background:var(--panel) !important;
  border:1px solid var(--rule) !important; border-radius:0 !important;
}
button[data-testid="stExpandSidebarButton"]:hover,
[data-testid="stSidebarCollapseButton"] button:hover { border-color:var(--pen) !important; }
button[data-testid="stExpandSidebarButton"]:focus-visible,
[data-testid="stSidebarCollapseButton"] button:focus-visible {
  outline:2px solid var(--pen) !important; outline-offset:2px;
}
[data-testid="stAppViewContainer"] { background:var(--stock); }
[data-testid="stSidebar"] { background:var(--panel); border-right:1px solid var(--rule); }
[data-testid="stAppViewBlockContainer"] { padding:2rem 2.6rem 4rem; max-width:1500px; }
[data-testid="stVerticalBlock"] { gap:0.85rem; }
html, body, [class*="css"] { font-family:var(--sans); color:var(--ink); }

/* -- document header --------------------------------------------------- */
.runline {
  font-family:var(--mono); font-size:10px; letter-spacing:.17em;
  text-transform:uppercase; color:var(--faint); margin:0 0 .45rem;
}
.masthead {
  font-family:var(--sans); font-size:27px; font-weight:700; letter-spacing:-.022em;
  color:#E6ECF4; margin:0 0 .15rem; line-height:1.05;
}
.masthead-sub { font-family:var(--mono); font-size:11px; color:var(--faint); margin:0 0 .9rem; }

.section {
  font-family:var(--mono); font-size:10px; letter-spacing:.17em;
  text-transform:uppercase; color:var(--faint);
  display:flex; align-items:center; gap:.75rem; margin:1.5rem 0 .6rem;
}
.section::after { content:""; flex:1; height:1px; background:var(--rule); }

/* -- replay position: a ruled measure, not a filled bar ---------------- */
.tape { height:6px; border:1px solid var(--rule); border-left:none; border-right:none;
        margin:.1rem 0 1.3rem; position:relative; background:repeating-linear-gradient(
          90deg, var(--rule) 0 1px, transparent 1px 40px); }
.tape > span { position:absolute; top:-1px; bottom:-1px; left:0;
               border-right:2px solid var(--pen); }

/* -- the totals line, as a statement rules it -------------------------- */
.figures { display:flex; border-top:1.5px solid var(--rule); border-bottom:1px solid var(--rule); }
.fig { flex:1; padding:.8rem 1rem; border-right:1px solid var(--rule); }
.fig:last-child { border-right:none; }
.fig-k { font-family:var(--mono); font-size:9.5px; letter-spacing:.14em;
         text-transform:uppercase; color:var(--faint); margin-bottom:.3rem; }
.fig-v { font-family:var(--mono); font-size:25px; font-weight:500;
         font-variant-numeric:tabular-nums lining-nums; color:#E6ECF4; line-height:1; }
.fig-v.pen { color:var(--pen); }
.fig-v.stamp { color:var(--stamp); }
.fig-n { font-family:var(--mono); font-size:10px; color:var(--faint); margin-top:.28rem; }

/* -- the marginal note ------------------------------------------------- */
.slip { border:1px solid var(--rule); border-left:3px solid var(--pen);
        background:var(--panel); padding:.7rem .9rem; margin-bottom:.9rem;
        font-family:var(--mono); font-size:11px; color:var(--faint); line-height:1.7; }
.slip b { color:#E6ECF4; font-weight:600; }
.slip .pen { color:var(--pen); font-weight:600; }
.slip .stamp { color:var(--stamp); font-weight:600; }

.reason { margin-bottom:.6rem; }
.reason-head { display:flex; justify-content:space-between; align-items:baseline;
               font-family:var(--mono); font-size:11px; margin-bottom:.22rem; }
.reason-f { color:var(--ink); }
.reason-v { color:var(--faint); }
.reason-c { color:var(--pen); font-variant-numeric:tabular-nums; }
.reason-bar { height:2px; background:var(--rule); }
.reason-bar > span { display:block; height:100%; background:var(--pen); }

/* -- the tally --------------------------------------------------------- */
.band-row { display:flex; align-items:center; gap:.65rem; margin-bottom:.28rem;
            font-family:var(--mono); font-size:10px; }
.band-k { width:50px; color:var(--faint); text-align:right;
          font-variant-numeric:tabular-nums; }
.band-track { flex:1; height:10px; border-bottom:1px solid var(--rule); position:relative; }
.band-track > i { position:absolute; left:0; top:1px; height:3px; background:var(--rule); }
.band-track > b { position:absolute; left:0; bottom:1px; height:3px; background:var(--pen); }
.legend { font-family:var(--mono); font-size:9.5px; color:var(--faint);
          letter-spacing:.1em; margin-top:.45rem; }
.legend em { font-style:normal; color:var(--pen); }

/* -- tables ------------------------------------------------------------ */
[data-testid="stDataFrame"] { border:1px solid var(--rule); width:100% !important; }
[data-testid="stDataFrame"] * { font-family:var(--mono) !important; font-size:11.5px !important;
                                font-variant-numeric:tabular-nums lining-nums; }
/* Column headings sit in the rule colour so the grid reads as printed. */
[data-testid="stDataFrame"] [role="columnheader"] { color:var(--faint) !important;
  text-transform:uppercase; letter-spacing:.08em; font-size:10px !important; }

/* -- controls ---------------------------------------------------------- */
[data-testid="stSidebar"] .stButton button {
  font-family:var(--mono); font-size:10.5px; letter-spacing:.12em; text-transform:uppercase;
  border:1px solid var(--rule); background:transparent; color:var(--ink);
  border-radius:0; transition:border-color .12s ease, color .12s ease;
}
[data-testid="stSidebar"] .stButton button:hover { border-color:var(--pen); color:var(--pen); }
[data-testid="stSidebar"] .stButton button:focus-visible { outline:2px solid var(--pen); outline-offset:2px; }
label, [data-testid="stWidgetLabel"] p {
  font-family:var(--mono) !important; font-size:10px !important;
  letter-spacing:.14em !important; text-transform:uppercase !important;
  color:var(--faint) !important;
}
.note { font-family:var(--mono); font-size:10px; color:var(--faint); line-height:1.7; }
.note b { color:#E6ECF4; font-weight:600; }

.empty { border:1px solid var(--rule); border-left:3px solid var(--rule);
         background:var(--panel); padding:2.2rem; }
.empty p { font-family:var(--mono); font-size:12px; color:#E6ECF4; margin:0; }
.empty p + p { margin-top:.35rem; color:var(--faint); font-size:10.5px; }

@media (prefers-reduced-motion: reduce) { * { transition:none !important; animation:none !important; } }
@media (max-width:900px) {
  .figures { flex-wrap:wrap; }
  .fig { flex:1 1 33%; border-bottom:1px solid var(--rule); }
  [data-testid="stAppViewBlockContainer"] { padding:1.3rem 1rem 3rem; }
}

/* The verdict slip: the product's answer, set larger than anything else on the
   check page. The rule on the left carries the tier, using the same pen blue
   for a hold and a dimmer rule for a challenge, so the ledger's vocabulary
   carries over. */
.verdict { border:1px solid var(--rule); border-left:4px solid var(--rule);
  background:var(--panel); padding:1.1rem 1.2rem; margin-bottom:1rem; }
.verdict.high { border-left-color:var(--pen); }
.verdict.medium { border-left-color:#8FA6C4; }
.verdict-k { font-family:var(--mono); font-size:10px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--faint); }
.verdict-v { font-family:var(--sans); font-size:28px; font-weight:700;
  color:#E6ECF4; line-height:1.15; margin:.15rem 0 .35rem; }
.verdict.high .verdict-v { color:var(--pen); }
.verdict-line { font-family:var(--mono); font-size:12px; color:var(--ink); line-height:1.8; }
.verdict-line b { color:#E6ECF4; font-weight:600; }
.scale { position:relative; height:6px; background:var(--rule); margin:.7rem 0 .25rem; }
.scale > span { position:absolute; top:-3px; width:2px; height:12px; background:var(--stamp); }
.scale > i { position:absolute; top:0; height:100%; background:#2A3546; }
.scale-k { display:flex; justify-content:space-between; font-family:var(--mono);
  font-size:9px; color:var(--faint); }
.why { margin-bottom:.7rem; }
.why-t { font-family:var(--sans); font-size:13px; color:#DCE3EC; line-height:1.4; }
.why-m { font-family:var(--mono); font-size:10px; color:var(--faint); margin:.1rem 0 .25rem; }
.why-bar { height:2px; background:var(--rule); }
.why-bar > span { display:block; height:100%; background:var(--pen); }
.why.against .why-bar > span { background:#5F6B7C; }
.why.against .why-t { color:var(--ink); }
</style>
"""


def figure_line(items: list[tuple[str, str, str | None, str]]) -> str:
    """The totals line: (label, value, note, mark) where mark is ''|'pen'|'stamp'."""
    cells = []
    for label, value, note, mark in items:
        note_html = f'<div class="fig-n">{note}</div>' if note else ""
        cells.append(
            f'<div class="fig"><div class="fig-k">{label}</div>'
            f'<div class="fig-v {mark}">{value}</div>{note_html}</div>'
        )
    return f'<div class="figures">{"".join(cells)}</div>'


def reason_rows(reasons: list[tuple[str, str, float]]) -> str:
    """Reason codes with a rule each, so contribution size is seen not read."""
    if not reasons:
        return '<p class="note">No positive contribution recorded for this entry.</p>'
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
    """Share of volume against share of fraud, per score band."""
    rows = []
    for label, volume_share, fraud_share in bands:
        rows.append(
            f'<div class="band-row"><span class="band-k">{label}</span>'
            f'<span class="band-track">'
            f'<i style="width:{min(100.0, volume_share * 100):.1f}%"></i>'
            f'<b style="width:{min(100.0, fraud_share * 100):.1f}%"></b>'
            f"</span></div>"
        )
    rows.append(
        '<div class="legend">&#9644; share of volume &nbsp;&nbsp; '
        "<em>&#9644; share of all fraud</em></div>"
    )
    return "".join(rows)


def ledger_style(frame, tiers, fraud):
    """Green-bar banding, with the margin marked as a reviewer would mark it."""
    held = list(tiers)
    confirmed = list(fraud)

    def row_band(row):
        shade = TOKENS["band"] if row.name % 2 else TOKENS["stock"]
        return [f"background-color:{shade};"] * len(row)

    def gutter(column):
        return [
            f'color:{TOKENS["pen"]};font-weight:600;' if t == "high"
            else f'color:{TOKENS["rule"]};' for t in held
        ]

    def stamp(column):
        return [
            f'color:{TOKENS["stamp"]};font-weight:600;' if f
            else f'color:{TOKENS["rule"]};' for f in confirmed
        ]

    styler = frame.style.apply(row_band, axis=1)
    if "" in frame.columns:
        styler = styler.apply(gutter, subset=[""])
    if "Outcome" in frame.columns:
        styler = styler.apply(stamp, subset=["Outcome"])
    return styler


def verdict_card(decision: dict, medium_lo: int, high_lo: int) -> str:
    """The product's answer: verdict, score on its scale, and what to do."""
    tier = decision["tier"]
    score = int(decision["risk_score"])
    timings = decision.get("timings_ms", {})
    flag = "yes" if decision.get("is_fraud_prediction") else "no"
    return (
        f'<div class="verdict {tier}">'
        f'<div class="verdict-k">Transaction {decision["transaction_id"]} &middot; '
        f'{decision["model"]}</div>'
        f'<div class="verdict-v">{decision["verdict"]}</div>'
        f'<div class="verdict-line">Risk score <b>{score}</b> / 100 &nbsp;&middot;&nbsp; '
        f'probability <b>{decision["probability"]:.2%}</b> &nbsp;&middot;&nbsp; '
        f'action <b>{decision["action"]}</b></div>'
        f'<div class="scale">'
        f'<i style="left:{medium_lo}%;width:{high_lo - medium_lo}%"></i>'
        f'<span style="left:calc({min(score, 99)}% - 1px)"></span></div>'
        f'<div class="scale-k"><span>0 approve</span><span>{medium_lo} challenge</span>'
        f'<span>{high_lo} hold</span><span>100</span></div>'
        f'<div class="verdict-line" style="margin-top:.6rem">Cost-sensitive fraud flag '
        f'<b>{flag}</b> (break-even probability {decision["break_even_probability"]:.2%} '
        f'for this amount) &nbsp;&middot;&nbsp; decided in '
        f'<b>{timings.get("total_ms", float("nan")):.1f} ms</b></div>'
        f'</div>'
    )


def plain_reasons(factors: list[dict], against: bool = False) -> str:
    """Reason codes as sentences, each with a rule sized by its contribution."""
    if not factors:
        return '<p class="note">None.</p>'
    largest = max(abs(f["contribution"]) for f in factors) or 1.0
    out = []
    for f in factors:
        width = max(2.0, 100.0 * abs(f["contribution"]) / largest)
        sign = "+" if f["contribution"] > 0 else "−"
        out.append(
            f'<div class="why{" against" if against else ""}">'
            f'<div class="why-t">{f["text"]}</div>'
            f'<div class="why-m">{f["group"]} &middot; {f["feature"]} &middot; '
            f'{sign}{abs(f["contribution"]):.3f} log-odds</div>'
            f'<div class="why-bar"><span style="width:{width:.1f}%"></span></div></div>'
        )
    return "".join(out)


def group_rows(groups: list[dict], limit: int = 6) -> str:
    """Net evidence by family: which kinds of signal decided the score."""
    shown = [g for g in groups if abs(g["contribution"]) > 1e-4][:limit]
    if not shown:
        return '<p class="note">No net evidence.</p>'
    largest = max(abs(g["contribution"]) for g in shown) or 1.0
    out = []
    for g in shown:
        width = max(2.0, 100.0 * abs(g["contribution"]) / largest)
        against = g["contribution"] < 0
        sign = "−" if against else "+"
        out.append(
            f'<div class="why{" against" if against else ""}">'
            f'<div class="why-m" style="color:var(--ink)">{g["group"]} &nbsp; '
            f'{sign}{abs(g["contribution"]):.3f}</div>'
            f'<div class="why-bar"><span style="width:{width:.1f}%"></span></div></div>'
        )
    return "".join(out)
