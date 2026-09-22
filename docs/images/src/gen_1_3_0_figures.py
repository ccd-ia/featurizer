"""Generate the SVG figures of the 1.3.0 docs pages, in the style of
docs/images/walkthrough-erd.svg (teal entities, amber as_of_dates, system font).

    python3 docs/images/src/gen_1_3_0_figures.py      # from the repo root

Writes asof-cut-timeline.svg, cte-flow.svg, row-order.svg,
paired-cohort-grid.svg and events-of-a-date.svg into docs/images/. Every figure
sits on an opaque white panel so it reads the same under the site's light and
dark themes and inside GitHub's README. The dates and values in the figures are
the ones the pages quote; change them here and the page together.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent
FONT = "-apple-system,'Segoe UI',Roboto,sans-serif"
MONO = "Menlo,Consolas,monospace"
TEAL, TEAL_DARK, TEAL_LIGHT = "#0f766e", "#134e4a", "#ccfbf1"
AMBER, AMBER_LIGHT = "#b45309", "#fef3c7"
INK, MUTED, GRID, PANEL = "#1a2332", "#5b6472", "#d7dbe0", "#ffffff"
RED = "#b91c1c"


def svg(width: int, height: int, label: str, body: str) -> str:
    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="{label}" font-family="{FONT}">\n'
        f'  <rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="12" '
        f'fill="{PANEL}" stroke="{GRID}"/>\n{body}</svg>\n'
    )


def text(
    x,
    y,
    s,
    *,
    size=12.5,
    fill=INK,
    anchor="start",
    weight=None,
    mono=False,
    italic=False,
):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    attrs = [f'x="{x}"', f'y="{y}"', f'font-size="{size}"', f'fill="{fill}"']
    if anchor != "start":
        attrs.append(f'text-anchor="{anchor}"')
    if weight:
        attrs.append(f'font-weight="{weight}"')
    if mono:
        attrs.append(f'font-family="{MONO}"')
    if italic:
        attrs.append('font-style="italic"')
    return f"  <text {' '.join(attrs)}>{s}</text>\n"


def line(x1, y1, x2, y2, *, stroke=MUTED, width=1.4, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"{d}/>\n'


def arrow_marker(name="arrow", color=TEAL):
    return (
        f'  <defs><marker id="{name}" markerWidth="10" markerHeight="10" refX="9" refY="5" '
        f'orient="auto" markerUnits="userSpaceOnUse"><path d="M0,1 L9,5 L0,9 z" fill="{color}"/></marker></defs>\n'
    )


def arrow(x1, y1, x2, y2, *, marker="arrow", stroke=TEAL, width=1.6):
    return (
        f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
        f'stroke-width="{width}" marker-end="url(#{marker})"/>\n'
    )


def box(x, y, w, h, title, lines, *, head=TEAL, fill="#ffffff", title_size=13):
    out = f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{head}" stroke-width="1.5"/>\n'
    out += f'  <rect x="{x}" y="{y}" width="{w}" height="26" rx="9" fill="{head}"/>\n'
    out += f'  <rect x="{x}" y="{y + 18}" width="{w}" height="8" fill="{head}"/>\n'
    out += text(
        x + w / 2,
        y + 18,
        title,
        size=title_size,
        fill="#ffffff",
        anchor="middle",
        weight="700",
        mono=True,
    )
    for i, s in enumerate(lines):
        out += text(x + 10, y + 46 + i * 17, s, size=11.5, mono=True)
    return out


# --------------------------------------------------------------- timelines


def axis(
    x0: int, x1: int, y: int, start: date, end: date, months: list[date]
) -> tuple[str, callable]:
    """A horizontal time axis; returns its SVG and a date→x mapping."""
    span = (end - start).days

    def at(d: date) -> float:
        return round(x0 + (x1 - x0) * (d - start).days / span, 1)

    out = line(x0, y, x1, y, stroke=INK, width=1.4)
    for m in months:
        x = at(m)
        out += line(x, y - 4, x, y + 4, stroke=INK, width=1.2)
        out += text(
            x, y + 20, m.strftime("%b %-d"), size=11, fill=MUTED, anchor="middle"
        )
    return out, at


def dot(x, y, *, read=True, r=6):
    if read:
        return f'  <circle cx="{x}" cy="{y}" r="{r}" fill="{TEAL}"/>\n'
    return f'  <circle cx="{x}" cy="{y}" r="{r}" fill="#ffffff" stroke="{RED}" stroke-width="1.8" stroke-dasharray="3 2"/>\n'


def square(x, y, *, emitted=True, s=13):
    if emitted:
        return f'  <rect x="{x - s / 2}" y="{y - s / 2}" width="{s}" height="{s}" rx="2" fill="{AMBER}"/>\n'
    return f'  <rect x="{x - s / 2}" y="{y - s / 2}" width="{s}" height="{s}" rx="2" fill="#ffffff" stroke="{RED}" stroke-width="1.8" stroke-dasharray="3 2"/>\n'


def asof_cut_timeline() -> str:
    W, H = 880, 400
    body = ""
    body += text(
        24,
        30,
        "One as-of date. What each read can see, and which target rows exist.",
        size=14,
        weight="700",
    )
    ax, at = axis(
        70,
        840,
        330,
        date(2024, 1, 1),
        date(2024, 9, 1),
        [date(2024, m, 1) for m in range(1, 10)],
    )
    asof = date(2024, 6, 1)
    xa = at(asof)
    # interval window
    w0 = at(date(2024, 5, 2))
    body += f'  <rect x="{w0}" y="70" width="{xa - w0}" height="180" fill="{AMBER_LIGHT}" opacity="0.7"/>\n'
    body += text(
        (w0 + xa) / 2,
        86,
        "interval P30D",
        size=11,
        fill=AMBER,
        anchor="middle",
        mono=True,
    )
    # as-of line
    body += line(xa, 56, xa, 330, stroke=AMBER, width=1.8, dash="6 4")
    body += text(
        xa + 6,
        62,
        "aod.as_of_date = 2024-06-01",
        size=11.5,
        fill=AMBER,
        weight="700",
        mono=True,
    )
    body += ax

    rows = [
        (
            140,
            "customer 1",
            date(2024, 1, 10),
            [date(2024, 2, 20), date(2024, 5, 12), date(2024, 7, 3)],
        ),
        (230, "customer 2", date(2024, 7, 8), [date(2024, 7, 28), date(2024, 8, 14)]),
    ]
    for y, name, signed, orders in rows:
        body += line(70, y, 840, y, stroke=GRID, width=1)
        body += text(
            70,
            y - 14,
            f"{name} · signed_up {signed.isoformat()}",
            size=11.5,
            weight="700",
        )
        body += square(at(signed), y, emitted=signed <= asof)
        for o in orders:
            body += dot(at(o), y, read=o <= asof)
    # annotations
    body += text(
        at(date(2024, 1, 10)),
        172,
        "row emitted",
        size=10.5,
        fill=AMBER,
        anchor="middle",
    )
    body += text(
        at(date(2024, 2, 20)), 172, "read", size=10.5, fill=TEAL, anchor="middle"
    )
    body += text(
        at(date(2024, 5, 12)),
        172,
        "read, in window",
        size=10.5,
        fill=TEAL,
        anchor="middle",
    )
    body += text(
        at(date(2024, 7, 3)), 172, "not read", size=10.5, fill=RED, anchor="middle"
    )
    body += text(
        at(date(2024, 7, 8)),
        262,
        "row not emitted (ADR-0017)",
        size=10.5,
        fill=RED,
        anchor="middle",
    )
    body += text(
        at(date(2024, 8, 14)), 262, "not read", size=10.5, fill=RED, anchor="middle"
    )
    # result strip
    body += text(
        70,
        296,
        "customer 1:  COUNT(orders.order_id) = 2 · |interval=P30D = 1",
        size=11.5,
        mono=True,
    )
    body += text(70, 312, "customer 2:  no row under this date", size=11.5, mono=True)
    # legend
    ly = 372
    body += dot(80, ly, read=True, r=5) + text(
        92, ly + 4, "read: temporal_ix <= aod.as_of_date", size=11, fill=MUTED
    )
    body += dot(330, ly, read=False, r=5) + text(
        342, ly + 4, "dated after the as-of date: never read", size=11, fill=MUTED
    )
    body += square(600, ly, emitted=True, s=11) + text(
        612, ly + 4, "target row emitted", size=11, fill=MUTED
    )
    body += square(745, ly, emitted=False, s=11) + text(
        757, ly + 4, "not emitted", size=11, fill=MUTED
    )
    return svg(
        W,
        H,
        "Timeline of two customers and their orders around one as-of date: orders dated after the date are never read, and a customer who signs up after the date has no row under it",
        body,
    )


def cte_flow() -> str:
    W, H = 880, 420
    body = arrow_marker()
    body += text(
        24,
        30,
        "The query for one config: one lateral per as-of date, six CTEs",
        size=14,
        weight="700",
    )
    # spine
    body += f'  <rect x="24" y="48" width="200" height="54" rx="9" fill="{AMBER_LIGHT}" stroke="{AMBER}" stroke-width="1.5"/>\n'
    body += text(
        124,
        70,
        "as_of_dates as aod",
        size=12.5,
        fill=AMBER,
        anchor="middle",
        weight="700",
        mono=True,
    )
    body += text(
        124, 88, "one row per as-of date", size=11, fill=AMBER, anchor="middle"
    )
    # lateral container
    body += f'  <rect x="24" y="118" width="832" height="280" rx="12" fill="none" stroke="{AMBER}" stroke-width="1.5" stroke-dasharray="7 5"/>\n'
    body += text(
        40,
        140,
        "cross join lateral ( with … select * from customers_transform ) as t   — evaluated once per aod.as_of_date",
        size=11.5,
        fill=AMBER,
        weight="700",
        mono=True,
    )
    body += arrow(124, 102, 124, 118, stroke=AMBER)
    # top row: child chain
    y1 = 158
    body += box(
        40,
        y1,
        250,
        96,
        "orders_synth",
        ["read orders", "where ordered_at <= aod.as_of_date", "identifiers + amount"],
    )
    body += box(
        320,
        y1,
        250,
        96,
        "orders_transform",
        ["lag(amount) over (", "  partition by order_id", "  order by ordered_at, …)"],
    )
    body += box(
        600,
        y1,
        250,
        96,
        "orders_aggs_for_customers",
        [
            "group by customer_id",
            "count(order_id), sum(amount)",
            "filter (where daterange(…))",
        ],
        title_size=11.5,
    )
    body += arrow(290, y1 + 48, 320, y1 + 48)
    body += arrow(570, y1 + 48, 600, y1 + 48)
    # bottom row: target chain
    y2 = 288
    body += box(
        600,
        y2,
        250,
        96,
        "customers_synth",
        [
            "read customers",
            "left join the aggregates",
            "where signed_up <= aod.as_of_date",
        ],
        head=TEAL_DARK,
        fill=TEAL_LIGHT,
    )
    body += box(
        320,
        y2,
        250,
        96,
        "customers_transform",
        [
            "lag(…) over (partition by",
            "  customer_id order by …)",
            "one-hots · the output columns",
        ],
        head=TEAL_DARK,
        fill=TEAL_LIGHT,
    )
    body += box(
        40,
        y2,
        250,
        96,
        "select * from customers_transform",
        [
            "as t: (as_of_date, customer_id, …)",
            "the executor indexes the frame",
            "by (as_of_date, customer_id)",
        ],
        head=TEAL_DARK,
        fill=TEAL_LIGHT,
        title_size=10.5,
    )
    body += arrow(725, y1 + 96, 725, y2, stroke=TEAL_DARK)
    body += arrow(600, y2 + 48, 570, y2 + 48, stroke=TEAL_DARK)
    body += arrow(320, y2 + 48, 290, y2 + 48, stroke=TEAL_DARK)
    body += text(
        735, y1 + 96 + 20, "left join on customer_id", size=10.5, fill=TEAL_DARK
    )
    return svg(
        W,
        H,
        "Flow of the six CTEs inside the lateral: the child is read and cut, transformed, aggregated; the target is read and cut, joined with the aggregates, transformed into the output",
        body,
    )


def row_order() -> str:
    W, H = 880, 430
    body = arrow_marker("walk", TEAL_DARK)
    body += text(
        24,
        30,
        "A window walks one partition in the entity's row order",
        size=14,
        weight="700",
    )
    cards = [  # (ordered_at, order_id, amount)
        ("2024-01-10", 3, 10),
        ("2024-03-05", 5, 7),
        ("2024-03-05", 6, 12),
        ("2024-04-20", 8, 9),
    ]

    def strip(y, title, order, lags, note_color):
        out = text(40, y - 34, title, size=12, weight="700")
        cw, gap, x0 = 170, 24, 40
        for i, idx in enumerate(order):
            d, oid, amount = cards[idx]
            x = x0 + i * (cw + gap)
            out += f'  <rect x="{x}" y="{y}" width="{cw}" height="72" rx="8" fill="#ffffff" stroke="{TEAL}" stroke-width="1.4"/>\n'
            out += (
                f'  <circle cx="{x + 16}" cy="{y + 16}" r="10" fill="{TEAL_DARK}"/>\n'
            )
            out += text(
                x + 16,
                y + 20,
                str(i + 1),
                size=10.5,
                fill="#ffffff",
                anchor="middle",
                weight="700",
            )
            out += text(x + 34, y + 20, f"ordered_at {d}", size=10.5, mono=True)
            out += text(x + 12, y + 40, f"order_id {oid}", size=10.5, mono=True)
            out += text(x + 12, y + 58, f"amount {amount}", size=10.5, mono=True)
            out += text(
                x + cw / 2,
                y + 92,
                f"lag_1(amount) = {lags[i]}",
                size=11,
                fill=note_color,
                anchor="middle",
                mono=True,
            )
            if i < len(order) - 1:
                out += arrow(
                    x + cw + 3,
                    y + 36,
                    x + cw + gap - 3,
                    y + 36,
                    marker="walk",
                    stroke=TEAL_DARK,
                    width=1.4,
                )
        return out

    # 1.3.0: ties resolved by order_id, then amount
    body += strip(
        100,
        "1.3.0 — order by ordered_at, then the other identifiers (order_id), then the declared variables (amount)",
        [0, 1, 2, 3],
        ["NULL", "10", "7", "12"],
        TEAL_DARK,
    )
    x_tie0, x_tie1 = 40 + 1 * 194, 40 + 2 * 194 + 170
    body += f'  <path d="M{x_tie0},94 L{x_tie0},86 L{x_tie1},86 L{x_tie1},94" fill="none" stroke="{AMBER}" stroke-width="1.5"/>\n'
    body += text(
        (x_tie0 + x_tie1) / 2,
        82,
        "same ordered_at: order_id decides",
        size=10.5,
        fill=AMBER,
        anchor="middle",
    )
    # before: physical order, here the table stored order 6 before order 5
    body += strip(
        272,
        "before 1.3.0 — order by ordered_at only; rows 2 and 3 as the table stored them (here 6 before 5)",
        [0, 2, 1, 3],
        ["NULL", "10", "12", "7"],
        RED,
    )
    body += text(
        40,
        410,
        "The same four rows, two answers for order 8's lag_1: 12 above, 7 below. A reload or a cluster could swap them.",
        size=11.5,
        fill=MUTED,
    )
    return svg(
        W,
        H,
        "Four orders of one customer laid out in the order a window walks them, once under the 1.3.0 row order and once under the physical order, with the lag_1 value each row receives",
        body,
    )


def paired_cohort_grid() -> str:
    W, H = 880, 330
    body = ""
    body += text(
        24, 30, "Which (date, entity) cells the query computes", size=14, weight="700"
    )
    dates = ["2024-02-01", "2024-03-01", "2024-04-01"]
    pairs = {
        ("2024-02-01", 1),
        ("2024-02-01", 2),
        ("2024-03-01", 2),
        ("2024-03-01", 3),
        ("2024-03-01", 5),
        ("2024-04-01", 1),
        ("2024-04-01", 4),
    }

    def grid(x0, title, sub, keep):
        out = text(x0, 62, title, size=12.5, weight="700")
        out += text(x0, 80, sub, size=11, fill=MUTED)
        cw, ch = 46, 40
        for j in range(6):
            out += text(
                x0 + 110 + j * cw + cw / 2,
                108,
                str(j + 1),
                size=11,
                fill=MUTED,
                anchor="middle",
                mono=True,
            )
        out += text(
            x0 + 110 + 3 * cw, 94, "customer_id", size=10.5, fill=MUTED, anchor="middle"
        )
        for i, d in enumerate(dates):
            y = 118 + i * ch
            out += text(
                x0 + 100,
                y + ch / 2 + 4,
                d,
                size=11,
                fill=MUTED,
                anchor="end",
                mono=True,
            )
            for j in range(6):
                x = x0 + 110 + j * cw
                on = keep(d, j + 1)
                fill = TEAL_LIGHT if on else "#ffffff"
                stroke = TEAL if on else GRID
                out += f'  <rect x="{x + 3}" y="{y + 3}" width="{cw - 6}" height="{ch - 6}" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>\n'
        return out

    body += grid(
        24,
        "Dense (default)",
        "every target row under every date: 18 rows",
        lambda d, c: True,
    )
    body += grid(
        470,
        "Paired: as_of_dates: {id_column: cohort_id}",
        "one row per declared pair: 7 rows",
        lambda d, c: (d, c) in pairs,
    )
    body += text(24, 262, "as_of_dates holds 3 dates", size=11.5, mono=True)
    body += text(
        470,
        262,
        "as_of_dates holds the 7 (as_of_date, cohort_id) pairs",
        size=11.5,
        mono=True,
    )
    body += text(
        24,
        292,
        "In the paired run a child that only the target aggregates is read for the date's cohort too:",
        size=11.5,
    )
    body += text(
        24,
        310,
        "the orders of customers 3, 4, 5 and 6 are not aggregated for 2024-02-01. Values on the 7 pairs equal the dense run's.",
        size=11.5,
    )
    return svg(
        W,
        H,
        "Two grids of as-of dates by customers: the dense run fills every cell, the paired run only the declared pairs",
        body,
    )


def events_of_a_date() -> str:
    W, H = 880, 420
    body = ""
    body += text(
        24,
        30,
        "One row per event: a game scored with what was known the day before",
        size=14,
        weight="700",
    )
    ax, at = axis(
        70,
        840,
        350,
        date(2024, 2, 10),
        date(2024, 3, 25),
        [date(2024, 2, 15), date(2024, 3, 1), date(2024, 3, 15)],
    )
    game = date(2024, 3, 8)
    asof = date(2024, 3, 7)
    xg, xa = at(game), at(asof)
    body += line(xa, 56, xa, 350, stroke=AMBER, width=1.8, dash="6 4")
    body += text(
        xa - 6,
        62,
        "as_of_date = 2024-03-07",
        size=11.5,
        fill=AMBER,
        weight="700",
        mono=True,
        anchor="end",
    )
    body += text(
        xa - 6, 78, "(played_on - 1)", size=11, fill=AMBER, mono=True, anchor="end"
    )
    body += ax
    # target row: the game
    y1 = 130
    body += line(70, y1, 840, y1, stroke=GRID, width=1)
    body += text(
        70,
        y1 - 14,
        "games · target, no temporal_ix  ·  game_id 3, played_on 2024-03-08, home_id 100",
        size=11.5,
        weight="700",
    )
    body += square(xg, y1, emitted=True, s=14)
    body += text(
        xg + 12,
        y1 + 4,
        "the pair (2024-03-07, 3): this row is emitted",
        size=10.5,
        fill=AMBER,
    )
    # child rows
    y2 = 230
    body += line(70, y2, 840, y2, stroke=GRID, width=1)
    body += text(
        70,
        y2 - 14,
        "team_games · child, temporal_ix played_on  ·  team_id 100",
        size=11.5,
        weight="700",
    )
    for d, goals in [(date(2024, 2, 20), 3), (date(2024, 3, 1), 1)]:
        body += dot(at(d), y2, read=True)
        body += text(
            at(d),
            y2 + 24,
            f"goals {goals}",
            size=10.5,
            fill=TEAL,
            anchor="middle",
            mono=True,
        )
    for d, goals, note in [
        (date(2024, 3, 8), 4, "the game's own day"),
        (date(2024, 3, 15), 2, "later"),
    ]:
        body += dot(at(d), y2, read=False)
        body += text(
            at(d),
            y2 + 24,
            f"goals {goals}",
            size=10.5,
            fill=RED,
            anchor="middle",
            mono=True,
        )
        body += text(at(d) + 12, y2 - 10, note, size=10, fill=RED)
    body += text(
        70,
        300,
        "COUNT(home.played_on) = 2    MEAN(home.goals) = 2.0",
        size=11.5,
        mono=True,
    )
    body += text(
        70,
        320,
        "Paired with its own date, with a temporal_ix on the",
        size=11,
        fill=MUTED,
    )
    body += text(
        70,
        336,
        "target, the 2024-03-08 row is read too (COUNT = 3).",
        size=11,
        fill=MUTED,
    )
    body += text(
        70,
        394,
        "as_of_dates:  select played_on - 1 as as_of_date, game_id as cohort_id from games",
        size=11.5,
        fill=AMBER,
        mono=True,
    )
    return svg(
        W,
        H,
        "Timeline of one game and its home team's earlier games: paired with the day before, the game's row is emitted and only the team's rows before that day are read",
        body,
    )


for name, fn in {
    "asof-cut-timeline.svg": asof_cut_timeline,
    "cte-flow.svg": cte_flow,
    "row-order.svg": row_order,
    "paired-cohort-grid.svg": paired_cohort_grid,
    "events-of-a-date.svg": events_of_a_date,
}.items():
    (OUT / name).write_text(fn())
    print("wrote", OUT / name)
