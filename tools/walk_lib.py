#!/usr/bin/env python3
"""The walk-through-the-year coupling's plain logic — windows, clocks, units.

Everything here is arithmetic a test can pin without a model run: how a year
splits into exchange windows, how a window's daily drainage becomes a
PFLOTRAN flux series on the walk's absolute clock, and how the window's own
days are cut out of the case's growing daily record. The model work stays
with the models — elm_wrapper for slices, the PFLOTRAN server's builders for
decks; tools/walk_job.py only wires them together.
"""
import calendar
import json
from datetime import date, timedelta
from pathlib import Path

DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

# 1 mm/day = 0.001 m/day * 365 d/y — the deck's flux unit, the same constant
# the PFLOTRAN server's own join applies to a daily series.
MM_DAY_TO_M_Y = 0.365

STATE_NAME = "walk_state.json"


def window_edges(year, months=None, days=None):
    """How the year splits: [{i, d0, d1, stop_n, stop_option}], d1 exclusive.

    d0/d1 are day-of-year offsets (0-based). Monthly windows follow the
    calendar, because ELM's nmonths stop lands on month boundaries and the
    PFLOTRAN window must cover exactly the same days. Day-count windows are
    fixed-size with the LAST taking the remainder (7-day windows: 52 of 7,
    then one of 1).

    A leap year is REFUSED, not approximated: every daily clock in this
    pipeline is /365 (the deck series, the walk's absolute times), and
    running 366 days against it would drop 31 December silently somewhere.
    """
    if calendar.isleap(int(year)):
        raise ValueError(
            f"{year} is a leap year and the walk's clock is /365 everywhere "
            f"— pick a 365-day year, or teach every daily series the 366th "
            f"day first")
    if (months is None) == (days is None):
        raise ValueError("exactly one of months= or days= chooses the window")
    out = []
    if months is not None:
        if int(months) != 1:
            raise ValueError("only months=1 is supported: ELM stops on "
                             "calendar months, and a 2-month window would "
                             "need its own day arithmetic nobody checks")
        d = 0
        for i, n in enumerate(DAYS_IN_MONTH):
            out.append({"i": i, "d0": d, "d1": d + n,
                        "stop_n": 1, "stop_option": "nmonths"})
            d += n
        return out
    days = int(days)
    if not 1 <= days <= 365:
        raise ValueError(f"days={days} is not a window")
    d = 0
    i = 0
    while d < 365:
        n = min(days, 365 - d)
        out.append({"i": i, "d0": d, "d1": d + n,
                    "stop_n": n, "stop_option": "ndays"})
        d += n
        i += 1
    return out


def flux_series(qdrai_mm_day, d0, t0_y):
    """One window's daily drainage as the deck's [(t_y, m_per_y)], absolute.

    t0_y is the walk's clock origin — the spin checkpoint's time — so day j
    of the window lands at t0_y + (d0 + j)/365, which is the time PFLOTRAN's
    continued clock will actually pass through.
    """
    return [[round(t0_y + (d0 + j) / 365.0, 8), float(v) * MM_DAY_TO_M_Y]
            for j, v in enumerate(qdrai_mm_day)]


def window_slice(dates, values, year, d0, d1):
    """The window's own days cut out of the case's full daily series.

    Refuses a hole rather than padding it: a missing or None day means the
    slice did not run or the extract dropped it, and a flux series quietly
    one day short would shift every later day of the walk by one.
    """
    start = date(int(year), 1, 1)
    expected = [str(start + timedelta(days=j)) for j in range(d0, d1)]
    idx = {str(d)[:10]: i for i, d in enumerate(dates or [])}
    missing = [d for d in expected if d not in idx]
    if missing:
        raise ValueError(
            f"the window {expected[0]}..{expected[-1]} is not fully on the "
            f"record: {len(missing)} day(s) missing, first {missing[0]} — "
            f"the record ends at {dates[-1] if dates else 'nothing'}")
    out = []
    for d in expected:
        v = values[idx[d]]
        if v is None:
            raise ValueError(f"{d}: the extract carries no value — a fill "
                             f"day cannot drive a deck")
        out.append(float(v))
    return out


def restart_date(restart_name):
    """The date an ELM restart file carries, 'YYYY-MM-DD', or None.

    A restart written at a stop is named for the midnight AFTER the last
    simulated day (7 days from 1 Jan -> ...elm.r.2010-01-08-00000.nc), so
    this is exactly 'day d1 of the walk' for a window ending at offset d1.
    """
    import re
    m = re.search(r"\.r\.(\d{4}-\d{2}-\d{2})-", str(restart_name))
    return m.group(1) if m else None


def window_end_date(year, d1):
    """The restart date a window ending at day-offset d1 must leave behind."""
    return str(date(int(year), 1, 1) + timedelta(days=int(d1)))


def read_built_cases(elm_run_dir):
    """{case_name: case_dir} from an ELM leg's build manifest.

    The manifest's top level is {"ok", "n_ok", "n_total", "cases"} (written
    by ensemble_job.py); the rows live under "cases", and a failed build's
    case_dir is falsy — skipped here, never handed on as Path(None).
    """
    p = Path(elm_run_dir) / "01_inputs" / "built_cases.json"
    rows = json.loads(p.read_text()).get("cases") or []
    return {r["case_name"]: r["case_dir"] for r in rows if r.get("case_dir")}


def load_pf_server():
    """The PFLOTRAN server repo's modules, however this machine reaches them.

    The editable pflotran-mcp install already maps `server` and its `tools`
    package into the environment; REACTION_MCP_DIR is the fallback for a
    machine without it. Returns (server, column_builder, simulation,
    site_data).
    """
    import os
    import sys
    try:
        import server as pf_server
    except ImportError:
        d = os.environ.get("REACTION_MCP_DIR")
        if not d or not Path(d).is_dir():
            raise RuntimeError(
                "cannot import the PFLOTRAN server: neither the editable "
                "pflotran-mcp install nor REACTION_MCP_DIR is available")
        sys.path.insert(0, d)
        import server as pf_server
    from tools import column_builder, simulation, site_data
    return pf_server, column_builder, simulation, site_data


def load_state(walk_dir):
    p = Path(walk_dir) / STATE_NAME
    if not p.is_file():
        return {"spun": False, "windows_done": 0, "checkpoints": {},
                "log": []}
    return json.loads(p.read_text())


def save_state(walk_dir, state):
    p = Path(walk_dir) / STATE_NAME
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(p)
    return p
