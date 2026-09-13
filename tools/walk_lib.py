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


# ── the forward flux and the lateral sink ───────────────────────────────
# docs/coupling/lateral_sink_design.md, sections 3, 5 and 6. Every constant
# and sign below is the design's; nothing here reads a model.

FORWARD_VARS = ("QDRAI", "QCHARGE")
NEGATIVE_POLICIES = ("clip", "pass")

# Water at the sink face (design, section 3).
MU_PA_S = 1.002e-3                   # dynamic viscosity
RHO_KG_M3 = 998.0                    # density
G_M_S2 = 9.81
SECONDS_PER_DAY = 86400.0
# PFLOTRAN's -mas.dat rates are kg per year; kg/y to mm/day divides by this.
DAYS_PER_YEAR_MAS = 365.25
# The column's plan area is 1 m^2 (dx = dy = 1 in the deck builder), so one
# kilogram of water over it is one millimetre.
KG_PER_MM = 1.0
SINK_COUPLER = "lateral_sink"
# ELM's own aquifer specific yield, the cap on the default Sy
# (set_water_table.py: 200 mm/m).
SY_CAP = 0.2

# Every per-column key of the sink contract (design, section 7).
SINK_CONTRACT_KEYS = (
    "sink_datum_source", "sink_datum_m", "sink_datum_applied_m",
    "sink_datum_clipped_to_domain", "sink_datum_raised_to_band", "sink_band_m", "sink_sy",
    "sink_sy_source", "sink_tau_days", "sink_tau_source",
    "sink_conductance_m", "sink_implied_tau_days", "hand_m", "hand_path_km",
    "hand_stream_area_km2", "hand_threshold_km2", "std_elev_m",
    "conus2_wtd_m")
# The ones the datum module computes; walk_setup derives the rest.
DATUM_MODULE_KEYS = ("sink_datum_source", "sink_datum_m", "hand_m",
                     "hand_path_km", "hand_stream_area_km2",
                     "hand_threshold_km2", "std_elev_m", "conus2_wtd_m")


def forward_values(data, forward_var):
    """The extract's daily series for the forward variable, [] when none.

    extract_column leaves a variable OUT of data["variables"] both when the
    history never carried it and when every value was fill (its "absent"
    and "empty" findings); either way there is nothing to drive a deck
    with, and the caller records that rather than tracing back.
    """
    v = ((data or {}).get("variables") or {}).get(forward_var) or {}
    return v.get("values") or []


def no_forward_reason(forward_var, meta=None):
    """The departure sentence for a column whose forward series is empty."""
    m = meta or {}
    why = ""
    if forward_var in (m.get("absent_variables") or []):
        why = " (not in the history output)"
    elif forward_var in (m.get("empty_variables") or []):
        why = " (present, but every value is fill)"
    return f"ELM wrote no {forward_var} for this column{why}"


def apply_negative_policy(values, policy):
    """One window's daily flux after the negative-day policy: (values, clipped_mm).

    A negative forward flux (QCHARGE on a day the aquifer feeds the soil)
    would extract water at an unsaturated top face, double-counting the
    capillary rise ELM already resolved, and works against the solver.
    "clip" zeroes those days and returns how much was removed, in mm over
    the window (mm/day times one day each); "pass" hands the series on
    unchanged and reports 0, for the experiment that wants it.
    """
    if policy not in NEGATIVE_POLICIES:
        raise ValueError(f"negative_forward={policy!r} is not one of "
                         f"{list(NEGATIVE_POLICIES)}")
    out, clipped = [], 0.0
    for j, v in enumerate(values):
        if v is None:
            raise ValueError(f"day {j}: the series carries no value, which "
                             f"no policy can clip or pass")
        v = float(v)
        if policy == "clip" and v < 0.0:
            clipped -= v
            v = 0.0
        out.append(v)
    return out, round(clipped, 6)


def sink_conductance_m(sy, tau_days, band_m):
    """C = Sy * mu / (tau * rho * g * B), metres (design, section 3).

    The sink is a Dirichlet-conductance face over a band B of side faces;
    for a water table h above the datum it leaks Q = C (rho g / mu) B h, a
    linear reservoir. Matching the observed recession Q = Sy h / tau gives
    this C. Naches (tau 30 d, Sy 0.10, B 1 m): 3.95e-15 m.
    """
    for name, v in (("sink_sy", sy), ("sink_tau_days", tau_days),
                    ("sink_band_m", band_m)):
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError(f"{name}={v!r} must be a positive number")
    if sy > 1.0:
        raise ValueError(f"sink_sy={sy} is not a specific yield (a fraction "
                         f"of 1)")
    tau_s = float(tau_days) * SECONDS_PER_DAY
    return float(sy) * MU_PA_S / (tau_s * RHO_KG_M3 * G_M_S2 * float(band_m))


def sink_outflow_mm_day(conductance_m, band_m, head_m):
    """What the sink leaks at a water table head_m above the datum, mm/day.

    Q = C (rho g / mu) B h in m^3/s per 1 m^2 of plan area, for h >= B
    and a saturated band; the design's check that one metre of head at
    the Naches conductance gives Sy * 1000 / tau = 3.33 mm/day.
    """
    q_m_s = (float(conductance_m) * RHO_KG_M3 * G_M_S2 / MU_PA_S
             * float(band_m) * float(head_m))
    return q_m_s * 1000.0 * SECONDS_PER_DAY


def datum_in_domain(datum_m, depth_m, band_m):
    """(sink_datum_applied_m, sink_datum_clipped_to_domain,
    sink_datum_raised_to_band) for a column H deep.

    The band [H - datum, H - datum + B] must lie inside the column, so the
    datum is held between B (the band's top at the surface) and H - B (its
    bottom at the base). A column shorter than its height above the stream
    drains at its base, and that clamp is recorded: this is where the
    parked domain-depth rule shows. A datum shallower than the band (CONUS2
    puts the steady water table at the surface in valley cells; job 778434
    met nine Naches columns at 0 to 0.05 m) would put the band's top above
    the surface and thin it below the B the conductance was mapped with,
    so it is raised to B, and that clamp is recorded too.
    """
    if not isinstance(datum_m, (int, float)) or datum_m < 0:
        raise ValueError(f"sink_datum_m={datum_m!r} is not a depth below "
                         f"the surface")
    deepest = float(depth_m) - float(band_m)
    if deepest <= 0:
        raise ValueError(f"depth_m={depth_m} leaves no room for a "
                         f"{band_m} m band")
    if datum_m > deepest:
        return round(deepest, 4), True, False
    if datum_m < band_m:
        return round(min(float(band_m), deepest), 4), False, True
    return round(float(datum_m), 4), False, False


def layer_at_depth(profile, depth_m):
    """The CONUS2 layer whose span holds depth_m (top inclusive), or None."""
    for L in (profile or {}).get("layers") or []:
        top, bot = L.get("depth_top_m"), L.get("depth_bot_m")
        if top is None or bot is None:
            continue
        if float(top) <= float(depth_m) < float(bot):
            return L
    return None


def implied_tau_days(sy, k_m_h, path_m, head_m):
    """The Dupuit timescale the column's own material implies, days, or None.

    A hillslope draining its own conductivity K over the path length L to
    the stream carries Q = K h^2 / L^2 per unit plan area; matching
    Q = Sy h / tau gives tau = Sy L^2 / (K h) at the head h the column
    stands at. Recorded for inspection only (design, section 3): the
    recession the gauges show is the timescale used. None when an input
    is missing or the water table stands below the datum (h <= 0).
    """
    vals = (sy, k_m_h, path_m, head_m)
    if any(not isinstance(v, (int, float)) for v in vals):
        return None
    if k_m_h <= 0 or path_m <= 0 or head_m <= 0:
        return None
    k_m_day = float(k_m_h) * 24.0
    return round(float(sy) * float(path_m) ** 2 / (k_m_day * float(head_m)), 3)


def read_mass_balance(path, coupler=SINK_COUPLER):
    """A PFLOTRAN -mas.dat's columns for one named coupler, or None.

    The file is one quoted, comma-separated header ("Time [y]",
    "<coupler> Water Mass [kg]", "<coupler> Water Mass [kg/y]", ...) over
    whitespace-separated rows. Returns {times_y, cumulative_kg, rate_kg_y}
    AS WRITTEN (outflow negative). None when the file or the coupler's
    columns are absent: an absence, never a zero.
    """
    import re
    p = Path(path)
    if not p.is_file():
        return None
    lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
    if not lines:
        return None
    names = re.findall(r'"([^"]*)"', lines[0])
    want = {"times_y": "Time [y]",
            "cumulative_kg": f"{coupler} Water Mass [kg]",
            "rate_kg_y": f"{coupler} Water Mass [kg/y]"}
    if any(n not in names for n in want.values()):
        return None
    idx = {k: names.index(n) for k, n in want.items()}
    out = {k: [] for k in want}
    for i, ln in enumerate(lines[1:], 2):
        f = ln.split()
        if len(f) < len(names):
            raise ValueError(f"{p} line {i}: {len(f)} fields against "
                             f"{len(names)} named in the header")
        for k, j in idx.items():
            out[k].append(float(f[j]))
    return out


def lateral_outflow_block(path, coupler=SINK_COUPLER):
    """The design's lateral_outflow block from a window's -mas.dat, or None.

    Sign flipped once, here, so positive = leaving the column; kg == mm
    over the 1 m^2 column; kg/y to mm/day divides by 365.25. The cumulative
    coupler mass is not checkpointed and every window runs in its own case
    directory, so the file's last cumulative value is the window's total.
    """
    mb = read_mass_balance(path, coupler)
    if mb is None:
        return None
    cum = mb["cumulative_kg"]
    return {
        "times_y": mb["times_y"],
        "cumulative_kg": cum,
        "rate_kg_y": mb["rate_kg_y"],
        "outflow_mm_day": [-r / KG_PER_MM / DAYS_PER_YEAR_MAS
                           for r in mb["rate_kg_y"]],
        "window_total_mm": (round(-cum[-1] / KG_PER_MM, 6) if cum else None),
        "units": {"times_y": "y", "cumulative_kg": "kg (as written, "
                  "negative = out)", "rate_kg_y": "kg/y (as written)",
                  "outflow_mm_day": "mm/day, positive = leaving",
                  "window_total_mm": "mm, positive = leaving"},
    }


# ── the return leg and ELM's own water balance (2026-09-13) ──────────────
# Measured on the first Naches sink walk: col_17 received 1191 mm of
# precipitation and exported 3262 mm of drainage plus 2330 mm of recharge
# with its storage unchanged over the year. Within each window ELM's storage
# fell by 150 to 370 mm and jumped back at the window boundary, where the
# return leg stamps PFLOTRAN's near-saturated soil profile into ELM's soil.
# ELM drained that stamped water as recharge and drainage, the stamp refilled
# it, the sink removed it: a pump. So the walk now records ELM's balance per
# window and the storage jump at every stamp, and the return leg is a dial.
RETURN_LEGS = ("wt+profile", "wt")
BALANCE_VARS = ("RAIN", "SNOW", "QOVER", "QDRAI", "QCHARGE",
                "QVEGE", "QVEGT", "QSOIL", "TWS",
                # carried by cases built after 2026-09-13; absent before
                "QRUNOFF", "QRGWL", "QSNWCPICE")


def elm_balance(series, windows):
    """ELM's water balance per window and for the year, from its daily series.

    series: {VAR: [daily values]} in mm/day (TWS in mm), all the same length.
    windows: [(d0, d1), ...] day indices, d1 exclusive, in order.

    Per window: P (RAIN + SNOW), ET (QVEGE + QVEGT + QSOIL), QOVER, QDRAI,
    QCHARGE, dTWS from the window's first day to its last, and the residual
    P - ET - QOVER - QDRAI - dTWS, which a water-conserving model keeps at
    zero (QCHARGE is soil to aquifer, internal to TWS). At each boundary the
    jump TWS[first day of the next window] - TWS[last day of this one] is the
    water the stamp added (one day of fluxes rides on it). A negative annual
    residual with positive jumps summing to about its size is the signature
    of water created by the return leg. None where a series is missing.
    """
    # SIGN. A NEGATIVE residual is water that appeared (the pump). A
    # POSITIVE residual is water that left through a flux this history does
    # not carry: older cases record only QOVER and QDRAI, so lake or wetland
    # runoff (QRGWL) and capped snow (QSNWCPICE) show as a positive residual;
    # when QRUNOFF is on the history it is used in their place. A positive
    # residual matching a negative stamp jump is the sink's removal mirrored
    # into ELM's aquifer, which is the exchange working.
    need = ("RAIN", "SNOW", "QOVER", "QDRAI", "QVEGE", "QVEGT", "QSOIL", "TWS")
    if any(not (series or {}).get(v) for v in need):
        return None
    n = min(len(series[v]) for v in need)
    P = [series["RAIN"][i] + series["SNOW"][i] for i in range(n)]
    ET = [series["QVEGE"][i] + series["QVEGT"][i] + series["QSOIL"][i]
          for i in range(n)]
    QO, QD, TWS = series["QOVER"][:n], series["QDRAI"][:n], series["TWS"][:n]
    QC = (series.get("QCHARGE") or [0.0] * n)[:n]
    # QRUNOFF is the whole runoff (QOVER + QDRAI + QRGWL + capped snow); when
    # the history carries it the balance closes on it, and QOVER + QDRAI are
    # reported for the reader only.
    runoff = series.get("QRUNOFF")
    OUT = (runoff[:n] if runoff and len(runoff) >= n
           else [QO[i] + QD[i] for i in range(n)])
    rows, jumps = [], []
    for k, (d0, d1) in enumerate(windows):
        d1 = min(d1, n)
        if d0 >= d1:
            break
        sl = slice(d0, d1)
        p, e, qo, qd, qc = (sum(P[sl]), sum(ET[sl]), sum(QO[sl]), sum(QD[sl]),
                            sum(QC[sl]))
        out = sum(OUT[sl])
        dt = TWS[d1 - 1] - TWS[d0]
        rows.append({"d0": d0, "d1": d1, "P_mm": round(p, 2),
                     "runoff_mm": round(out, 2),
                     "ET_mm": round(e, 2), "QOVER_mm": round(qo, 2),
                     "QDRAI_mm": round(qd, 2), "QCHARGE_mm": round(qc, 2),
                     "dTWS_mm": round(dt, 2),
                     "residual_mm": round(p - e - out - dt, 2)})
        if d1 < n:
            jumps.append(round(TWS[d1] - TWS[d1 - 1], 2))
    if not rows:
        return None
    last = rows[-1]["d1"]
    year = {"P_mm": round(sum(P[:last]), 1), "ET_mm": round(sum(ET[:last]), 1),
            "QOVER_mm": round(sum(QO[:last]), 1),
            "QDRAI_mm": round(sum(QD[:last]), 1),
            "QCHARGE_mm": round(sum(QC[:last]), 1),
            "dTWS_mm": round(TWS[last - 1] - TWS[0], 1)}
    year["runoff_mm"] = round(sum(OUT[:last]), 1)
    year["runoff_source"] = ("QRUNOFF" if (runoff and len(runoff) >= n)
                             else "QOVER + QDRAI (QRGWL and capped snow are "
                                  "not on this history)")
    year["residual_mm"] = round(year["P_mm"] - year["ET_mm"]
                                - year["runoff_mm"] - year["dTWS_mm"], 1)
    year["stamp_jumps_mm"] = round(sum(jumps), 1)
    year["residual_fraction_of_P"] = (round(year["residual_mm"] / year["P_mm"], 3)
                                      if year["P_mm"] else None)
    return {"windows": rows, "stamp_jumps_mm": jumps, "year": year}
