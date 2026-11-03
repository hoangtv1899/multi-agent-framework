#!/usr/bin/env python3
"""Each value an audited claim states, traced to the regional table.

The audit of step 3 checks that a claim's values are in the finding it cites;
the finding is a figure script's own result, so a value the script computed
wrongly passes. Here each value of a whole-domain claim is traced instead to
03_results/regional.json, the table the numbers check has reproduced from the
model output, by one of a small set of derivations:

  the value itself       any number of the table: a run's value for a region,
                         field and year; a run minus the control; the event's
                         normal, anomaly or percent; a burned area; a share
  a difference           of two values of the table with the same field and
                         year (two runs, or two regions), either way round
  a percent change       100 x (a - b) / b, of two such values
  a region's size        its cell count or area
  a month or a season    a run's value of one month or season, a run minus
                         the control in it, or the event's normal of it, and
                         a difference or percent change of two such values of
                         the same field and period; only when the claim names
                         that month ('July') or season ('June-August',
                         'summer', 'JJA')

A derivation may use only a field the claim names (its `variables`, a field
code in its text, or a framework field's words such as "burned area"); a
region's size and a changed share name no field. It may use only a region the
claim names, when it names any, and an elevation band only when the claim
names the band. A difference or a percent change pairs two runs in one
region, or two regions of one run, never both.

A value is traced when a derivation gives it EXACTLY (to 1e-9) when the
claim's `values` entry carries more digits than its text prints, or when the
text does not print it at all; a value given at the precision it is printed
to is traced within half a unit of its last printed digit (a whole number to
the unit: '30' is 29.5 to 30.5). A number is a year
only when it is one of the table's years and no unit follows it. A value no
derivation gives marks its claim "not traced to the numbers"; when a value of
the headline cannot be traced, REPORT.md is withheld.

  trace(claims, headline, table)   the derivations, written into each claim
  withheld_markdown(report)        REPORT.md when the headline cannot be traced
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

TABLE_FILE = "03_results/regional.json"
NOT_TRACED = "not traced to the numbers"
_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])-?\d+(?:,\d{3})*(?:\.\d+)?")
_YEARLIKE = re.compile(r"^(19|20)\d{2}$")
VEG_SUFFIX = ", vegetated"
_CODE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
# THE UNIT AFTER A NUMBER (2026-11-03). A year is a year only when no unit
# follows it (a vegetation carbon of 1950-2050 gC/m2 is a value); a whole
# number followed by a measured quantity's unit must trace like any other
# value, while one followed by a design label's unit ("2 K", "the 2000 m
# band") need not.
_UNIT_AFTER = re.compile(
    r"\s*(percent|per cent|%|gC\s*/\s*m\^?2|gC\s+m-2|kg\s*/\s*m\^?2|kg\s+m-2|"
    r"W\s*/\s*m\^?2|W\s+m-2|km\^?2|square kilomet\w*|mm|cm|km|m|K|cells?)"
    r"(?![A-Za-z0-9])")
MEASURED = {"percent", "gc/m2", "kg/m2", "w/m2", "km2", "mm"}
# How a derivation ranks when several give the value: the table's own number
# first, a pair of its numbers last.
_RANK = {"value": 0, "difference_table": 1, "event": 1, "size": 2, "difference": 3,
         "percent_change": 4}


def _short(region: str) -> str:
    return str(region).replace(VEG_SUFFIX, "")


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def norm_unit(u) -> Optional[str]:
    """A unit in one spelling: 'gC/m^2/yr' and 'gC/m2 per year' are 'gc/m2',
    '%' is 'percent', 'square kilometres' is 'km2'. None for no unit."""
    if u is None:
        return None
    x = str(u).strip().lower().replace("**", "").replace("^", "").replace(" per ", "/")
    if x in ("%", "percent", "per cent"):
        return "percent"
    if x.startswith("square kilomet"):
        return "km2"
    x = re.sub(r"\s*m-2", "/m2", x).replace(" ", "")
    x = re.sub(r"(/(yr|year|y|a|month|mon|season)|yr-1|y-1)$", "", x)
    return {"cell": "cells"}.get(x, x)


def unit_after(text: str, end: int) -> Optional[str]:
    """The unit that follows the number ending at `end`, in norm_unit's
    spelling, or None."""
    m = _UNIT_AFTER.match(str(text or ""), end)
    return norm_unit(m.group(1)) if m else None


def table_years(table: Dict[str, Any]) -> set:
    """The years a number of this table may be: its own and the event's."""
    ev = (table or {}).get("event") or {}
    out = set()
    for y in list((table or {}).get("years") or []) + list(ev.get("normal_years") or []) + \
            list(ev.get("target_years") or []):
        try:
            out.add(int(y))
        except (TypeError, ValueError):
            continue
    return out


def _is_year(word: str, unit: Optional[str], years: set) -> bool:
    w = str(word).replace(",", "").lstrip("-")
    return unit is None and bool(_YEARLIKE.match(w)) and int(w) in (years or set())


def field_units(table: Dict[str, Any]) -> Dict[str, str]:
    """{field: its unit, in norm_unit's spelling}, from the runs' field
    records and the observed blocks; a framework field named '..._km2' is in
    km2."""
    out: Dict[str, str] = {}
    for r in ((table or {}).get("runs") or {}).values():
        for f, meta in ((r or {}).get("fields") or {}).items():
            if (meta or {}).get("units"):
                out.setdefault(f, norm_unit(meta["units"]))
        for reg in ((r or {}).get("values") or {}).values():
            for f in (reg or {}):
                if f.endswith("_km2"):
                    out.setdefault(f, "km2")
    for name, o in ((table or {}).get("observed") or {}).items():
        out.setdefault(name, norm_unit((o or {}).get("units")) or
                       ("km2" if name.endswith("_km2") else None))
    return {k: v for k, v in out.items() if v}


def table_fields(table: Dict[str, Any]) -> set:
    out = set()
    for r in ((table or {}).get("runs") or {}).values():
        for reg in ((r or {}).get("values") or {}).values():
            out |= set(reg or {})
    out |= set(((table or {}).get("observed") or {}))
    return out


def named_fields(text: str, variables=None, known=()) -> set:
    """The fields a sentence names: its `variables`, the field codes in it,
    and a framework field whose words it uses ('burned area' names
    burned_area_km2)."""
    t = str(text or "")
    out = set(variables or []) | set(_CODE.findall(t))
    low = t.lower()
    for f in known or ():
        if f != f.upper():
            words = re.sub(r"_(km2|m2|mm|pct|percent)$", "", f).replace("_", " ").strip()
            if words and words in low:
                out.add(f)
    return out


_BAND = (re.compile(r"below (-?[\d.]+) m"), re.compile(r"(-?[\d.]+)-(-?[\d.]+) m"),
         re.compile(r"(-?[\d.]+) m and above"))
# Words that name the whole grid's vegetated cells.
_ALL_CELLS = ("vegetated cells", "domain", "whole grid", "the grid", "model box",
              "whole box")


def _band_edges(label: str) -> Optional[Tuple[float, ...]]:
    """The finite edges of an elevation band's name ('1000-1500 m' (1000,
    1500), 'below 1000 m' (1000,)), or None when it is not a band."""
    for rx in _BAND:
        m = rx.fullmatch(label.strip())
        if m:
            return tuple(float(g) for g in m.groups())
    return None


def _head_band(region: str) -> Tuple[str, Optional[str]]:
    """('watershed', '1000-1500 m') of 'watershed, vegetated, 1000-1500 m';
    a band over every vegetated cell has the head 'vegetated cells'."""
    parts = [p.strip() for p in str(region).split(",")]
    band = parts[-1] if _band_edges(parts[-1]) else None
    head = "vegetated cells" if (band and len(parts) == 1) else parts[0]
    return head, band


def allowed_regions(text: str, regions) -> Optional[set]:
    """The regions a claim's derivations may use, from the regions it names.

    REGIONS THE CLAIM NAMES (2026-11-03). On a table with elevation bands,
    every value of every band paired with every other made almost any small
    number. A band is used only when the claim names it (each of its edges is
    a number of the text: 'between 1500 and 2000 m' names '1500-2000 m'); a
    region is used only when the claim names it ('watershed', 'burned any
    year'; 'the domain' or 'vegetated cells' name the whole grid's vegetated
    cells), or names no region at all. None when nothing restricts."""
    low = str(text or "").lower()
    nums = {float(w.replace(",", "")) for w in _NUMBER.findall(str(text or ""))}
    regions = list(regions or [])
    heads = {h for h, _ in map(_head_band, regions)}

    def head_named(h):
        if h == "vegetated cells":
            return any(w in low for w in _ALL_CELLS)
        return h.replace("_", " ").lower() in low

    named_heads = {h for h in heads if head_named(h)}
    out = set()
    for r in regions:
        h, b = _head_band(r)
        if named_heads and h not in named_heads:
            continue
        if b and not all(e in nums for e in _band_edges(b)):
            continue
        out.add(r)
    return out


# MONTHS AND SEASONS A CLAIM NAMES (2026-11-03). The table carries each
# field by month and by season, and the report draws them, so a claim about a
# July or a summer may trace to those numbers; only when it names that month
# or season, so a year's number is never matched by one of its 16 periods by
# chance. The labels are the table's own (domain_analysis.month_key and
# season_key: "2015-07", "2015-JJA"; DJF is the year of its January).
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December")
SEASON_MONTHS = {"DJF": (12, 2), "MAM": (3, 5), "JJA": (6, 8), "SON": (9, 11)}
_SEASON_WORDS = {"winter": "DJF", "spring": "MAM", "summer": "JJA", "autumn": "SON"}
# a month by its name or its three-letter form, capitalised ('May' the month,
# not 'may' the verb; 'Sept' too)
_MONTH_WORDS = "|".join(sorted({m for n in MONTHS for m in (n, n[:3])} | {"Sept"},
                               key=len, reverse=True))
_MONTH_RX = re.compile(r"\b(" + _MONTH_WORDS + r")\b\.?")
_RANGE_RX = re.compile(r"\b(?P<a>" + _MONTH_WORDS + r")\b\.?\s*(?:-|\u2013|to|through|until)"
                       r"\s*(?P<b>" + _MONTH_WORDS + r")\b\.?")


def _month_number(word: str) -> Optional[int]:
    w = word.rstrip(".")
    for i, n in enumerate(MONTHS, 1):
        if w in (n, n[:3]) or (w == "Sept" and i == 9):
            return i
    return None


def named_periods(text: str) -> set:
    """The months and seasons a claim names, as ("month", "07") and
    ("season", "JJA"): a month by name ('July', 'Jul'); a season by its code
    ('JJA'), its months as a range ('June-August', 'Dec to Feb') or its word
    ('summer'; 'fall' only as 'the fall', 'in fall'). A range that is a season
    names the season, not its two end months."""
    t = str(text or "")
    out = set()
    for m in list(_RANGE_RX.finditer(t)):
        ab = (_month_number(m.group("a")), _month_number(m.group("b")))
        for name, ends in SEASON_MONTHS.items():
            if ab == ends:
                out.add(("season", name))
                t = t.replace(m.group(0), " ")
    for m in _MONTH_RX.finditer(t):
        out.add(("month", f"{_month_number(m.group(1)):02d}"))
    out |= {("season", c) for c in re.findall(r"\b(DJF|MAM|JJA|SON)\b", t)}
    low = t.lower()
    out |= {("season", c) for w, c in _SEASON_WORDS.items()
            if re.search(rf"\b{w}s?\b", low)}
    if re.search(r"\b(?:the|in|during|over|this|that|last|of)\s+fall\b", low):
        out.add(("season", "SON"))
    return out


def period_of(key) -> Optional[Tuple[str, str]]:
    """("month", "07") of "2015-07", ("season", "JJA") of "2015-JJA" or of
    "JJA", ("month", "07") of "07"; None of a year."""
    k = str(key)
    tail = k.split("-", 1)[1] if re.match(r"^\d{4}-", k) else k
    if tail in SEASON_MONTHS:
        return ("season", tail)
    if re.fullmatch(r"\d{2}", tail) and 1 <= int(tail) <= 12:
        return ("month", tail)
    return None


def when(key) -> str:
    """A table's year or period in words: '2015', 'July 2015', 'JJA 2015'."""
    p = period_of(key)
    k = str(key)
    if p is None:
        return k
    y = k.split("-", 1)[0] if re.match(r"^\d{4}-", k) else ""
    name = MONTHS[int(p[1]) - 1] if p[0] == "month" else p[1]
    return f"{name} {y}".strip()


def entries(table: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every number of the table, each with where it is, what it says and its
    unit (when the table records it)."""
    out: List[Dict[str, Any]] = []
    units = field_units(table)

    def add(x, kind, text, **where):
        if _is_num(x):
            u = where.pop("units", None) or units.get(where.get("field"))
            out.append(dict(where, value=float(x), how=kind, text=text, units=u))

    for run, r in (table.get("runs") or {}).items():
        for reg, fields in (r.get("values") or {}).items():
            for f, years in (fields or {}).items():
                for y, x in (years or {}).items():
                    add(x, "value", f"{f}, {_short(reg)}: {run}, {y}", run=run,
                        region=reg, field=f, year=str(y))
    for run, regs in (table.get("differences") or {}).items():
        for reg, fields in (regs or {}).items():
            for f, years in (fields or {}).items():
                for y, x in (years or {}).items():
                    add(x, "difference_table", f"{f}, {_short(reg)}: {run} minus control, {y}",
                        run=run, minus="control", region=reg, field=f, year=str(y))
    # each month and season: a run's value, and a run minus the control
    for kind in ("monthly", "seasonal"):
        for run, r in (table.get("runs") or {}).items():
            for reg, fields in (r.get(kind) or {}).items():
                for f, keys in (fields or {}).items():
                    for k, x in (keys or {}).items():
                        add(x, "value", f"{f}, {_short(reg)}: {run}, {when(k)}", run=run,
                            region=reg, field=f, year=str(k), period=period_of(k))
        for run, regs in (table.get(f"{kind}_differences") or {}).items():
            for reg, fields in (regs or {}).items():
                for f, keys in (fields or {}).items():
                    for k, x in (keys or {}).items():
                        add(x, "difference_table",
                            f"{f}, {_short(reg)}: {run} minus control, {when(k)}", run=run,
                            minus="control", region=reg, field=f, year=str(k),
                            period=period_of(k))
    ev = table.get("event") or {}
    ny = sorted(int(y) for y in ev.get("normal_years") or [])
    span = "" if not ny else str(ny[0]) if len(ny) == 1 else f"{ny[0]}-{ny[-1]}"
    for reg, fields in (ev.get("anomalies") or {}).items():
        for f, a in (fields or {}).items():
            base = f"{f}, {_short(reg)}: control"
            add(a.get("normal"), "event", f"{base}, the {span} normal", run="control",
                region=reg, field=f, what="normal")
            for k in ("normal_min", "normal_max", "normal_sd"):
                add(a.get(k), "event", f"{base}, the {span} {k.replace('normal_', '')}",
                    run="control", region=reg, field=f, what=k)
            for y, yy in (a.get("years") or {}).items():
                yy = yy or {}
                add(yy.get("anomaly"), "event", f"{base}, {y} minus the {span} normal",
                    run="control", region=reg, field=f, year=str(y), what="anomaly")
                add(yy.get("anomaly_pct"), "event",
                    f"{base}, {y} against the {span} normal, percent", run="control",
                    region=reg, field=f, year=str(y), what="anomaly_pct", units="percent")
    for kind in ("monthly_normal", "seasonal_normal"):
        for reg, fields in (ev.get(kind) or {}).items():
            for f, labels in (fields or {}).items():
                for lab, x in (labels or {}).items():
                    add(x, "event", f"{f}, {_short(reg)}: control, the {span} normal of "
                                    f"{when(lab)}", run="control", region=reg, field=f,
                        what=kind, period=period_of(lab))
    for reg, info in (table.get("regions") or {}).items():
        add((info or {}).get("cells"), "size", f"{_short(reg)}: its cells", region=reg,
            what="cells", units="cells")
        add((info or {}).get("area_km2"), "size", f"{_short(reg)}: its area, km2", region=reg,
            what="area_km2", units="km2")
    for run, cs in (table.get("changed_share") or {}).items():
        for reg, x in ((cs or {}).get("share") or {}).items():
            add(x, "value", f"{_short(reg)}: the share run {run}'s change reaches",
                run=run, region=reg, what="changed_share", units="share")
    for name, o in (table.get("observed") or {}).items():
        for run, regs in (o.get("model") or {}).items():
            for reg, years in (regs or {}).items():
                for y, x in (years or {}).items():
                    add(x, "value", f"{name}, {_short(reg)}: {run}, {y}", run=run,
                        region=reg, field=name, year=str(y))
        for reg, years in (o.get("values") or {}).items():
            for y, x in (years or {}).items():
                add(x, "value", f"{name}, {_short(reg)}: observed ({o.get('source')}), {y}",
                    run="observed", region=reg, field=name, year=str(y))
    return out


def _pairs(table: Dict[str, Any]) -> Dict[Tuple[str, str], List[Tuple[str, str, float]]]:
    """{(field, year): [(run, region, value)]}: the model's values that a
    difference or a percent change may pair. An older table's burned area
    beside an observation counts as the run's burned area."""
    out: Dict[Tuple[str, str], List[Tuple[str, str, float]]] = {}
    seen = set()
    for run, r in (table.get("runs") or {}).items():
        for reg, fields in (r.get("values") or {}).items():
            for f, years in (fields or {}).items():
                for y, x in (years or {}).items():
                    if _is_num(x):
                        out.setdefault((f, str(y)), []).append((run, reg, float(x)))
                        seen.add((run, reg, f, str(y)))
        # a month's or a season's values pair within that month or season
        for kind in ("monthly", "seasonal"):
            for reg, fields in (r.get(kind) or {}).items():
                for f, keys in (fields or {}).items():
                    for k, x in (keys or {}).items():
                        if _is_num(x):
                            out.setdefault((f, str(k)), []).append((run, reg, float(x)))
    for name, o in (table.get("observed") or {}).items():
        for run, regs in (o.get("model") or {}).items():
            for reg, years in (regs or {}).items():
                for y, x in (years or {}).items():
                    if _is_num(x) and (run, reg, name, str(y)) not in seen:
                        out.setdefault((name, str(y)), []).append((run, reg, float(x)))
    return out


def _unit(printed: str) -> float:
    """Half a unit of the last digit a number is printed with: '25.0' 0.05,
    '28' 0.5, '1950' 5 (a trailing zero of a whole number is not a digit)."""
    s = printed.replace(",", "").lstrip("-")
    if "." in s:
        return 0.5 * 10.0 ** -len(s.split(".")[1])
    tz = len(s) - len(s.rstrip("0")) if s.strip("0") else 0
    return 0.5 * 10.0 ** tz


def _three_sig(v: float) -> str:
    """v to 3 significant figures, trailing zeros kept: 25.0, 1950, 0.00199."""
    import math
    if v == 0 or not math.isfinite(v):
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    if e >= 2:
        return f"{round(abs(v), 2 - e):.0f}"
    return f"{abs(v):.{2 - e}f}"


def _printed(v: float, text: str, years: Optional[set] = None) -> Tuple[str, bool]:
    """(how the claim prints value v, whether its text prints it): the number
    in the text that states v at its own precision (closest first; a year of
    the table with no unit after it is a date, not a value), else v to 3
    significant figures."""
    best = None
    t = str(text or "")
    for m in _NUMBER.finditer(t):
        ww = m.group(0).replace(",", "")
        if _is_year(ww, unit_after(t, m.end()), years or set()):
            continue
        d = abs(abs(float(ww)) - abs(v))
        if d <= _unit(ww) + 1e-9 and (best is None or d < best[0]):
            best = (d, ww.lstrip("-"))
    return (best[1], True) if best else (_three_sig(abs(v)), False)


def printed_form(v: float, text: str, years: Optional[set] = None) -> str:
    """How the claim prints value v (see _printed)."""
    return _printed(v, text, years)[0]


def at_printed_precision(v: float, printed: str) -> bool:
    """True when v carries no digit beyond the ones `printed` shows: 25.3 as
    '25.3', 1950.0 as '1950'; False for -25.0384 as '25.0'."""
    s = printed.replace(",", "").lstrip("-")
    k = len(s.split(".")[1]) if "." in s else \
        -(len(s) - len(s.rstrip("0")) if s.strip("0") else 0)
    return abs(abs(v) - round(abs(v), k)) <= 1e-9 * max(1.0, abs(v))


def _signed(v: float, printed: str) -> str:
    """v, with its sign, at the precision the claim prints it: -25.0, 1950."""
    s = printed.replace(",", "")
    if "." in s:
        return f"{v:.{len(s.split('.')[1])}f}"
    tz = len(s) - len(s.rstrip("0")) if s.strip("0") else 0
    return f"{round(v, -tz):.0f}"


def trace_value(v: float, text: str, ents: List[Dict[str, Any]], pairs,
                hint_years: Optional[set] = None,
                hint_fields: Optional[set] = None,
                used: Optional[set] = None,
                sizes: Optional[Dict[str, int]] = None,
                years: Optional[set] = None,
                regions: Optional[set] = None,
                periods: Optional[set] = None) -> Dict[str, Any]:
    """The derivation of one value, or why there is none.

    hint_fields are the fields the claim names (named_fields; the field codes
    in `text` when None): a derivation of any other field is not one.
    `regions` are the regions it may use (allowed_regions over the regions
    of the entries when None); `periods` the months and seasons it names
    (named_periods of `text` when None): a month's or a season's number is
    used only when the claim names it, and is preferred then. When
    several derivations give the value, the table's own number before a pair
    of numbers, and among equals the one whose year and region the claim
    names, and one not already given to another value of the same claim (two
    zeros in "burns nothing in 2015 or 2016" are two years, not one twice)."""
    low = str(text or "").lower()
    fields = set(hint_fields) if hint_fields is not None else named_fields(text)
    periods = set(periods) if periods is not None else named_periods(text)
    if regions is None:
        regions = allowed_regions(text, {e["region"] for e in ents if e.get("region")})
    printed, in_text = _printed(v, text, years)
    exact = 1e-9 * max(1.0, abs(v))
    # EXACT WHEN THE CLAIM KNOWS MORE THAN IT PRINTS (2026-11-03). A value
    # given with more digits than its text shows, or not shown at all, is the
    # figure script's number: it must be the table's to 1e-9, as every demo
    # value is. Half a printed unit is allowed only for a value given at the
    # precision it is printed to.
    loose = in_text and at_printed_precision(v, printed)
    # a whole number given as such states its units digit: '30' is 29.5-30.5
    tol = (_half_unit(printed) if loose else 0.0) + exact
    fu = ents_units(ents)
    rec = {"value": v, "printed": printed, "traced": False, "file": TABLE_FILE,
           "precision": "half a unit of the last printed digit" if loose else "exact"}
    cands = []

    def score(c, d):
        reg = _short(c.get("region") or "").split(",")[0].strip().lower()
        return (d > exact, bool(periods) and c.get("period") not in periods,
                _RANK.get(c["how"], 9),
                not (hint_years and str(c.get("year") or "")[:4] in hint_years),
                # the smallest region the claim names: "the watershed's
                # vegetated cells" names the watershed
                (0, (sizes or {}).get(c.get("region"), 0)) if reg and reg in low
                else (1, 0),
                c["text"] in (used or set()), d)

    for c in ents:
        if c.get("field") is not None and c["field"] not in fields:
            continue
        if c.get("region") is not None and c["region"] not in regions:
            continue
        if c.get("period") is not None and c["period"] not in periods:
            continue
        d = abs(c["value"] - v)
        if d <= tol:
            cands.append((score(c, d), c))
    for (f, y), vals in pairs.items():
        if f not in fields:
            continue
        per = period_of(y)
        if per is not None and per not in periods:
            continue
        for ra, ga, a in vals:
            if ga not in regions:
                continue
            for rb, gb, b in vals:
                # two runs in one region, or two regions of one run
                if (ra == rb) == (ga == gb) or gb not in regions:
                    continue
                same = ga == gb
                where = (f"{f}, {_short(ga)}: {ra} minus {rb}, {when(y)}" if same else
                         f"{f}, {when(y)}: {ra} {_short(ga)} minus {rb} {_short(gb)}")
                d = abs((a - b) - v)
                if d <= tol:
                    c = {"value": a - b, "how": "difference", "text": where, "run": ra,
                         "minus": rb, "region": ga, "minus_region": gb, "field": f,
                         "year": y, "period": per, "units": fu.get(f)}
                    cands.append((score(c, d) + (not same,), c))
                if b != 0:
                    p = 100.0 * (a - b) / b
                    d = abs(p - v)
                    if d <= tol:
                        where = (f"{f}, {_short(ga)}: {ra} against {rb}, percent, {when(y)}"
                                 if same else
                                 f"{f}, {when(y)}: {ra} {_short(ga)} against {rb} "
                                 f"{_short(gb)}, percent")
                        c = {"value": p, "how": "percent_change", "text": where, "run": ra,
                             "against": rb, "region": ga, "against_region": gb, "field": f,
                             "year": y, "period": per, "units": "percent"}
                        cands.append((score(c, d) + (not same,), c))
    if not cands:
        rec["text"] = f"{_signed(v, printed)}: {NOT_TRACED}"
        rec["why"] = ("no value of the table of a field, region and period the claim names, "
                      "difference or percent change of two of its values with the same field "
                      "and year (two runs in one region, or two regions of one run), nor a "
                      "region's size, gives it "
                      + (f"to the precision it is printed with ({printed})" if loose else
                         "exactly (the claim's value carries more digits than it prints)"))
        return rec
    cands.sort(key=lambda t: t[0])
    c = cands[0][1]
    rec.update({k: c[k] for k in c if k not in ("text",)})
    rec.update(value=v, derived=c["value"], traced=True, source=c["text"],
               match="exact" if abs(c["value"] - v) <= exact else
               "within the printed precision",
               text=f"{_signed(v, printed)} = {c['text']}",
               n_alternatives=len(cands) - 1)
    return rec


def ents_units(ents: List[Dict[str, Any]]) -> Dict[str, str]:
    """{field: unit} as the entries record it (a run's value of the field)."""
    out: Dict[str, str] = {}
    for e in ents:
        if e.get("field") and e.get("units") and e.get("how") == "value":
            out.setdefault(e["field"], e["units"])
    return out


def _years(text: str) -> set:
    return {w for w in re.findall(r"\b(?:19|20)\d{2}\b", str(text or ""))}


def _is_year_value(v, text: str, years: set) -> bool:
    """A claim value is a date when it is one of the table's years and the
    claim's text prints it with no unit after it."""
    if not (float(v).is_integer() and int(v) in (years or set())):
        return False
    t = str(text or "")
    return any(m.group(0).replace(",", "").lstrip("-") == str(int(v))
               and unit_after(t, m.end()) is None for m in _NUMBER.finditer(t))


def trace(claims: List[Dict[str, Any]], headline: Optional[str],
          table: Dict[str, Any]) -> Dict[str, Any]:
    """Trace every claim value and the headline's values to the table.

    Each claim gains "derivations" (one per value) and "traced"; a claim with
    a value that cannot be traced gains "trace": "not traced to the numbers".
    Returns the summary analysis.json carries under "numbers_trace", with
    "withheld_because" set when a value of the headline cannot be traced."""
    table = table or {}
    ents = entries(table)
    pairs = _pairs(table)
    years = table_years(table)
    known = table_fields(table)
    sizes = {k: (v or {}).get("cells") or 0 for k, v in (table.get("regions") or {}
                                                         ).items()}
    regs = {e["region"] for e in ents if e.get("region")}
    n_vals = n_ok = 0
    not_traced = []
    for i, c in enumerate(claims, 1):
        text = str(c.get("claim") or "")
        hy = _years(text)
        hf = named_fields(text, c.get("variables"), known)
        ds = []
        for v in c.get("values") or []:
            if not _is_num(v):
                continue
            if _is_year_value(v, text, years):
                ds.append({"value": v, "traced": True, "how": "year",
                           "text": f"{int(v)} = a year"})
                continue
            d = trace_value(float(v), text, ents, pairs, hy, hf,
                            used={x.get("source") for x in ds}, sizes=sizes, years=years,
                            regions=allowed_regions(text, regs),
                            periods=named_periods(text))
            ds.append(d)
            n_vals += 1
            n_ok += bool(d["traced"])
        c["derivations"] = ds
        c["traced"] = all(d["traced"] for d in ds)
        if not c["traced"]:
            c["trace"] = NOT_TRACED
            not_traced.append(i)
    head = trace_headline(headline, claims, years=years, known=known)
    out = {"file": TABLE_FILE, "n_values": n_vals, "n_traced": n_ok,
           "claims_not_traced": not_traced, "headline": head,
           "derivations_allowed": [
               "the value itself (any number of the table)",
               "the difference of two table values of the same field and year: two runs "
               "in one region, or two regions of one run",
               "a percent change of one such value relative to another",
               "a region's cell count or area",
               "a month's or a season's value, run minus control or normal, and the "
               "difference or percent change of two values of one month or season, only "
               "when the claim names that month or season",
               "only of a field the claim names, and of a region it names (an elevation "
               "band only when it names the band; any region when it names none)"],
           "precision": "exact (1e-9) when the claim's value carries more digits than its "
                        "text prints, or its text does not print it; else half a unit of "
                        "the last digit the value is printed with"}
    bad = [h["printed"] for h in head if not h["traced"]]
    if bad:
        out["withheld_because"] = (f"the headline's value{'s' if len(bad) > 1 else ''} "
                                   f"{', '.join(bad)} cannot be traced to the numbers")
    return out


def _half_unit(word: str) -> float:
    """Half a unit of the last digit as written, a whole number to the unit:
    '40' 0.5, '8.2' 0.05 (the convention of step 3's same_number)."""
    s = word.replace(",", "").lstrip("-")
    return 0.5 * 10.0 ** -(len(s.split(".")[1]) if "." in s else 0)


def trace_headline(headline: Optional[str], claims: List[Dict[str, Any]],
                   years: Optional[set] = None, known=()) -> List[Dict[str, Any]]:
    """Each number the headline prints, with the claim values it states.

    A number states a claim value when they agree at the coarser of the two
    precisions (the headline's as written, a whole number to the unit; the
    claim's as its text prints it), the claim value is traced, and its
    derivation fits the headline: a percent for a percent, the same unit for a
    number with a unit, and a field the headline names when it names any. A
    year of the table with no unit after it is a date. A whole number that no
    claim states and that carries no measured unit (percent, gC/m2, kg/m2,
    W/m2, km2, mm) is a label ('2 K warmer', 'the 2000 m band'); any other
    number must state a traced claim value."""
    text = str(headline or "")
    vals = [(d, c) for c in claims for d in (c.get("derivations") or [])
            if _is_num(d.get("value")) and d.get("how") != "year"]
    names = {f for f in named_fields(text, None, known) if f in set(known or ())}
    out = []
    for m in _NUMBER.finditer(text):
        w = m.group(0).replace(",", "").lstrip("-")
        unit = unit_after(text, m.end())
        if _is_year(w, unit, years or set()):
            continue
        h = float(w)

        def fits(d):
            du = d.get("units")
            if unit == "percent" and du != "percent":
                return False
            if unit and unit != "percent" and (du == "percent" or (du and du != unit)):
                return False
            if names and d.get("field") and d["field"] not in names:
                return False
            tol = max(_half_unit(w), _unit(str(d.get("printed") or "0")))
            return abs(h - abs(float(d["value"]))) <= tol + 1e-9 * max(1.0, h)

        near = [d for d, _ in vals if fits(d)]
        hits = [d for d in near if d.get("traced")]
        if not near and "." not in w and unit not in MEASURED:
            continue
        rec = {"printed": w, "traced": bool(hits),
               "derivations": [d.get("text") for d in hits][:3]}
        if unit:
            rec["unit"] = unit
        if not hits:
            rec["why"] = ("a claim states it, but that value is not traced to the numbers"
                          if near else "no claim states this value"
                          + (f" in {unit}" if unit else ""))
        out.append(rec)
    return out


def withheld_markdown(report: Dict[str, Any]) -> str:
    """REPORT.md when the headline's values cannot be traced: what could not
    be traced, then every claim with its derivations. The answer and the
    headline are not presented as findings. A renderer of the record."""
    from agents.analysis.step4_report import _plain
    t = report.get("numbers_trace") or {}
    lines = [f"# WITHHELD: {t.get('withheld_because')}", "",
             f"*Question: {_plain(report.get('question') or '(no question recorded)')}*", "",
             "Each value a claim states was traced to the regional numbers "
             f"({t.get('file')}), which were made again from the model output before the "
             "Analyzer ran. The headline states a value that no claim traces to them, so "
             "this report is withheld. The record, analysis.json, is kept whole beside it.",
             "", f"The headline held back: \"{_plain(report.get('headline'))}\"", ""]
    for h in t.get("headline") or []:
        lines.append(f"- {h['printed']}: "
                     + ("traced, " + "; ".join(h.get("derivations") or [])
                        if h.get("traced") else (h.get("why") or NOT_TRACED)))
    lines += ["", f"## The claims and what their values trace to "
                  f"({t.get('n_traced')} of {t.get('n_values')} values traced)", ""]
    for i, c in enumerate(report.get("claims") or [], 1):
        lines.append(f"{i}. {_plain(c.get('claim'))}"
                     + (f" **({c['trace']})**" if c.get("trace") else ""))
        for d in c.get("derivations") or []:
            lines.append(f"   - {d.get('text') or (str(d.get('printed')) + ': ' + str(d.get('why')))}")
        lines.append("")
    return "\n".join(lines)
