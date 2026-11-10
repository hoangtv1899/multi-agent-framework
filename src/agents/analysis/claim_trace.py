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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents.analysis.analysis_plan import EXPLORATORY_LABEL, SCREENING_LABEL

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
    ('summer'; 'fall' only where it cannot be a decrease, _FALL_RX). A range
    that is a season names the season, not its two end months."""
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
    if any(m for m in _FALL_RX.finditer(low)):
        out.add(("season", "SON"))
    return out


# A RUN OF MONTHS IS NOT A SEASON'S VALUE (the user's decision of 2026-11-09).
# named_periods reads "June to August" as the season JJA, so a claim that
# went through the year month by month, on a record for every month ("below
# normal from June to August, largest in July"), failed for naming a season
# none of its records names, and its whole report was withheld. In the
# evidence trace such a range is now read as a run of months when no record
# of the claim is of the season: every month of the run must then be named by
# one of its records. The season's own evidence is still required whenever
# the claim names the season outright (its word or its code) or gives the
# range as one quantity (an aggregate word in the range's clause: "the June to
# August total"). The numbers rule is unchanged, so a season's value is never
# a value of monthly records.
AGGREGATE_WORDS = ("total", "totals", "totalled", "totaled", "sum", "summed", "mean",
                   "means", "average", "averaged", "averages", "cumulative", "combined",
                   "overall", "season", "seasonal", "altogether")
_CLAUSE_SPLIT = re.compile(r"[,;:.()]")


def month_runs(text: str) -> Dict[str, Dict[str, Any]]:
    """{season: {"words", "months", "aggregate"}} for each range of months in
    the text that spells a season ("June to August"): its words, its months
    ("06", "07", "08") and whether its clause gives it as one quantity
    (AGGREGATE_WORDS). A season spelled twice is an aggregate if either is."""
    t = str(text or "")
    out: Dict[str, Dict[str, Any]] = {}
    for m in _RANGE_RX.finditer(t):
        ab = (_month_number(m.group("a")), _month_number(m.group("b")))
        name = next((n for n, ends in SEASON_MONTHS.items() if ab == ends), None)
        if name is None:
            continue
        a = max((x.end() for x in _CLAUSE_SPLIT.finditer(t, 0, m.start())), default=0)
        b = next((x.start() for x in _CLAUSE_SPLIT.finditer(t, m.end())), len(t))
        words = set(re.findall(r"[a-z]+", t[a:b].lower()))
        months = [f"{(ab[0] - 1 + k) % 12 + 1:02d}" for k in range(3)]
        prev = out.get(name) or {}
        out[name] = {"words": m.group(0), "months": months,
                     "aggregate": bool(words & set(AGGREGATE_WORDS)) or bool(prev.get("aggregate"))}
    return out


def seasons_named_outright(text: str) -> set:
    """The seasons a text names by word or code ("summer", "JJA"), apart from
    any range of months that spells one."""
    t = _RANGE_RX.sub(" ", str(text or ""))
    return {lab for kind, lab in named_periods(t) if kind == "season"}


# 'FALL' IS A DECREASE MORE OFTEN THAN A SEASON (2026-11-03). "The fall in GPP"
# and "the rate of fall" named autumn, so a claim about a decrease traced to an
# SON number. 'fall' names the season only next to a year ('fall 2016', 'fall
# of 2016'), another season ('summer and fall'), or after a word of time ('in
# fall', 'during the fall'), and never when 'in', 'of', 'to', 'from' or 'by'
# follows it with anything but a year ('a fall in GPP', 'a fall of 4.6').
_NOT_A_DECREASE = r"(?!\s+(?:in|of|to|from|by)\s+(?!(?:19|20)\d{2}\b))"
_OTHER_SEASON = r"(?:summer|winter|spring|autumn)s?"
_FALL_RX = re.compile(
    r"\bfall\s+(?:of\s+)?(?:19|20)\d{2}\b"
    r"|\b(?:in|during|over|through|this|that|last|next|each|every)\s+"
    r"(?:the\s+)?(?:early\s+|late\s+)?fall\b" + _NOT_A_DECREASE +
    r"|\b" + _OTHER_SEASON + r"\s*(?:,|-|and|or|to|through)\s*fall\b" + _NOT_A_DECREASE +
    r"|\bfall\s*(?:,|-|and|or|to|through)\s*" + _OTHER_SEASON + r"\b")


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


def when_period(p) -> str:
    """A period of named_periods or period_of in words: 'July', 'JJA'."""
    kind, lab = tuple(p)
    return MONTHS[int(lab) - 1] if kind == "month" else lab


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


# A NUMBER IS ITS TYPE (the user's decision of 2026-11-08, "typed quantity
# rules"): "2000 m" is an elevation, never the year 2000. Question 3 of G part
# 2's run again lost five true claims, and question 2 one, to a band "above
# 2000 m" read as a year. A four-digit number names a year only with no unit
# after it, and not when it opens a range whose last number carries a length
# unit ("1500 to 2000 m", "2000-2500 m").
_RANGE_TAIL = re.compile(r"\s*(?:to|-|\u2013|and)\s*-?\d+(?:\.\d+)?\s*(?:km|cm|mm|m)"
                         r"(?![A-Za-z0-9])")


def years_named(text: str) -> set:
    """The years a claim's text names, each a four-digit number typed as a
    year: no unit after it, and not the first of a range in a length unit."""
    t = str(text or "")
    out = set()
    for m in re.finditer(r"\b(?:19|20)\d{2}\b", t):
        if unit_after(t, m.end()) is not None or _RANGE_TAIL.match(t, m.end()):
            continue
        out.add(m.group(0))
    return out


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
    number with a unit, a field the headline names when it names any, and a
    month or season only when the headline names it (named_periods). A
    year of the table with no unit after it is a date. A whole number that no
    claim states and that carries no measured unit (percent, gC/m2, kg/m2,
    W/m2, km2, mm) is a label ('2 K warmer', 'the 2000 m band'); any other
    number must state a traced claim value."""
    text = str(headline or "")
    vals = [(d, c) for c in claims for d in (c.get("derivations") or [])
            if _is_num(d.get("value")) and d.get("how") != "year"]
    names = {f for f in named_fields(text, None, known) if f in set(known or ())}
    # A SEASON'S NUMBER IS NOT A YEAR'S (2026-11-03): the claims' rule holds for
    # the headline too, so a headline that names no month or season cannot
    # print a claim's July or JJA value as the year's change.
    hp = named_periods(text)
    out = []
    for m in _NUMBER.finditer(text):
        w = m.group(0).replace(",", "").lstrip("-")
        unit = unit_after(text, m.end())
        if _is_year(w, unit, years or set()):
            continue
        h = float(w)

        def fits(d, by_period=True):
            du = d.get("units")
            if unit == "percent" and du != "percent":
                return False
            if unit and unit != "percent" and (du == "percent" or (du and du != unit)):
                return False
            if names and d.get("field") and d["field"] not in names:
                return False
            if by_period and d.get("period") and tuple(d["period"]) not in hp:
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
            other = sorted({when_period(d["period"]) for d, _ in vals
                            if d.get("traced") and d.get("period") and fits(d, False)})
            rec["why"] = ("a claim states it, but that value is not traced to the numbers"
                          if near else
                          f"a claim states it for {' and '.join(other)}, which the headline "
                          f"does not name" if other else "no claim states this value"
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
             "", f"The headline held back: \"{_plain(report.get('headline'))}\""
             + (f" {_plain(headline_caveat_line(report))}" if headline_caveat_line(report)
                else ""), ""]
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


# ═════════════════════════════════════════════════════════════════════════
# EVIDENCE RECORDS AND THE CLAIM TRACE OF A STUDY WITH A SUITE
# (package F1, 2026-11-05; plan sections 3.7 and 3.8)
# ═════════════════════════════════════════════════════════════════════════
# A study with a suite reaches its values only through the scoped, logged
# suite calls (the runner binds no whole table), so a claim is traced to the
# lookups, not to the regional table. Each answered lookup in
# 04_analysis/lookups.jsonl is an EVIDENCE RECORD, named by its line ("L7"):
# the call, run, field, period, region, method and settings, the lookup's
# hash, the comparison (against what, over which years), the support (years,
# share of area covered), the quality flags and reasons as WARNINGS, the
# evidence classes it can carry and the wording each class allows. A claim
# names its records (`evidence`), its class (`evidence_class`), the plan's
# expectation for a tested one (`expectation`) and the warnings it carries
# (`warnings`, by code, or the warning's words in its text).
#
# A claim FAILS when: it names no record, or one the log does not hold; a
# number it states is no value of its records; it names a year, month, season
# or region none of its records names; its class is missing or beyond its
# records (a tested expectation needs the plan's expectation and planned
# lookups only, never one found by screening, added by an amendment or made
# after the planned phase was closed (2026-11-08); an
# association needs an overlap; in-model attribution needs a difference from a
# run that differs from the control by one named change, confirmed by the
# server's checks); its wording goes beyond its class (a causal word on
# anything but in-model attribution, "caused by" in any form, the one
# in-model form being "in the model, the change caused"; "validated"); it
# gives a percent of a field whose notes allow none; it gives a timing shift
# of a record whose range across settings exceeds 7 days without the default,
# the smallest and the largest shift, each with its setting (G part 1
# decisions, 2026-11-08); or it leaves out a warning its records carry (among
# them that the settings reverse the direction of the shift, and that a
# record was made after the planned phase was closed). The headline: for a
# hypothesis only tested claims may lead, and never one resting on a lookup
# found by screening, amended or made after the close; for an exploratory question a
# finding from the screen may lead, labelled exploratory. An open cause
# proposes a run in the server's change vocabulary (domain_changes.resolve).
# Any failure withholds the report (evidence_withheld_markdown), as an
# untraced headline does.
LOOKUPS_LOG = "04_analysis/lookups.jsonl"
EVIDENCE_CLASSES = ("observation", "association", "tested expectation", "exploratory",
                    "in-model attribution")
IN_MODEL_WORDS = "in the model, the change caused"
CLASS_WORDING = {
    "observation": "what the run's values were or how they changed; a relation is "
                   "\"consistent with\"; no causal word",
    "association": "\"consistent with\" or \"went together with\": two patterns overlap; "
                   "never a causal word",
    "tested expectation": "the plan stated it before any lookup and it was found; "
                          "\"consistent with\" the expectation; no causal word",
    "exploratory": "found by looking, not stated first, and labelled exploratory; "
                   "\"consistent with\"; no causal word",
    "in-model attribution": f"\"{IN_MODEL_WORDS}\" only, for a run that differs from the "
                            "control by one named change: never a real-world cause, and it "
                            "does not isolate an intermediate step"}
HEADLINE_RULE = {
    "hypothesis": "a hypothesis question's headline comes from tested claims only",
    "exploratory": "an exploratory question's headline is labelled exploratory, every one "
                   "(the report puts the label on), and may come from the screen's display"}
FAILS_TRACE = "fails the evidence trace"
_CAUSAL = re.compile(r"\b(?:caus\w*|led to|leads? to|drove|drives?|driven|driving|due to|"
                     r"because of|result(?:s|ed|ing)? in|trigger\w*|induc\w*|explain\w*|"
                     r"responsible for|attribut\w*|owing to)\b", re.I)
_CAUSED_BY = re.compile(r"\bcaus\w*(?:\s+\w+){0,2}\s+by\b", re.I)
_BANNED = re.compile(r"\bvalidat\w*", re.I)
_PERCENT = re.compile(r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?\s*(?:%|percent\b|per cent\b)", re.I)
_COVER_KEYS = ("covered", "least_area_share", "covered_share", "inputs_area_share")
_LABEL_KEYS = ("settings", "normal_years", "event_years", "year", "years", "n_normal",
               "normal_years_used", "version", "cells", "region_area_km2")
_SCREEN_STATUS = {"insufficient support": ("insufficient_support", "insufficient support"),
                  "not scored yet": ("not_scored_yet", "not scored"),
                  "near-constant normal": ("near_constant", "near-constant")}


def _fin(x) -> bool:
    return _is_num(x) and x == x and abs(x) != float("inf")


def _elm_modules():
    from agents.analysis import analysis_plan as AP
    AP._elm()
    import domain_catalog as DCAT                                   # noqa: N812
    import domain_methods as M                                      # noqa: N812
    return AP, DCAT, M


def _short_reason(text: str) -> str:
    t = str(text).replace("not estimable: ", "")
    return re.split(r"\s\(|:", t)[0].strip()


def _read_log(analysis_dir) -> List[Dict[str, Any]]:
    import json
    from pathlib import Path
    p = Path(analysis_dir) / "lookups.jsonl"
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _years_in(x) -> set:
    out = set()
    for v in (x if isinstance(x, (list, tuple)) else [x]):
        if _is_num(v) and float(v).is_integer() and 1800 <= v <= 2200:
            out.add(str(int(v)))
        elif isinstance(v, str):
            out |= {w for w in re.findall(r"(?<!\d)(?:18|19|20|21)\d{2}(?!\d)", v)}
    return out


# A SERIES NAMES WHAT ITS DATES COVER (2026-11-08, after the third batch):
# a series' labels are kept one per entry ("labels.0", "labels.1", ...), and
# only a key named "labels" was read, so question 1's daily snow series named
# no year and its true "by the start of spring" in 2015 failed. A day label
# ("2015-03-12") now names its year, never its day as a month (the "-12" a
# month label ends in); every other label is read as before.
# STRICT, AND WINTER CONTIGUOUS (the user's decision of 2026-11-08): a series
# names a month only when it holds every day of it, and a season only when it
# holds its three months whole as one unbroken run. Winter is the suite's own
# (domain_methods.period_months): December of the year before with January
# and February, never a year's January, February and December, which are ten
# months apart.
_DAY_LABEL = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
# (year offset, month) of each season's months, from its January-or-later year
_SEASON_OF = {"DJF": ((-1, 12), (0, 1), (0, 2)), "MAM": ((0, 3), (0, 4), (0, 5)),
              "JJA": ((0, 6), (0, 7), (0, 8)), "SON": ((0, 9), (0, 10), (0, 11))}
# a month's days: February 28, or 29 when the series holds a 29 February
_MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _whole_months(days: Dict[Tuple[int, int], set]) -> set:
    """(year, month) of each month whose every day the series holds (`days`:
    (year, month) -> the days of it the series holds)."""
    out = set()
    for (y, m), ds in days.items():
        n = _MONTH_DAYS[m - 1] + (1 if m == 2 and 29 in ds else 0)
        if set(range(1, n + 1)) <= ds:
            out.add((y, m))
    return out


def _seasons_held(whole: set) -> set:
    """The seasons whose three months the series holds whole, one unbroken
    run each: winter from the December before (`whole`: _whole_months)."""
    return {k for k, parts in _SEASON_OF.items() for y in {y for y, _ in whole}
            if all((y + dy, mm) in whole for dy, mm in parts)}


def _labels(res: Dict[str, Any]) -> List[Any]:
    """A result's period words: its "period", "year" and "labels" keys, a
    flattened list's entries ("labels.12") among them."""
    out: List[Any] = []
    for k, v in res.items():
        parts = k.split(".")
        if parts[-1] in ("period", "labels", "year"):
            out += v if isinstance(v, list) else [v]
        elif len(parts) >= 2 and parts[-2] == "labels" and parts[-1].isdigit():
            out.append(v)
    return out


def _names(rec: Dict[str, Any], res: Dict[str, Any]) -> Dict[str, Any]:
    """The years, months, seasons and region a record names: its own period,
    its comparison's years, its support's years, its series' labels."""
    yrs, per, days = set(), set(), {}
    for lab in [rec.get("period")] + _labels(res):
        t = str(lab or "")
        yrs |= _years_in(lab)
        d = _DAY_LABEL.match(t)
        if d:
            days.setdefault((int(d.group(1)), int(d.group(2))), set()).add(int(d.group(3)))
            continue
        m = re.search(r"-(\d{2})$", t)
        if m:
            per.add(("month", m.group(1)))
        s = re.search(r"\b(DJF|MAM|JJA|SON)\b", t)
        if s:
            per.add(("season", s.group(1)))
    whole = _whole_months(days)
    per |= {("month", f"{mm:02d}") for _, mm in whole}
    per |= {("season", k) for k in _seasons_held(whole)}
    for k, v in res.items():
        if k.rsplit(".", 1)[-1] in ("normal_years", "event_years", "years",
                                    "normal_years_used", "against"):
            yrs |= _years_in(v)             # "against": a timing's "the normal years [...]"
    if rec.get("step") == "month":
        per |= {("month", f"{i:02d}") for i in range(1, 13)}
    if rec.get("step") == "season":
        per |= {("season", s) for s in SEASON_MONTHS}
    return {"years": yrs, "periods": per}


def _comparison(rec: Dict[str, Any], res: Dict[str, Any]) -> Dict[str, Any]:
    call, k = rec["call"], rec.get("keys") or {}
    if call == "anomaly":
        return {"against": "the normal years", "years": res.get("normal_years")}
    if call == "difference":
        return {"against": f"the control ({k.get('control') or 'control'})",
                "run": k.get("run"), "years": sorted(_years_in(rec.get("period")))}
    if call == "screen":
        return {"against": res.get("against"), "years": res.get("normal_years")}
    if call == "overlap":
        return {"against": "two maps over one region: an association"}
    if call == "elevation_split":
        return {"against": "the region's elevation bands and its other regions"}
    if rec.get("measure"):
        return {"against": res.get("against")}
    return {"against": None}


def _support(res: Dict[str, Any]) -> Dict[str, Any]:
    shares = [v for p, v in res.items() if p.rsplit(".", 1)[-1] in _COVER_KEYS and _fin(v)]
    shares += [x for p, v in res.items() if p.rsplit(".", 1)[-1] == "covered"
               and isinstance(v, list) for x in v if _fin(x)]
    yrs = res.get("support.years") or res.get("normal_years")
    return {"years": yrs, "area_share": min(shares) if shares else None,
            "n": res.get("support.n")}


def _present(v) -> bool:
    return (isinstance(v, list) and len(v) > 0) or (_fin(v) and v > 0)


def _warn(rec, code, reason):
    """One warning of a record: its code, the words that carry it in a claim
    (its label, from WARNINGS: an unregistered code raises) and its reason."""
    return {"code": code, "label": warning_label(code), "reason": reason, "record": rec["id"]}


def record_warnings(rec: Dict[str, Any], res: Dict[str, Any], notes=None) -> List[Dict[str, Any]]:
    """Every warning a record's value carries, each with its code, the words
    that carry it in a claim (label) and its reason: coverage, eligibility,
    restricted comparability, every reason code and flag, near-constant,
    insufficient support and not scored yet, provisional settings, found by
    screening or not planned, and the FPG configuration note."""
    AP, DCAT, M = _elm_modules()
    W: List[Dict[str, Any]] = []
    # 1. the area share covered, missing cell values, missing time records
    low = {p: v for p, v in res.items() if p.rsplit(".", 1)[-1] in _COVER_KEYS and _fin(v)
           and v < 1 - 1e-9}
    low.update({p: min(x for x in v if _fin(x)) for p, v in res.items()
                if p.rsplit(".", 1)[-1] == "covered" and isinstance(v, list)
                and any(_fin(x) and x < 1 - 1e-9 for x in v)})
    gone = {p: v for p, v in res.items() if ("missing_cell_values" in p
                                             or "missing_time_records" in p) and _present(v)}
    left = {p: v for p, v in res.items()
            if (p.startswith("normal_left_out.") and v not in (None, "", []))
            or (p.rsplit(".", 1)[-1] in ("normal_missing", "normal_incomplete") and _present(v))}
    if left:
        W.append(_warn(rec, "normal_left_out", "; ".join(
            f"{p}: {v}" for p, v in sorted(left.items()))))
    if low or gone:
        W.append(_warn(rec, "coverage", "; ".join(
            [f"area share covered {v:.3g} ({p})" for p, v in sorted(low.items())]
            + [f"{p}: {v}" for p, v in sorted(gone.items())])))
    # timing: the default variant's own; its alternatives are computed beside
    pref = "variants.default." if any(p.startswith("variants.default.") for p in res) else ""
    mine = {p: v for p, v in res.items() if not pref or p.startswith(pref)
            or p == "comparability"}
    # 2. the change of eligibility beside a shift
    el = {p[len(pref):]: v for p, v in mine.items() if ".shift.eligible." in "." + p[len(pref):]}
    # the groups of cells only: "region_area_km2" beside them is no group
    # (it was worded "region area km2 None cells"; 2026-11-09)
    parts = sorted({p.split("eligible.", 1)[1].split(".")[0] for p in el
                    if p.endswith(".cells")})
    if any(q != "both" and _fin(el.get(f"shift.eligible.{q}.cells")) and
           el[f"shift.eligible.{q}.cells"] > 0 for q in parts):
        W.append(_warn(rec, "eligibility", "the shift is on the cells eligible in "
                       "both; " + "; ".join(
                           f"{q.replace('_', ' ')} {el.get(f'shift.eligible.{q}.cells')} cells "
                           f"(area share {el.get(f'shift.eligible.{q}.area_share', 0):.3g})"
                           for q in parts)
                       + (f"; the shift covers area share {mine.get(pref + 'shift.area_share'):.3g}"
                          if _fin(mine.get(pref + "shift.area_share")) else "")))
    # 3. restricted comparability (a partial-record estimate in a shift)
    rc = [v for p, v in res.items() if p.endswith("comparability") and isinstance(v, str)
          and "restricted comparability" in v]
    if rc:
        W.append(_warn(rec, "restricted_comparability", rc[0]))
    # 4. every reason code and quality flag
    for p, v in sorted(mine.items()):
        m = re.search(r"(?:^|\.)reasons\.(\w+)$", p)
        if m and _fin(v) and v > 0 and m.group(1) in M.REASONS:
            text = M.REASONS[m.group(1)][1]
            W.append(_warn(rec, f"reason:{m.group(1)}", f"{int(v)} cells: {text}"))
        f = re.search(r"(?:^|\.)distribution\.flags\.(\w+)$", p)
        if f and _fin(v) and v > 0 and f.group(1) in M.FLAGS:
            text = M.FLAGS[f.group(1)][1]
            W.append(_warn(rec, f"flag:{f.group(1)}", f"{int(v)} cells: {text}"))
    # 5. near-constant, insufficient support, not scored yet (outside a screen row)
    if rec["call"] != "screen":
        for p, v in res.items():
            if p.rsplit(".", 1)[-1] == "near_constant" and v is True:
                W.append(_warn(rec, "near_constant", f"{p}: near-constant"))
            if p.rsplit(".", 1)[-1] == "status" and v in _SCREEN_STATUS:
                W.append(_warn(rec, _SCREEN_STATUS[v][0], f"{p}: {v}"))
    # 6. provisional settings (growth_peak's near and dip)
    mid = rec.get("method") or rec.get("measure")
    if mid in M.METHODS:
        prov = [s for s in M.METHODS[mid].settings if s.status == M.PROVISIONAL]
        if prov:
            W.append(_warn(rec, "provisional_settings",
                           f"{mid}'s settings {', '.join(s.name for s in prov)} are provisional: "
                           f"{(prov[0].review or {}).get('why')}"))
    # 7. found by screening, not tested; made after the planned phase was
    # closed (the user's decision on G part 2, 2026-11-08: exploratory, with
    # its own reason); not planned; the exploratory screen
    if rec.get("label") == AP.SCREENING_LABEL:
        W.append(_warn(rec, "found_by_screening",
                       "turned up by the screen, after the plan's lookups: not tested"))
    elif rec.get("after_close"):
        W.append(_warn(rec, "exploratory", AFTER_CLOSE_REASON))
    elif rec.get("label") == AP.EXPLORATORY_LABEL:
        W.append(_warn(rec, "exploratory",
                       "the exploratory question's screen: found by looking, not stated first"))
    if rec.get("amendment") or (rec.get("planned") is False and not rec.get("label")):
        W.append(_warn(rec, "not_planned",
                       f"amendment {rec.get('amendment')}: added after outcomes were seen, so "
                       f"never planned" if rec.get("amendment") else "no analysis of the plan "
                       "declared it"))
    # 8. the settings reverse the direction of the shift (the user's decisions
    # on G part 1, 2026-11-08): one setting gives an earlier date, another a
    # later one, on the cells eligible under every setting
    a = across_record(res)
    if a and a["direction_reversal"]:
        lo, hi = a.get("minimum") or {}, a.get("maximum") or {}
        W.append(_warn(rec, "settings_reverse_direction",
                       f"across the settings the shift goes from "
                       f"{M.format_value(lo.get('median_shift'))} days ({lo.get('variant')}) to "
                       f"{M.format_value(hi.get('median_shift'))} days ({hi.get('variant')}) on "
                       f"the {a.get('comparable_cells')} cells eligible under every setting: "
                       f"some settings give an earlier date, others a later one"))
    # 9. the FPG configuration note
    if rec.get("field") == "FPG":
        note = (notes or DCAT.load_notes()).get("FPG") or {}
        W.append(_warn(rec, "fpg_configuration",
                       " ".join(str(c) for c in note.get("cautions") or [])
                       + " See the catalog's line on the control's verified configuration."))
    return W


# THE RANGE ACROSS SETTINGS (the user's decisions on G part 1, 2026-11-08). A
# timing record gives, beside each setting's own shift, every setting's shift
# on the comparable cells (those eligible under every setting, in the year and
# in its baseline), the default, the smallest and the largest with their
# settings, and two flags: the largest minus the smallest exceeds 7 days
# (range_required) and the settings reverse the direction of the shift. A
# claim that gives a shift of a record whose range is required states the
# default, the smallest and the largest, each with its setting; a claim on a
# record whose settings reverse the direction carries that warning.
ACROSS = "across_settings."
REVERSAL_LABEL = "settings reverse the direction"
_SHIFT_VALUE = re.compile(r"^(?:variants\..+\.shift\.(?:median|q25|q75)|"
                          r"sensitivity\.\d+\.(?:default_)?median_shift|"
                          r"across_settings\..*median_shift)$")
_RANGE_PART_WORDS = {"default": "the default", "minimum": "the smallest",
                     "maximum": "the largest"}


# ═════════════════════════════════════════════════════════════════════════
# THE WARNINGS AND THE CLAIMS, DEFINED ONCE (the user's decision on G part 2,
# 2026-11-08)
# ═════════════════════════════════════════════════════════════════════════
# Question 1 of G part 2 wrote six claims, none naming a record or a class:
# the trace above reads `evidence`, `evidence_class`, `expectation`,
# `warnings` and `headline`, while the reply template step 3's model read
# listed only `claim`, `finding_id`, `values` and `caveats`. Both now come
# from one definition: CLAIM_PARTS and REPLY_PARTS say what a reply holds,
# step 3's template is rendered from them (claims_reply_lines), and WARNINGS
# is the one registry record_warnings, row_warnings and _sensitivity_rows
# take their labels from (an unregistered code raises), rendered into the
# same template (warning_lines).
AFTER_CLOSE_REASON = ("made after the planned analyses were closed (04_analysis/"
                      "planned_phase.json): found by looking, not stated first, so never a "
                      "tested expectation and never a hypothesis headline")
WARNINGS: Dict[str, Tuple[str, str]] = {
    "normal_left_out": ("left out of the normal", "a normal year lacks the period, or part "
                        "of it, so the normal is made without it"),
    "coverage": ("area share covered", "the method's inputs cover less than the whole region, "
                 "or cell values or time records are missing"),
    "eligibility": ("eligible", "a timing shift is on the cells eligible in both the year and "
                    "its baseline (or under both settings); some cells are eligible in one only, "
                    "or in neither"),
    "restricted_comparability": ("restricted comparability", "a partial-record estimate is in "
                                 "a shift: the water year starts before the run's records"),
    "near_constant": ("near-constant", "the normal years barely vary, so the change is given "
                      "with no score"),
    "insufficient_support": ("insufficient support", "too few normal years hold the period to "
                             "score it"),
    "not_scored_yet": ("not scored", "a field the screen does not score yet (a store, a "
                       "bounded index)"),
    "event_mark": ("marked", "a screen row marked for an event, such as where a run differs "
                   "from the control"),
    "provisional_settings": ("provisional", "the method's settings include provisional values, "
                             "approved for integration only"),
    "found_by_screening": (SCREENING_LABEL, "turned up by a hypothesis question's screen, after "
                           "its planned analyses: never a tested expectation"),
    "exploratory": (EXPLORATORY_LABEL, "found by looking, not stated first: an exploratory "
                    "question's screen, or any lookup made after the planned analyses were "
                    "closed"),
    "not_planned": ("not planned", "no analysis of the reviewed plan declared it, or an "
                    "amendment added it after outcomes were seen"),
    "settings_reverse_direction": (REVERSAL_LABEL, "across the method's settings some give an "
                                   "earlier date and others a later one"),
    "fpg_configuration": ("configuration", "FPG measures nitrogen limitation only, in the "
                          "control's verified configuration"),
}
# codes with a part after the colon, its vocabulary read from domain_methods
PARAMETERISED_WARNINGS = {
    "reason": ("the reason's own words", "a cell outcome of a timing measure other than a "
               "date, with its count of cells"),
    "flag": ("the flag's name in words", "a quality flag beside a timing date, with its "
             "count of cells")}


def warning_vocabulary(kind: str) -> List[str]:
    """The parts a parameterised code takes: the timing reasons (domain_methods.
    REASONS, apart from "date", the outcome they stand beside) or the flags."""
    M = _elm_modules()[2]
    if kind == "reason":
        return [k for k in M.REASONS if k != "date"]
    if kind == "flag":
        return list(M.FLAGS)
    raise KeyError(kind)


def warning_label(code: str) -> str:
    """The words that carry a warning in a claim, from WARNINGS (or, for
    reason:<x> and flag:<x>, from domain_methods). An unregistered code
    raises KeyError: a warning the registry does not hold cannot be made."""
    kind, _, x = str(code).partition(":")
    if x:
        if kind not in PARAMETERISED_WARNINGS or x not in warning_vocabulary(kind):
            raise KeyError(code)
        M = _elm_modules()[2]
        return _short_reason(M.REASONS[x][1]) if kind == "reason" else x.replace("_", " ")
    return WARNINGS[code][0]


def warning_lines() -> List[str]:
    """The registry as step 3's template shows it: every code, its words in
    quotes and its meaning."""
    L = ["THE WARNINGS a record can carry. A claim carries every warning of the records it",
         "names: the code in \"warnings\", or the words in quotes in its own sentence.",
         "Under each finding, its records list theirs."]
    for code, (label, means) in WARNINGS.items():
        L += _wrap(f"{code} (\"{label}\"): {means}", "  ", "      ")
    for kind, (words, means) in PARAMETERISED_WARNINGS.items():
        L += _wrap(f"{kind}:<x> ({words}): {means}; x is one of: "
                   + ", ".join(warning_vocabulary(kind)), "  ", "      ")
    return L


def _wrap(text: str, first: str, rest: str) -> List[str]:
    import textwrap
    return textwrap.wrap(text, 79, initial_indent=first, subsequent_indent=rest,
                         break_long_words=False, break_on_hyphens=False)


@dataclass(frozen=True)
class ReplyPart:
    """One part of step 3's reply: its name, how the template shows it, and
    what it means (shown in the template's rules)."""
    name: str
    shown: str
    means: str


CLAIM_PARTS = (
    ReplyPart("claim", '"<one sentence>"', "the claim, one sentence"),
    ReplyPart("finding_id", '"<the finding it rests on>"', "the finding the claim rests on"),
    ReplyPart("values", "[<every MEASURED number this claim asserts, copied exactly from the "
              "finding's result; NOT counts you made, NOT labels, NOT thresholds you chose; [] "
              "if none>]", "each measured number it states, unrounded"),
    ReplyPart("caveats", '["<ids of any blocking caveat this claim falls under>"]',
              "the blocking caveats it falls under"),
    ReplyPart("evidence", '["L7"]', "the evidence records it rests on, by id, as listed under "
              "its finding; a claim with none fails"),
    ReplyPart("evidence_class", " | ".join(f'"{c}"' for c in EVIDENCE_CLASSES),
              "its evidence class, one of: " + ", ".join(EVIDENCE_CLASSES)),
    ReplyPart("expectation", '"E1" | null', "for a tested expectation, the plan's expectation "
              "it tests"),
    ReplyPart("warnings", '["<every warning code its records carry>"]',
              "every warning its records carry, by code; a claim that leaves one out fails"),
)
REPLY_PARTS = (
    ReplyPart("claims", "[<the claims>]", "the claims"),
    # THE HEADLINE IS A VERIFIED CLAIM'S OWN TEXT (the user's decision of
    # 2026-11-08): step 3 ranks its claims, and the report's code takes the
    # first that traced and that the question type allows (select_headline)
    ReplyPart("headline_rank", "[<every claim's number once, 1 for the first claim; the one "
              "that answers the question most directly first>]",
              "the claims ranked for the headline: the report's headline is the text of the "
              "first of them that traces and that the question type allows"),
    ReplyPart("answer", '"<2-4 plain sentences: what was found, why, what it does not show; or '
              'saying plainly that this experiment cannot answer it and why>"', "the answer"),
    ReplyPart("verdict", '"sufficient" | "insufficient"', "the verdict"),
    ReplyPart("feedback", '"<if insufficient: what step 2 should do differently; empty if '
              'sufficient>"', "what step 2 should do differently"),
    ReplyPart("open_causes", '[{"cause": "...", "proposed_run": {"changes": [{"id": "<a change '
              'of the server\'s list>", "params": {...}}]}}]', "when a cause stays open, the "
              "run that would settle it, in the server's change vocabulary"),
)
OPEN_CAUSE_PARTS = ("cause", "proposed_run", "changes", "id", "params")


def claims_reply_lines() -> List[str]:
    """Step 3's reply template for a study with a suite, rendered from
    CLAIM_PARTS and REPLY_PARTS: every part check_claim, trace_evidence,
    the audit and interpret read."""
    L = ["Return ONLY JSON:", '{"claims": [']
    for i, p in enumerate(CLAIM_PARTS):
        lead = "    {" if i == 0 else "     "
        tail = "}" if i == len(CLAIM_PARTS) - 1 else ","
        L += _wrap(f'"{p.name}": {p.shown}{tail}', lead, "        ")
    L.append("  ],")
    rest = [p for p in REPLY_PARTS if p.name != "claims"]
    for i, p in enumerate(rest):
        L += _wrap(f'"{p.name}": {p.shown}' + ("}" if i == len(rest) - 1 else ","),
                   "  ", "      ")
    return L


def across_record(res: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A timing record's range across settings, read from its flat result
    (None when it has none): the flags, the spread and its limit, the
    comparable cells, and the default, the smallest and the largest shift,
    each with its setting."""
    if not any(str(p).startswith(ACROSS) for p in res):
        return None
    g = lambda k: res.get(ACROSS + k)                               # noqa: E731
    out: Dict[str, Any] = {
        "range_required": g("range_required") is True,
        "direction_reversal": g("direction_reversal") is True,
        "spread_days": g("spread_days"), "limit_days": g("limit_days"),
        "comparable_cells": g("comparable.cells"),
        "comparable_area_share": g("comparable.area_share"), "why": g("why"),
        "evaluated": [res[p] for p in sorted(
            (p for p in res if re.fullmatch(r"across_settings\.evaluated\.\d+\.variant", p)),
            key=lambda p: int(p.split(".")[2]))]}
    for part in _RANGE_PART_WORDS:
        if g(f"{part}.variant") is not None:
            out[part] = {"variant": g(f"{part}.variant"), "median_shift": g(f"{part}.median_shift")}
    return out


def _setting_named(variant: str, low: str, nums: List[float]) -> bool:
    """Whether a claim names a setting: the default by the word "default";
    an alternative by its label ("fraction=0.3") or by its setting's first
    word and its value ("fraction 0.3")."""
    if variant == "default":
        return "default" in low
    if variant.lower() in low:
        return True
    name, _, val = variant.partition("=")
    try:
        x = float(val)
    except ValueError:
        return val.lower() in low
    return name.split("_")[0].lower() in low and \
        any(abs(n - x) <= 1e-12 * max(1.0, abs(x)) for n in nums)


def range_problems(text: str, r: Dict[str, Any], paths: List[str]) -> List[str]:
    """The range rule for one record: a claim giving a shift of a record
    whose range across settings exceeds its limit states the default, the
    smallest and the largest shift, each with its setting."""
    a = r.get("across_settings")
    if not a or not a.get("range_required") or not any(_SHIFT_VALUE.match(p) for p in paths):
        return []
    low = str(text or "").lower()
    words = _number_words(text)
    nums = [float(w) for w, _, _ in words]
    lack = []
    for part, said in _RANGE_PART_WORDS.items():
        e = a.get(part) or {}
        x = e.get("median_shift")
        if not _fin(x):
            continue
        stated = any(abs(abs(float(w)) - abs(x)) <= _half_unit(w) + 1e-9 for w, _, _ in words)
        if not (stated and _setting_named(str(e.get("variant")), low, nums)):
            lack.append(f"{said}, {x:.3g} days at {e.get('variant')}")
    if not lack:
        return []
    return [f"its record {r['id']}'s shift moves by {a['spread_days']:.3g} days across the "
            f"settings, more than {a['limit_days']:g}: a claim giving the shift states the "
            f"default, the smallest and the largest shift, each with its setting; it lacks "
            + "; ".join(lack)]


def _screen_rows(res: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The screen's rows by their path ("rows.3", "display.groups.snow.0",
    "display.not_scored.1", "display.events.0")."""
    rows: Dict[str, Dict[str, Any]] = {}
    for p, v in res.items():
        m = re.match(r"^(rows\.\d+|display\.(?:groups\.[^.]+|not_scored|events)\.\d+)\.(\w+)$", p)
        if m:
            rows.setdefault(m.group(1), {})[m.group(2)] = v
    return rows


def row_warnings(rec: Dict[str, Any], row: Dict[str, Any], where: str) -> List[Dict[str, Any]]:
    """A screen row's own warnings: its status when it has no score, a
    near-constant normal, not scored, and its event mark."""
    W = []
    st = row.get("status")
    if st in _SCREEN_STATUS:
        W.append(_warn(rec, _SCREEN_STATUS[st][0], f"{row.get('field')} {row.get('period')}: "
                                                   f"{st}"))
    if row.get("near_constant") is True and st != "near-constant normal":
        W.append(_warn(rec, "near_constant",
                       f"{row.get('field')} {row.get('period')}: near-constant normal"))
    if row.get("label") == "not scored" and st != "not scored yet":
        W.append(_warn(rec, "not_scored_yet",
                       f"{row.get('field')} {row.get('period')}: not scored yet"))
    if row.get("mark") not in (None, ""):
        W.append(_warn(rec, "event_mark",
                       f"{row.get('field')} {row.get('period')}: marked {row.get('mark')}"))
    if row.get("field") == "FPG":
        W.append(_warn(rec, "fpg_configuration",
                       "FPG measures nitrogen limitation only; see the field note and the "
                       "catalog's line on the control's verified configuration"))
    return W


def _sensitivity_rows(rec: Dict[str, Any], res: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """A timing record's comparisons between settings ("sensitivity.0"): the
    change of date is on the cells with a date under both settings, and beside
    it stands the change of eligibility, a warning when any cell has a date
    under one setting only or neither."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for p, v in res.items():
        m = re.match(r"^(sensitivity\.\d+)\.eligible\.(\w+)\.cells$", p)
        if m and m.group(2) != "both" and _fin(v) and v > 0:
            parts = sorted({q.split(".")[3] for q in res
                            if q.startswith(m.group(1) + ".eligible.") and q.endswith(".cells")})
            out[m.group(1)] = [_warn(rec, "eligibility", "the change of date between "
                                     "settings is on the cells dated under both; " + "; ".join(
                                         f"{q.replace('_', ' ')} "
                                         f"{res.get(m.group(1) + '.eligible.' + q + '.cells')} "
                                         f"cells" for q in parts))]
    return out


# FULL DERIVED-RESULT PROVENANCE (the user's decision of 2026-11-08): an
# overlap or an elevation split names its maps only by their hashes, so its
# record named no year and no season, and question 2's "the snow and
# carbon-uptake changes of 2022" and question 3's "JJA" failed although their
# maps were 2022's and June-August's. Each input is now found, by its hash,
# among the answered lookups before it (as the numbers check finds it), and
# the record names that input's record, call, field, period and region. The
# record inherits its inputs' years and seasons, and an elevation split its
# input's field (the field of its parts' values, for its units and percent
# rule); never a region: its own region and parts are all a claim may name.
_DERIVED = {"overlap": ("a", "b"), "elevation_split": ("x",)}


def derived_inputs(log: List[Dict[str, Any]], i: int) -> List[Dict[str, Any]]:
    """The records the overlap or elevation split on line i was made from,
    in the order the call took them; a map no logged lookup returned (one a
    script made by its own arithmetic) is named as such."""
    ln = log[i - 1]
    keys = ln.get("keys") or {}
    out = []
    for role in _DERIVED.get(ln.get("call"), ()):
        sha, src = keys.get(role), None
        for j in range(i - 1, 0, -1):
            s_ = log[j - 1]
            if "sha256" not in s_ or s_.get("refused") or s_.get("failed"):
                continue
            if s_.get("call") == "get_map" and s_.get("sha256") == sha:
                src = (j, None)
                break
            hit = next((p for p, h in (s_.get("arrays") or {}).items()
                        if h == sha and s_.get("call") in ("anomaly", "difference")), None)
            if hit:
                src = (j, hit)
                break
        if src is None:
            out.append({"role": role, "record": None,
                        "why": "no logged lookup returned this map: a map a script made by "
                               "its own arithmetic"})
            continue
        j, path = src
        k = log[j - 1].get("keys") or {}
        out.append({"role": role, "record": f"L{j}", "call": log[j - 1].get("call"),
                    "path": path, "field": k.get("field"),
                    "period": None if k.get("period") is None else str(k.get("period")),
                    "region": k.get("region"), "run": k.get("run") or "control"})
    return out


# A QUANTITY THAT DEFINES A FIELD IS NO VALUE (2026-11-08, after the third
# batch, "typed quantity rules"): question 2's "summer water in the top 10 cm
# of soil" failed because "10 cm" was read as a measured number. It is the
# depth that defines SOILWATER_10CM ("liquid water and ice in the top 10 cm
# of soil"). A field's defining quantities, a length or depth with its unit,
# are read from its note's meaning and its code ("_10CM"), and a claim's
# number with that same unit is the field's label; a "10 mm" or a "10 cm"
# about another field is still a value to trace.
_LENGTH_UNITS = ("mm", "cm", "m", "km")


def field_quantities(field: Optional[str], notes: Dict[str, Any]) -> List[Tuple[float, str]]:
    """(value, unit) the definition of a field gives: its code's trailing
    length ("TSOI_10CM": 10 cm) and each length its note's meaning states."""
    out: List[Tuple[float, str]] = []
    if not field:
        return out
    m = re.search(r"_(\d+(?:\.\d+)?)(MM|CM|KM|M)$", str(field))
    if m:
        out.append((float(m.group(1)), m.group(2).lower()))
    for w, unit, _ in _number_words(str((notes.get(field) or {}).get("meaning") or "")):
        if unit in _LENGTH_UNITS and (float(w), unit) not in out:
            out.append((float(w), unit))
    return out


# ... AND ONLY WHERE THE CLAIM WRITES IT AS THE FIELD'S DEFINITION DOES (the
# user's decision of 2026-11-08, closing the false acceptance): "the top 10 cm
# of soil" names SOILWATER_10CM's layer, but "the soil water rose 10 cm" states
# an amount and must match a value. A defining length is its field's label
# only after the word the note's meaning puts before it ("top 10 cm", "at 2
# m"), before the "of" phrase that follows it there ("10 cm of soil"), or
# directly before a word the definition gives the field ("10 cm soil
# temperature", "2 m air temperature", "10-cm soil water").
_LENGTH_IN = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*-?\s*(mm|cm|km|m)\b")
_LABEL_STOP = frozenset(("the", "a", "an", "of", "in", "at", "and", "or", "from", "to", "on",
                         "for", "with", "by", "only", "each", "its", "per", "as")
                        + _LENGTH_UNITS)


def _words(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r"[A-Za-z]+", str(text or ""))]


def field_labels(field: Optional[str], notes: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Each length that defines a field (field_quantities), with what makes a
    claim's mention of it the field's label: the word before it in the note's
    meaning ("before"), the "of" phrase after it ("after"), and the words of
    the definition a length may modify ("words": the first word of the
    field's plain name and the content words after the length)."""
    meaning = str(((notes or {}).get(field) or {}).get("meaning") or "") if field else ""
    head = _words(_plain_name(meaning))[:1]
    out = []
    for value, unit in field_quantities(field, notes or {}):
        before, after, words = set(), set(), set(head) - _LABEL_STOP
        for m in _LENGTH_IN.finditer(meaning):
            if float(m.group(1)) != value or m.group(2) != unit:
                continue
            pre, post = _words(meaning[:m.start()]), _words(meaning[m.end():])
            if pre:
                before.add(pre[-1])
            if len(post) >= 2 and post[0] == "of":
                after.add(f"of {post[1]}")
            words |= set(post) - _LABEL_STOP
        out.append({"value": value, "unit": unit, "before": sorted(before),
                    "after": sorted(after), "words": sorted(words)})
    return out


def written_as_label(text: str, end: int, x: float, lab: Dict[str, Any]) -> bool:
    """Whether the number ending at `end` in `text` is `lab`'s defining
    length (field_labels), written as its field's definition writes it."""
    t = str(text or "")
    m = re.match(r"\s*-?\s*(mm|cm|km|m)\b", t[end:])
    if abs(x) != lab.get("value") or not m or m.group(1) != lab.get("unit"):
        return False
    j = end
    while j > 0 and (t[j - 1].isdigit() or t[j - 1] in ".,"):
        j -= 1
    pre, post = _words(t[:j]), _words(t[end + m.end():])
    return bool(pre and pre[-1] in (lab.get("before") or ())) or \
        bool(len(post) >= 2 and f"{post[0]} {post[1]}" in (lab.get("after") or ())) or \
        bool(post and post[0] in (lab.get("words") or ()))


def evidence_records(analysis_dir, notes=None) -> Dict[str, Dict[str, Any]]:
    """Every answered lookup of 04_analysis/lookups.jsonl as an evidence
    record, by its line ("L1" is the first line). Refused or failed calls and
    the four calls that read no outcome are not evidence."""
    AP, DCAT, M = _elm_modules()
    notes = notes if notes is not None else DCAT.load_notes()
    out: Dict[str, Dict[str, Any]] = {}
    # MADE AFTER THE CLOSE (the user's decision on G part 2, 2026-11-08): by
    # the gate's mark, or by its line, after the lines the close counted, so
    # a line logged without the mark is still read as after the close
    phase = AP.load_planned_phase(analysis_dir)
    before = phase.get("lookups_before") if phase else None
    log = _read_log(analysis_dir)
    for i, ln in enumerate(log, 1):
        if "sha256" not in ln or ln.get("support") or ln.get("refused") or ln.get("failed"):
            continue
        k = ln.get("keys") or {}
        res = ln.get("result") or {}
        call = ln.get("call")
        measure = k.get("measure") or (call if call in M.TIMING_IDS else None)
        period = k.get("period") if k.get("period") is not None else k.get("year")
        rec = {"id": f"L{i}", "lookup": i, "sha256": ln["sha256"], "at": ln.get("at"),
               "script": ln.get("script"), "call": call, "method": ln.get("method"),
               "measure": measure, "keys": k, "settings": ln.get("settings"),
               "run": k.get("run") or "control", "field": k.get("field") or res.get("field"),
               "period": None if period is None else str(period),
               "region": k.get("region") or res.get("region"), "step": k.get("step"),
               "planned": ln.get("planned"), "plan_items": ln.get("plan_items") or [],
               "amendment": ln.get("amendment"), "label": ln.get("label"),
               "after_screen": ln.get("after_screen"),
               "after_close": ln.get("after_close") is True or (
                   isinstance(before, int) and i > before),
               "values": {p: v for p, v in res.items() if _fin(v)},
               "words": {p: v for p, v in res.items() if isinstance(v, str)}}
        if call in _DERIVED:
            rec["inputs"] = derived_inputs(log, i)
            rec["fields"] = [x["field"] for x in rec["inputs"] if x.get("field")]
            if call == "elevation_split" and not rec.get("field") and rec["fields"]:
                rec["field"] = rec["fields"][0]
        rec["comparison"] = _comparison(rec, res)
        rec["support"] = _support(res)
        rec["across_settings"] = across_record(res) if measure else None
        if call == "screen":
            rec["screen_csv"] = ln.get("screen_csv")       # the whole table, beside the log
        rec["warnings"] = record_warnings(rec, res, notes)
        rows = _screen_rows(res) if call == "screen" else {}
        rec["rows"] = rows
        rec["row_warnings"] = {w: row_warnings(rec, r, w) for w, r in rows.items()}
        if measure:
            rec["row_warnings"].update(_sensitivity_rows(rec, res))
        nm = _names(rec, res)
        for x in rec.get("inputs") or []:             # a derived record's inputs' periods
            if x.get("period"):
                sub = _names({"period": x["period"], "step": None}, {})
                nm["years"] |= sub["years"]
                nm["periods"] |= sub["periods"]
        rec["names"] = {"years": sorted(nm["years"]), "periods": sorted(nm["periods"])}
        pc = (notes.get(rec["field"]) or {}).get("percent") if rec.get("field") else None
        rec["field_quantities"] = sorted({q for f in [rec.get("field")] + list(
            rec.get("fields") or []) for q in field_quantities(f, notes)})
        rec["field_labels"] = [lab for f in dict.fromkeys([rec.get("field")] + list(
            rec.get("fields") or [])) if f for lab in field_labels(f, notes)]
        rec["percent_allowed"] = None if not isinstance(pc, dict) else bool(pc.get("allowed"))
        rec["percent_why_not"] = None if not isinstance(pc, dict) else pc.get("why")
        out[rec["id"]] = rec
    return out


def lookups_since(analysis_dir, n_before: int) -> List[Dict[str, Any]]:
    """The evidence records of the lookups logged after line n_before (one
    figure's), compactly, for the finding step 3 reads: the id a claim names,
    the call and keys, planned or not, its label, and its warnings."""
    recs = evidence_records(analysis_dir)
    return [{"id": r["id"], "call": r["call"], "method": r.get("method"),
             "keys": r.get("keys"), "planned": r.get("planned"), "label": r.get("label"),
             "amendment": r.get("amendment"), "after_close": r.get("after_close"),
             "warnings": [{"code": w["code"], "label": w["label"]} for w in r["warnings"]],
             "row_warnings": {k: [w["code"] for w in v]
                              for k, v in r["row_warnings"].items() if v}}
            for r in recs.values() if r["lookup"] > n_before]


def study_runs(study_dir) -> Dict[str, Dict[str, Any]]:
    """Each experiment run of a study: its changes in the server's words and
    whether the server's checks confirm it differs from the control by them
    alone (every stage of its chain checked and passed: the case's settings
    against the control's, the smoke run, each year's start from the
    control's state)."""
    import json
    from pathlib import Path
    p = Path(str(study_dir)) / "experiment.json"
    e = json.loads(p.read_text()) if p.is_file() else {}
    chain = (e.get("chain") or {}).get("runs") or {}
    out = {}
    for r in (e.get("experiment") or {}).get("runs") or []:
        name = str((r or {}).get("run") or (r or {}).get("name") or "")
        if not name or name == "control":
            continue
        st = [s for s in ((chain.get(name) or {}).get("stages") or [])
              if s.get("check_pass") is not None]
        ok = bool(st) and all(s.get("check_pass") is True for s in st) and \
            any(s.get("stage") == "case" for s in st)
        out[name] = {"changes": [c for c in r.get("changes") or [] if isinstance(c, dict)],
                     "verified": ok,
                     "why": ("the server's checks of the case, the smoke run and every year "
                             "passed" if ok else "the server's checks do not confirm that "
                             "nothing else differs (a stage unchecked, failed or missing)")}
    return out


def _region_heads(text: str, regions) -> Tuple[set, set]:
    """The regions a text names: their heads ('watershed') and the bands it
    names by both edges ('watershed, 1500-2000 m')."""
    low = str(text or "").lower()
    nums = {float(w.replace(",", "")) for w in _NUMBER.findall(str(text or ""))}
    heads, bands = set(), set()
    for r in regions or []:
        h, b = _head_band(r)
        named = (any(w in low for w in _ALL_CELLS) if h == "vegetated cells"
                 else _short(h).replace("_", " ").lower() in low)
        if not named:
            continue
        if b:
            if all(e in nums for e in _band_edges(b)):
                bands.add(r)
        else:
            heads.add(_short(h))
    return heads, bands


def _label_numbers(recs, runs, plan) -> set:
    """Numbers a claim may print as labels, not values: its records' keys and
    settings, years and counts of the comparison, the settings of the runs its
    records read (the '2 K' of a warming), and the band edges of regions."""
    out = set()

    def add(x):
        if isinstance(x, dict):
            for v in x.values():
                add(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                add(v)
        elif _fin(x):
            out.add(float(x))
        elif isinstance(x, str):
            out.update(float(w.replace(",", "")) for w in _NUMBER.findall(x))
    for r in recs:
        add(r.get("keys"))
        add(r.get("settings"))
        add(r.get("comparison"))
        for p, v in r["values"].items():
            if p.rsplit(".", 1)[-1] in _LABEL_KEYS or ".settings." in "." + p:
                add(v)
        for run in {r.get("run"), (r.get("keys") or {}).get("run")} - {None, "control"}:
            add([c.get("params") for c in (runs.get(run) or {}).get("changes") or []])
        add([part for kind, part in _split_parts(r) if kind != "other_regions"])  # band edges
    return out


def _split_parts(r: Dict[str, Any]) -> List[Tuple[str, str]]:
    """(kind, name) of each part a record's result is split into: an
    elevation split's bands (parts) and other regions, an overlap's bands."""
    out = set()
    for p in list(r.get("values") or {}) + list(r.get("words") or {}):
        m = re.match(r"(parts|by_band|other_regions)\.([^.]+)\.", p)
        if m:
            out.add((m.group(1), m.group(2)))
    return sorted(out)


# A PERCENT OF A SHARE (the user's decision of 2026-11-08, "typed quantity
# rules"): step 3's own rule says a percent is a value times 100, and its
# audit accepts that, but the trace did not, so question 3's "55.6 percent"
# of a band share of 0.5557 failed. A percent figure may now be a record's
# value times 100 only when that value is a share (its path names one); no
# other value is read that way, and a share's percent is no percent change
# of a field, so the field's percent rule does not reach it.
_SHARE_PATH = re.compile(r"(?:^|\.)(?:\w*share|sign_shares\.\w+)$")


def share_matches(x: float, printed: Optional[str], recs) -> List[Tuple[Dict[str, Any], str]]:
    """The records' share values a printed percent states, times 100."""
    tol = _half_unit(printed) if printed else 0.0
    hits = []
    for r in recs:
        for p, v in r["values"].items():
            if _SHARE_PATH.search(p) and \
                    abs(abs(x) - abs(v) * 100.0) <= tol + 1e-9 * max(1.0, abs(v) * 100.0):
                hits.append((r, p))
    return hits


def _matches(v: float, printed: Optional[str], recs) -> List[Tuple[Dict[str, Any], str]]:
    tol = _half_unit(printed) if printed else 0.0
    hits = []
    for r in recs:
        for p, x in r["values"].items():
            if abs(abs(v) - abs(x)) <= tol + 1e-9 * max(1.0, abs(x)):
                hits.append((r, p))
    return hits


def _number_words(text: str) -> List[Tuple[str, Optional[str], int]]:
    """The numbers a text prints, with the unit after each; a unit's own
    exponent ('m^2', 'm-2', 's-1') is not a number."""
    t = re.sub(r"\^-?\d+|(?<=[A-Za-z])-\d+\b", lambda m: " " * len(m.group(0)),
               str(text or ""))
    return [(m.group(0).replace(",", ""), unit_after(t, m.end()), m.end())
            for m in _NUMBER.finditer(t)]


# A PERCENT ONLY WHERE THE FIELD'S NOTES ALLOW ONE, judged by one helper for
# the trace (check_claim) and for step 3's own audit (the user's decision on G
# part 2, 2026-11-08): question 1's audit kept "BTRAN 20.2 percent lower",
# a percent the field's notes give none, made by the script's own arithmetic.
def _plain_name(meaning: str) -> str:
    """A field's plain name, the head of its note's meaning: "relative
    humidity" of "relative humidity at 2 m, from the land model"."""
    return re.split(r":|,| \(| at | from | in each | in the top ", str(meaning or ""), 1)[0] \
        .strip().lower()


def field_mentions(text: str, notes: Dict[str, Any]) -> Dict[str, List[int]]:
    """{field: where in `text` each mention of it ends}: by its code
    ("BTRAN", case kept) or by its note's plain name ("relative humidity")."""
    t = str(text or "")
    low = t.lower()
    out: Dict[str, int] = {}
    for f, n in (notes or {}).items():
        hits = [m.end() for m in re.finditer(rf"\b{re.escape(f)}\b", t)]
        name = _plain_name((n or {}).get("meaning"))
        if len(name) > 2:
            hits += [m.end() for m in re.finditer(rf"\b{re.escape(name)}\b", low)]
        if hits:
            out[f] = hits
    return out


def percent_problems(text: str, records) -> List[str]:
    """The percent rule: a claim whose text gives a percent fails when a
    record it is judged on reads a field whose notes give no percent. One
    problem, naming the field the percent stands nearest after (by code or
    plain name), else the first such record's."""
    # a percent of a share (share_matches) is typed as a share, never a
    # percent change of a field: only the other percents meet the rule
    t = str(text or "")
    own = [m for m in _PERCENT.finditer(t)
           if not share_matches(float(re.match(r"-?\d+(?:\.\d+)?", m.group(0)).group(0)),
                                re.match(r"-?\d+(?:\.\d+)?", m.group(0)).group(0),
                                records or [])]
    if not own:
        return []
    bad = [r for r in records or [] if r.get("percent_allowed") is False]
    if not bad:
        return []
    at = own[0].start()
    pick = bad[0]
    try:
        DCAT = _elm_modules()[1]
        seen = field_mentions(str(text)[:at], DCAT.load_notes())
        named = [r for r in bad if r.get("field") in seen]
        if named:
            pick = max(named, key=lambda r: max(seen[r["field"]]))
    except Exception:                                               # noqa: BLE001
        pass
    # WHOSE PERCENT IT IS (2026-11-09): question 1's withheld claim gave
    # "summer leaf area was 0.0061 lower, a change of 0.52 percent" beside
    # "summer air was 1.73 K warmer", and its reason read "a percent of TSA",
    # though the percent was leaf area's. The rule still judges the claim
    # whole; the reason now names the field whose value the percent is, when
    # one of the claim's records holds it and that field allows one.
    of = percent_attribution(text, records)
    mine = [f for f in of.get(own[0].group(0), []) if f not in {r.get("field") for r in bad}]
    if mine:
        return [f"a percent in a claim that also gives {pick['field']}, whose notes give none "
                f"({pick.get('percent_why_not')}): its percent is of {', '.join(mine)}, "
                f"whose notes allow one; a claim naming {pick['field']} prints no percent, "
                f"so give each quantity a claim of its own"]
    return [f"a percent of {pick['field']}, whose notes give none: "
            f"{pick.get('percent_why_not')}"]


def percent_attribution(text: str, records) -> Dict[str, List[str]]:
    """{each percent as printed: the fields of the claim's records whose
    percent value it is (to the precision printed)}: whose quantity a
    percent is, read from the records themselves, not from the words."""
    out: Dict[str, List[str]] = {}
    for m in _PERCENT.finditer(str(text or "")):
        w = re.match(r"-?\d+(?:\.\d+)?", m.group(0)).group(0)
        x, tol = float(w), _half_unit(w)
        out[m.group(0)] = sorted({
            str(r.get("field")) for r in records or [] for p, v in (r.get("values") or {}).items()
            if p.rsplit(".", 1)[-1] == "percent" and _fin(v)
            and abs(abs(v) - abs(x)) <= tol + 1e-9 * max(1.0, abs(v)) and r.get("field")})
    return out


# THE SUMMARY HOLDS TO THE KEPT CLAIMS (the user's decision of 2026-11-09):
# question 1's summary ended "Warmer summer air, 1.73 K, was also present",
# a number only the claim the audit withheld gave. Each sentence of a summary
# is read as a claim is read (its numbers to the precision printed, years
# and labels aside) against the records of the claims the audit kept: a
# sentence with a number none of them gives is left out of the report's
# narrative, and the audit appendix says which and why.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def sentences(text: str) -> List[str]:
    """A summary's sentences (a decimal point is no sentence end)."""
    return [s.strip() for s in _SENTENCE.split(str(text or "").strip()) if s.strip()]


def summary_support(text: str, claims: List[Dict[str, Any]],
                    records: Dict[str, Dict[str, Any]], runs: Dict[str, Dict[str, Any]],
                    plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Each sentence of a summary: {"sentence", "claims" (the kept claims, by
    number from 1, whose records give its numbers), "unsupported" (the
    numbers as printed that no kept claim's record gives)}. `claims`: the
    kept claims, in the report's order."""
    recs_of = [[records[i] for i in (c.get("evidence") or []) if i in records] for c in claims]
    every = [r for rs in recs_of for r in rs]
    labels = _label_numbers(every, runs or {}, plan or {}) if every else set()
    out = []
    for s in sentences(text):
        years = {int(y) for r in every for y in (r.get("names") or {}).get("years") or []} | \
            {int(y) for y in years_named(s)}
        by, bad = set(), []
        for w, unit, end in _number_words(s):
            x = float(w)
            if _is_year(w.lstrip("-"), unit, years):
                continue
            tol = _half_unit(w) + 1e-9 * max(1.0, abs(x))
            # THE CLAIMS A NUMBER TIES ITS SENTENCE TO: those whose records give
            # it as the kind it is printed as (a percent at a percent's path,
            # or a share; any other number at any other path: "3.3 percent" is
            # not the 3.27 mm of another claim), else at any path; and among
            # them those that state it in their own values, when any does

            def pct(path):
                return path.rsplit(".", 1)[-1] == "percent"

            def claims_with(ok):
                return [i for i, rs in enumerate(recs_of, 1)
                        if any(ok(q) for _, q in _matches(x, w, rs))]
            if unit == "percent":
                tied = claims_with(pct) or [i for i, rs in enumerate(recs_of, 1)
                                            if share_matches(x, w, rs)] or claims_with(lambda q: True)
            else:
                tied = claims_with(lambda q: not pct(q)) or claims_with(lambda q: True)
            hit = [i for i in tied if any(_fin(v) and abs(abs(float(v)) - abs(x)) <= tol
                                          for v in claims[i - 1].get("values") or [])] or tied
            if hit:
                by.update(hit)
            elif not (abs(x) in labels or x in labels or any(
                    written_as_label(s, end, x, lab) for r in every
                    for lab in r.get("field_labels") or [])):
                raw = re.match(r"\s*([A-Za-z%][A-Za-z0-9%/^]*)", s[end:])
                bad.append(w + (f" {raw.group(1)}" if raw and unit else ""))
        out.append({"sentence": s, "claims": sorted(by), "unsupported": bad})
    return out


def check_claim(c: Dict[str, Any], records: Dict[str, Dict[str, Any]], plan: Dict[str, Any],
                runs: Dict[str, Dict[str, Any]], regions) -> Dict[str, Any]:
    """One claim against its evidence records: the problems, the warnings its
    records carry, and its class's wording."""
    AP = _elm_modules()[0]
    text = str(c.get("claim") or "")
    low = text.lower()
    probs: List[str] = []
    ids = [str(x) for x in (c.get("evidence") or [])]
    recs = [records[i] for i in ids if i in records]
    for i in ids:
        if i not in records:
            probs.append(f"no record: it names {i}, which is no answered lookup of "
                         f"{LOOKUPS_LOG}")
    if not ids:
        probs.append("no record: every empirical claim names the logged lookup it rests on")
    cls = c.get("evidence_class")
    out = {"records": ids, "class": cls, "warnings": [], "problems": probs,
           "wording_allowed": CLASS_WORDING.get(cls)}
    if not recs:
        return out
    # THE NUMBERS: each value it gives and each number its text prints is a
    # value of its records (to the precision printed), a year, or a label
    labels = _label_numbers(recs, runs, plan)
    years_all = {int(y) for r in recs for y in r["names"]["years"]} | \
        {int(y) for y in years_named(text)}
    matched: List[Tuple[Dict[str, Any], str]] = []
    for v in c.get("values") or []:
        if not _fin(v):
            continue
        if float(v).is_integer() and int(v) in years_all and _is_year_value(v, text, years_all):
            continue
        pr = next((w for w, _, _ in _number_words(text)
                   if abs(abs(float(w)) - abs(v)) <= _half_unit(w) + 1e-12), None)
        hits = _matches(float(v), pr, recs)
        if not hits and pr is not None and \
                any(w == pr and u == "percent" for w, u, _ in _number_words(text)):
            hits = share_matches(float(v), pr, recs)
        if not hits:
            probs.append(f"the number {v} matches no value of its records "
                         f"({', '.join(r['id'] for r in recs)})")
        matched += hits
    for w, unit, end in _number_words(text):
        x = float(w)
        if _is_year(w.lstrip("-"), unit, {int(y) for y in years_all}):
            continue
        hits = _matches(x, w, recs) or (share_matches(x, w, recs) if unit == "percent" else [])
        if hits:
            matched += hits
            continue
        if abs(x) in labels or x in labels or any(
                written_as_label(text, end, x, lab) for r in recs
                for lab in r.get("field_labels") or []):
            continue
        probs.append(f"the number {w}{(' ' + unit) if unit else ''} matches no value of its "
                     f"records ({', '.join(r['id'] for r in recs)})")
    # THE PERIODS AND REGIONS IT NAMES: only those its records name
    ny = set().union(*(set(r["names"]["years"]) for r in recs))
    npd = set().union(*({tuple(p) for p in r["names"]["periods"]} for r in recs))
    for y in sorted(years_named(text) - ny):
        probs.append(f"it names {y}, which none of its records names")
    spans, outright = month_runs(text), seasons_named_outright(text)
    for kind, lab in sorted(named_periods(text) - npd):
        run = spans.get(lab) if kind == "season" else None
        if run and lab not in outright and not run["aggregate"]:
            # a run of months, each on a record of its own (see month_runs)
            lack = [MONTHS[int(m) - 1] for m in run["months"] if ("month", m) not in npd]
            if lack:
                probs.append(f"it names {run['words']}: none of its records names the season "
                             f"{lab}, and as a run of months none names {', '.join(lack)}")
            continue
        word = MONTHS[int(lab) - 1] if kind == "month" else lab
        probs.append(f"it names {word}, which none of its records names")
    heads, bands = _region_heads(text, regions)
    rh = {_short(_head_band(r.get("region") or "")[0]) for r in recs if r.get("region")}
    rr = {r.get("region") for r in recs}
    # A RECORD NAMES THE PARTS IT IS SPLIT INTO (package F2, 2026-11-05): an
    # elevation split's bands and other regions, an overlap's bands, each
    # within the record's own region, so a band's share can name its band
    for r in recs:
        for kind, part in _split_parts(r):
            if kind == "other_regions":
                rh.add(_short(_head_band(re.sub(r"^not ", "", part))[0]))
            elif r.get("region"):
                rr.add(f"{r['region']}, {part}")
    for h in sorted(heads - rh):
        probs.append(f"it names the region {h!r}, which none of its records names")
    for b in sorted(bands - rr):
        probs.append(f"it names the band {b!r}, which none of its records names")
    # THE CLASS
    if cls not in EVIDENCE_CLASSES:
        probs.append(f"no evidence class: one of {', '.join(EVIDENCE_CLASSES)}")
    elif cls == "tested expectation":
        exps = {e.get("id"): e for e in plan.get("expectations") or [] if isinstance(e, dict)}
        e = exps.get(c.get("expectation"))
        if plan.get("question_type") != "hypothesis":
            probs.append("a tested expectation needs a hypothesis question")
        if e is None:
            probs.append(f"a tested expectation names the plan's expectation it tests "
                         f"(the plan's: {sorted(x for x in exps if x) or 'none'})")
        elif not any(e.get("analysis") in r["plan_items"] for r in recs):
            probs.append(f"the expectation {e.get('id')} is of the plan's analysis "
                         f"{e.get('analysis')}, which none of its records carries out")
        for r in recs:
            if r.get("label") == AP.SCREENING_LABEL:
                probs.append(f"{r['id']} was {AP.SCREENING_LABEL}: never a tested expectation")
            # MADE AFTER THE CLOSE (the user's decision on G part 2, 2026-11-08),
            # checked in its own right, not only through "not planned"
            if r.get("after_close"):
                probs.append(f"{r['id']} was made after the planned analyses were closed "
                             f"({AP.PLANNED_PHASE_FILE}): never a tested expectation")
            if r.get("amendment") or (r.get("planned") is not True
                                      and r.get("label") != AP.SCREENING_LABEL):
                probs.append(f"{r['id']} is not planned"
                             + (f" (amendment {r['amendment']})" if r.get("amendment") else "")
                             + ": never a tested expectation")
    elif cls == "association":
        if not any(r["call"] == "overlap" for r in recs):
            probs.append("an association rests on an overlap (a correlation or a quarter "
                         "split), and none of its records is one")
    elif cls == "in-model attribution":
        ok = []
        for r in recs:
            if r["call"] != "difference":
                continue
            run = runs.get((r.get("keys") or {}).get("run") or r["run"]) or {}
            if len(run.get("changes") or []) == 1 and run.get("verified"):
                ok.append(r)
        if not ok:
            why = "; ".join(f"{n}: {len(x.get('changes') or [])} change(s), {x.get('why')}"
                            for n, x in sorted(runs.items())) or "the study has no run"
            probs.append("in-model attribution needs a difference from a run that differs "
                         "from the control by one named change, confirmed by the server's "
                         f"checks ({why})")
    # THE WORDING ITS CLASS ALLOWS
    if _CAUSED_BY.search(text):
        probs.append(f"\"{_CAUSED_BY.search(text).group(0)}\" names a cause: no claim names a "
                     f"real-world cause, and in-model attribution says \"{IN_MODEL_WORDS}\"")
    else:
        rest = (re.sub(re.escape(IN_MODEL_WORDS), " ", text, flags=re.I)
                if cls == "in-model attribution" else text)
        m = _CAUSAL.search(rest)
        if m:
            probs.append(f"the causal word \"{m.group(0)}\" goes beyond "
                         f"{cls or 'its evidence'}: "
                         + (CLASS_WORDING.get(cls) or "\"consistent with\""))
    if _BANNED.search(text):
        probs.append(f"\"{_BANNED.search(text).group(0)}\" is a banned word")
    # A PERCENT ONLY WHERE THE FIELD'S NOTES ALLOW ONE (percent_problems, the
    # same helper step 3's audit applies)
    probs += percent_problems(text, recs)
    # THE RANGE ACROSS SETTINGS (G part 1 decisions, 2026-11-08): a shift of a
    # record whose range across settings exceeds 7 days comes with the
    # default, the smallest and the largest shift, each with its setting
    for r in recs:
        probs += range_problems(text, r, [p for rr_, p in matched if rr_ is r])
    # THE WARNINGS ITS RECORDS CARRY, each carried by the claim
    req: List[Dict[str, Any]] = []
    for r in recs:
        req += r["warnings"]
        # a row's own warnings (a screen row, a comparison between settings)
        # when the claim states one of its values
        mine = [p for rr_, p in matched if rr_ is r]
        for w in sorted(r["row_warnings"]):
            if any(p.startswith(w + ".") for p in mine):
                req += r["row_warnings"][w]
    seen, keep = set(), []
    for w in req:
        if (w["code"], w["record"], w["reason"]) not in seen:
            seen.add((w["code"], w["record"], w["reason"]))
            keep.append(w)
    out["warnings"] = keep
    carried = {str(x) for x in (c.get("warnings") or [])}
    for w in keep:
        if w["code"] not in carried and w["label"].lower() not in low:
            probs.append(f"it leaves out the warning \"{w['label']}\" ({w['code']}) its record "
                         f"{w['record']} carries: {w['reason']}")
    out["matched"] = sorted({f"{r['id']}:{p}" for r, p in matched})
    return out


def run_lines() -> List[str]:
    """THE RUNS THE SERVER CAN MAKE, from its own list (domain_changes, the
    list check_proposals resolves against; 2026-11-08): question 3 of G part
    2's run again proposed "snow_water_initial_increase", a run step 3 had no
    way to know was not on it. Each change by id, its summary, whether it is
    available, and its parameters with their kinds and ranges."""
    _elm_modules()
    import domain_changes as DCH                                    # noqa: N812
    L = ["THE RUNS THE SERVER CAN MAKE. An open cause's \"proposed_run\" names changes "
         "from this list only, by id, with its params; any other is refused:"]
    for c in DCH.declare()["changes"]:
        # EACH PARAMETER'S FORM, from the server's own definition beside its
        # check (domain_changes.param_form, 2026-11-08), and what it does
        ps = [f"{name}: {DCH.param_form(spec)}" + (f" ({spec['what']})" if spec.get("what")
                                                   else "")
              for name, spec in (c.get("params") or {}).items()]
        req = c.get("requires")
        req = "; ".join(map(str, req)) if isinstance(req, (list, tuple)) else req
        avail = "" if c.get("available", True) else f" NOT AVAILABLE ({req or 'not built'})"
        L += _wrap(f"{c['id']}{avail}: {c.get('summary') or c.get('meaning')}"
                   + (f"; params: {'; '.join(ps)}" if ps else "; no params"), "  - ", "      ")
    return L


def expectation_lines(plan: Dict[str, Any], records: Dict[str, Dict[str, Any]]) -> List[str]:
    """THE PLAN'S EXPECTATIONS, for step 3's brief (2026-11-08): question 3
    of G part 2's run again classed no claim as a tested expectation, and
    step 3 had never been shown the plan's expectations. Each by id, with its
    analysis, field, direction and words, and the planned records made
    before the close that carry out its analysis. [] for a plan with none."""
    exps = [e for e in plan.get("expectations") or [] if isinstance(e, dict)]
    if not exps:
        return []
    L = ["", "THE PLAN'S EXPECTATIONS, stated before any result was seen. A tested claim names "
         "one in \"expectation\", is classed \"tested expectation\", and rests on planned "
         "records of that expectation's analysis made before the close:"]
    for e in exps:
        ids = [r["id"] for r in records.values()
               if e.get("analysis") in (r.get("plan_items") or []) and r.get("planned") is True
               and not r.get("after_close")]
        L += _wrap(f"{e.get('id')} (analysis {e.get('analysis')}, field {e.get('field')}, "
                   f"expected {e.get('sign')}): {e.get('text')}", "    ", "        ")
        L += _wrap("records carrying out its analysis: " + (", ".join(ids) or "none"),
                   "        ", "            ")
    return L


def check_proposals(open_causes) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Each open cause's proposed run, resolved by the server's change
    vocabulary (domain_changes.resolve): its changes with what each means."""
    _elm_modules()
    import domain_changes as DCH                                    # noqa: N812
    out, probs = [], []
    # A RUN ON THE LIST, PROPOSED WRONGLY, IS SAID SO (2026-11-08): question 3's
    # weather_scale with months "6,7,8" was refused as "not a run the server
    # can make", which it is; its months were not in the form the server checks
    known = {x.get("id") for x in DCH.declare()["changes"]}
    for i, oc in enumerate(open_causes or [], 1):
        ch = ((oc or {}).get("proposed_run") or {}).get("changes") or []
        res = {"cause": (oc or {}).get("cause") or (oc or {}).get("question"), "changes": []}
        if not ch:
            probs.append(f"open cause {i} proposes no run in the server's change vocabulary")
        for c in ch:
            try:
                r = DCH.resolve({"id": c.get("id"), "params": c.get("params") or {}})
                res["changes"].append({k: r[k] for k in ("id", "params", "meaning",
                                                         "not_meaning")})
            except Exception as e:                              # noqa: BLE001
                probs.append(f"open cause {i}: {c.get('id')!r} "
                             + ("cannot be made as proposed" if c.get("id") in known
                                else "is not a run the server can make") + f": {e}")
        out.append(res)
    return out, probs


def _headline_claims(headline: str, claims: List[Dict[str, Any]]) -> List[int]:
    marked = [i for i, c in enumerate(claims) if c.get("headline") is True]
    if marked:
        return marked
    hits = []
    for w, unit, _ in _number_words(headline):
        if _is_year(w.lstrip("-"), unit, set(range(1800, 2201))):
            continue
        for i, c in enumerate(claims):
            if any(_fin(v) and abs(abs(v) - abs(float(w))) <= _half_unit(w) + 1e-9
                   for v in c.get("values") or []) and i not in hits:
                hits.append(i)
    return hits


def headline_bars(i: int, c: Dict[str, Any], records: Dict[str, Dict[str, Any]],
                  qt: str) -> List[str]:
    """What bars claim i (0-based) from heading the report of a question of
    type `qt`: the headline rule's conditions. A hypothesis question's
    headline is a tested expectation resting on no lookup found by
    screening, not planned, or made after the planned analyses were closed;
    an exploratory question's may be any claim."""
    if qt != "hypothesis":
        return []
    AP = _elm_modules()[0]
    out = []
    cls = c.get("evidence_class")
    recs = [records[x] for x in c.get("evidence") or [] if x in records]
    if cls != "tested expectation":
        out.append(f"claim {i + 1} is {cls or 'unclassed'}, not tested: "
                   + HEADLINE_RULE["hypothesis"])
    if any(r.get("label") == AP.SCREENING_LABEL or r.get("amendment") for r in recs):
        out.append(f"claim {i + 1} rests on a lookup found by screening or not planned: "
                   f"never a hypothesis headline")
    if any(r.get("after_close") for r in recs):
        out.append(f"claim {i + 1} rests on a lookup made after the planned analyses were "
                   f"closed: never a hypothesis headline")
    return out


# A HEADLINE KEEPS ITS CLAIM'S QUALIFICATIONS (the user's decision of
# 2026-11-08, preparing the phase's close): the accepted batch's question 1
# headline was claim 1's text, whose record is a partial-record estimate, and
# its "restricted comparability" was shown only under the claim, further down
# the page. Every warning the headline's claim carries (the trace has checked
# that it carries all its records') goes beside the headline wherever it is
# shown or exported, in the registry's own words: no model writes them.
def headline_caveats(claim: Dict[str, Any]) -> List[Dict[str, str]]:
    """Each warning a headline's claim carries, as {"code", "label",
    "means"}, once each: the codes it lists, then any its records carry that
    it carries only in its words (the trace's own list, so a label its
    sentence happens to hold still shows); the registry's words (WARNINGS),
    or for flag:<x> and reason:<x> domain_methods' own. A code the registry
    does not hold, or a "warnings" that is not a list, shows nothing: no
    caveat is in a model's words."""
    listed = claim.get("warnings")
    codes = [str(x) for x in listed] if isinstance(listed, list) else []
    codes += [str(w["code"]) for w in (claim.get("evidence_trace") or {}).get("warnings") or []
              if isinstance(w, dict) and w.get("code")]
    out: List[Dict[str, str]] = []
    for code in dict.fromkeys(codes):
        kind, _, x = code.partition(":")
        try:
            label = warning_label(code)
        except KeyError:
            continue
        if x:
            M = _elm_modules()[2]
            src = M.FLAGS if kind == "flag" else M.REASONS if kind == "reason" else {}
            means = (src.get(x) or (None, ""))[1]
        else:
            means = (WARNINGS.get(code) or ("", ""))[1]
        out.append({"code": code, "label": label, "means": means})
    return out


def _headline_claim(report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The claim the headline was selected from, or None."""
    ch = (report.get("headline_selection") or {}).get("chosen")
    claims = report.get("claims") or []
    if isinstance(ch, int) and 1 <= ch <= len(claims) and isinstance(claims[ch - 1], dict):
        return claims[ch - 1]
    return None


def headline_caveat_line(report: Dict[str, Any]) -> Optional[str]:
    """The words shown beside a headline: the caveats it carries, from which
    claim and records; None when its claim carries none. A report written
    before its caveats were recorded (2026-11-08) gets them from its
    selected claim, so a page rendered again from it still shows them."""
    c = _headline_claim(report)
    if "headline_caveats" in report:
        cav = report.get("headline_caveats") or []
    else:
        cav = headline_caveats(c) if c and report.get("headline") else []
    if not cav:
        return None
    src = ((report.get("headline_selection") or {}).get("source")
           or ({"claim": report["headline_selection"]["chosen"],
                "evidence": list(c.get("evidence") or [])} if c else {}))
    where = f"claim {src['claim']}" if src.get("claim") else "its claim"
    if src.get("evidence"):
        where += f" ({', '.join(map(str, src['evidence']))})"
    return (f"Caveats it carries from {where}: "
            + "; ".join(f"{c['label']} ({c['means']})" if c.get("means") else c["label"]
                        for c in cav) + ".")


def headline_text(report: Dict[str, Any]) -> Optional[str]:
    """The headline as plain text, with the caveats it carries beside it, for
    any place that shows the headline without the page around it."""
    h = report.get("headline")
    if not h:
        return None
    line = headline_caveat_line(report)
    return f"{h} {line}" if line else str(h)


def headline_order(claims: List[Dict[str, Any]], rank: Optional[Sequence[Any]]) -> List[int]:
    """The claims in the order they are offered for the headline (1-based):
    `rank` as given, every claim once, then any claim it leaves out in its
    own order. A reply recorded before 2026-11-08 gave no rank: the claims it
    marked "headline" come first."""
    if rank is None:
        rank = [i for i, c in enumerate(claims, 1) if c.get("headline") is True]
    order = [k for k in rank if isinstance(k, int) and not isinstance(k, bool)
             and 1 <= k <= len(claims)]
    order = list(dict.fromkeys(order))
    return order + [k for k in range(1, len(claims) + 1) if k not in order]


def select_headline(claims: List[Dict[str, Any]], trace: Dict[str, Any],
                    rank: Optional[Sequence[Any]]) -> Dict[str, Any]:
    """THE HEADLINE IS A VERIFIED CLAIM'S OWN TEXT (the user's decision of
    2026-11-08, after the third acceptance batch): the first claim of the
    order (headline_order) that traced and that the question type's rule
    allows (headline_bars), its text as it traced. "chosen" is None when no
    claim is eligible; each claim passed over is listed with why."""
    qt = trace.get("question_type") if trace.get("question_type") in HEADLINE_RULE \
        else "exploratory"
    failed = set(trace.get("claims_failed") or [])
    records = trace.get("records") or {}
    passed: List[Dict[str, Any]] = []
    order = headline_order(claims, rank)
    for k in order:
        why = ([f"claim {k} fails the evidence trace"] if k in failed
               else headline_bars(k - 1, claims[k - 1], records, qt))
        if not why:
            return {"rule": HEADLINE_RULE[qt], "order": order, "chosen": k,
                    "text": str(claims[k - 1].get("claim") or ""), "passed_over": passed}
        passed.append({"claim": k, "why": "; ".join(why)})
    return {"rule": HEADLINE_RULE[qt], "order": order, "chosen": None, "text": None,
            "passed_over": passed}


def trace_evidence(claims: List[Dict[str, Any]], headline: Optional[str], analysis_dir,
                   study_dir=None, regions=(), open_causes=None, notes=None
                   ) -> Dict[str, Any]:
    """The claim trace of a study with a suite: every claim checked against
    its evidence records (check_claim, written into the claim as
    "evidence_trace"), the headline against the question type's rule, and
    each open cause's proposed run. "withheld_because" is set when any of
    them fails."""
    import json
    from pathlib import Path
    AP = _elm_modules()[0]
    records = evidence_records(analysis_dir, notes=notes)
    plan = AP.load_plan(analysis_dir)
    qt = plan.get("question_type") if plan.get("question_type") in HEADLINE_RULE \
        else "exploratory"
    runs = study_runs(study_dir) if study_dir else {}
    regs = set(regions or ())
    for a in plan.get("analyses") or []:
        regs |= {str(x) for x in (a or {}).get("regions") or []}
    regs |= {r["region"] for r in records.values() if r.get("region")}
    if (plan.get("screen") or {}).get("region"):
        regs.add(plan["screen"]["region"])
    failed, per = [], {}
    for i, c in enumerate(claims, 1):
        t = check_claim(c, records, plan, runs, sorted(regs))
        c["evidence_trace"] = t
        if t["problems"]:
            c["trace"] = FAILS_TRACE
            failed.append(i)
            per[i] = t["problems"]
    head = {"rule": HEADLINE_RULE[qt], "claims": [], "problems": []}
    if headline:
        hc = _headline_claims(headline, claims)
        head["claims"] = [i + 1 for i in hc]
        if not hc:
            head["problems"].append("the headline states no claim's value and marks no claim")
        for i in hc:
            head["problems"] += headline_bars(i, claims[i], records, qt)
        # EVERY EXPLORATORY HEADLINE IS LABELLED (the user's decision of
        # 2026-11-08), not only one that comes from the screen
        if qt == "exploratory" and AP.EXPLORATORY_LABEL not in str(headline).lower():
            head["problems"].append(f"an exploratory question's headline is labelled "
                                    f"{AP.EXPLORATORY_LABEL}, and this one is not")
    proposals, pprobs = check_proposals(open_causes)
    sr = Path(analysis_dir) / AP.SCREEN_FILE
    out = {"file": LOOKUPS_LOG, "question_type": qt, "n_claims": len(claims),
           "claims_failed": failed, "problems": per, "headline": head,
           "records": records, "open_causes": proposals, "open_cause_problems": pprobs,
           "screen_record": json.loads(sr.read_text()) if sr.is_file() else None,
           "headline_rule": HEADLINE_RULE[qt], "wording": CLASS_WORDING,
           "runs": runs}
    why = []
    if failed:
        why.append(f"{len(failed)} of {len(claims)} claims fail the evidence trace "
                   f"(claims {', '.join(map(str, failed))})")
    if head["problems"]:
        why.append("the headline breaks its rule: " + "; ".join(head["problems"]))
    if pprobs:
        why.append("; ".join(pprobs))
    if why:
        out["withheld_because"] = "; ".join(why)
    return out


def describe_record(r: Dict[str, Any]) -> str:
    """A record in one line: the call, its keys, its comparison, its support."""
    what = r.get("method") or r.get("call")
    keys = ", ".join(str(x) for x in (r.get("measure") or r.get("field"), r.get("period"),
                                      r.get("region")) if x)
    comp = (r.get("comparison") or {}).get("against")
    yrs = (r.get("comparison") or {}).get("years")
    sup = r.get("support") or {}
    bits = [f"{r['id']}: {what} of run {r.get('run')}" + (f" ({keys})" if keys else "")]
    if comp:
        bits.append(f"against {comp}" + (f" {years_text_short(yrs)}" if yrs else ""))
    if r.get("inputs"):
        bits.append("made from " + ", ".join(
            (f"{x['record']} ({x.get('call')} {x.get('field')} {x.get('period')}"
             + (f", its {x['path']} map" if x.get("path") else ", its map") + ")")
            if x.get("record") else "a map no lookup returned" for x in r["inputs"]))
    if sup.get("area_share") is not None:
        bits.append(f"area share covered {sup['area_share']:.3g}")
    bits.append("planned" if r.get("planned") else "not planned")
    bits.append(f"hash {str(r.get('sha256'))[:12]}")
    return "; ".join(bits)


def years_text_short(yrs) -> str:
    ys = sorted({int(y) for y in (yrs or []) if _is_num(y) or str(y).isdigit()})
    if not ys:
        return ""
    if ys == list(range(ys[0], ys[-1] + 1)) and len(ys) > 2:
        return f"{ys[0]}-{ys[-1]}"
    return ", ".join(map(str, ys))


# ═════════════════════════════════════════════════════════════════════════
# THE RECORDS, SHOWN TO STEP 3 UNDER EACH FINDING (fix 1 of G part 2,
# 2026-11-08)
# ═════════════════════════════════════════════════════════════════════════
# Step 3 was asked to name each claim's records and to carry their warnings,
# and was shown neither: its brief listed each finding's id, question and
# result, never a record id. Under each finding of a study with a suite it now
# sees the records the trace checks (evidence_records of the study's own log,
# the ids from the finding's lookups): each record's line, planned or not, its
# label, its amendment, whether it was made after the planned analyses were
# closed, the warnings a claim citing it must carry, its field's percent rule
# and the range rule when it applies. The finding's own result numbers are
# split in two: values of its records (citable, with the record and the path)
# and numbers the script made itself, which are no value of any record, so a
# claim citing one fails the trace. Nothing here changes what the audit or the
# trace accepts.
#
# THE CAP, DETERMINISTIC: at most BRIEF_MAX_RECORDS records per finding in the
# log's order; under each, at most BRIEF_MAX_PER_RECORD of the result numbers
# that are its values, in the result's own order (a number several records
# hold is listed under the first, in the log's order); and at most
# BRIEF_MAX_NUMBERS of the numbers the script made itself, in the result's
# order. A list of more than LIST_SHOWN numbers is one entry. The count of
# the rest is said.
BRIEF_MAX_RECORDS = 60
BRIEF_MAX_PER_RECORD = 12
BRIEF_MAX_NUMBERS = 40
LIST_SHOWN = 6
REASON_SHOWN = 120


def _result_numbers(x, path: str = "") -> List[Tuple[str, Any]]:
    """(path, number) of every number in a figure's result, in its own order;
    a list of more than LIST_SHOWN numbers is one (path, [numbers])."""
    out: List[Tuple[str, Any]] = []
    if isinstance(x, dict):
        for k, v in x.items():
            out += _result_numbers(v, f"{path}.{k}" if path else str(k))
    elif isinstance(x, (list, tuple)):
        if len(x) > LIST_SHOWN and all(_is_num(v) or v is None for v in x):
            out.append((path, [v for v in x if _fin(v)]))
        else:
            for i, v in enumerate(x):
                out += _result_numbers(v, f"{path}.{i}")
    elif _fin(x):
        out.append((path, x))
    return out


def citable(result: Any) -> Dict[str, Any]:
    """A logged result's citable values: its finite numbers by path, as an
    evidence record keeps them ("values"), without the labels beside them
    (its normal years, a list) or the method's own version and settings."""
    if not isinstance(result, dict):
        return {}
    return {p: v for p, v in result.items() if _fin(v) and p != "version"
            and not str(p).startswith("settings.")}


def record_value(x: float, recs) -> Optional[Tuple[Dict[str, Any], str]]:
    """(record, path) of the first value of `recs` (in their order) equal to
    x at float precision, or None: results are the methods' unrounded values
    after a trip through JSON."""
    for r in recs or []:
        for p, v in (r.get("values") or {}).items():
            if abs(v - x) <= 1e-12 * max(1.0, abs(v), abs(x)):
                return r, p
    return None


def split_result_numbers(result, recs) -> Tuple[Dict[str, List[str]], List[str]]:
    """The finding's result numbers in two kinds: values of its records, by
    record ({record id: ["<result path> = <number> (its <record path>)",
    ...]}, in the result's order; the record's path left out where the
    result's path ends with it), and numbers the script made itself, a value
    of no record (["<result path> = <number>", ...]). A list of more than
    LIST_SHOWN numbers is one entry of the kind all its numbers share, else
    of the second."""
    citable: Dict[str, List[str]] = {}
    made: List[str] = []
    for path, x in _result_numbers(result):
        if isinstance(x, list):
            hits = [record_value(v, recs) for v in x]
            if hits and all(hits):
                ids = sorted({h[0]["id"] for h in hits}, key=lambda i: int(i[1:]))
                citable.setdefault(hits[0][0]["id"], []).append(
                    f"{path}: {len(x)} numbers (a list), each a value of " + ", ".join(ids))
            else:
                made.append(f"{path}: {len(x)} numbers (a list), {sum(not h for h in hits)} "
                            f"of them a value of no record")
            continue
        hit = record_value(x, recs)
        if hit:
            # the record's own path, said only where the result's path does
            # not already end with it
            same = path == hit[1] or path.endswith("." + hit[1])
            citable.setdefault(hit[0]["id"], []).append(
                f"{path} = {x!r}" + ("" if same else f" (its {hit[1]})"))
        else:
            made.append(f"{path} = {x!r}")
    return citable, made


def _range_rule(r: Dict[str, Any]) -> Optional[str]:
    a = r.get("across_settings")
    if not a:
        return None
    if a.get("range_required"):
        parts = []
        for part, said in _RANGE_PART_WORDS.items():
            e = a.get(part) or {}
            if _fin(e.get("median_shift")):
                parts.append(f"{said} ({e['median_shift']:.3g} days, {e.get('variant')})")
        return (f"range rule: applies; its shift moves by {a.get('spread_days'):.3g} days "
                f"across the settings, more than {a.get('limit_days'):g}: a claim giving the "
                f"shift states " + ", ".join(parts) + ", each with its setting")
    if _fin(a.get("spread_days")):
        return (f"range rule: does not apply (the shift moves by {a['spread_days']:.3g} days "
                f"across the settings, within {a.get('limit_days'):g})")
    return f"range rule: does not apply ({a.get('why') or 'no range'})"


def record_brief_lines(r: Dict[str, Any], in_result: Optional[List[str]] = None
                       ) -> List[str]:
    """One evidence record as step 3 reads it under its finding: its line,
    its marks, the warnings a claim citing it carries, its field's percent
    rule, the range rule when it applies, and the finding's result numbers
    that are its values."""
    L = [f"          {describe_record(r)}"]
    marks = [f"plan items {', '.join(r.get('plan_items') or []) or 'none'}"]
    if r.get("label"):
        marks.append(f"label: {r['label']}")
    if r.get("amendment"):
        marks.append(f"amendment {r['amendment']}")
    if r.get("after_close"):
        marks.append("made after the planned analyses were closed")
    L.append("              " + "; ".join(marks))
    ws = r.get("warnings") or []
    if ws:
        L.append("              warnings a claim citing it carries (code, words, reason): "
                 + "; ".join(f"{w['code']} (\"{w['label']}\"): {str(w['reason'])[:REASON_SHOWN]}"
                             for w in ws))
    else:
        L.append("              warnings a claim citing it carries: none")
    rows = {w["code"] for ws_ in (r.get("row_warnings") or {}).values() for w in ws_}
    if rows:
        n = sum(1 for ws_ in (r.get("row_warnings") or {}).values() if ws_)
        L.append(f"              row warnings ({n} rows; a claim stating a row's value carries "
                 f"that row's): " + ", ".join(sorted(rows)))
    if r.get("field"):
        L.append("              percent: allowed" if r.get("percent_allowed") is not False else
                 f"              percent: none ({r.get('percent_why_not')})")
    rr = _range_rule(r)
    if rr:
        L.append(f"              {rr}")
    got = list(in_result or [])
    L.append("              its values in the finding's result: "
             + ("; ".join(got[:BRIEF_MAX_PER_RECORD]) if got else "none")
             + (f"; and {len(got) - BRIEF_MAX_PER_RECORD} more"
                if len(got) > BRIEF_MAX_PER_RECORD else ""))
    return L


def finding_evidence_lines(finding: Dict[str, Any], records: Dict[str, Dict[str, Any]]
                           ) -> List[str]:
    """Under one finding of step 3's brief: its evidence records, each with
    the finding's result numbers that are its values, then the numbers the
    script made itself, a value of no record."""
    ids = [str(x.get("id")) for x in finding.get("lookups") or [] if isinstance(x, dict)]
    recs = [records[i] for i in ids if i in records]
    citable, made = split_result_numbers(finding.get("result"), recs)
    L = [f"        evidence records: {len(recs)} (a claim names those it rests on in "
         f"\"evidence\", by id; in the log's order, at most {BRIEF_MAX_RECORDS} shown, each "
         f"with at most {BRIEF_MAX_PER_RECORD} of its values in the result; a label, an "
         f"amendment or the close is shown only where a record has one)"]
    if not recs:
        L.append("          none: no number of this finding is a value of a record, so no "
                 "claim can rest on it")
    for r in recs[:BRIEF_MAX_RECORDS]:
        L += record_brief_lines(r, citable.get(r["id"]))
    if len(recs) > BRIEF_MAX_RECORDS:
        L.append(f"          and {len(recs) - BRIEF_MAX_RECORDS} more: "
                 + ", ".join(r["id"] for r in recs[BRIEF_MAX_RECORDS:]))
    L.append(f"        numbers the script made itself, a value of no record ({len(made)}; a "
             f"claim stating one fails the evidence trace; in the result's order, at most "
             f"{BRIEF_MAX_NUMBERS} shown):")
    L += [f"            {x}" for x in made[:BRIEF_MAX_NUMBERS]] or ["            none"]
    if len(made) > BRIEF_MAX_NUMBERS:
        L.append(f"            and {len(made) - BRIEF_MAX_NUMBERS} more")
    return L


CLASS_NEEDS = {
    "observation": "a value of its records, or how it changed",
    "association": "an overlap among its records (a correlation or a quarter split)",
    "tested expectation": "the plan's expectation, named in \"expectation\", and planned "
                          "lookups only: never one found by screening, added by an "
                          "amendment, or made after the planned analyses were closed",
    "exploratory": "a finding made by looking, not stated first, labelled exploratory",
    "in-model attribution": "a difference from a run that differs from the control by one "
                            "named change, confirmed by the server's checks"}


def class_lines() -> List[str]:
    """The evidence classes as step 3's template states them: what each
    needs and the wording it allows (CLASS_NEEDS, CLASS_WORDING)."""
    L = ["THE EVIDENCE CLASSES (\"evidence_class\"), each with what it needs and the",
         "wording it allows:"]
    for c in EVIDENCE_CLASSES:
        L += _wrap(f"\"{c}\": needs {CLASS_NEEDS[c]}; wording: {CLASS_WORDING[c]}",
                   "  ", "      ")
    return L


def evidence_withheld_markdown(report: Dict[str, Any]) -> str:
    """REPORT.md when claims fail the evidence trace: why, then every claim
    with its records, its class, its warnings and its problems. The answer
    and the headline are not presented as findings."""
    from agents.analysis.step4_report import _plain
    t = report.get("evidence_trace") or {}
    lines = [f"# WITHHELD: {t.get('withheld_because')}", "",
             f"*Question: {_plain(report.get('question') or '(no question recorded)')}*", "",
             f"Each claim was traced to the evidence records of the logged lookups "
             f"({t.get('file')}): its numbers, the periods and regions it names, its evidence "
             f"class and wording, and the warnings its records carry. A claim fails, so this "
             f"report is withheld. The record, analysis.json, is kept whole beside it.", "",
             f"The headline held back: \"{_plain(report.get('headline'))}\""
             + (f" {_plain(headline_caveat_line(report))}" if headline_caveat_line(report)
                else ""), ""]
    for p in (t.get("headline") or {}).get("problems") or []:
        lines.append(f"- headline: {p}")
    for p in t.get("open_cause_problems") or []:
        lines.append(f"- {p}")
    # the numbers the claims cite that did not reproduce from the model output
    # (package F2): each record, the value logged and the value made again
    from agents.analysis.numbers_check import evidence_withheld_lines
    nums = evidence_withheld_lines(report.get("evidence_numbers"))
    if nums:
        lines += [""] + nums
    lines += ["", "## The claims and their evidence", ""]
    recs = t.get("records") or {}
    for i, c in enumerate(report.get("claims") or [], 1):
        e = c.get("evidence_trace") or {}
        lines.append(f"{i}. {_plain(c.get('claim'))}"
                     + (f" **({c['trace']})**" if c.get("trace") else ""))
        lines.append(f"   - class: {e.get('class') or 'none'}")
        for rid in e.get("records") or []:
            lines.append(f"   - {describe_record(recs[rid])}" if rid in recs
                         else f"   - {rid}: no such record")
        for w in e.get("warnings") or []:
            lines.append(f"   - warning, {w['label']}: {_plain(w['reason'])}")
        for p in e.get("problems") or []:
            lines.append(f"   - FAILS: {_plain(p)}")
        lines.append("")
    return "\n".join(lines)
