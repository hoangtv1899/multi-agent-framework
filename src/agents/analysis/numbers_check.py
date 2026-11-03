#!/usr/bin/env python3
"""Every number in 03_results/regional.json, made again by a second code path.

A whole-domain experiment's numbers are made once, by the model server
(domain_analysis.py over gridded.py and regions.py), and everything after
reads them: the numbers report, the Analyzer, the e-mail. A wrong number there
is wrong everywhere, and no audit of what cites it can see it (the even-count
median of 2026-09-13 read 10.7 m where the data said 12.1 m, and every
citation of it checked out). So the numbers are made a second time here, from
each run's own history, by code that SHARES NO REDUCTION CODE with those three
modules, and the table is reported only when the two agree.

WHAT IS INDEPENDENT. The netCDF reading (netCDF4 and numpy directly), the time
weighting (each record counts for the part of its own time_bounds that falls in
the year, so the start convention, the year boundary and the run's first
30-minute step need no rule of their own), the area weights (area x landfrac,
read from the history), and every mask: the polygon test is matplotlib's on
the cell centres, a gridded-field region applies its rule to the file here,
and vegetated cells and elevation bands are read from the surface file the
control's lnd_in names, with the band edges read back from the band names the
table carries. WHAT IS SHARED is the definition of each quantity (a rate per
second is totalled over the year, anything else is the year's time-weighted
mean, a region's value is the area-weighted mean over its cells with a value)
and where the inputs are: the table names each run's output directory, the
manager's records name the control, and the control registry names each
registered region's file.

  check(run_dir)  the comparison record, written to 04_analysis/numbers_check.json
  gate(run_dir)   the analyze stage's gate: the check, and when it fails, the
                  numbers report written WITHHELD in place of the numbers

THE TOLERANCE. History is float32 (about 7 significant digits); both paths read
the same float32 values and reduce them in float64, so two right answers agree
to about 1e-12. A number is reproduced when |table - recomputed| <= RTOL x scale
+ ATOL, where scale is the size of the quantities it was made from (a
difference is judged against the two values it subtracts, a percent against
the value and its base). RTOL 1e-5 is about 100 float32 steps: loose enough for
a path that sums in float32, tight enough that a change in the fifth
significant digit fails. ATOL 1e-9 lets two zeros (a no-fire run's burned
area) agree.
"""
from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

CHECK_JSON = "numbers_check.json"
TABLE = "03_results/regional.json"
RTOL = 1e-5
ATOL = 1e-9
# The table rounds a changed share to 4 decimals (domain_analysis.changed_share),
# and so does the check; a share on a rounding edge may land one step apart.
SHARE_ATOL = 1e-4 + 1e-9
SECONDS_PER_DAY = 86400.0
FILL_ABS = 1e30
# Cell centres of two files are the same cells within this, in degrees.
COORD_TOL_DEG = 1e-4
# Two record bounds meet when they are this close, in days (about 0.1 s).
TIME_TOL_DAYS = 1e-6
# A field stored per second whose units do not say so (ELM's FireMod.F90
# writes FAREA_BURNED per second under the units 'proportion').
PER_SECOND_ANYWAY = ("FAREA_BURNED",)
BURNED_AREA, BURNED_FRACTION = "burned_area_km2", "FAREA_BURNED"
VEGETATED = "vegetated cells"
VEG_SUFFIX = ", vegetated"
# The keys of a regional table this check knows how to recompute; any other
# top-level key is listed as not checked, never failed and never silent.
KNOWN = ("regions", "years", "runs", "problems", "differences", "event",
         "changed_share", "observed", "error")
MAX_ROWS_SHOWN = 40


class CheckError(Exception):
    """The numbers cannot be recomputed, and the message says why."""


# ─────────────────────────────────────────────────────────────────────────────
# WHERE THE INPUTS ARE
# ─────────────────────────────────────────────────────────────────────────────
def _read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return {}


def _registry(name: Optional[str]) -> Dict[str, Any]:
    """The control's entry in the server's control registry, or {}.

    Read through the server's own loader (domain_controls.load), which finds the
    git-ignored registry the way the server does (IDEAS_DOMAIN_CONTROLS, then
    mcp/elm-mcp/domain_controls.json). A loader, not reduction code."""
    if not name:
        return {}
    try:
        import domain_controls as DCT                               # noqa: N812
    except ImportError:
        src = Path(__file__).resolve().parents[3] / "mcp" / "elm-mcp" / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        import domain_controls as DCT                               # noqa: N812
    return (DCT.load().get("controls") or {}).get(name) or {}


def inputs(run_dir, registry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """What the check reads, from the run's own records.

    The control: 01_inputs/domain_control_facts.json (the manager's record of the
    reused control), else 01_inputs/domain_runs.json (the server's build answer).
    Its regions: the registry entry of that name, as the manager hands them to
    domain_analysis. The surface file: fsurdat in the control's CaseDocs/lnd_in,
    as domain_analysis reads it, else in its run directory's lnd_in."""
    rd = Path(run_dir)
    facts = _read_json(rd / "01_inputs" / "domain_control_facts.json")
    built = _read_json(rd / "01_inputs" / "domain_runs.json")
    ctl = dict(facts.get("control") or built.get("control") or {})
    if not ctl.get("case_dir"):
        ent = next((r for r in built.get("runs") or [] if r.get("name") == "control"), {})
        ctl.setdefault("case_dir", ent.get("case_dir"))
    entry = registry if registry is not None else _registry(ctl.get("name"))
    case_dir = ctl.get("case_dir") or entry.get("case_dir")
    return {"control": ctl.get("name"), "case_dir": case_dir,
            "regions": dict(entry.get("regions") or {}),
            "observations": dict(entry.get("observations") or {}),
            "registry_entry": bool(entry)}


def _fsurdat(case_dir, run_dir=None) -> Path:
    for p in ([Path(case_dir) / "CaseDocs" / "lnd_in"] if case_dir else []) + \
             ([Path(run_dir) / "lnd_in"] if run_dir else []):
        if p.is_file():
            m = re.search(r"^\s*fsurdat\s*=\s*['\"]([^'\"]+)['\"]", p.read_text(),
                          re.MULTILINE | re.IGNORECASE)
            if m:
                return Path(m.group(1).strip())
            raise CheckError(f"{p} names no fsurdat")
    raise CheckError(f"no lnd_in in {case_dir}/CaseDocs or {run_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# READING THE HISTORY
# ─────────────────────────────────────────────────────────────────────────────
_HIST = re.compile(r"\.elm\.(h\d+)\.\d{4}-\d{2}(?:-\d{2}-\d{5})?\.nc$")


def _clean(a) -> np.ndarray:
    """float64, with masked, non-finite and |x| > 1e30 values as NaN."""
    x = np.ma.filled(np.ma.asarray(a).astype(np.float64), np.nan)
    x = np.asarray(x, dtype=np.float64)
    x[~np.isfinite(x) | (np.abs(x) > FILL_ABS)] = np.nan
    return x


def _lon180(lon) -> np.ndarray:
    x = np.asarray(lon, dtype=np.float64)
    return (x + 180.0) % 360.0 - 180.0


def _same_cells(lat, lon, glat, glon, what: str) -> None:
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    if lat.ndim == 1 and lon.ndim == 1 and (lat.size, lon.size) == glat.shape:
        lon, lat = np.meshgrid(lon, lat)
    if lat.shape != glat.shape or lon.shape != glat.shape:
        raise CheckError(f"{what}: cells of shape {lat.shape}, the history {glat.shape}")
    d = max(float(np.nanmax(np.abs(lat - glat))),
            float(np.nanmax(np.abs(_lon180(lon - glon)))))
    if not d <= COORD_TOL_DEG:
        raise CheckError(f"{what}: cell centres differ from the history's by {d:.3g} deg")


def per_second(var: str, units: str) -> bool:
    """A rate per second: its year is a total, never a mean.

    Units read as a quotient ('mm/s', 'gC/m^2/s', 'proportion/sec', 'per
    second'), a negative power ('kg m-2 s-1', 's^-1') or 'm2s' (umol/m2s), but
    not a second squared ('kg/m/s^2', 's-2'). FAREA_BURNED is per second
    whatever its units say."""
    if var in PER_SECOND_ANYWAY:
        return True
    u = " ".join(str(units or "").lower().split())
    quotient = re.search(r"(?:/|\bper\s)\s*(?:s|sec|second)(?![a-z0-9])", u)
    squared = re.search(r"(?:/|\bper\s)\s*(?:s|sec|second)\s*(?:\^|\*\*)\s*\+?[02-9]", u)
    power = re.search(r"(?<![a-z])s\s*(?:\^|\*\*)?\s*-\s*1(?!\d)", u)
    return bool((quotient and not squared) or power or re.search(r"m\^?2\s*s$", u))


class History:
    """One run's history files, read with netCDF4 and nothing else of ours."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        if not self.run_dir.is_dir():
            raise CheckError(f"{run_dir} is not a directory")
        self.files: Dict[str, List[Path]] = {}
        for p in sorted(self.run_dir.iterdir()):
            m = _HIST.search(p.name)
            if m and ".elm_" not in p.name:
                self.files.setdefault(m.group(1), []).append(p)
        if not self.files:
            raise CheckError(f"{run_dir} holds no ELM history")
        self._vars: Dict[str, set] = {}

    def streams(self) -> List[str]:
        return sorted(self.files, key=lambda s: int(s[1:]))

    def variables(self, stream: str) -> set:
        if stream not in self._vars:
            import netCDF4
            with netCDF4.Dataset(self.files[stream][0]) as d:
                self._vars[stream] = set(d.variables)
        return self._vars[stream]

    def stream_for(self, var: str) -> Optional[str]:
        return next((s for s in self.streams() if var in self.variables(s)), None)

    def grid(self, stream: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Cell centres and weights (area x landfrac, km2; 0 where either is
        missing), from the first file of a stream."""
        import netCDF4
        st = stream or self.streams()[0]
        with netCDF4.Dataset(self.files[st][0]) as d:
            for v in ("lat", "lon", "area", "landfrac"):
                if v not in d.variables:
                    raise CheckError(f"{self.files[st][0].name} has no {v}")
            lat, lon = _clean(d["lat"][:]), _clean(d["lon"][:])
            area, landfrac = _clean(d["area"][:]), _clean(d["landfrac"][:])
        if lat.ndim == 1 and lon.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)
        if area.shape != lat.shape:
            raise CheckError(f"area has shape {area.shape}, the cells {lat.shape}: "
                             f"only a lat x lon grid is checked here")
        w = area * landfrac
        return {"lat": lat, "lon": lon, "weight": np.where(np.isfinite(w), w, 0.0)}

    def units(self, stream: str, var: str) -> str:
        import netCDF4
        with netCDF4.Dataset(self.files[stream][0]) as d:
            return str(getattr(d[var], "units", "") or "")

    def year(self, stream: str, var: str, year: int, W: np.ndarray, cells: np.ndarray
             ) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[str]]:
        """(region means per record (n, regions), each record's days inside the
        year, why the year is not whole or None).

        W is (cells, regions): each region's weight on the cells `cells` (a flat
        index into the grid). A record counts for the overlap of its own
        time_bounds with [1 Jan year, 1 Jan year+1); the records found must tile
        that interval exactly, or the year is not whole."""
        import cftime
        import netCDF4
        means: List[np.ndarray] = []
        spans: List[Tuple[float, float]] = []
        length = None
        for p in self.files[stream]:
            with netCDF4.Dataset(p) as d:
                if var not in d.variables:
                    continue
                t = d["time"]
                bname = getattr(t, "bounds", None) or "time_bounds"
                if bname not in d.variables:
                    raise CheckError(f"{p.name} has no time bounds")
                cal = str(getattr(t, "calendar", "standard"))
                units = str(t.units)
                tb = np.asarray(d[bname][:], dtype=np.float64)
                y0 = float(netCDF4.date2num(cftime.datetime(year, 1, 1, calendar=cal),
                                            units, calendar=cal))
                y1 = float(netCDF4.date2num(cftime.datetime(year + 1, 1, 1, calendar=cal),
                                            units, calendar=cal))
                lo, hi = np.maximum(tb[:, 0], y0), np.minimum(tb[:, 1], y1)
                idx = np.flatnonzero(hi - lo > TIME_TOL_DAYS)
                if not idx.size:
                    continue
                v = d[var]
                v.set_auto_maskandscale(True)
                if v.ndim != 3:
                    raise CheckError(f"{var} has {v.ndim} dimensions; a regional number "
                                     f"is made from a (time, lat, lon) field")
                x = _clean(v[idx.tolist()]).reshape(len(idx), -1)[:, cells]
                ok = np.isfinite(x)
                num = np.where(ok, x, 0.0) @ W
                den = ok.astype(np.float64) @ W
                with np.errstate(invalid="ignore", divide="ignore"):
                    means.append(np.where(den > 0, num / np.where(den > 0, den, 1.0),
                                          np.nan))
                spans += [(float(lo[k] - y0), float(hi[k] - y0)) for k in idx]
                length = y1 - y0
        if not spans:
            return None, np.zeros(0), f"no record of {var} in {year}"
        order = np.argsort([s[0] for s in spans])
        sp = [spans[k] for k in order]
        m = np.concatenate(means, axis=0)[order]
        edge = 0.0
        for a, b in sp:
            if abs(a - edge) > TIME_TOL_DAYS:
                return None, np.zeros(0), (f"{year}: the records leave a gap or overlap at "
                                           f"day {edge:.4f}")
            edge = b
        if abs(edge - length) > TIME_TOL_DAYS:
            return None, np.zeros(0), f"{year}: the records end at day {edge:.4f} of {length:g}"
        return m, np.array([b - a for a, b in sp]), None


# ─────────────────────────────────────────────────────────────────────────────
# THE MASKS
# ─────────────────────────────────────────────────────────────────────────────
def _rings(geo) -> List[List[np.ndarray]]:
    if geo.get("type") == "FeatureCollection":
        gs = [f.get("geometry") or {} for f in geo.get("features") or []]
    elif geo.get("type") == "Feature":
        gs = [geo.get("geometry") or {}]
    else:
        gs = [geo]
    out = []
    for g in gs:
        parts = [g["coordinates"]] if g.get("type") == "Polygon" else \
            g["coordinates"] if g.get("type") == "MultiPolygon" else None
        if parts is None:
            raise CheckError(f"geometry {g.get('type')!r} is not a polygon")
        for p in parts:
            rs = []
            for r in p:
                a = np.asarray(r, dtype=np.float64)[:, :2].copy()
                a[:, 0] = np.unwrap(_lon180(a[:, 0]), period=360.0)
                rs.append(a)
            out.append(rs)
    if not out:
        raise CheckError("no polygon in the GeoJSON")
    return out


def polygon_cells(path, grid) -> np.ndarray:
    """Cells whose centre is inside the polygon: matplotlib's test, exterior
    minus holes, union over polygons, with the centres shifted by 360 degrees
    for a ring that runs past 180."""
    from matplotlib.path import Path as MPath
    pts = np.column_stack([_lon180(grid["lon"]).ravel(), grid["lat"].ravel()])
    inside = np.zeros(len(pts), dtype=bool)
    for rings in _rings(json.loads(Path(path).read_text())):
        lo = min(float(r[:, 0].min()) for r in rings)
        hi = max(float(r[:, 0].max()) for r in rings)
        for s in [0.0] + ([360.0] if hi > 180.0 else []) + ([-360.0] if lo < -180.0 else []):
            q = pts + np.array([s, 0.0])
            m = MPath(rings[0]).contains_points(q)
            for h in rings[1:]:
                m &= ~MPath(h).contains_points(q)
            inside |= m
    return inside.reshape(grid["lat"].shape)


_OPS = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal,
        "==": np.equal, "!=": np.not_equal}


def field_cells(r: Dict[str, Any], grid) -> np.ndarray:
    """Cells where the registered field's rule holds (var op value, combined
    over the leading axes by any or all); a missing value never selects."""
    import netCDF4
    op, over = r.get("op", ">"), r.get("over", "any")
    if op not in _OPS or over not in ("any", "all"):
        raise CheckError(f"region rule {op!r} over {over!r} is not one this check knows")
    with netCDF4.Dataset(r["field"]) as f:
        pair = next(((a, b) for a, b in (("lat", "lon"), ("LATIXY", "LONGXY"), ("yc", "xc"))
                     if a in f.variables and b in f.variables), None)
        if pair is None:
            raise CheckError(f"{Path(r['field']).name}: no cell centres")
        la, lo = f[pair[0]], f[pair[1]]
        _same_cells(_clean(la[:]), _clean(lo[:]), grid["lat"], grid["lon"],
                    Path(r["field"]).name)
        dims = (la.dimensions[0], lo.dimensions[0]) if la.ndim == 1 else tuple(la.dimensions)
        v = f[r["var"]]
        if tuple(v.dimensions[-2:]) != tuple(dims):
            raise CheckError(f"{r['var']} runs {v.dimensions}, its cells {dims}")
        a = _clean(v[:])
    with np.errstate(invalid="ignore"):
        sel = np.where(np.isfinite(a), _OPS[op](a, float(r.get("value", 0.0))), False)
    lead = tuple(range(a.ndim - 2))
    if lead:
        sel = sel.any(axis=lead) if over == "any" else sel.all(axis=lead)
    return np.asarray(sel, dtype=bool)


def band_rule(label: str):
    """(low, high) of an elevation band, read back from its name: 'below 1000 m',
    '1000-1500 m', '2000 m and above'. None when the name is not a band."""
    s = label.strip()
    m = re.fullmatch(r"below (-?[\d.]+) m", s)
    if m:
        return None, float(m.group(1))
    m = re.fullmatch(r"(-?[\d.]+)-(-?[\d.]+) m", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.fullmatch(r"(-?[\d.]+) m and above", s)
    if m:
        return float(m.group(1)), None
    return None


def _band(topo: np.ndarray, lohi) -> np.ndarray:
    lo, hi = lohi
    m = np.isfinite(topo)
    if lo is not None:
        m &= topo >= lo
    if hi is not None:
        m &= topo < hi
    return m


def region_mask(name: str, grid, surface: Dict[str, np.ndarray],
                registered: Dict[str, Dict], cache: Dict[str, np.ndarray]) -> np.ndarray:
    """The cells of one region the table names, rebuilt from its name:
    'vegetated cells'; a band; '<registered>, vegetated'; or '<registered>,
    vegetated, <band>'. Every region is restricted to the vegetated cells."""
    veg = surface["veg"]
    if name == VEGETATED:
        return veg.copy()
    b = band_rule(name)
    if b is not None:
        return veg & _band(surface["topo"], b)
    head, sep, tail = name.partition(VEG_SUFFIX)
    if not sep:
        raise CheckError(f"region {name!r}: not a name this check can rebuild")
    if head not in cache:
        r = registered.get(head)
        if not r:
            raise CheckError(f"region {head!r} is not in the control's registry entry")
        if r.get("polygon"):
            cache[head] = polygon_cells(r["polygon"], grid)
        elif r.get("field"):
            cache[head] = field_cells(r, grid)
        else:
            raise CheckError(f"region {head!r} names neither a polygon nor a field")
    m = cache[head] & veg
    tail = tail.lstrip(", ").strip()
    if tail:
        b = band_rule(tail)
        if b is None:
            raise CheckError(f"region {name!r}: {tail!r} is not an elevation band")
        m &= _band(surface["topo"], b)
    return m


def read_surface(path, grid) -> Dict[str, np.ndarray]:
    import netCDF4
    with netCDF4.Dataset(path) as s:
        for v in ("LATIXY", "LONGXY", "PCT_NATVEG", "TOPO"):
            if v not in s.variables:
                raise CheckError(f"{Path(path).name} has no {v}")
        _same_cells(_clean(s["LATIXY"][:]), _clean(s["LONGXY"][:]), grid["lat"],
                    grid["lon"], f"surface {Path(path).name}")
        nat, topo = _clean(s["PCT_NATVEG"][:]), _clean(s["TOPO"][:])
    return {"veg": np.isfinite(nat) & (nat >= 100.0 - 1e-6), "topo": topo}


# ─────────────────────────────────────────────────────────────────────────────
# THE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
class _Ledger:
    def __init__(self):
        self.n, self.absent, self.bad, self.worst = 0, 0, [], None

    def add(self, where: Dict[str, Any], table, mine, scale=None, atol=ATOL,
            why: Optional[str] = None, exact: bool = False) -> None:
        if table is None and mine is None:
            self.absent += 1
            return
        row = dict(where, table=table, recomputed=mine)
        if table is None or mine is None or isinstance(table, bool) or \
                not isinstance(table, (int, float)):
            self.n += 1
            row["why"] = why or ("the table has a number the check could not make"
                                 if mine is None else
                                 "the table has no number where the check made one"
                                 if table is None else "the table's entry is not a number")
            self.bad.append(row)
            return
        self.n += 1
        t, m = float(table), float(mine)
        d = abs(t - m)
        s = max(abs(m), abs(t)) if scale is None else float(scale)
        rel = d / s if s > 0 else (0.0 if d == 0 else math.inf)
        row.update(abs_diff=d, rel_diff=rel)
        if self.worst is None or rel > self.worst["rel_diff"]:
            self.worst = dict(row)
        allowed = 0.0 if exact else RTOL * s + atol
        if not (d <= allowed) or not math.isfinite(t):
            row["allowed"] = allowed
            self.bad.append(row)


def _year_value(means, days, rate: bool) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if means is None:
        return None, None
    if not np.all(np.isfinite(means)):
        return None, "a record has no value in the region"
    if rate:
        return (means * days[:, None]).sum(axis=0) * SECONDS_PER_DAY, None
    return (means * days[:, None]).sum(axis=0) / days.sum(), None


def recompute_runs(table, masks: Dict[str, np.ndarray], grid,
                   extra_fields: Sequence[str] = ()) -> Dict[str, Any]:
    """{run: {"values": {region: {field: {year: v}}}, "how": {field: ...},
    "stream": {field: ...}, "why": {...}}} from each run's own history."""
    regs = list(masks)
    out: Dict[str, Any] = {}
    years = sorted({int(y) for y in table.get("years") or []})
    for run, r in (table.get("runs") or {}).items():
        h = History(r["run_dir"])
        g = h.grid()
        if g["weight"].shape != grid["weight"].shape:
            raise CheckError(f"run {run}: its grid {g['weight'].shape} is not the "
                             f"control's {grid['weight'].shape}")
        union = np.zeros(g["weight"].shape, dtype=bool)
        for m in masks.values():
            union |= m
        cells = np.flatnonzero(union.ravel())
        W = np.stack([np.where(masks[k].ravel()[cells], g["weight"].ravel()[cells], 0.0)
                      for k in regs], axis=1)
        fields = [f for f in (r.get("fields") or {}) if f != BURNED_AREA]
        fields += [f for f in extra_fields if f not in fields]
        ys = sorted(set(years) | {int(y) for reg in (r.get("values") or {}).values()
                                  for f in (reg or {}).values() for y in (f or {})})
        rec = {"values": {k: {} for k in regs}, "how": {}, "stream": {}, "why": {}}
        for f in fields:
            st = h.stream_for(f)
            if st is None:
                rec["why"][f] = f"{f} is in none of the run's streams"
                continue
            rate = per_second(f, h.units(st, f))
            rec["how"][f], rec["stream"][f] = ("sum" if rate else "mean"), st
            for y in ys:
                means, days, why = h.year(st, f, y, W, cells)
                vals, why2 = _year_value(means, days, rate)
                for j, k in enumerate(regs):
                    rec["values"][k].setdefault(f, {})[str(y)] = \
                        None if vals is None else float(vals[j])
                if why or why2:
                    rec["why"][f"{f} {y}"] = why or why2
        out[run] = rec
    return out


def _anomaly_block(by: Dict[int, Optional[float]], normal: Sequence[int],
                   target: Sequence[int], percent: bool) -> Optional[Dict[str, Any]]:
    """The event numbers from recomputed years, by their definitions: the
    normal is the mean of the normal years, the anomaly the value minus it, the
    percent 100 x anomaly / normal, the rank 1 + the number of years lower."""
    nv = [by.get(y) for y in normal]
    if any(v is None for v in nv):
        return None
    nv = np.array(nv, dtype=np.float64)
    mean = float(nv.sum() / nv.size)
    have = {y: v for y, v in by.items() if v is not None}
    out = {"normal": mean, "normal_min": float(nv.min()), "normal_max": float(nv.max()),
           "normal_sd": (float(np.sqrt(((nv - mean) ** 2).sum() / (nv.size - 1)))
                         if nv.size > 1 else None),
           "by_year": {str(y): v for y, v in by.items()}, "years": {}}
    for y in target:
        v = by.get(y)
        if v is None:
            out["years"][str(y)] = None
            continue
        out["years"][str(y)] = {
            "value": v, "anomaly": v - mean,
            "anomaly_pct": (100.0 * (v - mean) / mean) if percent and mean != 0 else None,
            "rank_lowest_first": 1 + sum(1 for w in have.values() if w < v),
            "of": len(have)}
    return out


def _yearkey(d) -> Dict[str, Any]:
    return {str(k): v for k, v in (d or {}).items()}


def compare(table: Dict[str, Any], mine: Dict[str, Any], masks, grid,
            registered: Dict[str, Dict], observations: Dict[str, Dict]) -> Dict[str, Any]:
    L = _Ledger()
    notes: List[str] = []
    w = grid["weight"]
    area = {k: float(w[m].sum()) for k, m in masks.items()}
    # regions: cells exactly, area within the tolerance
    for k, info in (table.get("regions") or {}).items():
        L.add({"section": "regions", "region": k, "what": "cells"}, info.get("cells"),
              int(masks[k].sum()) if k in masks else None, exact=True)
        L.add({"section": "regions", "region": k, "what": "area_km2"}, info.get("area_km2"),
              area.get(k))

    def val(run, reg, f, y):
        if f == BURNED_AREA:
            x = ((mine.get(run) or {}).get("values") or {}).get(reg, {}).get(
                BURNED_FRACTION, {}).get(y)
            return None if x is None or reg not in area else x * area[reg]
        return ((mine.get(run) or {}).get("values") or {}).get(reg, {}).get(f, {}).get(y)

    # each run's values, and how each field's year is made
    for run, r in (table.get("runs") or {}).items():
        for f, meta in (r.get("fields") or {}).items():
            if f == BURNED_AREA:
                continue
            got = (mine.get(run) or {}).get("how", {}).get(f)
            if got and meta.get("how") and got != meta.get("how"):
                L.bad.append({"section": "fields", "run": run, "field": f,
                              "table": meta.get("how"), "recomputed": got,
                              "why": "the table and the check make this field's year "
                                     "differently (a total against a mean)"})
            st = (mine.get(run) or {}).get("stream", {}).get(f)
            if st and meta.get("stream") and st != meta.get("stream"):
                notes.append(f"run {run}, {f}: the table read {meta['stream']}, the check "
                             f"{st}")
        for reg, fields in (r.get("values") or {}).items():
            for f, years in (fields or {}).items():
                for y, x in _yearkey(years).items():
                    m = val(run, reg, f, y)
                    why = (mine.get(run) or {}).get("why", {}).get(f"{f} {y}") or \
                        (mine.get(run) or {}).get("why", {}).get(f)
                    L.add({"section": "values", "run": run, "region": reg, "field": f,
                           "year": y}, x, m, why=why)
    # each run minus the control
    for run, regs in (table.get("differences") or {}).items():
        for reg, fields in (regs or {}).items():
            for f, years in (fields or {}).items():
                for y, x in _yearkey(years).items():
                    a, b = val(run, reg, f, y), val("control", reg, f, y)
                    L.add({"section": "differences", "run": run, "region": reg, "field": f,
                           "year": y}, x, None if a is None or b is None else a - b,
                          scale=max(abs(a or 0.0), abs(b or 0.0)))
    # the event: each target year of the control against its normal years
    ev = table.get("event") or {}
    if ev.get("anomalies"):
        normal = [int(y) for y in ev.get("normal_years") or []]
        target = [int(y) for y in ev.get("target_years") or []]
        for reg, fields in ev["anomalies"].items():
            for f, a in (fields or {}).items():
                by = {int(y): val("control", reg, f, str(y))
                      for y in (a.get("by_year") or {}) or (normal + target)}
                b = _anomaly_block(by, normal, target, percent=True) or {}
                at = {"section": "event", "region": reg, "field": f}
                big = max([abs(v) for v in by.values() if v is not None] + [0.0])
                for k in ("normal", "normal_min", "normal_max"):
                    L.add(dict(at, what=k), a.get(k), b.get(k), scale=big)
                if a.get("normal_sd") is not None or b.get("normal_sd") is not None:
                    L.add(dict(at, what="normal_sd"), a.get("normal_sd"), b.get("normal_sd"),
                          scale=big)
                for y, x in _yearkey(a.get("by_year")).items():
                    L.add(dict(at, what="by_year", year=y), x, (b.get("by_year") or {}).get(y))
                for y, yt in _yearkey(a.get("years")).items():
                    ym = (b.get("years") or {}).get(y)
                    if yt is None or ym is None:
                        L.add(dict(at, what="year", year=y), None if yt is None else
                              yt.get("value"), None if ym is None else ym["value"])
                        continue
                    nrm = abs(b.get("normal") or 0.0)
                    L.add(dict(at, what="value", year=y), yt.get("value"), ym["value"])
                    L.add(dict(at, what="anomaly", year=y), yt.get("anomaly"), ym["anomaly"],
                          scale=big)
                    # a percent the table leaves out (a temperature, a zero
                    # normal) is not a number to compare
                    if yt.get("anomaly_pct") is not None:
                        L.add(dict(at, what="anomaly_pct", year=y), yt["anomaly_pct"],
                              ym["anomaly_pct"], scale=(100.0 * big / nrm) if nrm else None)
                    for k in ("rank_lowest_first", "of"):
                        if k in yt:
                            L.add(dict(at, what=k, year=y), yt.get(k), ym.get(k), exact=True)
    # the share of each region a confined change reaches (rounded to 4 places)
    for run, cs in (table.get("changed_share") or {}).items():
        where = (cs or {}).get("region")
        try:
            r = registered.get(where) or {}
            c = polygon_cells(r["polygon"], grid) if r.get("polygon") else \
                field_cells(r, grid) if r.get("field") else None
        except Exception as e:                                      # noqa: BLE001
            c, notes_e = None, f"{type(e).__name__}: {e}"
            notes.append(f"changed_share of run {run}: {notes_e}")
        for reg, x in ((cs or {}).get("share") or {}).items():
            m = None
            if c is not None and reg in masks and area.get(reg):
                m = round(float(w[masks[reg] & c].sum()) / area[reg], 4)
            L.add({"section": "changed_share", "run": run, "region": reg}, x, m,
                  atol=SHARE_ATOL)
    # an older table's burned area beside an observation (before 2026-10-27)
    for name, o in (table.get("observed") or {}).items():
        for run, regs in (o.get("model") or {}).items():
            for reg, years in (regs or {}).items():
                for y, x in _yearkey(years).items():
                    L.add({"section": "observed", "what": "model", "run": run, "region": reg,
                           "year": y}, x, val(run, reg, BURNED_AREA, y))
        obs = _observed(o, observations, masks, grid, notes)
        for reg, years in (o.get("values") or {}).items():
            for y, x in _yearkey(years).items():
                L.add({"section": "observed", "what": "observation", "region": reg,
                       "year": y}, x, (obs.get(reg) or {}).get(y))
    return {"ledger": L, "notes": notes}


def _observed(o, observations, masks, grid, notes) -> Dict[str, Dict[str, float]]:
    """An older table's observed burned area: the registered observation's
    burned fraction per cell and year, missing as none burned, times each
    cell's weight, summed over the region."""
    import netCDF4
    ent = next((e for e in observations.values() if e.get("what") == o.get("source")),
               None) or next((e for e in observations.values()
                              if e.get("var") == "burned_fraction_year"), None)
    if not ent:
        notes.append("the table's observed burned area: no registered observation to "
                     "recompute it from")
        return {}
    with netCDF4.Dataset(ent["path"]) as f:
        _same_cells(_clean(f["lat"][:]), _clean(f["lon"][:]), grid["lat"], grid["lon"],
                    Path(ent["path"]).name)
        years = [int(y) for y in f["year"][:]]
        bf = np.nan_to_num(_clean(f[ent["var"]][:]))
    w = grid["weight"]
    return {reg: {str(y): float((bf[k] * w)[m].sum()) for k, y in enumerate(years)}
            for reg, m in masks.items()}


def check(run_dir, registry: Optional[Dict[str, Any]] = None,
          write: bool = True) -> Dict[str, Any]:
    """Recompute every number of the run's regional table and compare.

    Returns, and writes to 04_analysis/numbers_check.json: pass, how many
    numbers were compared, every mismatch, the largest relative difference,
    the method and the tolerances. A table that cannot be recomputed at all
    (an input missing) does not pass, and says why under `error`."""
    rd = Path(run_dir)
    rec: Dict[str, Any] = {
        "kind": "numbers_check", "checked_at": datetime.now().isoformat(timespec="seconds"),
        "table": TABLE, "pass": False, "n_compared": 0, "n_mismatched": 0,
        "n_absent_in_both": 0, "mismatches": [], "largest_relative_difference": None,
        "tolerance": {"relative": RTOL, "absolute": ATOL, "changed_share_absolute": SHARE_ATOL,
                      "rule": "|table - recomputed| <= relative x scale + absolute, the "
                              "scale being the size of the quantities the number is made "
                              "from; cell counts and ranks must be equal"},
        "method": METHOD, "sources": {}, "notes": [], "not_checked": []}
    try:
        table = json.loads((rd / TABLE).read_text())
        if table.get("error"):
            raise CheckError(f"the table holds no numbers: {table['error']}")
        inp = inputs(rd, registry)
        ctl_run = ((table.get("runs") or {}).get("control") or {}).get("run_dir")
        if not ctl_run:
            raise CheckError("the table has no control run")
        h = History(ctl_run)
        grid = h.grid()
        surf_path = _fsurdat(inp["case_dir"], ctl_run)
        surface = read_surface(surf_path, grid)
        cache: Dict[str, np.ndarray] = {}
        masks: Dict[str, np.ndarray] = {}
        for name in (table.get("regions") or {}):
            masks[name] = region_mask(name, grid, surface, inp["regions"], cache)
        extra = [BURNED_FRACTION] if table.get("observed") else []
        mine = recompute_runs(table, masks, grid, extra)
        res = compare(table, mine, masks, grid, inp["regions"], inp["observations"])
        L = res["ledger"]
        rec.update({
            "n_compared": L.n, "n_absent_in_both": L.absent, "n_mismatched": len(L.bad),
            "mismatches": L.bad, "largest_relative_difference": L.worst,
            "pass": not L.bad and L.n > 0,
            "notes": res["notes"],
            "not_checked": [k for k in table if k not in KNOWN],
            "sources": {"control": inp["control"], "control_case_dir": inp["case_dir"],
                        "surface": str(surf_path),
                        "regions": {k: (v.get("polygon") or v.get("field"))
                                    for k, v in inp["regions"].items()},
                        "runs": {k: v.get("run_dir") for k, v in
                                 (table.get("runs") or {}).items()},
                        "registry_entry_found": inp["registry_entry"]}})
        if L.n == 0:
            rec["error"] = "the table holds no number to compare"
    except Exception as e:                                          # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    if write:
        out = rd / "04_analysis"
        out.mkdir(parents=True, exist_ok=True)
        (out / CHECK_JSON).write_text(json.dumps(rec, indent=1, default=str))
    return rec


METHOD = (
    "Each number in 03_results/regional.json made again from each run's own history "
    "by code that shares no reduction code with the server's domain_analysis.py, "
    "gridded.py or regions.py: history read with netCDF4; fill (masked, non-finite "
    "or above 1e30 in size) as no value; each record weighted by the days of its own "
    "time_bounds inside the calendar year, the records required to cover the year "
    "without gap or overlap; a region's value at a record is the mean over its cells "
    "with a value, weighted by area x landfrac from the history; a field whose units "
    "are per second (and FAREA_BURNED) is totalled over the year, any other field is "
    "the year's time-weighted mean; cells from the surface file the control's lnd_in "
    "names (vegetated: PCT_NATVEG at 100; elevation bands: TOPO, with each band's "
    "edges read from its name), a polygon region by matplotlib's point-in-polygon test "
    "on cell centres, a gridded-field region by its rule on the file; burned area is "
    "the year's burned fraction times the region's area; differences, the event's "
    "normal, anomalies, percents and ranks by their definitions from the recomputed "
    "values.")


# ─────────────────────────────────────────────────────────────────────────────
# THE GATE
# ─────────────────────────────────────────────────────────────────────────────
def _fmt(x) -> str:
    if x is None:
        return "none"
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return f"{float(x):.6g}"
    return str(x)


def _where(row: Dict[str, Any]) -> str:
    parts = [row.get("section")]
    for k in ("run", "region", "field", "what", "year"):
        if row.get(k) is not None:
            parts.append(str(row[k]))
    return ", ".join(str(p) for p in parts if p)


def reason(rec: Dict[str, Any]) -> str:
    """One line: why the numbers are withheld."""
    if rec.get("error") and not rec.get("n_compared"):
        return f"the regional numbers could not be recomputed ({rec['error']})"
    return (f"{rec.get('n_mismatched')} of {rec.get('n_compared')} regional numbers did "
            f"not reproduce when recomputed from the model output")


def withheld_markdown(rec: Dict[str, Any]) -> str:
    """The numbers report in place of the numbers: how many did not reproduce,
    and the mismatch table. A renderer: every value is the record's."""
    lines = [f"# WITHHELD: {reason(rec)}", "",
             "The numbers in 03_results/regional.json were made again from each run's own "
             "history by a second, independent code path (04_analysis/numbers_check.json "
             "says how). "
             + ("They could not be made again, so they are not shown."
                if rec.get("error") and not rec.get("n_compared") else
                f"{rec.get('n_mismatched')} of {rec.get('n_compared')} differ by more than "
                f"the tolerance (relative {RTOL:g}, with an absolute floor of {ATOL:g})."),
             "This report is withheld and the Analyzer was not run, until the difference "
             "is explained.", ""]
    rows = rec.get("mismatches") or []
    if rows:
        lines += ["| number | in the table | recomputed | difference | relative to its "
                  "scale | why |", "|---|---|---|---|---|---|"]
        for r in rows[:MAX_ROWS_SHOWN]:
            lines.append(f"| {_where(r)} | {_fmt(r.get('table'))} | "
                         f"{_fmt(r.get('recomputed'))} | {_fmt(r.get('abs_diff'))} | "
                         f"{_fmt(r.get('rel_diff'))} | {r.get('why') or ''} |")
        if len(rows) > MAX_ROWS_SHOWN:
            lines.append(f"\n{len(rows) - MAX_ROWS_SHOWN} more in 04_analysis/{CHECK_JSON}.")
        lines.append("")
    if rec.get("error"):
        lines += [f"The check stopped: {rec['error']}", ""]
    return "\n".join(lines)


def gate(run_dir, registry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The analyze stage's gate, run after regional.json is made and before the
    numbers report and the Analyzer.

    {"pass", "reason", "record", "report", "n_compared", "n_mismatched"}. On a
    failure, 04_analysis/DOMAIN_REPORT.md is written WITHHELD (its first line
    says how many numbers did not reproduce, then the mismatch table) and the
    caller runs neither the numbers report nor the Analyzer.

    None when the table holds no numbers (absent, or the server's error in
    their place): that failure is the numbers report's to state, and there is
    nothing here to recompute."""
    rd = Path(run_dir)
    t = _read_json(rd / TABLE)
    if not t or t.get("error"):
        return None
    try:
        rec = check(rd, registry=registry)
    except Exception as e:                                          # noqa: BLE001
        rec = {"pass": False, "n_compared": 0, "n_mismatched": 0,
               "error": f"{type(e).__name__}: {e}"}
    out = {"pass": bool(rec.get("pass")), "record": str(rd / "04_analysis" / CHECK_JSON),
           "n_compared": rec.get("n_compared"), "n_mismatched": rec.get("n_mismatched"),
           "reason": None, "report": None}
    if not out["pass"]:
        p = rd / "04_analysis" / "DOMAIN_REPORT.md"
        out["reason"] = reason(rec)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(withheld_markdown(rec))
            out["report"] = str(p)
        except OSError as e:
            out["reason"] += f" (the withheld report could not be written: {e})"
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: numbers_check.py <run_dir>")
    r = check(sys.argv[1])
    w = r.get("largest_relative_difference") or {}
    print(json.dumps({"pass": r["pass"], "n_compared": r["n_compared"],
                      "n_mismatched": r["n_mismatched"], "error": r.get("error"),
                      "largest_relative_difference": w.get("rel_diff"),
                      "at": _where(w) if w else None}, indent=1))
    sys.exit(0 if r["pass"] else 1)
