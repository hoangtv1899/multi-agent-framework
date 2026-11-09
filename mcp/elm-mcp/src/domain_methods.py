#!/usr/bin/env python3
"""The method catalog of a whole-domain study: each analysis method with its
contract, run on the suite or on the run's daily history.

THE METHOD CATALOG (2026-11-05). A method is reusable on any watershed: no
watershed's value lives in it. Regions, elevation bands, normal years,
vegetation and the month a year starts come from the study's records (the
suite's masks and manifest, the control's registry entry, the plan). Whether a
method applies is decided by code, per cell and period, from the data, and
"not reliably estimable, because ..." is a proper outcome with its reason and
its count: a method never fills in a value it cannot estimate. Settings are
explicit and versioned, and every result carries the method's id, version and
settings; a changed default raises the version.

Each method is a function carrying a Method record (fn.method), the contract
docs/ANALYSIS_METHODS.md renders (tools/make_methods_doc.py): what it
estimates, its inputs, its applicability checks (callables the code itself
calls, so the contract cannot drift from what runs), its settings with the
reason for each default and the named alternatives computed beside it, its
outcomes, outputs, limitations and the tests that hold it. People write that
text, never the framework's language models; an entry stays status
"draft" until the user has reviewed it (pytest --review-gate fails while one
is). On 2026-11-05 the user approved every contract and setting for
integration, the two-peak settings of growth_peak as provisional values: an
approval for integration, not a final scientific approval. Measured values of any study stay out of the records, since the Analyzer
reads the rendered file before it plans.

  anomaly(suite, field, period, normal_years, region=None)   value minus normal
  difference(run, control, field, period, region=None)       run minus control
  snow_peak, snow_gone, leaf_green_up, photosynthesis_onset, growth_peak,
  half_runoff (period)       per cell, a date or a reason, every setting
  timing_run(run_dir, years, control)   those six from the daily history,
                                        each year file read once
  timing_summary(...)        region distributions, shifts, sensitivity record
  screen(suite, region, year, normal_years, control=None)    the whole table
  overlap(a, b, mask, weights, bands=None)                   two maps
  elevation_split(x, mask, weights, bands, others=None, region=None)
                             one map by band, within the requested region

PRECISION. Maps are float32 in the suite and become float64 here; region
statistics, normals, scores and dates are float64, dates whole days. Every
region statistic is area-weighted (cell area times land fraction, the suite's
weight_km2, the weights regions.reduce_regions uses); a cell without a value
leaves the numerator and the denominator alike, and each statistic reports
the share of the region's area it covers. Quantiles are of equal area: the
rule of wquantile, which for equal areas is the ordinary median.

REVIEW ROUND 1 (2026-11-05). Whether a change is also given as a percent is
the field notes' reporting policy (percent.allowed), apart from the physical
scale, and a field kept in percent changes in percentage points. Every method
reports the share of the area its inputs cover and keeps missing cell values
apart from missing time records. Timing comparisons give the shift on the
cells eligible in both and, beside it, the change of eligibility (cells and
area eligible in both, in one only, in neither). Each contract states its
numerical contract, and each method carries four statuses kept apart
(METHOD_STATUS: review, implemented, tested, available to the Analyzer).

THE RANGE ACROSS SETTINGS (the user's decisions on G part 1, 2026-11-08).
Every timing result also gives each evaluated setting's shift on the cells
eligible under every setting (in the year and in its baseline), the default,
the smallest and the largest shift with their settings, whether the largest
minus the smallest exceeds 7 days (a claim giving the shift must then give
the three) and whether the settings reverse the direction of the shift
(across_settings). The six timing measures are at behaviour version 3.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field as dc_field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import domain_analysis as DA
import domain_catalog as DC
import gridded as G
import regions as R

DRAFT = DC.DRAFT_STATE
APPROVED = DC.APPROVED_STATE           # for integration, not a final scientific approval
PROVISIONAL = DC.PROVISIONAL_STATE     # approved for integration as a provisional value
REVIEWED = DC.REVIEWED_STATE
MECHANISMS_PATH = Path(__file__).resolve().parents[1] / "domain_mechanisms.json"
SEASON_NAMES = tuple(n for n, _ in DA.SEASONS)
KINDS = ("year", "wateryear", "season", "month")
# The registry keys that name the months a year and a water year start in
# (rule 1 of the plan: from the study's records, never from a method).
YEAR_START_KEY = "year_start_month"
WATER_YEAR_START_KEY = "water_year_start_month"
DEFAULT_YEAR_START = 1
DEFAULT_WATER_YEAR_START = 10
DAY_SECONDS = 86400.0


class MethodError(Exception):
    """A method cannot run as asked, and the message says why."""


# ─────────────────────────────────────────────────────────────────────────────
# THE CONTRACT
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Check:
    """One applicability condition: its words and the callable that decides it."""
    id: str
    text: str
    test: Callable[..., Any]


@dataclass(frozen=True, eq=False)
class Setting:
    """One setting: its default, why, and the alternatives computed beside it.
    Its review record is its method's (a setting has no versions of its own),
    set on registration from METHOD_REVIEW and SETTING_REVIEW."""
    name: str
    default: Any
    units: str
    reason: str
    alternatives: Tuple[Any, ...] = ()
    review: Dict[str, Any] = dc_field(default_factory=lambda: {"state": DRAFT})

    @property
    def status(self) -> str:
        return str((self.review or {}).get("state") or DRAFT)


@dataclass(frozen=True)
class Method:
    """A method's contract (rule 2 of the plan's section 3.4)."""
    id: str
    version: int
    estimates: str
    inputs: Tuple[Tuple[str, str, str], ...]          # (what, units, resolution)
    fields: Tuple[str, ...]                           # catalog fields read by name
    applicability: Tuple[Check, ...]
    settings: Tuple[Setting, ...]
    outcomes: Tuple[Tuple[str, str], ...]             # (outcome, when)
    outputs: Tuple[str, ...]
    limitations: Tuple[str, ...]
    validation: Tuple[Tuple[str, str], ...]           # (test, data): never values
    run: Optional[Callable[..., Any]] = None
    status: Dict[str, Any] = dc_field(default_factory=dict)   # METHOD_STATUS, on registration
    text_revision: int = 1                            # changes to the contract's words
    numerics: Tuple[Tuple[str, str], ...] = ()        # (part, text): the numerical contract

    def setting(self, name: str) -> Setting:
        for s in self.settings:
            if s.name == name:
                return s
        raise MethodError(f"{self.id} has no setting {name!r}")

    def defaults(self) -> Dict[str, Any]:
        return {s.name: s.default for s in self.settings}

    def variants(self) -> List[Tuple[str, Dict[str, Any]]]:
        """The default, then each named alternative, one setting changed at a
        time: what the suite computes with every result."""
        out = [("default", self.defaults())]
        for s in self.settings:
            for a in s.alternatives:
                d = self.defaults()
                d[s.name] = a
                out.append((f"{s.name}={a:g}" if isinstance(a, (int, float)) else
                            f"{s.name}={a}", d))
        return out

    def stamp(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        return {"method": self.id, "version": self.version, "settings": dict(settings)}


METHODS: Dict[str, Method] = {}


def _register(fn: Callable[..., Any], m: Method) -> Method:
    status = {**METHOD_STATUS[m.id], "review": dict(METHOD_REVIEW.get(m.id) or
                                                    METHOD_STATUS[m.id]["review"]),
              "available_to_analyzer": m.id in ANALYZER_ACCESS}
    settings = tuple(replace(s, review={**status["review"],
                                        **SETTING_REVIEW.get((m.id, s.name), {})})
                     for s in m.settings)
    m = Method(**{**m.__dict__, "run": fn, "status": status, "settings": settings})
    fn.method = m
    METHODS[m.id] = m
    return m


# THE PLAN'S SETTINGS REACH THE METHODS (package E, 2026-11-05). anomaly,
# difference, screen, overlap and elevation_split take `named`, the settings
# the study's recorded plan or an amendment names, and pass it here; what they
# compute for a given setting is unchanged, so no behaviour version moves.
def settings_for(method_id: str, given: Optional[Dict[str, Any]] = None,
                 named: Optional[Dict[str, Sequence[Any]]] = None) -> Dict[str, Any]:
    """The settings a call runs with: the defaults, changed only to a named
    alternative or to a value the study's plan (or an amendment) names in
    `named`. Anything else is refused (rule 5): a setting is chosen before
    the data are seen, not at the call."""
    m = METHODS[method_id]
    out = m.defaults()
    for k, v in (given or {}).items():
        s = m.setting(k)
        if v == s.default or v in s.alternatives or v in tuple((named or {}).get(k, ())):
            out[k] = v
            continue
        raise MethodError(f"{method_id}: {k}={v!r} is neither its default ({s.default!r}), "
                          f"a named alternative {list(s.alternatives)}, nor named by the plan")
    return out


# ONE RENDERER FOR A METHOD'S FULL CONTRACT (closing package E, 2026-11-05).
# docs/ANALYSIS_METHODS.md (tools/make_methods_doc.py) and the Analyzer's
# suite.method_contract call (domain_access) both take a method's section from
# contract_lines, so the call cannot say anything the document does not.
def format_value(v) -> str:
    return f"{v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def _setting_review(s: Setting) -> str:
    rv = s.review or {}
    if rv.get("state") == PROVISIONAL:
        return f"**provisional**: {rv.get('why') or 'pending a further review'}"
    return str(rv.get("state"))


def contract_lines(i: int, m: Method) -> List[str]:
    """Method m's section of docs/ANALYSIS_METHODS.md, numbered i: what it
    estimates, inputs, applicability, settings, outcomes, outputs, numerical
    contract, limitations, validation evidence and its four statuses."""
    L = [f"## {i}. `{m.id}` (version {m.version}, text revision {m.text_revision})", "",
         f"**Estimates.** {m.estimates}", "", "**Inputs.**", ""]
    for what, units, res in m.inputs:
        bits = [b for b in (units and f"units: {units}", res and f"resolution: {res}") if b]
        L.append(f"* {what}" + (f" ({'; '.join(bits)})" if bits else ""))
    L += ["", "**Applies where.** Checked by code, per cell and period:", ""]
    for c in m.applicability:
        L.append(f"* `{c.id}`: {c.text}")
    L += [""]
    if m.settings:
        L += ["**Settings.**", "",
              "| setting | default | units | why this default | alternatives computed beside it "
              "| status |",
              "|---|---|---|---|---|---|"]
        for s in m.settings:
            alts = ", ".join(format_value(a) for a in s.alternatives) or "none"
            L.append(f"| `{s.name}` | {format_value(s.default)} | {s.units} | {s.reason} | {alts} "
                     f"| {_setting_review(s)} |")
        L.append("")
    else:
        L += ["**Settings.** None.", ""]
    L += ["**Outcomes and quality checks.**", ""]
    for what, when in m.outcomes:
        L.append(f"* {what}: {when}")
    L += ["", "**Outputs.**", ""] + [f"* {o}" for o in m.outputs]
    L += ["", "**Numerical contract.**", ""] + [f"* {k}: {x}" for k, x in m.numerics]
    L += ["", "**Limitations.**", ""] + [f"* {x}" for x in m.limitations]
    L += ["", "**Validation evidence** (tests and data; results are in the test runs, not here).",
          ""] + [f"* `{t}`: {d}" for t, d in m.validation]
    L += ["", "**Status** (each kept apart):", ""] + [f"* {x}" for x in DC.status_lines(m.status)]
    return L + [""]


# ─────────────────────────────────────────────────────────────────────────────
# AREA-WEIGHTED STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def wquantile(values, weights, q: float) -> Optional[float]:
    """The q-quantile of values, each holding its weight's share of the area.

    EQUAL-AREA QUANTILES (2026-11-05). The smallest value by which the
    cumulative share of the area reaches q; where it reaches q exactly at the
    end of a value's share, the mean of that value and the next. The result
    is one of the values (a whole day, for dates) or the midpoint of two, and
    for equal areas it is the ordinary median (the mean of the two middle
    values of an even count): numpy's 'averaged_inverted_cdf'."""
    x = np.asarray(values, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    if not x.size:
        return None
    o = np.argsort(x, kind="stable")
    x, w = x[o], w[o]
    s = np.cumsum(w)
    t, tol = float(q) * s[-1], 1e-12 * s[-1]
    k = min(int(np.searchsorted(s, t - tol, side="left")), x.size - 1)
    if abs(s[k] - t) <= tol and k + 1 < x.size:
        return float(0.5 * (x[k] + x[k + 1]))
    return float(x[k])


def wmean(x, mask, w) -> Optional[float]:
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(x)
    return float((w[m] * x[m]).sum() / w[m].sum()) if m.any() and w[m].sum() > 0 else None


def covered_share(x, mask, w) -> float:
    m = np.asarray(mask, dtype=bool)
    tot = float(w[m].sum())
    return float(w[m & np.isfinite(x)].sum() / tot) if tot > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PERIODS
# ─────────────────────────────────────────────────────────────────────────────
def parse_period(label) -> Tuple[str, int, Any]:
    """('year', 2015, None), ('wateryear', 2016, None), ('season', 2015,
    'JJA') or ('month', 2015, 7) from the suite's period labels."""
    s = str(label).strip()
    if re.fullmatch(r"WY\d{4}", s):
        return "wateryear", int(s[2:]), None
    if re.fullmatch(r"\d{4}", s):
        return "year", int(s), None
    m = re.fullmatch(r"(\d{4})-(\d{2})", s)
    if m and 1 <= int(m.group(2)) <= 12:
        return "month", int(m.group(1)), int(m.group(2))
    m = re.fullmatch(r"(\d{4})-([A-Z]{3})", s)
    if m and m.group(2) in SEASON_NAMES:
        return "season", int(m.group(1)), m.group(2)
    raise MethodError(f"{label!r} is not a period label (2015, WY2016, 2015-JJA, 2015-07)")


def period_label(kind: str, year: int, part: Any = None) -> str:
    if kind == "year":
        return str(int(year))
    if kind == "wateryear":
        return f"WY{int(year)}"
    if kind == "season":
        return f"{int(year)}-{part}"
    if kind == "month":
        return f"{int(year)}-{int(part):02d}"
    raise MethodError(f"no period kind {kind!r}")


def period_months(kind: str, year: int, part: Any = None) -> List[Tuple[int, int]]:
    """(year, month) of every month a suite period holds: a water year from
    October of the year before (the suite's own water year), DJF with the
    December before."""
    y = int(year)
    if kind == "year":
        return [(y, m) for m in range(1, 13)]
    if kind == "wateryear":
        return [(y - 1, m) for m in range(10, 13)] + [(y, m) for m in range(1, 10)]
    if kind == "season":
        return DA.season_parts(y, part, dict(DA.SEASONS)[part])
    if kind == "month":
        return [(y, int(part))]
    raise MethodError(f"no period kind {kind!r}")


def kind_parts(kind: str) -> List[Any]:
    """The periods of one kind a reference range is taken over: the twelve
    months for a month row, the four seasons for every other row (a year or
    water year has one normal, and a range of one value is nothing)."""
    return list(range(1, 13)) if kind == "month" else list(SEASON_NAMES)


def left_out_why(kind: str, part: Any, normal_year: int,
                 event_years: Iterable[int]) -> Optional[str]:
    """Why a normal year's period stays out of the normal, or None.

    NO EVENT MONTH IN A NORMAL (2026-11-05). The regional table's rule
    (domain_analysis.period_normals): a normal year's DJF holds the December
    before it, so the DJF after an event year holds that event's December and
    is left out. The same holds for a water year, whose October-December are
    the year before's."""
    ev = {int(y) for y in event_years}
    hit = sorted({y for y, _ in period_months(kind, normal_year, part)
                  if y != int(normal_year) and y in ev})
    if not hit:
        return None
    return (f"{period_label(kind, normal_year, part)} holds months of "
            f"{', '.join(str(y) for y in hit)}, an event year, so it is left out of the normal")


def start_months(control: Optional[Dict[str, Any]]) -> Tuple[int, int]:
    """(the month a year starts in, the month a water year starts in), from
    the control's registry entry, January and October when it names none."""
    c = control or {}
    ys = int(c.get(YEAR_START_KEY) or DEFAULT_YEAR_START)
    ws = int(c.get(WATER_YEAR_START_KEY) or DEFAULT_WATER_YEAR_START)
    for k, v in ((YEAR_START_KEY, ys), (WATER_YEAR_START_KEY, ws)):
        if not 1 <= v <= 12:
            raise MethodError(f"the registry's {k} must be a month 1-12, not {v}")
    return ys, ws


# ─────────────────────────────────────────────────────────────────────────────
# THE SUITE, READ
# ─────────────────────────────────────────────────────────────────────────────
class Suite:
    """One built suite (domain_suite.build), read on demand: the methods'
    only door to a run's maps and region series."""

    def __init__(self, suite_dir):
        import netCDF4
        import domain_suite as S          # here, since domain_suite imports this module
        self.dir = Path(str(suite_dir))
        man = S.load(self.dir)
        if man is None:
            raise MethodError(f"no suite manifest in {self.dir}")
        self.manifest = man
        self.masks: Dict[str, np.ndarray] = {}
        self.region_area_km2: Dict[str, float] = {}
        with netCDF4.Dataset(self.dir / S.MASKS_FILE) as d:
            d.set_auto_mask(False)
            self.weights = np.asarray(d["weight_km2"][:], dtype=np.float64)
            for vn, v in d.variables.items():
                if vn.startswith("mask_"):
                    m = np.asarray(v[:]) == 1
                    self.masks[str(v.region)] = m
                    self.region_area_km2[str(v.region)] = float(self.weights[m].sum())
        self._series: Dict[str, Dict[str, Any]] = {}

    @property
    def fields(self) -> List[str]:
        return list(self.manifest.get("fields") or {})

    def units(self, field: str, kind: str) -> str:
        return ((self.manifest["fields"].get(field) or {}).get("units") or {}).get(kind, "")

    def how(self, field: str) -> str:
        return (self.manifest["fields"].get(field) or {}).get("how", "")

    def labels(self, kind: str) -> List[str]:
        return [str(x) for x in (self.manifest.get("periods") or {}).get(kind, [])]

    def absent_why(self, label: str) -> Optional[str]:
        kind = parse_period(label)[0]
        why = ((self.manifest.get("periods_absent") or {}).get(kind) or {}).get(str(label))
        if why:
            return why
        if str(label) not in self.labels(kind):
            return f"{label} is not among the run's {kind} periods"
        return None

    def coverage_mask(self) -> Tuple[np.ndarray, str]:
        """The cells a per-cell result's coverage is counted over: the
        vegetated cells when the suite holds them, else every cell with land."""
        if DA.VEGETATED in self.masks:
            return self.masks[DA.VEGETATED] & (self.weights > 0), DA.VEGETATED
        return self.weights > 0, "every cell with land"

    def mask(self, region: str) -> np.ndarray:
        if region not in self.masks:
            raise MethodError(f"no region {region!r} in the suite (it has {sorted(self.masks)})")
        return self.masks[region]

    def _series_file(self, kind: str) -> Dict[str, Any]:
        if kind not in self._series:
            import netCDF4
            p = self.dir / f"series_{kind}.nc"
            with netCDF4.Dataset(p) as d:
                d.set_auto_mask(False)
                regions = [str(x) for x in np.asarray(d["region"][:], dtype=object)]
                raw = np.asarray(d[kind][:])
                labels = [period_label(kind, int(x)) if kind in ("year", "wateryear") else str(x)
                          for x in raw.tolist()]
                vals = {vn: np.asarray(v[:], dtype=np.float64) for vn, v in d.variables.items()
                        if v.dimensions == ("region", kind)}
            self._series[kind] = {"regions": regions, "labels": labels,
                                  "pos": {lab: i for i, lab in enumerate(labels)}, "vals": vals}
        return self._series[kind]

    def value(self, field: str, region: str, label: str) -> Tuple[Optional[float], Optional[float]]:
        """(the region's value of the field for the period, its covered share),
        (None, None) where the suite holds none."""
        kind = parse_period(label)[0]
        f = self._series_file(kind)
        if field not in f["vals"] or region not in f["regions"] or label not in f["pos"]:
            return None, None
        i, j = f["regions"].index(region), f["pos"][label]
        v = float(f["vals"][field][i, j])
        c = f["vals"].get(field + "_covered")
        cov = float(c[i, j]) if c is not None else None
        return (v if np.isfinite(v) else None), cov

    def map(self, field: str, label: str) -> Optional[np.ndarray]:
        """The field's map for a year, water year or season, float64, NaN where
        a cell has no value; None where the suite holds none."""
        import netCDF4
        kind, year, part = parse_period(label)
        if kind == "month":
            raise MethodError("month maps are not in the suite: the monthly history holds "
                              "them (domain_suite: month_maps)")
        p = self.dir / f"maps_{kind}.nc"
        with netCDF4.Dataset(p) as d:
            d.set_auto_mask(False)
            if field not in d.variables:
                return None
            ys = [int(x) for x in np.asarray(d["year" if kind == "season" else kind][:])]
            if year not in ys:
                return None
            if kind == "season":
                ss = [str(x) for x in np.asarray(d["season"][:], dtype=object)]
                x = np.asarray(d[field][ys.index(year), ss.index(part)], dtype=np.float64)
            else:
                x = np.asarray(d[field][ys.index(year)], dtype=np.float64)
        if self.absent_why(label) or not np.isfinite(x).any():
            return None
        return x

    def band_masks(self, region: str) -> Dict[str, np.ndarray]:
        """The region cut by the control's elevation bands (the suite's band
        masks, from the edges its manifest records)."""
        edges = [float(e) for e in (self.manifest.get("band_edges_m") or [])]
        if not edges:
            raise MethodError("the suite records no elevation band edges")
        lo = [None] + edges
        hi = edges + [None]
        base = self.mask(region)
        out = {}
        for a, b in zip(lo, hi):
            name = R.band_label(a, b)
            if name not in self.masks:
                raise MethodError(f"the suite has no band mask {name!r}")
            out[name] = base & self.masks[name]
        return out

    def other_regions(self, region: str) -> Dict[str, np.ndarray]:
        """Every other registered region (not a band, not the vegetated cells
        the others are cut from), as itself and its complement inside
        `region`: for Naches, burned and unburned."""
        edges = [float(e) for e in (self.manifest.get("band_edges_m") or [])]
        bands = {R.band_label(a, b) for a, b in zip([None] + edges, edges + [None])}
        out = {}
        base = self.mask(region)
        for name, m in self.masks.items():
            if name in (region, DA.VEGETATED) or name in bands or \
                    any(name.endswith(", " + b) for b in bands):
                continue
            out[f"{name}"] = base & m
            out[f"not {name}"] = base & ~m
        return out


def _notes(notes: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    return notes if notes is not None else DC.load_notes()


def catalog_fields(notes: Optional[Dict[str, Dict[str, Any]]] = None) -> List[str]:
    """The catalog's rows: every field note that makes one (FIRE does not)."""
    return [k for k, e in _notes(notes).items() if DC.is_row(e)]


# ─────────────────────────────────────────────────────────────────────────────
# ANOMALY AND DIFFERENCE
# ─────────────────────────────────────────────────────────────────────────────
NEAR_ZERO_SHARE = 0.01


def percent_policy(note: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Whether a change is also given as a percent of its reference: the field
    notes' reporting policy (percent.allowed, with its why), apart from the
    physical scale, and never for an event or a bounded index.

    THE PERCENT POLICY (review round 1, 2026-11-05). The scale says what the
    quantity is physically (a signed flux such as NEE is ratio-scale); whether
    a percent is reported is a separate policy written in the notes, and a
    note without one gives no percent."""
    t = note.get("type")
    if t == "event":
        return False, "an event field has no percent change"
    if t == "bounded":
        return False, "a bounded index has no percent change"
    pc = note.get("percent")
    if not isinstance(pc, dict) or not isinstance(pc.get("allowed"), bool):
        return False, "the field notes give no percent policy for this field"
    if not pc["allowed"]:
        return False, f"the field notes' percent policy: {pc.get('why') or 'not allowed'}"
    return True, None


PERCENT_UNITS = ("%",)


def change_units(units: str) -> str:
    """A change's units: a field kept in percent changes in percentage points."""
    return "percentage points" if str(units or "").strip() in PERCENT_UNITS else units


def missing_cells(x, mask, w) -> Dict[str, Any]:
    """The cells of mask (with land) without a value: their count and their
    share of mask's area (missing cell values, apart from missing records)."""
    m = np.asarray(mask, dtype=bool).ravel() & (np.asarray(w).ravel() > 0)
    ww = np.asarray(w, dtype=np.float64).ravel()
    gone = m & ~np.isfinite(np.asarray(x, dtype=np.float64).ravel())
    tot = float(ww[m].sum())
    return {"cells": int(gone.sum()), "area_share": float(ww[gone].sum() / tot) if tot > 0 else 0.0}


def _coverage(shares: Dict[str, Optional[float]], lacking: Sequence[str]) -> Dict[str, Any]:
    """The share of the region's area each input value covers, the inputs
    with missing cell values (a share below 1), and apart from them the
    inputs with missing time records (a period the run lacks)."""
    have = [v for v in shares.values() if v is not None]
    return {"area_share": dict(shares), "least_area_share": min(have) if have else None,
            "missing_cell_values": [k for k, v in shares.items() if v is not None and v < 1.0],
            "missing_time_records": list(lacking)}


def not_near_zero(ref, ref_range, share: float = NEAR_ZERO_SHARE):
    """A reference (a normal, a control's value) far enough from zero for a
    percent: nonzero and at least `share` of the range of the field's normals
    over the periods of its kind. Works on numbers and on maps."""
    r = np.asarray(ref, dtype=np.float64)
    rr = np.asarray(ref_range, dtype=np.float64)
    return np.isfinite(r) & (r != 0) & ~(np.abs(r) < share * rr)


def _normal_of(get: Callable[[str], Tuple[Any, Any]], kind: str, part: Any,
               normal_years: Sequence[int], event_years: Iterable[int]) -> Dict[str, Any]:
    """The normal years' values of one period: those used, left out (and
    why), missing, and incomplete (a value that does not cover the region)."""
    used: Dict[int, Any] = {}
    left: Dict[int, str] = {}
    missing: List[int] = []
    incomplete: List[int] = []
    covered: Dict[int, Optional[float]] = {}
    for y in normal_years:
        why = left_out_why(kind, part, y, event_years)
        if why:
            left[int(y)] = why
            continue
        v, cov = get(period_label(kind, y, part))
        if v is None:
            missing.append(int(y))
            continue
        if cov is not None and np.all(np.asarray(cov) < 1.0):
            incomplete.append(int(y))
        covered[int(y)] = None if cov is None else float(cov)
        used[int(y)] = v
    return {"used": used, "left_out": left, "missing": missing, "incomplete": incomplete,
            "covered": covered}


def _stack_mean(vals: Dict[int, Any]):
    if not vals:
        return None
    a = np.stack([np.asarray(v, dtype=np.float64) for v in vals.values()])
    return a.mean(axis=0) if a.ndim > 1 else float(a.mean())


def _support(vals: Dict[int, Any]) -> Dict[str, Any]:
    a = np.array([float(v) for v in vals.values()], dtype=np.float64)
    return {"n": int(a.size), "years": sorted(vals),
            "sd": float(a.std(ddof=1)) if a.size > 1 else None,
            "min": float(a.min()) if a.size else None, "max": float(a.max()) if a.size else None}


def reference_range(get: Callable[[str], Tuple[Any, Any]], kind: str,
                    normal_years: Sequence[int], event_years: Iterable[int]):
    """The range of the field's normals over the periods of a kind (months
    for a month row, the four seasons for any other): the yardstick of 'near
    zero' and 'near constant', a range, so it does not depend on where a
    unit puts its zero."""
    rk = "month" if kind == "month" else "season"
    normals = []
    for part in kind_parts(kind):
        n = _normal_of(get, rk, part, normal_years, event_years)
        m = _stack_mean(n["used"])
        if m is not None:
            normals.append(np.asarray(m, dtype=np.float64))
    if not normals:
        return None
    a = np.stack(normals)
    return a.max(axis=0) - a.min(axis=0) if a.ndim > 1 else float(a.max() - a.min())


def anomaly(suite: Suite, field: str, period: str, normal_years: Sequence[int],
            region: Optional[str] = None, event_years: Optional[Sequence[int]] = None,
            notes: Optional[Dict[str, Dict[str, Any]]] = None,
            settings: Optional[Dict[str, Any]] = None,
            named: Optional[Dict[str, Sequence[Any]]] = None) -> Dict[str, Any]:
    """A period's value minus the mean of the normal years' same period, for
    a region (a series value) or per cell (a map, region None). `named`: the
    settings the study's plan or an amendment names (settings_for)."""
    st = settings_for("anomaly", settings, named)
    note = _notes(notes).get(field) or {}
    kind, year, part = parse_period(period)
    ev = [int(year)] if event_years is None else [int(y) for y in event_years]
    units = suite.units(field, kind)
    out = {**METHODS["anomaly"].stamp(st), "field": field, "period": period,
           "region": region, "units": units, "change_units": change_units(units),
           "normal_years": list(normal_years), "event_years": ev}
    if region is None:
        get = lambda lab: (suite.map(field, lab), None)          # noqa: E731
        cmask, over = suite.coverage_mask()
    else:
        suite.mask(region)
        get = lambda lab: suite.value(field, region, lab)        # noqa: E731
    absent = suite.absent_why(period)
    value, cov = (None, None) if absent else get(period)
    n = _normal_of(get, kind, part, normal_years, ev)
    normal = _stack_mean(n["used"])
    out.update({"value": value, "covered": cov, "normal": normal,
                "normal_left_out": n["left_out"], "normal_missing": n["missing"],
                "normal_incomplete": n["incomplete"]})
    lab_of = lambda y: period_label(kind, y, part)               # noqa: E731
    if region is None:
        share = lambda x: covered_share(np.asarray(x, dtype=np.float64), cmask,  # noqa: E731
                                        suite.weights)
        shares = {period: share(value)} if value is not None else {}
        shares.update({lab_of(y): share(x) for y, x in n["used"].items()})
    else:
        shares = {period: cov} if value is not None else {}
        shares.update({lab_of(y): n["covered"].get(y) for y in n["used"]})
    out["coverage"] = _coverage(shares, ([] if value is not None else [period]) +
                                [lab_of(y) for y in n["missing"]])
    if region is None:
        out["coverage"]["over"] = over
    if value is None:
        out.update(anomaly=None, percent=None,
                   why=f"the period is absent from the run: {absent or 'no value'}")
        return out
    if normal is None:
        out.update(anomaly=None, percent=None, why="not estimable: no normal year has this period")
        return out
    a = np.asarray(value, dtype=np.float64) - np.asarray(normal, dtype=np.float64)
    ok, why = percent_policy(note)
    rr = reference_range(get, kind, normal_years, ev)
    out["reference_range"] = rr
    if region is None:
        out["coverage"]["result"] = {"cells": int(cmask.sum()),
                                     "missing_cells": int((cmask & ~np.isfinite(a)).sum()),
                                     "area_share": covered_share(a, cmask, suite.weights)}
        stack = np.stack([np.asarray(v, dtype=np.float64) for v in n["used"].values()])
        out["support"] = {"n": int(stack.shape[0]), "years": sorted(n["used"]),
                          "sd": stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else None,
                          "min": stack.min(axis=0), "max": stack.max(axis=0)}
        pct = None
        if ok and rr is not None:
            good = not_near_zero(normal, rr, st["near_zero_share"])
            pct = np.where(good, 100.0 * a / np.where(good, normal, 1.0), np.nan)
        out.update(anomaly=a, percent=pct,
                   percent_why_not=why if not ok else ("cells whose normal is near zero have none"))
        return out
    out["support"] = _support(n["used"])
    a = float(a)
    pct, pwhy = None, why
    if ok:
        if rr is not None and bool(not_near_zero(normal, rr, st["near_zero_share"])):
            pct = 100.0 * a / float(normal)
        else:
            pwhy = ("the normal is near zero: below "
                    f"{100 * st['near_zero_share']:g} % of the range of the field's normals "
                    "over the periods of its kind")
    out.update(anomaly=a, percent=pct, percent_why_not=None if pct is not None else pwhy)
    return out


def difference(run: Suite, control: Suite, field: str, period: str,
               region: Optional[str] = None, normal_years: Optional[Sequence[int]] = None,
               notes: Optional[Dict[str, Dict[str, Any]]] = None,
               settings: Optional[Dict[str, Any]] = None,
               named: Optional[Dict[str, Sequence[Any]]] = None) -> Dict[str, Any]:
    """Run minus control, same period and region (or per cell), with the
    percent of the control under the anomaly's rules."""
    st = settings_for("difference", settings, named)
    note = _notes(notes).get(field) or {}
    kind, year, part = parse_period(period)
    units = control.units(field, kind)
    out = {**METHODS["difference"].stamp(st), "field": field, "period": period,
           "region": region, "units": units, "change_units": change_units(units)}
    if region is None:
        gr = lambda s, lab: (s.map(field, lab), None)             # noqa: E731
        getc = lambda lab: (control.map(field, lab), None)        # noqa: E731
        cmask, over = control.coverage_mask()
    else:
        gr = lambda s, lab: s.value(field, region, lab)           # noqa: E731
        getc = lambda lab: control.value(field, region, lab)      # noqa: E731
    ab = run.absent_why(period) or control.absent_why(period)
    rv, rcov = (None, None) if ab else gr(run, period)
    cv, ccov = (None, None) if ab else gr(control, period)
    out.update(run_value=rv, control_value=cv)
    shares, lacking = {}, []
    for who, v, c in (("run", rv, rcov), ("control", cv, ccov)):
        if v is None:
            lacking.append(f"{who} {period}")
        else:
            shares[f"{who} {period}"] = (covered_share(np.asarray(v, dtype=np.float64), cmask,
                                                       control.weights) if region is None else c)
    out["coverage"] = _coverage(shares, lacking)
    if region is None:
        out["coverage"]["over"] = over
    if rv is None or cv is None:
        out.update(difference=None, percent=None,
                   why=f"the period is absent from a run: {ab or 'no value'}")
        return out
    d = np.asarray(rv, dtype=np.float64) - np.asarray(cv, dtype=np.float64)
    if region is None:
        out["coverage"]["result"] = {"cells": int(cmask.sum()),
                                     "missing_cells": int((cmask & ~np.isfinite(d)).sum()),
                                     "area_share": covered_share(d, cmask, control.weights)}
    if normal_years:
        rr = reference_range(getc, kind, normal_years, [int(year)])
        out["reference_range_from"] = "the control's normals"
    else:
        rk = "month" if kind == "month" else "season"
        vals = [getc(period_label(rk, year, p))[0] for p in kind_parts(kind)]
        vals = [np.asarray(v, dtype=np.float64) for v in vals if v is not None]
        rr = (np.stack(vals).max(axis=0) - np.stack(vals).min(axis=0)) if vals else None
        out["reference_range_from"] = f"the control's {rk} values of {year}"
    out["reference_range"] = rr
    ok, why = percent_policy(note)
    pct = None
    if ok and rr is not None:
        good = not_near_zero(cv, rr, st["near_zero_share"])
        if region is None:
            pct = np.where(good, 100.0 * d / np.where(good, cv, 1.0), np.nan)
        elif bool(good):
            pct = 100.0 * float(d) / float(cv)
        else:
            why = "the control's value is near zero beside the range of its periods"
    out.update(difference=d if region is None else float(d), percent=pct,
               percent_why_not=None if pct is not None else why)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# THE TIMING MEASURES
# ─────────────────────────────────────────────────────────────────────────────
# A cell's outcome is a date or one of these reasons; codes are what
# timing.nc stores. Quality flags are bits beside a date.
REASONS: Dict[str, Tuple[int, str]] = {
    "date": (0, "a date"),
    "records_missing": (1, "not estimable: records missing (a day of the period has no "
                           "record in any cell)"),
    "before_record": (2, "not estimable: the period starts before the run's records "
                         "(for snow_peak: the peak may lie before the record starts)"),
    "no_snow": (3, "no snow"),
    "snow_persists": (4, "snow persists"),
    "no_leaves": (5, "no leaves"),
    "little_leaf_change": (6, "leaf area changes too little for a green-up date"),
    "no_dormant_season": (7, "no dormant season"),
    "no_photosynthesis": (8, "no photosynthesis"),
    "no_runoff": (9, "no runoff: the water year's total is not above zero"),
    "insufficient_follow_up": (10, "not estimable: insufficient follow-up (the snow falls below "
                                   "the threshold too close to the end of the days searched to "
                                   "see the full run of days)"),
    "cell_values_missing": (11, "not estimable: the cell lacks values on days the records hold"),
}
CODE = {k: v[0] for k, v in REASONS.items()}
REASON_OF_CODE = {v[0]: k for k, v in REASONS.items()}
FLAGS: Dict[str, Tuple[int, str]] = {
    "snow_returns": (1, "snow returns: after the melt-out, snow water stays at or above the "
                        "threshold for as many consecutive days as a melt-out needs"),
    "partial_record": (2, "partial-record estimate: the water year starts before the run's "
                          "records; a separate quantity, kept out of normals"),
    "two_peaks": (4, "another peak of the running mean comes within the 'near' share of the "
                     "largest, with a dip of more than the 'dip' share between the two"),
    "runoff_falls": (8, "the cumulative runoff falls on some day (negative daily runoff), "
                        "so it may cross the half mark more than once"),
}
FLAG = {k: v[0] for k, v in FLAGS.items()}
TIMING_FIELDS = ("H2OSNO", "TLAI", "GPP", "QRUNOFF")


@dataclass
class DailyPeriod:
    """One year or water year of daily values, cells flattened.

    data[field] is (days, cells) float64, NaN where a day has no record or a
    cell no value; a rate holds the day's total (per day), a state its mean.
    first_day is the date number of the first row: 1 for a year (1 = its
    first day), and for a water year the day of the year it ends counted back
    (31 December = 0, 1 October = -91 in a noleap calendar). record_start is
    the first row the run holds: above 0 only when the period starts before
    the run's first record."""
    kind: str
    year: int
    first_day: int
    data: Dict[str, np.ndarray]
    record_start: int = 0
    calendar: str = "noleap"
    shape: Tuple[int, ...] = ()

    @property
    def label(self) -> str:
        return period_label(self.kind, self.year)

    @property
    def starts_before_record(self) -> bool:
        return self.record_start > 0


# Applicability checks: the callables each measure calls (and its contract shows).
def records_complete(x: np.ndarray) -> np.ndarray:
    """Every day of the period has a value in the cell: any search or peak
    over the period crosses every record."""
    return np.isfinite(x).all(axis=0)


def missing_inputs(x: np.ndarray) -> np.ndarray:
    """Per cell, 0 where the cell has a value on every day of the period, else
    why not: records_missing for every cell where a day has no record (no cell
    holds a value that day: a missing time record), cell_values_missing where
    the cell alone lacks a value on a day the records hold (missing cell
    values). Review round 1 (2026-11-05): the two were one reason before."""
    fin = np.isfinite(x)
    if x.shape[0] and not fin.any(axis=1).all():
        return np.full(x.shape[1], CODE["records_missing"], dtype=np.int8)
    return np.where(fin.all(axis=0), 0, CODE["cell_values_missing"]).astype(np.int8)


def holds_snow(largest: np.ndarray, threshold_mm: float) -> np.ndarray:
    return largest >= threshold_mm


def has_leaves(peak: np.ndarray) -> np.ndarray:
    return peak > 0


def rise_enough(rise: np.ndarray, peak: np.ndarray, rise_mark: float) -> np.ndarray:
    """The rise from the low before the peak is at least rise_mark of the
    peak: a canopy that grows most of its leaves each year."""
    return rise >= rise_mark * peak


def photosynthesises(peak: np.ndarray) -> np.ndarray:
    return peak > 0


def dormant_season(low: np.ndarray, mark: np.ndarray) -> np.ndarray:
    return low < mark


def runoff_positive(total: np.ndarray) -> np.ndarray:
    return total > 0


def _first_true(cand: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    found = cand.any(axis=0)
    return found, np.where(found, cand.argmax(axis=0), -1)


def _running_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Means of every w consecutive days inside the period, row j holding
    days j..j+w-1 (dated by its middle day, j + w // 2)."""
    if w < 1 or w % 2 == 0:
        raise MethodError(f"a running mean needs an odd window, not {w}")
    if x.shape[0] < w:
        return np.full((0,) + x.shape[1:], np.nan)
    # each window summed on its own (not as a difference of running sums), so
    # two windows holding the same values have the same mean to the last bit
    # and a tie stays a tie; a window holding a missing day has no mean
    return np.lib.stride_tricks.sliding_window_view(x, w, axis=0).mean(axis=-1)


def _peak(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(largest, its first day): ties go to the earliest day."""
    xx = np.where(np.isfinite(x), x, -np.inf)
    return xx.max(axis=0), xx.argmax(axis=0)


def _low_before(x: np.ndarray, upto: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(the lowest value on days 0..upto, its first day), per cell."""
    idx = np.arange(x.shape[0])[:, None]
    v = np.where((idx <= upto[None, :]) & np.isfinite(x), x, np.inf)
    return v.min(axis=0), v.argmin(axis=0)


def _outcome(n_cells: int) -> Dict[str, np.ndarray]:
    return {"date": np.full(n_cells, np.nan), "reason": np.zeros(n_cells, dtype=np.int8),
            "flags": np.zeros(n_cells, dtype=np.int8)}


def _finish(out: Dict[str, np.ndarray], reason: np.ndarray, idx: np.ndarray,
            first_day: int) -> Dict[str, np.ndarray]:
    out["reason"] = reason.astype(np.int8)
    out["date"] = np.where(reason == CODE["date"], idx + first_day, np.nan).astype(np.float64)
    return out


def _snow_peak(p: DailyPeriod, threshold_mm: float) -> Dict[str, np.ndarray]:
    x = p.data["H2OSNO"]
    out = _outcome(x.shape[1])
    if p.starts_before_record:
        # a largest after the record starts is no proof: the peak may be before
        out["reason"][:] = CODE["before_record"]
        return out
    big, at = _peak(x)
    miss = missing_inputs(x)
    reason = np.where(miss != 0, miss,
                      np.where(~holds_snow(big, threshold_mm), CODE["no_snow"], CODE["date"]))
    out = _finish(out, reason, at, p.first_day)
    out["largest_mm"] = np.where(records_complete(x), big, np.nan)
    return out


def _trailing(b: np.ndarray) -> np.ndarray:
    """Per cell, how many of the last days in a row are True."""
    stop = ~b[::-1]
    return np.where(stop.any(axis=0), stop.argmax(axis=0), b.shape[0])


def _snow_gone(p: DailyPeriod, threshold_mm: float, days: int) -> Dict[str, np.ndarray]:
    r0 = p.record_start
    x = p.data["H2OSNO"][r0:]
    n, c = x.shape
    out = _outcome(c)
    days = int(days)
    big, at = _peak(x)
    below = np.isfinite(x) & (x < threshold_mm)
    above = np.isfinite(x) & ~(x < threshold_mm)
    j_idx = np.arange(max(n - days + 1, 0))[:, None]

    def runs(b):
        cs = np.concatenate([np.zeros((1, c), dtype=np.int64), np.cumsum(b, axis=0)])
        return (cs[days:] - cs[:-days]) == days if n >= days else np.zeros((0, c), bool)

    run_below, run_above = runs(below), runs(above)
    # the first lasting melt-out after the peak, the run of days inside the period
    found, j = _first_true(run_below & (j_idx >= at[None, :]))
    # INSUFFICIENT FOLLOW-UP (review round 1, 2026-11-05): days below the
    # threshold that start after the peak and reach the end of the days
    # searched (the water year's end) before they are `days` long may be the
    # melt-out, but its full run cannot be seen: not 'snow persists', which is
    # left for snow that never makes such a start
    tail = _trailing(below)
    short = (tail > 0) & (tail < days) & (n - tail > at)
    miss = missing_inputs(x)
    reason = np.where(miss != 0, miss,
                      np.where(~holds_snow(big, threshold_mm), CODE["no_snow"],
                               np.where(found, CODE["date"],
                                        np.where(short, CODE["insufficient_follow_up"],
                                                 CODE["snow_persists"]))))
    out = _finish(out, reason, j + r0, p.first_day)
    back = (run_above & (j_idx > j[None, :])).any(axis=0) & (reason == CODE["date"])
    out["flags"] = np.where(back, FLAG["snow_returns"], 0).astype(np.int8)
    if p.starts_before_record:
        out["flags"] = (out["flags"] | FLAG["partial_record"]).astype(np.int8)
    return out


def _leaf_green_up(p: DailyPeriod, fraction: float, rise_mark: float) -> Dict[str, np.ndarray]:
    x = p.data["TLAI"]
    out = _outcome(x.shape[1])
    peak, at = _peak(x)
    low, lowd = _low_before(x, at)
    rise = peak - low
    mark = low + fraction * rise
    idx = np.arange(x.shape[0])[:, None]
    found, j = _first_true(np.isfinite(x) & (x >= mark[None, :]) & (idx > lowd[None, :]))
    miss = missing_inputs(x)
    reason = np.where(miss != 0, miss,
                      np.where(~has_leaves(peak), CODE["no_leaves"],
                               np.where(~rise_enough(rise, peak, rise_mark),
                                        CODE["little_leaf_change"], CODE["date"])))
    reason = np.where((reason == CODE["date"]) & ~found, CODE["little_leaf_change"], reason)
    out = _finish(out, reason, j, p.first_day)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["rise_share"] = np.where(records_complete(x) & (peak > 0), rise / peak, np.nan)
    return out


def _photosynthesis_onset(p: DailyPeriod, fraction: float, window: int) -> Dict[str, np.ndarray]:
    x = p.data["GPP"]
    out = _outcome(x.shape[1])
    m = _running_mean(x, int(window))
    peak, at = _peak(m)
    low, lowj = _low_before(m, at)
    mark = fraction * peak
    jx = np.arange(m.shape[0])[:, None]
    found, j = _first_true(np.isfinite(m) & (m >= mark[None, :]) & (jx > lowj[None, :]))
    miss = missing_inputs(x)
    reason = np.where(miss != 0, miss,
                      np.where(~photosynthesises(peak), CODE["no_photosynthesis"],
                               np.where(~dormant_season(low, mark), CODE["no_dormant_season"],
                                        CODE["date"])))
    reason = np.where((reason == CODE["date"]) & ~found, CODE["no_dormant_season"], reason)
    return _finish(out, reason, j + int(window) // 2, p.first_day)


def _growth_peak(p: DailyPeriod, window: int, near: float, dip: float) -> Dict[str, np.ndarray]:
    x = p.data["GPP"]
    out = _outcome(x.shape[1])
    m = _running_mean(x, int(window))
    top, i0 = _peak(m)
    miss = missing_inputs(x)
    reason = np.where(miss != 0, miss,
                      np.where(~photosynthesises(top), CODE["no_photosynthesis"], CODE["date"]))
    out = _finish(out, reason, i0 + int(window) // 2, p.first_day)
    # TWO PEAKS: another day whose mean is within `near` of the largest, with
    # the mean between them falling below that day's own by more than `dip`
    # of the largest (the second peak stands out from the dip between).
    k = m.shape[0]
    idx = np.arange(k)[:, None]
    mm = np.where(np.isfinite(m), m, -np.inf)
    hi = (1.0 - near) * top
    fmin = np.minimum.accumulate(np.where(idx >= i0[None, :], mm, np.inf), axis=0)
    fwd = ((idx > i0[None, :]) & (mm >= hi[None, :]) & (fmin < mm - dip * top[None, :])).any(axis=0)
    rv, ri0 = mm[::-1], (k - 1) - i0
    bmin = np.minimum.accumulate(np.where(idx >= ri0[None, :], rv, np.inf), axis=0)
    bwd = ((idx > ri0[None, :]) & (rv >= hi[None, :]) & (bmin < rv - dip * top[None, :])).any(axis=0)
    out["flags"] = np.where((fwd | bwd) & (reason == CODE["date"]), FLAG["two_peaks"], 0
                            ).astype(np.int8)
    return out


def _half_runoff(p: DailyPeriod, fraction: float) -> Dict[str, np.ndarray]:
    x = p.data["QRUNOFF"]
    out = _outcome(x.shape[1])
    if p.starts_before_record:
        out["reason"][:] = CODE["before_record"]
        return out
    ok = records_complete(x)
    c = np.cumsum(np.where(np.isfinite(x), x, 0.0), axis=0)
    total = c[-1]                         # the curve's own end, so the half mark is on it
    found, j = _first_true(c >= fraction * total[None, :])
    miss = missing_inputs(x)
    reason = np.where(miss != 0, miss,
                      np.where(~runoff_positive(total), CODE["no_runoff"], CODE["date"]))
    out = _finish(out, reason, j, p.first_day)
    out["flags"] = np.where((np.isfinite(x) & (x < 0)).any(axis=0) & ok, FLAG["runoff_falls"], 0
                            ).astype(np.int8)
    return out


def _measure(method_id: str, compute: Callable[..., Dict[str, np.ndarray]]):
    def run(period: DailyPeriod, settings: Optional[Dict[str, Any]] = None,
            alternatives: bool = True) -> Dict[str, Any]:
        """Per cell, a date or a reason, for the default and (alternatives
        True) every named alternative. A setting changed from the default is
        computed as one more variant beside it, never in its place."""
        m = METHODS[method_id]
        if period.kind != TIMING_KIND[method_id]:
            raise MethodError(f"{method_id} belongs to the {TIMING_KIND[method_id]}, "
                              f"not to a {period.kind}")
        chosen = settings_for(method_id, settings)
        vs = m.variants() if alternatives else [("default", m.defaults())]
        if chosen != m.defaults():
            vs = vs + [("plan", chosen)]
        out = {**m.stamp(m.defaults()), "period": period.label, "kind": period.kind,
               "year": period.year, "first_day": period.first_day,
               "calendar": period.calendar, "shape": period.shape, "variants": {}}
        for lab, st in vs:
            r = compute(period, **st)
            out["variants"][lab] = {"settings": dict(st), **r}
        return out
    run.__name__ = method_id
    return run


TIMING_KIND = {"snow_peak": "wateryear", "snow_gone": "wateryear", "half_runoff": "wateryear",
               "leaf_green_up": "year", "photosynthesis_onset": "year", "growth_peak": "year"}
TIMING_IDS = ("snow_peak", "snow_gone", "leaf_green_up", "photosynthesis_onset",
              "growth_peak", "half_runoff")
TIMING_FIELD = {"snow_peak": "H2OSNO", "snow_gone": "H2OSNO", "leaf_green_up": "TLAI",
                "photosynthesis_onset": "GPP", "growth_peak": "GPP", "half_runoff": "QRUNOFF"}

snow_peak = _measure("snow_peak", _snow_peak)
snow_gone = _measure("snow_gone", _snow_gone)
leaf_green_up = _measure("leaf_green_up", _leaf_green_up)
photosynthesis_onset = _measure("photosynthesis_onset", _photosynthesis_onset)
growth_peak = _measure("growth_peak", _growth_peak)
half_runoff = _measure("half_runoff", _half_runoff)


# ─────────────────────────────────────────────────────────────────────────────
# THE DAILY HISTORY, READ ONCE PER YEAR
# ─────────────────────────────────────────────────────────────────────────────
def _month_offset(year: int, month: int, calendar: str) -> int:
    return sum(G.days_in_month(year, k, calendar) for k in range(1, int(month)))


def _span(kind: str, year: int, ys: int, ws: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """((year, month) the period starts, (year, month) it ends before)."""
    if kind == "year":
        return (year, ys), (year + 1, ys)
    if ws == 1:
        return (year, 1), (year + 1, 1)
    return (year - 1, ws), (year, ws)


def _first_day(kind: str, year: int, ws: int, calendar: str) -> int:
    if kind == "year" or ws == 1:
        return 1
    before = sum(G.days_in_month(year - 1, m, calendar) for m in range(ws, 13))
    return 1 - before


def read_year(stream, year: int, fields: Sequence[str]) -> Dict[str, np.ndarray]:
    """One calendar year of daily fields, (days, cells), in one pass of the
    year file (gridded.Stream.read_block_pass): a day without a record stays
    NaN, a rate becomes the day's total (the rate times 86,400 s)."""
    cal = stream.calendar
    nd = G.days_in_year(year, cal)
    nc = int(np.prod(stream.grid.shape))
    arr = {f: np.full((nd, nc), np.nan) for f in fields}

    def take(pb):
        for f in fields:
            v = pb.data[f].reshape(len(pb.keys), -1)
            if pb.per_second[f]:
                v = v * DAY_SECONDS
            for i, k in enumerate(pb.keys):
                arr[f][R.day_of_year(k, cal) - 1] = v[i]

    stream.read_block_pass(list(fields), int(year), reduce=take, require_complete=False)
    return arr


class PeriodAssembler:
    """Each (kind, year) as a DailyPeriod, from calendar years of daily
    values handed in any order: a period is made as soon as every calendar
    year it spans that the run holds has been handed in, and a year is let go
    once no period still to be made needs it. A span's year the run lacks
    stays NaN, counted in record_start when it lies before the run's first
    year (the water year whose October-December the run does not hold)."""

    def __init__(self, years: Sequence[int], have: Sequence[int], calendar: str,
                 shape: Tuple[int, ...], control: Optional[Dict[str, Any]] = None,
                 kinds: Sequence[str] = ("wateryear", "year"),
                 fields: Optional[Dict[str, Sequence[str]]] = None):
        self.cal, self.shape = calendar, tuple(shape)
        self.ys, self.ws = start_months(control)
        self.fields = fields or {"wateryear": ("H2OSNO", "QRUNOFF"), "year": ("TLAI", "GPP")}
        self.have = sorted(int(y) for y in have)
        if not self.have:
            raise MethodError("the stream holds no years")
        self.todo = []
        for y in sorted({int(y) for y in years}):
            for k in kinds:
                (a, am), (b, bm) = _span(k, y, self.ys, self.ws)
                self.todo.append((k, y, (a, am), (b, bm)))
        self.held: Dict[int, Dict[str, np.ndarray]] = {}

    @staticmethod
    def _years(item) -> range:
        _, _, (a, _), (b, bm) = item
        return range(a, (b if bm > 1 else b - 1) + 1)

    def needed(self) -> List[int]:
        """The calendar years of the run the periods still to be made span."""
        return sorted({c for it in self.todo for c in self._years(it) if c in self.have})

    def add(self, year: int, arrays: Dict[str, np.ndarray]) -> List[DailyPeriod]:
        self.held[int(year)] = arrays
        return self._ready(False)

    def finish(self) -> List[DailyPeriod]:
        return self._ready(True)

    def _ready(self, final: bool) -> List[DailyPeriod]:
        out, rest = [], []
        for it in self.todo:
            if final or all(c in self.held for c in self._years(it) if c in self.have):
                out.append(self._period(it))
            else:
                rest.append(it)
        self.todo = rest
        keep = {c for it in rest for c in self._years(it)}
        for c in [c for c in self.held if c not in keep]:
            del self.held[c]
        return out

    def _period(self, it) -> DailyPeriod:
        k, y, (a, am), (b, bm) = it
        nc = int(np.prod(self.shape))
        parts = {f: [] for f in self.fields[k]}
        lead = 0
        for cy in self._years(it):
            lo = _month_offset(cy, am, self.cal) if cy == a else 0
            hi = (_month_offset(cy, bm, self.cal) if (cy == b and bm > 1)
                  else G.days_in_year(cy, self.cal))
            for f in self.fields[k]:
                h = self.held.get(cy)
                parts[f].append(h[f][lo:hi] if h is not None and f in h else
                                np.full((hi - lo, nc), np.nan))
            if cy < self.have[0]:
                lead += hi - lo
        return DailyPeriod(kind=k, year=y, first_day=_first_day(k, y, self.ws, self.cal),
                           data={f: np.concatenate(v) for f, v in parts.items()},
                           record_start=lead, calendar=self.cal, shape=self.shape)


def daily_periods(stream, years: Sequence[int], control: Optional[Dict[str, Any]] = None,
                  kinds: Sequence[str] = ("wateryear", "year"),
                  fields: Optional[Dict[str, Sequence[str]]] = None):
    """Each (kind, year) as a DailyPeriod, every calendar year read once, in
    order; at most the two calendar years a period spans are held at once."""
    if getattr(stream, "cadence", "daily") != "daily":
        raise MethodError(f"the timing measures read a daily stream; {stream.name} is "
                          f"{stream.cadence}")
    asm = PeriodAssembler(years, stream.years(), stream.calendar, tuple(stream.grid.shape),
                          control, kinds, fields)
    flds = sorted({f for k in kinds for f in asm.fields[k]})
    for c in asm.needed():
        yield from asm.add(c, read_year(stream, c, flds))
    yield from asm.finish()


def timing_run(run_dir, years: Sequence[int], control: Optional[Dict[str, Any]] = None,
               measures: Sequence[str] = TIMING_IDS, stream: str = "h1",
               alternatives: bool = True) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Every timing measure for every year (water years filed under the year
    they end), from the run's daily history, each year file read once."""
    s = G.open_stream(run_dir, stream)
    kinds = sorted({TIMING_KIND[m] for m in measures})
    fields = {k: tuple(dict.fromkeys(TIMING_FIELD[m] for m in measures if TIMING_KIND[m] == k))
              for k in kinds}                  # only what the measures asked for read
    out: Dict[str, Dict[int, Dict[str, Any]]] = {m: {} for m in measures}
    for p in daily_periods(s, years, control, kinds=kinds, fields=fields):
        for mid in measures:
            if TIMING_KIND[mid] == p.kind:
                out[mid][p.year] = METHODS[mid].run(p, alternatives=alternatives)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# TIMING: REGION DISTRIBUTIONS, SHIFTS AND THE SENSITIVITY RECORD
# ─────────────────────────────────────────────────────────────────────────────
def distribution(v: Dict[str, np.ndarray], mask, weights) -> Dict[str, Any]:
    """Over a region: the area-weighted median and quartiles of the dates,
    and the counts of dates, of each reason and of each flag."""
    m = np.asarray(mask, dtype=bool).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    d = v["date"][m]
    r = v["reason"][m]
    f = v["flags"][m]
    has = np.isfinite(d)
    wm = w[m]
    tot = float(wm.sum())
    share = lambda b: float(wm[b].sum() / tot) if tot > 0 else 0.0     # noqa: E731
    rec_gone = np.isin(r, (CODE["records_missing"], CODE["before_record"]))
    cell_gone = r == CODE["cell_values_missing"]
    dated = float(wm[has].sum())
    # FLAGGED AREA (review round 1, 2026-11-05): a flag's cells, its area and
    # its share, each share with its denominator named
    flag_area = {}
    for k, (b, _) in FLAGS.items():
        fk = (f & b) > 0
        if fk.any():
            a = float(wm[fk].sum())
            flag_area[k] = {"cells": int(fk.sum()), "area_km2": a, "region_area_km2": tot,
                            "share_of_region_area": a / tot if tot > 0 else 0.0,
                            "cells_with_a_date": int((fk & has).sum()), "dated_area_km2": dated,
                            "share_of_dated_area": (float(wm[fk & has].sum()) / dated
                                                    if dated > 0 else None)}
    return {"cells": int(m.sum()), "dates": int(has.sum()),
            "inputs_area_share": share(~rec_gone & ~cell_gone),
            "missing_time_records": {"cells": int(rec_gone.sum()), "area_share": share(rec_gone)},
            "missing_cell_values": {"cells": int(cell_gone.sum()), "area_share": share(cell_gone)},
            "flag_area": flag_area,
            "date_area_share": float(w[m][has].sum() / tot) if tot > 0 else 0.0,
            "median": wquantile(d[has], w[m][has], 0.5),
            "q25": wquantile(d[has], w[m][has], 0.25),
            "q75": wquantile(d[has], w[m][has], 0.75),
            "reasons": {k: int((r == c).sum()) for k, (c, _) in REASONS.items() if k != "date"
                        and (r == c).any()},
            "flags": {k: int(((f & b) > 0).sum()) for k, (b, _) in FLAGS.items()
                      if ((f & b) > 0).any()}}


def eligibility(first, second, mask, weights,
                names: Tuple[str, str] = ("this", "reference")) -> Dict[str, Any]:
    """The cells eligible (holding a date) in both, only in the first, only
    in the second and in neither, each with its count and its share of the
    region's area: the change of eligibility a comparison of dates stands on
    (review round 1, 2026-11-05)."""
    m = np.asarray(mask, dtype=bool).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    a = m & np.isfinite(np.asarray(first, dtype=np.float64).ravel())
    b = m & np.isfinite(np.asarray(second, dtype=np.float64).ravel())
    tot = float(w[m].sum())
    parts = {"both": a & b, f"only_{names[0]}": a & ~b, f"only_{names[1]}": ~a & b,
             "neither": m & ~a & ~b}
    out: Dict[str, Any] = {k: {"cells": int(x.sum()),
                               "area_share": float(w[x].sum() / tot) if tot > 0 else 0.0}
                           for k, x in parts.items()}
    out["region_area_km2"] = tot
    return out


def normal_dates(by_year: Dict[int, Dict[str, Any]], normal_years: Sequence[int],
                 variant: str = "default") -> np.ndarray:
    """Per cell, the mean of its dates over the normal years, made only where
    every normal year gives a full-record date (a normal of four years is not
    a normal of five; a partial-record date never enters a normal)."""
    rows = []
    for y in normal_years:
        if y not in by_year:
            return np.full_like(next(iter(by_year.values()))["variants"][variant]["date"], np.nan)
        v = by_year[y]["variants"][variant]
        rows.append(np.where((v["flags"] & FLAG["partial_record"]) > 0, np.nan, v["date"]))
    a = np.stack(rows)
    return np.where(np.isfinite(a).all(axis=0), a.mean(axis=0), np.nan)


# THE RANGE ACROSS SETTINGS (the user's decisions on G part 1, 2026-11-08).
# The shifts compared between settings are on the comparable cells: those
# eligible (a date in the year and in its baseline) under every evaluated
# setting, so that a difference between settings is not a difference of
# cells. Beside it stand the default, the smallest and the largest shift,
# each with its setting; the range is required in a claim when the largest
# minus the smallest exceeds RANGE_LIMIT_DAYS, and the settings reverse the
# direction when one gives an earlier date and another a later one. A tie
# keeps the setting listed first (the default, then the alternatives in the
# contract's order). Each setting's own shift and eligibility stay as they
# were, in its variant's record.
RANGE_LIMIT_DAYS = 7.0
RANGE_EPS = 1e-9                     # days: below it a difference is float noise


def across_settings(res: Dict[str, Any], refs: Dict[str, Optional[np.ndarray]], mask,
                    weights) -> Dict[str, Any]:
    """Every evaluated setting's median shift on the comparable cells, the
    comparable cells' count, area and share of the region's area, the
    default, the smallest and the largest shift with their settings, their
    spread, whether it exceeds RANGE_LIMIT_DAYS and whether the settings
    reverse the direction of the shift."""
    m = np.asarray(mask, dtype=bool).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    labs = [lab for lab in ["default"] + [x for x in res["variants"] if x != "default"]
            if lab in res["variants"] and refs.get(lab) is not None]
    sh = {lab: res["variants"][lab]["date"] - refs[lab] for lab in labs}
    ok = m.copy()
    for lab in labs:
        ok &= np.isfinite(sh[lab])
    tot = float(w[m].sum())
    out: Dict[str, Any] = {
        "comparable": {"cells": int(ok.sum()), "area_km2": float(w[ok].sum()),
                       "area_share": float(w[ok].sum() / tot) if tot > 0 else 0.0},
        "evaluated": [{"variant": lab, "settings": dict(res["variants"][lab]["settings"]),
                       "median_shift": wquantile(sh[lab][ok], w[ok], 0.5)} for lab in labs],
        "limit_days": RANGE_LIMIT_DAYS}
    have = [e for e in out["evaluated"] if e["median_shift"] is not None]
    if not have:
        out.update(default=None, minimum=None, maximum=None, spread_days=None,
                   range_required=False, direction_reversal=False,
                   why="no cell is eligible under every setting in the year and in its baseline")
        return out
    lo = hi = have[0]
    for e in have[1:]:                                  # a tie keeps the first listed
        if e["median_shift"] < lo["median_shift"]:
            lo = e
        if e["median_shift"] > hi["median_shift"]:
            hi = e
    pick = lambda e: {k: e[k] for k in ("variant", "settings", "median_shift")}  # noqa: E731
    spread = float(hi["median_shift"] - lo["median_shift"])
    out.update(default=pick(have[0]) if have[0]["variant"] == "default" else None,
               minimum=pick(lo), maximum=pick(hi), spread_days=spread,
               range_required=bool(spread > RANGE_LIMIT_DAYS + RANGE_EPS),
               direction_reversal=bool(lo["median_shift"] < -RANGE_EPS
                                       and hi["median_shift"] > RANGE_EPS))
    return out


def timing_summary(measure: str, by_year: Dict[int, Dict[str, Any]], year: int, mask, weights,
                   normal_years: Optional[Sequence[int]] = None,
                   control: Optional[Dict[int, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """One measure, one year, one region: each variant's distribution, the
    per-cell shift in days against the normal years (or the control's same
    year) and its distribution, and the sensitivity record: per alternative,
    the setting, the region's median date and median shift beside the
    default's, and the median per-cell change of date. Every shift and every
    change of date is on the cells eligible in both, and beside it stands the
    change of eligibility (eligibility(): cells and area shares eligible in
    both, in one only, in neither) and the area the statistic covers. Then the
    range across the settings, on the cells eligible under every setting
    (across_settings)."""
    m = np.asarray(mask, dtype=bool).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    res = by_year[year]
    out = {**{k: res[k] for k in ("method", "version", "period", "kind", "year")},
           "against": ("the control's same year" if control is not None else
                       f"the normal years {list(normal_years or [])}"), "variants": {},
           "sensitivity": []}
    # the version is the one that made this summary; dates stored by another
    # (a suite's timing.nc not yet made again) say so
    if measure in METHODS and METHODS[measure].version != res.get("version"):
        out["version"] = METHODS[measure].version
        out["stored_dates"] = {"version": res.get("version")}
    partial = False
    refs: Dict[str, Optional[np.ndarray]] = {}
    for lab, v in res["variants"].items():
        rec = {"settings": v["settings"], "distribution": distribution(v, m, w)}
        ref = None
        if control is not None:
            cv = (control.get(year) or {}).get("variants", {}).get(lab)
            ref = cv["date"] if cv is not None else None
        elif normal_years:
            ref = normal_dates(by_year, normal_years, lab)
        refs[lab] = ref
        if ref is not None:
            sh = v["date"] - ref
            ok = m & np.isfinite(sh)
            rec["shift"] = {"cells": int(ok.sum()),
                            "area_share": covered_share(sh, m, w),
                            "area_km2": float(w[ok].sum()),
                            "median": wquantile(sh[ok], w[ok], 0.5),
                            "q25": wquantile(sh[ok], w[ok], 0.25),
                            "q75": wquantile(sh[ok], w[ok], 0.75),
                            "eligible": eligibility(v["date"], ref, m, w)}
            if ((v["flags"][ok] & FLAG["partial_record"]) > 0).any():
                rec["shift"]["comparability"] = "restricted comparability"
                partial = True
        out["variants"][lab] = rec
    d0 = res["variants"]["default"]["date"]
    base = out["variants"]["default"]
    for lab, v in res["variants"].items():
        if lab == "default":
            continue
        ch = v["date"] - d0
        ok = m & np.isfinite(ch)
        rec = out["variants"][lab]
        out["sensitivity"].append({
            "variant": lab, "settings": v["settings"],
            "median_date": rec["distribution"]["median"],
            "default_median_date": base["distribution"]["median"],
            "median_shift": (rec.get("shift") or {}).get("median"),
            "default_median_shift": (base.get("shift") or {}).get("median"),
            "median_cell_change": wquantile(ch[ok], w[ok], 0.5),
            "cells_with_both": int(ok.sum()),
            "cell_change_area_share": covered_share(ch, m, w),
            "cell_change_area_km2": float(w[ok].sum()),
            "eligible": eligibility(d0, v["date"], m, w, ("default", "alternative"))})
    if any(r is not None for r in refs.values()):
        out["across_settings"] = across_settings(res, refs, m, w)
    if partial:
        out["comparability"] = "restricted comparability: a partial-record date is in a shift"
    return out


# ─────────────────────────────────────────────────────────────────────────────
# THE SCREEN
# ─────────────────────────────────────────────────────────────────────────────
NEAR_CONSTANT_SHARE = 0.01
SCORED_TYPES = ("flux", "state")
EVENT_TYPE = "event"
NOT_SCORED_YET = ("store", "bounded")
SCREEN_COLUMNS = ("field", "group", "type", "scale", "kind", "period", "units", "change_units",
                  "value", "reference", "reference_is", "change", "percent", "score", "status",
                  "mark", "n_normal", "normal_years_used", "normal_sd", "normal_min",
                  "normal_max", "near_constant", "reference_range", "normal_largest",
                  "normal_years_with_fire", "covered", "missing_cell_values",
                  "missing_time_records", "note")
DISPLAY_RULE = (
    "Shown first, per process group, the groups in the model's order (the catalog's): the "
    "three rows with the largest absolute score, largest first, at most one per field (its "
    "highest-scored period; its other periods stay in the table); a tie keeps the field that "
    "comes first in the catalog and, within a field, the period that comes first in the table "
    "(the year, the water year, the seasons, the months); a group without a scored row shows "
    "none. Then every event row marked above_normal_maximum or differs_from_control, in the "
    "table's order. Then, for each field of a type not scored yet (stores, bounded indices), "
    "its largest absolute change with its period, labelled not scored, a tie keeping the "
    "earlier period. Every score is shown with the number of normal years behind it. Rows "
    "without a score (near-constant normal, insufficient support, no change, period absent) "
    "are not shown first and stay in the table. The whole table is the CSV beside this rule.")


def screen_periods(year: int) -> List[str]:
    """The periods of one year the screen covers: the year, its water year,
    its four seasons and its twelve months."""
    y = int(year)
    return ([period_label("year", y), period_label("wateryear", y)] +
            [period_label("season", y, s) for s in SEASON_NAMES] +
            [period_label("month", y, m) for m in range(1, 13)])


def _event_value(field: str, v: Optional[float], area_km2: float) -> Optional[float]:
    """An event's amount: the burned fraction becomes km2 (the region's
    fraction times its area, the regional table's burned area); carbon lost
    stays the region's mean."""
    if v is None:
        return None
    return float(v) * area_km2 if field == DA.BURNED_FRACTION else float(v)


def near_constant(sd: Optional[float], ref_range: Optional[float],
                  share: float = NEAR_CONSTANT_SHARE) -> bool:
    """A normal with no spread to speak of: its standard deviation zero or at
    most `share` of the range of the field's normals over its kind's periods."""
    if sd is None:
        return False
    return sd == 0.0 or (ref_range is not None and sd <= share * ref_range)


def screen(suite: Suite, region: str, year: int, normal_years: Sequence[int],
           control: Optional[Suite] = None, notes: Optional[Dict[str, Dict[str, Any]]] = None,
           fields: Optional[Sequence[str]] = None,
           settings: Optional[Dict[str, Any]] = None,
           named: Optional[Dict[str, Sequence[Any]]] = None) -> Dict[str, Any]:
    """Every catalog field and every period of one year over one region, as
    one table: the anomaly against the normal years (suite is the control),
    or the difference from the control (control given, suite the run), each
    scored by its variable type."""
    st = settings_for("screen", settings, named)
    nts = _notes(notes)
    flds = list(fields) if fields else catalog_fields(nts)
    base = control if control is not None else suite
    base.mask(region)
    area = base.region_area_km2[region]
    ev = [int(year)]
    rows: List[Dict[str, Any]] = []
    for f in flds:
        note = nts.get(f) or {}
        typ = note.get("type")
        in_suite = f in suite.fields and f in base.fields
        getb = (lambda lab, f=f: base.value(f, region, lab))
        for lab in screen_periods(year):
            kind, y, part = parse_period(lab)
            row = dict.fromkeys(SCREEN_COLUMNS)
            u = base.units(f, kind) if in_suite else ""
            row.update({"field": f, "group": note.get("group"), "type": typ,
                        "scale": note.get("scale"), "kind": kind, "period": lab,
                        "units": u, "change_units": change_units(u),
                        "reference_is": ("the control's value" if control is not None else
                                         "the normal years' mean")})
            rows.append(row)
            if not in_suite:
                row.update(status="not in the suite", note="the suite holds no products of it")
                continue
            ab = suite.absent_why(lab) or (control.absent_why(lab) if control is not None else None)
            if ab:
                row.update(status="period absent", note=ab, missing_time_records=ab)
                continue
            v, cov = suite.value(f, region, lab)
            row["covered"] = cov
            n = _normal_of(getb, kind, part, normal_years, ev)
            vals = [float(x) for x in n["used"].values()]
            sup = _support(n["used"])
            rr = reference_range(getb, kind, normal_years, ev)
            row.update(n_normal=sup["n"], normal_years_used=" ".join(str(x) for x in sup["years"]),
                       normal_sd=sup["sd"], normal_min=sup["min"], normal_max=sup["max"],
                       reference_range=rr)
            notes_txt = list(n["left_out"].values())
            # MISSING CELL VALUES APART FROM MISSING TIME RECORDS (review round
            # 1, 2026-11-05): a value short of the region's area, and a period
            # the run lacks, each in its own column
            mc, mt = [], []
            if cov is not None and cov < 1.0:
                mc.append(f"{lab}: its value covers {cov:.3g} of the region's area")
            if control is not None:
                ccov = control.value(f, region, lab)[1]
                if ccov is not None and ccov < 1.0:
                    mc.append(f"the control's {lab}: its value covers {ccov:.3g} of the "
                              f"region's area")
            if n["incomplete"]:
                mc.append("normal years whose value covers less than the whole region: " +
                          ", ".join(str(x) for x in n["incomplete"]))
            if n["missing"]:
                mt.append("normal years without the period: " +
                          ", ".join(str(x) for x in n["missing"]))
            row.update(missing_cell_values="; ".join(mc), missing_time_records="; ".join(mt))
            if typ == EVENT_TYPE:
                ref_raw = control.value(f, region, lab)[0] if control is not None else None
                vv = _event_value(f, v, area)
                nv = [_event_value(f, x, area) for x in vals]
                if f == DA.BURNED_FRACTION:
                    row["units"] = row["change_units"] = "km2"
                ref = (_event_value(f, ref_raw, area) if control is not None else
                       (float(np.mean(nv)) if nv else None))
                row.update(value=vv, reference=ref,
                           normal_largest=max(nv) if nv else None,
                           normal_years_with_fire=sum(1 for x in nv if x > 0))
                if vv is None or ref is None:
                    row.update(status="insufficient support",
                               note="; ".join(notes_txt + ["no value to compare"]))
                    continue
                row["change"] = vv - ref
                if row["change"] == 0.0:
                    row["status"] = "no change"
                elif control is not None:
                    row.update(status="marked", mark="differs_from_control")
                elif sup["n"] < 2 or n["incomplete"]:
                    row["status"] = "insufficient support"
                elif vv > max(nv):
                    row.update(status="marked", mark="above_normal_maximum")
                else:
                    row["status"] = "within the normal years' range"
                row["note"] = "; ".join(notes_txt)
                continue
            ref = control.value(f, region, lab)[0] if control is not None else \
                (float(np.mean(vals)) if vals else None)
            row.update(value=v, reference=ref)
            if v is None or ref is None:
                row.update(status="insufficient support",
                           note="; ".join(notes_txt + ["no value to compare"]))
                continue
            ch = v - ref
            row["change"] = ch
            ok, pwhy = percent_policy(note)
            if ok and rr is not None and bool(not_near_zero(ref, rr, NEAR_ZERO_SHARE)):
                row["percent"] = 100.0 * ch / ref
            if ch == 0.0:
                row["status"] = "no change"
            elif typ in NOT_SCORED_YET:
                row["status"] = "not scored yet"
            elif typ not in SCORED_TYPES:
                row["status"] = "no type in the field notes"
            elif sup["n"] < 2 or n["incomplete"]:
                row["status"] = "insufficient support"
                if n["incomplete"]:
                    notes_txt.append("no score: a normal year's value covers less than the "
                                     "whole region (missing_cell_values)")
            else:
                row["near_constant"] = near_constant(sup["sd"], rr, st["near_constant_share"])
                if row["near_constant"]:
                    row["status"] = "near-constant normal"
                else:
                    row.update(status="scored", score=ch / sup["sd"])
            row["note"] = "; ".join(notes_txt)
    out = {**METHODS["screen"].stamp(st), "region": region, "year": int(year),
           "normal_years": list(normal_years),
           "against": "the control" if control is not None else "the normal years",
           "rows": rows, "display_rule": DISPLAY_RULE}
    out["display"] = display(rows)
    return out


def display(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """What the Analyzer is shown first, by DISPLAY_RULE."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for g in list(DC.GROUPS) + sorted({r.get("group") or "" for r in rows} - set(DC.GROUPS)):
        best: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            if (r.get("group") or "") != g or r.get("status") != "scored":
                continue
            b = best.get(r["field"])
            if b is None or abs(r["score"]) > abs(b["score"]):
                best[r["field"]] = r
        top = sorted(best.values(), key=lambda r: -abs(r["score"]))[:3]
        if top:
            groups[g] = [_shown(r) for r in top]
    events = [_shown(r) for r in rows if r.get("status") == "marked"]
    nsy: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if r.get("status") == "not scored yet" and r.get("change") is not None:
            b = nsy.get(r["field"])
            if b is None or abs(r["change"]) > abs(b["change"]):
                nsy[r["field"]] = r
    return {"rule": DISPLAY_RULE, "groups": groups, "events": events,
            "not_scored": [{**_shown(r), "label": "not scored"} for r in nsy.values()]}


def _shown(r: Dict[str, Any]) -> Dict[str, Any]:
    return {k: r.get(k) for k in ("field", "group", "type", "period", "units", "change_units",
                                  "value",
                                  "reference", "change", "percent", "score", "status", "mark",
                                  "n_normal")}


def write_screen_csv(result: Dict[str, Any], path) -> Path:
    """The whole table, one row per field and period, as the report cites it."""
    p = Path(str(path))
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(SCREEN_COLUMNS)
        for r in result["rows"]:
            w.writerow(["" if r.get(c) is None else
                        (repr(r[c]) if isinstance(r[c], float) else r[c]) for c in SCREEN_COLUMNS])
    return p


# ─────────────────────────────────────────────────────────────────────────────
# OVERLAP AND ELEVATION SPLIT
# ─────────────────────────────────────────────────────────────────────────────
SIGN_SHARE_LIMIT = 0.8
SMALL_PART_SHARE = 0.01
NO_TIME_RECORDS = ("not applicable: each map is one period; a period the run lacks gives no "
                   "map, and the caller says so")


def wcorr(x, y, w) -> Optional[float]:
    ww = w / w.sum()
    xm, ym = (ww * x).sum(), (ww * y).sum()
    vx, vy = (ww * (x - xm) ** 2).sum(), (ww * (y - ym) ** 2).sum()
    if vx <= 0 or vy <= 0:
        return None
    return float((ww * (x - xm) * (y - ym)).sum() / np.sqrt(vx * vy))


def equal_area_quarters(rank_by, w) -> List[np.ndarray]:
    """Each cell's share of each quarter of the area, cells ranked by
    rank_by (ties kept in cell order): a cell across a quarter's edge is
    split between the two, so the quarters hold equal areas exactly."""
    o = np.argsort(rank_by, kind="stable")
    ws = w[o]
    hi = np.cumsum(ws)
    lo = hi - ws
    tot = hi[-1]
    out = []
    for k in range(4):
        a, b = tot * k / 4.0, tot * (k + 1) / 4.0
        part = np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0.0, None)
        q = np.zeros_like(w)
        q[o] = part
        out.append(q)
    return out


def _overlap_one(a, b, w, limit: float = SIGN_SHARE_LIMIT) -> Dict[str, Any]:
    if a.size < 3:
        return {"cells": int(a.size), "correlation": None,
                "why": "fewer than three cells hold both maps"}
    r = wcorr(a, b, w)
    quarters = []
    for q in equal_area_quarters(b, w):
        has = q > 0
        quarters.append({"area_km2": float(q.sum()),
                         "second_map_from": float(b[has].min()), "second_map_to": float(b[has].max()),
                         "first_map_mean": float((q * a).sum() / q.sum())})
    tot = float(w.sum())
    sh = {"first_positive": float(w[a > 0].sum() / tot), "first_negative": float(w[a < 0].sum() / tot),
          "second_positive": float(w[b > 0].sum() / tot), "second_negative": float(w[b < 0].sum() / tot)}
    out = {"cells": int(a.size), "area_km2": tot, "correlation": r,
           "correlation_why_not": None if r is not None else "a map is constant over the cells",
           "quarters_by_second_map": quarters, "sign_shares": sh}
    if max(sh.values()) > limit:
        out["same_sign_share"] = None
        out["same_sign_why_not"] = (f"one sign covers more than {100 * limit:g} % of "
                                    "the area, so agreement in sign would say little")
    else:
        same = ((a > 0) & (b > 0)) | ((a < 0) & (b < 0))
        out["same_sign_share"] = float(w[same].sum() / tot)
    return out


def overlap(a, b, mask, weights, bands: Optional[Dict[str, np.ndarray]] = None,
            settings: Optional[Dict[str, Any]] = None,
            named: Optional[Dict[str, Sequence[Any]]] = None) -> Dict[str, Any]:
    """Two per-cell maps over a region: their area-weighted correlation
    across cells, and the first map's area-weighted mean in each quarter of
    the region's area ranked by the second; the same by elevation band. An
    association, never a cause."""
    st = settings_for("overlap", settings, named)
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    m = np.asarray(mask, dtype=bool).ravel()
    both = m & np.isfinite(a) & np.isfinite(b) & (w > 0)
    mm = m & (w > 0)
    out = {**METHODS["overlap"].stamp(st), "covered_share": float(w[both].sum() / w[m].sum())
           if w[m].sum() > 0 else 0.0,
           "missing_cell_values": {"first": missing_cells(a, mm, w),
                                   "second": missing_cells(b, mm, w)},
           "missing_time_records": NO_TIME_RECORDS,
           **_overlap_one(a[both], b[both], w[both], st["sign_share_limit"])}
    if bands:
        out["by_band"] = {}
        for name, bm in bands.items():
            k = both & np.asarray(bm, dtype=bool).ravel()
            out["by_band"][name] = _overlap_one(a[k], b[k], w[k], st["sign_share_limit"])
    return out


def concentration(x, w) -> Optional[float]:
    """The smallest share of the area holding half of the total absolute
    change: cells taken largest |x| first, the last one by the part of its
    area that is needed, so cells tied in |x| give the same answer in any
    order."""
    ax = np.abs(x)
    tot = float((ax * w).sum())
    if tot <= 0:
        return None
    o = np.argsort(-ax, kind="stable")
    c = np.cumsum((ax * w)[o])
    a = np.cumsum(w[o])
    k = int(np.searchsorted(c, 0.5 * tot))
    before_c, before_a = (c[k - 1], a[k - 1]) if k > 0 else (0.0, 0.0)
    return float((before_a + (0.5 * tot - before_c) / ax[o][k]) / w.sum())


def elevation_split(x, mask, weights, bands: Dict[str, np.ndarray],
                    others: Optional[Dict[str, np.ndarray]] = None,
                    settings: Optional[Dict[str, Any]] = None,
                    region: Optional[str] = None,
                    named: Optional[Dict[str, Sequence[Any]]] = None) -> Dict[str, Any]:
    """One map summarised by elevation band (and by each other region and
    its complement) within the requested region: band means, each band's
    positive and negative contributions and its share of the total absolute
    change beside its share of the area, and the concentration of the change.

    WITHIN THE REQUESTED REGION (review round 1, 2026-11-05). Every band and
    every other region is cut to the region `mask` names, whatever masks the
    caller hands in, and counted there; a band with no cell of the region is
    flagged empty, and one under the small-part share of the region's area
    small, so neither passes for a band like the others."""
    st = settings_for("elevation_split", settings, named)
    x = np.asarray(x, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    base = np.asarray(mask, dtype=bool).ravel() & (w > 0)        # the requested region
    region_area = float(w[base].sum())
    m = base & np.isfinite(x)
    area = float(w[m].sum())
    tot_abs = float((np.abs(x[m]) * w[m]).sum())
    out = {**METHODS["elevation_split"].stamp(st), "region": region,
           "region_cells": int(base.sum()), "region_area_km2": region_area,
           "covered_share": covered_share(x, base, w),
           "missing_cell_values": missing_cells(x, base, w),
           "missing_time_records": NO_TIME_RECORDS,
           "area_km2": area, "mean": float((x[m] * w[m]).sum() / area) if area > 0 else None,
           "total_abs_change_x_area": tot_abs,
           "concentration": concentration(x[m], w[m]), "parts": {}}

    def part(pm):
        inside = base & np.asarray(pm, dtype=bool).ravel()      # cut to the region
        k = inside & np.isfinite(x)
        ra = float(w[inside].sum())
        rec = {"region_cells": int(inside.sum()), "cells": int(k.sum()),
               "region_area_share": ra / region_area if region_area > 0 else 0.0,
               "empty": not bool(inside.any()),
               "small": bool(inside.any() and ra < st["small_share"] * region_area),
               "missing_cell_values": missing_cells(x, inside, w)}
        if not k.any():
            rec["area_share"] = 0.0
            return rec
        a = float(w[k].sum())
        rec.update({"area_km2": a, "area_share": a / area,
                    "mean": float((x[k] * w[k]).sum() / a),
                    "positive_x_area": float((np.clip(x[k], 0, None) * w[k]).sum()),
                    "negative_x_area": float((np.clip(x[k], None, 0) * w[k]).sum()),
                    "abs_change_share": (float((np.abs(x[k]) * w[k]).sum()) / tot_abs
                                         if tot_abs > 0 else None),
                    "concentration": concentration(x[k], w[k])})
        return rec

    out["parts"] = {name: part(bm) for name, bm in bands.items()}
    out["flagged"] = {name: ("empty" if r["empty"] else "small")
                      for name, r in out["parts"].items() if r["empty"] or r["small"]}
    if others:
        out["other_regions"] = {name: part(om) for name, om in others.items()}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# THE MECHANISMS (candidate, never rules)
# ─────────────────────────────────────────────────────────────────────────────
def load_mechanisms(path=None) -> List[Dict[str, Any]]:
    p = Path(path) if path else MECHANISMS_PATH
    d = json.loads(p.read_text())
    return list(d.get("mechanisms") or [])


def mechanism_fields(e: Dict[str, Any]) -> List[str]:
    """The fields a mechanism runs through: those its links name, in the
    order they first appear. The links are explicit and may branch (review
    round 1, 2026-11-05), so there is no single ordered list."""
    return list(dict.fromkeys(f for k in e.get("links") or [] for f in (k["from"], k["to"])))


def check_names(fields: Iterable[str], methods: Iterable[str] = (), where: str = "",
                known_fields: Optional[Iterable[str]] = None) -> List[str]:
    """What a contract or a mechanism names that does not exist."""
    kf = set(known_fields) if known_fields is not None else set(catalog_fields())
    out = [f"{where}: no field {f!r}" for f in fields if f not in kf]
    out += [f"{where}: no method {m!r}" for m in methods if m not in METHODS]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# THE CONTRACTS
# ─────────────────────────────────────────────────────────────────────────────
_T = "tests/test_domain_methods.py"
_O = "tests/test_domain_methods_naches_oracle.py"
_SYN = "synthetic cells and series with known answers"
_NACHES_SUITE = "the Naches control's suite, against its regional table (2015-2023)"
_NACHES_DAILY = "the Naches control's daily history, 2015-2023, every vegetated watershed cell"
_NACHES_TIMING = "the Naches control's timing.nc, against the measures run on its daily history"


# THE STATUSES (review round 1, 2026-11-05). Four statuses per method, kept
# apart and each tied to the version it refers to (domain_catalog.
# status_problems): review (draft until the user reviews the contract at this
# text revision and behaviour version), implemented (the behaviour version the
# code runs), tested (the tests that passed, the date and the behaviour version
# they ran on) and available_to_analyzer (true where the Analyzer's access
# calls reach the method and a test exercises it: ANALYZER_ACCESS). The tests are named here as they passed, not taken from
# the validation list, so a test added later is not counted as passed.
_DATE = "2026-11-05"


def _status(text: int, behaviour: int, passed: Sequence[str], date: str = _DATE
            ) -> Dict[str, Any]:
    return {"review": {"state": DRAFT, "text_revision": text, "behaviour_version": behaviour},
            "implemented": {"behaviour_version": behaviour},
            "tested": {"behaviour_version": behaviour, "date": date, "passed": list(passed)},
            "available_to_analyzer": False}


def _in(path: str, *names: str) -> List[str]:
    return [f"{path}::{n}" for n in names]


# THE USER'S APPROVAL (2026-11-05). The user wrote: "record my approval of the
# revised entries for integration, retaining the two-peak settings as
# provisional". Each record names the text revision and behaviour version it
# approves, written out here rather than copied from the contract, so a later
# change to a contract leaves its approval behind (status_problems reports it
# and the review gate fails) instead of moving with it. A setting carries its
# method's record; growth_peak's two-peak settings are approved only as
# provisional values. An approval for integration is not a final scientific
# approval.
_APPROVAL = {"state": APPROVED, "by": "the user", "date": "2026-11-05"}
# THE G PART 1 FIXES (2026-11-08). The user authorized the fixes of decisions
# 1 and 2 (the range across settings, on the comparable cells, and the
# reversal flag), which raise the six timing measures to text revision 3 and
# behaviour version 3; their approval for integration is re-recorded at the
# new versions. No other entry's review changes.
_APPROVAL_G1 = {"state": APPROVED, "by": "the user", "date": "2026-11-08",
                "note": "re-recorded for the G part 1 fixes the user authorized on 2026-11-08 "
                        "(decisions 1 and 2)"}
METHOD_REVIEW: Dict[str, Dict[str, Any]] = {
    **{mid: {**_APPROVAL, "text_revision": 2, "behaviour_version": 2}
       for mid in ("anomaly", "difference", "screen", "overlap", "elevation_split")},
    **{mid: {**_APPROVAL_G1, "text_revision": 3, "behaviour_version": 3}
       for mid in ("snow_peak", "snow_gone", "leaf_green_up", "photosynthesis_onset",
                   "growth_peak", "half_runoff")}}
# AVAILABLE TO THE ANALYZER (package E, 2026-11-05): a method is, once the
# access path (domain_access.SuiteAccess) reaches it and a test exercises it
# through that path; each names the test.
_ACC = "tests/test_domain_access.py"
_ACC_O = "tests/test_domain_access_naches_oracle.py"
ANALYZER_ACCESS: Dict[str, List[str]] = {
    **{mid: [f"{_ACC}::test_every_method_is_a_gated_logged_call",
             f"{_ACC_O}::test_every_method_and_field_through_the_access_path"]
       for mid in ("anomaly", "difference", "screen", "overlap", "elevation_split",
                   "photosynthesis_onset", "growth_peak")},
    **{mid: [f"{_ACC_O}::test_every_method_and_field_through_the_access_path"]
       for mid in ("snow_peak", "snow_gone", "leaf_green_up", "half_runoff")}}
_TWO_PEAK = {"state": PROVISIONAL,
             "why": "approved for integration as a provisional value, pending review of "
                    "representative GPP curves"}
SETTING_REVIEW: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("growth_peak", "near"): _TWO_PEAK, ("growth_peak", "dip"): _TWO_PEAK}


_DATE_G1 = "2026-11-08"                 # the G part 1 fixes: the six timing measures
_TIMING_SHARED = _in(_T, "test_missing_cell_values_apart_from_missing_records",
                     "test_timing_shift_reports_eligibility_change",
                     "test_timing_range_across_settings_on_comparable_cells") + \
    _in(_O, "test_every_timing_measure_gives_a_date_or_a_reason",
        "test_every_timing_reason_follows_its_rule", "test_the_suites_timing_file_holds_the_same")
_PCT_TESTS = ("test_percent_follows_the_notes_policy_not_the_scale",
              "test_a_field_kept_in_percent_changes_in_percentage_points",
              "test_every_method_reports_its_coverage")
METHOD_STATUS: Dict[str, Dict[str, Any]] = {
    "anomaly": _status(2, 2, _in(_T, "test_anomaly_series_and_percent_rules",
                                 "test_anomaly_no_percent_for_interval_event_bounded_or_near_zero",
                                 "test_anomaly_map_per_cell", *_PCT_TESTS) +
                       _in(_O, "test_anomaly_reproduces_the_regional_table")),
    "difference": _status(2, 2, _in(_T, "test_difference_run_minus_control", *_PCT_TESTS)),
    "snow_peak": _status(3, 3, _in(_T, "test_snow_peak_dates_ties_and_no_snow",
                                   "test_water_year_without_its_start",
                                   "test_missing_records_give_records_missing") +
                         _TIMING_SHARED, _DATE_G1),
    "snow_gone": _status(3, 3, _in(_T, "test_snow_gone_searched_from_the_peak",
                                   "test_snow_that_returns_is_flagged", "test_cell_with_no_snow",
                                   "test_water_year_without_its_start",
                                   "test_alternatives_are_computed_beside_the_default",
                                   "test_snow_gone_insufficient_follow_up_is_not_snow_persists") +
                         _TIMING_SHARED, _DATE_G1),
    "leaf_green_up": _status(3, 3, _in(_T, "test_leaf_green_up_rules",
                                       "test_cell_with_weak_leaf_seasonality",
                                       "test_alternatives_are_computed_beside_the_default",
                                       "test_leaf_green_up_rise_mark_alternatives_and_eligibility_"
                                       "change") + _TIMING_SHARED, _DATE_G1),
    "photosynthesis_onset": _status(3, 3, _in(_T, "test_photosynthesis_onset_rules",
                                              "test_cell_without_a_dormant_season",
                                              "test_seasonal_flux_dates",
                                              "test_alternatives_are_computed_beside_the_default")
                                    + _TIMING_SHARED, _DATE_G1),
    "growth_peak": _status(3, 3, _in(_T, "test_growth_peak_two_peaks_and_ties",
                                     "test_alternatives_are_computed_beside_the_default",
                                     "test_growth_peak_flag_area_share_and_denominator") +
                           _TIMING_SHARED, _DATE_G1),
    "half_runoff": _status(3, 3, _in(_T, "test_half_runoff_negative_days",
                                     "test_water_year_without_its_start") + _TIMING_SHARED, _DATE_G1),
    "screen": _status(2, 2, _in(_T, "test_screen_scores_by_type", "test_screen_insufficient_support",
                                "test_screen_rare_event_series",
                                "test_screen_drifting_store_and_bounded_index_not_scored",
                                "test_screen_zero_change_is_no_change",
                                "test_near_constant_normal_kelvin_and_celsius",
                                "test_experiment_score_uses_the_controls_spread",
                                "test_display_rule_one_row_per_field",
                                "test_screen_csv_holds_the_whole_table", *_PCT_TESTS) +
                      _in(_O, "test_screen_2022_holds_every_field_and_period")),
    "overlap": _status(2, 2, _in(_T, "test_overlap_quarters_and_sign_share",
                                 "test_every_method_reports_its_coverage")),
    "elevation_split": _status(2, 2, _in(_T, "test_elevation_split_contributions_and_concentration",
                                         "test_elevation_split_flags_empty_and_small_bands_within_"
                                         "the_region", "test_every_method_reports_its_coverage",
                                         "test_suite_bands_and_other_regions")),
}

_PCT_RULE = Check(
    "percent_policy", "A percent of the reference only where the field notes' reporting policy "
    "allows it (percent.allowed, apart from the physical scale): not for temperatures in any "
    "unit, TWS, ZWT, signed or net quantities such as NEE or NPP, or RH2M, whose changes are in "
    "percentage points; never for an event or a bounded index; and none for a field whose "
    "note has no policy.", percent_policy)
_NEAR_ZERO = Check(
    "reference_not_near_zero", "No percent where the reference is zero or its size is below "
    "the near-zero share of the range of the field's normals over the periods of its kind "
    "(the twelve months for a month; the four seasons for a season, a year or a water year).",
    not_near_zero)
_NEAR_ZERO_SETTING = Setting(
    "near_zero_share", NEAR_ZERO_SHARE, "share of a range",
    "A reference this small beside the field's own range over the periods of its kind makes a "
    "percent large without meaning. The yardstick is a range, so it does not depend on where "
    "a unit puts its zero.")
_COVERAGE_OUT = ("coverage: the share of the region's area each input value covers (per cell, "
                 "counted over the vegetated cells), the inputs with missing cell values (a "
                 "share below 1) and, apart from them, the inputs with missing time records (a "
                 "period the run lacks)")
_MONTH_MAPS = ("Per-cell monthly values are not available yet: the suite references the "
               "monthly history files but has no access path to their per-cell values, so "
               "per-cell results are made for years, water years and seasons only.")
_PRECISION = ("precision", "maps are float32 in the suite and become float64 here; normals, "
                           "changes, percents and shares are float64")
_REGION_WEIGHTS = ("area weights", "a region's value is the suite's own, area-weighted by each "
                                   "cell's area times its land fraction (weight_km2); per cell "
                                   "nothing is weighted, and coverage is the share of that area")
_INPUTS = Check(
    "inputs_complete", "Every day of the period has a record, else not estimable: records "
    "missing, in every cell (a missing time record); and the cell has a value on each of those "
    "days, else not estimable: the cell lacks values (missing cell values, kept apart from "
    "missing records).", missing_inputs)
_TIMING_OUT = ("per region: the area-weighted median and quartiles of the dates; the counts of "
               "dates, of each reason and of each flag, with each flag's area, its share and the "
               "denominator of that share (the region's area, the area with a date); the share "
               "of the area whose inputs are complete, with missing time records and missing "
               "cell values counted apart; the shift in days against the normal years or the "
               "control on the cells with a date in both, beside the change of eligibility (cells "
               "and area shares with a date in both, only in this year, only in the reference, in "
               "neither) and the area the shift covers; the sensitivity record of each "
               "alternative setting with its own change of eligibility; and the range across "
               "the settings: each evaluated setting's median shift on the comparable cells "
               "(those eligible under every evaluated setting, in the year and in its "
               "baseline), the comparable cells' count and share of the region's area beside "
               "each setting's own eligibility, the default, the smallest and the largest shift "
               "each with its setting, whether the largest minus the smallest exceeds 7 days, "
               "and whether the settings reverse the direction of the shift (some give an "
               "earlier date, others a later one)")
_RANGE_LIMITATION = (
    "Its shift can move with the setting: when the largest minus the smallest shift across the "
    "settings exceeds 7 days, a claim giving the shift gives the default, the smallest and the "
    "largest, each with its setting, and a claim on a result whose settings reverse the "
    "direction of the shift carries that warning. The range is on the comparable cells, so it "
    "can differ from each setting's own shift, which is on that setting's eligible cells.")
_DAYS = ("units", "whole days: the day of the year, 1 = its first day, in the calendar the run's "
                  "files declare; a water-year measure counts the days before 1 January back "
                  "(31 December = 0); a shift is in days, positive when later")
_EQUAL_AREA = ("area weights", "region distributions are of equal area: each cell holds its "
                               "area times its land fraction (weight_km2), and a quantile is the "
                               "smallest date at which the cumulative share of the area reaches "
                               "it, the midpoint of two dates where it reaches it exactly")
_DAILY_PRECISION = ("precision", "daily values are float64; a running mean is the mean of each "
                                 "window on its own, so equal windows give equal means; dates "
                                 "are whole days, stored as float32 in timing.nc (exact for "
                                 "whole days); medians and quartiles can fall on half days")
_THRESHOLDS = ("tolerances", "none: a mark is passed when a value reaches it, a value is below "
                             "a threshold when strictly less, and ties go to the earliest day. "
                             "Across settings: the smallest and the largest shift are of the "
                             "settings' medians on the comparable cells, a tie keeping the "
                             "setting listed first (the default, then the alternatives in the "
                             "contract's order); the range exceeds 7 days when the largest "
                             "minus the smallest is more than 7 days by over 1e-9 of a day; a "
                             "median within 1e-9 of a day of zero is neither earlier nor later; "
                             "with no comparable cell there is no range")

_register(anomaly, Method(
    id="anomaly", version=2, text_revision=2,
    estimates="A period's value minus the mean of the normal years' values of the same period, "
              "for a region or per cell, with the normal years' support.",
    inputs=(("the suite's region series of the field for the period's kind (year, water year, "
             "season, month)", "the field's units per period", "one value per region and period"),
            ("the suite's maps, for a per-cell anomaly", "the field's units per period",
             "per cell, for years, water years and seasons"),
            ("the field notes' type and percent policy", "", "per field"),
            ("the normal years and the event years", "", "from the study's plan")),
    fields=(),
    applicability=(
        Check("normal_exists", "At least one normal year holds the period; else not "
              "estimable.", lambda n: n >= 1),
        Check("normal_period_clean", "A normal year's period that holds a month of an event "
              "year (the winter or water year after it) is left out of the normal, with the "
              "reason; the count of normal years shown drops.", left_out_why),
        _PCT_RULE, _NEAR_ZERO),
    settings=(_NEAR_ZERO_SETTING,),
    outcomes=(("the period is absent from the run", "the suite holds no value for it; the "
               "suite's own reason is given"),
              ("not estimable: no normal year has this period", "every normal year lacks it or "
               "is left out"),
              ("no percent, with the reason", "the field notes' policy allows none (or the "
               "note has no policy), an event, a bounded index, or a near-zero normal")),
    outputs=("value, normal (the mean of the normal years used), anomaly, percent or the reason "
             "for none, in the field's units per period; change_units, the units of the "
             "anomaly (percentage points for a field kept in percent)",
             "support: how many normal years, which, their standard deviation (ddof 1), "
             "smallest and largest; the years left out or missing, with reasons",
             _COVERAGE_OUT + "; per cell also the result's own share of the area and its cells "
             "without a value",
             "the reference range the near-zero rule used"),
    limitations=("The normal is the mean of the run's own normal years: it describes this "
                 "model's variability, not the observed climate's.",
                 "A percent is withheld rather than inflated where the normal is near zero; the "
                 "absolute anomaly is always given.",
                 _MONTH_MAPS),
    numerics=(("units", "the field's units per period (a rate's total over the period, a mean "
                        "otherwise); the anomaly is in the same units, and a field kept in "
                        "percent changes in percentage points; the percent is a percent of the "
                        "normal"),
              _REGION_WEIGHTS, _PRECISION,
              ("tolerances", "none: the normal is the plain mean of the normal years used, and "
                             "a change of zero is exactly zero"),
              ("near zero", "no percent where the normal is zero or its size is below the "
                            "near-zero share (0.01) of the range of the field's normals over "
                            "the periods of its kind; the absolute anomaly is always given")),
    validation=((f"{_T}::test_anomaly_series_and_percent_rules", _SYN),
                (f"{_T}::test_anomaly_no_percent_for_interval_event_bounded_or_near_zero", _SYN),
                (f"{_T}::test_anomaly_map_per_cell", _SYN),
                (f"{_T}::test_percent_follows_the_notes_policy_not_the_scale", _SYN),
                (f"{_T}::test_a_field_kept_in_percent_changes_in_percentage_points",
                 "synthetic series under the relative humidity field's own note"),
                (f"{_T}::test_every_method_reports_its_coverage", _SYN),
                (f"{_O}::test_anomaly_reproduces_the_regional_table", _NACHES_SUITE))))

_register(difference, Method(
    id="difference", version=2, text_revision=2,
    estimates="Run minus control for the same period and region, or per cell, with the percent "
              "of the control under the anomaly's rules.",
    inputs=(("both runs' suites: region series or maps of the field", "the field's units per "
             "period", "one value per region and period, or per cell"),
            ("the field notes' type and percent policy", "", "per field")),
    fields=(),
    applicability=(_PCT_RULE, _NEAR_ZERO),
    settings=(_NEAR_ZERO_SETTING,),
    outcomes=(("the period is absent from a run", "either suite lacks it"),
              ("no percent, with the reason", "the field notes' policy allows none (or the note "
               "has no policy), an event, a bounded index, or a near-zero control value")),
    outputs=("the run's value, the control's, the difference, the percent of the control or the "
             "reason for none; change_units, the units of the difference (percentage points "
             "for a field kept in percent)",
             _COVERAGE_OUT + ", for the run's value and the control's apart",
             "the reference range the near-zero rule used: the control's normals over the "
             "periods of the kind when normal years are given, else the control's values of "
             "the same year over those periods"),
    limitations=("A difference within one year says nothing about how unusual it is; the "
                 "screen sets it against the control's normal-year spread.",
                 _MONTH_MAPS),
    numerics=(("units", "the field's units per period; a field kept in percent changes in "
                        "percentage points; the percent is a percent of the control's value"),
              _REGION_WEIGHTS, _PRECISION,
              ("tolerances", "none: the difference is taken on the stored values"),
              ("near zero", "no percent where the control's value is zero or its size is below "
                            "the near-zero share (0.01) of the reference range; the difference "
                            "is always given")),
    validation=((f"{_T}::test_difference_run_minus_control", _SYN),
                (f"{_T}::test_percent_follows_the_notes_policy_not_the_scale", _SYN),
                (f"{_T}::test_a_field_kept_in_percent_changes_in_percentage_points",
                 "synthetic series under the relative humidity field's own note"),
                (f"{_T}::test_every_method_reports_its_coverage", _SYN))))

_SNOW_THR = Setting(
    "threshold_mm", 1.0, "mm of snow water",
    "It marks snow present or gone whatever the size of the snowpack, the same in a thin "
    "snowpack and a deep one.", alternatives=(5.0,))
_TIMING_VALIDATION = (
    (f"{_T}::test_missing_cell_values_apart_from_missing_records", _SYN),
    (f"{_T}::test_timing_shift_reports_eligibility_change", _SYN),
    (f"{_T}::test_timing_range_across_settings_on_comparable_cells", _SYN),
    (f"{_O}::test_every_timing_measure_gives_a_date_or_a_reason", _NACHES_DAILY),
    (f"{_O}::test_every_timing_reason_follows_its_rule", _NACHES_DAILY),
    (f"{_O}::test_the_suites_timing_file_holds_the_same", _NACHES_TIMING))

_register(snow_peak, Method(
    id="snow_peak", version=3, text_revision=3,
    estimates="The day of the water year's largest snow water (H2OSNO), per cell.",
    inputs=(("H2OSNO, daily, from the run's daily history", "mm", "per cell and day"),
            ("the water year's start month", "", "the control's registry entry, October when "
             "it names none")),
    fields=("H2OSNO",),
    applicability=(
        _INPUTS,
        Check("record_holds_the_start", "The run holds the water year's start; a water year "
              "whose first months the run lacks is not estimable in every cell: the peak may "
              "lie before the record starts (a largest after the record starts is no proof).",
              lambda p: not p.starts_before_record),
        Check("holds_snow", "The water year's largest is at least the threshold; else no "
              "snow.", holds_snow)),
    settings=(Setting("threshold_mm", 1.0, "mm of snow water",
                      "The same mark as snow_gone's: below it a cell holds no snowpack to "
                      "date."),),
    outcomes=tuple((REASONS[k][1], w) for k, w in (
        ("records_missing", "a day of the water year has no record"),
        ("cell_values_missing", "the cell lacks a value on a day the records hold"),
        ("before_record", "the run starts after the water year starts"),
        ("no_snow", "the water year's largest is below the threshold"))),
    outputs=("per cell and water year: the date (days of the year the water year ends in, "
             "earlier days counted back: 31 December = 0, 1 October = -91 in a noleap "
             "calendar) or the reason; the largest snow water",
             _TIMING_OUT),
    limitations=("The date of a single day's largest: a long flat peak is dated by its first "
                 "day (ties go to the earliest day).",
                 "A date says when the model's snowpack peaked, not why."),
    numerics=(_DAYS, _EQUAL_AREA, _DAILY_PRECISION, _THRESHOLDS,
              ("near zero", "a largest below the threshold is no snow, never a date; a missing "
                            "value is never taken for zero")),
    validation=((f"{_T}::test_snow_peak_dates_ties_and_no_snow", _SYN),
                (f"{_T}::test_water_year_without_its_start", _SYN),
                (f"{_T}::test_missing_records_give_records_missing", _SYN)) + _TIMING_VALIDATION))

_register(snow_gone, Method(
    id="snow_gone", version=3, text_revision=3,
    estimates="The first day of the first sustained melt-out after the water year's snow peak: "
              "the first of the run of days on which snow water stays below the threshold, per "
              "cell. It marks the end of the main snowpack, not the final disappearance of "
              "snow, which a later snowfall can postpone.",
    inputs=(("H2OSNO, daily, from the run's daily history", "mm", "per cell and day"),
            ("the water year's start month", "", "the control's registry entry, October when "
             "it names none")),
    fields=("H2OSNO",),
    applicability=(
        _INPUTS,
        Check("holds_snow", "The water year's largest is at least the threshold; else no "
              "snow.", holds_snow)),
    settings=(_SNOW_THR,
              Setting("days", 14, "days",
                      "Long enough that a short warm spell inside the snow season is not taken "
                      "for its end.", alternatives=(7,))),
    outcomes=tuple((REASONS[k][1], w) for k, w in (
        ("records_missing", "a day of the water year has no record"),
        ("cell_values_missing", "the cell lacks a value on a day the records hold"),
        ("no_snow", "the water year's largest is below the threshold"),
        ("insufficient_follow_up", "days below the threshold start after the peak but reach "
                                   "the end of the days searched (the water year's end) before "
                                   "the run is long enough: the melt-out may have begun, but "
                                   "its full run cannot be seen"),
        ("snow_persists", "no run of days below the threshold starts after the peak, and none "
                          "reaches the end of the water year: the snow stays"))) + (
        (FLAGS["snow_returns"][1], "a flag beside the date"),
        (FLAGS["partial_record"][1], "a flag on every cell of a water year that starts before "
                                     "the run's records: the search starts at the largest of "
                                     "the months the run holds")),
    outputs=("per cell and water year: the date (as snow_peak's) or the reason, and the flags",
             _TIMING_OUT),
    limitations=("Searched from the peak, not from a date, so a mid-winter melt before the main "
                 "snowpack is not taken for the season's end; a second, later snowpack is "
                 "flagged, not dated.",
                 "The first sustained melt-out, not the final disappearance of snow: snow that "
                 "returns later in the water year is flagged snow returns, and its later "
                 "melt-out is not dated.",
                 "The run of days must end inside the water year, so a water year's date never "
                 "depends on the next one; a melt-out that begins in the water year's last days "
                 "is insufficient follow-up, apart from snow that persists. A record that ends "
                 "before the water year does leaves the water year with records missing.",
                 "A partial-record date is a separate quantity: it never enters a normal, and a "
                 "shift made with one is labelled restricted comparability.",
                 _RANGE_LIMITATION),
    numerics=(_DAYS, _EQUAL_AREA, _DAILY_PRECISION, _THRESHOLDS,
              ("near zero", "snow water strictly below the threshold counts as gone; a largest "
                            "below the threshold is no snow, never a date")),
    validation=((f"{_T}::test_snow_gone_searched_from_the_peak", _SYN),
                (f"{_T}::test_snow_that_returns_is_flagged", _SYN),
                (f"{_T}::test_cell_with_no_snow", _SYN),
                (f"{_T}::test_water_year_without_its_start", _SYN),
                (f"{_T}::test_alternatives_are_computed_beside_the_default", _SYN),
                (f"{_T}::test_snow_gone_insufficient_follow_up_is_not_snow_persists", _SYN))
    + _TIMING_VALIDATION))

_register(leaf_green_up, Method(
    id="leaf_green_up", version=3, text_revision=3,
    estimates="The first day leaf area (TLAI) passes a fraction of the way from its low before "
              "the yearly peak to that peak, per cell.",
    inputs=(("TLAI, daily, from the run's daily history", "m2 leaf per m2 ground",
             "per cell and day"),
            ("the year's start month", "", "the control's registry entry, January when it names "
             "none")),
    fields=("TLAI",),
    applicability=(
        _INPUTS,
        Check("has_leaves", "Leaf area is above zero on some day; else no leaves.", has_leaves),
        Check("rise_enough", "The rise from the low before the peak to the peak is at least "
              "the rise mark of the peak: an eligibility rule on the year's seasonal amplitude, "
              "checked per cell and year; else leaf area changes too little for a green-up "
              "date. It does not diagnose the vegetation type or how often leaves turn over.",
              rise_enough)),
    settings=(Setting("fraction", 0.2, "share of the rise",
                      "Early in the rise, so the date marks the start of leaf growth, yet far "
                      "enough from the low that day-to-day noise does not set it.",
                      alternatives=(0.1, 0.5)),
              Setting("rise_mark", 0.5, "share of the peak",
                      "An eligibility rule on the seasonal amplitude: a date is made only where "
                      "the year's rise is at least half its peak, so that the date marks a clear "
                      "seasonal rise. It is not a diagnosis of the vegetation type or of leaf "
                      "turnover; the alternatives show how the eligible area moves with the "
                      "mark.", alternatives=(0.25, 0.75))),
    outcomes=tuple((REASONS[k][1], w) for k, w in (
        ("records_missing", "a day of the year has no record"),
        ("cell_values_missing", "the cell lacks a value on a day the records hold"),
        ("no_leaves", "leaf area is zero all year"),
        ("little_leaf_change", "the rise is below the rise mark of the peak"))),
    outputs=("per cell and year: the date (day of the year, 1 = its first day) or the reason, "
             "and the rise as a share of the peak",
             _TIMING_OUT + "; the rise-mark alternatives give the change of eligibility beside "
             "the default's"),
    limitations=("Leaf area, not photosynthesis: kept apart from photosynthesis_onset, never "
                 "merged into one map or column.",
                 "The rise mark is an amplitude rule, not a map of vegetation: a cell without a "
                 "date is a cell whose leaf area changes little within the year, whatever its "
                 "vegetation, and the eligible area moves with the mark.",
                 _RANGE_LIMITATION),
    numerics=(_DAYS, _EQUAL_AREA, _DAILY_PRECISION, _THRESHOLDS,
              ("near zero", "a peak of zero is no leaves, never a date; the rise is compared "
                            "with the peak as a share, so nothing is divided by zero")),
    validation=((f"{_T}::test_leaf_green_up_rules", _SYN),
                (f"{_T}::test_cell_with_weak_leaf_seasonality", _SYN),
                (f"{_T}::test_alternatives_are_computed_beside_the_default", _SYN),
                (f"{_T}::test_leaf_green_up_rise_mark_alternatives_and_eligibility_change",
                 _SYN)) + _TIMING_VALIDATION))

_register(photosynthesis_onset, Method(
    id="photosynthesis_onset", version=3, text_revision=3,
    estimates="The first day, after the lowest running-mean GPP before the year's peak, on "
              "which the running mean passes a fraction of that peak, per cell: the first "
              "crossing of the mark, not a sustained onset.",
    inputs=(("GPP, daily totals, from the run's daily history", "gC/m^2 per day",
             "per cell and day"),
            ("the year's start month", "", "the control's registry entry, January when it names "
             "none")),
    fields=("GPP",),
    applicability=(
        _INPUTS,
        Check("photosynthesises", "The year's peak running mean is above zero; else no "
              "photosynthesis.", photosynthesises),
        Check("dormant_season", "The low before the peak is below the mark (a dormant "
              "season); else no dormant season. A watershed whose dormant season falls mid-year "
              "names a year that starts in another month in its registry entry.",
              dormant_season)),
    settings=(Setting("fraction", 0.2, "share of the peak",
                      "Early in the season's rise, yet far enough above a dormant season's low "
                      "that a winter warm spell does not set it.", alternatives=(0.1, 0.3)),
              Setting("window", 7, "days",
                      "A week evens out day-to-day weather without blurring the season's turn; "
                      "the mean is dated by its middle day and uses only windows inside the "
                      "year.")),
    outcomes=tuple((REASONS[k][1], w) for k, w in (
        ("records_missing", "a day of the year has no record"),
        ("cell_values_missing", "the cell lacks a value on a day the records hold"),
        ("no_photosynthesis", "the year's peak running mean is not above zero"),
        ("no_dormant_season", "the low before the peak is not below the mark"))),
    outputs=("per cell and year: the date (day of the year) or the reason",
             _TIMING_OUT),
    limitations=("It measures photosynthetic activity, not leaf-out, and responds to light, "
                 "temperature and water as well as to leaves; kept apart from leaf_green_up.",
                 "The first crossing of the mark, not a sustained onset: a short early pulse of "
                 "photosynthesis that crosses the mark and falls back sets the date.",
                 "With several growing seasons in a year, only the rise to the year's largest "
                 "peak is dated, from the lowest running mean before it.",
                 _RANGE_LIMITATION),
    numerics=(_DAYS, _EQUAL_AREA, _DAILY_PRECISION, _THRESHOLDS,
              ("near zero", "a peak running mean of zero or less is no photosynthesis, never a "
                            "date; the mark is a share of the peak")),
    validation=((f"{_T}::test_photosynthesis_onset_rules", _SYN),
                (f"{_T}::test_cell_without_a_dormant_season", _SYN),
                (f"{_T}::test_seasonal_flux_dates", _SYN),
                (f"{_T}::test_alternatives_are_computed_beside_the_default", _SYN))
    + _TIMING_VALIDATION))

_register(growth_peak, Method(
    id="growth_peak", version=3, text_revision=3,
    estimates="The day of the largest running-mean GPP of the year, per cell, flagged where a "
              "second peak comes close.",
    inputs=(("GPP, daily totals, from the run's daily history", "gC/m^2 per day",
             "per cell and day"),),
    fields=("GPP",),
    applicability=(
        _INPUTS,
        Check("photosynthesises", "The largest running mean is above zero; else no "
              "photosynthesis (a largest of nothing is no date).", photosynthesises)),
    settings=(Setting("window", 7, "days", "A week evens out day-to-day weather without "
                      "blurring the season's peak.", alternatives=(15,)),
              Setting("near", 0.10, "share of the largest",
                      "PROVISIONAL until representative GPP curves are reviewed: a second peak "
                      "this close to the largest could have been the largest in slightly "
                      "different weather."),
              Setting("dip", 0.10, "share of the largest",
                      "PROVISIONAL until representative GPP curves are reviewed: the running "
                      "mean must fall this far between the two for the second to be a peak of "
                      "its own, not a shoulder of the first.")),
    outcomes=tuple((REASONS[k][1], w) for k, w in (
        ("records_missing", "a day of the year has no record"),
        ("cell_values_missing", "the cell lacks a value on a day the records hold"),
        ("no_photosynthesis", "the largest running mean is not above zero"))) + (
        (FLAGS["two_peaks"][1], "a flag beside the date"),),
    outputs=("per cell and year: the date (the middle day of the largest mean; ties go to the "
             "earliest) or the reason, and the two-peaks flag",
             _TIMING_OUT + "; for the two-peaks flag its cells, its area and its share of the "
             "area with a date and of the region's area"),
    limitations=("Where two peaks are flagged the date can jump between them from year to year; "
                 "a shift in such cells measures which peak won, not a change of season.",
                 "The two-peak settings (near, dip) are provisional until representative GPP "
                 "curves are reviewed; the flag's area is reported with its denominator, not "
                 "only its count of cells.",
                 _RANGE_LIMITATION),
    numerics=(_DAYS, _EQUAL_AREA, _DAILY_PRECISION, _THRESHOLDS,
              ("near zero", "a largest running mean of zero or less is no photosynthesis, never "
                            "a date; near and dip are shares of the largest")),
    validation=((f"{_T}::test_growth_peak_two_peaks_and_ties", _SYN),
                (f"{_T}::test_alternatives_are_computed_beside_the_default", _SYN),
                (f"{_T}::test_growth_peak_flag_area_share_and_denominator", _SYN))
    + _TIMING_VALIDATION))

_register(half_runoff, Method(
    id="half_runoff", version=3, text_revision=3,
    estimates="The first day on which the water year's cumulative runoff (QRUNOFF) reaches half "
              "of its total, per cell.",
    inputs=(("QRUNOFF, daily totals, from the run's daily history", "mm per day",
             "per cell and day"),
            ("the water year's start month", "", "the control's registry entry, October when "
             "it names none")),
    fields=("QRUNOFF",),
    applicability=(
        Check("record_holds_the_start", "The run holds the water year's start; else not "
              "estimable in every cell.", lambda p: not p.starts_before_record),
        _INPUTS,
        Check("runoff_positive", "The water year's total is above zero; else no runoff.",
              runoff_positive)),
    settings=(Setting("fraction", 0.5, "share of the total",
                      "Half the water year's runoff: the centre of its timing."),),
    outcomes=tuple((REASONS[k][1], w) for k, w in (
        ("records_missing", "a day of the water year has no record"),
        ("cell_values_missing", "the cell lacks a value on a day the records hold"),
        ("before_record", "the run starts after the water year starts"),
        ("no_runoff", "the total is not above zero"))) + (
        (FLAGS["runoff_falls"][1], "a flag beside the date"),),
    outputs=("per cell and water year: the date (as snow_peak's) or the reason, and the flag",
             _TIMING_OUT),
    limitations=("Daily runoff can be negative, so the cumulative curve can cross the half mark "
                 "more than once: the first crossing is the date, and the cell is flagged.",
                 "A cell's runoff is the water leaving that cell's land, not routed: it is not "
                 "the river at a gauge."),
    numerics=(_DAYS, _EQUAL_AREA, _DAILY_PRECISION, _THRESHOLDS,
              ("near zero", "a total of zero or less is no runoff, never a date")),
    validation=((f"{_T}::test_half_runoff_negative_days", _SYN),
                (f"{_T}::test_water_year_without_its_start", _SYN)) + _TIMING_VALIDATION))

_register(screen, Method(
    id="screen", version=2, text_revision=2,
    estimates="For every catalog field and every period of one year (the year, its water year, "
              "its seasons and months) over one region, the anomaly against the normal years or "
              "the difference from the control, as one complete table scored by variable type.",
    inputs=(("the suite's region series of every field", "the field's units per period",
             "one value per region and period"),
            ("the field notes' process group, variable type, scale and percent policy", "",
             "per field"),
            ("the normal years (and, for an experiment, the control's suite)", "",
             "from the study's plan")),
    fields=(),
    applicability=(
        Check("support", "Two or more normal values, none of them short of the region's area "
              "(missing cell values); else insufficient support and no score. Normal years "
              "without the period (missing time records) lower the count shown.",
              lambda n, incomplete=(): n >= 2 and not incomplete),
        Check("near_constant", "A normal whose standard deviation is zero or at most the "
              "near-constant share of the range of the field's normals over the periods of its "
              "kind (for a year or water-year row, over the four seasons) is flagged and gets no "
              "score, never a large one.", near_constant),
        Check("no_change", "A change of exactly zero is marked no change and gets no score.",
              lambda ch: ch != 0.0),
        _PCT_RULE),
    settings=(Setting("near_constant_share", NEAR_CONSTANT_SHARE, "share of a range",
                      "A normal with almost no spread would turn any change into a huge score; "
                      "the yardstick is a range, so it does not depend on where a unit puts its "
                      "zero."),),
    outcomes=(("scored", "fluxes and states: the change over the normal years' standard "
                         "deviation (ddof 1); for an experiment, the run minus the control over "
                         "the control's normal-year standard deviation for that period"),
              ("marked above_normal_maximum", "an event row (burned area in km2, carbon lost "
               "to fire) above the normal years' largest"),
              ("marked differs_from_control", "an experiment's event row unlike the control's"),
              ("within the normal years' range", "an event row that is no larger than in a "
               "normal year: it stays in the table, unmarked"),
              ("not scored yet", "stores and bounded indices: their absolute and relative "
                                 "change stay in the table"),
              ("no change", "the change is exactly zero"),
              ("near-constant normal", "flagged, never scored"),
              ("insufficient support", "fewer than two normal values or a normal value short of "
                                       "the region's area"),
              ("period absent", "the run lacks the period; the suite's reason is given")),
    outputs=("the table: one row per field and period with the value, the reference, the "
             "absolute change and its units (change_units: percentage points for a field kept "
             "in percent), the percent where the field notes' policy allows it (the anomaly's "
             "rules), the score where its type has one, the normal years' support (count, "
             "spread, near-constant flag), the field's group, type and scale, the share of the "
             "region's area the value covers, and two columns kept apart: missing cell values "
             "(a value, the control's or a normal year's short of the region's area) and missing "
             "time records (an absent period, normal years without it); written whole as a CSV",
             "the display rule, written beside the table, in full: " + DISPLAY_RULE,
             "the rows the display rule shows first"),
    limitations=("A score compares a change with the normal years' spread in this run; it is "
                 "never compared across variable types.",
                 "Stores (to be scored by their drift against the normal years' trend) and "
                 "bounded indices are not scored in this build.",
                 "What the screen turns up was not predicted: it is exploratory, never a tested "
                 "expectation."),
    numerics=(("units", "a row's value, reference and change are in the field's units per "
                        "period (burned area in km2); a field kept in percent changes in "
                        "percentage points; a score has no unit (a change over a standard "
                        "deviation)"),
              ("area weights", "the suite's region values, area-weighted as the suite makes "
                               "them (cell area times land fraction); the screen weights "
                               "nothing itself"),
              _PRECISION,
              ("tolerances", "a change of exactly zero is no change; there is no other "
                             "tolerance"),
              ("near zero", "a normal whose standard deviation is zero or at most the "
                            "near-constant share (0.01) of the range of the field's normals over "
                            "the periods of its kind gets no score; no percent where the "
                            "reference is near zero (the anomaly's rule)")),
    validation=((f"{_T}::test_screen_scores_by_type", _SYN),
                (f"{_T}::test_screen_insufficient_support", _SYN),
                (f"{_T}::test_screen_rare_event_series", _SYN),
                (f"{_T}::test_screen_drifting_store_and_bounded_index_not_scored", _SYN),
                (f"{_T}::test_screen_zero_change_is_no_change", _SYN),
                (f"{_T}::test_near_constant_normal_kelvin_and_celsius", _SYN),
                (f"{_T}::test_experiment_score_uses_the_controls_spread", _SYN),
                (f"{_T}::test_display_rule_one_row_per_field", _SYN),
                (f"{_T}::test_screen_csv_holds_the_whole_table", _SYN),
                (f"{_T}::test_percent_follows_the_notes_policy_not_the_scale", _SYN),
                (f"{_T}::test_a_field_kept_in_percent_changes_in_percentage_points",
                 "synthetic series under the relative humidity field's own note"),
                (f"{_T}::test_every_method_reports_its_coverage", _SYN),
                (f"{_O}::test_screen_2022_holds_every_field_and_period", _NACHES_SUITE))))

_register(overlap, Method(
    id="overlap", version=2, text_revision=2,
    estimates="How two per-cell maps go together over a region: their area-weighted "
              "correlation across cells, and the first map's mean in each quarter of the "
              "region's area ranked by the second.",
    inputs=(("two maps on the same grid", "each map's own units", "per cell"),
            ("the region's mask, the cell weights and the elevation bands", "km2",
             "per cell, from the suite's masks")),
    fields=(),
    applicability=(
        Check("enough_cells", "At least three cells hold both maps; else no correlation.",
              lambda n: n >= 3),
        Check("sign_share", "The share of the area agreeing in sign is reported only when "
              "neither sign of either map covers more than the sign-share limit of the area.",
              lambda shares, limit=SIGN_SHARE_LIMIT: max(shares) <= limit)),
    settings=(Setting("sign_share_limit", SIGN_SHARE_LIMIT, "share of the area",
                      "When one sign covers most of the area, agreement in sign is high by "
                      "chance and says little."),),
    outcomes=(("no correlation", "fewer than three cells, or a map constant over them"),
              ("no same-sign share", "one sign covers more than the limit")),
    outputs=("the correlation, the quarters (their area, the second map's range, the first "
             "map's mean), the sign shares, the same-sign share or why none; the same by band; "
             "the share of the region's area both maps cover, and each map's cells without a "
             "value (missing cell values) with their share of the area; missing time records do "
             "not apply, since each map is one period",),
    limitations=("An association across cells, never a cause: two maps can go together because "
                 "both follow a third.",),
    numerics=(("units", "the correlation has no unit; quarter means are in the first map's "
                        "units; shares are of the area"),
              ("area weights", "every statistic weights a cell by its area times its land "
                               "fraction; the quarters hold equal areas exactly, a cell across "
                               "a quarter's edge split between the two"),
              ("precision", "float64 throughout"),
              ("tolerances", "none: the signs are taken against zero exactly"),
              ("near zero", "a map constant over the cells gives no correlation; a value of "
                            "exactly zero has no sign")),
    validation=((f"{_T}::test_overlap_quarters_and_sign_share", _SYN),
                (f"{_T}::test_every_method_reports_its_coverage", _SYN))))

_register(elevation_split, Method(
    id="elevation_split", version=2, text_revision=2,
    estimates="One map summarised by the study's elevation bands and by each other registered "
              "region and its complement, within the requested region: where the change sits "
              "and how concentrated it is.",
    inputs=(("a map on the grid", "its own units", "per cell"),
            ("the requested region", "", "the suite's mask of it"),
            ("the elevation bands", "m", "the control's registry edges, as the suite's band "
             "masks; a control without edges gets them every 500 m across its elevations, a "
             "fallback recorded in its entry"),
            ("the other registered regions", "", "the suite's masks")),
    fields=(),
    applicability=(
        Check("has_bands", "The suite records band edges and their masks.",
              lambda edges: bool(edges)),
        Check("within_the_region", "Every band and every other region is cut to the requested "
              "region; a band with no cell of the region is flagged empty, and one under the "
              "small-part share of the region's area flagged small.",
              lambda share, small=SMALL_PART_SHARE: share >= small)),
    settings=(Setting("small_share", SMALL_PART_SHARE, "share of the region's area",
                      "A band this small holds too few cells for its mean and shares to say "
                      "much; it is flagged, not dropped."),),
    outcomes=(("an empty band", "no cell of the requested region lies in it: flagged empty, "
                                "zero cells"),
              ("a small band", "under the small-part share of the region's area: flagged "
                               "small, its values kept")),
    outputs=("per band and part, within the region: its cells in the region and with a value, "
             "its share of the region's area, the empty and small flags, its cells without a "
             "value (missing cell values), the area with a value and its share, the mean, the "
             "positive and negative contributions (change times area), the share of the total "
             "absolute change, and the concentration within it",
             "for the region: its name, cells and area, the share its values cover, its cells "
             "without a value, the mean, the concentration (the smallest share of the area "
             "holding half of the total absolute change) and the flagged bands; missing time "
             "records do not apply, since the map is one period"),
    limitations=("A share of a net change that nearly cancels would mislead, so shares are of "
                 "the absolute change and stand beside the share of the area.",
                 "The 500 m step for a control without edges is a fallback, not a scientific "
                 "partition. Edges are set in the registry before any result is seen and must "
                 "not be chosen after seeing results."),
    numerics=(("units", "band means in the map's units; contributions in the map's units times "
                        "km2; shares of the area or of the total absolute change have no unit"),
              ("area weights", "each cell weighs its area times its land fraction; shares of "
                               "the area are of the region's area with a value, and the "
                               "flags use the band's share of the region's whole area"),
              ("precision", "float64 throughout"),
              ("tolerances", "a band under the small-part share (0.01) of the region's area is "
                             "flagged small; there is no other tolerance"),
              ("near zero", "a total absolute change of zero gives no shares of it and no "
                            "concentration")),
    validation=((f"{_T}::test_elevation_split_contributions_and_concentration", _SYN),
                (f"{_T}::test_elevation_split_flags_empty_and_small_bands_within_the_region",
                 _SYN),
                (f"{_T}::test_every_method_reports_its_coverage", _SYN),
                (f"{_T}::test_suite_bands_and_other_regions", _SYN))))
