#!/usr/bin/env python3
"""The Analyzer's calls into a study's suites: maps, series, timing dates,
regions and the methods of domain_methods, each call gated and logged.

THE ACCESS CALLS (package E, 2026-11-05). A generated analysis script reaches
the results store only through one object, `suite`, built by the framework's
script runner. Every call goes through _call(), which does three things in
this order:

  1. asks the gate (a callable the caller supplies) with the call, its keys,
     method and settings, and raises AccessRefused when it says no, before
     anything is read. The framework's gate (analysis_plan.check_lookup)
     refuses every call until the study's analysis plan has passed step 3's
     review, then any lookup the reviewed plan (or an amendment appended
     before it) does not declare, and keeps the order the question type sets;
     what it returns (planned or not, the plan's items, the amendment) goes
     into the call's record;
  2. runs the call, a method's settings first checked by
     domain_methods.settings_for against the default, the contract's named
     alternatives and the settings the recorded plan or an amendment names
     (`named`), so any other setting is refused, never quietly run;
  3. hands a record to the log (a callable the caller supplies): the call, its
     keys, the method and settings, and a hash of what it returned, or why it
     was refused. The screen's record also carries what it ran over and by
     which rules (its scope, scoring rule and display rule), never a value.

Two calls are not lookups. method_contract(method_id) returns a method's full
contract (its section of docs/ANALYSIS_METHODS.md, from the Method record):
it holds no outcome, so it asks no gate and is logged as a contract read.
A timing setting is answered only from what timing.nc holds (the default and
the contract's alternatives it was built with); any other is refused with the
settings that exist, never computed on the fly or replaced by another.

Nothing here decides whether a lookup was planned or writes a file: the gate
and the log are the caller's (src/agents/analysis/analysis_plan.py), so this
module stays the model's side of the door and runs without the framework.
The data catalog's "How to load" section is generated from this class
(how_to_load), so it cannot name a call that does not exist.

Month maps are not in the suite (the monthly history holds them and has no
access path yet), so a month or a month's normal is refused with that reason.

  SuiteAccess(suites, gate, log, normal_years=(), named=None)
  contract(method_id)   a method's full contract, as its section of the document
  how_to_load()         the calls, their arguments and what each returns
  fingerprint(value)    the sha256 a lookup's record carries
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

import domain_methods as M

CONTROL = "control"
NORMAL = "normal"


class AccessError(Exception):
    """A call cannot be answered, and the message says why."""


class AccessRefused(AccessError):
    """The gate said no: nothing was read."""


def _canon(v: Any) -> Any:
    """A JSON-able stand-in for a value, arrays by their own hash."""
    if isinstance(v, np.ndarray):
        return {"array": _array_sha(v)}
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return repr(float(v))
    if isinstance(v, (np.integer, int)) and not isinstance(v, bool):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _array_sha(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(f"{a.dtype.str}{a.shape}".encode())
    if a.dtype.kind in "OUS":
        h.update(json.dumps(a.tolist(), default=str).encode())
    else:
        h.update(a.tobytes())
    return h.hexdigest()


def fingerprint(value: Any) -> str:
    """The sha256 of what a call returned: an array by its dtype, shape and
    bytes, anything else by its canonical JSON with arrays replaced by theirs.
    The numbers check (package F) recomputes a value and compares this."""
    if isinstance(value, np.ndarray):
        return _array_sha(value)
    blob = json.dumps(_canon(value), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


SHOWN_ROW = ("field", "group", "type", "period", "units", "change_units", "value",
             "reference", "change", "percent", "score", "status", "mark", "n_normal",
             "near_constant", "note")


def flat_result(call: str, value: Any) -> Optional[Dict[str, Any]]:
    """What an answered call returned, for its evidence record (package F1,
    2026-11-05): every number, word, flag and list of a method's or a series'
    result by dotted path (\"variants.default.shift.median\"), never a per-cell
    array. The screen keeps its own fields, its display, and of its table each
    row that holds a value, with the row's status and marks. None for a map
    or a call that returns no dict. Nothing is cut: the claim trace reads a
    claim's numbers, periods and warnings from here, and the numbers check
    (package F2) recomputes them."""
    if not isinstance(value, dict):
        return None
    src: Any = value
    if call == "screen":
        src = {k: v for k, v in value.items() if k != "rows"}
        src["rows"] = [{k: r.get(k) for k in SHOWN_ROW}
                       for r in value.get("rows") or []
                       if r.get("value") is not None or r.get("change") is not None]
    out: Dict[str, Any] = {}

    def scalar(x):
        if isinstance(x, (np.bool_, bool)):
            return bool(x)
        if isinstance(x, (np.integer, int)):
            return int(x)
        if isinstance(x, (np.floating, float)):
            return float(x)
        return x if x is None or isinstance(x, str) else str(x)

    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(x, np.ndarray):
            if x.ndim == 0:                       # a number; any array is per-cell
                out[path] = scalar(x.item())
        elif isinstance(x, (list, tuple)):
            if len(x) <= 64 and all(not isinstance(v, (dict, list, tuple, np.ndarray))
                                    for v in x):
                out[path] = [scalar(v) for v in x]
            else:
                for i, v in enumerate(x):
                    walk(v, f"{path}.{i}")
        else:
            out[path] = scalar(x)
    walk(src, "")
    return out


def array_hashes(value: Any, path: str = "") -> Dict[str, str]:
    """The hash of every array a dict result holds, by dotted path (package
    F2, 2026-11-05). An overlap or an elevation split names its maps only by
    their hashes; with these the numbers check finds which lookup returned
    each map (a per-cell anomaly's "anomaly", say) and makes it again."""
    out: Dict[str, str] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(array_hashes(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(value, np.ndarray) and value.ndim > 0:
        out[path] = _array_sha(value)
    return out


def _shape(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return list(value.shape)
    if isinstance(value, dict):
        return sorted(str(k) for k in value)[:20]
    return type(value).__name__


def contract(method_id: str) -> Dict[str, Any]:
    """A method's full contract from its Method record: its section of
    docs/ANALYSIS_METHODS.md as text (domain_methods.contract_lines, which that
    document is made with), its version, text revision, four statuses and
    settings with their reviews. It holds no outcome."""
    m = M.METHODS.get(str(method_id))
    if m is None:
        raise AccessError(f"no method {method_id!r} (the methods: {', '.join(M.METHODS)})")
    i = list(M.METHODS).index(m.id) + 1
    return {"method": m.id, "version": m.version, "text_revision": m.text_revision,
            "status": json.loads(json.dumps(m.status, default=str)),
            "settings": [{"name": s.name, "default": s.default, "units": s.units,
                          "reason": s.reason, "alternatives": list(s.alternatives),
                          "review": dict(s.review or {})} for s in m.settings],
            "text": "\n".join(M.contract_lines(i, m))}


def _screen_record(v: Dict[str, Any]) -> Dict[str, Any]:
    """What a screen ran over and by which rules, with no value in it: the
    record an exploratory study's other lookups wait for (plan section 3.5)."""
    fields: List[Any] = []
    periods: List[Any] = []
    for r in v.get("rows") or []:
        if r.get("field") not in fields:
            fields.append(r.get("field"))
        if r.get("period") not in periods:
            periods.append(r.get("period"))
    m = M.METHODS["screen"]
    scored = dict(m.outcomes).get("scored", "")
    checks = " ".join(c.text for c in m.applicability
                      if c.id in ("support", "near_constant", "no_change"))
    return {"screen_record": {
        "scope": {"region": v.get("region"), "year": v.get("year"),
                  "against": v.get("against"), "normal_years": v.get("normal_years"),
                  "fields": fields, "periods": periods, "rows": len(v.get("rows") or [])},
        "scoring_rule": f"scored (fluxes and states): {scored}. {checks}",
        "display_rule": v.get("display_rule"),
        "method": v.get("method"), "version": v.get("version"),
        "settings": v.get("settings")}}


def _screen_summary(v: Dict[str, Any]) -> Dict[str, Any]:
    """The screen's record (_screen_record) and its whole table, every row
    with every column, which the lookups log writes as a CSV beside itself
    (the user's decision 4 on G part 1, 2026-11-08); a log that writes no
    file keeps it in the line."""
    return {**_screen_record(v), "screen_table": [dict(r) for r in v.get("rows") or []]}


class SuiteAccess:
    """The `suite` object of a generated analysis script. Every call is
    gated and logged; a refused call raises AccessRefused or AccessError with
    the reason, and is logged as refused."""

    def __init__(self, suites: Dict[str, Any],
                 gate: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
                 log: Callable[[Dict[str, Any]], None], normal_years: Sequence[int] = (),
                 named: Optional[Dict[str, Dict[str, Sequence[Any]]]] = None,
                 study_dir: Optional[str] = None):
        if not callable(gate) or not callable(log):
            # NO DEFAULT GATE AND NO DEFAULT LOG: an object without them would
            # be an ungated, unlogged door to every outcome in the store
            raise AccessError("SuiteAccess needs a gate and a log")
        if CONTROL not in suites:
            raise AccessError("SuiteAccess needs the control's suite (run 'control')")
        self._dirs = {str(k): str(v) for k, v in suites.items()}
        self._open: Dict[str, M.Suite] = {}
        self._gate, self._log = gate, log
        self.normal_years = [int(y) for y in normal_years]
        self._named = {k: dict(v) for k, v in (named or {}).items()}
        self._study = str(study_dir) if study_dir else None   # for run_record
        for mid in M.TIMING_IDS:
            setattr(self, mid, self._timing_call(mid))

    # ── the door ────────────────────────────────────────────────────────
    @property
    def runs(self) -> List[str]:
        return list(self._dirs)

    def _suite(self, run: str) -> M.Suite:
        if run not in self._dirs:
            raise AccessError(f"no suite for run {run!r} (the study's suites: {self.runs})")
        if run not in self._open:
            self._open[run] = M.Suite(self._dirs[run])
        return self._open[run]

    def _call(self, call: str, keys: Dict[str, Any], fn: Callable[[], Any],
              method: Optional[str] = None, settings: Optional[Dict[str, Any]] = None,
              summary: Optional[Callable[[Any], Dict[str, Any]]] = None):
        rec = {"call": call, "keys": _canon(keys), "method": method,
               "settings": _canon(settings) if settings is not None else None}
        try:
            # the gate sees the call as made (values unconverted) and returns
            # the marks the record carries: planned or not, and under what
            marks = self._gate({"call": call, "keys": dict(keys), "method": method,
                                "settings": dict(settings or {})})
        except Exception as e:                                   # noqa: BLE001
            self._log(dict(rec, refused=f"the gate: {e}"))
            raise AccessRefused(str(e)) from None
        rec.update(marks or {})
        try:
            value = fn()
        except (AccessError, M.MethodError) as e:
            self._log(dict(rec, refused=str(e)))
            raise AccessError(str(e)) from None
        except Exception as e:                                   # noqa: BLE001
            # an unforeseen failure is logged too: every call leaves a record
            self._log(dict(rec, failed=f"{type(e).__name__}: {e}"))
            raise
        res = flat_result(call, value) if (marks or {}).get("support") is None else None
        arr = array_hashes(value) if isinstance(value, dict) else None
        self._log(dict(rec, sha256=fingerprint(value), shape=_shape(value),
                       **(summary(value) if summary else {}),
                       **({"result": res} if res else {}),
                       **({"arrays": arr} if arr else {})))
        return value

    def method_contract(self, method_id: str) -> Dict[str, Any]:
        """A method's full contract (as in docs/ANALYSIS_METHODS.md, with its
        statuses and versions): no outcome, so it needs no reviewed plan and
        is logged as a contract read, not a lookup."""
        rec = {"call": "method_contract", "keys": {"method": str(method_id)},
               "contract_read": True}
        try:
            out = contract(method_id)
        except AccessError as e:
            self._log(dict(rec, refused=str(e)))
            raise
        self._log(dict(rec, version=out["version"], text_revision=out["text_revision"],
                       sha256=fingerprint(out)))
        return out

    def run_record(self, run: str) -> Dict[str, Any]:
        """A run's full record, each change in the server's words: no outcome,
        so it needs no review and is logged apart from the lookups.

        The catalog prints a run compactly (package F1, 2026-11-05): each
        change's id, settings and meaning, and every "does not mean" warning
        once. This gives the rest on demand: the server's summary, meaning and
        warning of every change, the years planned and passed, and the full
        text the catalog's run row used to hold. Like method_contract it needs
        no reviewed plan and is logged as a contract read, never a lookup."""
        import domain_catalog as DCAT                      # noqa: N812 (no cycle)
        rec = {"call": "run_record", "keys": {"run": str(run)}, "contract_read": True}
        if str(run) == CONTROL:
            out = {"run": CONTROL, "changes": [], "text": "the control: no change"}
        else:
            study = DCAT.study_record(self._study) if self._study else {"runs": []}
            hit = [r for r in study.get("runs") or [] if r["run"] == str(run)]
            if not hit:
                why = (f"no run {run!r} in the study's experiment record (its runs: "
                       f"{[r['run'] for r in study.get('runs') or []]})")
                self._log(dict(rec, refused=why))
                raise AccessError(why)
            r = hit[0]
            out = {"run": r["run"], "changes": r["changes"], "years": study.get("years"),
                   "years_passed": study.get("years_passed"),
                   "text": " ".join(DCAT._change_text(c) for c in r["changes"])
                   or "no change recorded"}
        self._log(dict(rec, sha256=fingerprint(out)))
        return out

    def _settings(self, mid: str, settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """The settings a method call runs with: refused unless each is the
        default, a named alternative, or named by the plan or an amendment."""
        return M.settings_for(mid, settings, self._named.get(mid))

    def _normals(self, normal_years: Optional[Sequence[int]]) -> List[int]:
        ny = [int(y) for y in (normal_years if normal_years is not None else self.normal_years)]
        if not ny:
            raise AccessError("no normal years: the analysis plan names none")
        return ny

    # ── the products ────────────────────────────────────────────────────
    def get_map(self, run: str, field: str, period: str) -> np.ndarray:
        """A field's per-cell map, NaN where a cell has no value: a
        year (2015), water year (WY2016) or season (2015-JJA), or the normal
        years' mean of one (normal, normal-WY, normal-JJA). Months are refused:
        month maps are not in the suite."""
        def fn():
            s = self._suite(run)
            p = str(period)
            if p == NORMAL or p.startswith(NORMAL + "-"):
                part = p[len(NORMAL) + 1:]
                if not part:
                    kind, pt = "year", None
                elif part == "WY":
                    kind, pt = "wateryear", None
                elif part in M.SEASON_NAMES:
                    kind, pt = "season", part
                elif part.isdigit():
                    raise AccessError("month maps are not in the suite (the monthly history "
                                      "holds them and has no access path yet)")
                else:
                    raise AccessError(f"{p!r} is not a normal period (normal, normal-WY, "
                                      f"normal-JJA)")
                n = M._normal_of(lambda lab: (s.map(field, lab), None), kind, pt,
                                 self._normals(None), [])
                x = M._stack_mean(n["used"])
                if x is None:
                    raise AccessError(f"no normal year holds {field} for {p}")
                return np.asarray(x, dtype=np.float64)
            x = s.map(field, p)
            if x is None:
                raise AccessError(s.absent_why(p) or f"the suite holds no map of {field} "
                                  f"for {p}")
            return x
        return self._call("get_map", {"run": run, "field": field, "period": period}, fn)

    def get_series(self, run: str, field: str, region: str, step: str) -> Dict[str, Any]:
        """A region's series of a field at one step (day, month, season, year,
        wateryear): {labels, values, covered, units, how}. A value is NaN
        where the suite holds none; covered is the share of the region's area
        it covers."""
        def fn():
            s = self._suite(run)
            if step not in ("day", "month", "season", "year", "wateryear"):
                raise AccessError(f"no step {step!r} (day, month, season, year, wateryear)")
            s.mask(region)
            f = s._series_file(step)
            if field not in f["vals"]:
                raise AccessError(f"the suite holds no {step} series of {field}")
            i = f["regions"].index(region)
            cov = f["vals"].get(field + "_covered")
            return {"field": field, "region": region, "step": step,
                    "labels": list(f["labels"]),
                    "values": np.asarray(f["vals"][field][i], dtype=np.float64),
                    "covered": (np.asarray(cov[i], dtype=np.float64) if cov is not None
                                else None),
                    "units": s.units(field, step), "how": s.how(field)}
        return self._call("get_series", {"run": run, "field": field, "region": region,
                                         "step": step}, fn)

    def _stored(self, run: str, measure: str) -> List[Any]:
        """[(label, settings)] of every setting a run's timing.nc holds for a
        measure, read from the variables' attributes (no value is read)."""
        import netCDF4
        import domain_suite as S
        if measure not in M.TIMING_IDS:
            raise AccessError(f"no timing measure {measure!r} ({', '.join(M.TIMING_IDS)})")
        p = self._suite(run).dir / S.TIMING_FILE
        if not p.is_file():
            raise AccessError(f"the suite of {run} has no {S.TIMING_FILE}")
        out: Dict[str, Dict[str, Any]] = {}
        with netCDF4.Dataset(p) as d:
            for v in d.variables.values():
                if getattr(v, "method", None) == measure:
                    out.setdefault(str(v.setting), json.loads(v.settings))
        return list(out.items())

    def _stored_choice(self, run: str, measure: str,
                       settings: Optional[Dict[str, Any]]) -> Optional[str]:
        """The label of the stored setting a call names (None: every stored
        setting). A setting timing.nc does not hold is refused, naming it and
        the settings that exist: nothing is computed on the fly or replaced."""
        if not settings:
            return None
        m = M.METHODS[measure] if measure in M.METHODS else None
        if m is None:
            raise AccessError(f"no timing measure {measure!r} ({', '.join(M.TIMING_IDS)})")
        want = m.defaults()
        for k, v in settings.items():
            m.setting(k)                         # MethodError for a name it does not have
            want[k] = v
        stored = self._stored(run, measure)
        for lab, st in stored:
            if st == want:
                return lab
        asked = ", ".join(f"{k}={M.format_value(v)}" for k, v in settings.items())
        have = "; ".join(f"{lab} (" + ", ".join(f"{k}={M.format_value(x)}" for k, x in st.items())
                         + ")" for lab, st in stored)
        raise AccessError(f"{measure}: timing.nc of {run} holds no setting {asked}; it holds "
                          f"{have or 'none'}. A timing setting is never computed on the fly "
                          f"(that needs the daily history) or replaced by another")

    def _timing_year(self, run: str, measure: str, year: int) -> Dict[str, Any]:
        import netCDF4
        import domain_suite as S
        s = self._suite(run)
        p = s.dir / S.TIMING_FILE
        if measure not in M.TIMING_IDS:
            raise AccessError(f"no timing measure {measure!r} ({', '.join(M.TIMING_IDS)})")
        if not p.is_file():
            raise AccessError(f"the suite of {run} has no {S.TIMING_FILE}")
        out = {"method": measure, "kind": M.TIMING_KIND[measure], "year": int(year),
               "period": (M.period_label(M.TIMING_KIND[measure], int(year))),
               "variants": {}}
        with netCDF4.Dataset(p) as d:
            d.set_auto_mask(False)
            ys = [int(y) for y in np.asarray(d["year"][:])]
            if int(year) not in ys:
                raise AccessError(f"timing.nc of {run} holds no {year} (it holds {ys})")
            j = ys.index(int(year))
            parts: Dict[str, Dict[str, Any]] = {}
            for vn, v in d.variables.items():
                if getattr(v, "method", None) != measure:
                    continue
                part = vn.rsplit("__", 1)[-1]
                lab = str(v.setting)
                e = parts.setdefault(lab, {"settings": json.loads(v.settings)})
                out["version"] = int(v.version)
                e[part] = np.asarray(v[..., j]).ravel()
            if not parts:
                why = ((s.manifest.get("timing") or {}).get("not_built") or {}).get(measure)
                raise AccessError(f"timing.nc of {run} holds no {measure}"
                                  + (f": {why}" if why else ""))
        for lab, e in parts.items():
            e["date"] = np.asarray(e["date"], dtype=np.float64)
            e["reason"] = np.asarray(e["reason"], dtype=np.int64)
            e["flags"] = np.asarray(e["flags"], dtype=np.int64)
        out["variants"] = parts
        return out

    def get_timing(self, run: str, measure: str, year: int,
                   settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """A timing measure's per-cell dates for one year, every stored
        setting (or the one `settings` names, refused unless timing.nc holds
        it): {variant: {settings, date, reason, flags}}. Each is a flat
        array over the grid's cells (row by row), with a reason code where
        there is no date; a water year's measure is filed under the year it
        ends."""
        def fn():
            lab = self._stored_choice(run, measure, settings)
            got = self._timing_year(run, measure, year)["variants"]
            return got if lab is None else {lab: got[lab]}
        return self._call("get_timing", {"run": run, "measure": measure, "year": int(year)},
                          fn, method=measure, settings=settings)

    def region_mask(self, region: str) -> np.ndarray:
        """A region's cells, True inside, on the grid's (lat, lon)."""
        return self._call("region_mask", {"region": region},
                          lambda: np.asarray(self._suite(CONTROL).mask(region), dtype=bool))

    def cell_area(self) -> np.ndarray:
        """Each cell's area in km2 times its land fraction: the weight of
        every area-weighted statistic."""
        return self._call("cell_area", {}, lambda: np.asarray(self._suite(CONTROL).weights))

    def lat_lon(self) -> Dict[str, np.ndarray]:
        """The grid's latitude and longitude axes (for drawing a map)."""
        def fn():
            import netCDF4
            import domain_suite as S
            with netCDF4.Dataset(self._suite(CONTROL).dir / S.MASKS_FILE) as d:
                return {"lat": np.asarray(d["lat"][:], dtype=np.float64),
                        "lon": np.asarray(d["lon"][:], dtype=np.float64)}
        return self._call("lat_lon", {}, fn)

    def find_fields(self, words: str) -> List[Dict[str, Any]]:
        """The control's monthly fields whose name or long name holds every
        word: name, long name, units, in_catalog, in_suite. Names only: a
        field the suite does not hold has no access path yet."""
        def fn():
            import netCDF4
            import gridded as G
            s = self._suite(CONTROL)
            run_dir = s.manifest.get("run_dir")
            found = G.discover(run_dir) if run_dir and Path(str(run_dir)).is_dir() else {}
            streams = {n: G.open_stream(run_dir, n) for n in sorted(found)}
            mon = next((n for n in sorted(streams) if streams[n].cadence == "monthly"), None)
            if mon is None:
                raise AccessError("the control has no monthly stream to search")
            rows = set(M.catalog_fields())
            want = [w.lower() for w in str(words).split() if w.strip()]
            out = []
            with netCDF4.Dataset(streams[mon].files[0]) as d:
                for n, v in d.variables.items():
                    ln = str(getattr(v, "long_name", "") or "")
                    text = f"{n} {ln}".lower()
                    if want and all(w in text for w in want):
                        out.append({"name": n, "long_name": ln,
                                    "units": str(getattr(v, "units", "") or ""),
                                    "in_catalog": n in rows, "in_suite": n in s.fields})
            return out
        return self._call("find_fields", {"words": words}, fn)

    # ── the methods, as calls ───────────────────────────────────────────
    def anomaly(self, field: str, period: str, region: Optional[str] = None,
                run: str = CONTROL, normal_years: Optional[Sequence[int]] = None,
                settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """domain_methods.anomaly: a period's value minus the normal years'
        mean, for a region, or per cell when region is None; its values or
        its reasons."""
        def fn():
            st = self._settings("anomaly", settings)
            return M.anomaly(self._suite(run), field, period, self._normals(normal_years),
                             region=region, settings=st, named=self._named.get("anomaly"))
        return self._call("anomaly", {"run": run, "field": field, "period": period,
                                      "region": region, "normal_years": normal_years},
                          fn, method="anomaly", settings=settings)

    def difference(self, run: str, field: str, period: str, region: Optional[str] = None,
                   control: str = CONTROL, settings: Optional[Dict[str, Any]] = None
                   ) -> Dict[str, Any]:
        """domain_methods.difference: a run minus the control, same period and
        region (or per cell); its values or its reasons."""
        def fn():
            st = self._settings("difference", settings)
            return M.difference(self._suite(run), self._suite(control), field, period,
                                region=region, normal_years=self.normal_years or None,
                                settings=st, named=self._named.get("difference"))
        return self._call("difference", {"run": run, "control": control, "field": field,
                                         "period": period, "region": region},
                          fn, method="difference", settings=settings)

    def screen(self, year: int, region: str, run: str = CONTROL,
               normal_years: Optional[Sequence[int]] = None,
               settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """domain_methods.screen: every catalog field and period of one year
        over a region, scored by type, with its display rule. Against the
        normal years for the control, against the control for another run."""
        def fn():
            st = self._settings("screen", settings)
            ctl = None if run == CONTROL else self._suite(CONTROL)
            return M.screen(self._suite(run), region, int(year), self._normals(normal_years),
                            control=ctl, settings=st, named=self._named.get("screen"))
        return self._call("screen", {"run": run, "year": int(year), "region": region,
                                     "normal_years": normal_years},
                          fn, method="screen", settings=settings, summary=_screen_summary)

    def overlap(self, a: np.ndarray, b: np.ndarray, region: str, by_band: bool = False,
                settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """domain_methods.overlap: two per-cell maps over a region, their
        area-weighted correlation and quarter split (by band too when
        by_band); an association, never a cause."""
        def fn():
            st = self._settings("overlap", settings)
            s = self._suite(CONTROL)
            return M.overlap(a, b, s.mask(region), s.weights,
                             bands=s.band_masks(region) if by_band else None, settings=st,
                             named=self._named.get("overlap"))
        return self._call("overlap", {"region": region, "by_band": by_band,
                                      "a": _array_sha(np.asarray(a)),
                                      "b": _array_sha(np.asarray(b))},
                          fn, method="overlap", settings=settings)

    def elevation_split(self, x: np.ndarray, region: str,
                        settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """domain_methods.elevation_split: one per-cell map by elevation band
        within the region (and by each other region and its complement)."""
        def fn():
            st = self._settings("elevation_split", settings)
            s = self._suite(CONTROL)
            return M.elevation_split(x, s.mask(region), s.weights, s.band_masks(region),
                                     others=s.other_regions(region), settings=st, region=region,
                                     named=self._named.get("elevation_split"))
        return self._call("elevation_split", {"region": region,
                                              "x": _array_sha(np.asarray(x))},
                          fn, method="elevation_split", settings=settings)

    def _timing_call(self, mid: str):
        def call(year: int, region: str, run: str = CONTROL, against: str = NORMAL,
                 normal_years: Optional[Sequence[int]] = None,
                 settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            def fn():
                # refused unless timing.nc holds the setting (run's and control's)
                for r in dict.fromkeys([run] + ([CONTROL] if against == CONTROL else [])):
                    self._stored_choice(r, mid, settings)
                self._settings(mid, settings)
                s = self._suite(CONTROL)
                by = {int(year): self._timing_year(run, mid, year)}
                ctl = ny = None
                if against == CONTROL:
                    ctl = {int(year): self._timing_year(CONTROL, mid, year)}
                elif against == NORMAL:
                    ny = self._normals(normal_years)
                    for y in ny:
                        by.setdefault(int(y), self._timing_year(run, mid, y))
                else:
                    raise AccessError(f"against is {NORMAL!r} or {CONTROL!r}, not {against!r}")
                return M.timing_summary(mid, by, int(year), s.mask(region), s.weights,
                                        normal_years=ny, control=ctl)
            return self._call(mid, {"run": run, "year": int(year), "region": region,
                                    "against": against, "normal_years": normal_years},
                              fn, method=mid, settings=settings)
        call.__name__ = mid
        call.__doc__ = (f"domain_methods.{mid} from timing.nc: one year over a region, "
                        f"every stored setting, the shift against the normal years "
                        f"(against='normal') or the control's same year (against='control').")
        return call


def _bare(fn) -> str:
    """A call's signature without annotations or self: what a script types."""
    sig = inspect.signature(fn)
    ps = [p.replace(annotation=inspect.Parameter.empty) for p in sig.parameters.values()
          if p.name != "self"]
    return str(sig.replace(parameters=ps, return_annotation=inspect.Signature.empty))


def _first_sentence(doc: Optional[str]) -> str:
    text = " ".join((doc or "").split())
    i = text.find(". ")
    return text if i < 0 else text[:i + 1]


def how_to_load() -> List[Dict[str, str]]:
    """The calls a script makes on `suite`, read off SuiteAccess: each one's
    signature and the first sentence of its docstring; the six timing
    measures share one signature and are listed together."""
    out = []
    for name, fn in inspect.getmembers(SuiteAccess, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        out.append({"call": f"suite.{name}{_bare(fn)}", "does": _first_sentence(fn.__doc__)})
    probe = SuiteAccess._timing_call(object.__new__(SuiteAccess), "MEASURE")
    out.append({"call": f"suite.MEASURE{_bare(probe)}",
                "does": f"MEASURE is one of {', '.join(M.TIMING_IDS)}: "
                        + _first_sentence(probe.__doc__).replace("domain_methods.MEASURE ",
                                                                 "the measure ")})
    return out
