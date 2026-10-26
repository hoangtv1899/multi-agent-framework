#!/usr/bin/env python3
"""The changes a whole-domain experiment may make to its control run.

A whole-domain experiment is a control run and, for a what-if question, runs
that differ from it by named changes (docs/ELM_DOMAIN_EXPERIMENTS.md). This
module is the server's list of those changes, and the one place that knows
what each does to ELM:

  declare()          every change, available or not: what it alters in the
                     case, what the difference from the control means and
                     does not mean, how its effect is checked, what it costs
  check(...)         a proposed experiment against the list: wont_build,
                     unusual and facts, the three lists the conceptual sweep's
                     check returns. The server reports; it does not grade
  run_recipe(...)    one run's recipe: the control's, with its changes made
  build_run(...)     a run's recipe and checks, with the files its changes
                     write written first (weather_scale: domain_weather.py)
  expectations(...)  the settings and output checks a run's jobs apply
  settings_differences(...)  a run's resolved namelists against the
                     control's: what differs, and whether that is exactly
                     what its changes should alter

A change that needs a tool this server does not have is still listed, with
`available: false` and the tool it needs in `requires`, so a question that
needs it is answered `partial` with that tool named, never designed around.
"""
from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# A run's name becomes part of a case name and a folder name.
RUN_NAME = re.compile(r"[a-z][a-z0-9_]{0,23}")
# Hours per model year when the control's facts give none: the Naches box on
# one 40-core Compy node (0.78-0.83 h, chains 789148-789149 and 790947-790951).
DEFAULT_HOURS_PER_YEAR = 0.83

# THE LIST. Strings may hold {param} placeholders, filled when a change is
# resolved. `alters` is what the run's recipe carries: user_nl lines (component,
# key, value), XML settings (name, value), or {"inputs": writer}: files the
# writer makes for the run when it is built (build_run), with the user_nl
# lines and user files that read them. `checks.settings` are keys of the
# resolved namelists (CaseDocs/<file>) and their values after the change;
# `checks.output` are rules on history fields, checked by every year's gate;
# a writer adds its own. A param with a `default` may be left out.
_CHANGES: Tuple[Dict[str, Any], ...] = (
    {"id": "fire_off",
     "summary": "switch ELM's fire model off",
     "params": {},
     "alters": [{"user_nl": "elm", "key": "use_nofire", "value": ".true."}],
     "meaning": "the difference from the control is what the model's own fire "
                "does: the fire ELM's fire model lights and spreads, in its "
                "own places and years",
     "not_meaning": "the effect of an observed fire. The control's fire is the "
                    "model's own, unrelated in place and year to fires that "
                    "happened: compare its burned area with the observed one "
                    "in the control's facts",
     "checks": {"settings": [{"file": "lnd_in", "key": "use_nofire", "value": True}],
                "output": [{"stream": "h0", "field": "FAREA_BURNED", "rule": "all_zero",
                            "why": "no fire can burn with the fire model off"}]},
     "cost": "the control's: one job per model year",
     "available": True},
    {"id": "co2_ppmv",
     "summary": "hold atmospheric CO2 at another constant level",
     "params": {"ppmv": {"kind": "number", "lo": 180.0, "hi": 1200.0, "units": "ppmv",
                         "what": "the constant CO2 molar ratio"}},
     "alters": [{"xml": "CCSM_CO2_PPMV", "value": "{ppmv}"}],
     "needs": {"setting": "ELM_CO2_TYPE", "equals": "constant",
               "why": "a CO2 level can be held constant only when the control's CO2 is"},
     "meaning": "the difference from the control is the model's response to CO2 at "
                "{ppmv} ppmv instead of the control's level, through photosynthesis "
                "and stomata; this change alters nothing else",
     "not_meaning": "a climate-change experiment: CO2 alone changes here, so only its "
                    "direct effects on the land appear; the weather changes only where "
                    "another change of the same run alters it",
     "checks": {"settings": [{"file": "lnd_in", "key": "co2_ppmv", "value": "{ppmv}"}],
                "output": []},
     "cost": "the control's: one job per model year",
     "available": True},
    {"id": "ndep_year",
     "summary": "hold nitrogen deposition at another year's level",
     "params": {"year": {"kind": "integer", "lo": 1, "hi": 9999, "units": "year",
                         "what": "the year of the deposition dataset to repeat"}},
     "alters": [{"user_nl": "elm", "key": "stream_year_first_ndep", "value": "{year}"},
                {"user_nl": "elm", "key": "stream_year_last_ndep", "value": "{year}"}],
     "meaning": "the difference from the control is the model's response to nitrogen "
                "deposition at its {year} level, held every year, instead of the "
                "control's year",
     "not_meaning": "a change in any other forcing, or in deposition beyond what the "
                    "control's deposition dataset holds for {year}",
     "checks": {"settings": [{"file": "lnd_in", "key": "stream_year_first_ndep",
                              "value": "{year}"},
                             {"file": "lnd_in", "key": "stream_year_last_ndep",
                              "value": "{year}"}],
                "output": []},
     "cost": "the control's: one job per model year",
     "available": True},
    {"id": "nutrient_supplement",
     "summary": "supply nitrogen or phosphorus as plants demand it, or not at all",
     "params": {"nitrogen": {"kind": "choice", "choices": ["NONE", "ALL"],
                             "what": "suplnitro: ALL supplies all the nitrogen plants "
                                     "demand, NONE none beyond the model's own"},
                "phosphorus": {"kind": "choice", "choices": ["NONE", "ALL"],
                               "what": "suplphos, likewise for phosphorus"}},
     "alters": [{"user_nl": "elm", "key": "suplnitro", "value": "'{nitrogen}'"},
                {"user_nl": "elm", "key": "suplphos", "value": "'{phosphorus}'"}],
     "meaning": "the difference from the control shows how much nutrient limitation "
                "shapes the result: with ALL a nutrient no longer limits growth "
                "(here nitrogen {nitrogen}, phosphorus {phosphorus})",
     "not_meaning": "a fertilisation experiment with real amounts: ALL supplies "
                    "whatever plants demand, without limit",
     "checks": {"settings": [{"file": "lnd_in", "key": "suplnitro", "value": "{nitrogen}"},
                             {"file": "lnd_in", "key": "suplphos", "value": "{phosphorus}"}],
                "output": []},
     "cost": "the control's: one job per model year",
     "available": True},
    {"id": "weather_repeat_year",
     "summary": "make chosen model years read another year's weather, such as a dry "
                "year three times running",
     "params": {}, "alters": [],
     "meaning": "the difference from the control is the response to the weather of "
                "the repeated year in the chosen years",
     "not_meaning": "a drought of a kind the record has never held: only real years "
                    "are repeated",
     "checks": {"settings": [], "output": []},
     "cost": "the control's: one job per model year",
     "available": False,
     "requires": ["a weather tool that sets, year by year, which forcing year the "
                  "model reads (piece 4b)"]},
    {"id": "weather_scale",
     "summary": "warm or cool the air the land model receives, or multiply its rain "
                "and snow, in chosen months and cells",
     "params": {"temperature_K": {"kind": "number", "lo": -10.0, "hi": 10.0, "units": "K",
                                  "default": 0.0,
                                  "what": "added to the air temperature, every step"},
                "precip_factor": {"kind": "number", "lo": 0.0, "hi": 5.0, "units": "1",
                                  "default": 1.0,
                                  "what": "multiplies rain and snow alike, every step"},
                "months": {"kind": "months", "default": list(range(1, 13)),
                           "what": "the months it applies in, 1-12"},
                "region": {"kind": "region", "default": None,
                           "what": "a region registered for the control, all of its "
                                   "cells; the whole domain when absent"}},
     "alters": [{"inputs": "weather_scale"}],
     "meaning": "the difference from the control is the response to {weather_text}, "
                "in {months_text}{region_text}, with every other input the control's",
     # What the derived words read as in the published list, before a run's
     # parameters fill them (resolve() writes them from the parameters).
     "menu_text": {"weather_text": "the air temperature changed by temperature_K and "
                                   "rain and snow multiplied by precip_factor",
                   "months_text": "the chosen months",
                   "region_text": " over the chosen region (the whole domain when none "
                                  "is named)"},
     "not_meaning": "a consistent change of climate: the data atmosphere splits rain "
                    "from snow, caps the air's moisture at saturation and sets its "
                    "potential temperature from the recorded temperature before the "
                    "amount is added, so a warmer run does not turn snow into rain; the "
                    "air's moisture (specific humidity), radiation and wind are the "
                    "control's, and the land model computes relative humidity and air "
                    "density from the temperature it receives, so warmer air is also "
                    "drier air to it (relative humidity about 6-7 % of its value lower "
                    "per kelvin), which its fire model and leaves respond to; in a "
                    "cooler run the recorded moisture can exceed saturation for the "
                    "cooled air, so dew and frost form that the recorded weather would "
                    "not make",
     "checks": {"settings": [], "output": [],
                "by_writer": "settings: the run's datm_in names the anomaly streams, "
                             "every per-stream list grows by them alone, and the case "
                             "holds the stream files written for it; output, every year, "
                             "in every cell-day both runs report: daily TBOT minus the "
                             "control's equals the amount (to 0.001 K), and daily rain "
                             "and daily snow over the control's each equal the factor "
                             "(to 1e-5), in the chosen months and active cells, all "
                             "three the control's elsewhere; a cell-day reported by one "
                             "run only, or a change that reached no reported cell, "
                             "fails"},
     "cost": "the control's: one job per model year, plus a small anomaly file",
     "control": {"reuse": True,
                 "why": "its anomaly file is written on the grid of the control's "
                        "resolved datm_in, and every year is checked against the "
                        "control's output"},
     "limits": ["model years 2007-2299 only: the data atmosphere reads its anomaly "
                "streams for 2006-2300, the first year's January value takes effect "
                "just before that year begins, and the last call falls on 1 January "
                "after the last year",
                "the control's daily history (h1) must hold TBOT, RAIN and SNOW for "
                "every year of the experiment: each year is checked against it",
                "a run that changes the weather and something else gives their joint "
                "effect; each change's meaning is stated against the control alone"],
     "available": True},
    {"id": "burn_footprint",
     "summary": "burn chosen cells on a chosen date, such as an observed fire's "
                "footprint or a shifted copy of it",
     "params": {}, "alters": [],
     "meaning": "the difference from the control is the response to that burn",
     "not_meaning": "the model's own fire, which still burns elsewhere unless it is "
                    "also switched off",
     "checks": {"settings": [], "output": []},
     "cost": "the control's, plus stopping at the burn date",
     "available": False,
     "requires": ["a burn tool that imposes a fire footprint on chosen cells at a "
                  "date, by editing the saved state or by a code change that reads "
                  "a burn map"]},
    {"id": "parameter_value",
     "summary": "change one value in ELM's parameter file",
     "params": {}, "alters": [],
     "meaning": "the difference from the control is the sensitivity to that parameter",
     "not_meaning": "a calibration: one value changed, the rest as the control's",
     "checks": {"settings": [], "output": []},
     "cost": "the control's: one job per model year",
     "available": False,
     "requires": ["a parameter tool that writes a run's own copy of the parameter "
                  "file with one value changed"]},
)
_BY_ID = {c["id"]: c for c in _CHANGES}


class ChangeError(Exception):
    """A change cannot be resolved or applied, and the message says why."""


# ─────────────────────────────────────────────────────────────────────────────
# THE LIST
# ─────────────────────────────────────────────────────────────────────────────
def declare() -> Dict[str, Any]:
    """The change list, for reception and the planner, and the rules it obeys."""
    return {
        "what": "the changes a whole-domain experiment may make to its control "
                "run. Every run is the control plus changes from this list; a "
                "question needing one marked available: false is answered "
                "partial, with that change's `requires` named",
        "changes": [_published(c) for c in _CHANGES],
        "rules": [
            "a run's name is lowercase letters, digits and _ (at most 24), not "
            "'control'",
            "two changes in one run may not alter the same setting",
            "every run covers the experiment's years, consecutive, inside the "
            "forcing the control's case reads and, for a reused control, inside "
            "the years it has output for",
            "a run's resolved namelists must differ from the control's only by "
            "its changes, checked before its model is built",
        ],
    }


def _published(c: Dict[str, Any]) -> Dict[str, Any]:
    """A change as the list publishes it: words a run's parameters fill in
    (resolve) shown by their generic menu text, so no reader meets a
    placeholder that is not a parameter."""
    out = {k: v for k, v in c.items() if k != "menu_text"}
    for k, v in (c.get("menu_text") or {}).items():
        out["meaning"] = out["meaning"].replace("{" + k + "}", v)
    return out


def get(change_id: str) -> Dict[str, Any]:
    if change_id not in _BY_ID:
        raise ChangeError(f"no change {change_id!r}; the list is "
                          f"{', '.join(_BY_ID)}")
    return _BY_ID[change_id]


def _fill(x, params: Dict[str, Any]):
    """x with its {param} placeholders filled, recursively."""
    if isinstance(x, str):
        try:
            return x.format(**params)
        except (KeyError, IndexError, ValueError) as e:
            raise ChangeError(f"cannot fill {x!r}: {e}") from None
    if isinstance(x, list):
        return [_fill(v, params) for v in x]
    if isinstance(x, dict):
        return {k: _fill(v, params) for k, v in x.items()}
    return x


def _with_defaults(change: Dict, params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    for k, p in (change.get("params") or {}).items():
        if k not in out and "default" in p:
            out[k] = copy.deepcopy(p["default"])
    return out


def _param_problems(change: Dict, params: Dict[str, Any]) -> List[str]:
    out = []
    spec = change.get("params") or {}
    for k in sorted(set(params) - set(spec)):
        out.append(f"{change['id']} has no parameter {k!r} (it takes "
                   f"{sorted(spec) or 'none'})")
    for k, p in spec.items():
        if k not in params:
            out.append(f"{change['id']} needs {k} ({p.get('what', '')})")
            continue
        v = params[k]
        if p["kind"] == "months":
            if not isinstance(v, list) or not v or len(set(v)) != len(v) or any(
                    isinstance(m, bool) or not isinstance(m, int) or not 1 <= m <= 12
                    for m in v):
                out.append(f"{change['id']} {k}={v!r} is not a list of distinct months "
                           f"1-12")
            continue
        if p["kind"] == "region":
            if v is not None and (not isinstance(v, str) or not v.strip()):
                out.append(f"{change['id']} {k}={v!r} is not a region's name")
            continue
        if p["kind"] in ("number", "integer"):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or \
                    not math.isfinite(v):
                out.append(f"{change['id']} {k}={v!r} is not a number")
                continue
            if p["kind"] == "integer" and v != int(v):
                out.append(f"{change['id']} {k}={v!r} is not a whole number")
            if p.get("lo") is not None and v < p["lo"] or \
                    p.get("hi") is not None and v > p["hi"]:
                out.append(f"{change['id']} {k}={v} is outside {p['lo']}-{p['hi']} "
                           f"{p.get('units', '')}".rstrip())
        elif p["kind"] == "choice" and v not in p["choices"]:
            out.append(f"{change['id']} {k}={v!r} is not one of {p['choices']}")
    return out


def resolve(item: Dict[str, Any]) -> Dict[str, Any]:
    """One change of a run, {"id", "params"}, with its text, alterations and
    checks filled in. Raises ChangeError for an unknown, unavailable or
    malformed change."""
    if not isinstance(item, dict) or "id" not in item:
        raise ChangeError(f"a change is {{'id': ..., 'params': {{...}}}}, not {item!r}")
    c = get(item["id"])
    if not c["available"]:
        raise ChangeError(f"{c['id']} is not available here: it needs "
                          f"{'; '.join(c['requires'])}")
    params = _with_defaults(c, dict(item.get("params") or {}))
    bad = _param_problems(c, params)
    if bad:
        raise ChangeError("; ".join(bad))
    fmt = {k: (int(v) if c["params"][k]["kind"] == "integer" else v)
           for k, v in params.items()}
    if c["id"] == "weather_scale":
        import domain_weather as W
        if not W.active(fmt):
            raise ChangeError("weather_scale changes nothing: temperature_K is 0 and "
                              "precip_factor is 1")
        fmt = dict(fmt, months=sorted(fmt["months"]))
    out = {"id": c["id"], "params": fmt}
    text = dict(fmt, **_text(c["id"], fmt))
    for k in ("summary", "alters", "meaning", "not_meaning", "checks", "cost"):
        out[k] = _fill(copy.deepcopy(c[k]), text)
    return out


_MONTH = ("January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December")


def _text(change_id: str, p: Dict[str, Any]) -> Dict[str, str]:
    """Words a change's meaning is written with, made from its params."""
    if change_id != "weather_scale":
        return {}
    parts = []
    if abs(p["temperature_K"]) > 1e-12:
        parts.append(f"air {abs(p['temperature_K']):g} K "
                     f"{'warmer' if p['temperature_K'] > 0 else 'cooler'}")
    if abs(p["precip_factor"] - 1.0) > 1e-12:
        parts.append(f"rain and snow multiplied by {p['precip_factor']:g}")
    ms = p["months"]
    return {"weather_text": " and ".join(parts),
            "months_text": ("every month" if ms == list(range(1, 13)) else
                            ", ".join(_MONTH[m - 1] for m in ms)),
            "region_text": f", over {p['region']}" if p.get("region") else ""}


# The settings a writer's files are read through, for the one-change-per-
# setting rule.
_WRITER_TARGETS = {"weather_scale": ["user_nl_datm:anomaly_forcing", "user_nl_datm:tintalgo"]}


def _targets(resolved: Dict) -> List[str]:
    out = []
    for a in resolved["alters"]:
        if "inputs" in a:
            out += _WRITER_TARGETS.get(a["inputs"], [f"inputs:{a['inputs']}"])
        elif "user_nl" in a:
            out.append(f"user_nl_{a['user_nl']}:{a['key']}")
        else:
            out.append(f"xml:{a['xml']}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# A RUN'S RECIPE
# ─────────────────────────────────────────────────────────────────────────────
_KEY_LINE = r"^\s*{}\s*="


def _set_user_nl(lines: List[str], key: str, value: str) -> List[str]:
    """lines with `key = value`: the key's line replaced where it stands, or
    appended. A value continued over lines is not a key this list alters."""
    pat = re.compile(_KEY_LINE.format(re.escape(key)), re.IGNORECASE)
    out, done = [], False
    for line in lines:
        if pat.match(line):
            if not done:
                out.append(f"{key} = {value}")
                done = True
            continue
        out.append(line)
    if not done:
        out.append(f"{key} = {value}")
    return out


def run_case_name(control_case: str, run: str) -> str:
    return f"{control_case}_{run}"


def run_recipe(control, run: Dict[str, Any], written: Optional[Dict[str, Dict]] = None):
    """The recipe of one run: the control's recipe (a domain_case.Recipe) with
    the run's changes made, as a new case named <control case>_<run>.

    The control's recipe is not touched. Its notes gain one line naming the
    experiment's control and the changes, so the record says what the case is.
    A change that writes files for the run takes them from `written` (its
    id -> what the writer returned; build_run makes it) and raises without.
    """
    import domain_case as DC
    name = run.get("name", "")
    if not RUN_NAME.fullmatch(name or "") or name == "control":
        raise ChangeError(f"run name {name!r}: lowercase letters, digits and _ "
                          f"(at most 24), not 'control'")
    resolved = [resolve(c) for c in (run.get("changes") or [])]
    seen: Dict[str, str] = {}
    for r in resolved:
        for t in _targets(r):
            if t in seen:
                raise ChangeError(f"run {name}: {seen[t]} and {r['id']} both alter {t}")
            seen[t] = r["id"]
    r = DC.Recipe.from_json(control.to_json())
    r.case = run_case_name(control.case, name)
    r.source = {}
    r.copied_from = control.copied_from or str(control.case_dir)
    for c in resolved:
        for a in c["alters"]:
            if "inputs" in a:
                w = (written or {}).get(c["id"])
                if w is None:
                    raise ChangeError(f"run {name}: {c['id']} writes files for the run; "
                                      f"build it with build_run")
                for comp, key, value in w["user_nl"]:
                    r.user_nl[comp] = _set_user_nl(list(r.user_nl.get(comp, [])), key,
                                                   value)
                r.user_files.update(w["user_files"])
                continue
            if "user_nl" in a:
                comp = a["user_nl"]
                r.user_nl[comp] = _set_user_nl(list(r.user_nl.get(comp, [])), a["key"],
                                               a["value"])
            else:
                hits = [k for k, e in enumerate(r.xml) if e[0] == a["xml"]]
                if hits:
                    e = r.xml[hits[-1]]
                    r.xml[hits[-1]] = (e[0], str(a["value"])) + tuple(e[2:])
                else:
                    r.xml.append((a["xml"], str(a["value"])))
    r.notes = list(r.notes) + [
        f"run {name} of an experiment on {control.case}: "
        + ("; ".join(f"{c['id']}{c['params'] or ''}" for c in resolved) or "no changes")]
    bad = r.check()
    if bad:
        raise ChangeError(f"run {name}: the recipe cannot be used: " + "; ".join(bad))
    return r


def expectations(run: Dict[str, Any], control_case_dir: Optional[str] = None,
                 written: Optional[Dict[str, Dict]] = None) -> Dict:
    """What the run's jobs check: its resolved namelists against the
    control's (settings), and rules on its history fields every year (output);
    with a writer's own, from `written`."""
    resolved = [resolve(c) for c in (run.get("changes") or [])]
    settings: Dict[str, Dict[str, Any]] = {}
    output: List[Dict] = []
    case_files: Dict[str, str] = {}
    for c in resolved:
        for s in c["checks"]["settings"]:
            settings.setdefault(s["file"], {})[s["key"]] = s["value"]
        for o in c["checks"]["output"]:
            output.append(dict(o, change=c["id"]))
        w = (written or {}).get(c["id"])
        if w is None and any("inputs" in a for a in c["alters"]):
            raise ChangeError(f"run {run.get('name')}: {c['id']} writes files for the run, "
                              f"and its checks come from them; build it with build_run")
        if w:
            for f, kv in (w.get("settings") or {}).items():
                settings.setdefault(f, {}).update(kv)
            output += [dict(o, change=c["id"]) for o in w.get("output") or []]
            case_files.update(w.get("case_files") or {})
    out = {"run": run.get("name"), "control_case_dir": control_case_dir,
           "changes": [{k: c[k] for k in ("id", "params", "summary", "meaning",
                                           "not_meaning")} for c in resolved],
           "settings": settings, "output": output}
    if case_files:
        out["case_files"] = case_files
    return out


def build_run(control, run: Dict[str, Any], inputs_dir, years,
              control_case_dir=None, control_entry: Optional[Dict] = None):
    """(recipe, expectations) of one run, with the files its changes write
    written under inputs_dir first. The one call build_domain_runs makes."""
    resolved = [resolve(c) for c in (run.get("changes") or [])]
    written: Dict[str, Dict] = {}
    for c in resolved:
        if any("inputs" in a for a in c["alters"]):
            import domain_weather as W
            try:
                written[c["id"]] = W.inputs(c["params"], years, inputs_dir,
                                            control_case_dir or control.case_dir,
                                            control_entry)
            except W.WeatherError as e:
                raise ChangeError(f"run {run.get('name')}: {c['id']}: {e}") from None
    return (run_recipe(control, run, written),
            expectations(run, control_case_dir, written))


# ─────────────────────────────────────────────────────────────────────────────
# THE SETTINGS CHECK
# ─────────────────────────────────────────────────────────────────────────────
# The resolved namelists compared, run against control. drv_in holds paths
# under the case's own folders and the job's dates; it is not compared.
SETTINGS_FILES = ("lnd_in", "datm_in")


def _same(a, b) -> bool:
    """Two namelist values (lists from read_namelist) equal, numbers to 1e-9."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if isinstance(x, (int, float)) and not isinstance(x, bool) and \
                isinstance(y, (int, float)) and not isinstance(y, bool):
            if abs(float(x) - float(y)) > 1e-9 * max(1.0, abs(float(y))):
                return False
        elif x != y:
            return False
    return True


def _as_expected(v):
    """An expectation's value as read_namelist would read it."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().strip("'\"")
    if s.lower() in (".true.", "true"):
        return True
    if s.lower() in (".false.", "false"):
        return False
    try:
        return float(s)
    except ValueError:
        return s


def _want_text(want, control_v) -> str:
    """An expectation in words, for a reason."""
    if isinstance(want, dict) and "appends" in want:
        tail = want.get("values") or [f"{st}..." for st in want.get("starts") or []]
        return (f"the control's {list(control_v or [])} with "
                + (", ".join(repr(t) for t in tail) if tail else f"{want['appends']} entries")
                + " appended")
    return str(want)


def _meets(run_v, control_v, want) -> bool:
    """Does a run's namelist value meet one expectation? A scalar is the
    value after the change; a list is the whole list; {"appends": n} is the
    control's list kept as it was with n entries after it, each starting
    with `starts[i]` or equal to `values[i]` where those are given (a change
    that adds streams grows every per-stream list by its own entries only)."""
    run_v, control_v = list(run_v or []), list(control_v or [])
    if isinstance(want, dict) and "appends" in want:
        n = int(want["appends"])
        if len(run_v) != len(control_v) + n or not _same(run_v[:len(control_v)], control_v):
            return False
        tail = run_v[len(control_v):]
        starts = want.get("starts")
        if starts and not all(str(x).strip().startswith(st) for x, st in zip(tail, starts)):
            return False
        values = want.get("values")
        return not values or _same(tail, [_as_expected(v) for v in values])
    if isinstance(want, list):
        return _same(run_v, [_as_expected(v) for v in want])
    return _same(run_v, [_as_expected(want)])


def settings_differences(run_case_dir, control_case_dir, expected: Dict[str, Dict],
                         run_case: str, control_case: str) -> Dict[str, Any]:
    """The run's CaseDocs namelists against the control's: every key that
    differs, and whether the differences are exactly the expected ones with
    the expected values. Case names in values are put on one footing first."""
    from run_gate import read_namelist
    import tempfile
    files: Dict[str, Any] = {}
    reasons: List[str] = []
    for name in sorted(set(SETTINGS_FILES) | set(expected)):
        pr = Path(run_case_dir) / "CaseDocs" / name
        pc = Path(control_case_dir) / "CaseDocs" / name
        if not pr.is_file() or not pc.is_file():
            missing = [str(p) for p in (pr, pc) if not p.is_file()]
            reasons.append(f"{name}: cannot compare, missing {missing}")
            files[name] = {"missing": missing}
            continue
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "run", Path(d) / "control"
            a.write_text(pr.read_text(errors="replace").replace(run_case, "CASE"))
            b.write_text(pc.read_text(errors="replace").replace(control_case, "CASE"))
            nr, nc = read_namelist(a), read_namelist(b)
        want = {k.lower(): v for k, v in (expected.get(name) or {}).items()}
        diff = {k: {"control": nc.get(k), "run": nr.get(k)}
                for k in sorted(set(nr) | set(nc))
                if not _same(nr.get(k, []), nc.get(k, []))}
        unexpected = sorted(set(diff) - set(want))
        wrong = sorted(k for k in want if not _meets(nr.get(k, []), nc.get(k, []), want[k]))
        # A change to the value the control already has is not an error here;
        # check() reports it as unusual before anything is built.
        no_op = sorted(k for k in set(want) - set(diff) if k not in wrong)
        def show(v):
            return "unset" if v is None else v
        for k in unexpected:
            reasons.append(f"{name}: {k} differs from the control "
                           f"({show(diff[k]['control'])} -> {show(diff[k]['run'])}), and no "
                           f"change of this run alters it")
        for k in wrong:
            reasons.append(f"{name}: {k} is {show(nr.get(k))}, the change sets "
                           f"{_want_text(want[k], nc.get(k))}")
        files[name] = {"differences": diff, "expected": want, "unexpected": unexpected,
                       "wrong": wrong, "no_op": no_op}
    return {"pass": not reasons, "reasons": reasons, "files": files}


def case_files_differences(run_case_dir, want: Dict[str, str]) -> Dict[str, Any]:
    """Files the run's case must hold as they were written for it
    ({path in the case: sha256}), such as a user stream file CIME copies
    into CaseDocs: each one present and unchanged."""
    import domain_weather as W
    reasons, files = [], {}
    for rel, sha in sorted((want or {}).items()):
        p = Path(run_case_dir) / rel
        if not p.is_file():
            files[rel] = {"present": False}
            reasons.append(f"{rel} is missing: the case does not read the file written "
                           f"for the run")
            continue
        got = W.sha256(p)
        files[rel] = {"present": True, "same": got == sha}
        if got != sha:
            reasons.append(f"{rel} is not the file written for the run (sha256 differs)")
    return {"pass": not reasons, "reasons": reasons, "files": files}


# ─────────────────────────────────────────────────────────────────────────────
# THE CHECK OF AN EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────
def _years_problems(years, facts: Optional[Dict], reuse: bool) -> Tuple[List[str], List[str]]:
    bad, odd = [], []
    if not isinstance(years, list) or not years or \
            not all(isinstance(y, int) and not isinstance(y, bool) for y in years):
        return ["years must be a non-empty list of whole years"], odd
    if years != list(range(years[0], years[0] + len(years))):
        bad.append(f"years {years} are not consecutive: each continues from the "
                   f"one before")
    if not facts:
        odd.append("the control has no facts, so its forcing and output years "
                   "could not be confirmed")
        return bad, odd
    f0, f1 = (facts.get("forcing_years") or [None, None])[:2]
    if f0 is not None and (years[0] < f0 or years[-1] > f1):
        bad.append(f"years {years[0]}-{years[-1]} are outside the forcing the control "
                   f"reads ({f0}-{f1}); later years would recycle earlier weather")
    if reuse:
        o0, o1 = (facts.get("output_years") or [None, None])[:2]
        if o0 is None:
            bad.append("the reused control's output years are unknown")
        elif years[0] < o0 or years[-1] > o1:
            bad.append(f"years {years[0]}-{years[-1]} are outside the reused control's "
                       f"output ({o0}-{o1}): those years have nothing to compare with")
        start = facts.get("start_year")
        if start is not None and years[0] != start:
            bad.append(f"a run starts from the control's saved state at {start}, so "
                       f"its years start at {start}, not {years[0]} (compare later "
                       f"years only, after running from {start})")
    return bad, odd


def _setting(facts: Optional[Dict], name: str):
    return ((facts or {}).get("settings") or {}).get(name)


def check(experiment: Dict[str, Any], control: Optional[Dict[str, Any]] = None
          ) -> Dict[str, Any]:
    """A proposed experiment against the change list and the control's facts.

    experiment  {"control": {"name": ..., "reuse": true|false},
                 "runs": [{"name": ..., "changes": [{"id": ..., "params": {...}}]}],
                 "years": [first, ..., last]}
    control     the control's registry entry with its facts, or None

    Returns wont_build (each one a question for the user, never a silent
    correction), unusual (it will build, but is odd), facts (measurements the
    study's caveats are made from), the resolved runs and the cost.
    """
    wont, odd, facts = [], [], []
    exp = experiment or {}
    cfacts = (control or {}).get("facts")
    reuse = bool((exp.get("control") or {}).get("reuse"))
    years = exp.get("years")
    b, o = _years_problems(years, cfacts, reuse)
    wont += [{"where": "years", "what": x} for x in b]
    odd += [{"where": "years", "what": x} for x in o]
    runs = exp.get("runs")
    if not isinstance(runs, list):
        wont.append({"where": "runs", "what": "runs must be a list of runs; an empty list "
                                              "is the control alone"})
        runs = []
    elif not runs:
        facts.append({"where": "runs", "what": "no run differs from the control: the "
                                               "experiment " + ("reads the reused control "
                                               "alone and spends nothing" if reuse else
                                               "is the control, run alone")})
    names = [r.get("name") for r in runs if isinstance(r, dict)]
    for n in sorted({n for n in names if names.count(n) > 1}):
        wont.append({"where": f"run {n}", "what": "two runs have this name"})
    resolved_runs = []
    for run in runs:
        if not isinstance(run, dict):
            wont.append({"where": "runs", "what": f"a run is an object, not {run!r}"})
            continue
        name = run.get("name")
        where = f"run {name}"
        if not RUN_NAME.fullmatch(str(name or "")) or name == "control":
            wont.append({"where": where, "what": "a run's name is lowercase letters, "
                                                 "digits and _ (at most 24), not 'control'"})
        items = run.get("changes") or []
        if not items:
            odd.append({"where": where, "what": "this run has no changes: it repeats the "
                                                "control, which only checks that it "
                                                "reproduces"})
        rs, targets = [], {}
        for it in items:
            cid = (it or {}).get("id") if isinstance(it, dict) else None
            try:
                rc = resolve(it)
            except ChangeError as e:
                c = _BY_ID.get(cid)
                wont.append({"where": f"{where}, change {cid}", "what": str(e),
                             **({"requires": c["requires"]} if c and not c["available"]
                                else {})})
                continue
            for t in _targets(rc):
                if t in targets:
                    wont.append({"where": where, "what": f"{targets[t]} and {rc['id']} "
                                                         f"both alter {t}"})
                targets[t] = rc["id"]
            region = rc["params"].get("region") if rc["id"] == "weather_scale" else None
            need_ctl = get(rc["id"]).get("control") or {}
            if need_ctl.get("reuse") and not reuse:
                wont.append({"where": f"{where}, change {rc['id']}",
                             "what": f"{rc['id']} needs a reused control: "
                                     f"{need_ctl.get('why', '')}"})
            if rc["id"] == "weather_scale":
                import domain_weather as W
                here = f"{where}, change {rc['id']}"
                whole_years = isinstance(years, list) and years and all(
                    isinstance(y, int) and not isinstance(y, bool) for y in years)
                if whole_years:
                    bad_years = W.year_problem(years)
                    if bad_years:
                        wont.append({"where": here, "what": bad_years})
                h1 = ((cfacts or {}).get("streams") or {}).get("h1")
                if not h1 or h1.get("error"):
                    odd.append({"where": here, "what": "could not confirm that the "
                                "control's daily history (h1) holds TBOT, RAIN and SNOW: "
                                "every year of the run is checked against it"})
                else:
                    lack = [f for v in W.ANOMALY.values() for f in v["history"]
                            if f not in (h1.get("fields") or [])]
                    if lack:
                        wont.append({"where": here, "what": f"the control's daily history "
                                     f"(h1) lacks {lack}: no year of the run could be "
                                     f"checked"})
                    gap = sorted(set(years) - set(h1.get("years") or [])) \
                        if whole_years else []
                    if gap:
                        wont.append({"where": here, "what": f"the control's daily history "
                                     f"(h1) has no complete year {gap}: those years of "
                                     f"the run could not be checked"})
                if rc["params"].get("temperature_K", 0.0) < 0:
                    odd.append({"where": here, "what": "a cooler run: the recorded "
                                "moisture can exceed saturation for the cooled air, so "
                                "dew and frost form that the recorded weather would not "
                                "make (most Naches winter hours at -2 K)"})
            if region:
                regs = (control or {}).get("regions") or {}
                if region not in regs:
                    wont.append({"where": f"{where}, change {rc['id']}",
                                 "what": f"region {region!r} is not registered for the "
                                         f"control (registered: {sorted(regs) or 'none'})"})
            need = get(rc["id"]).get("needs")
            if need:
                have = _setting(cfacts, need["setting"])
                if have is None:
                    odd.append({"where": f"{where}, change {rc['id']}",
                                "what": f"could not confirm {need['setting']} = "
                                        f"{need['equals']} in the control ({need['why']})"})
                elif str(have) != need["equals"]:
                    wont.append({"where": f"{where}, change {rc['id']}",
                                 "what": f"the control's {need['setting']} is {have}: "
                                         f"{need['why']}"})
            for s in rc["checks"]["settings"]:
                have = _setting(cfacts, s["key"])
                if have is not None and _same([_as_expected(have)],
                                              [_as_expected(s["value"])]):
                    odd.append({"where": f"{where}, change {rc['id']}",
                                "what": f"the control already has {s['key']} = {have}: "
                                        f"this change changes nothing"})
            if rc["id"] == "ndep_year":
                span = _setting(cfacts, "ndep_stream_years")
                y = rc["params"]["year"]
                if span and not span[0] <= y <= span[1]:
                    wont.append({"where": f"{where}, change ndep_year",
                                 "what": f"{y} is outside the control's deposition "
                                         f"dataset ({span[0]}-{span[1]})"})
            rs.append(rc)
            facts.append({"where": f"{where}, change {rc['id']}",
                          "what": f"meaning: {rc['meaning']}. Not: {rc['not_meaning']}."})
        if any(c["id"] == "weather_scale" for c in rs) and len(rs) > 1:
            odd.append({"where": where, "what": "this run changes the weather and "
                        f"{', '.join(c['id'] for c in rs if c['id'] != 'weather_scale')}: "
                        "its difference from the control is their joint effect, and each "
                        "change's meaning is stated against the control alone"})
        if any(c["id"] == "fire_off" for c in rs):
            fire = (cfacts or {}).get("burned_area_km2") or {}
            if fire.get("model"):
                m = fire["model"]
                txt = (f"the control's own fire burns {min(m.values()):.0f}-"
                       f"{max(m.values()):.0f} km2 a year ({fire.get('pattern', '')})")
                obs = fire.get("observed")
                if obs:
                    txt += (f"; {fire.get('observed_source', 'observed')}: "
                            + ", ".join(f"{y} {v:.0f}" for y, v in sorted(obs.items()))
                            + " km2")
                facts.append({"where": f"{where}, change fire_off", "what": txt})
        resolved_runs.append({"name": name, "changes": rs, "executes": True})
    hp = (cfacts or {}).get("hours_per_model_year") or DEFAULT_HOURS_PER_YEAR
    n_years = len(years) if isinstance(years, list) else 0
    executed = len(resolved_runs) + (0 if reuse else 1)
    cost = {"runs_executed": executed, "model_years": executed * n_years,
            "hours_per_model_year": hp,
            "node_hours": round(executed * n_years * hp, 1),
            "note": ("the control is reused: it spends nothing" if reuse else
                     "the control runs too")}
    if reuse and (control or {}).get("reproduced"):
        rp = control["reproduced"]
        facts.append({"where": "control", "what": f"reused; reproduced {rp.get('how', '')} "
                      f"({'-'.join(str(y) for y in rp.get('years', []))})"
                      + (f", source fingerprint {rp['fingerprint'][:12]}"
                         if rp.get("fingerprint") else "")})
    elif reuse:
        odd.append({"where": "control", "what": "reused without a record that it "
                                                "reproduces: a run's difference from it "
                                                "may include differences in the build"})
    return {"ok": not wont, "wont_build": wont, "unusual": odd, "facts": facts,
            "runs": resolved_runs, "cost": cost}


if __name__ == "__main__":
    print(json.dumps(declare(), indent=1))
