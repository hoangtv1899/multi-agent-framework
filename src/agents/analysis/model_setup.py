#!/usr/bin/env python3
"""The model setup of a whole-domain experiment, for the top of its reports.

MODEL SETUP IN EVERY REPORT (2026-11-03). A reader of DOMAIN_REPORT.md or
REPORT.md could not tell from either how the control was set up: its grid,
its fire, its nutrients, its weather. Without that, a number is easy to read
as more than it is (the model's own fire read as a real one, a region's runoff
read as a river's flow). Both reports now open with a short block drawn from
the experiment's own records, never typed:

  01_inputs/domain_control_facts.json   the control's description and the
                                        setup sentences the model server read
                                        from the control's own files
                                        (domain_controls.setup_facts)
  experiment.json                       the runs and their changes, the
                                        years, the server's limitations
  03_results/regional.json              the years and fields reported

A RENDERER, NOT A REASONER. Every line quotes a record; the runoff line
quotes the server's own limitation (domain_analysis.LIMITATIONS, carried in
experiment.json) rather than phrasing it a second time.

  collect(run_dir, variables)   the block's record, from the run's records
  render(record)                the block's markdown lines. Pure.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

CONTROL_FACTS = "01_inputs/domain_control_facts.json"
HEADING = "## Model setup"
CONFIG_ORDER = (("compset", "compset"), ("carbon_nutrient", "carbon and nutrients"),
                ("fire", "fire"), ("co2", "CO2"), ("deposition", "deposition"),
                ("supplements", "nutrient supplements"), ("weather", "weather"))
MISSING = ("not in the control's facts (they were read before this part existed; "
           "refresh_domain_control_facts reads it)")

# WHAT OLDER FACTS DO HOLD (2026-11-03). Facts read before the fire, grid and
# configuration parts existed said MISSING for every line, although they hold
# the namelist settings (co2_type, co2_ppmv, use_nofire, suplnitro, suplphos,
# the nitrogen deposition years), the weather years, and a description that
# states the grid and the weather. Those are shown, each line saying which
# record it is quoted from; "not in the control's facts" is kept for what the
# older record truly lacks.
OLDER = "older facts"
FROM_DESCRIPTION = "the control's description"
NOT_HELD = "not in the control's facts"
# What a part of the description must name to be quoted for a line.
DESCRIPTION_WORDS = {"grid": ("degree", " box", " grid", "cells"), "weather": ("weather",),
                     "fire": ("fire",)}
# The supplement settings in words, as the server's sentence says them
# (domain_controls.SUPPLEMENT_WORDS).
SUPPLEMENT_WORDS = {"NONE": "not supplemented",
                    "ALL": "supplemented to meet plant demand everywhere",
                    "PROG_CROP_ONLY": "supplemented for prognostic crops only"}


def _read(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text()) if p.is_file() else {}
    except ValueError:
        return {}


def _span(ys) -> Optional[str]:
    ys = sorted(int(y) for y in ys or [])
    if not ys:
        return None
    if ys == list(range(ys[0], ys[-1] + 1)):
        return str(ys[0]) if ys[0] == ys[-1] else f"{ys[0]}-{ys[-1]}"
    return ", ".join(str(y) for y in ys)


def _fields(table: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for r in (table.get("runs") or {}).values():
        for v in (r.get("fields") or {}):
            if v not in out:
                out.append(v)
    return out


def reported_fields(table: Dict[str, Any], strategy: Dict[str, Any]) -> List[str]:
    """The fields the reports show: the plan's analysis variables the table
    holds, else every field it holds (domain_report.build's rule)."""
    have = _fields(table)
    an = ((strategy.get("domain_experiment") or {}).get("analysis") or {})
    return [v for v in an.get("variables") or [] if v in have] or have


def runoff_limitation(limitations: Dict[str, Any],
                      fields: List[str]) -> Optional[Dict[str, Any]]:
    """The server's limitation on runoff, when a reported field is one it
    names (the codes in its applies_to, as QRUNOFF, QOVER, QDRAI)."""
    for kind in (limitations or {}).values():
        for e in kind if isinstance(kind, list) else []:
            what = str(e.get("applies_to") or "")
            if "runoff" not in what.lower():
                continue
            named = re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", what)
            hit = [v for v in fields if v in named]
            if hit and e.get("caveat"):
                return {"fields": hit, "caveat": e["caveat"], "applies_to": what}
    return None


def collect(run_dir, variables: Optional[List[str]] = None) -> Dict[str, Any]:
    """The block's record, read from the run's records; a part the records
    lack is None, and render says so."""
    rd = Path(run_dir)
    exp = _read(rd / "experiment.json")
    cf = _read(rd / CONTROL_FACTS)
    strategy = _read(rd / "strategy.json")
    table = exp.get("regional") or _read(rd / "03_results" / "regional.json") or {}
    row = cf.get("facts") or {}
    facts = row.get("facts") or exp.get("control_facts") or {}
    ctl = (exp.get("experiment") or {}).get("control") or cf.get("control") or {}
    chain = exp.get("chain") or {}
    fields = list(variables) if variables is not None else reported_fields(table, strategy)
    runs = []
    for r in (exp.get("experiment") or {}).get("runs") or []:
        runs.append({"run": r.get("run"), "reused": bool(r.get("reused")),
                     "status": r.get("status"),
                     "changes": [{k: c.get(k) for k in ("id", "params", "summary")}
                                 for c in r.get("changes") or []]})
    return {
        "control": {"name": ctl.get("name") or row.get("name"),
                    "description": row.get("description"), "reused": ctl.get("reuse")},
        "grid": facts.get("grid"), "fire": facts.get("fire"),
        "configuration": facts.get("configuration"),
        # the settings every facts record holds, for a record older than the
        # parts above (older_lines)
        "older": {"settings": facts.get("settings") or {},
                  "forcing_years": facts.get("forcing_years")},
        "years": {"reported": table.get("years"), "planned": chain.get("years_planned"),
                  "passed": chain.get("years_passed"),
                  "control_output": facts.get("output_years"),
                  "start_year": facts.get("start_year")},
        "runs": runs, "fields": fields,
        "runoff": runoff_limitation(exp.get("limitations") or {}, fields),
        "sources": [s for s, ok in ((CONTROL_FACTS, bool(cf)),
                                    ("experiment.json", bool(exp)),
                                    ("03_results/regional.json", bool(table))) if ok]}


def _absent(part: Any) -> bool:
    """A part the facts were read without: absent, or the server's note that
    these facts predate it (domain_controls._setup_summary)."""
    return not isinstance(part, dict) or str(part.get("error") or "").startswith(
        "not in these facts")


def _clauses(text: Optional[str]) -> List[str]:
    """The description in its parts: split at ", " outside parentheses, the
    control's name before the first ": " left out."""
    out, cur, depth = [], "", 0
    for k, ch in enumerate(str(text or "")):
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if ch == "," and depth == 0 and text[k + 1:k + 2] == " ":
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    out.append(cur.strip())
    if out and ": " in out[0]:
        out[0] = out[0].split(": ", 1)[1]
    return [c for c in out if c]


def _described(rec: Dict[str, Any], what: str) -> Optional[str]:
    """The parts of the control's description that name `what` (a key of
    DESCRIPTION_WORDS), quoted, or None."""
    words = DESCRIPTION_WORDS[what]
    hit = [c for c in _clauses((rec.get("control") or {}).get("description"))
           if any(w in c.lower() for w in words)]
    return "; ".join(hit) or None


def _num(v) -> str:
    """A setting as the namelist reads: 367 not 367.0, true or false."""
    if isinstance(v, bool):
        return "true" if v else "false"
    try:
        x = float(v)
        return f"{x:g}"
    except (TypeError, ValueError):
        return str(v)


def older_lines(rec: Dict[str, Any]) -> Dict[str, str]:
    """For each line the newer parts would give, what the older record holds:
    {"grid", "fire", "co2", "deposition", "supplements", "weather"}, each
    saying in brackets which record it is quoted from; a line the older record
    does not hold is left out (render says NOT_HELD)."""
    st = ((rec.get("older") or {}).get("settings")) or {}
    fy = (rec.get("older") or {}).get("forcing_years")
    out: Dict[str, str] = {}
    d = _described(rec, "grid")
    if d:
        out["grid"] = f"{d} ({FROM_DESCRIPTION})"
    parts = []
    if st.get("use_nofire") is not None:
        parts.append(f"the switch that turns the fire model off, use_nofire, is "
                     f"{_num(st['use_nofire'])} ({OLDER})")
    d = _described(rec, "fire")
    if d:
        parts.append(f"{d} ({FROM_DESCRIPTION})")
    if parts:
        out["fire"] = "; ".join(parts)
    t, ppm = st.get("co2_type") or st.get("ELM_CO2_TYPE"), \
        st.get("co2_ppmv") if st.get("co2_ppmv") is not None else st.get("CCSM_CO2_PPMV")
    # ONLY WHAT THE RECORD HOLDS (2026-11-03): _settings writes a key the
    # namelist lacks as None, so an older record may hold the CO2 level without
    # its type, or one deposition year without the other; the missing one is
    # marked as not held, never quoted as "None".
    if t and ppm is not None:
        out["co2"] = ((f"held constant at {_num(ppm)} ppmv" if t == "constant" else
                       f"of type {t} (co2_ppmv {_num(ppm)})")
                      + f" (co2_type {t}, co2_ppmv {_num(ppm)}; {OLDER})")
    elif ppm is not None:
        out["co2"] = f"co2_ppmv {_num(ppm)} ({OLDER}); co2_type: {NOT_HELD}"
    elif t:
        out["co2"] = f"co2_type {t} ({OLDER}); co2_ppmv: {NOT_HELD}"
    a, b = st.get("stream_year_first_ndep"), st.get("stream_year_last_ndep")
    if a is not None or b is not None:
        span = st.get("ndep_stream_years")
        named = (f"; its file is named for {span[0]}-{span[1]}"
                 if isinstance(span, list) and len(span) == 2 else "")
        if a is not None and b is not None:
            said = (("nitrogen deposition held at year " + _num(a) if a == b else
                     f"nitrogen deposition read for {_num(a)}-{_num(b)}")
                    + f" (stream_year_first_ndep {_num(a)}, stream_year_last_ndep {_num(b)}"
                    + f"{named}; {OLDER})")
        else:
            held, k, other = ((a, "first", "last") if a is not None else (b, "last", "first"))
            said = (f"nitrogen deposition's {k} year is {_num(held)} "
                    f"(stream_year_{k}_ndep {_num(held)}{named}; {OLDER}); its {other} year "
                    f"(stream_year_{other}_ndep): {NOT_HELD}")
        out["deposition"] = f"{said}; phosphorus deposition: {NOT_HELD}"
    sup = [(k, st.get(f"supl{tag}")) for k, tag in (("nitrogen", "nitro"),
                                                     ("phosphorus", "phos"))]
    if any(v is not None for _, v in sup):
        out["supplements"] = "; ".join(
            f"{k} {SUPPLEMENT_WORDS.get(str(v).upper(), f'supplement setting {v}')} "
            f"(supl{'nitro' if k == 'nitrogen' else 'phos'} {v})"
            for k, v in sup if v is not None) + f" ({OLDER})"
    parts = []
    d = _described(rec, "weather")
    if d:
        parts.append(f"{d} ({FROM_DESCRIPTION})")
    if isinstance(fy, list) and len(fy) == 2:
        parts.append(f"weather for the model years {fy[0]}-{fy[1]} ({OLDER})")
    if st.get("DATM_MODE"):
        parts.append(f"DATM mode {st['DATM_MODE']} ({OLDER})")
    if parts:
        out["weather"] = "; ".join(parts)
    return out


def _sentence(part: Any, key: Optional[str] = None) -> str:
    """A part's sentence (or one of its sentences), or why there is none."""
    if not isinstance(part, dict):
        return MISSING
    if part.get("error"):
        return f"not read: {part['error']}"
    v = part.get("sentence") if key is None else (part.get("sentences") or {}).get(key)
    return str(v) if v else MISSING


def _changes(r: Dict[str, Any]) -> str:
    from agents.analysis.domain_report import _param_text
    out = []
    for c in r.get("changes") or []:
        p = c.get("params") or {}
        out.append(str(c.get("id") or "?")
                   + (" (" + ", ".join(_param_text(k, v) for k, v in p.items()) + ")"
                      if p else "")
                   + (f": {c['summary']}" if c.get("summary") else ""))
    return "; ".join(out) or "no change"


def render(rec: Optional[Dict[str, Any]]) -> List[str]:
    """The block's markdown lines, heading first, a blank line last. Pure.
    Empty for a record that is not a whole-domain experiment's."""
    if not rec or not isinstance(rec, dict) or rec.get("error"):
        return ([HEADING, "", f"- not drawn: {rec['error']}", ""]
                if isinstance(rec, dict) and rec.get("error") else [])
    ctl = rec.get("control") or {}
    conf = rec.get("configuration")
    old = (_absent(rec.get("grid")) or _absent(rec.get("fire")) or _absent(conf))
    held = older_lines(rec) if old else {}
    L = [HEADING, "",
         "*Read from this experiment's own records (" + ", ".join(rec.get("sources") or [])
         + "); the setup sentences are the model server's, read from the control's "
           "own files.*", ""]
    if old:
        L += [f"*The control's facts were read before the server read its grid, fire and "
              f"configuration (refresh_domain_control_facts reads them). A line marked "
              f"({OLDER}) quotes the settings that older record holds, one marked "
              f"({FROM_DESCRIPTION}) quotes the description above, and \"{NOT_HELD}\" marks "
              f"what neither holds.*", ""]

    def _line(part: Any, key: str, conf_key: Optional[str] = None) -> str:
        if old and _absent(part):
            return held.get(key) or NOT_HELD
        return _sentence(part, conf_key)

    L.append(f"- the control: **{ctl.get('name') or '?'}**"
             + (f", {ctl['description']}" if ctl.get("description") else ""))
    L.append(f"- grid: {_line(rec.get('grid'), 'grid')}")
    y = rec.get("years") or {}
    parts = []
    if _span(y.get("reported")):
        parts.append(f"reported {_span(y['reported'])}")
    if _span(y.get("passed")) and y.get("passed") != y.get("reported"):
        parts.append(f"run and checked {_span(y['passed'])}")
    co = y.get("control_output") or []
    if co:
        parts.append(f"the control's output covers {co[0]}-{co[-1]}"
                     + (f", from the saved state of {y['start_year']}"
                        if y.get("start_year") else ""))
    L.append("- years: " + ("; ".join(parts) or "not recorded"))
    L.append("- configuration (the control's; each run differs from it only by its "
             "changes, listed under runs):")
    for key, label in CONFIG_ORDER:
        text = (_line(rec.get("fire"), "fire") if key == "fire" else
                _line(conf, key, key))
        L.append(f"  - {label}: {text}")
    runs = rec.get("runs") or []
    if runs:
        L.append("- runs:")
    for r in runs:
        if r.get("reused") and not r.get("changes"):
            L.append(f"  - {r.get('run')}: the control, reused: its existing output was read")
        else:
            L.append(f"  - {r.get('run')}: the control with {_changes(r)}")
    ro = rec.get("runoff")
    if ro:
        L.append(f"- runoff ({', '.join(ro['fields'])}): {ro['caveat']}")
    L.append("")
    return L


def insert_after_answer(lines: List[str], block: List[str]) -> List[str]:
    """lines with block placed before the first heading after "## The answer",
    so the block sits near the top, right under the answer."""
    if not block:
        return lines
    try:
        k = lines.index("## The answer")
    except ValueError:
        return lines + [""] + block
    for j in range(k + 1, len(lines)):
        if lines[j].startswith("## "):
            return lines[:j] + block + lines[j:]
    return lines + [""] + block
