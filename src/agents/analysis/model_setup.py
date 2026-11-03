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
        "years": {"reported": table.get("years"), "planned": chain.get("years_planned"),
                  "passed": chain.get("years_passed"),
                  "control_output": facts.get("output_years"),
                  "start_year": facts.get("start_year")},
        "runs": runs, "fields": fields,
        "runoff": runoff_limitation(exp.get("limitations") or {}, fields),
        "sources": [s for s, ok in ((CONTROL_FACTS, bool(cf)),
                                    ("experiment.json", bool(exp)),
                                    ("03_results/regional.json", bool(table))) if ok]}


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
    L = [HEADING, "",
         "*Read from this experiment's own records (" + ", ".join(rec.get("sources") or [])
         + "); the setup sentences are the model server's, read from the control's "
           "own files.*", ""]
    L.append(f"- the control: **{ctl.get('name') or '?'}**"
             + (f", {ctl['description']}" if ctl.get("description") else ""))
    L.append(f"- grid: {_sentence(rec.get('grid'))}")
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
    conf = rec.get("configuration")
    for key, label in CONFIG_ORDER:
        text = (_sentence(rec.get("fire")) if key == "fire" else _sentence(conf, key))
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
