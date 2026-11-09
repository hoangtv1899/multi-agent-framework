#!/usr/bin/env python3
"""The data catalog of a whole-domain study: what can be analysed and how,
never what was found.

THE DATA CATALOG (2026-11-04). The Analyzer reads one file before it plans,
CATALOG.md: the study's runs and their changes, the regions, the periods, and
every field it can analyse, each with what it means, how a period's value is
made, its variable type and scale, its units, its products and the cautions
people wrote about it. Everything in it is generated at build time from the
code and the files, so it cannot claim a product or a method that does not
exist (the trap of prompts that encode capabilities); what people write
(meanings, groups, types, scales, cautions) comes from domain_field_notes.json,
merged in here and checked against the files.

  build(control, study_dir=None)  the catalog as a dict (catalog.json)
  render(catalog)                 CATALOG.md, at most MAX_CHARS characters
  write(catalog, out_dir)         both files
  load_notes() / check_notes()    the notes file, and what is wrong with it
  study_record(study_dir)         a study's runs and normal years, from its records

THE CATALOG SAYS ONLY WHAT EXISTS. Coverage facts (years, records, cells,
areas, units) are in; outcome values (anomalies, differences, ranks,
extremes) are out, and nothing here reads a field's values: a stream is
opened for its time stamps and a file for its header. A part not built yet
(the suite before it is built, the access calls) says so and names nothing.
The methods are listed by id, pointing to docs/ANALYSIS_METHODS.md, which
holds their contracts (review round 1, 2026-11-05).

REVIEW ROUND 1 (2026-11-05). The field notes carry a percent policy apart
from the physical scale, and every note, method and mechanism four statuses
kept apart (review, implemented, tested, available to the Analyzer: see
status_problems); the catalog shows them apart. The control's verified
configuration (nutrient cycles and supplements, fire) is read from its facts
cache, since how FPG reads depends on it. The same catalog serves every study of a control; a study adds
its run rows and its normal years, named as a definition. No hint of what a
field matters for, and nothing tuned to a question.

    python3 domain_catalog.py --control naches --out DIR [--study STUDY_DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import domain_analysis as DA
import gridded as G

NOTES_PATH = Path(__file__).resolve().parents[1] / "domain_field_notes.json"
CATALOG_VERSION = 1
# TWO PARTS, EACH WITH ITS OWN LIMIT (package F1, 2026-11-05). The Analyzer
# reads CATALOG.md whole, so it is never clipped: a part over its limit
# raises. The FIXED part (the control's row and configuration, regions,
# periods, fields, methods, how to load, other output) is the same for every
# study of a control and was 24,655 of these 25,000 characters for Naches. The
# STUDY part is the study's experiment runs. A run row used to quote every
# change whole (about 1,150 characters a run), so one experiment run could
# stop step 2a. Now a run is one compact line (its name, each change's id and
# settings with the server's sentence of what the difference means, its
# years and suite), every "does not mean" warning is kept whole in the
# server's words, printed once for the runs it applies to, and the run's full
# record (the server's summary too) is read on demand with
# suite.run_record(run), which reads no outcome and is logged apart from the
# lookups. Why this and not a longer limit alone: the warnings are the part
# that binds, so they stay in what is read whole; the rest grows with the
# runs and is a call away. The eight change kinds' warnings total about
# 1,900 characters, so eight runs fit in the study part with room to spare.
MAX_CHARS = 25000          # the fixed part
STUDY_MAX_CHARS = 10000    # the study part: up to 8 experiment runs fit
RUN_RECORD_CALL = "suite.run_record(run)"
CATALOG_JSON = "catalog.json"
CATALOG_MD = "CATALOG.md"
SUITE_FILE = "suite.json"

# The model's own order of its processes, from the weather that drives it to
# the fire that ends a stand; rows are alphabetical within a group.
GROUPS = ("weather in", "snow", "soil water and groundwater", "runoff",
          "water to the air", "energy", "carbon in and out", "leaves",
          "carbon stores", "nitrogen", "fire")
NO_NOTE_GROUP = "fields without a note"
TYPES = ("flux", "state", "event", "store", "bounded")
TYPE_WORDS = {"bounded": "bounded index"}
SCALES = ("ratio", "interval")
HOWS = ("total", "average")
CADENCES = ("daily", "monthly")
# THE USER'S APPROVAL (2026-11-05). The review states, in order: a draft; an
# approval for integration (the user's of 2026-11-05: the entry may be used by
# the framework, which is not a final scientific approval); provisional (a value
# approved for integration only as provisional, pending a further review); and
# reviewed (a final review, which no entry has yet). The --review-gate accepts
# the last three at the entry's current versions (is_approved) and fails on a
# draft or on a review that names an older version.
DRAFT_STATE = "draft"
APPROVED_STATE = "approved for integration"
PROVISIONAL_STATE = "provisional"
REVIEWED_STATE = "reviewed"
STATUSES = (DRAFT_STATE, APPROVED_STATE, PROVISIONAL_STATE, REVIEWED_STATE)
GATE_STATES = (APPROVED_STATE, PROVISIONAL_STATE, REVIEWED_STATE)
NOT_FINAL = "an approval for integration, not a final scientific approval"
STATUS_PARTS = ("review", "implemented", "tested", "available_to_analyzer")
TEST_ID = re.compile(r"tests/[A-Za-z0-9_]+\.py::[A-Za-z0-9_]+")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

METHODS_DOC = "docs/ANALYSIS_METHODS.md"
FACTS_DIR = "domain_control_facts"     # domain_controls.FACTS_DIR, beside the registry
NOT_BUILT = "not built yet"
SUITE_NOT_BUILT = "suite not built"


class CatalogError(Exception):
    """The catalog cannot be made, and the message says why."""


# ─────────────────────────────────────────────────────────────────────────────
# THE FIELD NOTES
# ─────────────────────────────────────────────────────────────────────────────

def load_notes(path=None) -> Dict[str, Dict[str, Any]]:
    """{field: entry} from the notes file (default NOTES_PATH)."""
    p = Path(path) if path else NOTES_PATH
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise CatalogError(f"the field notes {p} cannot be read: {e}")
    fields = d.get("fields") if isinstance(d, dict) else None
    if not isinstance(fields, dict) or not fields:
        raise CatalogError(f"the field notes {p} hold no fields")
    return fields


# STATUSES (review round 1, 2026-11-05). Every field note, method contract and
# mechanism carries four statuses, kept apart: review (draft or reviewed, with
# the text revision and behaviour version the review refers to), implemented
# (the behaviour version the code applies), tested (the tests that passed,
# their date and the behaviour version they ran on) and available_to_analyzer
# (false until the Analyzer's access calls are built). A review or a test
# record that refers to an older version is a problem, and an entry counts as
# reviewed only at its current text revision and behaviour version.
def status_problems(st: Any, text_revision: Any, behaviour_version: Any,
                    where: str = "") -> List[str]:
    """What is wrong with an entry's four statuses, given its current text
    revision and behaviour version."""
    pre = f"{where}: " if where else ""
    out: List[str] = []
    for k, v in (("text revision", text_revision), ("behaviour version", behaviour_version)):
        if not (isinstance(v, int) and not isinstance(v, bool) and v >= 1):
            out.append(f"{pre}its {k} must be a whole number from 1, not {v!r}")
    if not isinstance(st, dict) or tuple(st) != STATUS_PARTS:
        return out + [f"{pre}status must hold {', '.join(STATUS_PARTS)}, in that order"]
    rv = st["review"] if isinstance(st["review"], dict) else {}
    if rv.get("state") not in STATUSES:
        out.append(f"{pre}review state is {rv.get('state')!r}, not one of {STATUSES}")
    elif rv["state"] != DRAFT_STATE and not (str(rv.get("by") or "").strip() and
                                              DATE.fullmatch(str(rv.get("date") or ""))):
        out.append(f"{pre}a review that is not a draft names who made it (by) and when "
                   f"(date, YYYY-MM-DD)")
    if (rv.get("text_revision"), rv.get("behaviour_version")) != (text_revision,
                                                                   behaviour_version):
        out.append(f"{pre}the review refers to text revision {rv.get('text_revision')} and "
                   f"behaviour version {rv.get('behaviour_version')}; the entry is at "
                   f"{text_revision} and {behaviour_version}")
    im = st["implemented"] if isinstance(st["implemented"], dict) else {}
    if im.get("behaviour_version") != behaviour_version:
        out.append(f"{pre}implemented names behaviour version {im.get('behaviour_version')}, "
                   f"not {behaviour_version}")
    te = st["tested"]
    if te is not None:
        if not isinstance(te, dict) or te.get("behaviour_version") != behaviour_version:
            out.append(f"{pre}the tests ran on another behaviour version than "
                       f"{behaviour_version}: test it again")
        elif not DATE.fullmatch(str(te.get("date") or "")):
            out.append(f"{pre}tested has no date (YYYY-MM-DD)")
        elif not te.get("passed") or not all(TEST_ID.fullmatch(str(x)) for x in te["passed"]):
            out.append(f"{pre}tested must list the tests that passed, as path::name")
    if not isinstance(st["available_to_analyzer"], bool):
        out.append(f"{pre}available_to_analyzer must be true or false")
    return out


def _review_at(rv: Any, states: Tuple[str, ...], text_revision: Any,
               behaviour_version: Any) -> bool:
    return isinstance(rv, dict) and rv.get("state") in states and \
        (rv.get("text_revision"), rv.get("behaviour_version")) == (text_revision,
                                                                   behaviour_version)


def is_reviewed(st: Any, text_revision: Any, behaviour_version: Any) -> bool:
    """A final review by the user at the entry's current text revision and
    behaviour version: a review of an older one does not count."""
    rv = (st or {}).get("review") if isinstance(st, dict) else None
    return _review_at(rv, (REVIEWED_STATE,), text_revision, behaviour_version)


def review_passes(rv: Any, text_revision: Any, behaviour_version: Any) -> bool:
    """What the --review-gate accepts of one review record: approved for
    integration, provisional or reviewed, at the current text revision and
    behaviour version. A draft, or a review naming an older version, fails."""
    return _review_at(rv, GATE_STATES, text_revision, behaviour_version)


def is_approved(st: Any, text_revision: Any, behaviour_version: Any) -> bool:
    """review_passes on an entry's four statuses."""
    rv = (st or {}).get("review") if isinstance(st, dict) else None
    return review_passes(rv, text_revision, behaviour_version)


def review_words(rv: Dict[str, Any]) -> str:
    """One review record in plain words, saying when it is not a final
    scientific approval."""
    rv = rv or {}
    s = f"{rv.get('state')}"
    if rv.get("state") != DRAFT_STATE and rv.get("by"):
        s += f" by {rv.get('by')} on {rv.get('date')}"
    s += (f" (text revision {rv.get('text_revision')}, behaviour version "
          f"{rv.get('behaviour_version')})")
    if rv.get("why"):
        s += f": {rv['why']}"
    if rv.get("note"):
        s += f"; {rv['note']}"
    if rv.get("state") in (APPROVED_STATE, PROVISIONAL_STATE):
        s += f"; {NOT_FINAL}"
    return s


def status_lines(st: Dict[str, Any], names_only: bool = True) -> List[str]:
    """The four statuses as four lines of plain words, kept apart."""
    rv, im, te = st.get("review") or {}, st.get("implemented") or {}, st.get("tested")
    out = ["review: " + review_words(rv),
           f"implemented: behaviour version {im.get('behaviour_version')}"]
    if te:
        names = [x.split("::")[-1] if names_only else x for x in te.get("passed") or []]
        out.append(f"tested: behaviour version {te.get('behaviour_version')}, on "
                   f"{te.get('date')}, {len(names)} passed: " + ", ".join(names))
    else:
        out.append("tested: not yet")
    out.append("available to the Analyzer: " + ("yes (through the suite's access calls, "
                                                "domain_access)" if st.get("available_to_analyzer")
                                                else "no (the access calls do not reach it)"))
    return out


def note_status_problems(name: str, e: Dict[str, Any]) -> List[str]:
    return status_problems(e.get("status"), e.get("text_revision"), e.get("behaviour_version"),
                           name)


def is_row(entry: Dict[str, Any]) -> bool:
    """A note makes a catalog row unless it says row false (FIRE's caution)."""
    return entry.get("row", True) is not False


def check_notes(notes: Dict[str, Dict[str, Any]]) -> List[str]:
    """What is wrong with the notes' own content, entry by entry. Whether each
    names a field the run has is checked by build(), against the files."""
    out: List[str] = []
    for name, e in notes.items():
        if not isinstance(e, dict):
            out.append(f"{name}: the entry is not an object")
            continue
        out += note_status_problems(name, e)
        if e.get("cadence") not in CADENCES:
            out.append(f"{name}: cadence is {e.get('cadence')!r}, not one of {CADENCES}")
        cautions = e.get("cautions", [])
        if not isinstance(cautions, list) or not all(
                isinstance(c, str) and c.strip() for c in cautions):
            out.append(f"{name}: cautions must be a list of sentences")
        if not is_row(e):
            continue
        for key in ("meaning", "scale_why"):
            if not str(e.get(key) or "").strip():
                out.append(f"{name}: no {key}")
        for key, allowed in (("group", GROUPS), ("how", HOWS), ("type", TYPES),
                             ("scale", SCALES)):
            if e.get(key) not in allowed:
                out.append(f"{name}: {key} is {e.get(key)!r}, not one of {allowed}")
        pc = e.get("percent")
        if not (isinstance(pc, dict) and isinstance(pc.get("allowed"), bool)
                and str(pc.get("why") or "").strip()):
            out.append(f"{name}: percent must be {{allowed: true or false, why: a sentence}}")
        elif pc["allowed"] and e.get("type") in ("event", "bounded"):
            out.append(f"{name}: an {e.get('type')} field never has a percent change")
    return out


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# THE STREAMS: coverage from the time stamps, names from one header
# ─────────────────────────────────────────────────────────────────────────────

def _header(path: Path) -> Dict[str, Dict[str, Any]]:
    """Each variable's long name, dimensions and sizes, from one file's header.
    The stream's index keeps units and dimensions but not long names."""
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        return {n: {"long_name": str(getattr(v, "long_name", "") or ""),
                    "dims": [str(d) for d in v.dimensions],
                    "shape": [int(s) for s in v.shape]}
                for n, v in ds.variables.items()}


def years_text(years: Sequence[int]) -> str:
    """2015-2018, 2020: runs of consecutive years as ranges."""
    ys = sorted({int(y) for y in years})
    if not ys:
        return "none"
    parts, start, prev = [], ys[0], ys[0]
    for y in ys[1:] + [None]:
        if y is not None and y == prev + 1:
            prev = y
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        if y is not None:
            start = prev = y
    return ", ".join(parts)


def stream_facts(stream: G.Stream) -> Dict[str, Any]:
    """What one stream holds: cadence, records, whole and partial years, and
    the records left out with gridded's reason. Counts and labels only."""
    complete = stream.complete_years()
    partial = {}
    for y in stream.years():
        if y in complete:
            continue
        have, missing = stream.year_records(y)
        partial[str(y)] = {"records": len(have), "missing": len(missing),
                           "first_missing": G.label_text(missing[0], stream.cadence)}
    left_out = Counter(str(d.get("reason") or "no reason given") for d in stream.dropped)
    return {"name": stream.name, "cadence": stream.cadence,
            "steps_per_day": stream.steps_per_day, "calendar": stream.calendar,
            "files": len(stream.files), "first_file": stream.files[0].name,
            "last_file": stream.files[-1].name, "records": len(stream.records),
            "first_label": G.label_text(stream.records[0].key, stream.cadence),
            "last_label": G.label_text(stream.records[-1].key, stream.cadence),
            "complete_years": [int(y) for y in complete], "partial_years": partial,
            "left_out": [{"reason": r, "records": n} for r, n in sorted(left_out.items())],
            "extended": len(stream.extended), "fields": len(stream.variables)}


def _first_with(streams: Dict[str, G.Stream], cadence: str) -> Optional[str]:
    return next((n for n in sorted(streams) if streams[n].cadence == cadence), None)


def unit_rule(name: str, units: str) -> str:
    """How a period's value is made from the units alone: a rate stored per
    second is totalled, anything else averaged over time (domain_analysis)."""
    return "total" if G.is_per_second(name, units) else "average"


def amount_units(name: str, units: str) -> str:
    """A rate's units once totalled over a period: gC/m^2/s gives gC/m^2."""
    u = G.per_day_units(name, units)
    u = re.sub(r"\s*/\s*d$", "", u)
    return re.sub(r"\s+d\s*(?:\^|\*\*)?\s*-\s*1$", "", u)


# ─────────────────────────────────────────────────────────────────────────────
# THE SUITE (built later, by its own builder)
# ─────────────────────────────────────────────────────────────────────────────

# WHAT THE CATALOG READS OF A SUITE MANIFEST (2026-11-04). The suite builder
# comes after the catalog, so the keys read here are the ones it is to write:
# the version under SUITE_VERSION; "fields" as {field: {"products": [names]}};
# "fields_left_out" as {field: reason}; "periods" as {kind: [labels]}. A
# manifest without them lists no product, and the catalog says so per field.
SUITE_VERSION_KEYS = ("SUITE_VERSION", "suite_version")


def load_suite(suite_dir) -> Optional[Dict[str, Any]]:
    """The suite's manifest, or None when no suite was built there."""
    if not suite_dir:
        return None
    p = Path(str(suite_dir)) / SUITE_FILE
    if not p.is_file():
        return None
    try:
        m = json.loads(p.read_text())
    except ValueError as e:
        raise CatalogError(f"the suite manifest {p} cannot be read: {e}")
    return dict(m, _path=str(p)) if isinstance(m, dict) else None


def suite_version(manifest: Optional[Dict[str, Any]]) -> str:
    if manifest is None:
        return SUITE_NOT_BUILT
    v = next((manifest[k] for k in SUITE_VERSION_KEYS if manifest.get(k) is not None), None)
    return f"suite version {v}" if v is not None else "a suite whose manifest names no version"


def suite_products(manifest: Optional[Dict[str, Any]], field: str) -> Any:
    """The products the suite holds for a field, as its manifest lists them."""
    if manifest is None:
        return SUITE_NOT_BUILT
    why = (manifest.get("fields_left_out") or {}).get(field)
    if why:
        return f"left out of the suite: {why}"
    e = (manifest.get("fields") or {}).get(field)
    prods = e.get("products") if isinstance(e, dict) else None
    if isinstance(prods, list) and prods:
        return [str(p) for p in prods]
    return "the suite's manifest lists no product for it"


# ─────────────────────────────────────────────────────────────────────────────
# THE STUDY
# ─────────────────────────────────────────────────────────────────────────────

CHANGE_KEYS = ("id", "params", "summary", "meaning", "not_meaning")


def study_record(study_dir) -> Dict[str, Any]:
    """A study's runs (each with its change list as the server wrote it) and
    its normal years, from its experiment record: experiment.json, written
    when the runs are packaged, and the plan's 01_inputs/domain_experiment.json,
    whose analysis names the normal years. Before experiment.json exists the
    runs come from the plan, with whatever of the change list it holds."""
    d = Path(str(study_dir))
    out: Dict[str, Any] = {"dir": str(d), "name": d.name, "runs": [], "normal_years": [],
                           "years": [], "years_passed": None, "problems": []}
    exp, plan = d / "experiment.json", d / "01_inputs" / "domain_experiment.json"
    e = json.loads(exp.read_text()) if exp.is_file() else {}
    p = json.loads(plan.read_text()) if plan.is_file() else {}
    if not e and not p:
        out["problems"].append(f"{d} holds neither experiment.json nor "
                               f"01_inputs/domain_experiment.json")
        return out
    analysis = p.get("analysis") or {}
    out["normal_years"] = sorted(int(y) for y in analysis.get("normal_years") or [])
    chain = e.get("chain") or {}
    out["years"] = sorted(int(y) for y in (chain.get("years_planned") or p.get("years") or []))
    if "years_passed" in chain:
        out["years_passed"] = sorted(int(y) for y in chain.get("years_passed") or [])
    runs = (e.get("experiment") or {}).get("runs")
    if runs is None:
        runs = [dict(r, run=r.get("name")) for r in p.get("runs") or []]
    for r in runs:
        name = str(r.get("run") or r.get("name") or "")
        if not name or name == "control":
            continue                 # the control's row is made from the registry
        changes = [{k: c[k] for k in CHANGE_KEYS if c.get(k) not in (None, "", {}, [])}
                   for c in (r.get("changes") or []) if isinstance(c, dict)]
        out["runs"].append({"run": name, "changes": changes})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# THE CATALOG
# ─────────────────────────────────────────────────────────────────────────────

def _regions(control: Dict[str, Any], grid: G.Grid,
             manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The regions every product is made over: the vegetated cells, each
    registered region and the elevation bands, as domain_analysis.masks makes
    them on the control's grid.

    THE SUITE'S REGIONS (2026-11-04). Once the control's suite is built, its
    products are made over the regions its manifest lists, the elevation bands
    within each registered region among them, so those are the ones listed,
    with the cells, area and rule the manifest records."""
    registered = control.get("regions") or {}
    surface = DA._surface(control["case_dir"])
    what = {f"{n}, vegetated": str(r.get("what") or "") for n, r in registered.items()}
    listed = (manifest or {}).get("regions")
    if isinstance(listed, list) and listed:
        rows = [{"name": str(r.get("name")), "cells": int(r.get("cells") or 0),
                 "area_km2": float(r.get("area_km2") or 0.0), "rule": str(r.get("rule") or ""),
                 "what": what.get(str(r.get("name")), "")} for r in listed]
        return {"surface_file": str(surface), "regions": rows, "problems": [],
                "from": "the suite's manifest"}
    names = [DA.VEGETATED] + sorted(registered) + [DA.BANDS]
    mk = DA.masks(grid, surface, names, registered, control.get(DA.BAND_EDGES_KEY))
    rows = [{"name": n, "cells": int(m["n_cells"]), "area_km2": float(m["area_km2"]),
             "rule": m["rule"], "what": what.get(n, "")}
            for n, m in mk["masks"].summary().items()]
    return {"surface_file": str(surface), "regions": rows, "problems": list(mk["problems"])}


def _field_row(name: str, note: Optional[Dict[str, Any]], stream: G.Stream,
               header: Dict[str, Dict[str, Any]], manifest, problems: List[str]
               ) -> Dict[str, Any]:
    units = stream.units(name)
    rule = unit_rule(name, units)
    h = header.get(name) or {}
    levels = [{"dim": d, "size": s} for d, s in zip(h.get("dims", [])[1:], h.get("shape", [])[1:])
              if d in stream.levels(name)]
    row: Dict[str, Any] = {"name": name, "stream": stream.name, "cadence": stream.cadence,
                           "long_name": h.get("long_name", ""), "units": units,
                           "unit_rule": rule, "levels": levels,
                           "products": suite_products(manifest, name)}
    if note is None:
        problems.append(f"{name}: the {stream.cadence} stream {stream.name} holds it and the "
                        f"field notes have no entry for it")
        row.update(group=NO_NOTE_GROUP, meaning="no note yet", how=rule, type=None,
                   type_why=None, scale=None, scale_why=None, percent=None, cautions=[],
                   text_revision=None, behaviour_version=None, status=None)
    else:
        how = note.get("how")
        if how != rule and not note.get("how_why"):
            problems.append(f"{name}: the notes say {how}, its units ({units}) say {rule}, "
                            f"and the notes give no how_why; the units' rule is used")
            how = rule
        row.update({k: note.get(k) for k in ("group", "meaning", "type", "type_why", "scale",
                                              "scale_why", "percent", "text_revision",
                                              "behaviour_version", "status")},
                   how=how, how_why=note.get("how_why"),
                   cautions=list(note.get("cautions") or []))
        if note.get("cadence") != stream.cadence:
            problems.append(f"{name}: the notes say {note.get('cadence')}, the run saves it "
                            f"in the {stream.cadence} stream {stream.name}")
    if how_total(row):
        row["amount_units"] = amount_units(name, units)
    lack = stream.lacking([name]).get(name)
    if lack:
        problems.append(f"{name}: {len(lack)} file(s) of {stream.name} do not hold it, "
                        f"from {lack[0]}")
    return row


def how_total(row: Dict[str, Any]) -> bool:
    return row.get("how") == "total"


def _order(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rank = {g: i for i, g in enumerate(GROUPS + (NO_NOTE_GROUP,))}
    return sorted(rows, key=lambda r: (rank.get(r["group"], len(rank)), r["name"]))


def control_configuration(control: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The control's verified configuration, from its facts cache (the
    registry's domain_control_facts/<name>.json, keys configuration and fire),
    or None when the cache is not there. The nutrient reading of FPG in the
    field notes holds for the configuration recorded here."""
    reg, name = control.get("registry"), control.get("name")
    if not reg or not name:
        return None
    p = Path(str(reg)).parent / FACTS_DIR / f"{name}.json"
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    c = d.get("configuration") or {}
    cn = c.get("carbon_nutrient") or {}
    sup = c.get("supplements") or {}
    fire = d.get("fire") or {}
    return {"facts_file": str(p), "compset": c.get("compset"),
            "nutrient": (cn.get("build_flags") or {}).get("nutrient"),
            "nu_com": cn.get("nu_com"),
            "nitrogen_supplement": sup.get("nitrogen"),
            "phosphorus_supplement": sup.get("phosphorus"),
            "fire_model_on": fire.get("on"), "use_nofire": fire.get("use_nofire"),
            "prescribed_burned_area": fire.get("prescribed_burned_area")}


# HOW TO LOAD, FROM THE ACCESS CODE (package E, 2026-11-05). Section 6 lists
# the calls domain_access.SuiteAccess has, read off the class, so it cannot
# name a call that does not exist; until the class existed it said "not built
# yet".
def access_calls() -> Any:
    """The calls a script makes on `suite`, or NOT_BUILT."""
    try:
        import domain_access
    except ImportError:
        return NOT_BUILT
    return domain_access.how_to_load()


def _access_lines(access: Any) -> List[str]:
    if not isinstance(access, list) or not access:
        return [f"The calls that load values are {NOT_BUILT}."]
    L = ["A script reaches the suite only through `suite`: no call before the analysis "
         "plan has passed step 3's review, every call logged, and no method setting beyond "
         "the default, the contract's alternatives and those the plan names.", ""]
    L += [f"- `{a['call']}`: {_cell(a['does'])}" for a in access]
    return L


def methods_listed() -> Dict[str, Any]:
    """The methods that exist, by id, with their versions and statuses: the
    contracts themselves are in METHODS_DOC."""
    import domain_methods as M        # here, since domain_methods imports this module
    return {"doc": METHODS_DOC,
            "methods": [{"id": m.id, "version": m.version, "text_revision": m.text_revision,
                         "status": m.status} for m in M.METHODS.values()],
            "provisional_settings": [{"method": m.id, "setting": s.name,
                                      "why": s.review.get("why")}
                                     for m in M.METHODS.values() for s in m.settings
                                     if s.status == PROVISIONAL_STATE]}


def notes_status(notes: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The field notes' four statuses, counted apart."""
    cur = lambda e, part: (e.get("status") or {}).get(part) or {}             # noqa: E731
    tested = [e for e in notes.values()
              if cur(e, "tested").get("behaviour_version") == e.get("behaviour_version")]
    passing = [e for e in notes.values() if is_approved(
        e.get("status"), e.get("text_revision"), e.get("behaviour_version"))]
    return {"reviewed": sum(1 for e in notes.values() if is_reviewed(
                e.get("status"), e.get("text_revision"), e.get("behaviour_version"))),
            "approved": len(passing),
            "review_states": {k: sum(1 for e in passing if e["status"]["review"]["state"] == k)
                              for k in GATE_STATES},
            "approved_by": sorted({f"{e['status']['review'].get('by')} on "
                                   f"{e['status']['review'].get('date')}" for e in passing}),
            "implemented": sum(1 for e in notes.values() if cur(e, "implemented").get(
                "behaviour_version") == e.get("behaviour_version")),
            "tested": len(tested),
            "tested_on": sorted({str(cur(e, "tested").get("date")) for e in tested}),
            "available_to_analyzer": sum(1 for e in notes.values()
                                         if (e.get("status") or {}).get("available_to_analyzer")
                                         is True)}


def build(control: Dict[str, Any], study_dir=None, notes_path=None,
          suite_dir=None) -> Dict[str, Any]:
    """The catalog of a control, with a study's runs when study_dir is given.

    control    a registry entry (domain_controls.get): name, case_dir, run_dir,
               regions, and suite_dir once the control's suite is built
    study_dir  a study's directory (its experiment record), or None
    suite_dir  the control's suite, when not under the entry's suite_dir
    """
    notes_file = Path(notes_path) if notes_path else NOTES_PATH
    notes = load_notes(notes_file)
    problems = [f"field notes: {p}" for p in check_notes(notes)]
    run_dir = control.get("run_dir")
    found = G.discover(run_dir) if run_dir and Path(str(run_dir)).is_dir() else {}
    if not found:
        raise CatalogError(f"the control's run directory {run_dir} holds no history")
    streams = {s: G.open_stream(run_dir, s) for s in sorted(found)}
    daily, monthly = _first_with(streams, "daily"), _first_with(streams, "monthly")
    if daily is None:
        raise CatalogError(f"{run_dir} has no daily stream; its streams are "
                           f"{', '.join(f'{n} ({s.cadence})' for n, s in streams.items())}")
    headers = {s: _header(streams[s].files[0]) for s in (daily, monthly) if s}
    manifest = load_suite(suite_dir or control.get("suite_dir"))

    rows: List[Dict[str, Any]] = []
    for name in streams[daily].variables:
        rows.append(_field_row(name, notes.get(name) if notes.get(name) and is_row(notes[name])
                               else None, streams[daily], headers[daily], manifest, problems))
    in_rows = {r["name"] for r in rows}
    for name, e in sorted(notes.items()):
        if name in in_rows or not is_row(e):
            continue
        st = monthly if monthly and name in streams[monthly].variables else None
        if st is None:
            problems.append(f"{name}: the field notes name it, and neither the daily nor "
                            f"the monthly stream holds it")
            continue
        rows.append(_field_row(name, e, streams[st], headers[st], manifest, problems))
    rows = _order(rows)
    in_rows = {r["name"] for r in rows}

    outside = {}
    for name, e in sorted(notes.items()):
        if is_row(e):
            continue
        where = [s for s in sorted(streams) if name in streams[s].variables]
        if not where:
            problems.append(f"{name}: the field notes name it, and no stream holds it")
        outside[name] = {"cadence": e.get("cadence"), "streams": where,
                         "cautions": list(e.get("cautions") or []), "status": e.get("status")}

    other = {"monthly_fields": (len(set(streams[monthly].variables) - in_rows)
                                if monthly else 0),
             "monthly_stream": monthly,
             "subdaily": [{"stream": n, "fields": len(s.variables),
                           "steps_per_day": s.steps_per_day}
                          for n, s in sorted(streams.items()) if s.cadence == "subdaily"],
             "layers": [{"field": r["name"], "levels": r["levels"]}
                        for r in rows if r["levels"]]}

    regions = _regions(control, streams[daily].grid, manifest)
    problems += [f"regions: {p}" for p in regions["problems"]]

    study = study_record(study_dir) if study_dir else None
    facts = {n: stream_facts(s) for n, s in sorted(streams.items())}
    whole = sorted(set(facts[daily]["complete_years"]).intersection(
        *([facts[monthly]["complete_years"]] if monthly else [])))
    runs = [{"run": "control", "control": control.get("name"), "changes": [],
             "years": whole, "years_whole_in": "daily and monthly" if monthly else "daily",
             "suite": suite_version(manifest)}]
    if study:
        problems += [f"study: {p}" for p in study["problems"]]
        for r in study["runs"]:
            sdir = Path(study["dir"]) / "03_results" / "suite" / r["run"]
            runs.append({"run": r["run"], "changes": r["changes"], "years": study["years"],
                         "years_passed": study["years_passed"],
                         "suite": suite_version(load_suite(sdir))})
    ns = notes_status(notes)
    drafts = len(notes) - ns["approved"]     # a draft, or a review of an older version
    return {
        "catalog_version": CATALOG_VERSION,
        "control": {"name": control.get("name"), "run_dir": str(run_dir),
                    "case_dir": str(control.get("case_dir"))},
        "study": ({k: study[k] for k in ("dir", "name", "normal_years", "years",
                                         "years_passed")} if study else None),
        "notes": {"file": str(notes_file), "sha256": _sha256(notes_file),
                  "entries": len(notes), "drafts": drafts,
                  "rows": sum(1 for e in notes.values() if is_row(e)),
                  "outside_rows": {n: list(e.get("cautions") or []) for n, e in
                                   sorted(notes.items()) if not is_row(e)},
                  "status": ns},
        "runs": runs,
        "regions": regions,
        "streams": facts,
        "periods": {"daily_stream": daily, "monthly_stream": monthly,
                    "normal_years": study["normal_years"] if study else None,
                    "suite": (manifest.get("periods") if manifest else None),
                    "suite_absent": (manifest.get("periods_absent") if manifest else None)},
        "fields": rows,
        "notes_outside_rows": outside,
        "other_output": other,
        "configuration": control_configuration(control),
        "methods": methods_listed(),
        "access": access_calls(),
        "suite_manifest": manifest.get("_path") if manifest else None,
        "problems": problems,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CATALOG.md
# ─────────────────────────────────────────────────────────────────────────────

def _cell(text: Any) -> str:
    """Text safe inside a Markdown table cell."""
    return " ".join(str(text if text is not None else "").replace("|", "/").split())


def _int(n: int) -> str:
    return f"{int(n):,}"


def _change_text(c: Dict[str, Any]) -> str:
    parts = [f"{c.get('id', 'change')}"
             + (f" {json.dumps(c['params'], sort_keys=True)}" if c.get("params") else "")]
    if c.get("summary"):
        parts.append(str(c["summary"]))
    if c.get("meaning"):
        parts.append(f"Means: {c['meaning']}")
    if c.get("not_meaning"):
        parts.append(f"Does not mean: {c['not_meaning']}")
    return ". ".join(p.rstrip(".") for p in parts) + "."


def _units_text(row: Dict[str, Any]) -> str:
    u = f"{row['amount_units']} per period" if how_total(row) else str(row["units"])
    for lv in row.get("levels") or []:
        u += f", {lv['size']} layers ({lv['dim']})"
    return u


# A PERIOD LEFT OUT OF THE SUITE IS LEFT OUT OF ITS OWN PRODUCTS ONLY (the
# user's decision of 2026-11-08): both plans of the third batch's question 1
# read "WY2015 ... is left out" as "no melt-out date can be estimated for
# WY2015", and so planned no timing measure, while snow_gone dates that water
# year by its own contract. A period of a kind some timing measure uses now
# says that only its kind's products lack it and how each such measure
# answers it, read from the measures' contracts (a start-of-record check: not
# estimable; the partial-record outcome: a partial-record estimate). Kept
# short: the catalog's fixed part had 166 characters of room under MAX_CHARS.
def left_out_scope(kind: str, label: str) -> str:
    """What a left-out suite period of `kind` does not reach, for a kind a
    timing measure uses: only that kind's products lack it, and each timing
    measure of that kind answers it by its contract. "" for any other kind."""
    import domain_methods as DM       # here: the contracts, read only
    ms = [m for m in DM.TIMING_IDS if DM.TIMING_KIND.get(m) == kind]
    if not ms:
        return ""
    none = [m for m in ms if any(c.id == "record_holds_the_start"
                                 for c in DM.METHODS[m].applicability)]
    part = [m for m in ms if m not in none and any(
        o == DM.FLAGS["partial_record"][1] for o, _ in DM.METHODS[m].outcomes)]
    said = []
    if part:
        said.append(f"{' and '.join(part)} {'dates' if len(part) == 1 else 'date'} it as a "
                    f"partial-record estimate")
    if none:
        said.append(f"{' and '.join(none)} {'is' if len(none) == 1 else 'are'} not estimable")
    return f" Only the {kind} products lack it" + "".join("; " + x for x in said) + "."


def suite_product_names() -> List[str]:
    import domain_suite as S          # here: only the names of the products it makes
    return list(S.PRODUCTS)


def _products_text(p: Any, full: Sequence[str] = ()) -> str:
    """A row's products: 'all' when it has every product the suite makes,
    'all but' the one or two it lacks, else the list (review round 1,
    2026-11-05: the catalog must stay within MAX_CHARS with the suite built)."""
    if not isinstance(p, list):
        return str(p)
    if full and set(p) <= set(full):
        lack = [x for x in full if x not in p]
        if not lack:
            return "all"
        if len(lack) <= 2:
            return "all but " + ", ".join(lack)
    return ", ".join(p)


def _stream_line(f: Dict[str, Any], role: str) -> str:
    s = (f"- {role} stream {f['name']}: {f['first_label']} to {f['last_label']}, "
         f"{_int(f['records'])} records in {_int(f['files'])} files, {f['calendar']} "
         f"calendar; whole years {years_text(f['complete_years'])}")
    for y, p in sorted(f["partial_years"].items()):
        s += (f"; {y} partial ({_int(p['records'])} records, {_int(p['missing'])} missing "
              f"from {p['first_missing']})")
    for lo in f["left_out"]:
        s += f"; {_int(lo['records'])} record(s) left out: {lo['reason']}"
    return s + "."


def render(cat: Dict[str, Any]) -> str:
    """CATALOG.md from a catalog. Raises when its fixed part would pass
    MAX_CHARS or its study part STUDY_MAX_CHARS: the Analyzer reads it whole,
    so a longer one is a catalog to cut, not to clip."""
    return render_parts(cat)["text"]


def render_parts(cat: Dict[str, Any]) -> Dict[str, Any]:
    """CATALOG.md and its two parts: {"text", "fixed", "study"}, each part
    checked against its own limit (MAX_CHARS, STUDY_MAX_CHARS)."""
    L: List[str] = []
    title = f"# Data catalog: control {cat['control']['name']}"
    if cat.get("study"):
        title += f", study {cat['study']['name']}"
    L += [title, "",
          "What can be analysed and how each value is made, generated from the run's files, "
          "the registry's regions, the field notes and the suite. It holds no result: no "
          "anomaly, difference, rank or extreme.", ""]

    L += ["## 1. Runs", "", "| run | changes (quoted from the server) | years | suite |",
          "|---|---|---|---|"]
    for r in cat["runs"]:
        if r["run"] == "control":
            ch = f"none: the control ({r.get('control')})"
            yrs = f"{years_text(r['years'])} whole in its {r['years_whole_in']} output"
            L.append(f"| {_cell(r['run'])} | {_cell(ch)} | {_cell(yrs)} | "
                     f"{_cell(r['suite'])} |")
    at = len(L)                      # where the study part goes
    if not cat.get("study"):
        L += ["", "No study: this is the control's catalog. A study adds its runs here."]
    L += [""] + _configuration_lines(cat.get("configuration")) + [""]

    reg = cat["regions"]
    L += ["## 2. Regions", "",
          "Every region keeps only the vegetated cells: the cells whose land is entirely "
          "natural vegetation (PCT_NATVEG >= 100), not every cell with some vegetation. Area is "
          "each cell's area times its land fraction, summed.", "", "| region | cells | area (km2) | rule |",
          "|---|---|---|---|"]
    for r in reg["regions"]:
        rule = r["rule"] + (f"; {r['what']}" if r.get("what") else "")
        L.append(f"| {_cell(r['name'])} | {_int(r['cells'])} | {r['area_km2']:,.1f} | "
                 f"{_cell(rule)} |")
    L.append("")

    per = cat["periods"]
    L += ["## 3. Periods", ""]
    L.append(_stream_line(cat["streams"][per["daily_stream"]], "Daily"))
    if per.get("monthly_stream"):
        L.append(_stream_line(cat["streams"][per["monthly_stream"]], "Monthly"))
    if per.get("normal_years"):
        L.append(f"- Normal years, the plan's definition: {years_text(per['normal_years'])}.")
    elif cat.get("study"):
        L.append("- Normal years: the study's plan names none.")
    else:
        L.append("- Normal years: none; a study's plan names them.")
    if per.get("suite") is None:
        L.append(f"- Seasons, water years and the suite's other periods: {SUITE_NOT_BUILT}.")
    else:
        for kind, labels in sorted(per["suite"].items()):
            text = (f"{len(labels)} ({labels[0]} to {labels[-1]})"
                    if isinstance(labels, list) and labels else str(labels))
            L.append(f"- Suite periods, {kind}: {text}.")
        for kind, gone in sorted((per.get("suite_absent") or {}).items()):
            for lab, why in sorted((gone or {}).items()):
                L.append(f"- Suite periods left out, {kind} {lab}: {why}."
                         + left_out_scope(kind, lab))
    L.append("")

    rows = cat["fields"]
    groups = [g for g in GROUPS + (NO_NOTE_GROUP,) if any(r["group"] == g for r in rows)]
    notes = cat["notes"]
    L += ["## 4. Fields", "",
          f"{len(rows)} fields in {len(groups)} groups, in the model's order and "
          f"alphabetical within a group. how: a period's value is the total of a rate or the "
          f"average of anything else. type: flux, state, event, store or bounded index; a "
          f"store changes slowly from year to year with a weak seasonal cycle, so its drift "
          f"dominates, while a state has a strong seasonal cycle. scale: the physical scale, "
          f"ratio (zero means none, or no net flow for a signed quantity) or interval (the "
          f"zero is set by a unit or a baseline). percent: whether a change is also given as a "
          f"percent of its reference (the field notes' reporting policy, apart from the "
          f"scale); a field kept in percent changes in percentage points. units: a total's "
          f"are per period. saved: the stream the field is read from.", ""]
    full = suite_product_names() if any(isinstance(r["products"], list) for r in rows) else []
    if full:
        L += [f"products: the suite's products a field has; all means {', '.join(full)}.", ""]
    nopct: Dict[str, List[str]] = {}
    for r in rows:
        pc = r.get("percent") or {}
        if pc and not pc.get("allowed"):
            nopct.setdefault(str(pc.get("why")), []).append(r["name"])
    if nopct:
        L += ["No percent change, by the field notes' policy: " + "; ".join(
            f"{', '.join(sorted(ns))}: {_cell(why)}" for why, ns in nopct.items()) + ".", ""]
    L += _notes_lines(notes) + [""]
    for g in groups:
        L += [f"### {g}", "", "| field | meaning | how | type | scale | percent | units | saved "
              "| products |", "|---|---|---|---|---|---|---|---|---|"]
        cautions = []
        for r in (r for r in rows if r["group"] == g):
            pc = r.get("percent") or {}
            L.append(f"| {r['name']} | {_cell(r['meaning'])} | {r['how']} | "
                     f"{TYPE_WORDS.get(r['type'], r['type'] or '')} | {r['scale'] or ''} | "
                     f"{'yes' if pc.get('allowed') else 'no'} | "
                     f"{_cell(_units_text(r))} | {r['cadence']} | "
                     f"{_cell(_products_text(r['products'], full))} |")
            if r.get("type_why"):
                cautions.append(f"- {r['name']}: a {r['type']}: {_cell(r['type_why'])}")
            cautions += [f"- {r['name']}: {_cell(c)}" for c in r.get("cautions") or []]
            if r.get("how_why"):
                cautions.append(f"- {r['name']}: {r['how']}, not its units' "
                                f"{r['unit_rule']}: {_cell(r['how_why'])}")
        if cautions:
            L += ["", "Cautions:"] + cautions
        L.append("")

    L += ["## 5. Methods", ""] + _methods_lines(cat.get("methods")) + ["",
          "## 6. How to load", ""] + _access_lines(cat.get("access")) + ["",
          "## 7. Other output", ""]
    other = cat["other_output"]
    parts = []
    if other.get("monthly_stream"):
        parts.append(f"the monthly stream {other['monthly_stream']} holds "
                     f"{_int(other['monthly_fields'])} other fields")
    for s in other["subdaily"]:
        every = 24.0 / s["steps_per_day"] if s.get("steps_per_day") else None
        when = (f"every {every:g} hours" if every else "below daily")
        parts.append(f"the stream {s['stream']} holds {_int(s['fields'])} fields {when}")
    for lv in other["layers"]:
        for x in lv["levels"]:
            parts.append(f"{lv['field']} holds {_int(x['size'])} single layers ({x['dim']})")
    L.append(("Not reachable yet, since the calls that load values are not built: "
              + "; ".join(parts) + ".") if parts else "None.")
    L.append("")
    if cat.get("problems"):
        L += ["## Problems", ""] + [f"- {_cell(p)}" for p in cat["problems"]] + [""]
    fixed = "\n".join(L)
    if len(fixed) > MAX_CHARS:
        raise CatalogError(f"CATALOG.md's fixed part would be {len(fixed):,} characters, "
                           f"over the {MAX_CHARS:,} the Analyzer reads whole")
    study = study_lines(cat)
    n = len("\n".join(study))
    if n > STUDY_MAX_CHARS:
        raise CatalogError(f"CATALOG.md's study part ({len(cat['runs']) - 1} experiment "
                           f"runs) would be {n:,} characters, over the "
                           f"{STUDY_MAX_CHARS:,} the Analyzer reads whole")
    return {"text": "\n".join(L[:at] + study + L[at:]), "fixed": fixed,
            "study": "\n".join(study)}


def run_line(r: Dict[str, Any]) -> str:
    """One experiment run, compactly: each change's id and settings with the
    server's sentence of what the difference means, the years and the suite.
    The server's summary and its warnings are not repeated here (study_lines
    prints each warning once; suite.run_record gives the whole record)."""
    ch = " ".join(
        f"{c.get('id', 'change')}"
        + (f" {json.dumps(c['params'], sort_keys=True)}" if c.get("params") else "")
        + (f": {str(c['meaning']).rstrip('.')}." if c.get("meaning") else ".")
        for c in r["changes"]) or "no change recorded"
    yrs = f"planned {years_text(r['years'])}"
    if r.get("years_passed") is not None:
        yrs += f"; passed {years_text(r['years_passed'])}"
    return f"| {_cell(r['run'])} | {_cell(ch)} | {_cell(yrs)} | {_cell(r['suite'])} |"


def study_lines(cat: Dict[str, Any]) -> List[str]:
    """The study part: one line per experiment run in the runs table, then
    every change's "does not mean" warning whole, in the server's words, once
    for all the runs it applies to (in the order the runs first name it), and
    where each run's full record is read. Nothing is cut: a warning is never
    shortened, and the part raises in render when it passes its limit."""
    runs = [r for r in cat["runs"] if r["run"] != "control"]
    if not runs:
        return []
    out = [run_line(r) for r in runs]
    warn: Dict[str, List[str]] = {}
    for r in runs:
        for c in r["changes"]:
            if c.get("not_meaning"):
                who = warn.setdefault(str(c["not_meaning"]), [])
                tag = f"{r['run']} ({c.get('id', 'change')})"
                if tag not in who:
                    who.append(tag)
    if warn:
        out += ["", "What a run's difference from the control does NOT mean (the server's "
                    "words, each whole, once for the runs it applies to):"]
        out += [f"- {', '.join(who)}. Does not mean: {_cell(text).rstrip('.')}."
                for text, who in warn.items()]
    out += ["", f"A run's full record (each change's summary, meaning and does-not-mean in "
                f"the server's words, its years and its suite) is read with "
                f"{RUN_RECORD_CALL}: it reads no outcome, needs no review, and is logged "
                f"apart from the lookups."]
    return out


def _configuration_lines(cfg: Optional[Dict[str, Any]]) -> List[str]:
    """The control's verified configuration, from its facts cache: the
    nutrient cycles and supplements that settle how FPG reads, and the fire."""
    if not cfg:
        return ["Verified configuration of the control: its facts cache records none, so how "
                "FPG reads (the field notes) is not settled for this control."]
    nut = {"cnp": "carbon, nitrogen and phosphorus", "cn": "carbon and nitrogen"}.get(
        str(cfg.get("nutrient")), str(cfg.get("nutrient")))
    s = (f"Verified configuration of the control (its facts cache): nutrient cycles "
         f"{cfg.get('nutrient')} ({nut}), competition {cfg.get('nu_com')}; nitrogen supplement "
         f"{cfg.get('nitrogen_supplement')}, phosphorus supplement "
         f"{cfg.get('phosphorus_supplement')}")
    if (cfg.get("nutrient"), cfg.get("nitrogen_supplement"),
            cfg.get("phosphorus_supplement")) == ("cnp", "NONE", "ALL"):
        # STEP 0, CORRECTED WHEN CLOSING PACKAGE E (2026-11-05): the model
        # represents phosphorus limitation and the supplement switches it off, a
        # fact verified in one run and said only of it; FPG shows the nitrogen
        # limit at the time, not nitrogen's earlier effects on GPP
        s += (": phosphorus is supplemented to meet plant demand everywhere and nitrogen is "
              "not. The model represents phosphorus limitation, but the supplement switches "
              "it off (FPG_P, in the monthly output, was exactly 1 in every cell and month of "
              "2015-2023 in the Naches control run, checked on 2026-11-05: 160,728 finite "
              "cell-months a year), so in this configuration only nitrogen can limit growth "
              "by nutrients, the two cannot co-limit, and FPG measures only that nitrogen "
              "limitation of potential GPP, at the time. Nitrogen can also lower GPP through "
              "earlier effects FPG at the time does not show (less leaf area or leaf nitrogen "
              "grown in a nitrogen-poor period); water stress through BTRAN, temperature, "
              "light and leaf area limit growth separately.")
    else:
        s += ": not the configuration the field notes' reading of FPG was written for; read "
        s += "FPG for this one on its own (without the phosphorus supplement, FPG and FPG_P "
        s += "together are needed to tell nitrogen from phosphorus limitation)."
    out = [s]
    if cfg.get("fire_model_on") and not cfg.get("use_nofire") and \
            not cfg.get("prescribed_burned_area"):
        out.append("Fire in the control: ELM's own fire model, with no burned area "
                   "prescribed, so the model's fire is the only fire in the control; a no-fire "
                   "run switches that model off and so removes all of it.")
    return out


def _notes_lines(notes: Dict[str, Any]) -> List[str]:
    """The field notes' count and their four statuses, kept apart."""
    ns = notes.get("status") or {}
    outside = notes.get("outside_rows") or {}
    s = (f"Field notes: {notes['entries']} entries, one for each of the {notes.get('rows')} "
         f"fields below")
    for name, cautions in outside.items():
        s += f", plus {name}, kept only for its caution: {_cell('; '.join(cautions))}"
    s += "."
    tested_on = ", ".join(ns.get("tested_on") or []) or "no date"
    states = ns.get("review_states") or {}
    review = ", ".join(f"{n} {k}" for k, n in states.items() if n)
    if ns.get("approved_by"):
        review += f" (by {', '.join(ns['approved_by'])})"
    if states.get(APPROVED_STATE) or states.get(PROVISIONAL_STATE):
        review += f", {NOT_FINAL}"
    review = (review + "; " if review else "") + f"{notes['drafts']} drafts or reviews of an " \
        "older version"
    return [s, "", "Their statuses, kept apart:", "",
            f"- review: {review}",
            f"- implemented: {ns.get('implemented', 0)} at their behaviour version",
            f"- tested: {ns.get('tested', 0)} at their behaviour version, on {tested_on}",
            f"- available to the Analyzer: {ns.get('available_to_analyzer', 0)} (the "
            f"fields the suite holds, through the calls of section 6)"]


def _methods_lines(methods: Any) -> List[str]:
    """The methods that exist, by id, pointing to their contracts, with their
    four statuses kept apart."""
    if not isinstance(methods, dict) or not methods.get("methods"):
        return [f"The analysis methods are {NOT_BUILT}."]
    ms = methods["methods"]
    L = [f"The analysis methods, each with its contract (what it estimates, where it "
         f"applies, its settings, outcomes, numerical contract and limitations), are in "
         f"{methods['doc']}: " + ", ".join(m["id"] for m in ms) + ". Their statuses, kept "
         f"apart (versions are behaviour versions):", "",
         "| method | version | review | implemented | tested | available to the Analyzer |",
         "|---|---|---|---|---|---|"]
    for m in ms:
        st = m.get("status") or {}
        rv, im, te = st.get("review") or {}, st.get("implemented") or {}, st.get("tested")
        tested = (f"version {te.get('behaviour_version')}, on {te.get('date')}" if te
                  else "not yet")
        L.append(f"| {m['id']} | {m['version']} | {rv.get('state')} at text revision "
                 f"{rv.get('text_revision')}, version {rv.get('behaviour_version')} | "
                 f"version {im.get('behaviour_version')} | {tested} | "
                 f"{'yes' if st.get('available_to_analyzer') else 'no'} |")
    by = sorted({f"{(m.get('status') or {}).get('review', {}).get('by')} on "
                 f"{(m.get('status') or {}).get('review', {}).get('date')}" for m in ms
                 if ((m.get("status") or {}).get("review") or {}).get("state")
                 in (APPROVED_STATE, PROVISIONAL_STATE)})
    prov = [p for p in methods.get("provisional_settings") or []]
    if by or prov:
        L += [""]
        if by:
            L.append(f"Approved for integration by {', '.join(by)}: {NOT_FINAL}.")
        if prov:
            L.append("Settings approved only as provisional values: " + "; ".join(
                f"{p['method']}.{p['setting']} ({p.get('why') or 'pending a further review'})"
                for p in prov) + ".")
    return L


def write(cat: Dict[str, Any], out_dir) -> Dict[str, Path]:
    """catalog.json and CATALOG.md in out_dir (never a run or case directory)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = render(cat)
    pj, pm = out / CATALOG_JSON, out / CATALOG_MD
    pj.write_text(json.dumps(cat, indent=1) + "\n")
    pm.write_text(md)
    return {"json": pj, "md": pm}


def main(argv=None) -> int:
    import domain_controls
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--control", required=True, help="the registry's name of the control")
    ap.add_argument("--out", required=True, help="where catalog.json and CATALOG.md go")
    ap.add_argument("--study", help="a study's directory, for its runs and normal years")
    ap.add_argument("--registry", help="another registry file than the configured one")
    a = ap.parse_args(argv)
    cat = build(domain_controls.get(a.control, a.registry), study_dir=a.study)
    paths = write(cat, a.out)
    print(f"{paths['md']}: {len(paths['md'].read_text()):,} characters, "
          f"{len(cat['fields'])} fields, {len(cat['problems'])} problem(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
