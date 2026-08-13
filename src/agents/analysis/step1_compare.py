#!/usr/bin/env python3
"""
Analyzer step 1 — the model against the observations
src/agents/analysis/step1_compare.py

    in   ctx
    out  04_analysis/comparison.json + a figure per observable + the map

THE COMPARISON ITSELF LIVES IN THE ELM SERVER (2026-08-12). This step used to
own four modules of it — step1_compare_swe, _wtd, _streamflow and step1_maps,
about 1,400 lines that knew H2OSNO from ZWT, what a snow pillow is sited for,
and how deep ELM's soil column goes. None of that is model-agnostic, and the
Analyzer is the one box that has to be: it reads PFLOTRAN runs too. So the
knowledge moved to `mcp/elm-mcp/src/compare/`, where it sits beside the model it
is about (docs/ELM_MCP_PLAN.md §9 phase 5, §11 decision 3), and this file became
the caller.

WHAT STAYS HERE IS THE JUDGEMENT. The split follows the rule phase 4 already
set for the limitations payload: **rows and units come back from the model's
server; the framework attaches the caveats.** The comparison returns
measurements and refuses to grade them — "bias = -85.9 mm" and never "the model
underestimates snow". A caveat is the opposite kind of statement: it says what
those measurements are not allowed to support, which is a judgement about
inference, not a reading of a history file. Two different jobs, two sides of the
boundary, and this file is the only place they meet.

    ELM's server measures          this file judges
    ────────────────────           ────────────────
    bias, RMSE, NSE, KGE           whether a metric may be called skill
    16 of 16 columns below 3.8 m   that no drainage claim about them is
                                   a statement about groundwater
    2 stations, 0 wells            that the absence is a fact about the
                                   basin and not a failed comparison

WHY THE RECORD IS PERSISTED. Same reason step 0 is the only file reader: every
later step must be runnable against an archived study. Step 3 could not be
developed at all while its input existed only in memory, and the caveats raised
here — the ones that bind everything downstream — would have existed nowhere on
disk.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# WHICH MODEL'S COMPARISON, resolved from the run rather than assumed.
#
# A REGISTRY, NOT AN IMPORT (2026-08-13). This file hardcoded ELM's package and
# imported it whatever the run was — so a PFLOTRAN study reaching step 1 would
# have asked ELM's modules for H2OSNO and ZWT, found neither, and written four
# empty records plus their caveats as though the comparison had been made and
# found nothing. That is the exact failure this box exists to avoid: "we could
# not look" reported as "there is nothing there".
#
# So the model chooses the comparison, the same way core/backends.py lets the
# model choose the manager class. A backend with no comparison says so; it does
# not get ELM's.
#
#     model            module        where
#     elm              compare       mcp/elm-mcp/src/compare/
#     pflotran         —             none yet; see below
#
# ADDING PFLOTRAN is a package and one line here. It would declare its own
# observables — a PFLOTRAN column has no snowpack, and its comparands are
# concentrations and heads against wells — and the Spec/compare()/plot() shape
# is model-agnostic on purpose. Nothing else in the Analyzer changes.
#
# IN-PROCESS RATHER THAN OVER MCP, deliberately. The Analyzer must run against
# an ARCHIVED run directory with no servers configured — that is what makes
# step 3 re-runnable, and it is how this step is developed. A tool call would
# make the Analyzer's one input a live subprocess. It is the same edge
# core/backends.py already opens for ELMExpManager, and it goes when that does.
_SERVER_SRC = {
    "elm": Path(__file__).resolve().parents[3] / "mcp" / "elm-mcp" / "src",
}
_COMPARE_MODULE = {"elm": "compare"}

FILENAME = "comparison.json"

# ELM's hydrologically active soil column. Quoted in a caveat, so it is named
# once here and read from the comparison record when the record supplies it.
ACTIVE_SOIL_M = 3.8


def compare_all(ctx, out_dir, draw: bool = True) -> Dict[str, Any]:
    """The four comparisons, their figures, and what they forbid.

    `draw` exists so a caller that only needs the records — step 3 re-reading
    an archived run, a test — does not pay for five renders it will not look
    at.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = str((ctx.data or {}).get("model") or "").lower() or "elm"
    mod_name = _COMPARE_MODULE.get(model)
    if mod_name is None:
        # NOT AN ERROR, AND NOT AN EMPTY COMPARISON. This backend has no
        # observation comparison written yet; saying so is the finding. The
        # alternative — running ELM's — would report four observables as
        # compared-and-empty for a model that has none of those variables.
        return {"observables": {}, "summary": {}, "caveats": [], "findings": [],
                "figures": {},
                "skipped": f"no observation comparison is registered for model "
                           f"'{model}' — the comparison is model knowledge and "
                           f"lives in that model's server. Registered: "
                           f"{sorted(_COMPARE_MODULE)}."}

    src = _SERVER_SRC.get(model)
    if src and src.is_dir() and str(src) not in sys.path:
        sys.path.append(str(src))       # append: the framework's own modules
                                        # must still win a name collision
    try:
        _cmp = __import__(mod_name)
    except ImportError as e:                                    # noqa: BLE001
        raise RuntimeError(
            f"the comparison for '{model}' lives in that model's server "
            f"({src}) and could not be imported: {e}") from e

    # THE ROWS COME FROM ctx, not from a file this step opens. ctx.columns is
    # already the shape the comparison reads — case_name, lat/lon/elevation_m,
    # and each variable's daily block — because both were built from the same
    # extraction. Reading 03_results/extracted.json here instead would make
    # step 1 a second file reader AND would compare a different set of columns
    # from the one every other step reports on.
    rows = [r for r in ctx.columns if isinstance(r, dict)]
    reception = (ctx.sources or {}).get("reception")
    if not reception:
        raise RuntimeError("no reception.json in this run — the observations "
                           "are fetched once, by reception, and persisted "
                           "there; nothing else in the run has them")

    done = _cmp.compare_run(rows, str(reception), str(out_dir), draw=draw)
    out = done["comparison"]
    summary = done["summary"]

    findings = as_findings(summary, done["figures"])

    record = {
        "observables":  out.get("observables") or {},
        "summary":      summary,
        "domain":       out.get("domain"),
        "note":         out.get("note"),
        "n_observation_rows_dropped": out.get("n_observation_rows_dropped"),
        # KEPT AND ALWAYS EMPTY, deliberately. Step 4 and step 2 both read
        # `comparison["caveats"]`, and an archived run written before
        # 2026-08-13 has real entries here. Dropping the key would make a new
        # record and an old one differently shaped, which is a worse failure
        # than an empty list: the reader cannot tell "this step no longer
        # derives caveats" from "this file is truncated".
        "caveats":      [],
        "findings":     findings,
        "figures":      done["figures"],
    }
    # Rewritten over what compare_run just wrote. The measurements are the
    # server's, and this file is what step 3 reads when it is re-run alone
    # against an archived directory.
    (out_dir / FILENAME).write_text(json.dumps(record, indent=2, default=str))
    return record


def load(out_dir) -> Dict[str, Any]:
    """Read back a previous comparison, or {} if step 1 has not run."""
    p = Path(out_dir) / FILENAME
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

# ─────────────────────────────────────────────────────────────────────
# STEP 1 NO LONGER DERIVES CAVEATS  (2026-08-13)
# ─────────────────────────────────────────────────────────────────────
# It produced eight: blocking ones for an unrouted column against a routed
# gauge, a water table below the active soil, an unobservable snow onset and a
# gauge draining more ground than was simulated, plus qualify/context ones for
# SNOTEL siting, window overlap, gap-filled towers and absent stations.
#
# THEY ADDED NO INFORMATION. Every fact behind them is already a MEASUREMENT in
# the summary, and the summary is already in the brief the interpreter reads:
#
#     below_active_soil: {active_soil_depth_m=3.8, n_columns_mean_below=16, of=16, ...}
#     onset_censored_columns: {n=16, of=17, columns=[...]}
#     gauges_exceeding_modelled_domain: {n=0, of=2, basin_area_km2=2860.6, ...}
#
# The caveat restated one of those as a verdict about what may be claimed —
# which is precisely what the docstring at the top of this file says this step
# does not do. Measuring and ruling on the measurement are different jobs, and
# the second belongs to whoever interprets.
#
# AND IN PRACTICE IT WAS A TRAP, not a guardrail. step2_investigate builds its
# caveat block from ctx.caveats alone, so these were never quoted to the model
# choosing the figures. The number was shown; the rule was not; and step 3's
# `respects` audit then struck claims for breaking it. Brandywine lost a
# factually correct sentence that way.
#
# WHAT STILL BINDS: step 0's caveats, read out of experiment.json's
# `limitations` and `assumptions_ledger`. Those describe the experiment's scope
# and the model choices behind it, which is what a caveat was for.
#
# WHAT REPLACES THE ENFORCEMENT: nothing mechanical, deliberately. The
# interpreter is given the numbers and is expected to understand what the scope
# and the model choice permit — see the note in as_findings about what a
# finding still carries.


def _n(block: Optional[Dict], key: str, default=0):
    return (block or {}).get(key, default) or default


# ─────────────────────────────────────────────────────────────────────
# THE COMPARISON AS EVIDENCE STEP 3 MAY CITE
# ─────────────────────────────────────────────────────────────────────
def as_findings(summary: Dict[str, Any],
                figures: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Each observable as a finding, in step 2's shape.

    ADDED 2026-08-12, and it closes a hole. Step 3 audits every claim against
    the findings step 2 produced, and step 1's comparison was not among them —
    so the one part of the analysis that touches an OBSERVATION could not be
    cited, and any sentence about it was struck as unsupported. The interpreter
    was left able to describe the model and forbidden to say how it compared.

    The result is the SUMMARY block, not the full record: the audit checks a
    claim's numbers against the finding it cites, so the finding must hold
    exactly what the brief showed.

    `blocked_by` IS NOW ALWAYS EMPTY. It used to carry the observable's caveat
    ids, which is what made a streamflow claim owe the routing caveat. Step 1
    stopped deriving caveats on 2026-08-13 — see the note above — so the
    finding carries the measurement and nothing about what may be concluded
    from it. The block itself stays because step 2 and step 3 both read the
    key, and an absent key and an empty one are different shapes.
    """
    questions = {
        "swe": "How does modelled snow water equivalent stand against the "
               "snow pillows, and what did the snowpack do across the columns?",
        "water_table": "How does the modelled water table stand against the wells and "
               "the published references, and where does it sit in the soil?",
        "streamflow": "How does the ensemble-mean runoff stand against the "
                      "gauges, and how does water leave each column?",
        "et": "How does modelled evapotranspiration stand against the flux "
              "towers, and where does the ET come from?",
    }
    out = []
    for name, block in (summary or {}).items():
        if not isinstance(block, dict):
            continue
        out.append({
            "id": f"compare_{name}",
            "question": questions.get(name, f"{name}: model against observations"),
            "scale": "overall",
            "n": block.get("n_columns"),
            "result": block,
            "blocked_by": [],
            "figure": figures.get(name),
            "source": "step1_compare",
        })
    return out


# How many matched stations are written out per observable before the rest are
# counted. Brandywine has 20 gauges, all against the same ensemble mean; the
# first few show the spread and the twentieth adds nothing a prompt can use.
MAX_ROWS = 6


def _fmt(v: Any) -> str:
    """A value as text, NEVER rounded.

    An earlier version printed floats at four significant figures, which reads
    better and is a trap: the step-3 audit compares a claim's declared value
    against the raw record, so a reviewer quoting the 2861 it was shown gets
    struck for inventing a number that the record holds as 2860.6. What the
    prompt shows and what the audit accepts have to be the same string.
    """
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}={_fmt(x)}" for k, x in v.items()) + "}"
    return str(v)


def _clip(text: str, limit: int) -> str:
    """Cut at a separator, never mid-number. `0.72365` clipped to `0.72` is a
    different measurement, and one the audit will correctly refuse."""
    if len(text) <= limit:
        return text
    cut = max(text.rfind(", ", 0, limit), text.rfind("{", 0, limit))
    return text[:cut if cut > 40 else limit].rstrip(", ") + " …"


def format_comparison(summary: Dict[str, Any], max_rows: int = MAX_ROWS) -> str:
    """The comparison as prompt text. One writer, two readers (steps 2 and 3).

    Written from the SUMMARY rather than the record, so what a prompt shows and
    what the step-3 audit checks a claim against are the same object. A number
    visible here and absent there is how a correct sentence gets struck.
    """
    lines: List[str] = []
    for name, b in (summary or {}).items():
        if not isinstance(b, dict):
            continue
        head = [f"{b.get('n_columns')} columns", f"{b.get('n_stations')} station(s)"]
        if b.get("colocated") is False:
            head.append("NOT co-located — basin aggregate")
        lines.append(f"  {name} [{b.get('units')}] — " + ", ".join(head))
        # FIRST, because everything under it is meaningless without it. A
        # reader that does not know `streamflow` here means QOVER + QDRAI will
        # invent its own definition — one live run defined the drainage limb as
        # QDRAI + QCHARGE and reported two different ensemble totals in one
        # answer without noticing they were different quantities.
        if b.get("model_comparand"):
            lines.append(f"      model: {b['model_comparand']}")
        if b.get("obs_quantity"):
            lines.append(f"      obs:   {b['obs_quantity']}")
        for key in ("error", "skipped"):
            if b.get(key):
                lines.append(f"      {key}: {_clip(str(b[key]), 300)}")
        rows = b.get("matched") or []
        for m in rows[:max_rows]:
            bits = [f"{m.get('station_id')}"]
            if m.get("column"):
                bits.append(f"vs {m['column']}")
            for k in ("n_days", "separation_km", "bias", "rmse", "nse", "kge",
                      "model_days_later_than_gauge", "frac_measured"):
                if m.get(k) is not None:
                    bits.append(f"{k}={_fmt(m[k])}")
            lines.append("      " + "  ".join(bits))
        if len(rows) > max_rows:
            lines.append(f"      … and {len(rows) - max_rows} more station(s)")
        for k, v in b.items():
            if k in ("units", "colocated", "n_columns", "n_stations", "matched",
                     "error", "skipped", "model_comparand", "obs_quantity"):
                continue
            lines.append(f"      {k}: {_clip(_fmt(v), 500)}")
    return "\n".join(lines)
