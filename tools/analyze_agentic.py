#!/usr/bin/env python3
"""Agentic analysis of a completed run: choose figures, render, look, interpret.

Runs the loop a person runs. The Analyzer picks which figures answer the
question, deterministic code renders them from the registry, the model LOOKS at
each one and flags rendering problems, and the interpretation is written against
the JSON — bound by the verdicts validation already issued.

    source /qfs/people/tran289/IDEAS/env_compy.sh
    python3 tools/analyze_agentic.py --run-dir <dir> [--question "..."]
                                     [--no-vision] [--annotate]

Writes 04_analysis/analysis_plan.json (what it chose and why, so a rerun is
reproducible), figure_captions.json (verdicts live here, NOT on the images) and
interpretation.md.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from agents import drivers as _drv

from core.figure_registry import REGISTRY, available, detect_capabilities


def _load_tool(name):
    import importlib.util
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _read(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default if default is not None else {}


def render(name, ctx, out_dir):
    """Render one registry figure. Returns (path|None, provenance dict)."""
    entry = REGISTRY[name]
    tool = _load_tool(entry["tool"])
    fn = getattr(tool, entry["fn"])
    out = Path(out_dir) / f"{name}.png"
    args = []
    for a in entry["args"]:
        args.append(ctx[a])
    try:
        got = fn(*args, out)
        ok = bool(got) and out.exists()
        return (out if ok else None,
                {"figure": name, "provenance": "registry",
                 "renderer": f"{entry['tool']}.{entry['fn']}",
                 "rendered": ok})
    except Exception as e:
        return None, {"figure": name, "provenance": "registry",
                      "renderer": f"{entry['tool']}.{entry['fn']}",
                      "rendered": False, "error": str(e)[:200]}


def evidence_payload(hs, val, brief, plan, assumptions):
    """What the interpreter is allowed to reason from — and nothing else.

    Deliberately narrow. The interpretation may only state numbers that appear
    here, so anything absent is a number it cannot invent.
    """
    ok = [r for r in hs.get("experiments", []) if r.get("status") == "ok"]
    cols = []
    for r in ok:
        m = r["metrics"]
        wb = m.get("water_budget") or {}
        cols.append({"column": r["case_name"], "elevation_m": r.get("elevation_m"),
                     "precip_mm_yr": m.get("precip_mm_yr"),
                     "runoff_mm_yr": m.get("annual_runoff_mm_yr"),
                     "recharge_mm_yr": m.get("annual_recharge_mm_yr"),
                     "et_mm_yr": wb.get("et_mm_yr"),
                     "drainage_mm_yr": wb.get("drainage_mm_yr"),
                     "storage_change_mm": wb.get("storage_change_mm"),
                     "water_table_depth_m": m.get("water_table_depth_m"),
                     "peak_swe_mm": m.get("peak_swe_mm")})
    # the verdicts, carried verbatim — these BIND the interpretation
    verdicts = [{"variable": t.get("variable"), "status": t.get("status"),
                 "result": t.get("result"), "note": t.get("note")}
                for t in (val.get("targets") or [])]
    return {
        "question": brief.get("user_request"),
        "domain": brief.get("domain"),
        "feasibility": (plan or {}).get("feasibility"),
        "assumptions_ledger": assumptions,
        "limitations": hs.get("limitations"),
        "columns": cols,
        # computed here, not read: extraction no longer freezes them
        "spatial_summary": _drv.spatial_summary(hs.get("experiments") or []),
        "driver_matrix": _drv.driver_matrix(hs.get("experiments") or []),
        "soil_attribution": {k: v for k, v in (hs.get("soil_attribution") or {}).items()
                             if k != "by_recharge"},
        "validation_verdicts": verdicts,
        "domain_match": val.get("domain_match"),
        "runoff_ratio": val.get("runoff_ratio"),
        "catchment": val.get("catchment"),
        "hydrograph_metrics": {k: v for k, v in (val.get("hydrograph") or {}).items()
                               if k not in ("days", "obs", "mod",
                                            "obs_mm_day", "mod_mm_day")},
    }


def check_grounding(md, payload):
    """Flag numbers in the interpretation that do not appear in the evidence.

    Not a hard gate — prose legitimately carries years, counts and rounded
    forms. It surfaces anything that looks invented so a reader can check.

    Two normalisations, both learned from false positives on the first live
    run: the model writes a UNICODE minus (−55.8) where the JSON holds ASCII
    (-55.8), and it cites identifiers bare (12488500) where the JSON holds them
    prefixed (USGS-12488500). Neither is a fabrication and neither should be
    reported as one — a checker that cries wolf gets ignored, which costs more
    than it saves.
    """
    import re

    def norm(s):
        return (s.replace("\u2212", "-").replace("\u2013", "-")
                 .replace("\u2014", "-"))

    text = norm(json.dumps(payload, default=str))
    known = set(re.findall(r"-?\d+\.?\d*", text))
    known |= {k.lstrip("-") for k in list(known)}              # sign-insensitive
    known |= {k.rstrip("0").rstrip(".") for k in list(known) if "." in k}
    knownf = [float(k) for k in known if _isnum(k)]

    suspect = []
    for tok in re.findall(r"-?\d+\.?\d*", norm(md or "")):
        bare = tok.lstrip("-")
        if tok in known or bare in known:
            continue
        try:
            f = float(tok)
        except ValueError:
            continue
        if 1900 <= abs(f) <= 2100 or abs(f) <= 20:            # years, counts
            continue
        # a rounded form of something known is not an invention
        if any(abs(abs(f) - abs(k)) / max(abs(k), 1e-9) < .02 for k in knownf):
            continue
        suspect.append(tok)
    return sorted(set(suspect))


def _isnum(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser(description="Agentic analysis of a finished run")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--question", default=None,
                    help="defaults to the run's own reception brief")
    ap.add_argument("--model", default="claude-opus-4-8-project")
    ap.add_argument("--no-vision", action="store_true",
                    help="skip looking at the rendered figures")
    ap.add_argument("--cases-file", default="cases.json")
    args = ap.parse_args()

    rd = Path(args.run_dir)
    ana = rd / "04_analysis"
    ana.mkdir(parents=True, exist_ok=True)

    hs = _read(ana / "hydro_summary.json")
    if not hs:
        sys.exit(f"no {ana}/hydro_summary.json — run analyze_run.py first")
    val = _read(ana / "validation.json")
    brief = _read(rd / "reception_brief.json")
    plan = _read(rd / "plan.json")
    assumptions = _read(rd / "assumptions.json", [])
    cmeta = _read(rd / "columns.json")

    question = (args.question or brief.get("user_request")
                or "Summarise what this ensemble shows.")

    ar = _load_tool("analyze_run")
    results = {r["case_name"]: r for r in hs.get("experiments", [])}
    caps = detect_capabilities(results, val, cmeta)
    print(f"run capabilities: {', '.join(caps) or 'none'}")
    print(f"renderable figures: {', '.join(available(caps))}\n")

    from agents.analyzer_agent import AnalyzerAgent
    agent = AnalyzerAgent(model=args.model, vision=not args.no_vision)

    print("choosing figures …")
    sel = agent.select_figures(question, caps, brief, plan)
    for f in sel["figures"]:
        print(f"  → {f['name']}: {f.get('why', '')}")
    if sel.get("missing_capabilities"):
        print(f"  ! cannot draw: {sel['missing_capabilities']}")

    # deterministic render context — the LLM never touches these objects
    az_soil = None
    try:
        from core.elm_results_analyzer import ELMResultsAnalyzer
        exps = ar.build_experiments(rd, args.cases_file, "run_plan.json")
        _az = ELMResultsAnalyzer(exps, str(ana))
        _az.results = results
        az_soil = _az._compute_soil_attribution()
    except Exception:
        pass
    ctx = {"results": results, "run_dir": rd, "validation": val, "soil": az_soil}

    print("\nrendering …")
    captions, provenance = {}, []
    for f in sel["figures"]:
        name = f["name"]
        path, prov = render(name, ctx, ana)
        prov["why"] = f.get("why")
        if path:
            review = agent.review_figure(path, f.get("why") or REGISTRY[name]["question"])
            prov["review"] = review
            flag = "" if review.get("usable", True) else "  ⚠️ "
            probs = review.get("problems") or []
            print(f"  ✓ {name}{flag}" + (f" — {probs[0]}" if probs else ""))
        else:
            print(f"  ✗ {name}: {prov.get('error', 'not rendered')}")
        provenance.append(prov)
        captions[name] = {"answers": REGISTRY[name]["question"],
                          "chosen_because": f.get("why"),
                          "provenance": prov["provenance"]}

    (ana / "analysis_plan.json").write_text(json.dumps(
        {"question": question, "capabilities": caps, "selection": sel,
         "figures": provenance}, indent=2))
    (ana / "figure_captions.json").write_text(json.dumps(captions, indent=2))

    print("\ninterpreting …")
    payload = evidence_payload(hs, val, brief, plan, assumptions)
    md = agent.interpret(question, payload)
    ungrounded = check_grounding(md, payload)
    if ungrounded:
        md += ("\n\n> ⚠️ ungrounded figures flagged for review: "
               + ", ".join(ungrounded))
        print(f"  ⚠️  numbers not found in the evidence: {ungrounded}")
    (ana / "interpretation.md").write_text(md)
    print("\n" + md)
    print(f"\nwritten to {ana}/")


if __name__ == "__main__":
    main()
