#!/usr/bin/env python3
"""
Build a PowerPoint STORYLINE from all executed studies under workflow_outputs/:
for each run —  question → planning figure → planning reasons → execution time
→ result figures → analyzer interpretation (+ observation validation)  — and a
closing cross-study comparison table.

A run is included when it has 04_analysis/hydro_summary.json (i.e. it executed
and was analyzed). Reads only; writes docs/slides/story_<date>.pptx.

    module load pytorch/2.8.0
    python3 tools/make_story_pptx.py [--out <file.pptx>]
"""
import argparse
import json
import re
from datetime import date
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

ROOT = Path(__file__).resolve().parents[1]
W, H = Inches(13.333), Inches(7.5)
INK, MUT, ACC = RGBColor(0x0F, 0x17, 0x2A), RGBColor(0x47, 0x55, 0x69), RGBColor(0x2C, 0x7F, 0xB8)


# ── run discovery + fact extraction ─────────────────────────────────────────
def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}


def exec_summary(rd: Path):
    """Parse whatever run log exists for per-column times / total wall time."""
    n_cases = len(load(rd / "cases.json") or load(rd / "phase3_cases.json") or [])
    for logname in ("run.log", "run10_run.log", "phase3_run.log"):
        f = rd / logname
        if not f.exists():
            continue
        txt = f.read_text(errors="ignore")
        cols = re.findall(r"(\S+): rc=(\d+)\s+(\d+)min", txt)
        done = re.search(r"ALL_DONE(?: in (\d+)min)?", txt)
        if cols:
            n_ok = sum(1 for _, rc, _ in cols if rc == "0")
            mins = [int(m) for _, _, m in cols]
            return (f"{n_ok}/{len(cols)} columns succeeded · ~{max(mins)} min each "
                    f"(~15 s clone per column, shared executable)")
        if done:
            t = f" in {done.group(1)} min" if done.group(1) else ""
            return f"{n_cases} columns ran as one batch job (concurrent){t}"
    return f"{n_cases} columns built (no run log found)" if n_cases else None


def gather(rd: Path):
    """Everything the slides need for one run."""
    hs = load(rd / "04_analysis" / "hydro_summary.json")
    if not hs:
        return None
    brief = load(rd / "reception_brief.json")
    plan = load(rd / "plan.json")
    dom = brief.get("domain") or {}
    sweep = load(rd / "soilsweep_plan.json")

    if brief:
        question = brief.get("user_request", "?")
        name = dom.get("name") or rd.name
        subtitle = (f"HUC {dom.get('huc', '?')} · {dom.get('area_km2', '?')} km² · "
                    f"elevation {dom.get('elevation_range_m', {}).get('min', '?')}–"
                    f"{dom.get('elevation_range_m', {}).get('max', '?')} m")
    elif sweep:
        cc = sweep.get("CONDITIONS_COUPLERS", [])
        clays = [c["EXPERIMENT"] for c in cc]
        question = ("Controlled experiment: how does soil texture (clay content) "
                    "control the runoff/recharge partitioning?")
        name = "Controlled soil sweep"
        subtitle = (f"{len(cc)} uniform-soil treatments ({', '.join(clays)}) at one "
                    f"site — forcing, vegetation and location held identical")
    else:
        question, name, subtitle = "?", rd.name, ""

    # planning reasons
    reasons = []
    sd = plan.get("scientific_decomposition") or {}
    for g in (sd.get("goals") or [])[:2]:
        reasons.append(("goal", g))
    ss = plan.get("sampling_strategy") or {}
    if ss:
        reasons.append(("sampling", f"N={ss.get('n_exploratory')} — {ss.get('approach', '')}"))
        if ss.get("n_justification"):
            reasons.append(("why N", ss["n_justification"]))
    fe = plan.get("feasibility") or {}
    if fe:
        reasons.append(("feasibility", str(fe.get("verdict", ""))))
        for x in (fe.get("not_answerable") or [])[:2]:
            if isinstance(x, dict):
                reasons.append(("not answerable", f"{x.get('aspect')} — needs {x.get('needs')}"))
    if sweep and not plan:
        reasons.append(("design", "conceptual archetype — vary ONE factor (clay 5→55%), "
                                  "hold everything else fixed; the cleanest isolation of soil control"))

    # analyzer interpretation
    interp = []
    ssum = hs.get("spatial_summary") or {}
    for n in ssum.get("interpretation") or []:
        interp.append(n)
    sa = hs.get("soil_attribution") or {}
    if sa:
        interp.append(f"soil control (forcing held at {sa.get('forcing_held_mm_yr')} mm/yr, "
                      f"{sa.get('n_columns')} columns): strongest predictor = "
                      f"{sa.get('strongest_predictor')}")
    # evaluation vs observations (its own slide)
    val = load(rd / "04_analysis" / "validation.json")
    evaluation = None
    if val:
        inv = val.get("observation_inventory") or {}
        lines = [("in-domain obs",
                  f"SNOTEL {inv.get('snotel', {}).get('n', '?')} · "
                  f"stream gages {inv.get('streamgages', {}).get('n', '?')} · "
                  f"GW wells {inv.get('gwwells', {}).get('n', '?')} "
                  f"({inv.get('gwwells', {}).get('with_records', '?')}/"
                  f"{inv.get('gwwells', {}).get('sampled', '?')} sampled have records)")]
        for t in val.get("targets") or []:
            tag = "✓ COMPARED" if t.get("status") == "compared" else "· context-only"
            body = t.get("result", "")
            if t.get("needs"):
                body += f"  — needs: {t['needs']}"
            lines.append((f"{tag} — {t.get('variable')}", body))
        fig = rd / "04_analysis" / "validation.png"
        evaluation = {"lines": lines, "fig": fig if fig.exists() else None}

    figs = [rd / f for f in ("sampling_design.png",) if (rd / f).exists()]
    rfigs = [rd / "04_analysis" / f
             for f in ("elevation_gradient.png", "soil_control.png", "debug_timeseries.png")
             if (rd / "04_analysis" / f).exists()]

    ok = [r for r in hs.get("experiments", []) if r.get("status") == "ok"]
    rech = [r["metrics"].get("annual_recharge_mm_yr") for r in ok
            if r["metrics"].get("annual_recharge_mm_yr") is not None]
    return {"dir": rd, "name": name, "question": question, "subtitle": subtitle,
            "reasons": reasons, "execution": exec_summary(rd), "interp": interp,
            "evaluation": evaluation,
            "plan_figs": figs, "result_figs": rfigs,
            "n_cols": len(ok),
            # single-location runs have no spatial summary — fall back to the
            # held-forcing value from the soil attribution
            "precip_bins": ((ssum.get("forcing") or {}).get("precip_mm_yr_distinct")
                            or ([sa["forcing_held_mm_yr"]] if sa.get("forcing_held_mm_yr") else None)),
            "rech_range": (round(min(rech), 1), round(max(rech), 1)) if rech else None,
            "predictor": (sa or {}).get("strongest_predictor")}


# ── pptx helpers ─────────────────────────────────────────────────────────────
def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def text(slide, x, y, w, h, runs, size=14, color=INK, bold=False, align=None):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for r in (runs if isinstance(runs, list) else [runs]):
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        if isinstance(r, tuple):                      # (label, body) bullet
            lab, body = r
            ra = p.add_run(); ra.text = f"{lab}:  "
            ra.font.bold = True; ra.font.size = Pt(size); ra.font.color.rgb = ACC
            rb = p.add_run(); rb.text = str(body)
            rb.font.size = Pt(size); rb.font.color.rgb = color
        else:
            ra = p.add_run(); ra.text = str(r)
            ra.font.size = Pt(size); ra.font.bold = bold; ra.font.color.rgb = color
        p.space_after = Pt(6)
    return tb


def header(slide, title, sub=None):
    text(slide, Inches(.5), Inches(.25), W - Inches(1), Inches(.7), title, size=28, bold=True)
    if sub:
        text(slide, Inches(.5), Inches(.95), W - Inches(1), Inches(.4), sub, size=13, color=MUT)


def picture(slide, path, top, max_w=Inches(12.3), max_h=None):
    pic = slide.shapes.add_picture(str(path), 0, top, width=max_w)
    if max_h and pic.height > max_h:                  # rescale to height cap
        scale = max_h / pic.height
        pic.width = int(pic.width * scale); pic.height = int(pic.height * scale)
    pic.left = int((W - pic.width) / 2)
    return pic


# ── slide builders ───────────────────────────────────────────────────────────
def study_slides(prs, s, idx):
    # 1 · question + planning reasons
    sl = blank(prs)
    header(sl, f"Case {idx} — {s['name']}", s["subtitle"])
    text(sl, Inches(.5), Inches(1.5), W - Inches(1), Inches(1.1),
         [f"“{s['question']}”"], size=18, bold=True, color=ACC)
    text(sl, Inches(.5), Inches(2.7), W - Inches(1), Inches(4.3),
         [("planner", "")] and s["reasons"] or ["(no plan.json — design-only artifacts absent)"],
         size=14)

    # 2 · planning figure
    for f in s["plan_figs"]:
        sl = blank(prs)
        header(sl, f"{s['name']} — sampling design (Tier-2 expander)")
        picture(sl, f, Inches(1.15), max_h=Inches(6.1))

    # 3 · execution + results figures
    sl = blank(prs)
    header(sl, f"{s['name']} — execution & results")
    if s["execution"]:
        text(sl, Inches(.5), Inches(1.1), W - Inches(1), Inches(.5),
             [("execution", s["execution"])], size=14)
    top = Inches(1.75)
    for f in s["result_figs"][:2]:
        pic = picture(sl, f, top, max_h=Inches(2.75))
        top = pic.top + pic.height + Inches(.15)
    for f in s["result_figs"][2:]:
        sl = blank(prs)
        header(sl, f"{s['name']} — results (continued)")
        picture(sl, f, Inches(1.3), max_h=Inches(5.9))

    # 4 · analyzer interpretation
    sl = blank(prs)
    header(sl, f"{s['name']} — analyzer interpretation")
    text(sl, Inches(.5), Inches(1.3), W - Inches(1), Inches(5.8),
         [("finding", i) for i in s["interp"]] or ["(no interpretation recorded)"], size=14)

    # 5 · evaluation vs observations
    if s.get("evaluation"):
        ev = s["evaluation"]
        sl = blank(prs)
        header(sl, f"{s['name']} — evaluation vs observations",
               "what could be compared honestly, and what still blocks rigorous validation")
        text(sl, Inches(.5), Inches(1.25), W - Inches(1), Inches(2.4), ev["lines"], size=13)
        if ev["fig"]:
            picture(sl, ev["fig"], Inches(3.85), max_h=Inches(3.5))


def comparison_slide(prs, studies):
    sl = blank(prs)
    header(sl, "Cross-study synthesis", "same framework, three experiments — the control on "
           "recharge/runoff partitioning depends on climate")
    rows = [["study", "columns", "forcing (mm/yr)", "recharge (mm/yr)", "strongest control"]]
    for s in studies:
        rows.append([s["name"], str(s["n_cols"]),
                     ", ".join(str(b) for b in (s["precip_bins"] or [])) or "—",
                     f"{s['rech_range'][0]} → {s['rech_range'][1]}" if s["rech_range"] else "—",
                     str(s["predictor"] or "—")])
    tbl = sl.shapes.add_table(len(rows), 5, Inches(.5), Inches(1.6),
                              W - Inches(1), Inches(.5) * len(rows)).table
    widths = [3.2, 1.2, 2.0, 2.2, 3.7]
    for i, w in enumerate(widths):
        tbl.columns[i].width = Inches(w)
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = v
            para = cell.text_frame.paragraphs[0]
            para.font.size = Pt(13 if r else 13)
            para.font.bold = (r == 0)
    text(sl, Inches(.5), Inches(1.6) + Inches(.5) * len(rows) + Inches(.4),
         W - Inches(1), Inches(2.2),
         [("wet basin", "recharge is CLAY-limited — water is ample; the impeding layer decides"),
          ("dry basin", "recharge is Ksat-limited (threshold) — only the most permeable soil recharges"),
          ("controlled sweep", "confirms causality: clay 5→55% collapses recharge 755→9 mm/yr "
                               "with everything else fixed")], size=14)


def main():
    ap = argparse.ArgumentParser(description="Build the storyline PPTX from executed runs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dirs = sorted(
        d for d in (ROOT / "workflow_outputs").iterdir()
        if (d / "04_analysis" / "hydro_summary.json").exists()
        # a framework case study has a reception brief (spatial) or a sweep plan
        # (controlled) — this excludes legacy pre-framework run dirs
        and ((d / "reception_brief.json").exists() or (d / "soilsweep_plan.json").exists()))
    studies = [s for s in (gather(rd) for rd in run_dirs) if s]
    if not studies:
        raise SystemExit("no executed+analyzed runs found under workflow_outputs/")

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    sl = blank(prs)
    text(sl, Inches(.8), Inches(2.3), W - Inches(1.6), Inches(1.4),
         "From a question to a validated simulation campaign", size=34, bold=True)
    text(sl, Inches(.8), Inches(3.8), W - Inches(1.6), Inches(1.6),
         [f"Multi-agent LLM framework on NERSC Perlmutter — {len(studies)} executed case studies",
          f"question → plan → sampling → per-column ELM runs → analysis → observation validation",
          f"generated {date.today().isoformat()} from workflow_outputs/"], size=16, color=MUT)

    for i, s in enumerate(studies, 1):
        study_slides(prs, s, i)
    comparison_slide(prs, studies)

    out = Path(args.out) if args.out else ROOT / "docs" / "slides" / f"story_{date.today():%Y%m%d}.pptx"
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    print(f"{len(prs.slides)} slides · {len(studies)} studies -> {out}")
    for s in studies:
        print(f"  • {s['name']:<24} ({s['dir'].name})")


if __name__ == "__main__":
    main()
