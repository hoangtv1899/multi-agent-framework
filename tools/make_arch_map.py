#!/usr/bin/env python3
"""The architecture map, generated from the source tree.

    python3 tools/make_arch_map.py [--out workflow_outputs/arch_map.html]

WHY GENERATED AND NOT DRAWN. A hand-maintained diagram becomes a second source
of truth and rots — and a rotted architecture diagram is worse than none,
because it is consulted. Everything checkable is read from disk on every run:
which files exist, how long they are, which MCP servers are registered. Edit
this script, never the HTML.

WHAT IS STILL PROSE. The `flow` lines describing what happens inside a stage are
written here by hand. They are the part that can drift, and they are marked as
such rather than pretended otherwise. Deriving them from call order was
considered and rejected: the boundary is what moves, and the internals mostly
do not.

The stage data is injected with json.dumps rather than written as a JavaScript
literal. That is not fastidiousness — the first version of this page was hand-
written JS and shipped with a broken string escape (`\\"` inside a double-quoted
string, which closes it) that killed the whole script block and rendered a blank
page. json.dumps cannot make that mistake.
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def lines(rel: str) -> int:
    p = ROOT / rel
    try:
        return sum(1 for _ in p.open(errors="ignore"))
    except OSError:
        return 0


def ref(text, path, symbol):
    """One workflow step, with the function that implements it.

    Verified: `symbol` must be defined in `path`, or main() warns. A reference
    that has quietly gone stale is worse than none — it sends a reader to a file
    to look for something that is not there.
    """
    return [text, f"{path}::{symbol}"]


def check_ref(r):
    if not (isinstance(r, list) and len(r) == 2 and "::" in r[1]):
        return None
    path, sym = r[1].split("::", 1)
    src = ROOT / path
    if not src.is_file():
        return f"{path} does not exist ({sym})"
    body = src.read_text(errors="ignore")
    # A prompt file and a shell script have no `def`. Look for the token itself
    # there — the reference still has to point at something that exists, which is
    # the property worth checking; only the definition of "defined" differs.
    pat = (rf"^\s*(def|class)\s+{re.escape(sym)}\b" if path.endswith(".py")
           else re.escape(sym))
    if re.search(pat, body, re.M):
        return None
    return f"{sym} not found in {path}"


def f(*paths):
    """[path, line-count] per file, counted now."""
    return [[p, lines(p)] for p in paths]


def data_servers():
    """The registered MCP servers, minus elm, read from mcp_config.json.

    Read rather than listed: a server added to the config appears here without
    anyone remembering to update a diagram.
    """
    blurb = {
        "terrain":    "Watershed boundaries, DEM sampling, elevation grids.",
        "usgs_water": "Stream gauges and groundwater wells. Needs an API key — "
                      "without one it 429s and a basin silently loses its pins.",
        "snotel":     "Snow water equivalent at SNOTEL stations.",
        "ameriflux":  "Flux towers. Registration pending, so ET is offered and "
                      "never pinned.",
        "fan_wtd":    "Fan et al. 2013 water-table depth, as a prior.",
        "geology":    "Soil and geology characterisation at a point.",
        "reaction":   "PFLOTRAN reaction sandbox — not part of the ELM path.",
    }
    try:
        cfg = json.loads((ROOT / "mcp_config.json").read_text())["mcp_servers"]
    except Exception:                                           # noqa: BLE001
        return []
    out = []
    for name, spec in cfg.items():
        if name == "elm":
            continue
        args = [a for a in spec.get("args", []) if a.endswith(".py")]
        rel = args[0].split("multi-agent-framework/")[-1] if args else ""
        out.append({"n": name, "side": "data",
                    "sum": blurb.get(name, spec.get("description", "")),
                    "flow": ["Called by Reception to gather the basin and its "
                             "observations.",
                             "Returns DATA. It never decides what to sample and "
                             "never runs a model.",
                             "The observations it returns are what the Analyzer "
                             "will later compare against."],
                    "files": f(rel) if rel else []})
    return out


GROUPS = [
 {"name": "Reception", "note": "one sentence in, a brief out", "steps": [
  {"n": "Reception", "side": "fw",
   "sum": "Parse the request; gather the basin and every nearby station.",
   "flow": [ref("An LLM turns the sentence into a <b>brief</b>: basin, year, what is being asked.", "src/agents/reception_llm.py", "LLMReceptionAgent"),
            ref("Resolve the watershed polygon through the terrain MCP.", "src/core/data_gather.py", "gather_grid"),
            ref("Sample an elevation grid inside it.", "src/core/data_gather.py", "gather_grid"),
            ref("Fetch observations — gauges, wells, SNOTEL, flux towers.", "src/core/data_gather.py", "gather_observations"),
            ref("<b>Tag every station in or out of the basin.</b> Absent is not the same as outside: untested stations stay eligible, and conflating the two cost four basins their pinned columns.", "src/core/data_gather.py", "_tag_in_basin")],
   "files": f("src/agents/reception_llm.py", "src/core/data_gather.py",
              "src/agents/tool_loop.py")}]},

 {"name": "Data MCPs", "note": "observations and terrain, as data — they decide nothing",
  "steps": None},          # filled from mcp_config.json

 {"name": "Planner", "note": "the brief becomes a design", "steps": [
  {"n": "Planner", "side": "fw",
   "sum": "Decide how many columns, how to band them, what to pin.",
   "flow": [ref("Read the brief.", "src/agents/planner.py", "plan"),
            ref("Size the ensemble on <b>relief</b> — the simulated period does not enter this.", "src/agents/prompts/planner.txt", "BUDGET"),
            ref("Choose the elevation bands.", "src/agents/prompts/planner.txt", "bands"),
            ref("Choose which stations are worth pinning a column to.", "src/agents/prompts/planner.txt", "PIN"),
            ref("<b>Refuse a streamflow gauge.</b> A gauge integrates an upstream area that 1-D columns do not route, so it is never pinnable.", "src/agents/prompts/planner.txt", "NEVER"),
            ref("Strategy check, then emit the plan. No files yet.", "src/core/strategy_check.py", "check")],
   "files": f("src/agents/planner.py", "src/agents/prompts/planner.txt",
              "src/core/strategy_check.py")}]},

 {"name": "Experiment Manager", "note": "design → inputs → cases → run → results",
  "tag": "dissolving", "steps": [
  {"n": "Sample columns", "side": "fw", "sum": "The plan becomes real coordinates.",
   "flow": [ref("Clip the elevation grid to the watershed polygon.", "tools/expand_sampling.py", "expand"),
            ref("Equal-interval elevation bands.", "tools/expand_sampling.py", "_make_bands"),
            ref("Allocate columns per band, area-proportional with a floor of one.", "tools/expand_sampling.py", "_even_allocate"),
            ref("Farthest-point selection so points spread rather than clump.", "tools/expand_sampling.py", "_farthest_point_select"),
            ref("Pin columns at eligible stations.", "tools/expand_sampling.py", "_pinned_from_plan"),
            ref("Write <code>columns.json</code> — a <b>temporary input</b> to the MCP, not the final record.", "tools/expand_sampling.py", "expand")],
   "files": f("tools/expand_sampling.py", "tools/plot_sampling_design.py",
              "tools/figstyle.py")},
  {"n": "Build inputs", "side": "srv", "sum": "One call: six steps, ending in case_inputs.json.",
   "flow": [ref("<b>Warm start</b> — subset the CONUS restart per column, <b>snapping each to its donor gridcell</b>.", "mcp/elm-mcp/src/inputs.py", "warm_start"),
            ref("Donor soil — the column adopts that cell's profile. The restart names its own surfdata and the same indices slice both, so the two cannot disagree.", "mcp/elm-mcp/src/inputs.py", "attach_donor_soil"),
            ref("Run plan, returned to the framework, which persists it.", "mcp/elm-mcp/src/inputs.py", "to_run_plan"),
            ref("Surfaces and domains per column.", "mcp/elm-mcp/src/build_column_inputs.py", "build_all"),
            ref("Serialise to plain data — <code>runtime_config</code> carries every path the build needs.", "mcp/elm-mcp/src/inputs.py", "serialise_case_inputs"),
            ref("Write <code>case_inputs.json</code> and <code>elm_columns.json</code>, the final columns.", "mcp/elm-mcp/src/inputs.py", "write_case_inputs")],
   "flag": "This is the moment the columns move. Anything drawn from the sampled file shows a run that did not happen.",
   "files": f("mcp/elm-mcp/src/inputs.py", "mcp/elm-mcp/src/make_finidat_subset.py",
              "mcp/elm-mcp/src/build_column_inputs.py",
              "mcp/elm-mcp/src/elm_surface_generator.py")},
  {"n": "Build cases", "side": "srv", "sum": "The CIME compile. Job A, first half.",
   "flow": [ref("<code>create_newcase</code> for the reference column.", "mcp/elm-mcp/src/elm_wrapper.py", "_create_case"),
            ref("<code>xmlchange</code> the thirteen CIME keys, write the namelists.", "mcp/elm-mcp/src/elm_wrapper.py", "_configure_case"),
            ref("<code>case.setup</code>, then <code>case.build</code> — about 7½ minutes.", "mcp/elm-mcp/src/elm_wrapper.py", "_build_case"),
            ref("<code>create_clone --keepexe</code> for the rest, seconds each.", "mcp/elm-mcp/src/elm_wrapper.py", "_clone_case"),
            ref("Write <code>built_cases.json</code>.", "mcp/elm-mcp/scripts/ensemble_job.py", "main")],
   "flag": "A clone is left BUILD_COMPLETE=FALSE. Harmless today — the run path resolves EXEROOT itself — but the flag is not a usable readiness signal.",
   "files": f("mcp/elm-mcp/scripts/ensemble_job.py", "mcp/elm-mcp/src/elm_wrapper.py",
              "mcp/elm-mcp/src/elm_experiment_builder.py")},
  {"n": "Run", "side": "srv", "sum": "Every column concurrently. Job A, second half.",
   "flow": [ref("Read the case directories and resolve EXEROOT — a clone's executable lives in the reference case.", "mcp/elm-mcp/src/elm_wrapper.py", "run_simulation"),
            ref("<code>srun</code> each column concurrently, one task each.", "mcp/elm-mcp/scripts/ensemble_ab.sh", "srun"),
            ref("<b>Count history files rather than trusting the exit code.</b> A column that wrote nothing failed however srun exited.", "mcp/elm-mcp/scripts/ensemble_ab.sh", "N_OK"),
            ref("Report ENSEMBLE_DONE and stop. The analysis is job B's, submitted by the framework with --dependency=afterany.", "mcp/elm-mcp/main.py", "run_elm_ensemble")],
   "files": f("mcp/elm-mcp/scripts/ensemble_ab.sh", "mcp/elm-mcp/main.py")},
  {"n": "Extract", "side": "srv", "sum": "History files become rows of numbers.",
   "breach": True,
   "flow": [ref("Read each column's <code>*.elm.h0.*.nc</code>.", "mcp/elm-mcp/src/elm_results_analyzer.py", "ELMResultsAnalyzer"),
            ref("Pull the named series out — which variable means what is model knowledge.", "mcp/elm-mcp/src/elm_results_analyzer.py", "ELMResultsAnalyzer"),
            ref("Return rows.", "src/core/exp_manager_base.py", "_extract")],
   "flag": "Split today: the ELM reader is already MCP-side, the decision of when to run it is still framework stage machinery. Only that half moves.",
   "files": f("mcp/elm-mcp/src/elm_results_analyzer.py", "src/core/exp_manager_base.py")},
  {"n": "Package", "side": "fw", "sum": "Rows become the bundle the Analyzer reads.",
   "flow": [ref("Assemble rows into the standard result shape.", "src/core/exp_manager_base.py", "_package"),
            ref("Stays in the framework deliberately: that shape is <b>model-independent</b>, so one packaging format serves every model rather than one per server.", "src/core/exp_manager_base.py", "_package")],
   "files": f("src/core/exp_manager_base.py")}]},

 {"name": "Analyzer", "note": "results become an answer", "tag": "being redesigned",
  "steps": [
  {"n": "Compare to obs", "side": "fw", "sum": "Model against SNOTEL, wells, gauges.",
   "breach": True,
   "flow": [ref("Align the model series to each observation's time base.", "src/agents/analysis/step1_compare_streamflow.py", "gauge_daily"),
            ref("Metrics per variable — SWE, water table, runoff as specific discharge.", "src/agents/analysis/step1_compare_streamflow.py", "compare"),
            ref("A comparability verdict, which can be <b>refusal</b>: a gauge comparison is context-only until routing exists.", "src/agents/analysis/step1_compare_swe.py", "compare"),
            ref("Emit the caveats alongside the numbers.", "src/core/limitations.py", "select_limitations")],
   "flag": "Belongs in the MCP and has not moved. Knowing that QOVER + QDRAI is runoff is model knowledge, and so is the biggest caveat — columns generate locally, a gauge integrates upstream.",
   "files": f("src/agents/analysis/step1_compare_swe.py",
              "src/agents/analysis/step1_compare_streamflow.py",
              "src/agents/analysis/step1_compare_wtd.py")},
  {"n": "Interpret", "side": "fw", "sum": "What the numbers mean, and whether to trust them.",
   "flow": [ref("Build the context: results, assumptions ledger, and the limitations catalogue.", "src/agents/analysis/step0_context.py", "load"),
            ref("<b>Structural caveats are promoted to blocking</b> — they do not qualify a gauge comparison, they forbid the naive one.", "src/agents/analysis/step0_context.py", "_caveats"),
            ref("Derive and investigate, then interpret.", "src/agents/analysis/step2_derive.py", "driver_matrix"),
            ref("The LLM is told the catalogue binds it and that a refusal cannot be argued into evidence.", "src/agents/analyzer_agent.py", "AnalyzerAgent")],
   "files": f("src/agents/analyzer_agent.py", "src/agents/analysis/step0_context.py",
              "src/agents/analysis/step2_derive.py", "src/core/limitations.py")},
  {"n": "Report", "side": "fw", "sum": "The answer, with what is wrong with it attached.",
   "flow": [ref("Answer first, leading with the number that answers the question.", "src/agents/analysis/step3_interpret.py", "interpret"),
            ref("A Trust section that must acknowledge the caveats carried this far.", "src/agents/analysis/step3_interpret.py", "interpret"),
            ref("“This run cannot answer that, and here is what would” is a correct output.", "src/agents/analysis/step4_report.py", "build")],
   "files": f("src/agents/analysis/step3_interpret.py",
              "src/agents/analysis/step4_report.py")}]},
]

CSS = """
:root{--ground:#F1F4F3;--panel:#FFF;--ink:#141A1C;--ink-2:#43514F;--ink-3:#77837F;
--rule:#D5DEDA;--fw:#3A5B78;--fw-soft:#E7EEF4;--fw-line:#B6C7D6;--srv:#2C6A5C;
--srv-soft:#E1EEE9;--srv-line:#A9CCC1;--dat:#6B4A86;--dat-soft:#EDE7F3;
--dat-line:#C7B4D8;--breach:#A4522A;--breach-soft:#F7EADF;--breach-line:#E0BCA3;
--shadow:0 1px 2px rgba(20,26,28,.06),0 10px 28px -16px rgba(20,26,28,.22);
--serif:ui-serif,"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
--sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--ground:#0F1416;--panel:#161D1F;--ink:#E8EEEB;--ink-2:#A8B6B2;--ink-3:#75837F;
--rule:#26312F;--fw:#8FB4D4;--fw-soft:#17232E;--fw-line:#2E4356;--srv:#74C0AC;
--srv-soft:#132621;--srv-line:#28564A;--dat:#B79BD4;--dat-soft:#1E1826;
--dat-line:#413055;--breach:#DC9366;--breach-soft:#2B1C13;--breach-line:#5A3822;
--shadow:0 1px 2px rgba(0,0,0,.4),0 10px 28px -16px rgba(0,0,0,.7)}}
:root[data-theme="dark"]{--ground:#0F1416;--panel:#161D1F;--ink:#E8EEEB;
--ink-2:#A8B6B2;--ink-3:#75837F;--rule:#26312F;--fw:#8FB4D4;--fw-soft:#17232E;
--fw-line:#2E4356;--srv:#74C0AC;--srv-soft:#132621;--srv-line:#28564A;
--dat:#B79BD4;--dat-soft:#1E1826;--dat-line:#413055;--breach:#DC9366;
--breach-soft:#2B1C13;--breach-line:#5A3822;
--shadow:0 1px 2px rgba(0,0,0,.4),0 10px 28px -16px rgba(0,0,0,.7)}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);font-family:var(--sans);
line-height:1.55;margin:0;padding:clamp(1.2rem,4vw,3rem) clamp(1rem,4vw,2rem) 4rem}
.wrap{max-width:1060px;margin:0 auto}
.eyebrow{font-family:var(--mono);font-size:.7rem;letter-spacing:.14em;
text-transform:uppercase;color:var(--ink-3);margin:0 0 .6rem}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(1.9rem,4.4vw,2.9rem);
line-height:1.1;letter-spacing:-.015em;margin:0 0 .7rem;text-wrap:balance}
.lede{font-size:1.02rem;color:var(--ink-2);max-width:64ch;margin:0 0 1.5rem}
.lede b{color:var(--ink);font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:.45rem 1.1rem;margin:0 0 2rem;
font-size:.8rem;color:var(--ink-2)}
.key{display:inline-flex;align-items:center;gap:.45rem}
.dot{width:.62rem;height:.62rem;border-radius:2px;flex:none}
.group{margin-bottom:1.5rem;border:1px solid var(--rule);border-radius:11px;
background:var(--panel);box-shadow:var(--shadow);overflow:hidden}
.g-head{display:flex;flex-wrap:wrap;align-items:baseline;gap:.5rem .8rem;
padding:.85rem 1.1rem;border-bottom:1px solid var(--rule)}
.g-head h2{font-family:var(--serif);font-size:1.16rem;font-weight:600;margin:0;
letter-spacing:-.01em}
.g-head .g-note{font-size:.82rem;color:var(--ink-3);flex:1 1 18ch;min-width:0}
.g-tag{font-family:var(--mono);font-size:.6rem;letter-spacing:.09em;
text-transform:uppercase;padding:.18rem .42rem;border-radius:4px;
border:1px solid var(--breach-line);color:var(--breach);background:var(--breach-soft)}
.steps{display:flex;flex-wrap:wrap;gap:.5rem;padding:.9rem 1.1rem 1.1rem}
.step{flex:1 1 11rem;min-width:11rem;border:1px solid var(--rule);
border-radius:8px;background:var(--ground);overflow:hidden}
.step[data-side="fw"]{border-color:var(--fw-line);background:var(--fw-soft)}
.step[data-side="srv"]{border-color:var(--srv-line);background:var(--srv-soft)}
.step[data-side="data"]{border-color:var(--dat-line);background:var(--dat-soft)}
.step[data-breach="1"]{border-color:var(--breach-line);background:var(--breach-soft)}
.s-btn{appearance:none;font:inherit;color:inherit;background:none;border:0;
width:100%;text-align:left;cursor:pointer;padding:.6rem .7rem;display:flex;
flex-direction:column;gap:.22rem}
.s-btn:focus-visible{outline:2px solid var(--fw);outline-offset:-2px}
.s-top{display:flex;align-items:center;justify-content:space-between;gap:.5rem}
.s-name{font-weight:600;font-size:.87rem;line-height:1.25}
.s-where{font-family:var(--mono);font-size:.6rem;letter-spacing:.07em;
text-transform:uppercase;color:var(--ink-3);flex:none}
.chev{flex:none;width:.6rem;height:.6rem;border-right:1.6px solid var(--ink-3);
border-bottom:1.6px solid var(--ink-3);transform:rotate(45deg);
transition:transform .16s ease;margin-top:-.2rem}
.step.open .chev{transform:rotate(-135deg);margin-top:.15rem}
.s-sum{font-size:.78rem;color:var(--ink-2);line-height:1.4}
.s-body{display:none;padding:0 .7rem .75rem;border-top:1px dashed var(--rule);margin-top:.1rem}
.step.open .s-body{display:block}
.s-body h4{font-family:var(--mono);font-size:.62rem;letter-spacing:.11em;
text-transform:uppercase;color:var(--ink-3);margin:.75rem 0 .35rem;font-weight:500}
ol.flow{list-style:none;counter-reset:f;padding:0;margin:0;display:grid;gap:.3rem}
ol.flow li{counter-increment:f;position:relative;padding-left:1.5rem;
font-size:.78rem;color:var(--ink-2);line-height:1.42}
ol.flow li::before{content:counter(f);position:absolute;left:0;top:.05rem;
font-family:var(--mono);font-size:.6rem;color:var(--ink-3);border:1px solid var(--rule);
border-radius:3px;width:1.05rem;height:1.05rem;display:grid;place-items:center;
background:var(--panel)}
ol.flow li b{color:var(--ink);font-weight:600}
ul.files{list-style:none;padding:0;margin:0;display:grid;gap:.22rem}
ul.files li{display:flex;justify-content:space-between;gap:.7rem;align-items:baseline;
font-family:var(--mono);font-size:.7rem;padding:.26rem .45rem;border:1px solid var(--rule);
border-radius:5px;background:var(--panel)}
ul.files .fp{overflow-wrap:anywhere;color:var(--ink)}
ul.files .fl{color:var(--ink-3);font-variant-numeric:tabular-nums;flex:none}
.ref{display:block;font-family:var(--mono);font-size:.66rem;color:var(--ink-3);
margin-top:.15rem;overflow-wrap:anywhere}
.flag{font-size:.75rem;color:var(--breach);border-left:2px solid var(--breach);
padding:.35rem .55rem;background:var(--panel);border-radius:0 4px 4px 0;
margin-top:.6rem;line-height:1.4}
.foot{margin-top:2rem;padding-top:1rem;border-top:1px solid var(--rule);
color:var(--ink-3);font-size:.78rem;max-width:70ch}
.foot code{font-family:var(--mono);font-size:.74rem;color:var(--ink-2)}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

JS = """
const WHERE={fw:"framework",srv:"ELM MCP",data:"data MCP"};
const host=document.getElementById("groups");
G.forEach(g=>{
  const el=document.createElement("section");el.className="group";
  el.innerHTML=`<div class="g-head"><h2>${g.name}</h2>`+
    (g.tag?`<span class="g-tag">${g.tag}</span>`:"")+
    `<span class="g-note">${g.note}</span></div><div class="steps"></div>`;
  const row=el.querySelector(".steps");
  (g.steps||[]).forEach(s=>{
    const d=document.createElement("div");
    d.className="step";d.dataset.side=s.side;
    if(s.breach)d.dataset.breach="1";
    d.innerHTML=`<button class="s-btn" aria-expanded="false">
        <span class="s-top"><span class="s-name">${s.n}</span><span class="chev"></span></span>
        <span class="s-where">${WHERE[s.side]||s.side}${s.breach?" · needs moving":""}</span>
        <span class="s-sum">${s.sum}</span></button>
      <div class="s-body"><h4>Inside this stage</h4>
        <ol class="flow">${(s.flow||[]).map(x=>Array.isArray(x)
            ?`<li>${x[0]}<span class="ref">${x[1]}</span></li>`:`<li>${x}</li>`).join("")}</ol>`+
        (s.flag?`<div class="flag">${s.flag}</div>`:"")+
        `<h4>Source</h4><ul class="files">${(s.files||[]).map(x=>
          `<li><span class="fp">${x[0]}</span><span class="fl">${x[1]?x[1]+" lines":"—"}</span></li>`
        ).join("")}</ul></div>`;
    const b=d.querySelector(".s-btn");
    b.addEventListener("click",()=>{
      const open=d.classList.toggle("open");
      b.setAttribute("aria-expanded",open?"true":"false");});
    row.appendChild(d);});
  host.appendChild(el);});
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="workflow_outputs/arch_map.html")
    a = ap.parse_args()

    groups = [dict(g) for g in GROUPS]
    for g in groups:
        if g["steps"] is None:
            g["steps"] = data_servers()

    missing = [p for g in groups for s in g["steps"] for p, n in s["files"] if not n]
    stale = [w for g in groups for s in g["steps"] for r in (s.get("flow") or [])
             if (w := check_ref(r))]
    html = (f"<title>IDEAS pipeline — agents, stages, and what happens inside each</title>\n"
            f"<style>{CSS}</style>\n"
            f'<div class="wrap">\n'
            f'  <p class="eyebrow">IDEAS · multi-agent-framework · generated from the source</p>\n'
            f"  <h1>Agents, stages, and what happens inside each</h1>\n"
            f'  <p class="lede">Five groups own the run end to end. Each stage is coloured by\n'
            f"    <b>where it executes</b> — a different question from who orchestrates it.\n"
            f"    Click any stage for its internal workflow and its source.</p>\n"
            f'  <div class="legend">'
            f'<span class="key"><span class="dot" style="background:var(--fw)"></span>Framework</span>'
            f'<span class="key"><span class="dot" style="background:var(--dat)"></span>Data MCP</span>'
            f'<span class="key"><span class="dot" style="background:var(--srv)"></span>ELM MCP</span>'
            f'<span class="key"><span class="dot" style="background:var(--breach)"></span>Not yet where it belongs</span>'
            f"</div>\n"
            f'  <div id="groups"></div>\n'
            f'  <p class="foot">Regenerate with <code>python3 tools/make_arch_map.py</code>.\n'
            f"    File paths, line counts and the registered MCP servers are read from the\n"
            f"    source on every run; the step-by-step descriptions inside each stage are\n"
            f"    written by hand in that script and are the part that can drift.</p>\n"
            f"</div>\n<script>\nconst G = {json.dumps(groups, ensure_ascii=False)};\n{JS}</script>\n")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    n_steps = sum(len(g["steps"]) for g in groups)
    print(f"wrote {out}  ({len(groups)} groups, {n_steps} stages, "
          f"{sum(len(s['files']) for g in groups for s in g['steps'])} files)")
    n_ref = sum(1 for g in groups for s in g["steps"]
                for r in (s.get("flow") or []) if isinstance(r, list))
    print(f"   {n_ref} step(s) reference a function; {len(stale)} stale")
    for p in missing:
        print(f"   ⚠️  listed but not on disk: {p}")
    for w in stale:
        print(f"   ⚠️  stale reference: {w}")


if __name__ == "__main__":
    main()
