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
    # Absolute paths pass through: the reaction server is installed OUTSIDE the
    # repo, and reporting it as a 0-line file would read as "missing".
    p = Path(rel) if Path(rel).is_absolute() else ROOT / rel
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


# WHICH SERVERS RUN A MODEL. The split is not cosmetic: a data server answers
# "what is true at this place", and a model server "what happens if we simulate
# it". `reaction` sat in the data group for months because the grouping was
# "everything except elm", which is not a category — it is a leftover. It is
# PFLOTRAN's reaction sandbox: it writes decks and runs the model.
MODEL_SERVERS = {"elm", "reaction"}

# The two model servers get hand-written detail rather than the generic card the
# data servers get, so the paths are named once here. RXN is ABSOLUTE: the
# reaction server is installed into the conda env from a source tree outside
# this repo, which is a fact about the project worth showing rather than hiding.
ELM_MAIN = "mcp/elm-mcp/main.py"
RXN = "/qfs/people/tran289/IDEAS/reaction_sandbox_mcp-upstream/server.py"
# The compare package, named once: it is one directory with seven files and the
# per-observable sub-boxes reference it on nearly every line.
CMP = "mcp/elm-mcp/src/compare/"

# Display names, where the registered key is not what the thing is called.
SERVER_TITLE = {"reaction": "pflotran (reaction)"}


def _entry_point(spec) -> str:
    """The server's source file, for a config entry that may not name one.

    Most servers are `python <path>/main.py` and the path is right there. The
    reaction server is a console script installed into the conda env, so the
    config names a wrapper; follow it to the module it imports. Returned
    ABSOLUTE in that case, because it lives outside this repo — which is worth
    showing rather than hiding, since it is the one server whose source is not
    version-controlled here.
    """
    args = [a for a in spec.get("args", []) if a.endswith(".py")]
    if args:
        return args[0].split("multi-agent-framework/")[-1]
    cmd = spec.get("command", "")
    if not cmd or not Path(cmd).is_file():
        return ""
    try:                                    # `from server import main`
        import re as _re
        mod = _re.search(r"^from\s+(\w+)\s+import", Path(cmd).read_text(
            errors="ignore"), _re.M)
        if not mod:
            return ""
        import importlib.util
        s = importlib.util.find_spec(mod.group(1))
        return s.origin if s and s.origin else ""
    except Exception:                                           # noqa: BLE001
        return ""

BLURB = {
    "terrain":    "Watershed boundaries, DEM sampling, elevation grids.",
    "usgs_water": "Stream gauges and groundwater wells. Needs an API key — "
                  "without one it 429s and a basin silently loses its pins. "
                  "Every observation carries its OWN coordinates. The water "
                  "table is RECORDER WELLS ONLY — a daily series, never a "
                  "single visit — so an empty list means no logger, not no "
                  "wells, and the reason travels with it. min_days TAGS a "
                  "short gauge; min_obs on a well DROPS it (2026-08-12).",
    "snotel":     "Snow water equivalent at SNOTEL stations. Fetched on the "
                  "CALENDAR year like everything else since 2026-08-12 — the "
                  "water year returned months the run never simulated.",
    "ameriflux":  "Flux towers. Site discovery works; the flux SERIES needs a "
                  "registered account, so ET is offered and never pinned.",
    "hydrodata":  "Water table from HydroFrame, two kinds, never merged: the "
                  "ParFlow CONUS2 settled state — MODELLED, 1 km, every land "
                  "cell — written once per basin as a GeoTIFF; and Fan et al. "
                  "2013 — MEASURED, well sites with one long-term mean each. "
                  "Per-point tools and ma_2025 were deleted 2026-08-12: this "
                  "is a university's server and it gets ONE request per "
                  "basin. Catalogue is open; data needs a PIN.",
    "geology":    "Soil and geology characterisation at a point.",
    "reaction":   "PFLOTRAN reaction sandbox: builds decks, runs 1-D reactive "
                  "transport, and runs the LAMBDA network. Not on the ELM path.",
    "elm":        "E3SM Land Model: warm start, donor soil, surfaces, the CIME "
                  "build, and the ensemble. The whole simulation, end to end.",
}

DATA_FLOW = ["Called by Reception to gather the basin and its observations.",
             "Returns DATA. It never decides what to sample and never runs a "
             "model.",
             "The observations it returns are what the Analyzer will later "
             "compare against."]

MODEL_FLOW = ["Given what to simulate, owns HOW the model produces it.",
              "Returns data or a job id — never a blocking call on the science.",
              "Knows the model's file formats and physics; knows nothing about "
              "the study design that asked for the run."]


def _servers(want_models: bool):
    """Registered MCP servers of one kind, read from mcp_config.json.

    Read rather than listed: a server added to the config appears on the map
    without anyone remembering to update a diagram. Only the CLASSIFICATION is
    hand-maintained, in MODEL_SERVERS above — an unclassified server shows up
    as a data server, which is the safe default (it claims less).
    """
    try:
        cfg = json.loads((ROOT / "mcp_config.json").read_text())["mcp_servers"]
    except Exception:                                           # noqa: BLE001
        return []
    out = []
    for name, spec in cfg.items():
        is_model = name in MODEL_SERVERS
        if is_model != want_models:
            continue
        rel = _entry_point(spec)
        out.append({"n": SERVER_TITLE.get(name, name),
                    "side": "srv" if is_model else "data",
                    "sum": BLURB.get(name, spec.get("description", "")),
                    "flow": MODEL_FLOW if is_model else DATA_FLOW,
                    "files": f(rel) if rel else []})
    return out


GROUPS = [
 {"name": "Reception", "note": "one sentence in, a brief out", "steps": [
  {"n": "Reception", "side": "fw",
   "sum": "Parse the request; gather the basin and every nearby station.",
   "flow": [ref("An LLM turns the sentence into a <b>brief</b>: basin, year, what is being asked. <b>Two tools only.</b> Everything else is fetched by code afterwards, because what to fetch is not a decision — with twelve tools the loop spent most of its time choosing them and then summarised a truncated result.", "src/agents/reception_llm.py", "LLMReceptionAgent"),
            ref("Resolve the watershed polygon, then sample an elevation grid inside it. <b>Terrain and nothing else</b> — a function named for the grid returns only the grid.", "src/core/data_gather.py", "gather_grid"),
            ref("Fetch the observations — gauges, recorder wells, Fan's well means, SNOTEL, flux towers. <b>One calendar period for all five</b>: snow used to be asked for on the water year, which returned months the run never simulated.", "src/core/data_gather.py", "gather_observations"),
            ref("<b>How complete a record must be is a SHARE of the period</b>, never a fixed count — 90% for a gauge, 30% for a well. 300 days was 82% of a one-year study and 41% of a two-year one, so the bar fell quietly as studies got longer.", "src/core/data_gather.py", "_period_days"),
            ref("<b>Each observation carries the point it was measured at.</b> Read from the record itself, not looked up in a site catalogue — that lookup was capped at 500 of 5,312 wells and cost Naches 236 of 251 well coordinates, so a basin with 56 in-basin wells reported none (fixed 2026-08-11).", "src/core/data_gather.py", "gather_observations"),
            ref("The modelled water table arrives as a <b>GeoTIFF written once</b>, not as values at points. Sampling it per column was 116 requests to a university's server, and could only answer at points the sampler was about to move when it snapped columns to the CONUS grid.", "src/core/static_wtd.py", "sample"),
            ref("<b>Tag every station in or out of the basin.</b> Absent is not the same as outside: untested stations stay eligible, and conflating the two cost four basins their pinned columns.", "src/core/data_gather.py", "_tag_in_basin"),
            ref("<b>The planner is handed a summary, never the series</b> — 0.9 KB against 170 kB — and never the fetch thresholds, because it is not the one choosing them.", "src/core/data_gather.py", "summarise")],
   "flag": "A lat/lon bounding box is a ROTATED quadrilateral on the CONUS2 grid — about 17° at Naches. Building the raster window from two corners left the basin's other two corners outside it, and any column there read back as 'no water table': a plausible absence with a projection behind it (fixed 2026-08-12, window now walks the whole perimeter).",
   "files": f("src/agents/reception_llm.py", "src/core/data_gather.py",
              "src/core/static_wtd.py", "src/agents/tool_loop.py")}]},

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
   "files": f("tools/expand_sampling.py", "mcp/elm-mcp/src/sampling_design.py",
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
  {"n": "Submit A + B", "side": "fw", "sum": "Two sbatch calls, then the framework exits.",
   "flow": [ref("Ask the server for job A — build and run, one job.", "mcp/elm-mcp/main.py", "run_elm_ensemble"),
            ref("Submit <b>job B</b> here, with <code>--dependency=afterany</code> on A. <b>afterany, never afterok</b>: with afterok a failed ensemble means B never runs and no mail is ever sent.", "mcp/elm-mcp/src/elm_exp_manager.py", "_submit_job_b"),
            ref("Record the job id in <code>run_state.json</code> and stop. That file is the only thing that reaches job B — a different process, on a different node, hours later.", "src/core/exp_manager_base.py", "_advance"),
            ref("Refuse a node-local run directory. A compute node cannot see this node's <code>/tmp</code>, and both jobs die in two seconds with empty logs.", "mcp/elm-mcp/src/elm_exp_manager.py", "_submit_job_b")],
   "flag": "B is submitted by the FRAMEWORK, not by A. The tool this replaced ended its job by running workflow.py --finalize — the server executing its client's code.",
   "files": f("mcp/elm-mcp/src/elm_exp_manager.py", "src/core/exp_manager_base.py")},
  {"n": "Build cases", "side": "srv", "sum": "The CIME compile. Job A, first half.",
   "flow": [ref("Reuse the existing build only if <code>built_cases.json</code> is ok <b>and every case directory it names is still on disk</b> — one missing directory condemns the manifest.", "mcp/elm-mcp/scripts/ensemble_ab.sh", "reusable"),
            ref("<code>create_newcase</code> for the reference column.", "mcp/elm-mcp/src/elm_wrapper.py", "_create_case"),
            ref("<code>xmlchange</code> the thirteen CIME keys, write the namelists.", "mcp/elm-mcp/src/elm_wrapper.py", "_configure_case"),
            ref("<code>case.setup</code>, then <code>case.build</code> — about 7½ minutes.", "mcp/elm-mcp/src/elm_wrapper.py", "_build_case"),
            ref("<code>create_clone --keepexe</code> for the rest, seconds each.", "mcp/elm-mcp/src/elm_wrapper.py", "_clone_case"),
            ref("Write <code>built_cases.json</code>.", "mcp/elm-mcp/scripts/ensemble_job.py", "main"),
            ref("Draw <code>column_surfaces.png</code> — the soil each column ACTUALLY got, read through the case's own <code>run/lnd_in</code>. It can only happen here: that file does not exist until the case is built.", "mcp/elm-mcp/scripts/ensemble_job.py", "_plot_setups")],
   "flag": "A clone is left BUILD_COMPLETE=FALSE. Harmless today — the run path resolves EXEROOT itself — but the flag is not a usable readiness signal.",
   "files": f("mcp/elm-mcp/scripts/ensemble_job.py", "mcp/elm-mcp/src/elm_wrapper.py",
              "mcp/elm-mcp/src/elm_experiment_builder.py")},
  {"n": "Run", "side": "srv", "sum": "Every column concurrently. Job A, second half.",
   "flow": [ref("Read the case directories and resolve EXEROOT — a clone's executable lives in the reference case.", "mcp/elm-mcp/src/elm_wrapper.py", "run_simulation"),
            ref("<code>srun</code> each column concurrently, one task each.", "mcp/elm-mcp/scripts/ensemble_ab.sh", "srun"),
            ref("<b>Count history files rather than trusting the exit code.</b> A column that wrote nothing failed however srun exited.", "mcp/elm-mcp/scripts/ensemble_ab.sh", "N_OK"),
            ref("Report ENSEMBLE_DONE and stop. The analysis is job B's.", "mcp/elm-mcp/main.py", "run_elm_ensemble")],
   "files": f("mcp/elm-mcp/scripts/ensemble_ab.sh", "mcp/elm-mcp/main.py")},
  {"n": "Collect", "side": "fw", "sum": "What landed, from the filesystem. Job B.",
   "flow": [ref("Ask the scheduler whether A has finished; the built case directories come back inline with the answer.", "mcp/elm-mcp/main.py", "check_elm_job"),
            ref("<b>A column succeeded if ELM wrote it a history file.</b> The scheduler being done is what ends the wait; the files decide the outcome.", "mcp/elm-mcp/src/elm_exp_manager.py", "_outcomes_from_disk"),
            ref("Write <code>execution_report.txt</code> and <code>results_summary.csv</code>.", "mcp/elm-mcp/src/elm_exp_manager.py", "_collect")],
   "flag": "This stage NEVER submits. It used to, and a resumed study re-ran every column over the top of the history files the first pass wrote (fixed 2026-08-10).",
   "files": f("mcp/elm-mcp/src/elm_exp_manager.py", "tools/notify_study.py")},
  {"n": "Extract", "side": "srv", "sum": "History files become rows of numbers.",
   "breach": True, "tag": "being replaced",
   "flow": [ref("<b>ONE reader of the history files.</b> Daily series per column, layers kept, no derived ratios, written to <code>03_results/extracted.json</code>. Finding the files, opening them and trimming the warm start live here and nowhere else — the results class kept its own copy of all three until 2026-08-13.", "mcp/elm-mcp/src/extract.py", "extract_column"),
            ref("<b>A day with one timestep is not a day.</b> ELM stamps a history record at the END of its window, so the final average of 31 December lands on 1 January with an eighth of the sampling — and a daily resample weighted it like any other day. Dropping it is what makes the published series and the raw record agree.", "mcp/elm-mcp/src/extract.py", "_drop_partial_days"),
            ref("<b>EVERY computation is downstream of the extraction</b> (the user's rule, 2026-08-13). Annual means, the water budget and the per-variable reductions are derived from <code>extracted.json</code> and from nothing else — so anyone holding the run directory can recompute them. Computed while the NetCDF was open, they stopped being checkable the moment scratch was purged.", "mcp/elm-mcp/src/column_metrics.py", "column_metrics"),
            ref("Normalise whatever the backend built into the stage's JSON contract — rows, units, extra_summary. <b>Model-AGNOSTIC and shared with PFLOTRAN</b>, which is why it is framework-side and stays there.", "src/core/exp_manager_base.py", "_as_extract"),
            ref("Attach the honesty payload: applicable limitations + the assumptions ledger.", "mcp/elm-mcp/src/elm_exp_manager.py", "_extract")],
   "flag": "The breach is NOT the row-shaping — that is shared glue and belongs where it is. It is the backwards imports: elm_exp_manager pulls core.limitations, analyze_agentic and analyze_run pull agents.analysis, and the class still inherits ExperimentManagerBase. limitations.py is LOAD-BEARING, not a leftover: elm_exp_manager._extract fills extra_summary with it, _ensemble_blocks folds that into experiment.json, and step0_context turns it into the blocking and qualifying caveats that bind the interpreter. Deleting it would silently remove every caveat from every run. Moving extract behind a tool call was proposed 2026-08-06 and REJECTED: FIELD_SEMANTICS would end up on the far side of the boundary.",
   "files": f("mcp/elm-mcp/src/extract.py", "mcp/elm-mcp/src/column_metrics.py",
              "mcp/elm-mcp/src/column_rows.py",
              "src/core/exp_manager_base.py")},
  {"n": "Package", "side": "fw", "sum": "Rows become the bundle the Analyzer reads.",
   "flow": [ref("Assemble rows into the standard result shape.", "src/core/exp_manager_base.py", "_package"),
            ref("Stays in the framework deliberately: that shape is <b>model-independent</b>, so one packaging format serves every model rather than one per server.", "src/core/exp_manager_base.py", "_package")],
   "files": f("src/core/exp_manager_base.py")}]},

 {"name": "ELM MCP", "note": "8 tools · 15 modules · 8 scripts — the whole ELM simulation",
  "steps": [
  {"n": "Tools", "side": "srv", "sum": "The eight calls the framework may make.",
   "flow": [ref("<code>describe_elm_capabilities()</code> — the workflow, a CHECKED inventory of every external dependency, and what this server does not do. Exhaustive, and a test enforces that against the registry.", ELM_MAIN, "describe_elm_capabilities"),
            ref("<code>build_elm_inputs_from_location(run_dir, columns, …)</code> — six steps in ONE call, returning the SNAPPED columns and writing <code>case_inputs.json</code>.", ELM_MAIN, "build_elm_inputs_from_location"),
            ref("<code>get_column_metadata(run_dir)</code> — the columns as they will be RUN. Ask here, not from the columns.json you sampled: that one is what you ASKED FOR.", ELM_MAIN, "get_column_metadata"),
            ref("<code>run_elm_ensemble(run_dir, …)</code> — JOB A. Build every case and run every column, then stop. Returns a job id in seconds.", ELM_MAIN, "run_elm_ensemble"),
            ref("<code>build_elm_cases(run_dir, …)</code> — the build ALONE, for inspecting cases before spending node time. run_elm_ensemble does this too.", ELM_MAIN, "build_elm_cases"),
            ref("<code>check_elm_job(job_id, run_dir)</code> — what SLURM is doing, plus the built case directories once a build lands.", ELM_MAIN, "check_elm_job"),
            ref("<code>extract_elm_output(run_dir, …)</code> — the history files become daily series in <code>03_results/extracted.json</code>. RAW SERIES ONLY; no derived ratios cross this boundary. Added 2026-08-11.", ELM_MAIN, "extract_elm_output"),
            ref("<code>compare_to_obs(run_dir, …)</code> — model against observations for swe / water_table / streamflow / et. <b>Measurements, never a verdict.</b>", ELM_MAIN, "compare_to_obs")],
   "flag": "Every tool returns quickly or returns a job id. An MCP client opens a fresh session per call and tearing it down kills this server's children — so a 10-minute CIME build inside a call is not a slow call, it is a half-built case directory.",
   "files": f("mcp/elm-mcp/main.py", "mcp/elm-mcp/src/paths.py")},
  {"n": "Input build", "side": "srv", "sum": "Locations in → a runnable case list out.",
   "flow": [ref("Snap each column to its CONUS donor gridcell and subset the restart → per-column <code>finidat</code>.", "mcp/elm-mcp/src/inputs.py", "warm_start"),
            ref("Take the donor cell's soil. The restart NAMES its own surfdata, and the same ixy/jxy slice both, so the two cannot disagree.", "mcp/elm-mcp/src/inputs.py", "attach_donor_soil"),
            ref("Cut the per-column initial state out of the CONUS 1 km restart.", "mcp/elm-mcp/src/make_finidat_subset.py", "main"),
            ref("Generate each column's surface file from the CONUS surfdata.", "mcp/elm-mcp/src/elm_surface_generator.py", "ELMSurfaceGenerator"),
            ref("Generate each column's domain file.", "mcp/elm-mcp/src/elm_domain_generator.py", "ELMDomainGenerator"),
            ref("Turn columns into the executable plan — one coupler per column, each with its OWN lat/lon.", "mcp/elm-mcp/src/columns_to_plan.py", "columns_to_elm_plan"),
            ref("Serialise to plain data: <code>runtime_config</code> carries every path the build needs.", "mcp/elm-mcp/src/inputs.py", "serialise_case_inputs"),
            ref("Write <code>case_inputs.json</code> + <code>elm_columns.json</code>, the final columns.", "mcp/elm-mcp/src/inputs.py", "write_case_inputs")],
   "flag": "This is the moment the columns MOVE. Anything drawn from the sampled file afterwards shows a run that did not happen.",
   "files": f("mcp/elm-mcp/src/inputs.py", "mcp/elm-mcp/src/make_finidat_subset.py",
              "mcp/elm-mcp/src/make_warmstart.py", "mcp/elm-mcp/src/build_column_inputs.py",
              "mcp/elm-mcp/src/elm_surface_generator.py",
              "mcp/elm-mcp/src/elm_domain_generator.py",
              "mcp/elm-mcp/src/columns_to_plan.py")},
  {"n": "Case build + run", "side": "srv", "sum": "CIME compile, clone, then every column. Job A.",
   "flow": [ref("Reuse the existing build only if <code>built_cases.json</code> is ok AND every case directory it names still exists.", "mcp/elm-mcp/scripts/ensemble_ab.sh", "reusable"),
            ref("<code>create_newcase</code> → <code>xmlchange</code> the thirteen CIME keys → namelists.", "mcp/elm-mcp/src/elm_wrapper.py", "_configure_case"),
            ref("<code>case.setup</code>, then <code>case.build</code> — about 7½ minutes for the reference column.", "mcp/elm-mcp/src/elm_wrapper.py", "_build_case"),
            ref("<code>create_clone --keepexe</code> for the rest, seconds each, with a serial retry pass for the known parallel-filesystem race.", "mcp/elm-mcp/src/elm_experiment_builder.py", "build_cases"),
            ref("Write <code>built_cases.json</code>, then draw <code>column_surfaces.png</code> from each case's GENERATED fsurdat.", "mcp/elm-mcp/scripts/ensemble_job.py", "_plot_setups"),
            ref("<code>srun</code> every column concurrently, one task each, <code>--exclusive</code> not <code>--exact</code> (Slurm 18.08).", "mcp/elm-mcp/scripts/ensemble_ab.sh", "srun"),
            ref("Count history files rather than trusting the exit code. A column that wrote nothing failed however srun exited.", "mcp/elm-mcp/scripts/ensemble_ab.sh", "N_OK")],
   "flag": "A clone is left BUILD_COMPLETE=FALSE. Harmless today — the run path resolves EXEROOT itself — but the flag is not a usable readiness signal.",
   "files": f("mcp/elm-mcp/scripts/ensemble_ab.sh", "mcp/elm-mcp/scripts/ensemble_job.py",
              "mcp/elm-mcp/scripts/build_cases.py", "mcp/elm-mcp/src/elm_wrapper.py",
              "mcp/elm-mcp/src/elm_experiment_builder.py",
              "mcp/elm-mcp/src/elm_input_agent.py")},
  {"n": "Compare to obs", "side": "srv", "sum": "Model beside observations. Numbers only.",
   "flow": [ref("Reuse <code>03_results/extracted.json</code> when it is already there; call the extract tool when it is not. ONE artifact, written by every path that reads history files.", ELM_MAIN, "_extracted_rows"),
            ref("Read the observations out of <b>reception.json in the shape reception wrote it</b> — the only source. A tool that re-fetched them into a CSV returned 14,102 values identical to what was already on disk, and was deleted. Unparseable rows are DROPPED AND COUNTED, never coerced to zero.", CMP + "_common.py", "load_observations"),
            ref("<b>One module per observable</b>, each a SPEC + compare() + plot(); this is pure dispatch, so a fifth observable is a file and one registry line, never a branch in shared code.", CMP + "__init__.py", "compare_all"),
            ref("<b>Outside the divide, out of the comparison</b> — applying reception's tag, never recomputing it. Only an explicit False excludes: unchecked is not outside, and refusing untested stations cost two basins all four pinned columns when one elevation request timed out. Each module applies it now: <code>standard_compare</code>, which used to, is deleted.", CMP + "streamflow.py", "compare"),
            ref("<b>A designed PIN wins outright</b> — where the sampler placed a column at a station, that pairing is honoured with no distance or elevation test. Re-deriving it geometrically was guessing at an answer already recorded in <code>columns.json</code>.", CMP + "_common.py", "pair_stations"),
            ref("For the rest: a column qualifies on BOTH limits, then the CLOSEST wins. Elevation FILTERS, distance DECIDES. Limits are <b>per observable</b> and declared on the Spec — SWE is 150 m and 5 km. One pass over the columns in name order, and a station leaves the pool when taken, so col_01 has first refusal.", CMP + "_common.py", "pair_stations"),
            ref("An unpaired station carries the number that disqualified it, and <b>&quot;no coordinates&quot; is its own reason</b> — collapsing it into &quot;no column qualified&quot; is what let 236 coordinate-less wells read as a pairing failure.", CMP + "_common.py", "pair_stations"),
            ref("Then the numbers, the same for all four: inner-join model and observation <b>on the date, never interpolated</b>, and report bias / MAE / RMSE / r / NSE / KGE with the measured-versus-gap-filled split beside them.", CMP + "_common.py", "metrics")],
   "subs_label": "One module per observable",
   "subs": [
     {"n": "swe", "sum": "H2OSNO against a snow pillow. MATCHES FIRST, then compares.",
      "flow": [ref("<b>Match, then compare</b> — the only observable that does, so far. Every station used to be compared against every column: 8,772 paired days on a 17-column Naches run to keep 516. The cost was never the arithmetic, it was that the record then carried 34 complete metric sets beside one &quot;assignment&quot;, and the flattering one is always in there somewhere.", CMP + "swe.py", "compare"),
               ref("<b>The season's timing, both sides, never differenced here</b>: peak, peak day-of-water-year, first snow, melt-out, days above 25 mm. A thin pack lasting months and a deep one melting fast share a mean, and bias cannot say which you have.", CMP + "swe.py", "snow_season_timing"),
               ref("<b>The timing belongs to ONE season</b> — the one holding the peak, split where the ground is bare for over 30 days. Reception fetches SWE on the calendar year now, so a series can hold a melt limb AND the next winter accumulating; melt-out was the last above-threshold day of the WHOLE series, which in a full-year record lands just before New Year. A 7-day midwinter thaw must not split a season, and does not.", CMP + "swe.py", "_seasons"),
               ref("Day-of-water-year, Oct 1 = 1. A calendar day-of-year puts a 15 December peak at 349 and a 5 January peak at 5 — adjacent events 344 apart, which makes the timing panel unreadable.", CMP + "swe.py", "day_of_water_year"),
               ref("Both sides clipped to the window they SHARE before any metric. A run can start mid-year, as Gunnison's did on 2019-01-15, and a mean over two different denominators is not a comparison.", CMP + "swe.py", "shared_window"),
               ref("Three panels: every column with the matched ones in their station's colour, every matched day against 1:1, and peak AND melt-out timing together. All 17 columns stay on the figure — at Naches 2 of them got a station, so the other 15 are the study and the observations are a thin check on a corner of it.", CMP + "swe.py", "plot")],
      "flag": "Onset is now censored in every snow basin, and that is the honest answer rather than a loss: ELM was not running the previous November, so this run cannot observe its own snow onset. The old water-year fetch reported an observed onset date with no model counterpart.",
      "files": f(CMP + "swe.py")},
     {"n": "water_table", "sum": "ZWT against recorder wells, and against two static fields.",
      "flow": [ref("Recorder wells differenced against the model, on a log axis, with the 3.8 m active-soil count.", CMP + "water_table.py", "compare"),
               ref("<b>Its `references` panel is STALE.</b> It still expects a per-column <code>{fan, parflow_clm}</code> pair and calls Fan &quot;an equilibrium surface&quot;. Fan is now WELL OBSERVATIONS with one long-term mean per site; the modelled field is a GeoTIFF read by static_wtd. Neither arrives in the shape this expects.", CMP + "water_table.py", "plot")],
      "files": f(CMP + "water_table.py")},
     {"n": "streamflow", "sum": "QOVER + QDRAI against a gauge. Context, never a score.",
      "flow": [ref("<b>The scale mismatch as a NUMBER</b>: a gauge's drainage area against the column's 1 m², in orders of magnitude. Nothing is rescaled by it.", CMP + "streamflow.py", "compare"),
               ref("A gauge integrates and routes an upstream area a 1-D column has no lateral transport to represent, so <code>colocated</code> is false and no metric here scores the model. The hydrograph SHAPE is still worth seeing.", CMP + "streamflow.py", "plot")],
      "files": f(CMP + "streamflow.py")},
     {"n": "et", "sum": "QSOIL + QVEGE + QVEGT against a flux tower.",
      "flow": [ref("<b>Metrics computed TWICE</b>, measured-only and all pairs, so a reader sees how much agreement came from the gap-filling model rather than the instrument. At US-NR1 in 2016 the gap-filled annual total is 464 mm against 89 mm from the measured half-hours alone.", CMP + "et.py", "compare")],
      "files": f(CMP + "et.py")},
     {"n": "map", "sum": "Where we are even looking. All observables, one figure.",
      "flow": [ref("Columns as grey points, stations coloured by observable and <b>ringed when matched</b>. A basin with every station in the valley and every column on the ridge is a design problem no metric surfaces.", CMP + "maps.py", "plot_all"),
               ref("Reads BOTH record shapes — swe's matched pairs and the other three's all-pairs-plus-assignment — so the map keeps working while the observables are converted one at a time.", CMP + "maps.py", "plot_all")],
      "flag": "No elevation and no basin outline. Snow pairs on ELEVATION and two columns adjacent on this map can be 1,000 m apart vertically. Subplots per variable were asked for 2026-08-12 and are still to come.",
      "files": f(CMP + "maps.py")}],
   "flag": "MEASUREMENTS, NOT VERDICTS — 'bias = -41 mm' is here, 'the model underestimates snowpack' is not. Every comparison is CONTEXT rather than a skill claim, which is what makes streamflow admissible at all. limitations.py is deliberately not used. THE FORCING RESOLUTION IS NOT A PAIRING CRITERION (removed 2026-08-11): one NLDAS-2 cell answers 'did these two points get the same rain', which is a fact about the atmosphere, not about what was measured. MATCH FIRST, THEN COMPARE — swe, water_table and et each match ONE column to ONE station and compare only those; streamflow matches NOTHING, because a gauge measures an area, and stands the ensemble mean against each gauge instead.",
   "files": f(CMP + "__init__.py", CMP + "_common.py",
              CMP + "swe.py", CMP + "water_table.py",
              CMP + "streamflow.py", CMP + "et.py",
              CMP + "maps.py")},
  {"n": "Results + figures", "side": "srv", "sum": "History NetCDFs → rows and plots.",
   "breach": True,
   "flow": [ref("Rows built FROM <code>extracted.json</code> — identity, series and metrics, none of it re-read from NetCDF. Replaced ELMResultsAnalyzer 2026-08-13, which was the second writer of that artifact.", "mcp/elm-mcp/src/column_rows.py", "build_rows"),
            ref("Per-column surface and time-series figures.", "mcp/elm-mcp/src/plot_columns.py", "plot_surfaces"),
            ref("Re-plot a finished run without re-running it.", "mcp/elm-mcp/scripts/replot.py", "regenerate_setup_plots"),
            ref("Standalone CLI over the same reader, for a run driven by hand.", "mcp/elm-mcp/scripts/analyze_run.py", "main")],
   "flag": "STILL SPLIT: the ELM reader lives here, but WHEN to run it is framework stage machinery, and analyze_run/analyze_agentic still import core.limitations and agents.analysis. Closing that is the Analyzer redesign.",
   "files": f("mcp/elm-mcp/src/column_rows.py", "mcp/elm-mcp/src/plot_columns.py",
              "mcp/elm-mcp/scripts/replot.py", "mcp/elm-mcp/scripts/analyze_run.py",
              "mcp/elm-mcp/scripts/analyze_agentic.py")}]},

 {"name": "PFLOTRAN MCP", "note": "20 tools · reactive transport, ensembles, and the LAMBDA network",
  "tag": "outside this repo", "steps": [
  {"n": "Decks", "side": "srv", "sum": "Write and check a PFLOTRAN input deck.",
   "flow": [ref("<code>create_pflotran_input</code> — a deck from scratch.", RXN, "create_pflotran_input"),
            ref("<code>create_column_deck</code> — a 1-D Richards column from a sampled column dict: depth, water table, recharge, van Genuchten soil.", RXN, "create_column_deck"),
            ref("<code>check_column_schema</code> — does this column dict carry what a deck needs?", RXN, "check_column_schema"),
            ref("<code>configure_reaction_sandbox</code> — attach a reaction network to the deck.", RXN, "configure_reaction_sandbox"),
            ref("<code>validate_pflotran_input</code> — parse the deck before spending a run on it.", RXN, "validate_pflotran_input")],
   "files": [[RXN, lines(RXN)]]},
  {"n": "Run", "side": "srv", "sum": "Inline for one column, sbatch for an ensemble.",
   "flow": [ref("<code>run_pflotran_simulation</code> — 1-D columns take seconds, so this runs INLINE and returns the result, no job id.", RXN, "run_pflotran_simulation"),
            ref("<code>submit_pflotran_ensemble</code> — many decks as one batch job.", RXN, "submit_pflotran_ensemble"),
            ref("<code>check_pflotran_job</code> / <code>check_simulation_status</code> — the scheduler, and the run.", RXN, "check_pflotran_job"),
            ref("<code>create_parameter_ensemble</code> — sweep a parameter across decks.", RXN, "create_parameter_ensemble")],
   "flag": "The opposite contract to ELM's: PFLOTRAN is fast enough to answer in the call. That is why this server has both an inline path and a job path, and ELM has only the job path.",
   "files": [[RXN, lines(RXN)]]},
  {"n": "Results", "side": "srv", "sum": "Observations out of finished runs.",
   "flow": [ref("<code>collect_pflotran_results</code> — gather an ensemble's output.", RXN, "collect_pflotran_results"),
            ref("<code>extract_observations</code> — pull the observation points out.", RXN, "extract_observations"),
            ref("<code>convert_pflotran_to_netcdf</code> — into a form the rest of the world reads.", RXN, "convert_pflotran_to_netcdf"),
            ref("<code>create_dart_config</code> — wire a run into DART for data assimilation.", RXN, "create_dart_config")],
   "files": [[RXN, lines(RXN)]]},
  {"n": "LAMBDA network", "side": "srv", "sum": "Organic-matter chemistry from FTICR-MS samples.",
   "flow": [ref("<code>run_lambda_preprocessing</code> — raw sample data into the pipeline.", RXN, "run_lambda_preprocessing"),
            ref("<code>run_lambda_binning</code> — bin compounds into reactive classes.", RXN, "run_lambda_binning"),
            ref("<code>generate_lambda_reaction_database</code> — the network PFLOTRAN will actually integrate.", RXN, "generate_lambda_reaction_database"),
            ref("<code>calculate_thermodynamic_properties</code> — per-compound thermodynamics.", RXN, "calculate_thermodynamic_properties"),
            ref("<code>list_available_samples</code> / <code>visualize_binning_results</code> — what is on hand, and what the binning did.", RXN, "list_available_samples")],
   "flag": "A generated network carries a header saying the binning is RECONSTRUCTED. Not a calibration to any site — the demonstration parameterisation. That fact reaches the Analyzer through the assumptions ledger, because the figures would otherwise look exactly like a calibrated study's.",
   "files": [[RXN, lines(RXN)]]}]},

 {"name": "Analyzer", "note": "results become an answer", "tag": "being redesigned",
  "steps": [
  {"n": "Compare to obs", "side": "fw", "sum": "Call the model's comparison; judge what it may support.",
   "flow": [ref("The measurements come from the MODEL'S server — one call, four observables, the figures with them. The 1,400 lines that used to do this here knew H2OSNO from ZWT and how deep ELM's soil goes, in the one box that also reads PFLOTRAN runs.", "src/agents/analysis/step1_compare.py", "compare_all"),
            ref("<b>Measurements only, and no caveats since 2026-08-13.</b> The server measures and refuses to grade: &quot;bias = -85.9 mm&quot;, never &quot;the model underestimates snow&quot;. This step used to add eight caveats saying what those numbers may not support — but each restated a measurement the summary already carried (<code>below_active_soil</code>, <code>onset_censored_columns</code>), so it was measuring AND ruling on the measurement. Understanding what the scope permits is the interpreter's job.", "src/agents/analysis/step1_compare.py", "compare_all"),
            ref("The comparison becomes evidence the interpreter may CITE — one finding per observable, carrying its figure. Until this existed, a sentence about an observation had nothing to cite and the audit struck it. <code>blocked_by</code> is now always empty: the finding carries the measurement, not a ruling on it.", "src/agents/analysis/step1_compare.py", "as_findings")],
   "files": f("src/agents/analysis/step1_compare.py")},
  {"n": "Interpret", "side": "fw", "sum": "What the numbers mean, and whether to trust them.",
   "flow": [ref("Build the context: results, assumptions ledger, and the limitations catalogue.", "src/agents/analysis/step0_context.py", "load"),
            ref("<b>Structural caveats are promoted to blocking</b> — they do not qualify a gauge comparison, they forbid the naive one.", "src/agents/analysis/step0_context.py", "_caveats"),
            ref("An LLM decides which figures answer the user's question and writes the code; pandas computes. It is shown what the comparison MEASURED, not how many rows each record held.", "src/agents/analysis/step2_investigate.py", "context_brief"),
            ref("<b>The LLM judges; code audits the judgement.</b> Every claim cites a finding — step 2's figures or step 1's comparison — every number it declares must appear in that finding, and a claim inside a blocking caveat's scope must carry the caveat's id or it is struck with the reason.", "src/agents/analysis/step3_interpret.py", "audit")],
   "files": f("src/agents/analysis/step0_context.py",
              "src/agents/analysis/step2_investigate.py")},
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
/* SCOPED TO THE STEP'S OWN BUTTON. Unscoped, an open step also rotated every
   chevron of its sub-boxes, so a closed sub read as open. */
.step.open > .s-btn .chev{transform:rotate(-135deg);margin-top:.15rem}
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
/* SUB-BOXES — one level deeper than a stage. A stage that dispatches to
   several independent implementations (compare_to_obs -> one module per
   observable) reads as a single flow otherwise, and the flows are not the
   same: swe matches before it compares, the other three do not yet. */
.subs{display:grid;gap:.4rem;margin-top:.1rem}
.sub{border:1px solid var(--rule);border-radius:6px;background:var(--panel);
overflow:hidden}
.sub-btn{appearance:none;font:inherit;color:inherit;background:none;border:0;
width:100%;text-align:left;cursor:pointer;padding:.45rem .55rem;display:flex;
flex-direction:column;gap:.18rem}
.sub-btn:focus-visible{outline:2px solid var(--srv);outline-offset:-2px}
.sub-top{display:flex;align-items:center;justify-content:space-between;gap:.5rem}
.sub-name{font-family:var(--mono);font-size:.74rem;font-weight:600;color:var(--ink)}
.sub-sum{font-size:.74rem;color:var(--ink-2);line-height:1.38}
.sub-body{display:none;padding:.1rem .55rem .6rem;border-top:1px dashed var(--rule)}
.sub.open .sub-body{display:block}
.sub.open > .sub-btn .chev{transform:rotate(-135deg);margin-top:.15rem}
.sub-body ol.flow{margin-top:.45rem}
.foot{margin-top:2rem;padding-top:1rem;border-top:1px solid var(--rule);
color:var(--ink-3);font-size:.78rem;max-width:70ch}
.foot code{font-family:var(--mono);font-size:.74rem;color:var(--ink-2)}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

JS = """
const WHERE={fw:"framework",srv:"model MCP",data:"data MCP"};
const host=document.getElementById("groups");
G.forEach(g=>{
  const el=document.createElement("section");el.className="group";
  el.innerHTML=`<div class="g-head"><h2>${g.name}</h2>`+
    (g.tag?`<span class="g-tag">${g.tag}</span>`:"")+
    `<span class="g-note">${g.note}</span></div><div class="steps"></div>`;
  const row=el.querySelector(".steps");
  const flow=x=>`<ol class="flow">${(x||[]).map(y=>Array.isArray(y)
      ?`<li>${y[0]}<span class="ref">${y[1]}</span></li>`:`<li>${y}</li>`).join("")}</ol>`;
  const files=x=>`<ul class="files">${(x||[]).map(y=>
      `<li><span class="fp">${y[0]}</span><span class="fl">${y[1]?y[1]+" lines":"—"}</span></li>`
    ).join("")}</ul>`;
  (g.steps||[]).forEach(s=>{
    const d=document.createElement("div");
    d.className="step";d.dataset.side=s.side;
    if(s.breach)d.dataset.breach="1";
    const subs=(s.subs||[]).map(u=>`<div class="sub">
        <button class="sub-btn" aria-expanded="false">
          <span class="sub-top"><span class="sub-name">${u.n}</span><span class="chev"></span></span>
          <span class="sub-sum">${u.sum}</span></button>
        <div class="sub-body">${flow(u.flow)}`+
        (u.flag?`<div class="flag">${u.flag}</div>`:"")+
        (u.files&&u.files.length?`<h4>Source</h4>${files(u.files)}`:"")+
        `</div></div>`).join("");
    d.innerHTML=`<button class="s-btn" aria-expanded="false">
        <span class="s-top"><span class="s-name">${s.n}</span><span class="chev"></span></span>
        <span class="s-where">${WHERE[s.side]||s.side}${s.breach?" · needs moving":""}</span>
        <span class="s-sum">${s.sum}</span></button>
      <div class="s-body"><h4>Inside this stage</h4>${flow(s.flow)}`+
        (subs?`<h4>${s.subs_label||"Inside that, one per kind"}</h4>
          <div class="subs">${subs}</div>`:"")+
        (s.flag?`<div class="flag">${s.flag}</div>`:"")+
        `<h4>Source</h4>${files(s.files)}</div>`;
    const b=d.querySelector(".s-btn");
    b.addEventListener("click",()=>{
      const open=d.classList.toggle("open");
      b.setAttribute("aria-expanded",open?"true":"false");});
    d.querySelectorAll(".sub-btn").forEach(sb=>{
      sb.addEventListener("click",()=>{
        const open=sb.parentElement.classList.toggle("open");
        sb.setAttribute("aria-expanded",open?"true":"false");});});
    row.appendChild(d);});
  host.appendChild(el);});
"""


def main():
    ap = argparse.ArgumentParser()
    # The published artifact is keyed to this FILE PATH: republishing the same
    # path keeps the same URL, a different path mints a new one. Changing this
    # default orphans the link people have — the old arch_map.html artifact was
    # lost that way on 2026-08-10 and had to be re-published as v2.
    ap.add_argument("--out", default="workflow_outputs/ideas_arch_map_v2.html")
    a = ap.parse_args()

    groups = [dict(g) for g in GROUPS]
    for g in groups:
        if g["steps"] is None:
            g["steps"] = _servers(want_models=g.get("models", False))

    # A MODEL SERVER WITH NO CARD IS A HOLE IN THE MAP. The data servers are
    # generated from mcp_config.json so a new one appears by itself; the two
    # model servers are hand-detailed, which means adding a third would silently
    # show nothing. Checked here against the same config the data side reads.
    named = " ".join(g["name"] for g in groups).lower()
    uncovered = [s for s in MODEL_SERVERS
                 if s not in named and SERVER_TITLE.get(s, s).split()[0] not in named]

    missing = [p for g in groups for s in g["steps"] for p, n in s["files"] if not n]
    # SUB-BOXES ARE CHECKED TOO. They were not when they were added, and the
    # count silently fell from 119 references to 115 — the four that vanished
    # were the ones now nested, so the map gained a level of detail and lost
    # the guarantee that it points at anything real. A reference that is never
    # verified is exactly the rot this generator exists to prevent.
    def _flows(step):
        yield from (step.get("flow") or [])
        for sub in (step.get("subs") or []):
            yield from (sub.get("flow") or [])

    stale = [w for g in groups for s in g["steps"] for r in _flows(s)
             if (w := check_ref(r))]
    html = (f"<title>IDEAS pipeline — agents, stages, and what happens inside each</title>\n"
            f"<style>{CSS}</style>\n"
            f'<div class="wrap">\n'
            f'  <p class="eyebrow">IDEAS · multi-agent-framework · generated from the source</p>\n'
            f"  <h1>Agents, stages, and what happens inside each</h1>\n"
            f'  <p class="lede">Seven groups own the run end to end. Each stage is coloured by\n'
            f"    <b>where it executes</b> — a different question from who orchestrates it.\n"
            f"    Click any stage for its internal workflow and its source.</p>\n"
            f'  <div class="legend">'
            f'<span class="key"><span class="dot" style="background:var(--fw)"></span>Framework</span>'
            f'<span class="key"><span class="dot" style="background:var(--dat)"></span>Data MCP</span>'
            f'<span class="key"><span class="dot" style="background:var(--srv)"></span>Model MCP</span>'
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
    for s in uncovered:
        print(f"   ⚠️  model server '{s}' is registered but has no group on the "
              f"map — add one, or it silently does not exist to a reader")


if __name__ == "__main__":
    main()
