# Runbook — question → simulation → results

End-to-end recipe for running the multi-agent ELM framework, with **a plot to
debug at every step**. Two tracks: a **spatial / site study** (sample a real
watershed) and a **controlled soil sweep** (vary one factor at one site).

> Each step says *where it runs* — **login node** for planning/building/analysis,
> **compute node** (`salloc`) only for the actual ELM runs.

## Setup (once per terminal)
```bash
cd ~/RCSFA/multi-agent
module load pytorch/2.8.0          # the env for everything below (openai, mcp, xarray…)
```
The reusable reference build (shared executable, clone from it):
```bash
REF=/pscratch/sd/h/hvtran/E3SMv3/1D_ELM.3c13216be8.2026-06-19-150916.elm_phase0
```

---

## One-command wrapper — REMOVED 2026-08-13

`tools/run_watershed.sh` is gone. It predated `workflow.py` and had been broken
since the ELM code moved into `mcp/elm-mcp/`: four of the six paths it invoked
no longer existed, and it died at the third — before building anything. So
`--execute` and `--submit` had not built or submitted a run in some time.

**Use `workflow.py`**, which drives Reception → Planner → Experiment Manager →
Analyzer and reaches ELM through the MCP tools (`run_elm_ensemble` submits job A
and the manager chains job B). The manual steps below still work for driving or
debugging a single stage.

---

## Track A — spatial / site study

### 1 · Plan — question → strategy  *(login, LLM, ~2–3 min)*
```bash
python3 tools/run_pipeline.py "Partitioning of runoff/recharge in the Naches sub-watershed (HUC8 17030002), validate with observations"
RD=workflow_outputs/pipeline_<TIMESTAMP>      # paste the dir it printed
```
Produces `reception_brief.json`, `plan.json` (incl. the **`feasibility`** verdict).
**Debug:** `python3 -m json.tool $RD/plan.json` — check `feasibility`, `sampling_strategy`, `requires_capabilities`.

### 2 · Materialize  *(login, ~1–2 min)*
```bash
python3 tools/expand_sampling.py --run-dir $RD
```
→ `$RD/columns.json` (real lat/lon per column)

The design figure is NOT drawn here any more. It shows post-warm-start values —
the snapped coordinates, the donor gridcell's soil, the initial soil water read
out of each finidat — none of which exist until step 3 builds the inputs. It
appears then, as `$RD/sampling_design.png`: the columns over a hillshade, NLDAS
forcing against donor elevation, the initial soil water, and the sand / clay /
organic profiles the model was handed.

*The first call hits the terrain MCP; re-running it is a free pure-read.*

### 3 · Adapter — columns → executable plan  *(login, instant)*
```bash
python3 src/core/columns_to_plan.py $RD/columns.json --yr-start 1995 --yr-end 1995 --out $RD/run_plan.json
```

### 4 · Build/clone the cases  *(login, ~1 min)*
```bash
python3 tools/build_cases.py --plan $RD/run_plan.json --ref $REF --out-dir $RD
```
→ `$RD/cases.json` + `$RD/exe_path.txt`.
**🖼 Debug (do this before running!):**
```bash
python3 tools/plot_columns.py --run-dir $RD --cases-file cases.json --surfaces
```
→ `$RD/04_analysis/debug_surfaces.png` — confirms **each column got a distinct, correct
soil profile** (clay/sand vs depth). Catches surface-file collisions before you spend node time.

### 5 · Execute  *(compute node — `salloc`, ~3 min/column)*
```bash
salloc -N 1 -t 60:00 -q interactive -C cpu -A m3780
EXE=$(cat $RD/exe_path.txt)
CASES=$(python3 -c "import json;print(' '.join(json.load(open('$RD/cases.json'))))")
bash tools/run_cases.sh "$EXE" $CASES | tee $RD/run.log
exit                                           # release the node
```
Each line shows `rc=0  3min  history_files=9` on success.

**Faster — one batch job, all columns concurrent (no interactive node):**
```bash
bash tools/submit_cases.sh $RD -q debug          # submit; returns a job id
squeue -j <JID>                                  # watch ; tail -f $RD/run.log
```
Columns are tiny (1 task / 2 cores), so they all run *at once* on a single
`debug` node — ~3 min wall regardless of N, submit-and-forget. `--dry` writes the
sbatch script without submitting; `-t` sets the walltime.

### 6 · Analyze + 🖼 RESULT plots  *(login, ~30 s)*
```bash
python3 tools/analyze_run.py --run-dir $RD --cases-file cases.json --plan-file run_plan.json --plot
python3 tools/plot_columns.py --run-dir $RD --cases-file cases.json --timeseries   # debug
```
→ `$RD/04_analysis/`: `elevation_gradient.png`, `soil_control.png`, `hydro_summary.json`,
and `debug_timeseries.png` (per-column recharge/runoff over the run).

---

## Track B — controlled soil sweep
Same machinery, but **replace steps 1–3** with one command (vary only soil at one site;
no planning plot — it's controlled, not spatial):
```bash
SW=workflow_outputs/soil_sweep
python3 tools/make_soil_sweep.py --out-dir $SW           # clay 5→55% at a fixed site
# then steps 4–6 with these substitutions:
python3 tools/build_cases.py  --plan $SW/soilsweep_plan.json --ref $REF --out-dir $SW
python3 tools/plot_columns.py --run-dir $SW --cases-file cases.json --surfaces   # verify 7 distinct soils
#   ... salloc run (step 5) ...
python3 tools/analyze_run.py  --run-dir $SW --cases-file cases.json --plan-file soilsweep_plan.json --plot
```

---

## Plot-to-debug at every step

| step | command | figure | what it verifies |
|------|---------|--------|------------------|
| 1 plan | `python3 -m json.tool $RD/plan.json` | — (text) | feasibility verdict, sampling strategy, required capabilities |
| 2 materialize | `expand_sampling.py --run-dir $RD` | `columns.json` | where the columns landed — bands, watershed |
| 3 build inputs | (the elm MCP draws it) | `sampling_design.png` | the ensemble ELM will integrate: snapped positions, donor soil, initial soil water |
| 4 build | `plot_columns.py --run-dir $RD --surfaces` | `04_analysis/debug_surfaces.png` | **each column got a distinct, correct soil** (before you run) |
| 5 run | `plot_columns.py --run-dir $RD --timeseries` | `04_analysis/debug_timeseries.png` | runs produced sensible, differentiated dynamics |
| 6 analyze | `analyze_run.py --run-dir $RD --plot` | `04_analysis/elevation_gradient.png`, `soil_control.png` | the science result + honest driver attribution |

All `.png` files render directly in VS Code. The durable outputs live in `$RD/` and
`$RD/04_analysis/`; raw ELM history files live on `$PSCRATCH` (purge-eligible), so the
`04_analysis/` summaries are the record worth keeping.

## Pick an observation-rich watershed first (optional)
```bash
python3 tools/scout_watersheds.py "Naches" "Walla Walla" 17090010
```
Resolves each (name or HUC8) and tallies SNOTEL / stream gages / GW wells + relief.
**Lower relief ⇒ less coarse-forcing confound** (a cleaner spatial study).

## Validate against observations (step 7)
```bash
python3 tools/validate_run.py --run-dir $RD     # after step 6
```
→ `$RD/04_analysis/validation.json` + `validation.png`. Three real confrontations:
**water table** (model ZWT vs Fan 2013 vs observed USGS wells with records, as
distributions), **streamflow** (modeled yield vs observed *specific discharge* —
gauge mean flow ÷ drainage area — for in-domain gauges with daily records in the
simulated year), and **snow** (observed peak SWE per SNOTEL station; context-only —
the run has no SWE output). Honest per-target status + caveats. Hits live MCP
sources + the USGS OGC daily API.

## Where things run
- **Login node:** steps 1, 2, 3, 4, 6 + all debug plots (planning, building, analysis).
- **Compute node (`salloc`):** only step 5 (the ELM runs).
