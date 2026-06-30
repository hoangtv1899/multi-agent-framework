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

## One-command wrapper (recommended)
`run_watershed.sh` chains the login-node steps (1–4) + the soil pre-flight, then
prints the `salloc` run block and the resume command. Add `--execute` to also run
on a node and analyze — the whole pipeline in one go.

```bash
# safe default: plan -> plot -> build -> pre-flight, then it prints steps 5 & 6
bash tools/run_watershed.sh "Partitioning of runoff/recharge in the <WATERSHED> (HUC8 <CODE>), validate with observations"
bash tools/run_watershed.sh --analyze workflow_outputs/pipeline_<TIMESTAMP>   # step 6, after the salloc run

bash tools/run_watershed.sh --execute "…question…"   # all-in-one (also runs the salloc step)
# -i = let reception clarify an ambiguous name;  override with  YR_START= YR_END= REF=
```
The manual steps below are exactly what the wrapper runs — use them to drive or
debug any single stage.

---

## Track A — spatial / site study

### 1 · Plan — question → strategy  *(login, LLM, ~2–3 min)*
```bash
python3 tools/run_pipeline.py "Partitioning of runoff/recharge in the Naches sub-watershed (HUC8 17030002), validate with observations"
RD=workflow_outputs/pipeline_<TIMESTAMP>      # paste the dir it printed
```
Produces `reception_brief.json`, `plan.json` (incl. the **`feasibility`** verdict).
**Debug:** `python3 -m json.tool $RD/plan.json` — check `feasibility`, `sampling_strategy`, `requires_capabilities`.

### 2 · Materialize + 🖼 the PLANNING plot  *(login, ~1–2 min)*
```bash
python3 tools/expand_sampling.py --run-dir $RD --plot
```
→ `$RD/columns.json` (real lat/lon + soil per column) **and `$RD/sampling_design.png`**
(terrain map + watershed outline + elevation bands + Fan water-table + columns-per-band).
*First call hits the terrain MCP; re-running `--plot` later is a free pure-read.*

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
| 2 materialize | `expand_sampling.py --run-dir $RD --plot` | `sampling_design.png` | where the columns landed — bands, watershed, water table |
| 4 build | `plot_columns.py --run-dir $RD --surfaces` | `04_analysis/debug_surfaces.png` | **each column got a distinct, correct soil** (before you run) |
| 5 run | `plot_columns.py --run-dir $RD --timeseries` | `04_analysis/debug_timeseries.png` | runs produced sensible, differentiated dynamics |
| 6 analyze | `analyze_run.py --run-dir $RD --plot` | `04_analysis/elevation_gradient.png`, `soil_control.png` | the science result + honest driver attribution |

All `.png` files render directly in VS Code. The durable outputs live in `$RD/` and
`$RD/04_analysis/`; raw ELM history files live on `$PSCRATCH` (purge-eligible), so the
`04_analysis/` summaries are the record worth keeping.

## Where things run
- **Login node:** steps 1, 2, 3, 4, 6 + all debug plots (planning, building, analysis).
- **Compute node (`salloc`):** only step 5 (the ELM runs).
