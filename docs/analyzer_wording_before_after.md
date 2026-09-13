# The Analyzer's wording, before and after the readability gate

Date 2026-09-12. Same runs, same figures, same findings, same audit. Only the
reviewer's prompt and a code gate on the wording changed; every number below
was checked against the run's findings exactly as before. The old wording is
kept in each run's `04_analysis/before_<stamp>/`.

## What changed in the code

- `step3_interpret.py`: the reviewer is told to write for a reader who will put
  the text on a slide (sentences of at most 30 words, every variable code glossed
  once, no column ids, numbers rounded to 3 significant figures in the sentence
  with the exact value in `values`, no dashes as punctuation), and it must
  return a one-sentence `headline`. A deterministic gate checks all of that
  after the reply; a failing text is sent back once for a wording-only rewrite,
  then audited exactly as before. What the gate still objects to is recorded.
- `step4_report.py`: writes `REPORT.md` beside `analysis.json`, the presenter's
  copy: headline, answer, each audited claim with its values shown to 3
  significant figures and its finding id, the binding caveats, what the audit
  withheld, provenance. Values in the deck are shown the same way; the exact
  values stay in `analysis.json`.
- `tools/reinterpret_run.py`: re-runs steps 3 and 4 on an archived run without
  recomputing a figure (one model call), keeping the previous wording on record.

## Naches 1979, ELM, 19 columns (demo 1)

`workflow_outputs/elm_run_20260814_121346`

### Before

In this run the partitioning is not a smooth function of elevation. Locally generated surface runoff stays a small share of precipitation everywhere (largest at the lowest column, QOVER/P = 0.37 at 471 m, and only 0.15 at 1913 m), while sub-surface drainage is bimodal: where the diagnosed water table is shallow enough to intersect the drainage parameterisation, QDRAI/P reaches 0.78–0.84 at 1206–1913 m, but 10 of 19 columns — including several high ones — drain below 0.01 mm for the year even though they recharge hundreds of mm (e.g. col_11, 804.5 mm QCHARGE with zero QDRAI), because the water table sits below the 3.8 m active soil in all 18 columns (median 12.1 m). Timing does follow elevation: melt and the drainage response shift from November–December at 471 m to a May melt peak with a June drainage peak at 1913 m. The comparison with observations is limited to SWE at two co-located SNOTEL pillows (bias -85.9 and -43.9 mm, NSE 0.29 and 0.72, with the 12 km forcing-cell caveat); the runoff/recharge split itself cannot be validated, since these are unrouted 1-D columns with no run-on and the basin has no flux tower or recorder well for 1979.

*9 claims made, 1 struck by the audit; no headline.*

### After

**Runoff share falls from 37 percent at 471 m to about 15 percent higher up, while soil drainage reaches 84 percent at 1913 m**

Across the 19 columns the share of precipitation leaving as local surface runoff is highest at the lowest site, 37 percent, and stays near 15 percent at the highest sites. Deep drainage, not runoff, dominates high up, reaching 84 percent at 1913 m. But 10 of 19 columns drain nothing, because their water table sits below the 3.8 m of active soil. In those columns water leaves instead as aquifer recharge. Both pathways follow snowmelt, with the high column peaking in May and June. The run cannot test this split against river flow, because the columns do not exchange water or route it. So the two gauge comparisons are context only, and the sole in basin check is snow water equivalent at two pillows.

*9 claims made, 0 struck; readability problems 3 before the rewrite, 0 after. Presenter copy: `workflow_outputs/elm_run_20260814_121346/04_analysis/REPORT.md`.*

## Upper Gunnison 1995, PFLOTRAN, 17 columns (demo 2)

`workflow_outputs/pflotran_run_20260818_193601`

### Before

In this run water moves down through the unsaturated column as a sharp block front that, over the simulated year, reaches 11.8–296.3 m below land surface in the 12 columns that had an unsaturated zone (deepest at high elevation: 296.3 m, 271.8 m, 212.8 m); 5 low-elevation columns start essentially saturated and show no vertical transit. In every column the front stops at or above the pre-existing CONUS2 water table (up to 99.9% of the initial unsaturated thickness), and the saturated boundary itself moved upward by at most 44.5 m. These depths are, however, a property of the CONUS2 initial water table used here (model median 209.5 m versus 3.03 m median in the 13 Fan-2013 wells in the basin), so they should not be read as evidence of hundred-metre percolation in the real Upper Gunnison; no transit time or velocity can be derived from the 4 available snapshots.

*6 claims made, 0 struck by the audit; no headline.*

### After

**Water wetted the soil from the surface down to the given water table, 0 to about 296 m deep in the sampled columns.**

In the 17 sampled columns the saturation change reached from 0 m to about 296 m below the surface. In several columns it stopped within 1 percent of the depth of the water table the column was given. The columns began extremely dry above that water table, so the year of applied precipitation wetted the entire unsaturated zone rather than a shallow layer. The depths are therefore set by the starting water table. That water table sits at a median of 210 m against 3.0 m in 13 basin wells, so the depths are not a realistic percolation depth. The run also cannot show the speed or the seasonal timing of that downward movement, only its final depth extent.

*7 claims made, 0 struck; readability problems 2 before the rewrite, 0 after. Presenter copy: `workflow_outputs/pflotran_run_20260818_193601/04_analysis/REPORT.md`.*

## Conceptual loam column, PFLOTRAN, 4 columns (demo 3)

`workflow_outputs/pflotran_run_20260818_220421`

### Before

In this run the columns are not dry when the rain year starts: after the spin-up, all four are already wet through their entire unsaturated zone (top-cell saturation ~0.672-0.677 at t = 10.25 y), so the question 'how far does one year of 500 mm get' cannot be answered as a virgin wetting front — the profile is at quasi-steady state and the year's rain only adds a thin near-surface pulse (Δsat ≤ ~0.022, deepest cell with Δsat > 0.01 at 0.75 m in the 2, 5 and 10 m columns) with compensating drying (down to -0.088) below it. Water-table depth therefore does not control penetration depth of the annual pulse in any resolvable way: the solved 101325 Pa surface stays within a few centimetres of its prescribed depth in every column (rain-year movement between -0.029 and +0.033 m), moving up and down rather than showing a discrete arrival. The one exception is the 20 m column, whose rain-year change peaks deep (0.0236 at 19.75 m) and whose water table has risen 0.151 m relative to the input — most consistent with that longest column still completing its spin-up drainage/wetting rather than with the single rain year reaching 20 m. Absolute wetting depths are an upper bound because all precipitation is applied with no ET, runoff or snow, and the soil is a single loam parameter mean.

*7 claims made, 0 struck by the audit; no headline.*

### After

**One 500 mm rain year wets mainly the top 0.75 m; the water table moves less than 0.04 m in every column.**

In the columns with the water table at 2, 5 and 10 m, the rain year raises wetness by at most about 0.022, and only down to 0.75 m. Only the 20 m column shows a gain within half a cell of its water table, and even there the water table rises just 0.033 m. The reason is that ten earlier years of the same rain already left the whole column wet. Extra rain mostly replaces water draining out, and parts of the profile below one metre actually dry slightly. This run therefore does not show a dry column being wetted from scratch. It also gives no drainage flux, travel time or water budget, so no recharge fraction can be quoted.

*8 claims made, 0 struck; readability problems 7 before the rewrite, 0 after. Presenter copy: `workflow_outputs/pflotran_run_20260818_220421/04_analysis/REPORT.md`.*
