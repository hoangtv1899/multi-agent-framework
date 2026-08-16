# Conceptual ELM studies — implementation plan

A conceptual study answers a question about a *mechanism* rather than a place:
*"How does soil texture split rain between surface runoff and drainage?"* There
is no basin, no observation to compare against, and no spatial sampling. What
there is instead is a **controlled sweep** — a set of columns identical in every
respect but one.

This document is the plan for adding that. It is not a description of what
exists; the status table at the end says what is built.

---

## Two rules that constrain everything below

**1 · Additive only. The site path must not move.**

Every change is either a new file, a new key that site runs never set, or a
branch whose `else` is the existing code untouched. The proof is mechanical:

```
64 failed, 726 passed, 72 skipped          # baseline, before any conceptual work
```

The exact `FAILED` list is snapshotted. Every step diffs against it and the
diff must be empty. A step that changes the set — in either direction — is
wrong until explained. (Counts may rise: a new source file adds a case to
`test_no_shadowed_methods.py`, which parametrises over files.)

**2 · No ELM knowledge in the framework.**

Anything that is true because of how ELM works belongs in `mcp/elm-mcp/`.
The framework may hold archetype arithmetic — how many columns a factorial
design has, what it holds fixed — because a PFLOTRAN sweep needs the identical
logic and an entirely different factor list.

This rule is why `src/core/factors.py` was written, reduced, and finally
**deleted on 2026-08-15**. Its first version restated `RUNTIME_KEYS` and knew
what a soil profile looks like, which broke the rule outright. Rewritten
model-agnostic, it broke nothing — and turned out to be needed by nobody: the
rule pulls every one of its jobs to the server that owns the answer. Checking
a factor is real, spotting two factors that collide, and knowing what is held
fixed all belong where the menu is declared. Counting columns is one
multiplication, done in `conceptual._n_columns` and
`strategy_check._sweep_n_columns`. Nothing ever imported it.

---

## The shape

A conceptual study is **a column list where one field varies and the rest do
not.** Everything downstream of `columns.json` already works, because the
Experiment Manager does not care *why* two columns differ.

So all the new work is upstream of that file, and the two paths merge there.

```
                    site                          conceptual
                     |                                 |
   the axis      elevation                      a model input
   the levels    bands                          factor values
   who places    tools/expand_sampling.py       the ELM server
                     |                                 |
                     +------------ columns.json -------+
                                       |
                          everything below is UNCHANGED
```

### One block, not two

`sampling` gains a discriminator rather than growing a sibling:

```json
"sampling": {
  "approach":      "elevation_bands" | "factor_sweep",
  "n_columns":     7,
  "justification": "clay 5-55% in 7 steps, one site, one year",

  // approach = elevation_bands — exactly as today, untouched
  "n_bands": 4, "per_band": 3, "n_validation": 2,

  // approach = factor_sweep — new, and never present on a site run
  "factors":    [{"name": "soil_texture", "levels": [5,10,18,27,35,45,55]}],
  "held_fixed": {"lat": 47.11, "lon": -121.39, "years": [1995,1995]}
}
```

`approach`, `n_columns` and `justification` mean the same thing in both designs
— they are the shared spine. Nothing is removed or moved, so every current
reader of `sampling` is unaffected.

Two reasons this beats a separate top-level `factors` key:

- `step0_context.py:631` copies `sampling` wholesale into `ctx.plan`, so the
  conceptual design reaches the Analyzer with no new plumbing.
- `approach` is free text today, read by nothing. Making it an enum gives
  `strategy_check` something to validate, which is where the conceptual shape
  check wants to live anyway.

The cost: `approach` becomes load-bearing and must be validated, because a
wrong value now silently selects the wrong shape.

---

## What each stage does

### Reception — the only stage that talks

Today it classifies a no-place question as conceptual correctly, skips fetching
observations, and stops with a nearly empty brief.

It gains a conversation:

1. Ask the ELM server **what can be varied**, and offer that as the menu.
2. Recommend a model, with the reason stated. Say separately which models are
   runnable — availability and fitness are different facts and must not be
   merged into one sentence.
3. Elicit what the question left open: which factor, where the weather comes
   from, which years, what is held constant.
4. Record each answer **with how it was settled** — stated by the user, chosen
   from a suggestion, or defaulted by the framework. "The user chose seven clay
   levels and the framework picked the site" is a different study from "the
   user chose both", and the caveats need to know which.

Machinery it already has: the `ask_user` tool in `tool_loop.py`, enabled by
`workflow.py --interactive`.

**Suggest, with a default:** level count and values, organic matter, bulk
density, profile depth, run length.

**Ask, and never default quietly:** the site's climate (a sweep at 1,373 mm/yr
and one at 300 mm/yr answer differently, and this becomes a blocking caveat),
and uniform versus layered profiles (a different experiment, not a different
setting).

### Planner — the same job, a different answer

It already decides the only thing that matters here: along what axis the columns
differ, and how many. For a site it writes *"relief 1471 m → 4 bands × 3 = 12,
+2 pinned = 14"*. For a sweep it writes the factor and its levels, and the same
count and arithmetic.

It is **shown** the factor declaration the way it is already shown a short
observations summary — Reception fetches, the Planner designs against it. It
has no tools and needs none.

`validation` stays empty. There are no stations.

**The Planner must not judge whether a value is sensible.** One call, no tools,
answering from memory is exactly the guessing this design removes. It proposes;
the server rules.

### The ELM server — three new tools and one guard

**`describe_conceptual_factors()`** — what can be varied, sensible ranges, and
the caveats each choice carries. Derived from `RUNTIME_KEYS` and the surface
generator rather than restated beside them, so the menu cannot claim a
capability the wrapper lacks. Joins the existing `describe_elm_capabilities`
family and follows its rot-resistance rule: a claim names the symbol whose
absence would falsify it.

**`check_conceptual_design(design)`** — takes the whole design, not one field.
Returns three separate lists:

| list | meaning | what Reception does with it |
|---|---|---|
| `wont_build` | arithmetic or buildability failure | turn into a question for the user |
| `unusual` | builds, but far from anything measured | becomes a caveat on the study |
| `facts` | measurements about the design | may be fine, may be the whole problem |

It must run **before compute**, and it must share its validation code with the
column builder — otherwise a design passes the check and fails the build.

**`build_conceptual_columns(design)`** — the design becomes `columns.json` in
the identical shape `expand_sampling.py` emits.
`mcp/elm-mcp/scripts/make_soil_sweep.py` already does the science; it writes a
run plan directly and needs to write the column list instead.

**The guard — `inputs.py:355`, `attach_donor_soil()`:**

```python
c["soil_profile"] = prof        # unconditional today
```

For a site run this is correct: a warm start keeps the donor gridcell's
surfdata, so a profile gathered at sampling time is characterisation rather
than what ELM runs on. For a soil sweep it is fatal in the worst way — all
columns sit at one lat/lon, so all of them receive the *same* donor profile,
the prescribed gradient is erased, every column runs, every check passes, and
the Analyzer reports a clean `n = 7` for a sweep with no gradient in it.

The guard is one condition, keyed on a field only conceptual columns set. A
site column can never reach it.

**Decision:** keep the warm start, keep the prescribed soil. Cold-starting
instead would discard the initialisation that is deliberately the default. The
cost is that the restart's initial water content belongs to a column of
different texture, so early time is unreliable — recorded as a caveat, not left
as a silent mismatch.

### Experiment Manager, materialize — one branch

Read `sampling.approach`. If `elevation_bands`, everything happens exactly as
today. If `factor_sweep`, ask the server for the columns instead of calling the
sampler. After that the paths join and are identical.

**Conceptual columns must not carry `band`.** `planned_vs_actual()` at
`step0_context.py:472` compares planned bands against delivered bands. On a
conceptual run both are currently `None`, so the claim passes **by accident**.
Set `band: 1` — the natural thing when mirroring the site column shape — and it
reports a failure to deliver bands nobody asked for, which now flows into
`preflight().unmet_plan_targets` and into the report.

Fix the claim to be conditional rather than relying on the accident.

### Analyzer — one skip and a caveat family

Step 1 skips on archetype, reusing the `skipped` record shape that already
exists for an unregistered backend. The reason string matters: *"this study has
no observational component by design"*, not *"no observations found"*.

Steps 2, 3 and 4 are unchanged. `soil_summary` is already in
`COLUMN_METADATA` and already reaches step 2's metadata catalogue, so "runoff
against clay" is a figure the model can write today.

New caveats, generated from `factors`/`held_fixed` rather than hand-written:

| id | severity | why |
|---|---|---|
| `conceptual_no_observations` | blocking | every claim is about this configuration of ELM, not about soil |
| `conceptual_one_climate` | blocking | the sweep answers for the cell that was picked |
| `conceptual_uniform_soil` | qualify | organic matter and bulk density pinned; the method working, and why the columns are not real soils |
| `conceptual_donor_state` | qualify | initial water content came from a column of different texture |
| the existing 1-D routing limitation | blocking | carries over unchanged — "surface runoff" is local generation, not streamflow |

**What was held constant is the finding.** A clay sweep in which rooting depth
also moved is not a soil experiment, and nothing downstream can detect that from
the outputs. It is recorded from the design, before compute.

---

## Bad requests: refuse, flag, or ask

Three different problems, three different owners, and **nothing silently
corrects**.

| problem | example | owner | response |
|---|---|---|---|
| cannot be built | clay 60% + sand 70% = 130% | ELM server | **refuse**, naming the field |
| far from anything real | clay at 95% | ELM server | **build it**, attach a caveat |
| means something other than the user thinks | "make the soil 2 m deep" | Reception | **ask** |

The third deserves its example. ELM's soil column is a fixed grid — 15 layers,
1.75 cm at the top to 13.85 m at the bottom, 42.10 m total, the first 10 layers
making the 3.80 m that does the hydrology. A study does not configure that.
"2 m of soil" says what fills the top 2 m; the substrate setting fills the rest,
and the existing sweep extrapolates the bottom layer downward. Someone
picturing 2 m of soil on bedrock will get 2 m on top of 40 m more of the same.
No check fires, and none should — it is a conversation.

Clamping 120% clay to 100% and running is the worst available outcome: a
finished study, a clean column count, figures, and a claim about a soil nobody
chose.

### The server states; the framework grades

`check_conceptual_design` reports **what is true** about a design. It does not
decide how much that matters.

This is the rule step 1 already follows — it used to add eight caveats and had
them removed, because the server measures and refuses to grade. The server says
*"45% clay and above is outside the fitted range."* Whether that blocks a claim
or merely qualifies it is assigned where every other caveat's severity is
assigned. Two places deciding what a study may conclude will eventually
disagree.

---

## Deliberately absent

Named here so that adding one means adding the builder **first** and the menu
row second:

| not built | what it needs |
|---|---|
| PFT parameter sweeps | a `paramfile` in `user_nl_elm`; `RUNTIME_KEYS` does not carry one |
| graded initial water table | perturbed restart files — a netCDF edit on `WA`/`ZWT`. Small, but absent. |

**Written weather moved off this list on 2026-08-15.** It was the one item here
described as "the most-wanted conceptual study", and it is now the
`prescribed_weather` factor plus `held_fixed.weather`. The two are different
studies and both are offered: a factor makes the columns differ in weather, and
`held_fixed.weather` gives every column the *same* written weather, which is
what stops a soil result from being bounded by a borrowed climate. Four fills —
`copy`, `uniform`, `scale`, `offset`.

One caution is carried in the menu rather than enforced: **uniform precipitation
removes rainfall intensity**, which is exactly the thing that splits rain
between runoff and drainage. A texture sweep under constant rain understates how
much texture matters, and the understatement reads like a finding. `scale` keeps
the storm structure and is the better instrument for that question. Reception
says so and then builds whichever the user chooses.

The last unproven link: **no ELM case has yet run on a written stream.** The
writer round-trips a real NLDAS cell with zero differences across all seven
variables, so the files are right; what is untested is that ELM reads them,
which would fail at model init rather than quietly.

A menu entry is a capability claim the Planner designs against.

---

## Order and status

| # | step | where | status |
|---|---|---|---|
| 1 | ~~model-agnostic factor arithmetic~~ | ~~`src/core/factors.py`~~ | **dropped** — nothing imported it; the work belongs to the server that owns the menu |
| 2 | factor declaration | ELM server | |
| 3 | design check | ELM server | |
| 4 | column builder | ELM server | |
| 5 | prescribed soil survives the warm start | `inputs.py` | |
| 6 | conditional band claim | `step0_context.py` | |
| 7 | `approach` dispatch | Exp Manager materialize | |
| 8 | `sampling` schema + validation | `planner.txt`, `strategy_check.py` | |
| 9 | the conversation | `reception_agentic.txt`, `reception_llm.py` | |
| 10 | step 1 skip + conceptual caveats | Analyzer | not started |
| 11 | written weather end to end | `forcing.py` → `elm_wrapper.py` | files + stream written at case build; **no case has run on one yet** |

Steps 1–7 can be built and tested from a hand-written `strategy.json` with no
model call. Steps 8–10 replace the hand-written file with a real conversation.
The hand-written file is also the test fixture.
