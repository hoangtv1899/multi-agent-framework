# Driving ELM and PFLOTRAN from Claude Code

How to sit at a Compy terminal, start Claude Code, and have it run the two
model servers **directly** — no `workflow.py`, no Python driver, no LLM agents
of ours in the loop. You type a sentence; Claude calls the MCP tool; the
scheduler does the work.

Two servers are wired up:

| server | what it is | tools |
|---|---|---|
| `elm` | E3SM Land Model, 1-D columns. Compiles and runs CIME cases. | 5 |
| `reaction` | PFLOTRAN — variably-saturated flow, reactive transport, the LAMBDA sandbox. | 44 |

**What was actually run.** Examples 1, 2, 4, 5 and 6 were executed on Compy on
**2026-08-03** through a real MCP client, and every output shown for them is the
real one — including two environment bugs they exposed and the fixes that
followed (Part 4). Examples 3, 7 and 8 are compositions of tools that have each
been run, with numbers quoted from the runs that measured them — they say so
where it matters.

**Re-verified 2026-08-05, after the reaction server moved to upstream `bb773fd`
+ `compy-port`.** Every example below still holds as written — the tool names
and arguments did not change, because the four tools they use are ours and came
across untouched. Re-measured on the new tree: Example 1 reproduces exactly
(11 cells, 13.0 m, `wt_in_domain: true`, exit 0 in **2.65 s** against 2.38 s
before); Example 2's ensemble ran as job **770901**, 2/2 attributed, 6.0-6.3 s
per deck; Example 4's reactive deck exits 0 in 3.9 s, and both of its dead ends
survive — `list_available_samples` still returns `n_samples: 0`, and the shipped
`batch_reaction_lambda_network.in` is byte-identical to the one that failed five
ways. The ELM examples never touched this server.

Written for someone who has never opened Claude Code. Part 1 is the terminal
setup; Part 3 is eight independent worked examples — you can run any one of
them without having run the others.

---

## Part 1 — Claude Code in the terminal, step by step

### 1. Get the binary

You already have one on this machine: the VS Code extension ships the same CLI.

```bash
ls -l ~/.vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude
```

**Do not alias a versioned path.** The extension auto-updates and that path
disappears underneath you — `2.1.221` became `2.1.223` in the middle of writing
this guide, and a command that worked an hour earlier answered `No such file or
directory`. `~/bin/claude` is already installed here as a two-line wrapper that
resolves the newest build at run time:

```bash
claude --version        # 2.1.223 (Claude Code)  — via ~/bin/claude
```

`~/bin` is on your PATH in every login shell, so this needs nothing in
`.bashrc`. Set `CLAUDE_BIN` to override it.

For a standalone install that does not depend on VS Code at all (this reaches
the internet from a Compy login node — checked):

```bash
curl -fsSL https://claude.ai/install.sh | bash     # installs into ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"               # add to ~/.bashrc
claude --version
```

Both are the same program. Use whichever you like; the rest of this guide just
says `claude`.

### 2. Log in

The terminal CLI and the VS Code extension share `~/.claude.json` and
`~/.claude/.credentials.json`, so if you are signed in inside VS Code you are
already signed in here — nothing to do. Otherwise run `claude` and use
`/login`.

### 2b. Which account answers — and which one pays

Three backends are wired up on this machine. **The shell decides**, so nothing
is global and nothing is committed: start a terminal, source one file (or
neither), and that terminal is on that backend for its lifetime.

| backend | how you get it | model | who pays |
|---|---|---|---|
| your claude.ai login | do nothing | `opus` | your personal subscription |
| **PNNL AI Incubator** | `source /qfs/people/tran289/IDEAS/env_compy.sh` | `claude-opus-5-project` | the project charge code behind `PNNL_API_KEY` |
| **AWS Bedrock** | `source /qfs/people/tran289/IDEAS/claude_bedrock.sh` | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | the project AWS account 009144422082 |

```bash
# PNNL AI Incubator — verified 2026-08-03, returns modelUsage: claude-opus-5-project
source /qfs/people/tran289/IDEAS/env_compy.sh
cd /qfs/people/tran289/IDEAS/multi-agent-framework && claude

# AWS Bedrock — same shell, flips over; re-source env_compy.sh to flip back
source /qfs/people/tran289/IDEAS/claude_bedrock.sh
source /qfs/people/tran289/IDEAS/claude_bedrock.sh us.anthropic.claude-opus-4-1-20250805-v1:0
```

Each script clears the other's variables. That is not tidiness: an
`ANTHROPIC_BASE_URL` left over next to `CLAUDE_CODE_USE_BEDROCK=1`, or an
Incubator model id like `claude-opus-5-project` carried into Bedrock, produces a
configuration that looks right and cannot resolve a model.

**Incubator models available to this project** (from `/v1/models`, checked
2026-08-03): `claude-opus-5-project`, `claude-sonnet-5-project`,
`claude-opus-4-8-project`, `claude-opus-4-7-project`, `claude-opus-4-6-v1-project`,
`claude-sonnet-4-6-project`, `claude-haiku-4-5-20251001-v1-project`, and the
4-5/4-0 generations. These are newer than the ones in the Forge setup page —
Opus 5 and Sonnet 5 are both live, so the mapping here uses them.

**Bedrock credentials expire.** They come from the AWS access portal
("Get credentials" → environment variables) and last hours, not weeks. Paste a
fresh set into `~/.aws/bedrock_creds.sh` (mode 600, outside every repo, already
created with placeholders) and re-source `claude_bedrock.sh` — you never source
the credentials file yourself; the switcher reads it. Include
`AWS_SESSION_TOKEN` — portal credentials are temporary and SigV4 fails without
it. The script tells you which of the three it can see, and says
`PLACEHOLDER` rather than `set` when it finds an un-edited template value.

**Bedrock models confirmed working on account 009144422082** (2026-08-03, real
credentials, `provider: "bedrock"` in every response):
`us.anthropic.claude-sonnet-4-5-20250929-v1:0` (the default here),
`us.anthropic.claude-opus-4-1-20250805-v1:0`, and
`us.anthropic.claude-haiku-4-5-20251001-v1:0` — the last is the background
model, and it is exercised only outside this repo, because
`DISABLE_NON_ESSENTIAL_MODEL_CALLS=1` in `.claude/settings.json` suppresses
those calls here. An MCP tool call through the `elm` server on Bedrock also
came back clean, so tool use is not affected by the backend.

**Verify a backend in one command:**

```bash
claude -p "Reply with exactly: OK" --output-format json | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print(d['result'], list(d['modelUsage']))"
# Incubator -> OK ['claude-opus-5-project']
# Bedrock   -> OK ['us.anthropic.claude-sonnet-4-5-20250929-v1:0']
```

**Symptoms worth recognising.** Expired or placeholder AWS credentials do not
fail fast — the session hangs about two minutes and ends in
`error_during_execution` with an empty `modelUsage`; that is a credential
problem, not a Claude Code one. Repeated 403s from the Incubator are what
`CLAUDE_CODE_AGENT_AUTH_TOKEN_PASSTHROUGH=1` is for, and it is already set in
this repo's `.claude/settings.json`. And when a token is exported, Claude Code
says so on startup — *"takes precedence over your claude.ai login"* — which is
the one-line confirmation that you are not spending your own subscription.

> Note on `total_cost_usd` in `--output-format json`: it is computed from
> Anthropic list prices. On the Incubator or Bedrock the real charge lands on
> the project account; treat the number as a size, not an invoice.

### 3. Start it in the right directory

**Always start Claude Code from the framework repo**, because that is where the
MCP config lives:

```bash
source /qfs/people/tran289/IDEAS/env_compy.sh     # optional, see the note below
cd /qfs/people/tran289/IDEAS/multi-agent-framework
claude
```

> `env_compy.sh` is *not* required for the MCP servers — both resolve every path
> they need themselves (that is the whole point of the environment rule at the
> end of Part 2). Source it when you want the PNNL Incubator backend (§2b), or
> when you also want to run framework scripts by hand in the same terminal.

First launch in a new directory asks you to trust the folder — say yes; it is
your own repo.

### 4. Approve the two servers, once

`.mcp.json` in the repo root declares `elm` and `reaction`. Claude Code will not
start a project-declared server until you approve it, so the first `claude` in
this directory shows a prompt like *"This project has 2 MCP servers — use
them?"*. Choose **Use this and all future MCP servers in this project**.

Check any time with `/mcp` inside the session, or from the shell:

```bash
claude mcp list
# elm: ... - ✔ Connected
# reaction: ... - ✔ Connected
```

Before approval they read "⏸ Pending approval (run claude to approve)". That is
normal, not an error.

**Wiring them up somewhere else** (another checkout, or a colleague's account)
— either copy `.mcp.json`, or register them by hand:

```bash
claude mcp add elm      /qfs/people/tran289/.conda/envs/ideas/bin/python3 \
    /qfs/people/tran289/IDEAS/multi-agent-framework/mcp/elm-mcp/main.py
claude mcp add reaction /qfs/people/tran289/.conda/envs/ideas/bin/pflotran-mcp
```

`claude mcp add` defaults to *local* scope (this directory, your account only);
`-s user` makes a server available in every directory; `-s project` writes to
`.mcp.json` for everyone who clones the repo.

### 5. Prove it works before you trust it

Inside the session, type:

```
Call describe_elm_capabilities and tell me whether this machine is ready.
```

Verified answer on Compy, 2026-08-03: `ready: true`, `missing_requirements: []`.
If it says otherwise, read Part 4 before doing anything else — a missing E3SM
tree or an absent `sbatch` is a one-line answer *here* and a baffling failure
eight minutes into a case build.

### 6. Keys and commands worth knowing

| | |
|---|---|
| `Esc` | interrupt Claude mid-answer (it stops cleanly, keeps the conversation) |
| `Shift+Tab` | cycle permission mode: ask each time → auto-accept edits → plan mode |
| `Ctrl+C` twice | quit |
| `/help` | everything |
| `/mcp` | which servers are connected, and their tools |
| `/model` | switch model |
| `/status` | account, model, working directory |
| `/clear` | drop the conversation, keep the session |
| `claude -c` | continue the last conversation in this directory |
| `claude --resume` | pick an older one from a list |

### 7. Permissions

Every tool call asks the first time. Answer **"Yes, and don't ask again for
this tool"** for the read-only ones (`describe_elm_capabilities`,
`check_elm_job`, `validate_pflotran_input`, `check_simulation_status`). Keep the ask
on anything that submits a job, at least at first — those spend node hours.

To pre-authorize a whole server for a session:

```bash
claude --allowedTools "mcp__reaction"          # every reaction tool
claude --allowedTools "mcp__elm__check_elm_job"  # exactly one
```

Both forms are checked and working (`permission_denials: []`).

### 8. Headless, for scripts and cron

`-p` runs one prompt and exits — no TTY needed, so it works inside a batch job:

```bash
cd /qfs/people/tran289/IDEAS/multi-agent-framework
claude -p "Call describe_elm_capabilities and reply with the ready field only." \
       --mcp-config .mcp.json --strict-mcp-config \
       --allowedTools "mcp__elm__describe_elm_capabilities" \
       --model sonnet --output-format json
```

`--mcp-config` loads the servers straight from the file (no approval prompt),
`--strict-mcp-config` ignores every other MCP source, and `--output-format json`
gives you `.result` plus token cost for a script to parse. The verified run of
exactly this command returned `ready: true` in 8 s for about $0.10 on Sonnet.

---

## Part 2 — what the two servers are, and are not

### `elm` — six tools

```
describe_elm_capabilities()                     what this does, needs, and does NOT do
build_elm_inputs_from_location(run_dir, ...)    -> DATA     (warm start, donor soil,
                                                             surfaces, case_inputs.json)
get_column_metadata(run_dir)                    -> DATA     (the columns as they will RUN)
run_elm_ensemble(run_dir, queue, walltime)      -> JOB ID   (JOB A: build + run, one job)
build_elm_cases(run_dir, queue, walltime)       -> JOB ID   (the build alone; ~8-10 min)
check_elm_job(job_id, run_dir)                  what Slurm is doing; case dirs when built
```

**Two tools went away on 2026-08-10**, both because they shelled out to scripts
in the framework — a server running its client's code. `run_elm_study` ran
`tools/run_study.sh`, whose job ended by calling `workflow.py --finalize`;
`run_elm_ensemble` plus a job B of your own is the same study with the callback
inverted (Example 7). `submit_elm_ensemble` ran `tools/submit_cases.sh` to run
already-built cases; `run_elm_ensemble` does that now — it verifies
`built_cases.json` against the case directories on disk and skips the compile
when they are still there.

**The rule:** *this server owns the simulation end to end; the caller decides
what to simulate and interprets what came out.* Give it locations and it does
the warm start, the donor soil, the surfaces and the case list
(`build_elm_inputs_from_location`), then compiles and runs them. What it does
not do is choose the columns, hold the observations, or say what the numbers
mean. Everything crosses the boundary as data in `<run_dir>/01_inputs/`.

*(This is the second rule, adopted 2026-08-08. The first said the server only
compiled and ran, and the caller generated every input — read that way, the
sentence "it does not warm-start" was true of the same server four days
earlier.)*

**The contract:** *every tool returns quickly or returns a job id.* An MCP
client opens a fresh session per call and tearing it down kills the server's
children, so a ten-minute CIME build inside a call is not a slow call — it is a
half-built case directory. Both slow stages sbatch and hand back an id.

So the ELM loop is always: **submit → poll `check_elm_job` → act on `active:
false`**. Claude is good at this; ask it to "check every couple of minutes until
it lands" and it will.

### `reaction` — PFLOTRAN, 44 tools

**Which tree is live:** `/qfs/people/tran289/IDEAS/reaction_sandbox_mcp-upstream`,
branch `compy-port` — upstream `main` (`bb773fd`, 2026-08-05) plus our four
tools and a packaging fix, installed into the `ideas` env so `.mcp.json` can
launch it as the console script `pflotran-mcp`. The older unzipped
`reaction_sandbox_mcp-main/` is no longer wired to anything, including via
`.env` (fixed 2026-08-06): `LAMBDA_PFLOTRAN_DIR` now points at a copy of the
Lambda stub living inside `-upstream` itself, and `JAX_PFLOTRAN_SANDBOX_DIR`
at `-upstream`'s own tracked `jax_surrogate/` (aliased via a
`jax_pflotran_sandbox` symlink, since `tools/jax_surrogate.py` imports it
under that name). `-main` stays on disk only as a rollback/reference; nothing
reads from it at runtime.

The ones that matter for column work:

```
describe_pflotran_capabilities()                READ THIS FIRST — the model, the constraints,
                                                and `working_order`: the rules + call order for
                                                driving this server by hand
check_installation()                            every env var this server reads, resolved
create_column_deck(column, out_dir, ...)        a RUNNABLE 1-D soil column deck
create_decks_from_columns(columns, out_dir, ..) one deck per column; with site_dir it joins the
                                                site data itself and returns a run plan
describe_conceptual_factors()                   the CONTROLLED-SWEEP menu (water table, soil,
                                                soil depth, recharge, written rain)
check_conceptual_design(design)                 will a sweep build — every reason named, pre-compute
build_conceptual_columns(design)                the sweep as columns, in the deck tool's shape
create_pflotran_input(...)                      a deck SKELETON (grid/time/output only) — says runnable=False
configure_reaction_sandbox(input_file, ...)     add LAMBDA/microbial/CLM-CN chemistry
validate_pflotran_input(input_file)             TEXT SCAN + `runnable` (are the blocks a run needs there?)
check_deck_reads(input_file, timeout)           PFLOTRAN ITSELF reads a COPY of the deck and returns
                                                its own error lines — run it on any hand-touched deck
run_pflotran_simulation(input_file, ...)        run it HERE, inline (seconds); num_cores defaults to 1
check_simulation_status(output_dir, prefix)     completed / running (with progress) / failed, from the .out
extract_column_series(cases, out_file)          profiles per column (saturation, pressure vs depth), to a file
extract_observations(output_file, variables)    time series out of -obs-0.tec / .h5
create_parameter_ensemble(...)                  LHS/Sobol parameter sets
```

**The working order, short form** (the full version rides
`describe_pflotran_capabilities().working_order` — this exists because a
hand-written deck that "validated" and PFLOTRAN refused is the main way a
session here goes wrong):

1. Never hand-write or hand-edit a deck — the builders write runnable ones.
2. If a deck was hand-made anyway, `check_deck_reads` must say `reads: true`
   before it is worth a real run. `validate_pflotran_input` is only a text
   scan: its `is_valid` does not mean runnable, and neither catches a wrong
   keyword — PFLOTRAN does, and this tool asks it.
3. Never state a capability, keyword, or default from memory — read the
   capabilities report, the sweep menu, or the tool docstrings.
4. Judge a run by `exit_codes` (0 = clean) or `check_simulation_status` with
   `prefix=<deck name>`; always pass `timeout`.

The other 30-odd tools cover the LAMBDA DOM-respiration pipeline
(preprocessing, binning, thermodynamics), DART assimilation, particle
tracking, and the JAX/KIM surrogates; `list_available_samples` is the cheap
way to see what the LAMBDA side has on disk.

`create_column_deck` vs `create_pflotran_input`: the second writes simulation
type, grid, time and output and stops — no materials, no regions, no strata, no
flow conditions, so PFLOTRAN rejects it at read time. The first writes the
physics too, and is what you want for a soil column.

### The environment rule — why `.mcp.json` has `env` blocks

**Which variables reach a stdio server depends on the client, and the two we
use disagree.** The Python SDK's `get_default_environment()` forwards only
`HOME`, `LOGNAME`, `PATH`, `SHELL` and `USER` — not `LD_LIBRARY_PATH`, not
`MODULEPATH`, nothing `env_compy.sh` exports. Claude Code forwards the parent
environment instead (measured: with `env_compy.sh` sourced, the reaction server
saw that shell's `MPI_COMMAND`; without it, it fell back to its own `.env`).
Our own `MCPManager` forwards all of `os.environ`.

So a server can work through one launcher and fail under another on the same
machine, in both directions: a path the framework supplied can go missing under
a plain client, and a variable your shell exports can silently override what
the server configured for itself. Both servers therefore resolve every path
they need *themselves*, and the sources rank:

```
.mcp.json "env" block   >   the launching shell   >   the server's own .env / defaults
```

The `elm` server sets its defaults in code; the `reaction` server uses
upstream's `.env` (see `.env.example` there for every variable it reads).
`.mcp.json` overrides only what must not vary:

| variable | server | default |
|---|---|---|
| `IDEAS_FRAMEWORK_DIR` | elm | the repo this file lives in |
| `E3SM_SRC_DIR` | elm | `/qfs/people/tran289/E3SM` |
| `PSCRATCH` | elm | `/compyfs/tran289` |
| `IDEAS_SLURM_ACCOUNT` / `_QUEUE` | elm | `e3sm` / `short` |
| `MODULEPATH`, `MODULESHOME` | elm | Compy's five module trees |
| `PFLOTRAN_EXECUTABLE` | reaction | `.env` → `/qfs/people/tran289/pflotran/src/pflotran/pflotran` |
| `MPI_COMMAND` | reaction | `.env` → `/share/apps/openmpi/4.0.1/gcc/10.2.0/bin/mpirun` |
| `LAMBDA_PFLOTRAN_DIR` | reaction | `.env` → the stub in `reaction_sandbox_mcp-upstream/lambda_pflotran_refactor` |
| `JAX_PFLOTRAN_SANDBOX_DIR` | reaction | `.env` → `reaction_sandbox_mcp-upstream` (root, so `jax_pflotran_sandbox` resolves as a package) |

On another machine, edit the reaction server's `.env` and export the `elm` ones
before starting `claude`.

**Give absolute paths, even for things on your PATH.** `env_compy.sh` used to
export `MPI_COMMAND=mpirun`; because Claude Code forwards the shell, that bare
name overrode the absolute path in `.env`, and it only resolves while the
openmpi module is loaded. It is now the full path — same binary, one fewer way
to be wrong.

---

## Part 3 — eight independent examples

Each is self-contained: the prompt you type, the tool call it produces, the
real output, and what to check afterwards.

---

### Example 1 — one PFLOTRAN column, steady recharge (2.4 s, no scheduler)

*The smallest thing that is still real science: a layered 1-D column, hydrostatic
at its water table, with 150 mm/yr on top for 20 years.*

**You type:**

```
Build a 1-D PFLOTRAN column with the reaction MCP and run it here.
Site: water table 8 m down, three horizons —
  0-30 cm   loam:      theta_s 0.43, theta_r 0.045, alpha 1.45 /m, m 0.627, Ksat 8.25e-5 m/s
  30-120 cm silt loam: theta_s 0.41, theta_r 0.065, alpha 0.75 /m, m 0.471, Ksat 1.23e-5 m/s
  120-300 cm clay loam: theta_s 0.39, theta_r 0.10, alpha 0.59 /m, m 0.312, Ksat 2.89e-6 m/s
Steady recharge 150 mm/yr, 20 years, write it under /compyfs/tran289/demo2.
Validate the deck before running it, and tell me the saturation profile at the end.
```

**What Claude calls** — `mcp__reaction__create_column_deck`:

```json
{"column": {"id": "demo_col", "fan_wtd_m": 8.0,
            "soil_profile": {"layers": [
              {"component": "demo", "depth_top_cm": 0, "depth_bot_cm": 30,
               "van_genuchten": {"theta_s": 0.43, "theta_r": 0.045,
                                 "alpha_per_m": 1.45, "m": 0.627, "ksat_ms": 8.25e-5}},
              {"component": "demo", "depth_top_cm": 30, "depth_bot_cm": 120,
               "van_genuchten": {"theta_s": 0.41, "theta_r": 0.065,
                                 "alpha_per_m": 0.75, "m": 0.471, "ksat_ms": 1.23e-5}},
              {"component": "demo", "depth_top_cm": 120, "depth_bot_cm": 300,
               "van_genuchten": {"theta_s": 0.39, "theta_r": 0.10,
                                 "alpha_per_m": 0.59, "m": 0.312, "ksat_ms": 2.89e-6}}]}},
 "out_dir": "/compyfs/tran289/demo1", "recharge_mm_yr": 150.0, "years": 20.0}
```

**Returned** (verified):

```json
{"case_dir": ".../demo1/demo_col", "input_file": ".../demo_col/demo_col.in",
 "n_cells": 11, "depth_m": 13.0, "water_table_m": 8.0, "wt_in_domain": true,
 "soil_horizons": 3, "recharge_mm_yr": 150.0, "transient": false,
 "validation_status": "success"}
```

then `validate_pflotran_input` → `{"errors": [], "warnings": [], "is_valid": true}`,
then `run_pflotran_simulation` with `{"input_file": "...", "num_cores": 1, "timeout": 300}` →

```json
{"exit_codes": [0], "execution_time": 2.38,
 "output_files": [".../demo_col-000.tec", ... , ".../demo_col-004.tec"]}
```

**What to check.** `wt_in_domain: true` — if it is false the column never
saturates and you are answering a different question (see Part 4). `n_cells: 11`
for a 13 m domain is three horizon cells plus a geometrically coarsening
substrate; resolution lives near the surface where the wetting front is.
`exit_codes: [0]` is the only success signal that means anything — see the
`check_simulation_status` trap in Part 4.

**Reading the answer.** The `.tec` files are TECPLOT POINT snapshots, one per
output time, columns `X Y Z Pressure Saturation MaterialID`. Ask Claude to plot
saturation vs depth from `demo_col-004.tec` (the 20-year state), or point it at
`tools/analyze_pflotran_run.py`, which already turns these into water-table
depth through time.

---

### Example 2 — a PFLOTRAN ensemble through Slurm (REMOVED 2026-08-18)

The three scheduler tools this example used — `submit_pflotran_ensemble`,
`check_pflotran_job`, `collect_pflotran_results` — were deleted from the
server (commit `fac929a`): they were ours, nothing called them, and a tool
that can `sbatch` behind the conversation is what nobody wanted. A framework
column solves in 0.3-3 s, so `run_pflotran_simulation` inline (Example 1) is
the path. For a long or wide reactive ensemble, write the batch script
yourself and ask before submitting it; inside the job, `tools.simulation.
run_simulation(mode="ensemble_parallel", timeout=...)` still runs a list of
decks and returns `results_by_input`, so each deck's outcome stays attributed.

**One argument you should always pass:** `timeout`. Without it a single
non-converging column stalls the whole call.

---

### Example 3 — a column driven by an ELM run (one-way coupling, REBUILT 2026-08-19)

*This is the coupling the project exists for: ELM does the surface
partitioning, PFLOTRAN does the deep fate.*

The old form of this example — `elm_daily_flux()` in
`tools/build_pflotran_cases.py`, a local deck builder, `--bottom fan` — is
GONE: that path bypassed the model server and its results never reached the
Analyzer. Coupling is the framework's **coupling archetype** now, and it is
one request:

```
python workflow.py --request "A coupling follow-up: drive PFLOTRAN with the
sub-surface drainage (QDRAI) from the prior ELM run <elm_run_...>. Reuse that
run's columns exactly, anchored at ELM's own solved water table."
```

What happens, and where each fact lives:

- **Reception** (coupling archetype) gathers NOTHING new — the prior run's
  grid, observations and rain are carried into this run's `reception.json`,
  its site files are copied beside it, and the provenance says
  `CARRIED FORWARD, NOT FETCHED`.
- **The PFLOTRAN manager's** `_build_coupled_columns` turns the prior run's
  RECORD (experiment.json + extracted.json) into columns: the prior's columns
  **verbatim** (same ids, coordinates, bands, pinned stations), each carrying
  its own daily `QDRAI` series (`daily_flux_mm_day`, mm/day) and **ELM's own
  solved water table** as its anchor — not CONUS2's.
- **The server's** `create_decks_from_columns` sees a column that carries its
  flux and water table, joins only the subsurface, and writes the column's
  `flux_description` into the deck's `forcing_caveat` — so the record never
  claims "no ET was removed" of a flux that had ET removed by ELM.
- **The Analyzer** runs on the coupled run like any other; the two runs line
  up column by column.

Driving one column by hand through the MCP is still possible — build the deck
with `create_column_deck(recharge_series=...)` — but the ensemble path above
is the framework's and keeps every fact on the record.

When a column's ELM `QDRAI` is ~0 (a deep ELM water table), its PFLOTRAN
column is driven by ~nothing: the deck builds STEADY at the near-zero mean and
its row says so. The two models agreeing about which sites are dry-at-depth is
a consistency check, not a coincidence.

---

### Example 4 — reactive transport with the LAMBDA sandbox (2.7 s, verified)

*The reaction sandbox is what this server was originally built for: organic-
matter respiration chemistry on top of a flow deck. This runs today — but only
by the route below, and the two routes that look more obvious are both dead
ends. That is most of what this example is for.*

**You type:**

```
Build the coupled Richards + LAMBDA reactive-transport deck for col_01 at
100 mm/yr with tools/build_reactive_demo.py, then run it with the reaction MCP
and tell me where the oxygen is consumed.
```

**What happens.** The deck comes from our own builder (Bash), because it is the
one thing on this machine that writes a LAMBDA block this PFLOTRAN accepts:

```bash
python3 tools/build_reactive_demo.py --column col_01 --recharge 100 10
```

Then the MCP runs it — `mcp__reaction__run_pflotran_simulation`:

```json
{"input_file": ".../workflow_outputs/gunnison_reactive/r100/col_01_r100.in",
 "num_cores": 1, "timeout": 600}
```

(the timing below was measured on a copy of that deck under
`/compyfs/tran289/reactive_demo` — same file, somewhere writable)

**Returned** (verified 2026-08-03, through a real MCP client):

```json
{"exit_codes": [0], "execution_time": 2.73, "validation_status": "success"}
```
with five `.tec` snapshots, and `validate_pflotran_input` clean beforehand
(`CHEMISTRY`, `TRANSPORT_CONDITION` and `CONSTRAINT` all present in
`blocks_found`). The science it shows: aerobic CH2O oxidation concentrated in
the infiltration zone, pushed deeper at higher recharge — compare `r100` with
`r10`.

**The LAMBDA block this PFLOTRAN accepts**, straight from
`src/pflotran/reaction_sandbox_pnnl_lambda.F90`: `REACTION_NETWORK`, `MU_MAX`
(1/sec), `VH` (**m^3** — a volume, not a concentration), `CC` (M), `K_DEG`
(1/sec), `NH4_INHIBIT` (M), `INHIBITION_TYPE` (THRESHOLD | SMOOTHSTEP),
`SCALING_MINERAL`, `CARBON_CONSUMPTION_SPECIES`, `ACTIVATION_ENERGY` (J/mol),
`REFERENCE_TEMPERATURE`. Write new decks against **that** list.

**Dead end 1 — the sample-driven path.** `list_available_samples` returns
`n_samples: 0` on Compy (checked 2026-08-03), so `create_pflotran_input(...,
lambda_sample_id="SPS_0001")` has nothing to auto-discover: no Phase 1/2
outputs are on this machine, and `run_lambda_preprocessing` needs the
Lambda-PFLOTRAN-Refactor repo that is still NERSC-only. The tools answer, they
just answer empty.

**Dead end 2 — the shipped example deck.** `batch_reaction_lambda_network.in`
in `reaction_sandbox_mcp-main/` was written for a *different* LAMBDA build.
Ported against our v7.0 binary it fails five times over — `VH 1.0e-6 M` (wants
`m^3`), then `O2_THRESHOLD`/`TEMPERATURE`/`F_ACT`/`LOG_FORMULATION` (not
keywords here), then `SATURATION_FUNCTION` (v7.0 wants `CHARACTERISTIC_CURVES`),
then `pH` in an OUTPUT VARIABLES list, then `PRESSURE` (deprecated, use
`LIQUID_PRESSURE`) — and then stops for good at its ten `C##-DONOR` species,
which exist in no database on this machine. Do not start from it.

**Keep the simulated duration at or under ~9 years.** The demo deck hits a hard,
reproducible NaN wall at exactly t = 9.31319 yr: biomass washes out under the
demo's `k_deg`/`MU_MAX` and the log formulation goes singular. It is a
deck-parameterisation limit, not a build problem — PFLOTRAN's own
`reaction_sandbox_lambda` regression case passes clean at its native 21 days.

**A real reactive ensemble wants a batch job you submit yourself**, not the
inline path. Two seconds is the demo; chemistry over a deep column and a long
record turns into hours, and that cannot run inside an MCP call at all,
whatever the timeout says. (The server's own scheduler tools were removed
2026-08-18 — see Example 2.)

---

### Example 5 — is this machine ready for ELM? (instant)

*Run this before anything expensive. It is the cheapest tool on either server
and it answers the question that otherwise surfaces eight minutes into a build.*

**You type:**

```
Call describe_elm_capabilities. Is everything present?
```

**Returned** (verified 2026-08-03, abridged):

```json
{"server": "elm", "ready": true, "missing_requirements": [], "broken_imports": [],
 "requirements": {
   "framework":   {"path": ".../multi-agent-framework", "present": true},
   "e3sm_source": {"path": "/qfs/people/tran289/E3SM",  "present": true},
   "cime":        {"path": "/qfs/people/tran289/E3SM/cime", "present": true},
   "scratch":     {"path": "/compyfs/tran289", "present": true},
   "sbatch":      {"path": "/usr/bin/sbatch", "present": true}},
 "does_not": ["choose where to put columns ...", "warm-start anything ...",
              "generate surface or domain files", "read history files ...",
              "decide whether a study is worth running ..."]}
```

Every entry is **checked, not asserted** — a capabilities tool that lists its
assumptions is worth nothing on the machine where one of them is false, which
is exactly the machine it exists for.

---

### Example 6 — build and run a 2-column ELM ensemble (the split flow)

*Two jobs, and a poll in between, so you can look at the cases before spending
node time on them.*

**Prerequisite — the one thing this server will not do for you.**
`build_elm_cases` reads `<run_dir>/01_inputs/case_inputs.json` and nothing else.
That file is a list, one entry per column, each carrying a `runtime_config` of
absolute paths that already exist:

```json
[{"case_name": "col_01",
  "runtime_config": {
    "FSURDAT": "/qfs/people/tran289/IDEAS/1d_elm/input_files/surfaces/Surfacedata_38.6292_-107.5875_native_extrapolate_soil-conus_96469d.nc",
    "FINIDAT": ".../workflow_outputs/elm_run_20260730_113010/warmstart/finidat_col_01.nc",
    "LND_DOMAIN_FILE": "Domainfile_38.6292_-107.5875.nc",
    "LND_DOMAIN_PATH": "/qfs/people/tran289/IDEAS/1d_elm/input_files/domains",
    "ATM_DOMAIN_FILE": "Domainfile_38.6292_-107.5875.nc",
    "ATM_DOMAIN_PATH": "/qfs/people/tran289/IDEAS/1d_elm/input_files/domains",
    "STOP_N": "1", "STOP_OPTION": "nyears", "RUN_STARTDATE": "2020-01-01",
    "DATM_CLMNCEP_YR_START": "2020", "DATM_CLMNCEP_YR_END": "2020",
    "REST_N": "1", "REST_OPTION": "nyears"}}]
```

`FSURDAT` and `FINIDAT` are the warm start's two products — the donor
gridcell's soil and its equilibrated state. They are decisions about the
science, not compilation steps, which is precisely why they are inputs here.

Three ways to get the file:

* **the framework writes it** — a `python3 workflow.py --interactive --model elm`
  run does the sampling design, the warm start and the input generation, and
  `_save_case_inputs` drops `case_inputs.json` into the run dir on its way past;
* **by hand from existing inputs** — the file is a list of the JSON above, and
  the pieces come from `tools/expand_sampling.py` (columns) →
  `tools/make_warmstart.py` (FINIDAT + donor soil) → `tools/build_column_inputs.py`
  (surfaces and domains);
* **copy a ready-made one** — `workflow_outputs/_unattended_check/run/01_inputs/case_inputs.json`
  is a working 2-column Gunnison pair, and copying it into a fresh run dir is
  exactly how this example was set up. Drop the `case_dir` field if an entry
  carries one: that is a record of where a previous build landed, not an input.

**You type:**

```
In workflow_outputs/mcp_demo_elm I have 01_inputs/case_inputs.json with two
columns. Build the CIME cases on the short queue with an hour of walltime, poll
until the build lands, then run both columns as one ensemble and poll that too.
Tell me the case directories and whether every column produced history files.
```

**The four calls, with the real outputs from jobs 770826 and 770827
(2026-08-03):**

```jsonc
// 1. build_elm_cases
{"run_dir": ".../workflow_outputs/mcp_demo_elm", "queue": "short", "walltime": "01:00:00"}
// -> {"job_id": "770826", "n_cases": 2, "stage": "build_cases",
//     "log_path": ".../mcp_demo_elm/build_cases.log", "next": "check_elm_job"}

// 2. check_elm_job  (every couple of minutes)
{"job_id": "770826", "run_dir": ".../workflow_outputs/mcp_demo_elm"}
// still going -> {"state": "RUNNING", "active": true, "scheduler_answered": true}
// landed     -> {"state": "COMPLETED", "active": false, "stage": "build_cases",
//                "ready": true, "ok": true, "n_ok": 2, "n_total": 2,
//                "cases": [{"case_name": "col_01",
//                  "case_dir": "/compyfs/tran289/E3SMv3/1D_ELM.b198763.2026-08-03-230406.col_01"},
//                           {"case_name": "col_02",
//                  "case_dir": "/compyfs/tran289/E3SMv3/1D_ELM.b198763.2026-08-03-231051.col_02"}]}

// 3. run_elm_ensemble — the cases are built, so this skips the compile and
//    goes straight to the columns. (Until 2026-08-10 this step was a separate
//    tool, submit_elm_ensemble, which took the case_dirs from step 2 as an
//    argument. It no longer needs them: the job reads built_cases.json.)
{"run_dir": ".../workflow_outputs/mcp_demo_elm",
 "queue": "short", "walltime": "00:40:00"}
// -> {"job_id": "770827", "n_cases": 2, "stage": "ensemble",
//     "log_path": ".../ensemble_A.log"}

// 4. check_elm_job again — branch on state/active, and read run.log for the columns
{"job_id": "770827", "run_dir": ".../workflow_outputs/mcp_demo_elm"}
// -> {"state": "COMPLETED", "active": false, "scheduler_answered": true, ...}
```

```
$ tail -3 workflow_outputs/mcp_demo_elm/run.log
  1D_ELM.b198763.2026-08-03-231051.col_02: rc=0 history=9
  1D_ELM.b198763.2026-08-03-230406.col_01: rc=0 history=9
ALL_DONE in 11min
```

**Measured, this run:** build 436 s for 2 cases (one full compile, the second
cloned with `--keepexe`), run 690 s for both columns concurrently, 1126 s
end to end including the polling. Scaling is the point of the design — a
19-column ensemble took 2406 s of run time, because the columns run
concurrently on one node, not because each one is fast.

**What to check.** `n_ok` equals `n_total` after the build. Every column in
`run.log` shows `rc=0` **and** a non-zero `history=` — a column that "COMPLETES"
in 2 s with `history=0` did not run (Part 4). And `active: false` **with**
`scheduler_answered: true`: when the scheduler cannot be reached the server
reports `state: null, active: true`, because a `squeue` timeout must never be
read as a finished ensemble.

**One sharp edge in step 4.** Once `01_inputs/built_cases.json` exists,
`check_elm_job` with a `run_dir` answers with the *build's* payload — `stage:
"build_cases"`, `ready: true`, the case list — even when the job id you asked
about is the ensemble. `state`, `active` and `scheduler_answered` still describe
the job you asked about, so branching on `active` is correct; just do not read
`ready`/`ok`/`cases` as statements about the run. For an unambiguous answer on a
run job, call `check_elm_job` **without** `run_dir` and read `run.log` yourself.

**Reading the results is your job, not the server's** — that boundary is
deliberate. Point Claude at `tools/analyze_run.py --run-dir <rd> --cases-file
cases.json --plot`, or ask it to open the history files directly with xarray.

---

### Example 7 — the whole ELM study unattended (jobs A and B)

*Use this when you want to close the laptop. One call builds and runs
everything; a second job you submit yourself reports when it lands.*

**You type:**

```
Run the study in workflow_outputs/mcp_demo_elm end to end — short queue, two
hours — then submit a follow-up job that depends on it and mails
hoang.tran@pnnl.gov when it finishes.
```

**The call** — `mcp__elm__run_elm_ensemble`:

```json
{"run_dir": ".../workflow_outputs/mcp_demo_elm", "queue": "short",
 "walltime": "02:00:00"}
```

```json
{"job_id": "...", "n_cases": 2, "stage": "ensemble",
 "log_path": ".../ensemble_A.log",
 "next": "submit job B with --dependency=afterany:<id>, then exit"}
```

**Then job B, which is yours, not the server's:**

```bash
sbatch --dependency=afterany:<A> ensemble_B.sbatch   # analysis + mail
```

**Why two jobs and not one.** The tool this replaces, `run_elm_study`, ended
its job by running the framework's `workflow.py --finalize` — the server's job
executing the client's code, a circular dependency the boundary rule forbids.
A and B invert it: the caller asks for the model run and arranges its own
follow-up, and nothing inside the server calls back out. The cost to you is
identical — one sitting, one email — because SLURM presses the second button.

**`afterany`, never `afterok`.** With `afterok` a failed ensemble means B never
runs and *no mail is ever sent*, which is the silent failure that happened twice
on 2026-08-06. `afterany` means B always runs and always reports, including "the
ensemble failed, here is why". Slurm mail is confirmed delivering to @pnnl.gov
from Compy (job 770794) — it goes through the scheduler, so it does not need an
MTA on the compute node.

**Walltime covers build + run.** ~8-10 min of compile, then the columns. Two
hours is the `short` partition's ceiling and fits a 19-column study. Anything
bigger: `queue: "slurm"` (4-day limit) and a longer walltime. B is separate and
needs minutes, not hours.

**It is restartable, and it now means it.** Job A skips the compile when
`built_cases.json` reports success *and* every case directory it names is still
on disk — one missing directory condemns the whole manifest and triggers a
rebuild, rather than launching columns at an executable that is not there. Until
2026-08-10 the tool deleted that file before every submission, so the "reuse"
check could never fire and each resubmission paid the ~7 min compile again.

---

### Example 8 — both servers in one sitting

*What a real morning looks like: ask ELM for the surface fluxes, hand them to
PFLOTRAN, compare.*

**You type:**

```
Start the ELM study in workflow_outputs/mcp_demo_elm as one unattended job and
mail me. While it runs, build PFLOTRAN columns for the same two sites from
columns.json at 100 mm/yr steady, run them inline, and tell me where each
water table settles. When the ELM job lands, rebuild the PFLOTRAN columns with
the real QINFL series and tell me what changed.
```

This is one conversation using both servers, and it works because of the
contract in Part 2: `run_elm_ensemble` returns a job id in seconds, so nothing is
blocked while the PFLOTRAN columns run inline in the same session. Claude polls
`check_elm_job` between the PFLOTRAN calls.

The scientific point of the sequence: the steady-recharge columns are a
*control* for the coupled ones. The steady run answers "where would the water
table sit under a climatological average", the ELM-driven run answers "when does
this year's infiltration actually arrive at the water table" — and the
difference between them is the travel-time signal the coupling exists to
measure.

---

## Part 4 — the traps, all of them found the hard way

**A tool that works for us can fail for you, on the same machine.** Only
`HOME`, `LOGNAME`, `PATH`, `SHELL` and `USER` reach a stdio server. Two real
instances, both fixed in the servers themselves on 2026-08-03: the ELM build
died 60 s in with `python3: error while loading shared libraries:
libpython3.11.so.1.0` (the first `python3` on a stock PATH needs an
`LD_LIBRARY_PATH` nobody forwarded, and CIME's `create_newcase` is
`#!/usr/bin/env python3`), then died again 30 s in with `ERROR: No module path
defined` (CIME loads its compiler modules through `modulecmd`, which needs
`MODULEPATH`). If you add a tool that shells out, resolve its environment in
the server, with a default.

**`check_simulation_status` used to say `running` for a finished run** — it
grepped for `*** SIMULATION COMPLETED`, which PFLOTRAN v7.0 never prints.
FIXED 2026-08-18 (`fac929a`): it now reads the `Wall Clock Time` line for
completed, the `Step N Time=` lines against the deck's `FINAL_TIME` for
progress, `ERROR` lines for failed, and falls back to the one `.out` in the
directory when `prefix` is not the deck's name. `exit_codes` from
`run_pflotran_simulation` remains the primary word.

**`extract_observations` does not read snapshot `.tec` files.** It wants an
observation *time series* — `-obs-0.tec` or `.h5`. Pointed at a
`demo_col-004.tec` snapshot it returns `Extraction failed: "['Time'] not in
index"`. Column decks from `create_column_deck` emit snapshots, so read those
directly (six columns: X Y Z Pressure Saturation MaterialID) or use
`tools/analyze_pflotran_run.py`, which already does.

**Never put a run directory on `/tmp`.** `/tmp` is node-local: a job submitted
from a login node cannot see it, and the ensemble silently finds nothing. Use
`/compyfs/tran289` (`$PSCRATCH`) or the repo's `workflow_outputs/`.

**`rc=0` is not proof an ELM column ran.** If you ever see `rc=0 history=0`,
distrust `rc` and read `<case>/run/srun.out`. Two Compy-specific causes, both
fixed in `mcp/elm-mcp/scripts/ensemble_ab.sh` but worth knowing: `srun --exact` does not
exist on Slurm 18.08 (use `--exclusive`), and Intel-MPI cases need
`srun --mpi=pmi2` or they hang in bootstrap.

**Deep water tables are not a bug.** ELM's hydrologically active soil is
0-3.8 m; a warm-started column can legitimately come back with ZWT of 20-69 m
and negative `WA`. For those columns `QDRAI` stays ~0 and that is correct — no
drainage at the bottom of a 3.8 m column when the water table is 66 m down.
Do not clamp it, and do not ask PFLOTRAN to deepen its domain past
`depth_cap` to hide it.

**`mpirun` is not on a stock PATH here.** It does not matter for 1-D columns,
because `run_pflotran_simulation` with `num_cores: 1` executes the binary
directly (and the PFLOTRAN build has its PETSc/HDF5 paths baked in as RPATHs,
which is why it needs no `LD_LIBRARY_PATH`). For `num_cores > 1`, set
`MPI_COMMAND` to an absolute path or load the MPI module before starting
`claude`.

**The LAMBDA example decks in `reaction_sandbox_mcp-main/` do not run on this
PFLOTRAN**, and `list_available_samples` is empty here. Reactive transport
*does* run — from decks written by `tools/build_reactive_demo.py`. The five
incompatibilities, the keyword set v7.0 accepts, and the 9.3-year NaN wall are
all in Example 4.

**Login-node etiquette.** Inline PFLOTRAN runs execute in the server's own
process tree — on a login node. Two or three tiny columns are fine; forty
concurrent `mpirun`s are not. Past a handful, submit.

**Long calls are not slow, they are dead.** Every MCP client has a per-call
timeout and tearing the session down kills the server's children. That is why
neither server has a tool that blocks on the science, and why you should never
"fix" a timeout by raising it past a few minutes.

---

## Part 5 — crib sheet

```bash
# pick a backend (§2b) — nothing sourced = your own subscription
source /qfs/people/tran289/IDEAS/env_compy.sh        # PNNL AI Incubator
source /qfs/people/tran289/IDEAS/claude_bedrock.sh   # AWS Bedrock (creds expire)

# start
cd /qfs/people/tran289/IDEAS/multi-agent-framework && claude
/mcp                      # are elm and reaction connected?

# ELM, split (build first, inspect the cases, then run them)
"describe_elm_capabilities"                              # ready: true?
"build_elm_cases on <run_dir>, short queue, 1 h"         # -> job id
"check_elm_job <id> with run_dir <run_dir>"              # -> cases when done
"run_elm_ensemble on <run_dir>, 40 min"                  # -> job id (skips the compile)
"check_elm_job <id> again"                               # -> state/active
tail -3 <run_dir>/ensemble_A.log                         # -> rc= and history= per column

# ELM, unattended (jobs A + B)
"run_elm_ensemble on <run_dir>, 2 h"                     # -> job id  = A
sbatch --dependency=afterany:<A> ensemble_B.sbatch       # your job B: report + mail

# PFLOTRAN, inline
"create_column_deck for <site> at <wtd> m, <r> mm/yr, <y> years, out_dir <dir>"
"validate_pflotran_input on it"
"run_pflotran_simulation, num_cores 1, timeout 300"      # exit_codes: [0]

# PFLOTRAN, reactive (LAMBDA) — build the deck with our builder, run it with the MCP
python3 tools/build_reactive_demo.py --column col_01 --recharge 100 10
"run_pflotran_simulation on that deck, num_cores 1, timeout 600"

# headless
claude -p "<prompt>" --mcp-config .mcp.json --strict-mcp-config \
       --allowedTools "mcp__elm mcp__reaction" --output-format json
```

**Further reading in this repo:** `docs/ELM_MCP_PLAN.md` (why the ELM server has
the boundary it has), `docs/PFLOTRAN_PLAN.md` (the coupling science),
`docs/RUNBOOK.md` (the same work driven by hand, without Claude),
`ARCHITECTURE.md` (where the MCP servers sit in the framework).
