#!/usr/bin/env python3
"""
ELM Experiment Manager
mcp/elm-mcp/src/elm_exp_manager.py

Single responsibility: orchestrate the ELM execution pipeline — the
Experiment Manager box of the framework (see ARCHITECTURE.md).

NOT AN MCP TOOL, despite living in the server's src/. The seven tools are
registered in mcp/elm-mcp/main.py; this is the framework's per-model manager
class, which moved here with the rest of the ELM code in phase 1b and is
imported back across the boundary by core/backends.py. It is deleted after
phase 4 (docs/ELM_MCP_PLAN.md §1e) — the input half is already a shim around
one MCP call, and what remains is the SLURM job-A/job-B orchestration that has
to live outside a tool, because a tool must return fast or return an id.

Output directory structure:

    elm_run_YYYYMMDD_HHMMSS/
        ├── columns.json  run_plan.json  plan.json  reception_brief.json
        ├── assumptions.json                        (the honesty ledger)
        ├── sampling_design.png                     (step 0, materialize)
        ├── warmstart/                              (step 0b, only if warm)
        │   ├── Warmstart_<col>.nc
        │   └── warmstart.json
        ├── 01_inputs/
        │   └── experiment_summary.json
        ├── 03_results/
        │   ├── execution_report.txt  results_summary.csv
        │   └── extracted.json          (the record of what was READ)
        ├── 04_analysis/
        │   ├── comparison.json  investigation.json  interpretation.json
        │   ├── analysis.json           (the Analyzer's boundary file)
        │   ├── partitioning.png  controls.png  soil_control.png
        │   └── wtd_columns.png
        │   ├── <col>/<col>.in + outputs
        ├── experiment.json             (this box's product)
        └── RUN_SUMMARY.json

ANALYSIS_REPORT.json and LLM_ANALYSIS_INPUT.json used to sit at the top level.
Both belonged to the one-shot report agent, deleted 2026-08-13; the Analyzer
writes 04_analysis/analysis.json instead. validation.json is listed by no
current writer either.

ELM cases live at: $PSCRATCH/E3SMv3/1D_ELM.*/
"""
import csv
import glob
import os
import json
import sys
from pathlib  import Path
from datetime import datetime
from typing   import Dict, Any, List, Optional

sys.path.insert(0, "src")

from core.exp_manager_base import ExperimentManagerBase, Pending
from core.keyset import KeySet


# _ROOT and _load_tool ARE GONE (2026-08-19) with step 4d: their only user
# was the PFLOTRAN coupling CLIs, and a coupling is a run of the PFLOTRAN
# manager now — see exp_manager_base._is_coupling.


# ─────────────────────────────────────────────────────────────────────
# ELM EXPERIMENT MANAGER
# ─────────────────────────────────────────────────────────────────────
class ELMExpManager(ExperimentManagerBase):
	"""
	Executes ELM experiment plans. EVERY ELM COMPUTATION IS THE MCP'S.

	What is left here is the four things that are not ELM knowledge: WHEN to
	call the server (the stage order is the base's), WHERE the run directory
	is, WHAT the derived fields mean (FIELD_SEMANTICS), and arranging the
	framework's own follow-up job. The stages read:

	    materialize        one MCP call — warm start, donor soil, surfaces,
	                       case_inputs.json, the run plan  (_refine_columns)
	    build_case_inputs  read back what that call wrote
	    build_cases        submit JOB A (build + run), submit JOB B, stop
	    run                collect from disk — A already ran the columns
	    extract            read the history NetCDFs
	    couple             optional one-way handoff to PFLOTRAN

	THERE IS NO LOCAL PATH. It was deleted 2026-08-10: _refine_columns requires
	the `elm` client, so every branch that ran ELM by import was unreachable
	from the first stage onward and only looked like a fallback. With it went
	_run_batch, _submit_via_mcp, the ELMExperimentBuilder handle, and the last
	use of tools/submit_cases.sh from this side of the boundary.

	Figures, observation validation and interpretation are NOT here — they are
	the Analyzer (src/agents/analyzer.py), which execute_plan invokes after
	extraction.
	"""
	MODEL = "elm"

	def __init__(self,
				 base_output_dir: str = "./workflow_outputs",
				 run_dir: str = None):
		super().__init__(base_output_dir, run_dir)
		print(f"\n{'=' * 60}")
		print(f"ELM Experiment Manager")
		print(f"Run dir : {self.run_dir}")
		print(f"Cases   : $PSCRATCH/E3SMv3/")
		print(f"{'=' * 60}\n")

	# ─────────────────────────────────────────────────────────
	# BACKEND HOOKS — what the base cannot know about ELM
	# ─────────────────────────────────────────────────────────
	def _already_executable(self, plan: Dict[str, Any]) -> bool:
		"""CONDITIONS_COUPLERS is ELM's executable payload (the legacy shape)."""
		return bool(plan.get("CONDITIONS_COUPLERS"))

	def _build_sweep_columns(self, design: Dict[str, Any],
							 config: Dict[str, Any]) -> Dict[str, Any]:
		"""A controlled sweep's columns, built by the server that runs them.

		The base knows a sweep is one column per level combination. Only this
		server knows that a soil_texture level of 27 becomes a synthetic
		surface dataset with those percentages in it, or that two coordinates
		in one NLDAS cell would share their weather exactly — so the design
		crosses the MCP boundary and comes back as columns.

		CHECKED FIRST, AND SEPARATELY. build_conceptual_columns refuses an
		unbuildable design on its own, but its refusal is an exception with a
		list in it. Calling check first means the run stops with every reason
		named at once, including the ones that would NOT have refused — a level
		outside the fitted range builds fine and still belongs in the log
		before an hour of compute rather than after.
		"""
		client = self._mcp(config)
		if client is None:
			raise RuntimeError(
				"the elm MCP is required to build a controlled sweep — what a "
				"factor level becomes is this server's knowledge. Register an "
				"`elm` client in mcp_config.json.")

		verdict = self._mcp_call(client, "check_conceptual_design",
								 {"design": design}) or {}
		for f in (verdict.get("facts") or []):
			print(f"   · {f.get('what')}: {f.get('detail')}")
		for u in (verdict.get("unusual") or []):
			print(f"   ⚠️  {u.get('factor')}: {u.get('why')}")
		if not verdict.get("buildable", False):
			why = "\n".join(f"     - {w.get('factor')}: {w.get('why')}"
							for w in (verdict.get("wont_build") or []))
			raise RuntimeError(
				f"the elm server refused this sweep design:\n{why}")

		out = self._mcp_call(client, "build_conceptual_columns",
							 {"design": design}) or {}
		if not out.get("columns"):
			raise RuntimeError(
				"build_conceptual_columns returned no columns for a design the "
				"server had just called buildable — the check and the builder "
				"disagree, which they share code precisely to prevent.")
		return out

	def _build_coupled_columns(self, config: Dict[str, Any]) -> Dict[str, Any]:
		"""The prior PFLOTRAN run's columns, each carrying its next start.

		THE RETURN LEG OF TWO-WAY COUPLING. The forward leg drove PFLOTRAN
		with ELM's drainage; this one hands PFLOTRAN's SOLVED water table
		back as each column's initial state. The columns are the prior run's
		verbatim (which are the original ELM run's — same ids, coordinates,
		bands), so an iteration is the same ensemble walked to consistency:

		    initial_water_table_m   PFLOTRAN's pressure-crossing depth at its
		                            final output time, read by the ONE public
		                            function that owns that definition
		                            (compare/water_table.solved_water_table_m,
		                            loaded by path from beside the PFLOTRAN
		                            server — the meaning of that model's
		                            output stays on that model's side)
		    water_table_prior_m     what ELM said last time (the anchor the
		                            PFLOTRAN column was given)
		    water_table_delta_m     solved minus prior — the number an
		                            iteration watches; near zero means the
		                            two models agree and the loop is done

		The warm start then stamps each column's fresh finidat with the new
		water table (inputs.py -> set_water_table.py), reporting clamps —
		ELM's aquifer ends at 28.802 m, and a deeper PFLOTRAN water table is
		honestly clamped, not silently believed.
		"""
		brief = (config or {}).get("brief") or {}
		coupling = {**((config.get("strategy") or {}).get("coupling") or {}),
					**(brief.get("coupling") or {})}
		src = coupling.get("prior_run_dir") or config.get("coupling_source")
		if not src or not Path(src).is_dir():
			raise RuntimeError(
				"a coupled study needs the prior run's directory — reception "
				f"resolves it into brief.coupling.prior_run_dir; got {src!r}")
		src = Path(src).resolve()
		cols_f = src / "columns.json"
		ext_f = src / "03_results" / "extracted.json"
		for f in (cols_f, ext_f):
			if not f.is_file():
				raise RuntimeError(
					f"the prior run {src.name} has no {f.name} — it must have "
					f"finished its extract before it can drive another model")

		# the one function that owns "where PFLOTRAN's water table IS"
		import importlib.util
		pkg = Path(__file__).resolve().parents[2] / "pflotran-mcp" / "compare"
		spec = importlib.util.spec_from_file_location(
			"compare_pflotran", pkg / "__init__.py",
			submodule_search_locations=[str(pkg)])
		mod = importlib.util.module_from_spec(spec)
		sys.modules.setdefault("compare_pflotran", mod)
		spec.loader.exec_module(mod)
		from compare_pflotran import water_table as _pwt

		prior = json.loads(cols_f.read_text())
		prior = prior.get("columns", prior) if isinstance(prior, dict) else prior
		data = (json.loads(ext_f.read_text()) or {}).get("columns") or {}

		KEEP = ("id", "lat", "lon", "elevation_m", "band", "band_range_m",
				"pinned", "station_id", "station_variable")
		columns: List[Dict[str, Any]] = []
		for c in prior:
			cid = c.get("id")
			solved = _pwt.solved_water_table_m(data.get(cid) or {})
			if solved is None:
				raise RuntimeError(
					f"{cid}: the prior run's final profile has no water table "
					f"in the domain (bottom cell unsaturated) — there is "
					f"nothing to hand back for this column")
			prior_wt = c.get("water_table_m")
			col: Dict[str, Any] = {k: c.get(k) for k in KEEP
								   if c.get(k) is not None}
			col["initial_water_table_m"] = float(solved)
			col["coupled_from"] = src.name
			col["coupling_variable"] = "water_table"
			if isinstance(prior_wt, (int, float)):
				col["water_table_prior_m"] = round(float(prior_wt), 3)
				col["water_table_delta_m"] = round(float(solved) - float(prior_wt), 3)
			columns.append(col)

		deltas = [c.get("water_table_delta_m") for c in columns
				  if c.get("water_table_delta_m") is not None]
		print(f"   {len(columns)} column(s) from {src.name}, each taking "
			  f"PFLOTRAN's solved water table as its next start"
			  + (f"; moved vs ELM's last answer: "
				 + ", ".join(f"{d:+.3f} m" for d in deltas) if deltas else ""))
		return {
			"approach": "coupled",
			"coupled_from": str(src),
			"coupling_variable": "water_table",
			"n_columns": len(columns),
			"bands": [],
			"columns": columns,
		}

	def _refine_columns(self, columns, config: Dict[str, Any]) -> Dict[str, Any]:
		"""ONE MCP call: warm start, donor soil, surfaces, case_inputs.json.

		Runs inside the base's materialize, BEFORE columns.json is written and
		before the design figure is drawn — the warm start snaps each column to
		its donor gridcell, so anything persisted earlier describes a plan the
		run will not follow.

		WHY THE WHOLE INPUT BUILD HAPPENS HERE, in a hook named for refinement.
		The MCP does all six input steps in one call and the manager used to
		split them across two stages, with the columns.json write wedged
		between. That split is why routing this through the MCP was impossible
		for a week: calling the tool from the later stage re-ran the warm start,
		and calling it from here could not work while the tool's only input was
		a file that does not exist yet at this moment. The tool now takes the
		columns as DATA, so the call lands here — the one moment that satisfies
		the ordering constraint — and the two stages after it have nothing left
		to compute.

		THE COLUMNS ARE REPLACED IN PLACE. The base persists the same list
		object it handed us, so returning a new one would silently persist the
		pre-snap coordinates.
		"""
		client = self._mcp(config)
		if client is None:
			raise RuntimeError(
				"the elm MCP is required to build ELM inputs — the local "
				"by-import path was deleted 2026-08-10 (docs/EXP_MANAGER_ELM.md "
				"§4). Register an `elm` client in mcp_config.json.")
		yr_start = int((config or {}).get("yr_start", 1995))
		ws = (config or {}).get("warm_start", True)
		out = self._mcp_call(client, "build_elm_inputs_from_location", {
			"run_dir":       str(self.run_dir),
			"columns":       columns,
			"yr_start":      yr_start,
			"yr_end":        int((config or {}).get("yr_end", yr_start)),
			"soil_config":   str((config or {}).get("soil_config", "native")),
			"substrate":     str((config or {}).get("substrate", "extrapolate")),
			# FORWARDED EXPLICITLY. Only conus_restart used to cross, so
			# warm_start=False could not be expressed at all and a cold sweep
			# was built warm while its columns claimed otherwise.
			"warm_start":    ws is not False,
			"conus_restart": str((ws or {}).get("conus_restart", "")
								 if isinstance(ws, dict) else ""),
		}, budget=900)
		if not out.get("ok"):
			raise RuntimeError(
				f"build_elm_inputs_from_location failed: {out.get('error')}")
		columns[:] = out["columns"]          # snapped — see the docstring
		self._mcp_inputs = out
		print(f"✓ {out['n_cases']} case input(s) built via the elm MCP")
		return {"mcp_inputs": out}

	def _draw_design(self, res: Dict[str, Any], config: Dict[str, Any]) -> None:
		"""sampling_design.png — what ELM will actually integrate.

		Six panels: the columns over a hillshade, NLDAS forcing against donor
		elevation, the initial soil water read out of each column's finidat, and
		the sand / clay / organic profiles the donor gridcell handed over. Every
		value is POST-warm-start, which is why this can only run after
		_refine_columns — the snap moves every column and swaps its soil.

		IT LIVES HERE BECAUSE IT IS ELM KNOWLEDGE. Reading a finidat and knowing
		what a donor gridcell is are this server's business; docs/ELM_MCP_PLAN.md
		said so on 2026-08-07 and listed the figure among the input tool's
		returns, but `plots.sampling_design` was never written and the framework
		kept drawing its own from tools/expand_sampling.plot_columns — unstyled,
		and describing the columns as SAMPLED rather than as run. Moved
		2026-08-13; that plotter is deleted.
		"""
		sampling_design = self._sibling_module("sampling_design")
		png = sampling_design.render_run(
			self.run_dir,
			reception = self.run_dir / "reception.json",
			year      = int((config or {}).get("yr_start", 0) or 0),
			out       = self.run_dir / "sampling_design.png")
		print(f"✓ sampling design → {Path(png).name}")

	def _to_run_plan(self, plan, columns, config, refine) -> Dict[str, Any]:
		"""The plan the MCP already built, handed back.

		Rebuilding it here would be the same computation twice with two chances
		to disagree; the base still persists it as run_plan.json and resumes
		from it, so it is returned by the call rather than recomputed.
		"""
		out = (refine or {}).get("mcp_inputs") or getattr(self, "_mcp_inputs", None)
		if not out or "run_plan" not in out:
			raise RuntimeError(
				"no run plan from the elm MCP — _refine_columns must run first")
		return out["run_plan"]

	# ─────────────────────────────────────────────────────────
	# MAIN ENTRY POINT
	# ─────────────────────────────────────────────────────────
	# execute_plan now lives in ExperimentManagerBase. ELM keeps the stages
	# and the declarations; the sequence and its ordering guarantees are the
	# base's, so a second backend cannot re-implement them differently.
	NEEDS_CASE_BUILD   = True          # CIME case build, ~8 min for the first
	NEEDS_SCHEDULER = True          # sbatch + wait

	# WHAT build_elm_inputs_from_location ADDS TO A COLUMN, and what the base's
	# package carries from columns.json onto the row (composed with the
	# framework's own COLUMN_METADATA; a key neither names raises). Moved here
	# from the base list on 2026-08-18: which soil a column got and from where
	# is ELM knowledge, so its declaration lives beside the server.
	#
	#   forcing_cell, soil_*   WHICH FORCING CELL, and WHAT SOIL — two
	#                          independent facts about where the column ended
	#                          up, both settled at input time and neither
	#                          derivable from the row without them. Columns
	#                          sharing forcing_cell got the same rain, so a
	#                          difference between them is soil or terrain;
	#                          that reading is the reader's to make, and
	#                          nothing here precomputes it. The
	#                          soil-attribution figure and any claim that soil
	#                          explains a gradient rest on soil_*.
	#   warm_start             cold or warm. Decides which initialisation
	#                          caveat applies and how much of the early record
	#                          is the start rather than the soil —
	#                          extract.resolve_spinup reads it and trims 14
	#                          days for one and a year for the other.
	COLUMN_METADATA_EXTRA = KeySet(
		"ELM_COLUMN_METADATA",
		keep = ("forcing_cell", "soil_summary", "soil_top_texture",
				"soil_layers", "soil_source", "soil_profile",
				"warm_start",
				# A COUPLED FOLLOW-UP'S OWN FACTS (_build_coupled_columns +
				# the warm start's override): which run drove this one, the
				# water table it asked for, the one the finidat actually got
				# (clamps differ), the sentence saying so, and how far the
				# driving model moved it from ELM's own last answer.
				"coupled_from", "coupling_variable",
				"initial_water_table_m", "initial_water_table_written_m",
				"initial_water_table_note",
				"water_table_prior_m", "water_table_delta_m"),
		# A site run starts warm by default and its columns do not say so
		# per column; a sweep writes it. Neither absence is a hole — nor is
		# any coupling key on a run nothing drove.
		optional = ("warm_start",
					"coupled_from", "coupling_variable",
					"initial_water_table_m", "initial_water_table_written_m",
					"initial_water_table_note",
					"water_table_prior_m", "water_table_delta_m"),
		drop = {
			"outside_design_band": "set by warm_start when the snapped donor "
								   "leaves the band the sampler drew. A "
								   "sampling-design fact, read by the design "
								   "figure, not a property of the results",
		},
		source = "columns.json -> columns[*], as build_elm_inputs_from_location wrote them",
		where  = "mcp/elm-mcp/src/elm_exp_manager.py :: COLUMN_METADATA_EXTRA",
	)

	# What ELM's derived metrics MEAN. Lives here, not on the base, because
	# every entry names an ELM history variable: on the base it was silently
	# inherited by PFLOTRAN, whose experiment.json then advertised a surface
	# water budget it never computed. See ExperimentManagerBase.FIELD_SEMANTICS.
	FIELD_SEMANTICS = {
		"precip_mm_yr":            {"units": "mm/yr", "from": ["RAIN", "SNOW"],
									"note": "TOTAL precipitation: rain + snow"},
		"rainfall_mm_yr":          {"units": "mm/yr", "from": ["RAIN"]},
		"snowfall_mm_yr":          {"units": "mm/yr", "from": ["SNOW"]},
		# FIVE ENTRIES WENT ON 2026-08-13, and the note each of them needed is
		# the reason. `runoff_fraction` and `recharge_fraction` were the
		# recharge-vs-runoff SPLIT, denominator (QCHARGE + QOVER), and needed a
		# paragraph here saying they were not fractions of precipitation —
		# because the name says they are. `recharge_to_runoff_ratio` reached
		# 4,288,570 on a column with no runoff. `precip_total_mm_yr` was
		# `precip_mm_yr` under a second name. `annual_runoff_mm_yr` and
		# `annual_recharge_mm_yr` duplicated the water-budget terms at a
		# different rounding.
		#
		# Everything they answered is in `water_budget`, where every term
		# carries `_frac_of_P` and the denominator is IN THE NAME. A catalogue
		# entry that has to warn the reader what its key does not mean is a key
		# that should be renamed, not documented.
		"water_budget":            {"units": "mm/yr and fractions of P",
									"from": ["RAIN", "SNOW", "QOVER", "QDRAI",
											 "QCHARGE", "QINFL", "QSOIL",
											 "QVEGE", "QVEGT", "TWS"],
									"note": "each term as <name>_mm_yr AND "
											"<name>_frac_of_P. recharge is "
											"INTERNAL to TWS, not an export — "
											"see recharge_is_internal. "
											"unaccounted_mm_yr is what the "
											"balance cannot place, and it is "
											"0-14% of P on real runs"},
		"water_table_depth_m":     {"units": "m", "from": ["ZWT"],
									"note": "positive downward from the surface"},
		"peak_swe_modelled_mm":    {"units": "mm", "from": ["H2OSNO"],
									"note": "MODELLED peak. `peak_swe_mm` "
											"elsewhere is the OBSERVED peak at "
											"a snow pillow — different thing"},
		"tws_seasonal_range_mm":   {"units": "mm", "from": ["TWS"]},
		"n_days_in_record":        {"units": "days", "from": [],
									"note": "every mm/yr above is a daily rate "
											"x 365.25, so a short record is an "
											"EXTRAPOLATION by 365.25/this"},
	}

	# ─────────────────────────────────────────────────────────
	# THE MCP — the only path there is
	# ─────────────────────────────────────────────────────────
	# Not a default with a fallback behind it: _refine_columns raises without a
	# client, so a study that reaches any stage below has one. run_via_mcp=False
	# now fails at the first stage instead of quietly running a second
	# implementation of ELM that nobody has exercised since the move.
	MCP_NAME = "elm"

	def _announce(self, experiments, config) -> str:
		"""Say what is about to happen. PRINTS ONLY — it never asks.

		Asking belongs to the coordinator: it owns every other question put to
		the user (interactive_reception, the --resume prompt) and it owns the
		TTY guard that keeps those from hanging a scripted run. A backend that
		called input() here would be a compute stage holding the terminal open,
		and it only ended up here because the run directory and the column
		count happened to be in scope.

		The address is whatever the coordinator, the environment, or the
		remembered preference already settled on.
		"""
		n = len(experiments or [])
		email = self._notify_email(config)
		print()
		print(f"⚙️  This ELM study runs unattended.")
		print(f"    {n} column(s) · build + run + analysis, ~25-40 min in the queue.")
		print(f"    Nothing to watch. Results will appear in:")
		print(f"      {self.analysis_dir}")
		print(f"    ✉  {email}" if email else
			  "    (no notification address — nothing will tell you when it lands)")
		print()
		return email

	# ─────────────────────────────────────────────────────────
	# STEP 2 — JOBS A AND B  (the whole of the compute)
	# ─────────────────────────────────────────────────────────
	def _build_cases(self, experiments: List[Dict], config: Dict[str, Any] = None):
		"""JOB A through the MCP, then JOB B by dependency. Returns a Pending.

		Named build_cases because that is the base's slot for "a stage that may
		hand back a job id"; job A does the build AND the run, so _run has
		nothing left to submit by the time it is reached. ONE JOB, ALWAYS — the
		split flow cost three invocations for a 25-40 minute study (submit the
		build, come back and submit the run, come back and analyze), which made
		the person the scheduler.

		A builds the cases and runs every column, and stops. B is submitted HERE,
		with --dependency=afterany on A's id, and does the analysis. Then this
		returns and the framework exits — SLURM starts B by itself.

		WHY B IS SUBMITTED BY THE FRAMEWORK AND NOT BY A. The tool this replaces,
		run_elm_study, ended its job by running `workflow.py --finalize`: the
		server's job executing the client's code, which is the dependency the
		boundary rule forbids and the reason the two could not be separated.
		Submitting B from this side inverts it — the framework asks for the model
		run and arranges its own follow-up, and nothing in the server calls back.

		afterany, NEVER afterok. With afterok a failed ensemble means B never runs
		and no mail is ever sent — the silent failure of 2026-08-06, twice. afterany
		means B always runs and always reports, including "the ensemble failed".
		"""
		config = config or {}
		client = self._mcp(config)
		if client is None:
			raise RuntimeError(
				"the elm MCP is required to build and run ELM cases — the "
				"local by-import path was deleted 2026-08-10 "
				"(docs/EXP_MANAGER_ELM.md §4). Register an `elm` client in "
				"mcp_config.json.")
		# The server reads 01_inputs/case_inputs.json and nothing else. It does
		# not generate inputs and never opens a surfdata file: FSURDAT and
		# FINIDAT — the warm start's two products — crossed the boundary as
		# data when _refine_columns wrote this file.
		src = self.input_dir / self.CASE_INPUTS
		if not src.is_file():
			raise RuntimeError(
				f"{src} is missing — _refine_columns should have written it "
				f"via the elm MCP")
		email = self._announce(experiments, config)
		out = self._mcp_call(client, "run_elm_ensemble", {
			"run_dir":  str(self.run_dir),
			"queue":    str(config.get("queue", "")),
			"walltime": str(config.get("study_walltime", "02:00:00")),
		}, budget=600)
		if out.get("error"):
			raise RuntimeError(f"run_elm_ensemble: {out['error']}")
		job_a = str(out["job_id"])
		job_b = self._submit_job_b(job_a, config, email)
		if email:
			print(f"   ✉  {email} will be mailed when job B lands")
		# BOTH IDS. `job_id` is what the framework POLLS — job B when there is
		# one, because B ending means the study is over. But B is the reporter;
		# A is the job that built and ran the columns, and its id was dropped
		# here and recorded nowhere else, so the run state named B's id beside
		# A's log and nothing could say how long the model actually took.
		# Measured 2026-08-13: the only runtime any artifact carried was the
		# framework's own 593.8 s tail against an 18 m 23 s ensemble.
		return Pending(job_b or job_a, n_cases=len(experiments or []),
					   job_id_a=job_a, job_id_b=job_b or None,
					   log=out.get("log_path"), via="mcp", scope="study",
					   notify=email or None)

	def _submit_job_b(self, job_a: str, config: Dict[str, Any],
					  email: str = "") -> Optional[str]:
		"""JOB B — the analysis, held until A ends. Returns its id, or None.

		Non-fatal: A is already queued and its model output does not depend on B.
		Losing B costs the analysis, which can be run by hand; raising here would
		strand a running ensemble with nothing recording that it exists.
		"""
		import shutil, subprocess
		if not shutil.which("sbatch"):
			print("   ⚠️  no sbatch — job B not submitted; analyse by hand")
			return None
		# REFUSE A NODE-LOCAL RUN DIRECTORY. A compute node cannot see this node's
		# /tmp: the first A/B attempt (770938/770939) put its scripts there and both
		# jobs died in two seconds with completely empty logs, which is
		# indistinguishable from a scheduler fault.
		#
		# It also makes this method safe to reach from a test. pytest's tmp_path is
		# under /tmp, and `sbatch` exists on a login node, so a test that got past
		# the mocked client would otherwise submit a real job into the queue.
		#
		# IT FAILS OPEN, AND SAYS SO. `df` not answering is not evidence that the
		# directory is node-local, so refusing on it would break every machine
		# where df is absent or slow. But an unmade check that prints nothing is
		# indistinguishable from a check that passed — which is exactly how this
		# went unnoticed once already, when a harness broke `df` and the bare
		# `except` turned the failure into a silent green light.
		try:
			fs = subprocess.check_output(
				["df", "-P", str(self.run_dir)], text=True,
				timeout=20).splitlines()[-1].split()[0]
		except Exception as e:                                  # noqa: BLE001
			print(f"   ⚠️  could not check whether {self.run_dir} is on shared "
				  f"storage ({type(e).__name__}: {e}) — submitting job B anyway. "
				  f"If it dies in seconds with an empty log, this is why.")
			fs = ""
		if fs.startswith("/dev/"):
			print(f"   ⚠️  {self.run_dir} is on {fs}, a node-local filesystem — "
				  f"job B not submitted. Put the run directory on $PSCRATCH.")
			return None
		fw = Path(__file__).resolve().parents[3]
		sb = self.run_dir / "ensemble_B.sbatch"
		mail = (f"#SBATCH --mail-user={email}\n#SBATCH --mail-type=END,FAIL"
				if email else "")
		sb.write_text(f"""#!/bin/bash
#SBATCH -J elm_B
#SBATCH -N 1
#SBATCH -p {config.get("queue") or os.environ.get("IDEAS_SLURM_QUEUE", "short")}
#SBATCH -A {config.get("account") or os.environ.get("IDEAS_SLURM_ACCOUNT", "e3sm")}
#SBATCH -t {config.get("analysis_walltime", "00:30:00")}
#SBATCH -o {self.run_dir}/ensemble_B.log
{mail}
cd {fw}
# JOB B RUNS THE ANALYSIS. It was deferred while the Analyzer was rebuilt, and
# the default stayed 0 long after — so every run since has stopped at the
# ensemble and been finished by hand, while SLURM recorded job B as FAILED in
# 2 s (notify_study exits non-zero on an incomplete study). Default flipped
# 2026-08-13, once the Analyzer had been verified end to end on both archived
# studies. IDEAS_RUN_ANALYSIS=0 still defers it.
if [ "${{IDEAS_RUN_ANALYSIS:-1}}" = "1" ]; then
  # --finalize, NOT --resume. This runs INSIDE job B, so --resume polls the
  # scheduler, finds this very job active, and stops one line short of the
  # analysis it was submitted to produce — "Job <B> is still running, N
  # column(s) queued", exit 1, 40 seconds. finalize_run exists precisely for
  # this and its docstring describes the failure; the template just never used
  # it. Observed on 773411 and 773413 (2026-08-13), both after job A had
  # COMPLETED and the output was on disk.
  {sys.executable} workflow.py --finalize {self.run_dir}
  {sys.executable} tools/notify_study.py {self.run_dir}
else
  {sys.executable} tools/notify_study.py {self.run_dir} --deferred
fi
exit $?
""")
		try:
			jid = subprocess.check_output(
				["sbatch", "--parsable", f"--dependency=afterany:{job_a}", str(sb)],
				text=True, timeout=120).strip().split(";")[0]
		except Exception as e:                                  # noqa: BLE001
			print(f"   ⚠️  job B not submitted ({e}) — A is running; analyse by hand")
			return None
		print(f"   job A {job_a} (build+run) → job B {jid} (afterany, reports)")
		return jid

	def _poll(self, record: Dict[str, Any], experiments, config):
		"""Has job A landed? The case directories if so, None if not.

		ONE SHAPE, because there is one job. build_cases is the only stage that
		returns a Pending now, so the dispatch on record['stage'] is gone with
		the split flow — and with it the way that dispatch went wrong: polling a
		build+run job down the `build_cases` branch handed back case dirs and let
		execute_plan walk into _run, which submitted the ensemble A HAD ALREADY
		RUN. Every column would have run twice, and the second pass would have
		overwritten the history files of the first.

		The case directories come back from check_elm_job itself. A stage that
		hands back a job id needs somewhere to hand back its ANSWER, and a few
		short strings can travel inline — unlike results, which is why reading
		those stayed on this side.
		"""
		client = self._mcp(config or {})
		if client is None:
			raise RuntimeError(
				f"job {record.get('job_id')} is recorded as pending but there "
				f"is no `elm` client to ask about it — register one in "
				f"mcp_config.json, or check it by hand with squeue")
		st = self._mcp_call(client, "check_elm_job",
							{"job_id": str(record.get("job_id")),
							 "run_dir": str(self.run_dir)})
		if st.get("active", True):
			print(f"   job {record.get('job_id')} is "
				  f"{st.get('state') or 'unanswered'} — nothing to collect yet")
			return None
		if not st.get("ok"):
			raise RuntimeError(
				f"the case build failed: {st.get('error')}"
				+ (f"\n{st.get('log_tail')}" if st.get("log_tail") else ""))
		by_name = {c.get("case_name"): c.get("case_dir")
				   for c in (st.get("cases") or [])}
		for e in experiments:
			if by_name.get(e.get("case_name")):
				e["case_dir"] = by_name[e["case_name"]]
		print(f"   ✓ {st.get('n_ok')}/{st.get('n_total')} case(s) built")
		return experiments

	def _build_case_inputs(self,
			   plan:   Dict[str, Any],
			   config: Dict[str, Any]) -> List[Dict]:
		"""Read what the MCP already wrote. Computes nothing.

		_refine_columns made the one call that produced case_inputs.json, so
		this stage exists only because the base's pipeline has a slot for it.
		The rows come back as PLAIN DATA with no live ELMAgent, and that is now
		the ONLY state they are ever in — not a resume-only special case. The
		--keepexe builder handle went with the local path; the job side does
		the cloning, on the compute node, out of case_inputs.json.
		"""
		rows = self._rehydrate_case_inputs()
		if not rows:
			raise RuntimeError(
				f"{self.input_dir / self.CASE_INPUTS} is missing or empty — "
				f"_refine_columns should have written it via the elm MCP")
		return rows

	# NO BUILD-TIME SURFACE FIGURE (deleted 2026-08-14, by request).
	# column_surfaces.png was drawn by the MCP's build job because it read each
	# case's generated FSURDAT through its `run/lnd_in`, which exists only after
	# the build. sampling_design.png shows the same soil from the donor data,
	# post-warm-start, and is the figure that gets looked at.

	BUILT_CASES = "built_cases.json"

	def adopt_completed_run(self) -> Dict[str, Any]:
		"""ELM also adopts the case build, and re-attaches where each landed.

		built_cases.json is the one thing the finalize cannot work out for
		itself: case directories carry a creation timestamp, so they cannot be
		derived from the case names. Without re-attaching them, _extract globs
		`{case_dir}/run/*.elm.h0.*.nc` against a case_dir of None and every
		column reports zero history files — a finished ensemble read as a
		total failure.
		"""
		experiments = self._rehydrate_case_inputs() or []
		src = self.input_dir / self.BUILT_CASES
		adopted: Dict[str, Any] = {}
		if src.is_file() and experiments:
			try:
				built = json.loads(src.read_text())
				by_name = {c.get("case_name"): c.get("case_dir")
						   for c in (built.get("cases") or [])}
				n = 0
				for e in experiments:
					if by_name.get(e.get("case_name")):
						e["case_dir"] = by_name[e["case_name"]]
						n += 1
				self._save_case_inputs(experiments)
				self._mark("build_cases", n_experiments=len(experiments),
						   n_ok=n, adopted_from=self.BUILT_CASES)
				adopted["build_cases"] = n
			except Exception as e:                              # noqa: BLE001
				print(f"   ⚠️  could not adopt {self.BUILT_CASES} ({e}) — "
					  f"the case directories will be missing")
		adopted.update(super().adopt_completed_run())
		return adopted

	# ─────────────────────────────────────────────────────────
	# STEP 3 — RUN (writes to 03_results/)
	# ─────────────────────────────────────────────────────────
	def _run(self,
			 experiments: List[Dict],
			 config:      Dict[str, Any]):
		"""COLLECT. Job A already ran the columns; this never submits anything.

		The stage is still here because the base's sequence has a slot for it
		and because everything it does after the columns stop — the run
		summaries, execution_report.txt, results_summary.csv — still has to
		happen. What it does NOT do is run ELM: by the time execute_plan reaches
		this line, either job A has just been polled (the case dirs came back
		from _poll) or the run state says build_cases is done, and both mean the
		ensemble has already been through the queue.

		SUBMITTING HERE WAS A BUG, not a simplification removed. A --resume of a
		landed A+B study walked out of _poll into this stage and re-submitted
		every column, over the top of the history files the first pass wrote.
		Nothing had exercised it only because the analysis is deferred.

		Success is what is ON DISK: a column succeeded if ELM wrote it a history
		file. The scheduler being finished is what ends the wait; the files
		decide the outcome.
		"""
		return self._collect(experiments, self._outcomes_from_disk(experiments))

	def _collect(self,
				 experiments: List[Dict],
				 results:     Dict[str, bool]) -> Dict[str, bool]:
		"""Everything that happens once the columns have stopped running."""
		for exp in experiments:
			try:
				exp['run_summary'] = self._run_summary_for(exp)
			except Exception as e:                              # noqa: BLE001
				print(f"   ⚠️  run summary failed for {exp['case_name']}: {e}")

		n_ok   = sum(1 for v in results.values() if v)
		n_fail = len(results) - n_ok
		print(f"✓ {n_ok} succeeded, {n_fail} failed")

		# Post-run reporting → 03_results/
		self._write_execution_report(experiments, results)
		self._write_results_csv(experiments, results)

		return results

	@staticmethod
	def _run_summary_for(exp: Dict) -> Dict[str, Any]:
		"""The case's run summary. Off disk, always.

		There is no live agent to ask any more: the cases are built inside job
		A, and what comes back to this side is JSON. Asking the string repr of
		an ELMAgent for a summary is how this once reported zero history files
		for cases that had plenty — the branch that did it is gone rather than
		guarded, so there is nothing left to fall back FROM.
		"""
		cd = exp.get('case_dir')
		return {'history_files': sorted(glob.glob(f"{cd}/run/*.elm.h0.*.nc"))
								 if cd else []}

	def _outcomes_from_disk(self, experiments: List[Dict]) -> Dict[str, bool]:
		"""Which columns produced output — the one definition of success: a
		column succeeded if ELM wrote it a history file."""
		return {e['case_name']: bool(e.get('case_dir') and
									 glob.glob(f"{e['case_dir']}/run/*.elm.h0.*.nc"))
				for e in experiments}

	# ONE definition, in inputs.py, because two lists that must agree are two
	# lists that will not. A key added there and forgotten here would drop a
	# field from case_inputs.json silently — the file would exist, parse, and be
	# missing something the build needs.
	@property
	def CASE_KEYS(self):                                    # noqa: N802
		import inputs
		return inputs.CASE_KEYS

	def _serialise_case_inputs(self, experiments):
		"""Delegates to inputs.serialise_case_inputs — see there."""
		import inputs
		return inputs.serialise_case_inputs(experiments)

	def _rehydrate_handles(self, experiments, plan, config) -> None:
		"""Drop the ELMAgent string reprs a reloaded case_inputs.json carries.

		An agent is an object and the manifest is JSON, so what comes back is a
		string: truthy, with no methods, and therefore a thing that fails at the
		call site rather than here. Nothing on this side calls one any more —
		the build happens inside job A — so they are dropped and the key stays
		absent.
		"""
		stale = [e for e in experiments if isinstance(e.get('elm_agent'), str)]
		for e in stale:
			e.pop('elm_agent', None)
		if stale:
			print(f"   ↻ {len(stale)} case handle(s) are not JSON — working "
				  f"from the case directories on disk")

	def _write_execution_report(self,
								experiments: List[Dict],
								results:     Dict[str, bool]) -> None:
		"""Write a plain-text run report to 03_results/execution_report.txt."""
		report_file = self.results_dir / "execution_report.txt"

		n_total   = len(experiments)
		n_success = sum(results.values())

		lines = []
		lines.append("=" * 60)
		lines.append("ELM EXECUTION REPORT")
		lines.append("=" * 60)
		lines.append(f"Generated:         {datetime.now().isoformat()}")
		lines.append(f"Run directory:     {self.run_dir}")
		lines.append(f"Total experiments: {n_total}")
		lines.append(f"Successful:        {n_success}")
		lines.append(f"Failed:            {n_total - n_success}")
		lines.append("")

		for exp in experiments:
			case_name   = exp['case_name']
			status      = ('completed' if results.get(case_name)
						   else 'failed')
			run_summary = exp.get('run_summary', {}) or {}
			n_hist      = len(run_summary.get('history_files', []))

			lines.append(f"── {case_name}")
			lines.append(f"   status:           {status}")
			lines.append(f"   scenario:         {exp.get('scenario_name', '?')}")
			lines.append(f"   forcing_period:   {exp.get('forcing_period', '?')}")
			lines.append(f"   forcing_years:    "
						 f"{exp.get('forcing_start', '?')}-"
						 f"{exp.get('forcing_end', '?')}")
			lines.append(f"   soil_config:      {exp.get('soil_config', '?')}")
			if exp.get('substrate'):
				lines.append(f"   substrate:        {exp['substrate']}")
			lines.append(f"   case_dir:         {exp.get('case_dir', '?')}")
			lines.append(f"   history_files:    {n_hist}")
			lines.append("")

		report_file.write_text('\n'.join(lines))
		print(f"✓ execution_report.txt saved")

	def _write_results_csv(self,
						   experiments: List[Dict],
						   results:     Dict[str, bool]) -> None:
		"""Write a tabular summary to 03_results/results_summary.csv."""
		csv_file = self.results_dir / "results_summary.csv"
		fields = ['case_name', 'scenario_name', 'forcing_period',
				  'forcing_start', 'forcing_end', 'soil_config',
				  'substrate', 'status', 'history_file_count',
				  'case_dir']

		with open(csv_file, 'w', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=fields)
			writer.writeheader()
			for exp in experiments:
				case_name   = exp['case_name']
				run_summary = exp.get('run_summary', {}) or {}
				writer.writerow({
					'case_name':          case_name,
					'scenario_name':      exp.get('scenario_name', ''),
					'forcing_period':     exp.get('forcing_period', ''),
					'forcing_start':      exp.get('forcing_start', ''),
					'forcing_end':        exp.get('forcing_end', ''),
					'soil_config':        exp.get('soil_config', ''),
					'substrate':          exp.get('substrate', '') or '',
					'status':             ('completed'
										   if results.get(case_name)
										   else 'failed'),
					'history_file_count': len(run_summary.get(
						'history_files', [])),
					'case_dir':           exp.get('case_dir', ''),
				})
		print(f"✓ results_summary.csv saved")

	# ─────────────────────────────────────────────────────────
	# STEP 4 — ANALYZE (writes to 04_analysis/)
	# ─────────────────────────────────────────────────────────
	def _extract(self,
				 experiments,
				 plan:   Dict[str, Any] = None,
				 config: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Read the ELM history NetCDFs, then build the rows FROM what was read.

		TWO STEPS, IN THIS ORDER, AND NOTHING BETWEEN THEM:

		    extract_run   history files -> 03_results/extracted.json
		    build_rows    extracted.json -> the per-column rows

		This used to be one object, ELMResultsAnalyzer, which read and computed
		in the same pass and called write_extracted itself. That made TWO
		writers of one artifact — it and extract_run — reached by two paths.
		They did not disagree, but the shape is the one that already cost a
		day: an extracted.json from one morning beside an experiment.json from
		the next, 351 days against 352, nothing saying so. One writer now, and
		the rows are reconstructible from the file it wrote.

		OVERWRITE, deliberately. This stage has just run the model, so it is
		the authoritative read; reuse belongs to the tool and to the comparison,
		which may legitimately open an artifact somebody else made. A resumed
		run does not pay for it twice — execute_plan skips a completed stage
		outright, so this body does not run at all.

		This is extraction, not analysis: it knows ELM's output format and
		nothing about what the numbers mean. Figures, observation comparison
		and interpretation belong to the Analyzer (src/agents/analyzer.py).

		Also attaches the honesty payload (structural + configuration
		limitations, assumptions ledger), which _ensemble_blocks folds into
		experiment.json and step 0 turns into the caveats that bind the
		interpreter.
		"""
		cfg  = config or {}
		plan = plan or {}
		honesty = {}
		try:
			from core.limitations import select_limitations
			couplers = plan.get("CONDITIONS_COUPLERS") or [{}]
			y0 = int(couplers[0].get("DATM_CLMNCEP_YR_START", cfg.get("yr_start", 1995)))
			y1 = int(couplers[0].get("DATM_CLMNCEP_YR_END",   cfg.get("yr_end", y0)))
			honesty = {
				"limitations": select_limitations(
					n_years       = max(1, y1 - y0 + 1),
					warm_start    = bool(couplers[0].get("FINIDAT")),
					forcing       = cfg.get("forcing", "nldas"),
					spinup_years  = int(cfg.get("spinup_years", 0)),
					# WHICH warm start, and whether the soil was kept with it.
					# Without these the caveat cannot tell an equilibrated,
					# self-consistent state from the mismatch that made year
					# one a relaxation — and it defaulted to warning about both.
					# ALWAYS "conus" — there is one warm start (2026-08-12).
					# This used to dig a `source` out of the config, which by
					# then could be True, a dict, or a string; with one source
					# the question is answered here instead of re-derived.
					warm_source   = "conus",
					# WHAT THE RUN ACTUALLY DID, not what a site run always
					# does. Both of these were hardcoded on the premise that a
					# warm start is required and the donor's surfdata is always
					# what ELM runs on. A conceptual sweep is cold and
					# prescribes its own soil, so both were simply false there,
					# and the caveat list described a different run.
					soil_source   = str(couplers[0].get("SOIL_SOURCE") or "conus"),
					archetype     = str((cfg.get("strategy") or {}).get("archetype")
										or (cfg.get("brief") or {}).get("design_archetype")
										or ""),
					written_weather = bool(couplers[0].get("PRESCRIBED_WEATHER")),
				),
				"assumptions_ledger": (
					json.loads((self.run_dir / "assumptions.json").read_text())
					if (self.run_dir / "assumptions.json").exists() else []),
			}
		except (NameError, AttributeError, TypeError) as e:
			# A BUG HERE IS NOT A MISSING PAYLOAD, and the handler used to
			# report it as one. A bare `except Exception` swallowed a NameError
			# — a stale variable left by a refactor — and printed "limitations
			# payload unavailable", which reads as "this run has no
			# limitations". It does not: it means this code did not run. The
			# consequence is silent and large, because extra_summary is what
			# _ensemble_blocks folds into experiment.json and step 0 turns into
			# the caveats that bind the interpreter. Found 2026-08-13 by
			# re-running the stage over an archived study.
			print(f"   ✗ BUG in the limitations payload — {type(e).__name__}: "
				  f"{e}. This is a code fault, not a run without caveats; "
				  f"experiment.json will carry no limitations.")
			raise
		except Exception as e:
			# Genuinely absent input: no assumptions.json, an unreadable one, a
			# plan with no couplers. The run stands; the caveats do not.
			print(f"   ⚠️  limitations payload unavailable ({type(e).__name__}: "
				  f"{e}) — experiment.json will carry no limitations")

		from extract import VARIABLE_UNITS, extract_run
		from column_rows import build_rows, spinup_dropped

		res = extract_run(str(self.run_dir), overwrite=True)
		if not res.get("ok"):
			# NOT fatal here. _package still runs and records that the run
			# produced nothing readable, which is the only account of what
			# happened; raising would throw away a finished ensemble.
			print(f"   ✗ extraction failed: {res.get('error')}")
			return {"rows": [], "units": dict(VARIABLE_UNITS),
					"extra_summary": honesty}
		print(f"   ✓ {res['n_ok']}/{res['n_columns']} column(s) read "
			  f"-> 03_results/extracted.json ({res.get('size_mb')} MB)")

		rows = build_rows(self.run_dir)
		print(f"   ✓ {len(rows)} row(s) built from the artifact")

		# READ OFF THE ARTIFACT, not remembered from the extraction pass — the
		# same rule as the rows. It travels in the stage contract so the base
		# can fold it into experiment.json, where step 4 has always looked for
		# it and never found it.
		drop = spinup_dropped(self.run_dir)
		if drop:
			span = (f", from {drop['from']} to {drop['to']}"
					if drop.get("applied") else "")
			print(f"   ✓ {drop.get('basis', 'start-up')} trim recorded: "
				  f"{drop['days']} d, {drop['timesteps_dropped']} timestep(s)"
				  f"{span}")
			if drop.get("note"):
				print(f"   ! {drop['note']}")

		# THE STAGE'S OUTPUT IS DATA — a plain dict in the contract's shape, not
		# a live object the base has to read with getattr. That was the last
		# holdout, and it is what makes this stage crossable by a tool call.
		return {"rows": list(rows.values()),
				"units": dict(VARIABLE_UNITS),
				"extra_summary": honesty,
				"spinup_dropped": drop}

	# ─────────────────────────────────────────────────────────
	# STEP 4d — ONE-WAY ELM → PFLOTRAN COUPLING
	# ─────────────────────────────────────────────────────────
