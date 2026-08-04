#!/usr/bin/env python3
"""
ELM Experiment Manager
src/core/elm_exp_manager.py

Single responsibility: orchestrate the ELM execution pipeline — the
Experiment Manager box of the framework (see ARCHITECTURE.md).

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
        ├── 02_setup_plots/
        │   └── column_surfaces.png                 (real FSURDAT per column)
        ├── 03_results/
        │   ├── execution_report.txt
        │   └── results_summary.csv
        ├── 04_analysis/
        │   ├── hydro_summary.json  validation.json  interpretation.md
        │   ├── partitioning.png  controls.png  soil_control.png
        │   ├── wtd_columns.png
        │   └── validation_{hydrograph,yield,water_table,swe,context}.png
        ├── 05_pflotran/                            (step 4d, only if coupled)
        │   ├── <col>/<col>.in + outputs
        │   ├── pflotran_summary_coupled.json  pflotran_study.json
        │   └── pflotran_coupled.png
        ├── ANALYSIS_REPORT.json
        ├── LLM_ANALYSIS_INPUT.json
        └── RUN_SUMMARY.json

ELM cases live at: $PSCRATCH/E3SMv3/1D_ELM.*/
"""
import csv
import glob
import os
import json
import re
import sys
from pathlib  import Path
from datetime import datetime
from typing   import Dict, Any, List, Optional

sys.path.insert(0, "src")

from agents.analyzer          import Analyzer
from core.exp_manager_base       import ExperimentManagerBase, Pending
from core.elm_experiment_builder import ELMExperimentBuilder
from core.elm_results_analyzer   import ELMResultsAnalyzer
from core.columns_to_plan        import columns_to_elm_plan


# Repo root — this file is <root>/src/core/elm_exp_manager.py
_ROOT = Path(__file__).resolve().parents[2]



def _load_tool(name: str):
	"""Import tools/<name>.py by path — tools/ is a script dir, not a package.

	The manager and the standalone CLIs then share ONE implementation of each
	stage instead of drifting apart: expand_sampling (materialize),
	make_warmstart (0b), plot_columns (setup figures), build_pflotran_cases +
	analyze_pflotran_coupled (4d).
	"""
	import importlib.util
	if name in sys.modules:
		return sys.modules[name]
	path = _ROOT / "tools" / f"{name}.py"
	spec = importlib.util.spec_from_file_location(name, path)
	mod  = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	spec.loader.exec_module(mod)
	return mod


# ─────────────────────────────────────────────────────────────────────
# ELM EXPERIMENT MANAGER
# ─────────────────────────────────────────────────────────────────────
class ELMExpManager(ExperimentManagerBase):
	"""
	Executes ELM experiment plans.

	Stages: materialize -> warm start -> build_case_inputs -> build_cases -> run -> extract
	-> couple -> package. Steps 0, 0b, 4 and 4d delegate to the tools/ CLIs
	via _load_tool(), so both entry points share one implementation.

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

	def _refine_columns(self, columns, config: Dict[str, Any]) -> Dict[str, Any]:
		"""Warm start, then adopt the donor cell's soil.

		Runs inside the base's materialize, BEFORE columns.json is written and
		before the design figure is drawn — the warm start snaps each column to
		its donor gridcell, so anything persisted earlier describes a plan the
		run will not follow.
		"""
		finidat_map = self._warmstart(columns, config)
		self._attach_donor_soil(columns, finidat_map)
		return {"finidat_map": finidat_map}

	def _to_run_plan(self, plan, columns, config, refine) -> Dict[str, Any]:
		"""Columns → CONDITIONS_COUPLERS.

		FINIDAT is a per-coupler key the builder reads at build time, which is
		why the warm start had to happen in _refine_columns rather than here.
		"""
		yr_start = int(config.get("yr_start", 1995))
		yr_end   = int(config.get("yr_end",   yr_start))
		return columns_to_elm_plan(
			columns,
			yr_start      = yr_start,
			yr_end        = yr_end,
			soil_config   = config.get("soil_config", "native"),
			substrate     = config.get("substrate",   "extrapolate"),
			finidat_map   = refine.get("finidat_map") or {},
			period_source = (config.get("period_source")
							 or ("reception" if config.get("yr_start") else "DEFAULT")),
		)

	# ─────────────────────────────────────────────────────────
	# MAIN ENTRY POINT
	# ─────────────────────────────────────────────────────────
	# execute_plan now lives in ExperimentManagerBase. ELM keeps the stages
	# and the declarations; the sequence and its ordering guarantees are the
	# base's, so a second backend cannot re-implement them differently.
	NEEDS_CASE_BUILD   = True          # CIME case build, ~8 min for the first
	NEEDS_SCHEDULER = True          # sbatch + wait
	COUPLES_TO      = "pflotran"    # one-way QINFL handoff, when asked for

	# What ELM's derived metrics MEAN. Lives here, not on the base, because
	# every entry names an ELM history variable: on the base it was silently
	# inherited by PFLOTRAN, whose experiment.json then advertised a surface
	# water budget it never computed. See ExperimentManagerBase.FIELD_SEMANTICS.
	FIELD_SEMANTICS = {
		"precip_mm_yr":            {"units": "mm/yr", "from": ["RAIN", "SNOW"],
									"note": "TOTAL precipitation: rain + snow"},
		"rainfall_mm_yr":          {"units": "mm/yr", "from": ["RAIN"]},
		"snowfall_mm_yr":          {"units": "mm/yr", "from": ["SNOW"]},
		"annual_runoff_mm_yr":     {"units": "mm/yr", "from": ["QOVER"],
									"note": "SURFACE runoff only. QDRAI "
											"(subsurface drainage) is NOT "
											"included, so this is not the "
											"streamflow-comparable total"},
		"annual_recharge_mm_yr":   {"units": "mm/yr", "from": ["QCHARGE"]},
		# NOT fractions of precipitation. Both denominators are
		# (QCHARGE + QOVER): these say how the drainage SPLITS between
		# recharge and runoff, and they sum to 1 by construction even when
		# both terms are ~0. Reading them as runoff/P is wrong by a factor
		# of P/(QCHARGE+QOVER) — on the 2019 Gunnison run that made columns
		# draining ~0 mm/yr report recharge_fraction = 1.00.
		"runoff_fraction":         {"units": "1",
									"from": ["QOVER", "QCHARGE"],
									"note": "QOVER / (QCHARGE + QOVER) — the "
											"recharge-vs-runoff SPLIT, not a "
											"fraction of precipitation"},
		"recharge_fraction":       {"units": "1",
									"from": ["QCHARGE", "QOVER"],
									"note": "QCHARGE / (QCHARGE + QOVER) — the "
											"recharge-vs-runoff SPLIT, not a "
											"fraction of precipitation"},
		"recharge_to_runoff_ratio": {"units": "1", "from": ["QCHARGE", "QOVER"]},
		"precip_total_mm_yr":      {"units": "mm/yr", "from": ["RAIN", "SNOW"],
									"note": "water-budget total; "
											"precip_mm_yr is the same sum"},
		"water_budget":            {"units": "mm/yr", "from": ["RAIN", "SNOW",
															   "QOVER", "QDRAI",
															   "QCHARGE", "TWS"]},
		"water_table_depth_m":     {"units": "m", "from": ["ZWT"],
									"note": "positive downward from the surface"},
		"peak_swe_mm":             {"units": "mm", "from": ["H2OSNO"]},
		"tws_seasonal_range_mm":   {"units": "mm", "from": ["TWS"]},
	}

	def _couple(self, plan, config):
		"""Base calls this when COUPLES_TO is set; delegate to the existing
		implementation so the coupling code has one home."""
		return self._couple_pflotran(plan, config)

	def _attach_donor_soil(self, columns, finidat_map):
		"""Replace each warm-started column's soil with the donor's own.

		A warm start keeps the donor gridcell's surfdata, so the SSURGO profile
		gathered in step 0 is characterisation, not what ELM runs on. The design
		figure and columns.json must show the dataset the experiment actually
		uses, or the plan on the page and the run on the machine disagree.

		This is the ONLY soil the run has. Sampling no longer gathers a second
		profile, so there is no other dataset to confuse it with and nothing in
		experiment.json that ELM did not actually see.
		"""
		fs = _load_tool("make_finidat_subset")
		n = 0
		for c in columns:
			entry = finidat_map.get(c.get("id")) or {}
			sd = entry.get("surface_template")
			if not sd or not Path(sd).exists():
				raise RuntimeError(
					f"{c.get('id')}: warm start reported a donor but its "
					f"surface template is missing ({sd}). Skipping would leave "
					f"this column with no soil while its siblings have the "
					f"donor's.")
			prof = fs.donor_soil_profile(sd)
			if not prof:
				raise RuntimeError(
					f"{c.get('id')}: no soil profile readable from the donor "
					f"surface template {sd}.")
			c["soil_profile"] = prof
			c["soil_layers"] = prof["num_layers"]
			c["soil_top_texture"] = prof["layers"][0]["texture_class"]
			c["soil_source"] = "conus"
			n += 1
		if n:
			print(f"   soil for the design/figure taken from the CONUS donor "
				  f"cells ({n} column(s)) — the dataset the run uses")

	def _warmstart(self, columns, config: Dict[str, Any]):
		"""Build a finidat per column straight from the CONUS 1-km restarts.

		No carrier, no prior run, no template library: the CONUS restart already
		holds every variable, so one gridcell is subset out into a standalone
		single-column file. Works on a domain that has never been run.

		config['warm_start'] = True | {'conus_restart': manifest|file|dir}

		Two things this does that are easy to get wrong:

		  * It SNAPS each column to its donor gridcell (~250-400 m). The domain,
		    surfdata and finidat must agree on coordinates or ELM aborts at init
		    on a surfdata/fatmgrid mismatch, so the donor's lat/lon wins and the
		    shift is recorded in the assumptions ledger rather than left silent.
		  * It also subsets the CONUS gridcell's own surfdata and passes it on as
		    the surface TEMPLATE, so fsurdat and finidat describe the same
		    gridcell (ELM's check_weights gate). SSURGO soil is still overwritten
		    on top of it by the surface generator — the per-column soil science is
		    unchanged; only the vegetation source improves, 0.5 degree -> 1 km.

		FATAL on failure. This used to return None and let the ensemble cold
		start, which sounds forgiving and is not: the start type is decided PER
		COLUMN downstream, so a partial failure produced a mixed ensemble —
		some columns warm on the donor's CONUS soil, others cold on a different
		soil dataset — inside one run, signalled by nothing but a "(cold)" in a
		print. Every cross-column comparison then spans two experiments. A run
		that cannot warm start is a different experiment from the one that was
		planned, so it stops here instead of quietly becoming one.
		"""
		ws = config.get("warm_start", True)
		if isinstance(ws, str):
			ws = {"source": ws}
		elif ws is True:
			ws = {}

		try:
			fs = _load_tool("make_finidat_subset")
			mw = _load_tool("make_warmstart")
			spec = ws.get("conus_restart") or mw.DEFAULT_CONUS_MANIFEST
			bands = mw.ConusBandSet(mw.resolve_conus_sources(spec))

			print("\n🌡️  STEP 0b: Warm Start (CONUS subset)")
			print("-" * 40)
			manifest = fs.build_finidats(columns, self.run_dir / "warmstart", bands)
			if not manifest:
				raise RuntimeError(
					"warm start produced no finidat for any column. The CONUS "
					"restart source is unreadable or the domain lies outside "
					"its coverage.")
			missing = [c.get("id") for c in columns if c.get("id") not in manifest]
			if missing:
				raise RuntimeError(
					f"warm start covered {len(manifest)}/{len(columns)} columns; "
					f"no donor for {', '.join(missing)}. A partial warm start "
					f"would put columns with different initial states and "
					f"different soil datasets in one ensemble.")

			# Snap to the donor cell so domain/surfdata/finidat agree exactly.
			for c in columns:
				m = manifest.get(c.get("id"))
				if m:
					c["lat"], c["lon"] = m["donor_lat"], m["donor_lon"]

			(self.run_dir / "warmstart" / "warmstart.json").write_text(
				json.dumps(manifest, indent=2))
			snap = max(m["dist_km"] for m in manifest.values())
			print(f"✓ {len(manifest)}/{len(columns)} column(s) warm-started from "
				  f"CONUS (snapped <= {snap} km) → warmstart/warmstart.json")
			return manifest
		except Exception as e:
			raise RuntimeError(f"warm start failed: {e}") from e

	# ─────────────────────────────────────────────────────────
	# STEP 1 — BUILD (writes to 01_inputs/)
	# ─────────────────────────────────────────────────────────
	def _build_column_inputs(self, config: Dict[str, Any]) -> Dict[str, Any]:
		"""domain.nc + surface.nc per column, before any CIME work.

		The builder generates these itself, per coupler, deep inside
		_build_one(). Doing it here first changes nothing about WHAT ELM
		receives — the generators key their output on coordinates plus a
		content hash, so the builder's calls become cache hits on the very
		files written here. Verified against the 19-column 2019 Upper
		Gunnison run: every filename matches what that run actually used.

		What it changes is when you find out. Surface generation reads the
		CONUS donor and can fail for a column; buried in _build_one that
		surfaces partway through case creation, after CIME work has begun.
		Here it fails before anything expensive starts, names the column, and
		leaves 01_inputs/column_inputs.json saying which columns have inputs
		and which do not.

		Non-fatal by design: the builder retains its own generation path, so
		a failure here costs the early warning and the manifest, not the run.
		"""
		try:
			bci = _load_tool("build_column_inputs")
			res = bci.build_all(
				self.run_dir,
				soil_config = config.get("soil_config", "native"),
				substrate   = config.get("substrate",   "extrapolate"),
				quiet       = True,
			)
			n_ok, n_bad = len(res.get("built") or {}), len(res.get("failed") or {})
			print(f"✓ column inputs: {n_ok} built"
				  + (f", {n_bad} FAILED" if n_bad else "")
				  + ("  (warm)" if res.get("warm_started") else "  (cold)"))
			for cid, why in (res.get("failed") or {}).items():
				print(f"   ⚠️  {cid}: {why}")
			return res
		except Exception as e:                                  # noqa: BLE001
			print(f"   ⚠️  pre-building column inputs failed ({e}) — the "
				  f"builder will generate them per column instead")
			return {}

	# ─────────────────────────────────────────────────────────
	# THE MCP PATH — Phase 4
	# ─────────────────────────────────────────────────────────
	# D5: the MCP is the DEFAULT when an `elm` client is registered. With no
	# client in mcp_config.json the local path still runs — the choice is made
	# by what is configured, not by a failure, so this is not the silent
	# demotion commit d61eaed forbade. Set run_via_mcp=False to force local.
	MCP_NAME = "elm"

	def _mcp(self, config: Dict[str, Any]):
		"""The elm MCP client, or None to run locally."""
		if not (config or {}).get("run_via_mcp", True):
			return None
		return ((config or {}).get("mcp_clients") or {}).get(self.MCP_NAME)

	@staticmethod
	def _mcp_call(client, tool: str, args: Dict[str, Any],
				  budget: Optional[float] = None) -> Dict[str, Any]:
		"""One MCP call, with the client's timeout raised for its duration.

		The per-server timeout is sized for the quick tools. build_elm_cases
		generates per-column surfaces and collect_elm_results reads NetCDF, and
		neither is quick for 19 columns — a client ceiling that was 15x too
		small is exactly how the reaction MCP failed before commit 19fffc3.
		"""
		prev = getattr(client, "timeout", None)
		try:
			if budget and prev is not None and prev < budget:
				client.timeout = budget
			out = client.call_tool_json(tool, args)
		finally:
			if prev is not None:
				client.timeout = prev
		if out is None:
			raise RuntimeError(
				f"the elm MCP did not answer {tool} within its timeout")
		# `error` means the CALL could not be made. A payload carrying `ok` is
		# reporting an OUTCOME — a build that failed, an ensemble with no
		# results — and the caller has more to say about that than this does,
		# including the job's log. Raising here would make those messages dead
		# code and replace them with a one-liner.
		if isinstance(out, dict) and out.get("error") and "ok" not in out:
			raise RuntimeError(f"elm MCP {tool}: {out['error']}")
		return out


	def _build_cases_via_mcp(self, experiments, config, client):
		"""D1: the CIME build is a JOB. Returns a Pending, not case dirs.

		The server reads 01_inputs/case_inputs.json and nothing else — it does
		not generate inputs and does not open a surfdata file. _save_case_inputs has
		already written that file with each case's runtime_config, which is
		where FSURDAT and FINIDAT (the warm start's two products) cross the
		boundary as data.
		"""
		src = self.input_dir / self.CASE_INPUTS
		if not src.is_file():
			raise RuntimeError(
				f"{src} is missing — the framework builds the inputs and the "
				f"elm MCP only compiles cases against them")
		# UNATTENDED: one job for the whole study — build, run, and the
		# framework's own tail. The split flow costs three invocations, each
		# waiting on a queue; this costs one, and Slurm's END mail arrives
		# after the analysis is written rather than before it starts.
		#
		# Still the default-off switch rather than the default, because it
		# gives up the chance to look at the cases between building and
		# running — which is exactly what you want while something is wrong.
		if self._unattended(config):
			return self._run_study_via_mcp(experiments, config, client)

		print("   via the elm MCP — the case build is sbatch'd (D1)")
		out = self._mcp_call(client, "build_elm_cases", {
			"run_dir":  str(self.run_dir),
			"queue":    str(config.get("queue", "")),
			"walltime": str(config.get("prepare_walltime", "02:00:00")),
		}, budget=600)
		return Pending(out["job_id"], n_cases=len(experiments or []),
					   log=out.get("log_path"), via="mcp")

	@staticmethod
	def _unattended(config: Dict[str, Any]) -> bool:
		"""Should the whole study go as ONE job?

		Config wins over the environment so a single run can opt out of a
		machine-wide default.
		"""
		cfg = (config or {})
		if "unattended" in cfg:
			return bool(cfg["unattended"])
		return os.environ.get("IDEAS_UNATTENDED", "").strip().lower() in (
			"1", "true", "yes", "on")

	@staticmethod
	def _notify_email(config: Dict[str, Any]) -> str:
		"""Where Slurm should send the END,FAIL mail. Empty = do not ask for one.

		Never guessed from the username: a wrong address means the one signal
		the unattended flow depends on goes silently nowhere.
		"""
		return str((config or {}).get("notify_email")
				   or os.environ.get("IDEAS_NOTIFY_EMAIL", "")).strip()

	def _run_study_via_mcp(self, experiments, config, client):
		"""The entire study as one job: build, run, extract, package, analyze.

		Returns a Pending like the split path does, so the ledger and the
		resume machinery are unchanged — but the job itself finishes the run,
		so a resume is never actually needed. Anyone who does resume finds
		every stage already marked done, written by the job's own --finalize.
		"""
		src = self.input_dir / self.CASE_INPUTS
		if not src.is_file():
			raise RuntimeError(
				f"{src} is missing — the framework builds the inputs and the "
				f"elm MCP only compiles cases against them")
		email = self._notify_email(config)
		print("   via the elm MCP — the WHOLE study is one job "
			  "(build → run → analyze)")
		if not email:
			print("   ⚠️  no notify_email/IDEAS_NOTIFY_EMAIL — the study will "
				  "run unattended but nothing will tell you when it lands")
		out = self._mcp_call(client, "run_elm_study", {
			"run_dir":  str(self.run_dir),
			"queue":    str(config.get("queue", "")),
			"walltime": str(config.get("study_walltime", "02:00:00")),
			"email":    email,
		}, budget=600)
		if email:
			print(f"   ✉  {email} will be mailed when it finishes")
		return Pending(out["job_id"], n_cases=len(experiments or []),
					   log=out.get("log_path"), via="mcp", scope="study",
					   notify=email or None)

	def _submit_via_mcp(self, experiments, config, client):
		"""The ensemble as one batch job. Returns a Pending."""
		cases = [e.get("case_dir") for e in experiments if e.get("case_dir")]
		if not cases:
			raise RuntimeError(
				"no case directories — build_cases must run before the ensemble")
		out = self._mcp_call(client, "submit_elm_ensemble", {
			"run_dir":   str(self.run_dir),
			"case_dirs": cases,
			"queue":     str(config.get("queue", "")),
			"walltime":  str(config.get("walltime", "00:40:00")),
			"email":     str(config.get("email", "")),
		}, budget=600)
		print(f"   submitted {out.get('n_cases')} column(s) as job "
			  f"{out['job_id']} via the elm MCP")
		return Pending(out["job_id"], n_cases=out.get("n_cases"),
					   log=out.get("log_path"), via="mcp")

	def _poll_via_mcp(self, record, experiments, config, client):
		"""Ask the server whether this stage's job has landed.

		Dispatches on record['stage'], which Phase 3b added for exactly this:
		a finished CIME build hands back case directories and a finished
		ensemble hands back outcomes, and a job id does not say which.
		"""
		stage = record.get("stage")
		st = self._mcp_call(client, "check_elm_job",
							{"job_id": str(record.get("job_id")),
							 "run_dir": str(self.run_dir)})
		if st.get("active", True):
			print(f"   job {record.get('job_id')} is "
				  f"{st.get('state') or 'unanswered'} — nothing to collect yet")
			return None

		if stage == "build_cases":
			# The case directories come back from check_elm_job itself. A stage
			# that hands back a job id needs somewhere to hand back its ANSWER,
			# and a few short strings can travel inline — unlike results, which
			# is why reading those stayed on this side.
			if not st.get("ok"):
				raise RuntimeError(
					f"the case build failed: {st.get('error')}"
					+ (f"\n{st.get('log_tail')}" if st.get("log_tail") else ""))
			got = st
			by_name = {c.get("case_name"): c.get("case_dir")
					   for c in (got.get("cases") or [])}
			for e in experiments:
				if by_name.get(e.get("case_name")):
					e["case_dir"] = by_name[e["case_name"]]
			print(f"   ✓ {got.get('n_ok')}/{got.get('n_total')} case(s) built")
			return experiments

		# stage == "run": the scheduler is done, so the FILES decide.
		print(f"   job {record.get('job_id')} finished "
			  f"({st.get('state')}) — collecting")
		return self._collect(experiments, self._outcomes_from_disk(experiments))


	def _build_case_inputs(self,
			   plan:   Dict[str, Any],
			   config: Dict[str, Any]) -> List[Dict]:
		"""Build the ELMAgentAdapter list from the plan (per-column surfaces)."""
		self._build_column_inputs(config)

		builder     = ELMExperimentBuilder(plan)
		self._builder = builder          # reused by _build_cases for the fast path
		experiments = builder.build_experiments()

		# Save experiment_summary.json to 01_inputs/
		summary_file = self.input_dir / "experiment_summary.json"
		with open(summary_file, 'w') as f:
			json.dump(
				builder.get_experiment_summary(),
				f, indent=2, default=str
			)

		print(f"✓ {len(experiments)} experiment(s) built")
		return experiments

	def _plot_setups(self, cases: List[str]) -> None:
		"""02_setup_plots/column_surfaces.png — the soil each column ACTUALLY got.

		Runs after _build_cases, because it reads every case's generated FSURDAT via
		its `run/lnd_in`; that is the only way to see what ELM will really use.
		It also cross-checks each surface's lat/lon against the case's domain
		file and shouts if they disagree — that mismatch aborts ELM at init.

		This replaced a plan-driven version that drew soil from ELM_CONFIG
		(which never carries soil — it is per-coupler) and labelled every x-axis
		"Qian 1948-2004" while the pipeline runs NLDAS. It was confidently wrong
		on both counts, which is worse than having no figure.

		Non-fatal.
		"""
		if not cases:
			return
		try:
			pc  = _load_tool("plot_columns")
			out = self.setup_plots_dir / "column_surfaces.png"
			pc.plot_surfaces(list(cases), str(out))
			print(f"✓ setup plot → 02_setup_plots/{out.name}")
		except Exception as e:
			print(f"   ⚠️  surface plot failed: {e}")

	# ─────────────────────────────────────────────────────────
	# STEP 2 — BUILD CASES (cases live at $PSCRATCH)
	# ─────────────────────────────────────────────────────────
	def _build_cases(self, experiments: List[Dict], config: Dict[str, Any] = None):
		"""
		Build all ELM cases.

		Uses ELMExperimentBuilder.build_cases(): builds the FIRST case from
		scratch (~8-10 min CIME compile) and clones the rest with
		--keepexe (~30 s each, in parallel, with a serial retry pass for the
		known parallel-filesystem race). The previous implementation called
		prepare_case() per experiment with no ref_case_dir, i.e. a full
		compile for EVERY column — ~2 h for a 14-column watershed instead of
		~12 min.
		"""
		client = self._mcp(config or {})
		if client is not None:
			return self._build_cases_via_mcp(experiments, config or {}, client)

		builder = getattr(self, "_builder", None)
		if builder is None and getattr(self, "_resume_plan", None) is not None:
			# Resumed past _build_case_inputs, and _build_cases turns out to be needed after
			# all. The builder is reconstructible from the plan — that is the
			# whole reason the plan is persisted — so rebuild it here, where
			# the cost is actually incurred, rather than on every resume.
			print("   ↻ rebuilding the case builder from the persisted plan")
			builder = ELMExperimentBuilder(self._resume_plan)
			builder.build_experiments()
			self._builder = builder
		if builder is not None:
			case_dirs = builder.build_cases(output_dir=str(self.run_dir))
			for exp, cd in zip(experiments, case_dirs):
				exp['case_dir'] = cd
			n_ok = sum(1 for c in case_dirs if c)
			if not n_ok:
				raise RuntimeError("No ELM cases could be built.")
			if n_ok != len(case_dirs):
				print(f"   ⚠️  {len(case_dirs) - n_ok} case(s) failed to build")
		else:
			# Fallback: no builder handle (e.g. a caller that bypassed _build_case_inputs)
			for exp in experiments:
				exp['case_dir'] = exp['elm_agent'].prepare_case(
					output_dir = str(self.run_dir)
				)

		# Now that each case has a run/lnd_in, plot the soil it actually got.
		self._plot_setups([e['case_dir'] for e in experiments if e.get('case_dir')])

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
		"""
		Run all simulations. Returns {case_name: bool} when it waited, or a
		Pending marker when config['detach'] asked it to submit and return.

		Detached is the interesting mode: a 19-column ELM ensemble is ~40
		minutes of queue plus wall clock, and waiting for it holds a Python
		process — and whatever session started it — for the duration. Submitting
		and recording the job id lets the study be picked up later, from
		anywhere, by re-entering with resume=True.
		"""
		client = self._mcp(config)
		if client is not None:
			return self._submit_via_mcp(experiments, config, client)

		detach  = bool(config.get("detach"))
		results = self._run_batch(experiments, config, wait=not detach)
		if isinstance(results, Pending):
			return results
		if results is None:
			# Fallback: bare srun, serially. Requires an interactive node.
			# There is no job id to hand back here, so detach cannot apply.
			if detach:
				print("   ⚠️  detach requested but batch submission is not "
					  "usable — running serially instead")
			results = {}
			for exp in experiments:
				results[exp['case_name']] = exp['elm_agent'].run_simulation()

		return self._collect(experiments, results)

	def _collect(self,
				 experiments: List[Dict],
				 results:     Dict[str, bool]) -> Dict[str, bool]:
		"""Everything that happens once the columns have stopped running.

		Split out of _run because a detached ensemble reaches this point from
		_poll instead, in a later session — and the reports it writes are the
		same reports either way.
		"""
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
		"""The case's run summary, from its live agent when there is one.

		A resumed run has no live agent — the build manifest is JSON — so the
		one field anything downstream reads (history_files) is taken off disk
		instead. Asking the string repr of an ELMAgent for a summary is how
		this reported zero history files for cases that had plenty.
		"""
		agent = exp.get('elm_agent')
		if hasattr(agent, "get_run_summary"):
			return agent.get_run_summary()
		cd = exp.get('case_dir')
		return {'history_files': sorted(glob.glob(f"{cd}/run/*.elm.h0.*.nc"))
								 if cd else []}

	def _outcomes_from_disk(self, experiments: List[Dict]) -> Dict[str, bool]:
		"""Which columns produced output. The definition of success used by
		both the waiting and the detached path: a column succeeded if ELM
		wrote it a history file."""
		return {e['case_name']: bool(e.get('case_dir') and
									 glob.glob(f"{e['case_dir']}/run/*.elm.h0.*.nc"))
				for e in experiments}

	def _poll(self, record: Dict[str, Any], experiments, config):
		"""Has the detached ensemble finished? Results if so, None if not.

		Two sources, in order:

		  1. SLURM. While it still owns the job, nothing else matters — a
		     half-written history file from a running column would otherwise
		     read as success.
		  2. run.log's ALL_DONE. Used only when SLURM will not answer (sacct
		     purged, squeue unreachable), because the alternative is to call an
		     ensemble finished on no evidence at all.

		Note what is NOT here: a column whose job COMPLETED but which wrote no
		history file is a failure, not a reason to keep waiting. The scheduler
		being done is what ends the wait; the files decide the outcome.
		"""
		client = self._mcp(config or {})
		if client is not None:
			return self._poll_via_mcp(record, experiments, config or {}, client)

		jid   = record.get("job_id")
		state = self._slurm_state(jid)
		if state in self.ACTIVE_JOB_STATES:
			print(f"   job {jid} is {state} — nothing to collect yet")
			return None
		if state is None:
			log = self.run_dir / "run.log"
			done = log.exists() and "ALL_DONE" in log.read_text(errors="replace")
			if not done:
				print(f"   ⚠️  SLURM will not say what job {jid} is doing and "
					  f"run.log has no ALL_DONE — treating it as still running")
				return None
			print(f"   job {jid} is gone from SLURM but run.log says ALL_DONE")
		else:
			print(f"   job {jid} finished ({state}) — collecting")

		return self._collect(experiments, self._outcomes_from_disk(experiments))

	# Keys of an experiment that are plain data and mean something downstream.
	# LISTED, not "everything except elm_agent": a new object added upstream
	# would otherwise silently become a repr in the case-inputs file.
	CASE_KEYS = (
		"scenario_index", "scenario_name", "case_name", "forcing_period",
		"soil_config", "substrate", "forcing_start", "forcing_end", "stop_n",
		"start_date", "description", "lat", "lon", "elevation_m", "band",
		"case_dir",
	)

	def _serialise_case_inputs(self, experiments):
		"""Each case as data, with the adapter replaced by what BUILT it.

		runtime_config is the adapter's whole input — FSURDAT, FINIDAT, the
		domain paths, STOP_N, the forcing years — so the elm MCP can construct
		an identical adapter without the object ever crossing. That config is
		also where the warm start's two products (the subset restart and the
		donor-soil surface) leave this side as plain paths.
		"""
		out = []
		for e in experiments:
			row = {k: e.get(k) for k in self.CASE_KEYS if k in e}
			rc = getattr(e.get("elm_agent"), "runtime_config", None)
			if isinstance(rc, dict):
				row["runtime_config"] = dict(rc)
			elif isinstance(e.get("runtime_config"), dict):
				row["runtime_config"] = dict(e["runtime_config"])
			out.append(row)
		return out

	def _rehydrate_handles(self, experiments, plan, config) -> None:
		"""A rehydrated ELM build has no ELMAgents and no builder.

		Both are objects; the manifest is JSON. The agents come back as string
		reprs, which are truthy and have no methods — so they are dropped here
		rather than left to fail at the call site. The builder is not
		reconstructed eagerly: rebuilding it re-runs per-column file generation,
		which a resume that only needs to collect finished output should not
		pay for. _build_cases rebuilds it on demand from this plan.
		"""
		self._resume_plan = plan
		stale = [e for e in experiments if isinstance(e.get('elm_agent'), str)]
		for e in stale:
			e.pop('elm_agent', None)
		if stale:
			print(f"   ↻ {len(stale)} case handle(s) are not JSON — resuming "
				  f"from the case directories on disk")

	def _run_batch(self,
				   experiments: List[Dict],
				   config:      Dict[str, Any],
				   wait:        bool = True):
		"""
		Run every column as ONE sbatch job via tools/submit_cases.sh.

		Works from a login node (the bare-srun path needs a pre-held
		allocation) and runs the columns concurrently on one node.

		wait=True  → --wait, blocks, returns {case_name: bool}
		wait=False → submits and returns Pending(job_id)

		Either way returns None if batch submission isn't usable, in which case
		the caller falls back to serial srun.
		"""
		import shutil, subprocess
		if config.get("no_batch") or not shutil.which("sbatch"):
			return None

		cases = [e.get('case_dir') for e in experiments if e.get('case_dir')]
		if not cases:
			return None
		try:
			exeroot = subprocess.check_output(
				["./xmlquery", "EXEROOT", "--value"], cwd=cases[0],
				env={**__import__("os").environ,
					 "LC_ALL": "en_US.utf8", "LANG": "en_US.utf8"},
				text=True).strip()
		except Exception as e:
			print(f"   ⚠️  could not resolve EXEROOT ({e}) — serial fallback")
			return None

		# submit_cases.sh reads these two files from the run dir
		(self.run_dir / "cases.json").write_text(json.dumps(cases, indent=1))
		(self.run_dir / "exe_path.txt").write_text(
			str(Path(exeroot) / "e3sm.exe") + "\n")

		cmd = ["bash", str(_ROOT / "tools" / "submit_cases.sh"),
			   str(self.run_dir),
			   "-q", str(config.get("queue", "short")),
			   "-t", str(config.get("walltime", "00:40:00"))]
		if wait:
			cmd.append("--wait")
		if config.get("email"):
			cmd += ["-m", str(config["email"])]
		print(f"   submitting {len(cases)} column(s) as one batch job "
			  f"(queue={config.get('queue', 'short')}"
			  f"{'' if wait else ', detached'}) ...")
		try:
			# Detached: capture stdout to read the job id back out of it. The
			# waiting path keeps streaming to the terminal, where the per-column
			# summary the script prints is the point.
			proc = subprocess.run(cmd, cwd=str(_ROOT), check=False,
								  capture_output=not wait, text=True)
		except Exception as e:
			print(f"   ⚠️  batch submission failed ({e}) — serial fallback")
			return None

		if not wait:
			out = (proc.stdout or "") + (proc.stderr or "")
			print(out.rstrip())
			m = re.search(r"submitted job (\d+)", out)
			if not m:
				# No job id means nothing to resume with. Say so and fall back
				# rather than returning a Pending nobody can ever poll.
				print("   ⚠️  submitted but no job id in the output — cannot "
					  "detach; falling back")
				return None
			return Pending(m.group(1), n_cases=len(cases),
						   queue=str(config.get("queue", "short")),
						   walltime=str(config.get("walltime", "00:40:00")),
						   log=str(self.run_dir / "run.log"))

		# Success == the column actually produced history files.
		return self._outcomes_from_disk(experiments)

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
				 config: Dict[str, Any] = None) -> ELMResultsAnalyzer:
		"""Read the ELM history NetCDFs and pull the numbers out → 04_analysis/.

		This is extraction, not analysis: it knows ELM's output format and
		nothing about what the numbers mean. Figures, observation comparison
		and interpretation belong to the Analyzer (src/agents/analyzer.py) and
		are no longer reachable from here.

		Also attaches the honesty payload (structural + configuration
		limitations, assumptions ledger). ELMResultsAnalyzer already computes
		the confounding notes, fit r2 and soil attribution; only
		tools/analyze_run.py used to add `extra_summary`, so runs driven
		through this manager silently lost the caveats.
		"""
		analyzer = ELMResultsAnalyzer(
			experiments  = experiments,
			analysis_dir = str(self.analysis_dir),
		)

		cfg  = config or {}
		plan = plan or {}
		try:
			from core.limitations import select_limitations
			couplers = plan.get("CONDITIONS_COUPLERS") or [{}]
			y0 = int(couplers[0].get("DATM_CLMNCEP_YR_START", cfg.get("yr_start", 1995)))
			y1 = int(couplers[0].get("DATM_CLMNCEP_YR_END",   cfg.get("yr_end", y0)))
			analyzer.extra_summary = {
				"limitations": select_limitations(
					n_years       = max(1, y1 - y0 + 1),
					warm_start    = bool(couplers[0].get("FINIDAT")),
					forcing       = cfg.get("forcing", "nldas"),
					spinup_years  = int(cfg.get("spinup_years", 0)),
					# WHICH warm start, and whether the soil was kept with it.
					# Without these the caveat cannot tell an equilibrated,
					# self-consistent state from the mismatch that made year
					# one a relaxation — and it defaulted to warning about both.
					warm_source   = ((cfg.get("warm_start") or {}).get("source")
									 if isinstance(cfg.get("warm_start"), dict)
									 else cfg.get("warm_start")) or "conus",
					soil_source   = "conus",   # warm start is required; the
					# donor's surfdata is always what ELM runs on
				),
				"assumptions_ledger": (
					json.loads((self.run_dir / "assumptions.json").read_text())
					if (self.run_dir / "assumptions.json").exists() else []),
			}
		except Exception as e:
			print(f"   ⚠️  limitations payload unavailable ({e})")

		analyzer.extract_all()

		return analyzer

	# ─────────────────────────────────────────────────────────
	# STEP 4d — ONE-WAY ELM → PFLOTRAN COUPLING
	# ─────────────────────────────────────────────────────────
	def _couple_pflotran(self,
						 plan:   Dict[str, Any],
						 config: Dict[str, Any]) -> bool:
		"""Drive a 1-D PFLOTRAN column per ELM column with that column's own
		daily QINFL -> 05_pflotran/. Non-fatal.

		Runs when the planner emitted a coupling_design (the archetype the
		Reception and Planner prompts already speak), or when config forces it
		with pflotran={'run': True}. Both prompts have described this coupling
		for months; nothing executed it, so "one-way coupling" was a design the
		framework could plan but never perform.

		The columns are the SAME ones ELM just ran — same lat/lon, same SSURGO
		profile, same Fan water table — so the two models answer one question
		jointly: ELM partitions the surface water, PFLOTRAN carries it down.
		1-D Richards columns take seconds, so this runs serially in-process
		with no batch queue.
		"""
		cfg = (config or {}).get("pflotran") or {}
		coupling = plan.get("coupling_design") or {}
		archetype = ((plan.get("model_choice") or {}).get("design_archetype") or "")
		wanted = bool(cfg.get("run")) or bool(coupling) or archetype == "coupling"
		if not wanted:
			return False

		print("\n🪨 STEP 4d: Coupling ELM → PFLOTRAN")
		print("-" * 40)
		if coupling.get("driver"):
			print(f"   driver: {coupling['driver']}")

		# Checked before the try so a missing file reports what is missing,
		# not a bare errno from json.loads.
		cols_f = self.run_dir / "columns.json"
		cols = []
		if cols_f.exists():
			cols = json.loads(cols_f.read_text())
			cols = cols.get("columns", cols) if isinstance(cols, dict) else cols
		if not cols:
			print("   ⚠️  no columns.json in this run — skipping coupling")
			return False

		try:
			bp  = _load_tool("build_pflotran_cases")
			out = self.run_dir / "05_pflotran"

			res = bp.build_ensemble(
				cols, out,
				flux_from  = str(self.run_dir),      # this run's QINFL
				run        = cfg.get("run", True),
				timeout    = int(cfg.get("timeout", 1800)),
				spin_years = float(cfg.get("spin_years", 10.0)),
				depth_cap  = float(cfg.get("depth_cap", 50.0)),
				bottom     = cfg.get("bottom", "fan"),
			)
			cases = res.get("cases") or []
			if not cases:
				print("   ⚠️  no PFLOTRAN decks built (no ELM flux?) — skipping")
				return False

			n_ok = sum(1 for m in cases if m.get("run_ok"))
			print(f"✓ {n_ok}/{len(cases)} PFLOTRAN column(s) ran → 05_pflotran/")
			if not n_ok:
				return False

			ap = _load_tool("analyze_pflotran_coupled")
			ap.analyze_coupled(out, quiet=True)
			print("✓ lag + attenuation → 05_pflotran/pflotran_summary_coupled.json")
			return True
		except Exception as e:
			print(f"   ⚠️  coupling failed ({e}) — the ELM study stands")
			return False
