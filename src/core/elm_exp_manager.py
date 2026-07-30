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
import json
import sys
from pathlib  import Path
from datetime import datetime
from typing   import Dict, Any, List

sys.path.insert(0, "src")

from agents.analyzer          import Analyzer
from core.exp_manager_base       import ExperimentManagerBase
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

	Stages: materialize -> warm start -> build -> prepare -> run -> extract
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
	def execute_plan(self,
					 experiment_plan: Dict[str, Any],
					 config:          Dict[str, Any]
					 ) -> Dict[str, Any]:
		"""Execute complete ELM experiment plan."""
		start_time = datetime.now()

		try:
			# Step 0 — Materialize sampling (strategy → concrete columns)
			# The capability-aware planner emits sampling_strategy, NOT
			# CONDITIONS_COUPLERS. Turning one into the other is the
			# "Materialize sampling" stage of the Experiment Manager; it
			# used to live only in tools/expand_sampling.py, so a plan
			# straight from the planner could never be executed.
			experiment_plan = self._materialize(experiment_plan, config)

			# Step 1 — Build → 01_inputs/ (setup plots come after step 2,
			# once each case has a run/lnd_in to read its real FSURDAT from)
			print("📋 STEP 1: Building Experiments")
			print("-" * 40)
			experiments = self._build(
				experiment_plan, config
			)

			# Step 2 — Prepare (cases live at $PSCRATCH)
			print("\n⚙️  STEP 2: Preparing Cases")
			print("-" * 40)
			self._prepare(experiments)

			# Step 3 — Run + write 03_results/
			print("\n🌿 STEP 3: Running Simulations")
			print("-" * 40)
			results = self._run(experiments, config)

			# Step 4 — Extract → 04_analysis/. Reading ELM's own history
			# format is this backend's job; saying what the numbers mean is
			# not, and everything past this line belongs to the Analyzer.
			print("\n📊 STEP 4: Extracting Results")
			print("-" * 40)
			analyzer = self._extract(
				experiments, plan=experiment_plan, config=config)

			# Step 4b — PACKAGE FIRST. experiment.json is this manager's
			# product and the Analyzer's input, so it is written before
			# anything interpretive runs. Ordered the other way, a crash in
			# the Analyzer took the results package with it — the ensemble
			# was computed, and nothing on disk said so.
			print("\n📦 STEP 4b: Packaging Results")
			print("-" * 40)
			# Guarded for the same reason the Analyzer is: the compute has
			# already succeeded and _extract has already written
			# 04_analysis/hydro_summary.json, so a packaging bug must not
			# discard an ensemble that cost a queue slot and an hour. Loud,
			# because experiment.json is what everything downstream reads.
			try:
				self._package(experiment_plan, analyzer, config)
			except Exception as e:                              # noqa: BLE001
				print(f"   ✗ PACKAGING FAILED ({e}) — the results are still "
					  f"in 04_analysis/hydro_summary.json")

			# Step 4c — the Analyzer box: figures, observation comparison,
			# interpretation. Non-fatal AS A WHOLE, not merely stage by
			# stage: an unexpected failure here must not cost the compute
			# that produced the numbers.
			print("\n🔭 STEP 4c: Analyzer")
			print("-" * 40)
			try:
				Analyzer(str(self.run_dir)).run(results=analyzer, config=config)
			except Exception as e:                              # noqa: BLE001
				print(f"   ⚠️  analyzer failed ({e}) — experiment.json stands")

			# Step 4d — one-way ELM → PFLOTRAN coupling, when the plan asks
			# for it. Non-fatal: the ELM study stands on its own.
			try:
				self._couple_pflotran(experiment_plan, config)
			except Exception as e:                              # noqa: BLE001
				print(f"   ⚠️  PFLOTRAN coupling failed ({e}) — the ELM run "
					  f"stands")

			# Step 5 — the alias the standalone report tools open by name.
			# experiment.json was already written at 4b.
			print("\n📦 STEP 5: Packaging LLM Input")
			print("-" * 40)
			self._save_llm_input(experiment_plan, analyzer)

			# Create + save run summary (top level)
			end_time    = datetime.now()
			run_summary = self._create_run_summary(
				experiment_plan, experiments,
				results, start_time, end_time
			)
			self._save_run_summary(run_summary)

			n_ok  = run_summary['experiments_success']
			n_tot = run_summary['experiments_total']
			rt    = run_summary['total_runtime_seconds']
			print(f"\n{'=' * 60}")
			print(f"ELM COMPLETE: {n_ok}/{n_tot} | {rt:.1f}s")
			print(f"Output: {self.run_dir}")
			print(f"{'=' * 60}\n")

			return run_summary

		except Exception as e:
			self._save_error(e, start_time)
			raise
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
	def _build_inputs(self, config: Dict[str, Any]) -> Dict[str, Any]:
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

	def _build(self,
			   plan:   Dict[str, Any],
			   config: Dict[str, Any]) -> List[Dict]:
		"""Build the ELMAgentAdapter list from the plan (per-column surfaces)."""
		self._build_inputs(config)

		builder     = ELMExperimentBuilder(plan)
		self._builder = builder          # reused by _prepare for the fast path
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

		Runs after _prepare, because it reads every case's generated FSURDAT via
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
	# STEP 2 — PREPARE (cases live at $PSCRATCH)
	# ─────────────────────────────────────────────────────────
	def _prepare(self, experiments: List[Dict]) -> None:
		"""
		Prepare (build) all ELM cases.

		Uses ELMExperimentBuilder.prepare_cases(): builds the FIRST case from
		scratch (~8-10 min CIME compile) and clones the rest with
		--keepexe (~30 s each, in parallel, with a serial retry pass for the
		known parallel-filesystem race). The previous implementation called
		prepare_case() per experiment with no ref_case_dir, i.e. a full
		compile for EVERY column — ~2 h for a 14-column watershed instead of
		~12 min.
		"""
		builder = getattr(self, "_builder", None)
		if builder is not None:
			case_dirs = builder.prepare_cases(output_dir=str(self.run_dir))
			for exp, cd in zip(experiments, case_dirs):
				exp['case_dir'] = cd
			n_ok = sum(1 for c in case_dirs if c)
			if not n_ok:
				raise RuntimeError("No ELM cases could be prepared.")
			if n_ok != len(case_dirs):
				print(f"   ⚠️  {len(case_dirs) - n_ok} case(s) failed to prepare")
		else:
			# Fallback: no builder handle (e.g. a caller that bypassed _build)
			for exp in experiments:
				exp['case_dir'] = exp['elm_agent'].prepare_case(
					output_dir = str(self.run_dir)
				)

		# Now that each case has a run/lnd_in, plot the soil it actually got.
		self._plot_setups([e['case_dir'] for e in experiments if e.get('case_dir')])

	# ─────────────────────────────────────────────────────────
	# STEP 3 — RUN (writes to 03_results/)
	# ─────────────────────────────────────────────────────────
	def _run(self,
			 experiments: List[Dict],
			 config:      Dict[str, Any]) -> Dict[str, bool]:
		"""
		Run all simulations via srun (blocking; requires interactive node).
		After runs complete, write execution_report.txt and
		results_summary.csv to 03_results/.
		"""
		results = self._run_batch(experiments, config)
		if results is None:
			# Fallback: bare srun, serially. Requires an interactive node.
			results = {}
			for exp in experiments:
				results[exp['case_name']] = exp['elm_agent'].run_simulation()

		for exp in experiments:
			try:
				exp['run_summary'] = exp['elm_agent'].get_run_summary()
			except Exception as e:
				print(f"   ⚠️  run summary failed for {exp['case_name']}: {e}")

		n_ok   = sum(1 for v in results.values() if v)
		n_fail = len(results) - n_ok
		print(f"✓ {n_ok} succeeded, {n_fail} failed")

		# Post-run reporting → 03_results/
		self._write_execution_report(experiments, results)
		self._write_results_csv(experiments, results)

		return results

	def _run_batch(self,
				   experiments: List[Dict],
				   config:      Dict[str, Any]):
		"""
		Run every column as ONE sbatch job via tools/submit_cases.sh --wait.

		Works from a login node (the bare-srun path needs a pre-held
		allocation) and runs the columns concurrently on one node. Returns
		{case_name: bool} or None if batch submission isn't usable, in which
		case the caller falls back to serial srun.
		"""
		import shutil, subprocess, glob
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
			   str(self.run_dir), "--wait",
			   "-q", str(config.get("queue", "short")),
			   "-t", str(config.get("walltime", "00:40:00"))]
		if config.get("email"):
			cmd += ["-m", str(config["email"])]
		print(f"   submitting {len(cases)} column(s) as one batch job "
			  f"(queue={config.get('queue', 'short')}) ...")
		try:
			subprocess.run(cmd, cwd=str(_ROOT), check=False)
		except Exception as e:
			print(f"   ⚠️  batch submission failed ({e}) — serial fallback")
			return None

		# Success == the column actually produced history files.
		results = {}
		for exp in experiments:
			cd = exp.get('case_dir')
			results[exp['case_name']] = bool(
				cd and glob.glob(f"{cd}/run/*.elm.h0.*.nc"))
		return results

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
