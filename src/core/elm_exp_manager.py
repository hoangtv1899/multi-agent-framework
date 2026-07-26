#!/usr/bin/env python3
"""
ELM Experiment Manager
src/core/elm_exp_manager.py

Single responsibility: orchestrate the ELM execution pipeline — the
Experiment Manager box of the framework (see ARCHITECTURE.md).

Output directory structure:

    elm_run_YYYYMMDD_HHMMSS/
        ├── columns.json  run_plan.json  plan.json  reception_brief.json
        ├── sampling_design.png                     (step 0, materialize)
        ├── 01_inputs/
        │   └── experiment_summary.json
        ├── 02_setup_plots/
        │   └── column_surfaces.png                 (real FSURDAT per column)
        ├── 03_results/
        │   ├── execution_report.txt
        │   └── results_summary.csv
        ├── 04_analysis/
        │   ├── hydro_summary.json  validation.json  interpretation.md
        │   ├── elevation_gradient.png  soil_control.png
        │   └── water_budget.png  driver_response.png  wtd_columns.png
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

from core.elm_experiment_builder import ELMExperimentBuilder
from core.elm_results_analyzer   import ELMResultsAnalyzer
from core.columns_to_plan        import columns_to_elm_plan


# Repo root — this file is <root>/src/core/elm_exp_manager.py
_ROOT = Path(__file__).resolve().parents[2]


def _load_tool(name: str):
	"""Import tools/<name>.py by path — tools/ is a script dir, not a package.

	The manager and the standalone CLIs then share ONE implementation of each
	stage instead of drifting apart: expand_sampling (materialize),
	plot_columns (setup figures), validate_run (4b), interpret_run (4c).
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
class ELMExpManager:
	"""
	Executes ELM experiment plans.

	Stages: materialize -> build -> prepare -> run -> analyze -> validate
	-> interpret -> package. Steps 0 and 4/4b/4c delegate to the tools/
	CLIs via _load_tool(), so both entry points share one implementation.
	"""

	def __init__(self,
				 base_output_dir: str = "./workflow_outputs"):
		self.base_output_dir = Path(base_output_dir)
		timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
		self.run_dir = (
			self.base_output_dir / f"elm_run_{timestamp}"
		)
		self.run_dir.mkdir(parents=True, exist_ok=True)

		# Four numbered subdirs — names match PFLOTRAN exactly
		self.input_dir       = self.run_dir / "01_inputs"
		self.setup_plots_dir = self.run_dir / "02_setup_plots"
		self.results_dir     = self.run_dir / "03_results"
		self.analysis_dir    = self.run_dir / "04_analysis"

		for d in [self.input_dir, self.setup_plots_dir,
				  self.results_dir, self.analysis_dir]:
			d.mkdir(exist_ok=True)

		print(f"\n{'=' * 60}")
		print(f"ELM Experiment Manager")
		print(f"Run dir : {self.run_dir}")
		print(f"Cases   : $PSCRATCH/E3SMv3/")
		print(f"{'=' * 60}\n")

	# ─────────────────────────────────────────────────────────
	# MAIN ENTRY POINT — mirrors ExpManager exactly
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

			# Step 4 — Analyze → 04_analysis/
			print("\n📊 STEP 4: Analyzing Results")
			print("-" * 40)
			analyzer = self._analyze(
				experiments, plan=experiment_plan, config=config)

			# Step 4b/4c — validate against observations, then interpret.
			# Both are non-fatal; the run stands without them.
			print("\n🔭 STEP 4b: Validating Against Observations")
			print("-" * 40)
			self._validate(config)

			print("\n🧭 STEP 4c: Interpreting")
			print("-" * 40)
			self._interpret(config)

			# Step 5 — Package for LLM (top level)
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

	# ─────────────────────────────────────────────────────────
	# STEP 0 — MATERIALIZE SAMPLING (strategy → concrete columns)
	# ─────────────────────────────────────────────────────────
	def _materialize(self,
					 plan:   Dict[str, Any],
					 config: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Turn a capability-aware planner strategy into CONDITIONS_COUPLERS.

		No-op when the plan already carries CONDITIONS_COUPLERS (the legacy
		single-site plan shape), so old plans keep working unchanged.

		Needs, via config:
			brief        — reception brief (domain bbox + optional HUC)
			mcp_clients  — dict of live MCP clients (terrain/fan_wtd/geology)
		Optional: yr_start, yr_end, soil_config, substrate, n_columns, n_bands.
		"""
		if plan.get("CONDITIONS_COUPLERS"):
			return plan

		exp = _load_tool("expand_sampling")
		brief   = config.get("brief") or {}
		clients = config.get("mcp_clients") or {}

		bbox = exp._bbox_from_brief(brief)
		if not bbox:
			raise ValueError(
				"Cannot materialize sampling: no domain bbox in the reception "
				"brief, and the plan has no CONDITIONS_COUPLERS. Pass "
				"config['brief'] with domain.bbox, or supply an explicit plan."
			)
		if not clients.get("terrain"):
			raise ValueError(
				"Cannot materialize sampling: the 'terrain' MCP client is "
				"required. Pass config['mcp_clients']."
			)

		n_total = config.get("n_columns") or exp._n_from_plan(plan)
		if not n_total:
			raise ValueError(
				"Cannot materialize sampling: no column count in the plan "
				"(sampling_strategy.n_exploratory) and none in config."
			)
		bands = config.get("n_bands") or len(
			(brief.get("heterogeneity") or {}).get("elevation_bands") or []) or 4

		print("🗺️  STEP 0: Materializing Sampling")
		print("-" * 40)
		print(f"   bbox={bbox} N={n_total} bands={bands}")

		# Clip the rectangular DEM sample to the real basin when we know the HUC.
		# Without a boundary the sample is the raw bbox, so columns can land
		# OUTSIDE the basin — the run still completes, but the ensemble no
		# longer represents the watershed that was asked about. That is a
		# scientific difference, so say so loudly rather than degrade silently.
		boundary = None
		huc = (brief.get("domain") or {}).get("huc")
		if huc:
			try:
				b = clients["terrain"].call_tool_json(
					"get_watershed_boundary",
					{"huc": huc, "huc_level": len(str(huc))}) or {}
				boundary = b.get("rings")
			except Exception as e:
				print(f"   ⚠️  boundary lookup FAILED for HUC {huc} ({e})")
		if not boundary:
			why = ("no HUC in the reception brief"
				   if not huc else f"boundary lookup returned nothing for HUC {huc}")
			print(f"   ⚠️  WARNING: sampling the RAW BBOX, not the watershed "
				  f"— {why}.")
			print(f"   ⚠️  Columns may fall outside the basin; the ensemble is "
				  f"a bounding-box sample, not '{(brief.get('domain') or {}).get('name') or 'the watershed'}'.")
			print(f"   ⚠️  Fix: ensure reception resolves a HUC, or pass "
				  f"config['boundary'] explicitly.")
		boundary = config.get("boundary", boundary)

		res = exp.expand(clients, bbox, n_total, bands, boundary=boundary)
		if res.get("error"):
			raise RuntimeError(f"Sampling expansion failed: {res['error']}")
		columns = res.get("columns", [])
		if not columns:
			raise RuntimeError("Sampling expansion produced no columns.")

		# Keep the WBD polygon in the result: plot_columns() draws it as the
		# basin outline in panel (a), and it makes columns.json self-contained
		# (grid + boundary) so a later re-plot needs no MCP fetch.
		if boundary:
			res["boundary"] = boundary

		# Provenance: was this a true watershed sample or a bbox fallback?
		res["sampling_domain"] = {
			"clipped_to_watershed": bool(boundary),
			"huc": huc,
			"name": (brief.get("domain") or {}).get("name"),
			"bbox": bbox,
			"caveat": None if boundary else (
				"Columns were sampled from the bounding box, NOT clipped to the "
				"watershed boundary (no HUC resolved). Some columns may lie "
				"outside the basin; treat the ensemble as a bbox sample."),
		}

		# Persist the materialized columns next to the run's other inputs, so
		# analyze_run.py / make_warmstart.py can consume this run like any other.
		(self.input_dir / "columns.json").write_text(json.dumps(res, indent=2))
		(self.run_dir  / "columns.json").write_text(json.dumps(res, indent=2))

		yr_start = int(config.get("yr_start", 1995))
		yr_end   = int(config.get("yr_end",   yr_start))

		# The real sampling-design figure (domain map + watershed outline,
		# hypsometry with band edges, SSURGO soil configs, Fan WTD vs
		# elevation, per-band allocation, NLDAS precip gradient). Same
		# function tools/expand_sampling.py --plot uses; passing
		# forcing_year populates the precip-vs-elevation panel that is
		# otherwise blank.
		try:
			png = exp.plot_columns(
				res, str(self.run_dir / "sampling_design.png"),
				forcing_year = yr_start)
			print(f"✓ sampling design → {Path(png).name}")
		except Exception as e:
			print(f"   ⚠️  sampling_design.png failed ({e}) — non-fatal")
		executable = columns_to_elm_plan(
			columns,
			yr_start    = yr_start,
			yr_end      = yr_end,
			soil_config = config.get("soil_config", "native"),
			substrate   = config.get("substrate",   "extrapolate"),
		)

		merged = {**plan, **executable}
		(self.run_dir / "run_plan.json").write_text(json.dumps(merged, indent=2))
		# Make the run dir self-describing, with the SAME filenames the
		# standalone tools expect. validate_run.build_validation() needs
		# reception_brief.json (domain bbox) and interpret_run.interpret()
		# needs plan.json (goals + feasibility verdict). Writing them here
		# means an integrated run is consumable by every existing tool.
		(self.run_dir / "reception_brief.json").write_text(json.dumps(brief, indent=2))
		(self.run_dir / "plan.json").write_text(json.dumps(plan, indent=2))
		print(f"✓ {len(columns)} column(s) materialized "
			  f"({yr_start}-{yr_end}) → CONDITIONS_COUPLERS")
		return merged

	# ─────────────────────────────────────────────────────────
	# STEP 1 — BUILD (writes to 01_inputs/)
	# ─────────────────────────────────────────────────────────
	def _build(self,
			   plan:   Dict[str, Any],
			   config: Dict[str, Any]) -> List[Dict]:
		"""Build the ELMAgentAdapter list from the plan (per-column surfaces)."""
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
	def _analyze(self,
				 experiments,
				 skip_plotting: bool = False,
				 plan:          Dict[str, Any] = None,
				 config:        Dict[str, Any] = None) -> ELMResultsAnalyzer:
		"""Extract variables from ELM history files into 04_analysis/.

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
				),
				"assumptions_ledger": (
					json.loads((self.run_dir / "assumptions.json").read_text())
					if (self.run_dir / "assumptions.json").exists() else []),
			}
		except Exception as e:
			print(f"   ⚠️  limitations payload unavailable ({e})")

		analyzer.extract_all()

		if not skip_plotting:
			self._plot_analysis(analyzer)

		return analyzer

	def _plot_analysis(self, analyzer: ELMResultsAnalyzer) -> None:
		"""04_analysis/ figures — the same five tools/analyze_run.py --plot draws.

		These are the science figures: elevation_gradient, soil_control,
		water_budget, driver_response, wtd_columns. They read the ensemble, so
		they answer questions about the ensemble.

		They replaced ELMResultsAnalyzer.plot_all(), which drew a 2-panel
		<case>_overview.png per column plus one comparison figure. That set had
		no elevation gradient, no soil attribution, no water-budget closure and
		no WTD panel — everything the study is actually about — and it meant an
		integrated run produced a strictly weaker figure set than the same run
		analysed from the command line. Non-fatal.
		"""
		try:
			ar      = _load_tool("analyze_run")
			spatial = analyzer._compute_spatial_summary()
			soil    = analyzer._compute_soil_attribution()

			basin = "Spatial ensemble"
			bf    = self.run_dir / "reception_brief.json"
			if bf.exists():
				basin = ((json.loads(bf.read_text()).get("domain") or {})
						 .get("name") or basin)

			n = 0
			if spatial:
				ar.plot_gradient(spatial, self.analysis_dir / "elevation_gradient.png",
								 basin=basin); n += 1
			if soil:
				ar.plot_soil(soil, self.analysis_dir / "soil_control.png"); n += 1
			ar.plot_budget(analyzer.results, self.analysis_dir / "water_budget.png")
			ar.plot_relations(analyzer.results, self.analysis_dir / "driver_response.png")
			ar.plot_wtd(analyzer.results, self.run_dir,
						self.analysis_dir / "wtd_columns.png")
			print(f"✓ {n + 3} analysis figure(s) → 04_analysis/")
		except Exception as e:
			print(f"   ⚠️  analysis plots failed: {e}")

	# ─────────────────────────────────────────────────────────
	# STEP 4b — VALIDATE AGAINST OBSERVATIONS
	# ─────────────────────────────────────────────────────────
	def _validate(self, config: Dict[str, Any]) -> bool:
		"""Compare the run against in-domain observations (USGS wells and
		gauges, SNOTEL SWE) -> 04_analysis/validation.json + validation.png.

		Reuses tools/validate_run.build_validation() rather than
		reimplementing it. Non-fatal: needs live MCP + network, and a run is
		still useful without it.
		"""
		clients = (config or {}).get("mcp_clients") or {}
		if not clients:
			print("   ⚠️  no MCP clients — skipping observation validation")
			return False
		try:
			vr = _load_tool("validate_run")
			val = vr.build_validation(self.run_dir, clients)
			(self.analysis_dir / "validation.json").write_text(
				json.dumps(val, indent=2, default=str))
			try:
				vr.plot_validation(val, self.analysis_dir / "validation.png")
			except Exception as e:
				print(f"   ⚠️  validation plot failed: {e}")
			n = sum(1 for t in (val.get("targets") or [])
					if t.get("status") == "compared")
			print(f"✓ validation: {n} target(s) compared → 04_analysis/validation.json")
			return True
		except Exception as e:
			print(f"   ⚠️  observation validation failed ({e}) — continuing")
			return False

	# ─────────────────────────────────────────────────────────
	# STEP 4c — INTERPRET (numbers + feasibility + validation)
	# ─────────────────────────────────────────────────────────
	def _interpret(self, config: Dict[str, Any]) -> bool:
		"""LLM interpretation grounded in the computed numbers, the plan's
		feasibility verdict and the observation validation
		-> 04_analysis/interpretation.md. Non-fatal."""
		try:
			ir = _load_tool("interpret_run")
			ir.interpret(self.run_dir,
						 model = (config or {}).get(
							 "interpreter_model", "claude-opus-4-8-project"),
						 quiet = True)
			print("✓ interpretation → 04_analysis/interpretation.md")
			return True
		except Exception as e:
			print(f"   ⚠️  interpretation failed ({e}) — continuing")
			return False

	# ─────────────────────────────────────────────────────────
	# STEP 5 — PACKAGE LLM INPUT (top level)
	# ─────────────────────────────────────────────────────────
	def _save_llm_input(self,
						plan:     Dict[str, Any],
						analyzer: ELMResultsAnalyzer) -> None:
		"""Save LLM_ANALYSIS_INPUT.json at the top level of run_dir."""
		llm_input = analyzer.get_llm_analysis_input()
		llm_input['experiment_plan'] = plan
		llm_input['run_directory']   = str(self.run_dir)

		# get_llm_analysis_input() omits extra_summary, so the honesty payload
		# and the observation verdicts never reached the report agent. Attach
		# them here so the written report can be held to the same standard as
		# 04_analysis/interpretation.md.
		if getattr(analyzer, 'extra_summary', None):
			llm_input['limitations'] = analyzer.extra_summary.get('limitations')
			llm_input['assumptions_ledger'] = \
				analyzer.extra_summary.get('assumptions_ledger')
		vp = self.analysis_dir / "validation.json"
		if vp.exists():
			try:
				val = json.loads(vp.read_text())
				llm_input['validation'] = {
					"targets": val.get("targets"),
					"observation_inventory": val.get("observation_inventory"),
					"hydrograph_metrics": {
						k: v for k, v in (val.get("hydrograph") or {}).items()
						if k not in ("days", "obs", "mod")},
				}
			except Exception as e:
				print(f"   ⚠️  could not attach validation to LLM input: {e}")
		ip = self.analysis_dir / "interpretation.md"
		if ip.exists():
			llm_input['interpretation_md'] = ip.read_text()

		llm_file = self.run_dir / "LLM_ANALYSIS_INPUT.json"
		with open(llm_file, 'w') as f:
			json.dump(llm_input, f, indent=2, default=str)
		print(f"✓ LLM_ANALYSIS_INPUT.json saved")

	# ─────────────────────────────────────────────────────────
	# RUN SUMMARY
	# ─────────────────────────────────────────────────────────
	def _create_run_summary(self,
							plan:        Dict[str, Any],
							experiments: List[Dict],
							results:     Dict[str, bool],
							start_time:  datetime,
							end_time:    datetime
							) -> Dict[str, Any]:
		n_total   = len(experiments)
		n_success = sum(results.values())

		exp_details = [
			{
				'name':              e['scenario_name'],
				'case_name':         e['case_name'],
				'status':            'completed'
									 if results.get(e['case_name'])
									 else 'failed',
				'forcing_period':    e['forcing_period'],
				'forcing_start':     e['forcing_start'],
				'forcing_end':       e['forcing_end'],
				'model_type':        'elm',
				'runtime_seconds':   0,
				'timesteps':         0,
				'newton_iterations': 0,
			}
			for e in experiments
		]

		return {
			'run_directory':         str(self.run_dir),
			'start_time':            start_time.isoformat(),
			'end_time':              end_time.isoformat(),
			'total_runtime_seconds': (
				end_time - start_time
			).total_seconds(),
			'experiments_total':     n_total,
			'experiments_success':   n_success,
			'experiments_failed':    n_total - n_success,
			'experiments':           exp_details,
			'convergence_warnings':  [],
			'output_files': {
				'inputs':             str(self.input_dir),
				'setup_plots':        str(self.setup_plots_dir),
				'results':            str(self.results_dir),
				'analysis':           str(self.analysis_dir),
				'experiment_summary': str(
					self.input_dir / "experiment_summary.json"),
				'execution_report':   str(
					self.results_dir / "execution_report.txt"),
				'results_csv':        str(
					self.results_dir / "results_summary.csv"),
				'hydro_summary':      str(
					self.analysis_dir / "hydro_summary.json"),
				'llm_input':          str(
					self.run_dir / "LLM_ANALYSIS_INPUT.json"),
			},
			'model_type': 'elm',
		}

	def _save_run_summary(self,
						  run_summary: Dict[str, Any]) -> None:
		summary_file = self.run_dir / "RUN_SUMMARY.json"
		with open(summary_file, 'w') as f:
			json.dump(run_summary, f, indent=2, default=str)
		print(f"✓ RUN_SUMMARY.json saved")

	def _save_error(self,
					error:      Exception,
					start_time: datetime) -> None:
		import traceback
		error_log = {
			'run_directory': str(self.run_dir),
			'start_time':    start_time.isoformat(),
			'end_time':      datetime.now().isoformat(),
			'status':        'failed',
			'error':         str(error),
			'traceback':     traceback.format_exc(),
			'model_type':    'elm',
		}
		error_file = self.run_dir / "ERROR_LOG.json"
		with open(error_file, 'w') as f:
			json.dump(error_log, f, indent=2)


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("ELMExpManager — call via workflow.py or test directly:")
    print("  from core.elm_exp_manager import ELMExpManager")
    print("  mgr = ELMExpManager()")
    print("  run_summary = mgr.execute_plan(plan, {})")