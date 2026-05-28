#!/usr/bin/env python3
"""
ELM Experiment Manager
src/core/elm_exp_manager.py

Single responsibility: orchestrate the ELM execution pipeline.
Mirrors PFLOTRAN's ExpManager interface AND directory layout so
workflow.py and downstream tools (e.g. create_slides.py) work
with both PFLOTRAN and ELM unchanged.

Output directory structure (mirrors PFLOTRAN exactly):

    elm_run_YYYYMMDD_HHMMSS/
        ├── 01_inputs/
        │   └── experiment_summary.json
        ├── 02_setup_plots/
        │   ├── exp_001_<case_name>/
        │   │   └── domain_configuration.png
        │   ├── exp_NNN_<case_name>/
        │   │   └── domain_configuration.png
        │   ├── comparison_experiments.png
        │   └── comparison_forcing_conditions.png
        ├── 03_results/
        │   ├── execution_report.txt
        │   └── results_summary.csv
        ├── 04_analysis/
        │   ├── hydro_summary.json
        │   ├── comparison_all_times.png
        │   └── <case_name>_*.png
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


# ─────────────────────────────────────────────────────────────────────
# ELM EXPERIMENT MANAGER
# ─────────────────────────────────────────────────────────────────────
class ELMExpManager:
	"""
	Executes ELM experiment plans.

	Mirrors PFLOTRAN ExpManager.execute_plan() interface AND output
	directory structure, so workflow.py and create_slides.py work
	identically for both model types.
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
			# Step 1 — Build + setup plots → 01_inputs/, 02_setup_plots/
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
			analyzer = self._analyze(experiments)

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
	# STEP 1 — BUILD (writes to 01_inputs/ + 02_setup_plots/)
	# ─────────────────────────────────────────────────────────
	def _build(self,
			   plan:   Dict[str, Any],
			   config: Dict[str, Any]) -> List[Dict]:
		"""Build ELMAgentAdapter list from plan + generate setup plots."""
		builder     = ELMExperimentBuilder(plan)
		experiments = builder.build_experiments()

		# Save experiment_summary.json to 01_inputs/
		summary_file = self.input_dir / "experiment_summary.json"
		with open(summary_file, 'w') as f:
			json.dump(
				builder.get_experiment_summary(),
				f, indent=2, default=str
			)

		# Phase B: generate setup plots → 02_setup_plots/
		self._plot_setups(experiments, plan)

		print(f"✓ {len(experiments)} experiment(s) built")
		return experiments

	def _plot_setups(self,
					 experiments: List[Dict],
					 plan:        Dict[str, Any]) -> None:
		"""
		Generate setup-time plots into 02_setup_plots/.

		Per-experiment: 02_setup_plots/exp_NNN_<case_name>/domain_configuration.png
		Cross-experiment: 02_setup_plots/comparison_{experiments,forcing_conditions}.png

		Failures here are non-fatal — pipeline continues.
		"""
		if not experiments:
			return

		try:
			from core.elm_setup_plotting import (
				plot_domain_configuration,
				compare_experiments,
				compare_forcing_conditions,
			)
		except ImportError as e:
			print(f"   ⚠️  elm_setup_plotting unavailable — "
				  f"skipping setup plots ({e})")
			return

		elm_config = plan.get('ELM_CONFIG', {}) if plan else {}
		n_ok = 0

		# Per-experiment plots
		for i, exp in enumerate(experiments, 1):
			case_name = exp.get('case_name', f'exp_{i:03d}')
			exp_dir   = self.setup_plots_dir / f"exp_{i:03d}_{case_name}"
			exp_dir.mkdir(exist_ok=True)
			try:
				ok = plot_domain_configuration(
					experiment  = exp,
					elm_config  = elm_config,
					output_path = str(exp_dir / "domain_configuration.png"),
				)
				if ok:
					n_ok += 1
			except Exception as e:
				print(f"   ⚠️  domain plot failed for {case_name}: {e}")

		# Cross-experiment comparisons (only useful with 2+ experiments)
		if len(experiments) >= 2:
			try:
				ok = compare_experiments(
					experiments,
					output_path = str(
						self.setup_plots_dir / "comparison_experiments.png"),
				)
				if ok:
					n_ok += 1
			except Exception as e:
				print(f"   ⚠️  compare_experiments failed: {e}")

			try:
				ok = compare_forcing_conditions(
					experiments,
					output_path = str(
						self.setup_plots_dir /
						"comparison_forcing_conditions.png"),
				)
				if ok:
					n_ok += 1
			except Exception as e:
				print(f"   ⚠️  compare_forcing_conditions failed: {e}")

		print(f"✓ {n_ok} setup plot(s) → 02_setup_plots/")

	# ─────────────────────────────────────────────────────────
	# STEP 2 — PREPARE (cases live at $PSCRATCH)
	# ─────────────────────────────────────────────────────────
	def _prepare(self, experiments: List[Dict]) -> None:
		"""Prepare (build) all ELM cases."""
		for exp in experiments:
			case_dir = exp['elm_agent'].prepare_case(
				output_dir = str(self.run_dir)
			)
			exp['case_dir'] = case_dir

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
		results = {}

		for exp in experiments:
			case_name = exp['case_name']
			success   = exp['elm_agent'].run_simulation()
			results[case_name] = success
			exp['run_summary'] = exp['elm_agent'].get_run_summary()

		n_ok   = sum(results.values())
		n_fail = len(results) - n_ok
		print(f"✓ {n_ok} succeeded, {n_fail} failed")

		# Post-run reporting → 03_results/
		self._write_execution_report(experiments, results)
		self._write_results_csv(experiments, results)

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
				 skip_plotting: bool = False) -> ELMResultsAnalyzer:
		"""Extract variables from ELM history files into 04_analysis/."""
		analyzer = ELMResultsAnalyzer(
			experiments  = experiments,
			analysis_dir = str(self.analysis_dir),
		)
		analyzer.extract_all()

		if not skip_plotting and hasattr(analyzer, 'plot_all'):
			analyzer.plot_all()

		return analyzer

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