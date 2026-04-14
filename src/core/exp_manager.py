#!/usr/bin/env python3
"""
PFLOTRAN Experiment Manager
Pure execution engine: Takes experiment plan → Executes → Returns run summary
No planning logic, no agent coordination - just execution
"""
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add src directories to path
sys.path.insert(0, "src")
sys.path.insert(0, "src/core")

from core.experiment_builder import ExperimentBuilder
from core.experiment_manager import ExperimentManager
from core.pflotran_input_agent import PFLOTRANInputAgent
from core.results_analyzer import ResultsAnalyzer
from core.pflotran_plotting import (
    compare_experiments,
    compare_flux_conditions,
    plot_all_experiment_figures
)


class ExpManager:
	"""
	Executes PFLOTRAN experiment plans
	Input: Experiment plan (from Planner Agent)
	Output: Run summary (for Analysis Agent)
	"""
	
	def __init__(self, base_output_dir: str = "./workflow_outputs"):
		"""Initialize executor with output directory"""
		self.base_output_dir = Path(base_output_dir)
		
		# Create timestamped run directory
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		self.run_dir = self.base_output_dir / f"run_{timestamp}"
		self.run_dir.mkdir(parents=True, exist_ok=True)
		
		# Define subdirectories
		self.input_dir = self.run_dir / "01_inputs"
		self.setup_plots_dir = self.run_dir / "02_setup_plots"
		self.results_dir = self.run_dir / "03_results"
		self.analysis_dir = self.run_dir / "04_analysis"
		
		for d in [self.input_dir, self.setup_plots_dir, self.results_dir, self.analysis_dir]:
			d.mkdir(exist_ok=True)
		
		#print(f"{'='*70}")
		#print(f"PFLOTRAN Manager - Run Directory: {self.run_dir.name}")
		#print(f"{'='*70}\n")
	
	# ═════════════════════════════════════════════════════════════════════════
	# MAIN ENTRY POINT
	# ═════════════════════════════════════════════════════════════════════════
	
	def execute_plan(self, experiment_plan: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Execute complete experiment plan
		
		Args:
			experiment_plan: Plan from Planner Agent
			config: Execution configuration {
				'pflotran_exe': path to executable,
				'time_indices': list of time indices for analysis,
				'skip_plotting': bool
			}
		
		Returns:
			run_summary: {
				'run_directory': path,
				'experiments_total': int,
				'experiments_success': int,
				'experiments_failed': int,
				'total_runtime_seconds': float,
				'experiments': [experiment details],
				'output_files': file locations,
				'convergence_warnings': list
			}
		"""
		start_time = datetime.now()
		
		try:
			# Step 1: Build experiments
			builder, experiments = self._build_experiments(experiment_plan)
			
			# Step 2: Plot setups (optional)
			plot_objects = None
			if not config.get('skip_plotting', False):
				plot_objects = self._plot_experiment_setups(experiments)
			
			# Step 3: Run experiments
			manager = self._run_experiments(experiments, config['pflotran_exe'])
			
			# Step 4: Analyze results
			time_indices = self._get_time_indices_from_plan(experiment_plan)
			analyzer = self._analyze_results(manager,time_indices)
			
			# Step 5: Prepare LLM analysis input
			self._prepare_llm_analysis_input(experiment_plan, manager, analyzer)
			
			# Create run summary
			end_time = datetime.now()
			run_summary = self._create_run_summary(
				experiment_plan, experiments, manager, analyzer, 
				start_time, end_time
			)
			
			# Save run summary
			summary_file = self.run_dir / "RUN_SUMMARY.json"
			with open(summary_file, 'w') as f:
				json.dump(run_summary, f, indent=2, default=str)
			
			#print(f"\n{'='*70}")
			#print(f"EXECUTION COMPLETE")
			#print(f"{'='*70}")
			#print(f"✓ Success: {run_summary['experiments_success']}/{run_summary['experiments_total']}")
			#print(f"✓ Runtime: {run_summary['total_runtime_seconds']:.1f} seconds")
			#print(f"✓ Output: {self.run_dir}")
			#print(f"{'='*70}\n")
			
			return run_summary
			
		except Exception as e:
			error_summary = self._create_error_summary(e, start_time)
			error_file = self.run_dir / "ERROR_LOG.json"
			with open(error_file, 'w') as f:
				json.dump(error_summary, f, indent=2, default=str)
			raise
	
	# ═════════════════════════════════════════════════════════════════════════
	# EXECUTION STEPS
	# ═════════════════════════════════════════════════════════════════════════
	
	def _build_experiments(self, experiment_plan: Dict[str, Any]):
		"""Build experiment configurations from plan"""
		#print(f"{'='*70}")
		#print("STEP 1: Building Experiments")
		#print(f"{'='*70}")
		
		builder = ExperimentBuilder(experiment_plan)
		experiments = builder.build_experiments(PFLOTRANInputAgent)
		builder.prepare_cases(str(self.input_dir))
		
		input_files = list(self.input_dir.glob("**/*.in"))
		#print(f"✓ Built {len(experiments)} experiments")
		#print(f"✓ Generated {len(input_files)} input files\n")
		
		return builder, experiments
	
	def _plot_experiment_setups(self, experiments: List):
		"""Generate plots of experiment setups"""
		#print(f"{'='*70}")
		#print("STEP 2: Plotting Experiment Setups")
		#print(f"{'='*70}")
		
		plot_objects = []
		
		for i, exp in enumerate(experiments, 1):
			# Extract experiment info
			if isinstance(exp, dict):
				exp_name = exp.get('scenario_name', f'exp_{i:03d}')
				agent = exp.get('pflotran_agent', None)
			else:
				exp_name = getattr(exp, 'scenario_name', f'exp_{i:03d}')
				agent = getattr(exp, 'pflotran_agent', None)
			
			if agent is None:
				continue
			
			try:
				exp_output_dir = self.setup_plots_dir / f"exp_{i:03d}_{exp_name}"
				exp_output_dir.mkdir(exist_ok=True)
				
				figures, plot_obj = plot_all_experiment_figures(
					agent, str(exp_output_dir), prefix=f"{exp_name}_"
				)
				plot_objects.append(plot_obj)
			except Exception:
				pass  # Continue on plotting errors
		
		# Comparison plots
		if len(plot_objects) > 1:
			self._generate_comparison_plots(plot_objects)
		
		#print(f"✓ Generated {len(plot_objects)} setup plots\n")
		return plot_objects
	
	def _run_experiments(self, experiments: List, pflotran_exe: str):
		"""Run PFLOTRAN simulations"""
		#print(f"{'='*70}")
		#print("STEP 3: Running PFLOTRAN Simulations")
		#print(f"{'='*70}")
		#print(f"Executable: {pflotran_exe}")
		#print(f"Experiments: {len(experiments)}\n")
		
		manager = ExperimentManager(experiments)
		manager.run_all_experiments(pflotran_exe=pflotran_exe)
		
		# Generate execution reports
		report_file = self.results_dir / "execution_report.txt"
		manager.generate_report(str(report_file))
		
		df = manager.export_to_dataframe()
		csv_file = self.results_dir / "results_summary.csv"
		df.to_csv(csv_file, index=False)
		
		perf_plot = self.results_dir / "performance_summary.png"
		manager.plot_performance_summary(save_path=str(perf_plot))
		
		#print(f"✓ Execution complete\n")
		return manager
	
	def _analyze_results(self, manager: ExperimentManager, time_indices: List[int]):
		"""Analyze simulation results"""
		#print(f"{'='*70}")
		#print("STEP 4: Analyzing Results")
		#print(f"{'='*70}")
		
		# time_indices already provided from plan, no need to auto-detect!
		#print(f"   Using {len(time_indices)} time snapshots from experiment plan\n")
		
		analyzer = ResultsAnalyzer(manager)
		
		# Generate plots
		figures = analyzer.plot_all_experiments(
			time_indices=time_indices,
			save_dir=str(self.analysis_dir)
		)
		
		# Generate reports
		report_file = self.analysis_dir / "analysis_report.txt"
		analyzer.generate_analysis_report(
			save_path=str(report_file),
			time_indices=time_indices
		)
		
		# Save saturation data
		sat_data_file = self.analysis_dir / "saturation_data.json"
		analyzer.save_saturation_data(
			save_path=str(sat_data_file),
			time_indices=time_indices
		)
		
		#print(f"✓ Generated {len(figures)} analysis plots\n")
		return analyzer
	
	def _prepare_llm_analysis_input(self, 
									experiment_plan: Dict[str, Any],
									manager: ExperimentManager,
									analyzer: ResultsAnalyzer):
		"""Prepare structured input for LLM Analysis Agent"""
		#print(f"{'='*70}")
		#print("STEP 5: Preparing LLM Analysis Input")
		#print(f"{'='*70}")
		
		llm_input = {
			"run_directory": str(self.run_dir),
			"experiment_plan": experiment_plan,
			"execution_summary": self._extract_execution_data(manager),
			"analysis_data": self._extract_analysis_data(analyzer),
			"file_locations": {
				"execution_report": str(self.results_dir / "execution_report.txt"),
				"analysis_report": str(self.analysis_dir / "analysis_report.txt"),
				"saturation_data": str(self.analysis_dir / "saturation_data.json"),
				"results_csv": str(self.results_dir / "results_summary.csv")
			}
		}
		
		# Save LLM input
		llm_input_file = self.run_dir / "LLM_ANALYSIS_INPUT.json"
		with open(llm_input_file, 'w') as f:
			json.dump(llm_input, f, indent=2, default=str)
		
		#print(f"✓ LLM analysis input ready\n")
	
	# ═════════════════════════════════════════════════════════════════════════
	# SUMMARY GENERATION
	# ═════════════════════════════════════════════════════════════════════════
	
	def _create_run_summary(self,
					   experiment_plan: Dict[str, Any],
					   experiments: List,
					   manager: ExperimentManager,
					   analyzer: ResultsAnalyzer,
					   start_time: datetime,
					   end_time: datetime) -> Dict[str, Any]:
		"""Create comprehensive run summary"""
		
		# Get execution status from manager's dataframe
		try:
			df = manager.export_to_dataframe()
			
			# Count successes and failures
			experiments_success = 0
			experiments_failed = 0
			experiment_details = []
			convergence_warnings = []
			
			for _, row in df.iterrows():
				status = row.get('status', 'unknown')
				case_name = row.get('case_name', 'N/A')
				
				# Count status
				if status == 'completed':
					experiments_success += 1
				elif status == 'failed':
					experiments_failed += 1
					convergence_warnings.append({
						'experiment': case_name,
						'issue': 'Simulation failed - check logs'
					})
				
				# Build experiment details
				experiment_details.append({
					'name': case_name,
					'status': status,
					'runtime_seconds': row.get('wall_clock_time', 0),
					'timesteps': row.get('total_steps', 0),
					'newton_iterations': row.get('total_newton_iterations', 0)
				})
			
		except Exception as e:
			#print(f"⚠️  Warning: Could not extract detailed status from manager: {e}")
			# Fallback to basic counts
			experiments_success = len(experiments)
			experiments_failed = 0
			experiment_details = [{'name': f'exp_{i}', 'status': 'unknown'} for i in range(len(experiments))]
			convergence_warnings = []
		
		return {
			'run_directory': str(self.run_dir),
			'start_time': start_time.isoformat(),
			'end_time': end_time.isoformat(),
			'total_runtime_seconds': (end_time - start_time).total_seconds(),
			'experiments_total': len(experiments),
			'experiments_success': experiments_success,
			'experiments_failed': experiments_failed,
			'experiments': experiment_details,
			'convergence_warnings': convergence_warnings,
			'output_files': {
				'inputs': str(self.input_dir),
				'setup_plots': str(self.setup_plots_dir),
				'results': str(self.results_dir),
				'analysis': str(self.analysis_dir),
				'llm_input': str(self.run_dir / "LLM_ANALYSIS_INPUT.json")
			}
		}
	
	def _create_error_summary(self, error: Exception, start_time: datetime) -> Dict[str, Any]:
		"""Create error summary"""
		import traceback
		return {
			'run_directory': str(self.run_dir),
			'start_time': start_time.isoformat(),
			'end_time': datetime.now().isoformat(),
			'status': 'failed',
			'error': str(error),
			'traceback': traceback.format_exc()
		}
	
	# ═════════════════════════════════════════════════════════════════════════
	# UTILITIES
	# ═════════════════════════════════════════════════════════════════════════
	
	def _get_time_indices_from_plan(self, experiment_plan: Dict[str, Any]) -> List[int]:
		"""
		Extract time indices from experiment plan
		This matches what setup plots use, so it's consistent!
		"""
		try:
			output_config = experiment_plan.get('OUTPUT', {})
			output_times = output_config.get('TIMES', [])
			
			if output_times:
				n_times = len(output_times)
				
				# Convert to actual time values for display
				time_values = []
				for t_str in output_times:
					try:
						t_val = self._fortran_to_float(t_str)
						time_values.append(t_val)
					except:
						pass
				
				#print(f"   📊 Using {n_times} output times from plan")
				if time_values:
					#print(f"   ✓ Time range: {time_values[0]:.2f} to {time_values[-1]:.2f} years")
					sample_times = [f"{t:.2f}" for t in time_values[:6]]
					if len(time_values) > 6:
						sample_times.append("...")
					#print(f"   ✓ Times: {', '.join(sample_times)}")
				
				# Return indices 0 to n_times-1
				return list(range(n_times))
			else:
				#print("   ⚠️  No OUTPUT.TIMES in plan, using default")
				return list(range(20))
		
		except Exception as e:
			#print(f"   ⚠️  Could not extract times from plan: {e}")
			return list(range(20))
	
	def _fortran_to_float(self, fortran_str: str) -> float:
		"""Convert Fortran notation to float"""
		if not isinstance(fortran_str, str):
			return float(fortran_str)
		python_str = fortran_str.lower().replace('d', 'e')
		return float(python_str)
	
	def _generate_comparison_plots(self, plot_objects: List):
		"""Generate comparison plots"""
		try:
			import matplotlib.pyplot as plt
			
			comp_fig, _ = compare_experiments(plot_objects)
			if comp_fig is not None:
				comp_file = self.setup_plots_dir / "comparison_experiments.png"
				comp_fig.savefig(comp_file, dpi=300, bbox_inches='tight')
				plt.close(comp_fig)
			
			flux_fig, _ = compare_flux_conditions(plot_objects)
			if flux_fig is not None:
				flux_file = self.setup_plots_dir / "comparison_flux_conditions.png"
				flux_fig.savefig(flux_file, dpi=300, bbox_inches='tight')
				plt.close(flux_fig)
		except Exception:
			pass  # Silently fail on comparison plots
	
	def _extract_execution_data(self, manager: ExperimentManager) -> Dict[str, Any]:
		"""Extract execution data for LLM"""
		df = manager.export_to_dataframe()
		return {
			'summary': df.to_dict('records'),
			'total_experiments': len(df),
			'successful': len(df[df['status'] == 'completed']) if 'status' in df.columns else 0
		}
	
	def _extract_analysis_data(self, analyzer: ResultsAnalyzer)  -> Dict[str, Any]:
		"""Extract analysis data for LLM"""
		# Read saturation data if available
		sat_data_file = self.analysis_dir / "saturation_data.json"
		if sat_data_file.exists():
			with open(sat_data_file, 'r') as f:
				return json.load(f)
		return {}


if __name__ == "__main__":
    print("="*70)
    print("Experiment Manager")
    print("="*70)
    print("\nThis is a pure execution engine.")
    print("It should be called by the Coordinator, not directly.")
    print("\nUsage:")
    print("  from core.exp_manager import ExpManager")
    print("  executor = ExpManager()")
    print("  run_summary = executor.execute_plan(plan, config)")
    print("="*70)