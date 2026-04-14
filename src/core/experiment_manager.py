#!/usr/bin/env python3
"""
Experiment Manager - Manages execution and tracking of PFLOTRAN simulations
"""
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd


class ExperimentManager:
	"""
	Manages execution and monitoring of PFLOTRAN experiments.
	Runs simulations serially and collects runtime statistics.
	"""
	
	def __init__(self, experiments: List[Dict[str, Any]]):
		"""
		Initialize ExperimentManager.
		
		Args:
			experiments: List of experiment dictionaries from ExperimentBuilder
		"""
		self.experiments = experiments
		self.results = {}
		self.status = {}
		
		# Initialize status for all experiments
		for exp in self.experiments:
			case_name = exp['case_name']
			self.status[case_name] = 'pending'
			self.results[case_name] = {
				'status': 'pending',
				'start_time': None,
				'end_time': None,
				'stats': None,
				'error_message': None
			}
	
	def run_all_experiments(self, pflotran_exe: str = "pflotran", 
						   timeout: Optional[int] = None):
		"""
		Run all experiments serially.
		
		Args:
			pflotran_exe: Path to PFLOTRAN executable
			timeout: Maximum runtime per simulation in seconds (optional)
		
		Returns:
			Summary dictionary with results
		"""
		print("\n" + "="*70)
		print("RUNNING ALL EXPERIMENTS")
		print("="*70)
		
		total_experiments = len(self.experiments)
		
		for idx, exp in enumerate(self.experiments, 1):
			case_name = exp['case_name']
			agent = exp['pflotran_agent']
			
			print(f"\n[{idx}/{total_experiments}] Running: {exp['scenario_name']}")
			print(f"    Case: {case_name}")
			
			self.run_experiment(agent, pflotran_exe, timeout)
		
		print("\n" + "="*70)
		print("ALL EXPERIMENTS COMPLETED")
		print("="*70)
		
		# Print summary
		self._print_summary()
		
		return self.get_summary()
	
	def run_experiment(self, agent, pflotran_exe: str = "pflotran", 
					  timeout: Optional[int] = None):
		"""
		Run a single experiment.
		
		Args:
			agent: PFLOTRANInputAgent instance
			pflotran_exe: Path to PFLOTRAN executable
			timeout: Maximum runtime in seconds (optional)
		
		Returns:
			Dictionary with execution results
		"""
		case_name = agent.case_name
		
		# Update status
		self.status[case_name] = 'running'
		self.results[case_name]['status'] = 'running'
		self.results[case_name]['start_time'] = datetime.now()
		
		try:
			# Run simulation
			print(f"    Status: Running...")
			start_time = time.time()
			
			result = agent.run_simulation(pflotran_exe=pflotran_exe, timeout=timeout)
			
			elapsed_time = time.time() - start_time
			
			# Check if successful
			if result.returncode == 0:
				self.status[case_name] = 'completed'
				self.results[case_name]['status'] = 'completed'
				print(f"    Status: ✓ Completed ({elapsed_time:.2f}s)")
				
				# Parse output file
				out_file = os.path.join(agent.case_dir, f"{case_name}.out")
				if os.path.exists(out_file):
					stats = self.parse_output_file(out_file)
					self.results[case_name]['stats'] = stats
					print(f"    Stats: {stats['total_steps']} steps, "
						  f"{stats['total_newton_iterations']} Newton iters, "
						  f"{stats['simulation_time']:.3f}s")
				else:
					print(f"    Warning: Output file not found: {out_file}")
			else:
				self.status[case_name] = 'failed'
				self.results[case_name]['status'] = 'failed'
				self.results[case_name]['error_message'] = f"Return code: {result.returncode}"
				print(f"    Status: ✗ Failed (return code: {result.returncode})")
				
		except Exception as e:
			self.status[case_name] = 'failed'
			self.results[case_name]['status'] = 'failed'
			self.results[case_name]['error_message'] = str(e)
			print(f"    Status: ✗ Failed ({str(e)})")
		
		finally:
			self.results[case_name]['end_time'] = datetime.now()
		
		return self.results[case_name]
	
	def parse_output_file(self, out_file_path: str) -> Dict[str, Any]:
		"""
		Parse PFLOTRAN .out file and extract statistics.
		
		Args:
			out_file_path: Path to .out file
		
		Returns:
			Dictionary with extracted statistics
		"""
		stats = {
			'n_processes': None,
			'n_grid_cells': None,
			'grid_extent': {},
			'total_steps': None,
			'total_cuts': None,
			'total_newton_iterations': None,
			'total_linear_iterations': None,
			'wasted_linear_iterations': None,
			'wasted_newton_iterations': None,
			'avg_newton_per_step': None,
			'final_time': None,
			'simulation_time': None,
			'wall_clock_time': None,
			'timestep_stats': {},
			'convergence_reasons': []
		}
		
		with open(out_file_path, 'r') as f:
			content = f.read()
		
		# Extract number of processes
		match = re.search(r'Number of processes:\s+(\d+)', content)
		if match:
			stats['n_processes'] = int(match.group(1))
		
		# Extract number of grid cells
		match = re.search(r'Number of grid cells:\s+(\d+)', content)
		if match:
			stats['n_grid_cells'] = int(match.group(1))
		
		# Extract grid extent
		x_match = re.search(r'X:\s+([\d.E+-]+)\s+-\s+([\d.E+-]+)', content)
		y_match = re.search(r'Y:\s+([\d.E+-]+)\s+-\s+([\d.E+-]+)', content)
		z_match = re.search(r'Z:\s+([\d.E+-]+)\s+-\s+([\d.E+-]+)', content)
		
		if x_match and y_match and z_match:
			stats['grid_extent'] = {
				'x_min': float(x_match.group(1)),
				'x_max': float(x_match.group(2)),
				'y_min': float(y_match.group(1)),
				'y_max': float(y_match.group(2)),
				'z_min': float(z_match.group(1)),
				'z_max': float(z_match.group(2))
			}
		
		# Extract time stepping summary (at the end)
		# Example: "FLOW TS SNES steps = 263  newton = 706  linear = 706  cuts = 3"
		match = re.search(
			r'FLOW TS SNES steps\s*=\s*(\d+)\s+newton\s*=\s*(\d+)\s+linear\s*=\s*(\d+)\s+cuts\s*=\s*(\d+)',
			content
		)
		if match:
			stats['total_steps'] = int(match.group(1))
			stats['total_newton_iterations'] = int(match.group(2))
			stats['total_linear_iterations'] = int(match.group(3))
			stats['total_cuts'] = int(match.group(4))
			
			# Calculate average
			if stats['total_steps'] > 0:
				stats['avg_newton_per_step'] = stats['total_newton_iterations'] / stats['total_steps']
		
		# Extract wasted iterations
		match = re.search(r'FLOW TS SNES Wasted Linear Iterations\s*=\s*(\d+)', content)
		if match:
			stats['wasted_linear_iterations'] = int(match.group(1))
		
		match = re.search(r'FLOW TS SNES Wasted Newton Iterations\s*=\s*(\d+)', content)
		if match:
			stats['wasted_newton_iterations'] = int(match.group(1))
		
		# Extract final time
		# Example: "Step    263 Time=  1.00000E+01"
		time_matches = re.findall(r'Step\s+\d+\s+Time=\s+([\d.E+-]+)', content)
		if time_matches:
			stats['final_time'] = float(time_matches[-1])
		
		# Extract simulation time
		# Example: "Total Time: 7.9293E-02 seconds"
		match = re.search(r'Total Time:\s+([\d.E+-]+)\s+seconds', content)
		if match:
			stats['simulation_time'] = float(match.group(1))
		
		# Extract wall clock time
		# Example: "Wall Clock Time:  9.6487E-02 [sec]"
		match = re.search(r'Wall Clock Time:\s+([\d.E+-]+)\s+\[sec\]', content)
		if match:
			stats['wall_clock_time'] = float(match.group(1))
		
		# Extract timestep statistics
		dt_values = re.findall(r'Dt=\s+([\d.E+-]+)', content)
		if dt_values:
			dt_floats = [float(dt) for dt in dt_values]
			stats['timestep_stats'] = {
				'min': min(dt_floats),
				'max': max(dt_floats),
				'avg': sum(dt_floats) / len(dt_floats),
				'count': len(dt_floats)
			}
		
		# Extract convergence reasons
		conv_reasons = re.findall(r'conv_reason:\s*(\d+)', content)
		if conv_reasons:
			stats['convergence_reasons'] = [int(cr) for cr in conv_reasons]
		
		return stats
	
	def get_runtime_stats(self, case_name: str) -> Optional[Dict[str, Any]]:
		"""
		Get runtime statistics for a specific experiment.
		
		Args:
			case_name: Name of the experiment case
		
		Returns:
			Dictionary with statistics or None if not found
		"""
		if case_name in self.results:
			return self.results[case_name].get('stats')
		return None
	
	def get_summary(self) -> Dict[str, Any]:
		"""
		Get summary of all experiments.
		
		Returns:
			Dictionary with summary information
		"""
		summary = {
			'total_experiments': len(self.experiments),
			'completed': sum(1 for s in self.status.values() if s == 'completed'),
			'failed': sum(1 for s in self.status.values() if s == 'failed'),
			'pending': sum(1 for s in self.status.values() if s == 'pending'),
			'experiments': []
		}
		
		for exp in self.experiments:
			case_name = exp['case_name']
			result = self.results[case_name]
			
			exp_summary = {
				'case_name': case_name,
				'scenario_name': exp['scenario_name'],
				'status': result['status'],
				'start_time': result['start_time'],
				'end_time': result['end_time']
			}
			
			if result['stats']:
				exp_summary.update({
					'total_steps': result['stats']['total_steps'],
					'total_newton_iterations': result['stats']['total_newton_iterations'],
					'total_cuts': result['stats']['total_cuts'],
					'simulation_time': result['stats']['simulation_time'],
					'wall_clock_time': result['stats']['wall_clock_time'],
					'n_processes': result['stats']['n_processes']
				})
			
			if result['error_message']:
				exp_summary['error_message'] = result['error_message']
			
			summary['experiments'].append(exp_summary)
		
		return summary
	
	def get_failed_experiments(self) -> List[str]:
		"""
		Get list of failed experiment names.
		
		Returns:
			List of case names that failed
		"""
		return [name for name, status in self.status.items() if status == 'failed']
	
	def retry_failed(self, pflotran_exe: str = "pflotran", 
					timeout: Optional[int] = None):
		"""
		Retry all failed experiments.
		
		Args:
			pflotran_exe: Path to PFLOTRAN executable
			timeout: Maximum runtime per simulation in seconds
		
		Returns:
			Summary dictionary
		"""
		failed_cases = self.get_failed_experiments()
		
		if not failed_cases:
			print("No failed experiments to retry.")
			return self.get_summary()
		
		print(f"\nRetrying {len(failed_cases)} failed experiments...")
		
		for case_name in failed_cases:
			# Find the experiment
			exp = next((e for e in self.experiments if e['case_name'] == case_name), None)
			if exp:
				agent = exp['pflotran_agent']
				print(f"\nRetrying: {case_name}")
				self.run_experiment(agent, pflotran_exe, timeout)
		
		return self.get_summary()
	
	def _print_summary(self):
		"""Print summary to console."""
		summary = self.get_summary()
		
		print(f"\nSummary:")
		print(f"  Total experiments: {summary['total_experiments']}")
		print(f"  Completed: {summary['completed']}")
		print(f"  Failed: {summary['failed']}")
		print(f"  Pending: {summary['pending']}")
		
		if summary['failed'] > 0:
			print(f"\nFailed experiments:")
			for exp in summary['experiments']:
				if exp['status'] == 'failed':
					print(f"  - {exp['case_name']}: {exp.get('error_message', 'Unknown error')}")
	
	def generate_report(self, save_path: str):
		"""
		Generate detailed text report.
		
		Args:
			save_path: Path to save report
		"""
		summary = self.get_summary()
		
		with open(save_path, 'w') as f:
			f.write("="*80 + "\n")
			f.write("PFLOTRAN EXPERIMENT MANAGER - SIMULATION REPORT\n")
			f.write("="*80 + "\n\n")
			
			f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
			
			f.write("SUMMARY\n")
			f.write("-"*80 + "\n")
			f.write(f"Total experiments:  {summary['total_experiments']}\n")
			f.write(f"Completed:          {summary['completed']}\n")
			f.write(f"Failed:             {summary['failed']}\n")
			f.write(f"Pending:            {summary['pending']}\n\n")
			
			f.write("DETAILED RESULTS\n")
			f.write("="*80 + "\n\n")
			
			for idx, exp in enumerate(summary['experiments'], 1):
				f.write(f"{idx}. {exp['scenario_name']}\n")
				f.write(f"   Case name: {exp['case_name']}\n")
				f.write(f"   Status: {exp['status'].upper()}\n")
				
				if exp.get('start_time'):
					f.write(f"   Start time: {exp['start_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
				if exp.get('end_time'):
					f.write(f"   End time: {exp['end_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
				
				if exp['status'] == 'completed':
					f.write(f"\n   Performance Metrics:\n")
					f.write(f"   - CPU cores used:        {exp.get('n_processes', 'N/A')}\n")
					f.write(f"   - Total timesteps:       {exp.get('total_steps', 'N/A')}\n")
					f.write(f"   - Newton iterations:     {exp.get('total_newton_iterations', 'N/A')}\n")
					f.write(f"   - Timestep cuts:         {exp.get('total_cuts', 'N/A')}\n")
					f.write(f"   - Simulation time:       {exp.get('simulation_time', 'N/A'):.4f} s\n")
					f.write(f"   - Wall clock time:       {exp.get('wall_clock_time', 'N/A'):.4f} s\n")
					
					# Get detailed stats
					stats = self.get_runtime_stats(exp['case_name'])
					if stats and stats.get('avg_newton_per_step'):
						f.write(f"   - Avg Newton/step:       {stats['avg_newton_per_step']:.2f}\n")
					
					if stats and stats.get('timestep_stats'):
						ts = stats['timestep_stats']
						f.write(f"\n   Timestep Statistics:\n")
						f.write(f"   - Min dt:                {ts['min']:.6e}\n")
						f.write(f"   - Max dt:                {ts['max']:.6e}\n")
						f.write(f"   - Avg dt:                {ts['avg']:.6e}\n")
				
				elif exp['status'] == 'failed':
					f.write(f"\n   Error: {exp.get('error_message', 'Unknown error')}\n")
				
				f.write("\n" + "-"*80 + "\n\n")
		
		print(f"✓ Report saved: {save_path}")
	
	def export_to_dataframe(self) -> pd.DataFrame:
		"""
		Export results to pandas DataFrame.
		
		Returns:
			DataFrame with experiment results
		"""
		data = []
		
		for exp in self.experiments:
			case_name = exp['case_name']
			result = self.results[case_name]
			
			row = {
				'case_name': case_name,
				'scenario_name': exp['scenario_name'],
				'status': result['status'],
				'initial_condition': exp.get('initial_condition', 'N/A'),
				'surface_recharge': exp.get('surface_recharge', 'N/A'),
				'deep_boundary': exp.get('deep_boundary', 'N/A')
			}
			
			if result['stats']:
				stats = result['stats']
				row.update({
					'n_processes': stats.get('n_processes'),
					'n_grid_cells': stats.get('n_grid_cells'),
					'total_steps': stats.get('total_steps'),
					'total_newton_iterations': stats.get('total_newton_iterations'),
					'total_linear_iterations': stats.get('total_linear_iterations'),
					'total_cuts': stats.get('total_cuts'),
					'avg_newton_per_step': stats.get('avg_newton_per_step'),
					'wasted_newton_iterations': stats.get('wasted_newton_iterations'),
					'wasted_linear_iterations': stats.get('wasted_linear_iterations'),
					'final_time': stats.get('final_time'),
					'simulation_time': stats.get('simulation_time'),
					'wall_clock_time': stats.get('wall_clock_time'),
					'min_timestep': stats.get('timestep_stats', {}).get('min'),
					'max_timestep': stats.get('timestep_stats', {}).get('max'),
					'avg_timestep': stats.get('timestep_stats', {}).get('avg')
				})
			
			data.append(row)
		
		return pd.DataFrame(data)
	
	def plot_performance_summary(self, save_path: str = None, show: bool = False):
		"""
		Plot performance summary across all experiments.
		
		Args:
			save_path: Path to save figure (optional)
			show: Whether to display the plot (default: False)
		"""
		import matplotlib.pyplot as plt
		
		# Get completed experiments
		completed = [exp for exp in self.get_summary()['experiments'] 
					if exp['status'] == 'completed']
		
		if not completed:
			print("No completed experiments to plot")
			return
		
		fig, axes = plt.subplots(2, 2, figsize=(14, 10))
		fig.suptitle('Simulation Performance Summary', fontsize=16, fontweight='bold')
		
		case_names = [exp['case_name'] for exp in completed]
		
		# Plot 1: Total timesteps
		ax = axes[0, 0]
		steps = [exp['total_steps'] for exp in completed]
		bars = ax.bar(range(len(case_names)), steps, color='steelblue', edgecolor='black')
		ax.set_xlabel('Experiment', fontsize=11, fontweight='bold')
		ax.set_ylabel('Total Timesteps', fontsize=11, fontweight='bold')
		ax.set_title('Total Timesteps', fontsize=12, fontweight='bold')
		ax.set_xticks(range(len(case_names)))
		#ax.set_xticklabels([f'E{i+1}' for i in range(len(case_names))], rotation=0)
		ax.set_xticklabels(case_names, rotation=90)
		ax.grid(axis='y', alpha=0.3)
		for i, (bar, val) in enumerate(zip(bars, steps)):
			ax.text(i, val, str(val), ha='center', va='bottom', fontsize=9)
		
		# Plot 2: Newton iterations
		ax = axes[0, 1]
		newton = [exp['total_newton_iterations'] for exp in completed]
		bars = ax.bar(range(len(case_names)), newton, color='coral', edgecolor='black')
		ax.set_xlabel('Experiment', fontsize=11, fontweight='bold')
		ax.set_ylabel('Total Newton Iterations', fontsize=11, fontweight='bold')
		ax.set_title('Newton Iterations', fontsize=12, fontweight='bold')
		ax.set_xticks(range(len(case_names)))
		#ax.set_xticklabels([f'E{i+1}' for i in range(len(case_names))], rotation=0)
		ax.set_xticklabels(case_names, rotation=90)
		ax.grid(axis='y', alpha=0.3)
		for i, (bar, val) in enumerate(zip(bars, newton)):
			ax.text(i, val, str(val), ha='center', va='bottom', fontsize=9)
		
		# Plot 3: Wall clock time
		ax = axes[1, 0]
		times = [exp['wall_clock_time'] for exp in completed]
		bars = ax.bar(range(len(case_names)), times, color='lightgreen', edgecolor='black')
		ax.set_xlabel('Experiment', fontsize=11, fontweight='bold')
		ax.set_ylabel('Wall Clock Time (s)', fontsize=11, fontweight='bold')
		ax.set_title('Execution Time', fontsize=12, fontweight='bold')
		ax.set_xticks(range(len(case_names)))
		#ax.set_xticklabels([f'E{i+1}' for i in range(len(case_names))], rotation=0)
		ax.set_xticklabels(case_names, rotation=90)
		ax.grid(axis='y', alpha=0.3)
		for i, (bar, val) in enumerate(zip(bars, times)):
			ax.text(i, val, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
		
		# Plot 4: Timestep cuts
		ax = axes[1, 1]
		cuts = [exp['total_cuts'] for exp in completed]
		bars = ax.bar(range(len(case_names)), cuts, color='salmon', edgecolor='black')
		ax.set_xlabel('Experiment', fontsize=11, fontweight='bold')
		ax.set_ylabel('Total Timestep Cuts', fontsize=11, fontweight='bold')
		ax.set_title('Timestep Cuts', fontsize=12, fontweight='bold')
		ax.set_xticks(range(len(case_names)))
		#ax.set_xticklabels([f'E{i+1}' for i in range(len(case_names))], rotation=0)
		ax.set_xticklabels(case_names, rotation=90)
		ax.grid(axis='y', alpha=0.3)
		for i, (bar, val) in enumerate(zip(bars, cuts)):
			ax.text(i, val, str(val), ha='center', va='bottom', fontsize=9)
		
		plt.tight_layout()
		
		if save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
			print(f"✓ Performance summary plot saved: {save_path}")
		
		if show:
			plt.show()
		else:
			plt.close()


if __name__ == "__main__":
    print("Experiment Manager Module")
    print("=" * 50)
    print("\nThis module manages execution of PFLOTRAN experiments.")
    print("\nUsage:")
    print("  from experiment_manager import ExperimentManager")
    print("\n  manager = ExperimentManager(experiments)")
    print("  manager.run_all_experiments(pflotran_exe='pflotran')")
    print("  manager.generate_report('report.txt')")