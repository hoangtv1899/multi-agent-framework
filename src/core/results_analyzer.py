#!/usr/bin/env python3
"""
Results Analyzer - Simple analysis and visualization of PFLOTRAN simulation results
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

# Import PFLOTRAN Python utilities
try:
    pflotran_dir = os.environ['PFLOTRAN_DIR']
except KeyError:
    print('Warning: PFLOTRAN_DIR not set. Using fallback method for data loading.')
    pflotran_dir = None

if pflotran_dir:
    sys.path.append(pflotran_dir + '/src/python')
    import pflotran as pft
    PFLOTRAN_UTILS_AVAILABLE = True
else:
    PFLOTRAN_UTILS_AVAILABLE = False


class ResultsAnalyzer:
    """
    Simple analyzer for PFLOTRAN simulation results.
    Focuses on saturation profile visualization and comparison.
    """
    
    def __init__(self, experiment_manager):
        """
        Initialize ResultsAnalyzer.
        
        Args:
            experiment_manager: ExperimentManager instance with completed simulations
        """
        self.manager = experiment_manager
        self.pflotran_dir = pflotran_dir
        
        if not PFLOTRAN_UTILS_AVAILABLE:
            print("Warning: PFLOTRAN Python utilities not available.")
            print("Set PFLOTRAN_DIR environment variable for full functionality.")
    
    def get_output_files(self, case_name: str, time_indices: Optional[List[int]] = None):
        """
        Get Tecplot output filenames for a case.
        
        Args:
            case_name: Name of the experiment case
            time_indices: List of time indices (e.g., [0, 1, 2, 5])
            
        Returns:
            List of full file paths
        """
        if not PFLOTRAN_UTILS_AVAILABLE:
            print("Error: PFLOTRAN utilities not available")
            return []
        
        # Get case directory
        exp = next((e for e in self.manager.experiments if e['case_name'] == case_name), None)
        if not exp:
            print(f"Error: Case '{case_name}' not found")
            return []
        
        agent = exp['pflotran_agent']
        case_dir = agent.case_dir
        
        if not case_dir or not os.path.exists(case_dir):
            print(f"Error: Case directory not found for {case_name}")
            return []
        
        # Get Tecplot filenames
        if time_indices is None:
            time_indices = range(10)  # Default: try first 10 files
        
        files = pft.get_tec_filenames(case_name, time_indices)
        filenames = pft.get_full_paths([case_dir], files)
        
        # Filter to only existing files
        existing_files = [f for f in filenames if os.path.exists(f)]
        
        return existing_files
    
    def load_dataset(self, filepath: str, x_column: int = 5, y_column: int = 3):
        """
        Load a Tecplot dataset.
        
        Args:
            filepath: Path to Tecplot file
            x_column: Column index for x-axis data (saturation)
            y_column: Column index for y-axis data (depth)
            
        Returns:
            PFLOTRAN Dataset object
        """
        if not PFLOTRAN_UTILS_AVAILABLE:
            print("Error: PFLOTRAN utilities not available")
            return None
        
        try:
            data = pft.Dataset(filepath, x_column, y_column)
            return data
        except Exception as e:
            print(f"Error loading dataset from {filepath}: {e}")
            return None
    
    def extract_time_from_title(self, data_title: str) -> Optional[float]:
        """
        Extract time value (in years) from Tecplot dataset title.
        
        Args:
            data_title: Title string from dataset (e.g., "Time=  1.0000E+01 y")
            
        Returns:
            Time value in years, or None if not found
        """
        import re
        
        # Match pattern like "Time=  1.0000E+01 y" or "1.0000E+01"
        match = re.search(r'([\d.E+-]+)\s*y', data_title)
        if match:
            return float(match.group(1))
        
        # Try to extract just the number
        match = re.search(r'([\d.E+-]+)', data_title)
        if match:
            return float(match.group(1))
        
        return None
    
    def extract_saturation_data(self, time_indices: List[int] = None) -> Dict[str, Any]:
        """
        Extract saturation profile data from all completed experiments.
        
        Args:
            time_indices: List of time indices to extract (default: [0,1,2,3,4,5])
            
        Returns:
            Dictionary with saturation data for each experiment
        """
        if time_indices is None:
            time_indices = [0, 1, 2, 3, 4, 5]
        
        saturation_data = {}
        
        # Get completed experiments
        completed = [exp for exp in self.manager.experiments
                    if self.manager.status.get(exp['case_name']) == 'completed']
        
        for exp in completed:
            case_name = exp['case_name']
            scenario_name = exp.get('scenario_name', case_name)
            
            try:
                # Get available snapshots
                snapshots = self.get_available_snapshots(case_name)
                
                if not snapshots:
                    continue
                
                # Extract data for requested time indices
                times = []
                time_labels = []
                depths_data = []
                saturation_data_arrays = []
                
                # Get files for requested time indices
                available_indices = [idx for idx in time_indices if idx in snapshots]
                filenames = self.get_output_files(case_name, available_indices)
                
                for idx, filename in zip(available_indices, filenames):
                    data = self.load_dataset(filename, x_column=5, y_column=3)
                    if data:
                        time_val = self.extract_time_from_title(data.title)
                        depths = data.get_array('y')
                        saturations = data.get_array('x')
                        
                        if time_val is not None:
                            times.append(time_val)
                            time_labels.append(f"{time_val:.2f} y")
                        else:
                            times.append(idx)
                            time_labels.append(f"Index {idx}")
                        
                        depths_data.append(depths.tolist())
                        saturation_data_arrays.append(saturations.tolist())
                
                # Get overall depth and saturation ranges
                if depths_data and saturation_data_arrays:
                    all_depths = np.concatenate([np.array(d) for d in depths_data])
                    all_sats = np.concatenate([np.array(s) for s in saturation_data_arrays])
                    
                    depth_range = f"{all_depths.min():.2f} - {all_depths.max():.2f} m"
                    sat_range = f"{all_sats.min():.3f} - {all_sats.max():.3f}"
                else:
                    depth_range = "N/A"
                    sat_range = "N/A"
                
                saturation_data[scenario_name] = {
                    'case_name': case_name,
                    'scenario_name': scenario_name,
                    'num_snapshots': len(snapshots),
                    'available_indices': snapshots,
                    'extracted_indices': available_indices,
                    'times': times,
                    'time_labels': time_labels,
                    'depths': depths_data,
                    'saturations': saturation_data_arrays,
                    'depth_range': depth_range,
                    'saturation_range': sat_range,
                    'num_profiles': len(times)
                }
                
            except Exception as e:
                print(f"Warning: Could not extract saturation data from {scenario_name}: {e}")
                continue
        
        return saturation_data
    
    def save_saturation_data(self, save_path: str, time_indices: List[int] = None):
        """
        Extract and save saturation data to JSON file.
        
        Args:
            save_path: Path to save JSON file
            time_indices: List of time indices to extract
            
        Returns:
            Path to saved file
        """
        saturation_data = self.extract_saturation_data(time_indices)
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(saturation_data, f, indent=2, default=str)
        
        print(f"✓ Saturation data saved to: {save_path}")
        
        return save_path
    
    def plot_saturation_data_comparison(self, save_path: str, time_indices: List[int] = None):
        """
        Create and save saturation profile comparison plot directly from extracted data.
        
        Args:
            save_path: Path to save the plot
            time_indices: List of time indices to plot
            
        Returns:
            Path to saved plot
        """
        if time_indices is None:
            time_indices = [0, 1, 2, 3, 4, 5]
        
        saturation_data = self.extract_saturation_data(time_indices)
        
        if not saturation_data:
            print("No saturation data available to plot")
            return None
        
        # Create multi-panel figure (one panel per time index)
        n_times = len(time_indices)
        n_cols = min(3, n_times)
        n_rows = (n_times + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        
        if n_times == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_times > 1 else axes
        
        fig.suptitle('Saturation Profile Comparison Across Experiments',
                    fontsize=16, fontweight='bold')
        
        # Colors for different experiments
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(saturation_data)))
        line_styles = ['-', '--', '-.', ':', '-']
        
        # Plot each time snapshot in its own subplot
        for time_idx_pos, (ax, time_idx) in enumerate(zip(axes, time_indices)):
            plotted = False
            
            for exp_idx, (exp_name, exp_data) in enumerate(saturation_data.items()):
                # Check if this time index is available for this experiment
                if time_idx not in exp_data['extracted_indices']:
                    continue
                
                # Find the position of this time index in the data arrays
                try:
                    data_pos = exp_data['extracted_indices'].index(time_idx)
                    depths = exp_data['depths'][data_pos]
                    sats = exp_data['saturations'][data_pos]
                    time_label = exp_data['time_labels'][data_pos]
                    
                    ls_idx = exp_idx % len(line_styles)
                    
                    ax.plot(sats, depths,
                           label=exp_name,
                           ls=line_styles[ls_idx],
                           color=colors[exp_idx],
                           linewidth=2.5)
                    
                    plotted = True
                    
                except (IndexError, ValueError):
                    continue
            
            if plotted:
                ax.set_xlabel('Liquid Saturation [-]', fontsize=11, fontweight='bold')
                ax.set_ylabel('Depth [m]', fontsize=11, fontweight='bold')
                
                # Use first experiment's time label for title
                first_exp_data = list(saturation_data.values())[0]
                if time_idx in first_exp_data['extracted_indices']:
                    idx_pos = first_exp_data['extracted_indices'].index(time_idx)
                    time_title = first_exp_data['time_labels'][idx_pos]
                else:
                    time_title = f"Time Index {time_idx}"
                
                ax.set_title(f"Time = {time_title}", fontsize=12, fontweight='bold')
                ax.set_xlim(0., 1.)
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3, linestyle='--')
            else:
                ax.set_visible(False)
        
        # Hide unused subplots
        for idx in range(len(time_indices), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saturation comparison plot saved to: {save_path}")
        
        return save_path
    
    def plot_saturation_profiles(self, case_name: str, time_indices: List[int] = None,
                                 save_path: Optional[str] = None, show: bool = False):
        """
        Plot saturation vs depth profiles for one experiment at multiple times.
        
        Args:
            case_name: Name of the experiment case
            time_indices: List of time snapshot indices to plot
            save_path: Path to save figure (optional)
            show: Whether to display the plot (default: False)
            
        Returns:
            fig, ax: matplotlib figure and axes objects
        """
        if not PFLOTRAN_UTILS_AVAILABLE:
            print("Error: PFLOTRAN utilities not available")
            return None, None
        
        if time_indices is None:
            time_indices = [0, 1, 2, 3, 4, 5]
        
        # Get output files
        filenames = self.get_output_files(case_name, time_indices)
        if not filenames:
            print(f"No output files found for {case_name}")
            return None, None
        
        # Create figure
        fig = plt.figure(figsize=(8, 10))
        plt.subplot(1, 1, 1)
        
        # Get experiment info
        exp = next((e for e in self.manager.experiments if e['case_name'] == case_name), None)
        scenario_name = exp['scenario_name'] if exp else case_name
        
        fig.suptitle(f"Saturation Profiles: {scenario_name}", fontsize=14, fontweight='bold')
        plt.xlabel('Liquid Saturation [-]', fontsize=12, fontweight='bold')
        plt.ylabel('Depth [m]', fontsize=12, fontweight='bold')
        plt.xlim(0., 1.)
        
        # Line styles for different times
        line_styles = ['-', '-', '-', '--', '--', '-.', '-.', ':', ':', '-']
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(filenames)))
        
        # Plot each time snapshot
        for ifile, (filename, color) in enumerate(zip(filenames, colors)):
            data = self.load_dataset(filename, x_column=5, y_column=3)
            if data:
                # Extract time from title
                time_value = self.extract_time_from_title(data.title)
                if time_value is not None:
                    label = f"{time_value:.2f} y"
                else:
                    label = data.title
                
                ls_idx = ifile % len(line_styles)
                plt.plot(data.get_array('x'), data.get_array('y'),
                        label=label,
                        ls=line_styles[ls_idx],
                        color=color,
                        linewidth=2)
        
        plt.legend(loc='best', title='Time', fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        fig.subplots_adjust(hspace=0.2, wspace=0.2,
                          bottom=.12, top=.92,
                          left=.14, right=.94)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saturation profile plot saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig, plt.gca()
    
    def compare_saturation_profiles_combined(self, case_names: List[str],
                                            time_indices: List[int] = None,
                                            save_path: Optional[str] = None,
                                            show: bool = False):
        """
        Compare saturation profiles across multiple experiments at multiple times.
        Creates a multi-panel figure with one panel per time snapshot.
        
        Args:
            case_names: List of experiment case names to compare
            time_indices: List of time snapshot indices to compare
            save_path: Path to save figure (optional)
            show: Whether to display the plot (default: False)
            
        Returns:
            fig, axes: matplotlib figure and axes objects
        """
        if not PFLOTRAN_UTILS_AVAILABLE:
            print("Error: PFLOTRAN utilities not available")
            return None, None
        
        if time_indices is None:
            time_indices = [0, 1, 2, 3, 4, 5]
        
        # Calculate subplot layout
        n_times = len(time_indices)
        n_cols = min(3, n_times)
        n_rows = (n_times + n_cols - 1) // n_cols
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        
        # Handle single subplot case
        if n_times == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_times > 1 else axes
        
        fig.suptitle('Saturation Profile Comparison Across Experiments',
                    fontsize=16, fontweight='bold')
        
        # Colors for different experiments
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(case_names)))
        line_styles = ['-', '--', '-.', ':', '-']
        
        # Plot each time snapshot in its own subplot
        for time_idx, ax in zip(time_indices, axes):
            plotted = False
            time_label = None
            
            for i, (case_name, color) in enumerate(zip(case_names, colors)):
                # Get output file for this time index
                filenames = self.get_output_files(case_name, [time_idx])
                
                if not filenames:
                    continue
                
                # Load data
                data = self.load_dataset(filenames[0], x_column=5, y_column=3)
                
                if data:
                    # Extract time from first dataset for subplot title
                    if time_label is None:
                        time_value = self.extract_time_from_title(data.title)
                        if time_value is not None:
                            time_label = f"Time = {time_value:.2f} years"
                        else:
                            time_label = f"Time Index = {time_idx}"
                    
                    # Get experiment info for label
                    exp = next((e for e in self.manager.experiments
                              if e['case_name'] == case_name), None)
                    label = exp['scenario_name'] if exp else case_name
                    
                    ls_idx = i % len(line_styles)
                    ax.plot(data.get_array('x'), data.get_array('y'),
                           label=label,
                           ls=line_styles[ls_idx],
                           color=color,
                           linewidth=2.5)
                    plotted = True
            
            if plotted:
                ax.set_xlabel('Liquid Saturation [-]', fontsize=11, fontweight='bold')
                ax.set_ylabel('Depth [m]', fontsize=11, fontweight='bold')
                ax.set_title(time_label, fontsize=12, fontweight='bold')
                ax.set_xlim(0., 1.)
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3, linestyle='--')
            else:
                ax.set_visible(False)
        
        # Hide unused subplots
        for idx in range(len(time_indices), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Combined comparison plot saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig, axes
    
    def compare_saturation_profiles(self, case_names: List[str], time_index: int = 5,
                                   save_path: Optional[str] = None, show: bool = False):
        """
        Compare saturation profiles across multiple experiments at one time.
        
        Args:
            case_names: List of experiment case names to compare
            time_index: Time snapshot index to compare
            save_path: Path to save figure (optional)
            show: Whether to display the plot (default: False)
            
        Returns:
            fig, ax: matplotlib figure and axes objects
        """
        if not PFLOTRAN_UTILS_AVAILABLE:
            print("Error: PFLOTRAN utilities not available")
            return None, None
        
        fig = plt.figure(figsize=(8, 10))
        plt.subplot(1, 1, 1)
        
        # Colors for different experiments
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(case_names)))
        line_styles = ['-', '--', '-.', ':', '-']
        
        plotted = False
        time_label = None
        
        for i, (case_name, color) in enumerate(zip(case_names, colors)):
            # Get output file for this time index
            filenames = self.get_output_files(case_name, [time_index])
            
            if not filenames:
                print(f"Warning: No output file at time index {time_index} for {case_name}")
                continue
            
            # Load data
            data = self.load_dataset(filenames[0], x_column=5, y_column=3)
            
            if data:
                # Extract time from first dataset for title
                if time_label is None:
                    time_value = self.extract_time_from_title(data.title)
                    if time_value is not None:
                        time_label = f"Time = {time_value:.2f} years"
                    else:
                        time_label = f"Time Index = {time_index}"
                
                # Get experiment info for label
                exp = next((e for e in self.manager.experiments
                          if e['case_name'] == case_name), None)
                label = exp['scenario_name'] if exp else case_name
                
                ls_idx = i % len(line_styles)
                plt.plot(data.get_array('x'), data.get_array('y'),
                        label=label,
                        ls=line_styles[ls_idx],
                        color=color,
                        linewidth=2.5)
                plotted = True
        
        if not plotted:
            print("Error: No data to plot")
            plt.close()
            return None, None
        
        fig.suptitle(f"Saturation Profile Comparison\n{time_label}",
                    fontsize=14, fontweight='bold')
        plt.xlabel('Liquid Saturation [-]', fontsize=12, fontweight='bold')
        plt.ylabel('Depth [m]', fontsize=12, fontweight='bold')
        plt.xlim(0., 1.)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        fig.subplots_adjust(hspace=0.2, wspace=0.2,
                          bottom=.12, top=.88,
                          left=.14, right=.94)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comparison plot saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig, plt.gca()
    
    def plot_all_experiments(self, time_indices: List[int] = None,
                           save_dir: str = "./results"):
        """
        Generate saturation profile plots for all completed experiments.
        Creates individual plots and a combined comparison plot.
        
        Args:
            time_indices: List of time snapshot indices to plot
            save_dir: Directory to save figures
            
        Returns:
            Dictionary with paths to generated figures
        """
        if time_indices is None:
            time_indices = [0, 1, 2, 3, 4, 5]
        
        os.makedirs(save_dir, exist_ok=True)
        
        figures = {}
        
        print("\n" + "="*70)
        print("GENERATING SATURATION PROFILE PLOTS")
        print("="*70)
        
        # Get completed experiments
        completed = [exp for exp in self.manager.experiments
                    if self.manager.status.get(exp['case_name']) == 'completed']
        
        if not completed:
            print("No completed experiments to plot")
            return figures
        
        # Plot individual experiments
        print("\nGenerating individual experiment plots...")
        for exp in completed:
            case_name = exp['case_name']
            print(f"  • {exp['scenario_name']}")
            
            save_path = os.path.join(save_dir, f"{case_name}_saturation_profiles.png")
            self.plot_saturation_profiles(
                case_name=case_name,
                time_indices=time_indices,
                save_path=save_path
            )
            figures[case_name] = save_path
        
        # Create combined comparison plot (all times in one figure)
        if len(completed) > 1:
            print(f"\nCreating combined comparison plot...")
            case_names = [exp['case_name'] for exp in completed]
            
            save_path = os.path.join(save_dir, "comparison_all_times.png")
            self.compare_saturation_profiles_combined(
                case_names=case_names,
                time_indices=time_indices,
                save_path=save_path
            )
            figures['comparison_combined'] = save_path
        
        print(f"\n✓ Generated {len(figures)} plots in {save_dir}")
        
        return figures
    
    def generate_analysis_report(self, save_path: str, time_indices: List[int] = None):
        """
        Generate detailed analysis report.
        
        Args:
            save_path: Path to save report
            time_indices: Time indices analyzed
        """
        if time_indices is None:
            time_indices = [0, 1, 2, 3, 4, 5]
        
        with open(save_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PFLOTRAN RESULTS ANALYZER - ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary from manager
            summary = self.manager.get_summary()
            f.write("SIMULATION SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total experiments:      {summary['total_experiments']}\n")
            f.write(f"Completed:              {summary['completed']}\n")
            f.write(f"Failed:                 {summary['failed']}\n\n")
            
            f.write("ANALYSIS CONFIGURATION\n")
            f.write("-"*80 + "\n")
            f.write(f"Time indices analyzed:  {time_indices}\n")
            f.write(f"Output format:          Tecplot\n")
            f.write(f"Primary variable:       Liquid Saturation\n")
            f.write(f"Time units:             Years\n\n")
            
            f.write("EXPERIMENT DETAILS\n")
            f.write("="*80 + "\n\n")
            
            # Get completed experiments
            completed = [exp for exp in self.manager.experiments
                        if self.manager.status.get(exp['case_name']) == 'completed']
            
            for idx, exp in enumerate(completed, 1):
                case_name = exp['case_name']
                
                f.write(f"{idx}. {exp['scenario_name']}\n")
                f.write(f"   Case name: {case_name}\n")
                f.write(f"   Initial condition: {exp.get('initial_condition', 'N/A')}\n")
                f.write(f"   Surface recharge:  {exp.get('surface_recharge', 'N/A')}\n")
                f.write(f"   Deep boundary:     {exp.get('deep_boundary', 'N/A')}\n\n")
                
                # Get simulation stats
                stats = self.manager.get_runtime_stats(case_name)
                if stats:
                    f.write(f"   Simulation Performance:\n")
                    f.write(f"   - Grid cells:          {stats.get('n_grid_cells', 'N/A')}\n")
                    f.write(f"   - Total timesteps:     {stats.get('total_steps', 'N/A')}\n")
                    f.write(f"   - Final time reached:  {stats.get('final_time', 'N/A')} y\n")
                    f.write(f"   - Wall clock time:     {stats.get('wall_clock_time', 'N/A'):.4f} s\n")
                    f.write(f"   - CPU cores used:      {stats.get('n_processes', 'N/A')}\n")
                    f.write(f"   - Newton iterations:   {stats.get('total_newton_iterations', 'N/A')}\n")
                    f.write(f"   - Avg Newton/step:     {stats.get('avg_newton_per_step', 'N/A'):.2f}\n")
                    f.write(f"   - Timestep cuts:       {stats.get('total_cuts', 'N/A')}\n\n")
                    
                    if stats.get('timestep_stats'):
                        ts = stats['timestep_stats']
                        f.write(f"   Timestep Statistics:\n")
                        f.write(f"   - Min dt:              {ts.get('min', 'N/A'):.6e}\n")
                        f.write(f"   - Max dt:              {ts.get('max', 'N/A'):.6e}\n")
                        f.write(f"   - Avg dt:              {ts.get('avg', 'N/A'):.6e}\n\n")
                
                # Check for output files
                output_files = self.get_output_files(case_name, time_indices)
                f.write(f"   Output Files:\n")
                f.write(f"   - Number of snapshots: {len(output_files)}\n")
                if output_files:
                    f.write(f"   - Available times:\n")
                    for output_file in output_files:
                        data = self.load_dataset(output_file)
                        if data:
                            time_val = self.extract_time_from_title(data.title)
                            if time_val is not None:
                                f.write(f"     • {time_val:.4f} years\n")
                            else:
                                f.write(f"     • {os.path.basename(output_file)}\n")
                
                f.write("\n" + "-"*80 + "\n\n")
            
            # Add failed experiments if any
            failed = [exp for exp in self.manager.experiments
                     if self.manager.status.get(exp['case_name']) == 'failed']
            
            if failed:
                f.write("FAILED EXPERIMENTS\n")
                f.write("="*80 + "\n\n")
                for idx, exp in enumerate(failed, 1):
                    case_name = exp['case_name']
                    result = self.manager.results[case_name]
                    f.write(f"{idx}. {exp['scenario_name']}\n")
                    f.write(f"   Case name: {case_name}\n")
                    f.write(f"   Error: {result.get('error_message', 'Unknown error')}\n\n")
        
        print(f"✓ Analysis report saved: {save_path}")
    
    def get_available_snapshots(self, case_name: str) -> List[int]:
        """
        Get list of available time snapshot indices for a case.
        
        Args:
            case_name: Name of the experiment case
            
        Returns:
            List of available time indices
        """
        if not PFLOTRAN_UTILS_AVAILABLE:
            return []
        
        # Try to find files with indices 0-99
        available = []
        for i in range(100):
            files = self.get_output_files(case_name, [i])
            if files:
                available.append(i)
        
        return available
    
    def list_available_data(self):
        """
        Print summary of available output data for all experiments.
        """
        print("\n" + "="*70)
        print("AVAILABLE OUTPUT DATA")
        print("="*70)
        
        completed = [exp for exp in self.manager.experiments
                    if self.manager.status.get(exp['case_name']) == 'completed']
        
        for exp in completed:
            case_name = exp['case_name']
            snapshots = self.get_available_snapshots(case_name)
            
            print(f"\n{exp['scenario_name']}")
            print(f"  Case: {case_name}")
            print(f"  Available snapshots: {len(snapshots)}")
            if snapshots:
                print(f"  Indices: {snapshots[:10]}{'...' if len(snapshots) > 10 else ''}")
                
                # Show actual times
                if len(snapshots) > 0:
                    files = self.get_output_files(case_name, snapshots[:5])
                    times = []
                    for f in files:
                        data = self.load_dataset(f)
                        if data:
                            time_val = self.extract_time_from_title(data.title)
                            if time_val is not None:
                                times.append(f"{time_val:.2f} y")
                    if times:
                        print(f"  Times: {', '.join(times)}{'...' if len(snapshots) > 5 else ''}")


if __name__ == "__main__":
    print("Results Analyzer Module")
    print("=" * 50)
    print("\nThis module analyzes PFLOTRAN simulation results.")
    print("\nRequires:")
    print("  - PFLOTRAN_DIR environment variable set")
    print("  - Completed ExperimentManager with simulation results")
    print("\nMain Features:")
    print("  - Extracts time values in YEARS from output files")
    print("  - Plots saturation vs depth profiles")
    print("  - Creates combined multi-panel comparison plots")
    print("  - Exports saturation data to JSON")
    print("\nNew Methods:")
    print("  - extract_saturation_data(time_indices)")
    print("  - save_saturation_data(save_path, time_indices)")
    print("  - plot_saturation_data_comparison(save_path, time_indices)")
    print("\nUsage:")
    print("  analyzer = ResultsAnalyzer(manager)")
    print("  analyzer.plot_all_experiments(time_indices=[0,1,2,5])")
    print("  analyzer.save_saturation_data('saturation_data.json')")
    print("  analyzer.plot_saturation_data_comparison('comparison.png')")
    print("  analyzer.generate_analysis_report('report.txt')")