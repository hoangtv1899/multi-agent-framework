#!/usr/bin/env python3
"""
PFLOTRAN Plotting Module - Simplified visualization for PFLOTRAN experiments
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Optional


class ExperimentPlot:
    """
    Container for experiment plot data - used for comparison plots.
    """
    def __init__(self, case_name, layer_thicknesses, layer_names, 
                 initial_condition, surface_bc, bottom_bc, total_depth,
                 water_table_depth=None, flux_data=None):
        self.case_name = case_name
        self.layer_thicknesses = layer_thicknesses
        self.layer_names = layer_names
        self.initial_condition = initial_condition
        self.surface_bc = surface_bc
        self.bottom_bc = bottom_bc
        self.total_depth = total_depth
        self.water_table_depth = water_table_depth
        self.flux_data = flux_data  # Dictionary with 'times', 'fluxes', 'units'


class PFLOTRANPlotter:
	"""
	Simplified plotting utilities for PFLOTRAN Input Agent.
	"""
	
	@staticmethod
	def plot_experiment_overview(agent, save_path: Optional[str] = None, 
								 show: bool = False):
		"""
		Plot simplified overview: layer profile with water table + flux recharge.
		
		Args:
			agent: PFLOTRANInputAgent instance
			save_path: Path to save figure (optional)
			show: Whether to display the plot (default: False)
		
		Returns:
			ExperimentPlot object containing plot data for comparison
		"""
		fig, axes = plt.subplots(1, 2, figsize=(14, 8))
		
		# Main title
		fig.suptitle(f'Experiment Overview: {agent.case_name}', 
					fontsize=16, fontweight='bold')
		
		# Subplot 1: Layer profile with water table
		ax1 = axes[0]
		water_table_depth = PFLOTRANPlotter._plot_layer_profile(ax1, agent)
		
		# Subplot 2: Flux recharge conditions
		ax2 = axes[1]
		flux_data = PFLOTRANPlotter._plot_flux_recharge(ax2, agent)
		
		plt.tight_layout()
		
		if save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
			print(f"✓ Experiment overview saved: {save_path}")
		
		if show:
			plt.show()
		else:
			plt.close()
		
		# Return plot object for comparison
		return PFLOTRANPlotter._create_plot_object(agent, water_table_depth, flux_data)
	
	@staticmethod
	def _plot_layer_profile(ax, agent):
		"""Plot layer column with water table."""
		total_depth = sum(agent.layer_thicknesses)
		
		# Get water table depth from initial condition
		initial_fc = agent.flow_conditions[0] if agent.flow_conditions else None
		water_table_depth = None
		
		if initial_fc and 'datum' in initial_fc:
			water_table_depth = initial_fc['datum'][2]
		
		# Generate colors
		cmap = plt.cm.get_cmap('RdYlBu_r')
		colors = [cmap(i/len(agent.layer_thicknesses)) 
				 for i in range(len(agent.layer_thicknesses))]
		
		# Draw layers
		cumulative_depth = 0
		
		for i, (thickness, color) in enumerate(zip(agent.layer_thicknesses, colors)):
			# Determine saturation status
			if water_table_depth is not None:
				if cumulative_depth + thickness <= water_table_depth:
					# Fully saturated
					alpha = 0.9
					edge_color = 'darkblue'
					edge_width = 2
				elif cumulative_depth >= water_table_depth:
					# Fully unsaturated
					alpha = 0.4
					edge_color = 'black'
					edge_width = 1
				else:
					# Partially saturated
					alpha = 0.7
					edge_color = 'blue'
					edge_width = 2
			else:
				alpha = 0.7
				edge_color = 'black'
				edge_width = 1.5
			
			rect = mpatches.Rectangle((0, cumulative_depth), 1, thickness,
									 facecolor=color, edgecolor=edge_color, 
									 linewidth=edge_width, alpha=alpha)
			ax.add_patch(rect)
			
			# Add layer label
			layer_name = agent.layer_order[i] if i < len(agent.layer_order) else f"L{i+1}"
			ax.text(0.5, cumulative_depth + thickness/2, 
				   f'{layer_name}\n{thickness:.3f}m',
				   ha='center', va='center', fontsize=8, fontweight='bold')
			
			cumulative_depth += thickness
		
		# Draw water table line if available
		if water_table_depth is not None:
			ax.axhline(y=water_table_depth, color='blue', linewidth=3, 
					  linestyle='--', label=f'Water Table @ {water_table_depth:.2f}m')
			
			# Add water table annotation
			ax.annotate('Water Table', 
					   xy=(0.5, water_table_depth), 
					   xytext=(1.5, water_table_depth),
					   fontsize=11, fontweight='bold', color='blue',
					   arrowprops=dict(arrowstyle='->', color='blue', lw=2),
					   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
			
			# Add zone labels
			if water_table_depth > 0:
				ax.text(-0.3, water_table_depth/2, 'Saturated\nZone',
					   ha='right', va='center', fontsize=10, fontweight='bold',
					   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
			
			if water_table_depth < total_depth:
				unsaturated_mid = (water_table_depth + total_depth) / 2
				ax.text(-0.3, unsaturated_mid, 'Unsaturated\nZone',
					   ha='right', va='center', fontsize=10, fontweight='bold',
					   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
		
		# Get boundary condition names
		surface_bc = "N/A"
		bottom_bc = "N/A"
		
		for bc in agent.boundary_conditions:
			if bc['region'] == 'top':
				surface_bc = bc['flow_condition']
			elif bc['region'] == 'bottom':
				bottom_bc = bc['flow_condition']
		
		# Add boundary labels
		ax.text(1.3, total_depth - 0.3, f"Surface BC:\n{surface_bc}",
			   fontsize=9, va='top', ha='left',
			   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
		
		ax.text(1.3, 0.3, f"Bottom BC:\n{bottom_bc}",
			   fontsize=9, va='bottom', ha='left',
			   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
		
		# Initial condition
		initial_name = agent.flow_conditions[0]['name'] if agent.flow_conditions else "N/A"
		ax.text(-0.5, total_depth/2, f"Initial:\n{initial_name}",
			   fontsize=9, va='center', ha='right',
			   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
		
		ax.set_xlim(-1.5, 2.5)
		ax.set_ylim(0, total_depth * 1.05)
		ax.set_ylabel('Depth (m)', fontsize=12, fontweight='bold')
		ax.set_title('Layer Profile & Water Table', fontsize=13, fontweight='bold')
		ax.set_xticks([])
		ax.grid(axis='y', alpha=0.3, linestyle='--')
		
		if water_table_depth is not None:
			ax.legend(loc='upper right', fontsize=9)
		
		return water_table_depth
	
	# In _plot_flux_recharge() — store conditions separately
	@staticmethod
	def _plot_flux_recharge(ax, agent):
		"""Plot flux/recharge time series."""
		flux_conditions = [fc for fc in agent.flow_conditions
						   if 'flux_data' in fc]
	
		if not flux_conditions:
			ax.text(0.5, 0.5, 'No flux/recharge conditions defined',
					ha='center', va='center', fontsize=12,
					transform=ax.transAxes,
					bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
			ax.set_title('Flux/Recharge Conditions', fontsize=13, fontweight='bold')
			ax.axis('off')
			return None
	
		colors = plt.cm.tab10(np.linspace(0, 1, len(flux_conditions)))
	
		# ✅ Store each condition separately — not merged
		conditions_list = []
	
		for fc, color in zip(flux_conditions, colors):
			flux_data  = fc['flux_data']
			times      = [val[0] for val in flux_data['values']]
			fluxes     = [val[1] for val in flux_data['values']]
	
			conditions_list.append({
				'name':   fc['name'],
				'times':  times,
				'fluxes': fluxes,
			})
	
			ax.step(times, fluxes, where='post', linewidth=2.5,
					color=color, label=fc['name'], alpha=0.8)
			ax.fill_between(times, 0, fluxes, step='post',
							alpha=0.2, color=color)
			ax.plot(times, fluxes, 'o', markersize=7, color=color,
					markeredgecolor='black', markeredgewidth=0.5)
	
		time_units = flux_conditions[0]['flux_data']['time_units']
		data_units = flux_conditions[0]['flux_data']['data_units']
	
		ax.set_xlabel(f"Time ({time_units})", fontsize=12, fontweight='bold')
		ax.set_ylabel(f"Flux ({data_units})", fontsize=12, fontweight='bold')
		ax.set_title('Flux/Recharge Conditions', fontsize=13, fontweight='bold')
		ax.grid(alpha=0.3, linestyle='--')
		ax.legend(loc='best', fontsize=9, framealpha=0.9)
	
		all_fluxes = [f for c in conditions_list for f in c['fluxes']]
		if all_fluxes:
			ax.set_ylim(bottom=0, top=max(all_fluxes) * 1.1)
	
		# ✅ Return structured list of conditions
		return {
			'conditions':  conditions_list,
			'time_units':  time_units,
			'data_units':  data_units,
		}
	
	@staticmethod
	def _create_plot_object(agent, water_table_depth, flux_data):
		"""Create ExperimentPlot object from agent."""
		# Get boundary condition names
		surface_bc = "N/A"
		bottom_bc = "N/A"
		
		for bc in agent.boundary_conditions:
			if bc['region'] == 'top':
				surface_bc = bc['flow_condition']
			elif bc['region'] == 'bottom':
				bottom_bc = bc['flow_condition']
		
		initial_cond = agent.flow_conditions[0]['name'] if agent.flow_conditions else "N/A"
		
		return ExperimentPlot(
			case_name=agent.case_name,
			layer_thicknesses=agent.layer_thicknesses,
			layer_names=agent.layer_order,
			initial_condition=initial_cond,
			surface_bc=surface_bc,
			bottom_bc=bottom_bc,
			total_depth=sum(agent.layer_thicknesses),
			water_table_depth=water_table_depth,
			flux_data=flux_data
		)


def compare_experiments(plot_objects: List[ExperimentPlot], 
                       save_path: Optional[str] = None, 
                       show: bool = False):
    """
    Create side-by-side comparison of multiple experiments.
    
    Args:
        plot_objects: List of ExperimentPlot objects
        save_path: Path to save figure (optional)
        show: Whether to display the plot (default: False)
    
    Returns:
        fig, axes: matplotlib figure and axes objects
    """
    n_exp = len(plot_objects)
    n_cols = min(4, n_exp)
    n_rows = (n_exp + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 6*n_rows))
    
    if n_exp == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_exp > 1 else axes
    
    fig.suptitle('Experiment Comparison', fontsize=16, fontweight='bold')
    
    for idx, plot_obj in enumerate(plot_objects):
        if n_exp == 1:
            ax = axes[0] if isinstance(axes, np.ndarray) else axes
        else:
            ax = axes[idx]
        
        # Draw layer column
        cmap = plt.cm.get_cmap('RdYlBu_r')
        colors = [cmap(i/len(plot_obj.layer_thicknesses)) 
                 for i in range(len(plot_obj.layer_thicknesses))]
        
        cumulative_depth = 0
        for i, (thickness, color) in enumerate(zip(plot_obj.layer_thicknesses, colors)):
            # Determine saturation
            if plot_obj.water_table_depth is not None:
                if cumulative_depth + thickness <= plot_obj.water_table_depth:
                    alpha = 0.9
                    edge_color = 'darkblue'
                    edge_width = 2
                elif cumulative_depth >= plot_obj.water_table_depth:
                    alpha = 0.4
                    edge_color = 'black'
                    edge_width = 1
                else:
                    alpha = 0.7
                    edge_color = 'blue'
                    edge_width = 2
            else:
                alpha = 0.7
                edge_color = 'black'
                edge_width = 1.5
            
            rect = mpatches.Rectangle((0, cumulative_depth), 1, thickness,
                                     facecolor=color, edgecolor=edge_color, 
                                     linewidth=edge_width, alpha=alpha)
            ax.add_patch(rect)
            
            layer_name = plot_obj.layer_names[i] if i < len(plot_obj.layer_names) else f"L{i+1}"
            ax.text(0.5, cumulative_depth + thickness/2, layer_name,
                   ha='center', va='center', fontsize=7, fontweight='bold')
            
            cumulative_depth += thickness
        
        # Draw water table if available
        if plot_obj.water_table_depth is not None:
            ax.axhline(y=plot_obj.water_table_depth, color='blue', 
                      linewidth=2, linestyle='--', alpha=0.7)
        
        # Add boundary condition labels
        ax.text(1.2, plot_obj.total_depth - 0.2, f"S: {plot_obj.surface_bc}",
               fontsize=7, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        ax.text(1.2, 0.2, f"B: {plot_obj.bottom_bc}",
               fontsize=7, va='bottom', ha='left',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        ax.set_xlim(-0.3, 2)
        ax.set_ylim(0, plot_obj.total_depth * 1.05)
        ax.set_ylabel('Depth (m)', fontsize=10, fontweight='bold')
        ax.set_title(plot_obj.case_name, fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_exp, len(axes) if isinstance(axes, np.ndarray) else 1):
        if isinstance(axes, np.ndarray):
            axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, axes


def compare_flux_conditions(plot_objects: List[ExperimentPlot],
                             save_path: Optional[str] = None,
                             show: bool = False):
    """Compare flux conditions across experiments."""
    flux_plots = [p for p in plot_objects if p.flux_data is not None]

    if not flux_plots:
        print("No flux conditions found to compare")
        return None, None

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Flux/Recharge Comparison Across Experiments',
                 fontsize=14, fontweight='bold')

    # ✅ Collect all unique conditions across all experiments
    seen        = {}   # name → (times, fluxes)
    time_units  = "y"
    data_units  = "cm/y"

    for plot_obj in flux_plots:
        if not plot_obj.flux_data:
            continue
        time_units = plot_obj.flux_data.get('time_units', 'y')
        data_units = plot_obj.flux_data.get('data_units', 'cm/y')

        for cond in plot_obj.flux_data.get('conditions', []):
            name = cond['name']
            if name not in seen:
                seen[name] = {
                    'times':  cond['times'],
                    'fluxes': cond['fluxes'],
                }

    if not seen:
        print("No flux conditions found")
        return None, None

    # ✅ Plot each unique condition once
    colors = plt.cm.tab10(np.linspace(0, 1, len(seen)))

    for (name, data), color in zip(seen.items(), colors):
        times  = data['times']
        fluxes = data['fluxes']

        ax.step(times, fluxes, where='post', linewidth=2.5,
                color=color, label=name, alpha=0.8)
        ax.fill_between(times, 0, fluxes, step='post',
                        alpha=0.15, color=color)
        ax.plot(times, fluxes, 'o', markersize=6,
                color=color, markeredgecolor='black',
                markeredgewidth=0.5)

    ax.set_xlabel(f"Time ({time_units})", fontsize=11, fontweight='bold')
    ax.set_ylabel(f"Flux ({data_units})", fontsize=11, fontweight='bold')
    ax.set_title('Recharge Flux Conditions', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Flux comparison plot saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig, ax

def plot_all_experiment_figures(agent, output_dir: str = "./", prefix: str = ""):
    """
    Generate overview plot for a single experiment.
    
    Args:
        agent: PFLOTRANInputAgent instance
        output_dir: Directory to save figures
        prefix: Optional prefix for filenames
    
    Returns:
        Dictionary with path to generated figure and plot object
    """
    os.makedirs(output_dir, exist_ok=True)
    
    figures = {}
    
    # Experiment overview (2 subplots: layer profile + flux)
    plot_obj = PFLOTRANPlotter.plot_experiment_overview(
        agent, 
        save_path=os.path.join(output_dir, f"{prefix}overview.png")
    )
    figures['overview'] = os.path.join(output_dir, f"{prefix}overview.png")
    
    print(f"✓ Generated overview plot for {agent.case_name}")
    
    return figures, plot_obj


if __name__ == "__main__":
    print("PFLOTRAN Plotting Module - Simplified")
    print("=" * 50)
    print("\nThis module provides streamlined plotting for PFLOTRAN experiments.")
    print("\nMain functions:")
    print("  - PFLOTRANPlotter.plot_experiment_overview(agent)")
    print("    → 2-panel plot: layer profile + flux conditions")
    print("\n  - compare_experiments(plot_objects)")
    print("    → Side-by-side comparison of multiple experiments")
    print("\n  - compare_flux_conditions(plot_objects)")
    print("    → Overlay flux conditions from multiple experiments")
    print("\n  - plot_all_experiment_figures(agent, output_dir)")
    print("    → Convenience function to generate all plots")