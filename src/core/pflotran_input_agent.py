#!/usr/bin/env python3
"""
PFLOTRAN Input Generator and Simulation Runner
"""
import os
import subprocess
import numpy as np
from elm_soildata import ELMSoilData, fortran_format

# Import plotting utilities
try:
    from pflotran_plotting import PFLOTRANPlotter, plot_all_experiment_figures
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: pflotran_plotting module not found. Plotting functions will not be available.")


class PFLOTRANInputAgent:
	"""
	Generates PFLOTRAN input files and manages simulation execution.
	Handles grid configuration, regions, flow conditions, and material properties.
	"""
	
	def __init__(self, nx, ny, dx, dy, layer_thicknesses, case_name="pflotran_case"):
		"""
		Initialize PFLOTRAN input agent.
		
		Args:
			nx: Number of cells in x-direction
			ny: Number of cells in y-direction
			dx: Cell size in x-direction (meters)
			dy: Cell size in y-direction (meters)
			layer_thicknesses: Thickness of layers from bottom to top (meters)
			case_name: Name for the simulation case
		"""
		self.nx = nx
		self.ny = ny
		self.dx = dx
		self.dy = dy
		self.layer_thicknesses = layer_thicknesses
		self.case_name = case_name
		self.case_dir = None
		
		total_height = sum(layer_thicknesses)
		
		self.grid_config = {
			"type": "STRUCTURED",
			"dimensions": [nx, ny, len(layer_thicknesses)],
			"thicknesses": layer_thicknesses,
		}
		
		self.layer_order = []
		self.layer_regions = []
		
		self.required_regions = [
			{"name": "all", "coords": [0.0, 0.0, 0.0, dx*nx, dy*ny, total_height]},
			{"name": "top", "coords": [0.0, 0.0, total_height, dx*nx, dy*ny, total_height], "face": "TOP"},
			{"name": "bottom", "coords": [0.0, 0.0, 0.0, dx*nx, dy*ny, 0.0], "face": "BOTTOM"},
		]
		
		self.regions = self.required_regions.copy()
		
		self.flow_conditions = [
			{
				"name": "initial",
				"type": "LIQUID_PRESSURE HYDROSTATIC",
				"datum": (0.0, 0.0, total_height / 2.0),
				"liquid_pressure": 101325.0,
			},
		]
		
		self.initial_conditions = []
		self.boundary_conditions = []
		self.strata = []
		
		# Material properties and characteristic curves
		self.material_properties = []
		self.characteristic_curves = []
		
		self.time_config = {
			"final_time": 10.0,
			"final_time_units": "y",
			"initial_timestep": 1.0,
			"initial_timestep_units": "h",
			"maximum_timestep": 0.1,
			"maximum_timestep_units": "y"
		}
		
		self.output_config = {
			"times": [0.5, 1.0, 2.0, 5.0, 10.0],
			"time_units": "y",
			"format": "TECPLOT POINT"
		}
	
	def set_layer_order(self, material_names):
		"""Set material order from bottom to top and create regions."""
		self.layer_order = material_names.copy()
		layer_bottoms, layer_tops = self.compute_layer_depths()
		
		self.layer_regions = [
			{
				"name": mat_name,
				"coords": [0.0, 0.0, layer_bottoms[i], self.dx*self.nx, self.dy*self.ny, layer_tops[i]]
			}
			for i, mat_name in enumerate(material_names)
		]
		
		self.regions = self.required_regions + self.layer_regions
	
	def add_strata_from_layer_order(self):
		"""Add strata based on layer order."""
		self.strata = [{"region": mat_name, "material": mat_name} for mat_name in self.layer_order]
	
	def compute_layer_depths(self):
		"""Compute cumulative layer depths (bottom to top)."""
		bottom_depths = [0.0]
		for thickness in self.layer_thicknesses:
			bottom_depths.append(bottom_depths[-1] + thickness)
		return bottom_depths[:-1], bottom_depths[1:]
	
	def add_flow_condition(self, condition):
		"""Add a flow condition."""
		self.flow_conditions.append(condition)
	
	def add_initial_condition(self, flow_condition_name, region_name):
		"""Add initial condition coupling."""
		self.initial_conditions.append({"flow_condition": flow_condition_name, "region": region_name})
	
	def add_boundary_condition(self, name, flow_condition_name, region_name):
		"""Add boundary condition coupling."""
		self.boundary_conditions.append({
			"name": name,
			"flow_condition": flow_condition_name,
			"region": region_name
		})
	
	def add_strata(self, region_name, material_name):
		"""Add strata (material-region coupling)."""
		self.strata.append({"region": region_name, "material": material_name})
	
	def add_material_property(self, material):
		"""
		Add material property.
		
		Args:
			material: Dictionary with keys 'name', 'id', 'porosity', 'permeability'
		"""
		self.material_properties.append(material)
	
	def add_characteristic_curve(self, curve):
		"""
		Add characteristic curve.
		
		Args:
			curve: Dictionary with Van Genuchten parameters
		"""
		self.characteristic_curves.append(curve)
	
	def prepare_case(self, output_dir="./", elm_data=None):
		"""Prepare simulation case directory and generate input file."""
		self.case_dir = os.path.join(output_dir, self.case_name)
		os.makedirs(self.case_dir, exist_ok=True)
		self.elm_data = elm_data
		self._regenerate_input()
		return self.case_dir
	
	def _regenerate_input(self):
		"""Internal method to regenerate input file."""
		input_file = os.path.join(self.case_dir, f"{self.case_name}.in")
		self._write_input_file(input_file, self.elm_data)
	
	def update_input_file(self):
		"""
		Regenerate input file after modifications.
		Use this after changing materials, flow conditions, or boundaries.
		"""
		if not self.case_dir:
			raise RuntimeError("Case not prepared. Call prepare_case() first.")
		self._regenerate_input()
		print(f"✓ Input file regenerated with updated configuration")
	
	def run_simulation(self, pflotran_exe="pflotran", timeout=None):
		"""
		Execute PFLOTRAN simulation.
		
		Args:
			pflotran_exe: Path to PFLOTRAN executable
			timeout: Maximum runtime in seconds
			
		Returns:
			CompletedProcess object with returncode, stdout, stderr
		"""
		if not self.case_dir:
			raise RuntimeError("Case not prepared. Call prepare_case() first.")
		
		input_file = f"{self.case_name}.in"
		
		try:
			result = subprocess.run(
				[pflotran_exe, "-pflotranin", input_file],
				cwd=self.case_dir,
				capture_output=True,
				text=True,
				timeout=timeout
			)
			
			if result.returncode == 0:
				print(f"✓ Simulation completed successfully in {self.case_dir}")
			else:
				print(f"✗ Simulation failed with return code {result.returncode}")
				if result.stderr:
					print(f"STDERR: {result.stderr}")
			
			return result
			
		except subprocess.TimeoutExpired:
			print(f"✗ Simulation timed out after {timeout} seconds")
			raise
		except FileNotFoundError:
			print(f"✗ PFLOTRAN executable not found: {pflotran_exe}")
			raise
	
	# ============================================================================
	# Plotting Methods (delegated to PFLOTRANPlotter)
	# ============================================================================
	
	def plot_experiment_overview(self, save_path=None, show=False):
		"""
		Plot overview of experiment configuration.
		
		Args:
			save_path: Path to save figure (optional)
			show: Whether to display the plot (default: False)
		
		Returns:
			ExperimentPlot object containing plot data for comparison
		"""
		if not PLOTTING_AVAILABLE:
			print("Error: Plotting module not available")
			return None
		
		return PFLOTRANPlotter.plot_experiment_overview(self, save_path, show)
	
	def plot_flux_recharge(self, save_path=None, show=False):
		"""
		Plot flux/recharge time series.
		
		Args:
			save_path: Path to save figure (optional)
			show: Whether to display the plot (default: False)
		
		Returns:
			fig, axes: matplotlib figure and axes objects
		"""
		if not PLOTTING_AVAILABLE:
			print("Error: Plotting module not available")
			return None, None
		
		return PFLOTRANPlotter.plot_flux_recharge(self, save_path, show)
	
	def plot_water_table(self, save_path=None, show=False):
		"""
		Plot water table depth and position.
		
		Args:
			save_path: Path to save figure (optional)
			show: Whether to display the plot (default: False)
		
		Returns:
			fig, axes: matplotlib figure and axes objects
		"""
		if not PLOTTING_AVAILABLE:
			print("Error: Plotting module not available")
			return None, None
		
		return PFLOTRANPlotter.plot_water_table(self, save_path, show)
	
	def plot_all_figures(self, output_dir="./", prefix=""):
		"""
		Generate all available plots for this experiment.
		
		Args:
			output_dir: Directory to save figures
			prefix: Optional prefix for filenames
		
		Returns:
			Dictionary with paths to all generated figures and ExperimentPlot object
		"""
		if not PLOTTING_AVAILABLE:
			print("Error: Plotting module not available")
			return {}, None
		
		return plot_all_experiment_figures(self, output_dir, prefix)
	
	# ============================================================================
	# Internal Methods for Writing Input File
	# ============================================================================
	
	def _write_input_file(self, filepath, elm_data=None):
		"""Write complete PFLOTRAN input file (internal method)."""
		with open(filepath, 'w') as f:
			self._write_simulation(f)
			self._write_numerical_methods(f)
			self._write_regression(f)
			self._write_discretization(f)
			
			if elm_data:
				elm_data.write_material_properties(file_handle=f)
				elm_data.write_characteristic_curves(file_handle=f)
			else:
				if self.material_properties:
					self._write_material_properties(f)
				if self.characteristic_curves:
					self._write_characteristic_curves(f)
			
			self._write_output(f)
			self._write_time(f)
			self._write_regions(f)
			self._write_flow_conditions(f)
			self._write_initial_conditions(f)
			self._write_boundary_conditions(f)
			self._write_strata(f)
			f.write("END_SUBSURFACE\n")
	
	def _write_simulation(self, f):
		"""Write SIMULATION section."""
		f.write("#Description: 1D variably saturated flow problem with ELM-compatible layering\n")
		f.write("SIMULATION\n")
		f.write("  SIMULATION_TYPE SUBSURFACE\n")
		f.write("  PROCESS_MODELS\n")
		f.write("    SUBSURFACE_FLOW flow\n")
		f.write("      MODE RICHARDS\n")
		f.write("    /\n")
		f.write("  /\n")
		f.write("END\n\n")
		f.write("SUBSURFACE\n\n")
	
	def _write_numerical_methods(self, f):
		"""Write NUMERICAL_METHODS section."""
		f.write("#=========================== numerical methods ================================\n")
		f.write("NUMERICAL_METHODS FLOW\n")
		f.write("  LINEAR_SOLVER\n")
		f.write("    SOLVER DIRECT\n")
		f.write("  /\n")
		f.write("END\n\n")
	
	def _write_regression(self, f):
		"""Write REGRESSION section."""
		f.write("#=========================== regression =======================================\n")
		f.write("REGRESSION\n")
		f.write(f"  CELLS_PER_PROCESS {len(self.layer_thicknesses)}\n")
		f.write("END\n\n")
	
	def _write_discretization(self, f):
		"""Write GRID discretization section."""
		f.write("#=========================== discretization ===================================\n")
		f.write("GRID\n")
		f.write("  TYPE STRUCTURED\n")
		f.write(f"  NXYZ {self.grid_config['dimensions'][0]} {self.grid_config['dimensions'][1]} {self.grid_config['dimensions'][2]}\n")
		f.write("  DXYZ\n")
		f.write(f"    {fortran_format(self.dx)}\n")
		f.write(f"    {fortran_format(self.dy)}\n")
		thickness_str = ' '.join([fortran_format(t) for t in self.grid_config['thicknesses']])
		f.write(f"    {thickness_str}\n")
		f.write("  /\n")
		f.write("END\n\n")
	
	def _write_material_properties(self, f):
		"""Write MATERIAL_PROPERTY section."""
		f.write("#=========================== material properties ==============================\n")
		
		for material in self.material_properties:
			f.write(f"MATERIAL_PROPERTY {material['name']}\n")
			f.write(f"  ID {material.get('id', 1)}\n")
			f.write(f"  POROSITY {fortran_format(material['porosity'])}\n")
			
			perm = material['permeability']
			if isinstance(perm, dict):
				if 'PERM_X' in perm:
					f.write(f"  PERMEABILITY\n")
					f.write(f"    PERM_X {fortran_format(perm['PERM_X'])}\n")
					f.write(f"    PERM_Y {fortran_format(perm.get('PERM_Y', perm['PERM_X']))}\n")
					f.write(f"    PERM_Z {fortran_format(perm.get('PERM_Z', perm['PERM_X']))}\n")
					f.write(f"  /\n")
				else:
					f.write(f"  PERMEABILITY\n")
					f.write(f"    PERM_ISO {fortran_format(perm.get('PERM_ISO', 1.0e-12))}\n")
					f.write(f"  /\n")
			else:
				f.write(f"  PERMEABILITY\n")
				f.write(f"    PERM_ISO {fortran_format(perm)}\n")
				f.write(f"  /\n")
			
			f.write(f"  CHARACTERISTIC_CURVES {material['name']}\n")
			f.write("END\n\n")
	
	def _write_characteristic_curves(self, f):
		"""Write CHARACTERISTIC_CURVES section."""
		f.write("#=========================== characteristic curves ============================\n")
		
		for curve in self.characteristic_curves:
			f.write(f"CHARACTERISTIC_CURVES {curve['name']}\n")
			
			sat_func = curve.get('saturation_function', 'VAN_GENUCHTEN')
			f.write(f"  SATURATION_FUNCTION {sat_func}\n")
			
			if sat_func == 'VAN_GENUCHTEN':
				f.write(f"    ALPHA {fortran_format(curve.get('alpha', 1.0e-4))}\n")
				f.write(f"    M {fortran_format(curve.get('m', 0.5))}\n")
				f.write(f"    LIQUID_RESIDUAL_SATURATION {fortran_format(curve.get('liquid_residual_saturation', 0.1))}\n")
			
			f.write("  /\n")
			
			perm_func_type = curve.get('permeability_function_type', 'MUALEM_VG_LIQ')
			f.write(f"  PERMEABILITY_FUNCTION {perm_func_type}\n")
			
			if 'MUALEM' in perm_func_type:
				f.write(f"    M {fortran_format(curve.get('permeability_m', curve.get('m', 0.5)))}\n")
				f.write(f"    LIQUID_RESIDUAL_SATURATION {fortran_format(curve.get('permeability_liquid_residual_saturation', curve.get('liquid_residual_saturation', 0.1)))}\n")
			
			f.write("  /\n")
			f.write("END\n\n")
	
	def _write_output(self, f):
		"""Write OUTPUT section."""
		f.write("#=========================== output options ===================================\n")
		f.write("OUTPUT\n")
		times_str = ' '.join([fortran_format(t) for t in self.output_config['times']])
		f.write(f"  TIMES {self.output_config['time_units']} {times_str}\n")
		f.write(f"  FORMAT {self.output_config['format']}\n")
		f.write("END\n\n")
	
	def _write_time(self, f):
		"""Write TIME section."""
		f.write("#=========================== times ============================================\n")
		f.write("TIME\n")
		f.write(f"  FINAL_TIME {fortran_format(self.time_config['final_time'])} {self.time_config['final_time_units']}\n")
		f.write(f"  INITIAL_TIMESTEP_SIZE {fortran_format(self.time_config['initial_timestep'])} {self.time_config['initial_timestep_units']}\n")
		f.write(f"  MAXIMUM_TIMESTEP_SIZE {fortran_format(self.time_config['maximum_timestep'])} {self.time_config['maximum_timestep_units']}\n")
		f.write("END\n\n")
	
	def _write_regions(self, f):
		"""Write REGION section."""
		f.write("#=========================== regions ==========================================\n")
		for region in self.regions:
			f.write(f"REGION {region['name']}\n")
			if "face" in region:
				f.write(f"  FACE {region['face']}\n")
			f.write("  COORDINATES\n")
			
			def format_coord(val):
				if abs(val - round(val)) < 1e-10:
					return f"{int(round(val))}.d0"
				else:
					return f"{val:.3f}d0"
			
			coords = region['coords']
			f.write(f"    {format_coord(coords[0])} {format_coord(coords[1])} {format_coord(coords[2])}\n")
			f.write(f"    {format_coord(coords[3])} {format_coord(coords[4])} {format_coord(coords[5])}\n")
			f.write("  /\n")
			f.write("END\n\n")
	
	def _write_flow_conditions(self, f):
		"""Write FLOW_CONDITIONS section."""
		f.write("#=========================== flow conditions ==================================\n")
		for condition in self.flow_conditions:
			f.write(f"FLOW_CONDITION {condition['name']}\n")
			f.write("  TYPE\n")
			f.write(f"    {condition['type']}\n")
			f.write("  /\n")
			
			if condition['name'] == "initial" or "HYDROSTATIC" in condition['type']:
				f.write(f"  DATUM {fortran_format(condition['datum'][0])} {fortran_format(condition['datum'][1])} {fortran_format(condition['datum'][2])}\n")
				f.write(f"  LIQUID_PRESSURE {fortran_format(condition['liquid_pressure'])}\n")
			elif "flux_data" in condition:
				flux_data = condition["flux_data"]
				f.write(f"  LIQUID_FLUX LIST\n")
				f.write(f"    TIME_UNITS {flux_data['time_units']}\n")
				f.write(f"    DATA_UNITS {flux_data['data_units']}\n")
				for time, flux in flux_data["values"]:
					f.write(f"    {fortran_format(time)}    {fortran_format(flux)}\n")
				f.write("  /\n")
			
			f.write("END\n\n")
	
	def _write_initial_conditions(self, f):
		"""Write INITIAL_CONDITION section."""
		f.write("#=========================== condition couplers ===============================\n")
		
		if not self.initial_conditions:
			f.write("INITIAL_CONDITION\n")
			f.write("  FLOW_CONDITION initial\n")
			f.write("  REGION all\n")
			f.write("END\n\n")
		else:
			for ic in self.initial_conditions:
				f.write("INITIAL_CONDITION\n")
				f.write(f"  FLOW_CONDITION {ic['flow_condition']}\n")
				f.write(f"  REGION {ic['region']}\n")
				f.write("END\n\n")
	
	def _write_boundary_conditions(self, f):
		"""Write BOUNDARY_CONDITION section."""
		for bc in self.boundary_conditions:
			f.write(f"BOUNDARY_CONDITION {bc['name']}\n")
			f.write(f"  FLOW_CONDITION {bc['flow_condition']}\n")
			f.write(f"  REGION {bc['region']}\n")
			f.write("END\n\n")
	
	def _write_strata(self, f):
		"""Write STRATA section."""
		f.write("#=========================== stratigraphy couplers ============================\n")
		for stratum in self.strata:
			f.write("STRATA\n")
			f.write(f"  REGION {stratum['region']}\n")
			f.write(f"  MATERIAL {stratum['material']}\n")
			f.write("END\n\n")
	

if __name__ == "__main__":
    print("PFLOTRAN Input Agent")
    print("=" * 50)
    print("\nThis module provides tools for generating PFLOTRAN input files.")
    print("\nUsage:")
    print("  from pflotran_input_agent import PFLOTRANInputAgent")
    print("\n  agent = PFLOTRANInputAgent(nx=1, ny=1, dx=1.0, dy=1.0,")
    print("                             layer_thicknesses=[1.0, 2.0, 3.0],")
    print("                             case_name='my_case')")
    print("\n  agent.prepare_case(output_dir='./cases')")
    print("  agent.plot_experiment_overview(save_path='overview.png')")
    print("  agent.run_simulation()")