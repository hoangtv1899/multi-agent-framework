#!/usr/bin/env python3
"""
Experiment Builder - Converts experiment plans into PFLOTRANInputAgent instances
"""
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple
from pathlib import Path


class ExperimentBuilder:
	"""
	Builds PFLOTRAN simulation experiments from AI-generated experiment plans.
	Converts structured experiment plans into configured PFLOTRANInputAgent instances
	ready for simulation.
	"""
	
	def __init__(self, experiment_plan: Dict[str, Any]):
		"""
		Initialize builder with experiment plan.
		
		Args:
			experiment_plan: Dictionary containing complete experiment plan from planner
		"""
		self.plan = experiment_plan
		self.experiments = []
		
		# Cache parsed configurations
		self._domain_config = None
		self._grid_params = None
		self._materials = None
	
	@property
	def domain_config(self) -> Dict[str, Any]:
		"""Lazy load and cache domain configuration."""
		if self._domain_config is None:
			self._domain_config = self.plan.get('DOMAIN_CONFIGURATION', {})
		return self._domain_config
	
	@property
	def grid_params(self) -> Dict[str, Any]:
		"""Lazy load and cache parsed grid parameters."""
		if self._grid_params is None:
			self._grid_params = self._parse_grid_parameters()
		return self._grid_params
	
	@property
	def materials(self) -> List[Dict[str, Any]]:
		"""Lazy load and cache material properties."""
		if self._materials is None:
			self._materials = self.plan.get('MATERIAL_PROPERTIES_AND_CHARACTERISTIC_CURVES', [])
		return self._materials
	
	def _parse_grid_parameters(self) -> Dict[str, Any]:
		"""Parse and validate grid parameters from domain configuration."""
		nxyz = self.domain_config.get('nxyz', {})
		dxyz = self.domain_config.get('dxyz', {})
		
		# Parse grid counts
		nx = self._parse_fortran_value(nxyz.get('x', 1))
		ny = self._parse_fortran_value(nxyz.get('y', 1))
		nz = self._parse_fortran_value(nxyz.get('z', 13))
		
		# Parse cell dimensions
		dx = self._parse_fortran_value(dxyz.get('x', '1.0d0'))
		dy = self._parse_fortran_value(dxyz.get('y', '1.0d0'))
		
		# Parse Z layer thicknesses
		dz_list = dxyz.get('z', [])
		if isinstance(dz_list, list):
			layer_thicknesses = np.array([self._parse_fortran_value(dz) for dz in dz_list])
		else:
			# Handle uniform spacing
			layer_thicknesses = np.full(int(nz), self._parse_fortran_value(dz_list))
		
		return {
			'nx': int(nx),
			'ny': int(ny),
			'nz': int(nz),
			'dx': float(dx),
			'dy': float(dy),
			'layer_thicknesses': layer_thicknesses,
			'total_depth': float(np.sum(layer_thicknesses))
		}
	
	def build_experiments(self, pflotran_agent_class) -> List[Dict[str, Any]]:
		"""
		Build all experiment scenarios from the plan.
		
		Args:
			pflotran_agent_class: Your PFLOTRANInputAgent class
			
		Returns:
			List of experiment configurations with agents
		"""
		print("\n" + "="*70)
		print("BUILDING EXPERIMENTS FROM PLAN")
		print("="*70)
		
		# Extract configurations
		layer_regions = self.plan.get('LAYER_REGIONS_AND_COORDINATES', [])
		flow_conditions = self.plan.get('FLOW_CONDITIONS', {})
		condition_couplers = self.plan.get('CONDITIONS_COUPLERS', [])
		output_config = self.plan.get('OUTPUT', {})
		time_config = self.plan.get('TIME', {})
		
		# Display grid configuration
		self._print_grid_summary()
		
		print(f"\n📋 Materials: {len(self.materials)} defined")
		print(f"📍 Layers: {len(layer_regions)} regions")
		print(f"🌊 Flow Conditions: {len(flow_conditions)} defined")
		print(f"🧪 Experiments: {len(condition_couplers)} scenarios")
		
		# Build each experiment
		for idx, coupler in enumerate(condition_couplers, start=1):
			experiment = self._build_single_experiment(
				coupler, idx, len(condition_couplers),
				pflotran_agent_class, layer_regions,
				flow_conditions, time_config, output_config
			)
			self.experiments.append(experiment)
		
		print(f"\n✅ All {len(self.experiments)} experiments built successfully!")
		return self.experiments
	
	def _print_grid_summary(self):
		"""Print grid configuration summary."""
		params = self.grid_params
		print(f"\n📐 Grid Configuration:")
		print(f"   - Grid size: {params['nx']} × {params['ny']} × {params['nz']}")
		print(f"   - Cell dimensions: {params['dx']} × {params['dy']} m (horizontal)")
		print(f"   - Z-layer thicknesses: {len(params['layer_thicknesses'])} layers")
		print(f"   - Total depth: {params['total_depth']:.4f} m")
		
		# Show thickness distribution
		thicknesses = params['layer_thicknesses']
		print(f"   - Layer thickness range: [{thicknesses.min():.4f}, {thicknesses.max():.4f}] m")
	
	def _build_single_experiment(
		self,
		coupler: Dict[str, str],
		idx: int,
		total: int,
		pflotran_agent_class,
		layer_regions: List[Dict],
		flow_conditions: Dict,
		time_config: Dict,
		output_config: Dict
	) -> Dict[str, Any]:
		"""Build a single experiment configuration."""
		experiment_name = coupler.get('EXPERIMENT', f'experiment_{idx}')
		case_name = experiment_name.lower().replace(' ', '_')
		
		print(f"\n   [{idx}/{total}] {experiment_name}")
		
		# Create PFLOTRAN agent
		print(f"      • Creating PFLOTRAN agent...")
		params = self.grid_params
		agent = pflotran_agent_class(
			nx=params['nx'],
			ny=params['ny'],
			dx=params['dx'],
			dy=params['dy'],
			layer_thicknesses=params['layer_thicknesses'],
			case_name=case_name
		)
		
		# Setup components
		print(f"      • Setting up materials and layers...")
		self._setup_materials(agent, layer_regions)
		
		print(f"      • Adding material properties...")
		self._setup_material_properties(agent, layer_regions)
		
		print(f"      • Adding characteristic curves...")
		self._setup_characteristic_curves(agent, layer_regions)
		
		print(f"      • Applying initial condition...")
		initial_fc_name = coupler.get('INITIAL_CONDITION', 'hydrostatic_shallow')
		self._setup_initial_condition(agent, flow_conditions, initial_fc_name)
		
		print(f"      • Configuring flow conditions...")
		self._setup_flow_conditions(agent, flow_conditions)
		
		print(f"      • Setting time parameters...")
		self._setup_time_parameters(agent, time_config, output_config)
		
		print(f"      • Adding boundary conditions...")
		self._add_boundary_conditions(agent, coupler, flow_conditions)
		
		# Finalize agent setup
		agent.add_strata_from_layer_order()
		
		print(f"      ✅ Ready")
		
		# Create experiment record
		return {
			"scenario_index": idx - 1,
			"scenario_name": experiment_name,
			"case_name": case_name,
			"initial_condition": initial_fc_name,
			"surface_recharge": coupler.get('BOUNDARY_CONDITION_SURFACE_RECHARGE', 'unknown'),
			"deep_boundary": coupler.get('BOUNDARY_CONDITION_DEEP_BOUNDARY', 'unknown'),
			"pflotran_agent": agent
		}
	
	def _parse_fortran_value(self, value: Union[str, int, float]) -> float:
		"""
		Parse Fortran format numbers (e.g., 1.0d0, 5.0d-8).
		
		Args:
			value: Value in Fortran format or regular number
			
		Returns:
			Parsed float value
		"""
		if isinstance(value, (int, float)):
			return float(value)
		
		if isinstance(value, str):
			# Replace Fortran 'd' or 'D' with 'e' for scientific notation
			value_clean = value.strip().replace('d', 'e').replace('D', 'e')
			try:
				return float(value_clean)
			except ValueError:
				print(f"Warning: Could not parse '{value}', defaulting to 0.0")
				return 0.0
		
		return 0.0
	
	def _setup_materials(self, agent, layer_regions: List[Dict[str, Any]]):
		"""Setup materials and layer order from layer regions."""
		material_names = [region.get('layer_name', f'layer_{i}') 
						 for i, region in enumerate(layer_regions)]
		
		if len(material_names) > 6:
			display_names = f"{material_names[:3]} ... {material_names[-3:]}"
		else:
			display_names = material_names
			
		print(f"         Material order ({len(material_names)} layers): {display_names}")
		
		# Set layer order using agent method
		agent.set_layer_order(material_names)
	
	def _setup_material_properties(self, agent, layer_regions: List[Dict[str, Any]]):
		"""
		Setup material properties for each layer based on material IDs.
		Maps layer regions to material properties from the JSON.
		"""
		for idx, region in enumerate(layer_regions):
			layer_name = region.get('layer_name', f'layer_{idx}')
			
			# Material ID corresponds to the layer index + 1
			# (assuming materials are numbered 1-13 for layers 1-13)
			material_id = idx + 1
			
			# Find material properties with this ID
			material_props = self._get_material_by_id(material_id)
			
			if material_props:
				# Extract properties
				porosity = self._parse_fortran_value(material_props.get('POROSITY', '0.25d0'))
				
				# Get permeability
				perm_data = material_props.get('PERMEABILITY', {})
				if 'PERM_ISO' in perm_data:
					permeability = self._parse_fortran_value(perm_data['PERM_ISO'])
				else:
					permeability = 1.0e-12  # default
				
				# Add material property to agent
				material = {
					"name": layer_name,
					"id": material_id,
					"porosity": porosity,
					"permeability": permeability
				}
				
				agent.add_material_property(material)
	
	def _setup_characteristic_curves(self, agent, layer_regions: List[Dict[str, Any]]):
		"""
		Setup characteristic curves for each layer.
		Extracts Van Genuchten parameters from material properties.
		"""
		for idx, region in enumerate(layer_regions):
			layer_name = region.get('layer_name', f'layer_{idx}')
			material_id = idx + 1
			
			# Find material properties with this ID
			material_props = self._get_material_by_id(material_id)
			
			if material_props:
				# Extract characteristic curve parameters
				char_curve = material_props.get('CHARACTERISTIC_CURVE', {})
				perm_func = material_props.get('PERMEABILITY_FUNCTION', {})
				
				saturation_function = char_curve.get('SATURATION_FUNCTION', 'VAN_GENUCHTEN')
				
				# Van Genuchten parameters
				alpha = self._parse_fortran_value(char_curve.get('ALPHA', '1.0d-4'))
				m = self._parse_fortran_value(char_curve.get('M', '0.5d0'))
				sr_liquid = self._parse_fortran_value(
					char_curve.get('LIQUID_RESIDUAL_SATURATION', '0.1d0')
				)
				
				# Permeability function parameters
				perm_m = self._parse_fortran_value(perm_func.get('M', m))
				perm_sr = self._parse_fortran_value(
					perm_func.get('LIQUID_RESIDUAL_SATURATION', sr_liquid)
				)
				
				# Add characteristic curve to agent
				curve = {
					"name": layer_name,
					"saturation_function": saturation_function,
					"alpha": alpha,
					"m": m,
					"liquid_residual_saturation": sr_liquid,
					"permeability_function_type": perm_func.get('TYPE', 'MUALEM_VG_LIQ'),
					"permeability_m": perm_m,
					"permeability_liquid_residual_saturation": perm_sr
				}
				
				agent.add_characteristic_curve(curve)
	
	def _get_material_by_id(self, material_id: int) -> Optional[Dict[str, Any]]:
		"""
		Get material properties by ID.
		
		Args:
			material_id: Material ID number
			
		Returns:
			Material properties dictionary or None if not found
		"""
		for material in self.materials:
			if material.get('ID') == material_id:
				return material
		return None
	
	def _setup_initial_condition(
		self,
		agent,
		flow_conditions: Dict[str, Dict],
		fc_name: str
	):
		"""Setup initial condition using flow condition from plan."""
		initial_fc = flow_conditions.get(fc_name, {})
		
		if not initial_fc:
			print(f"         Warning: Initial condition '{fc_name}' not found, using default")
			initial_condition = {
				"name": "initial",
				"type": "LIQUID_PRESSURE HYDROSTATIC",
				"datum": (0.0, 0.0, 5.0),
				"liquid_pressure": 101325.0,
			}
		else:
			# Parse datum and pressure
			datum = self._parse_datum(initial_fc.get('DATUM', '0.0d0 0.0d0 5.0d0'))
			pressure = self._parse_fortran_value(initial_fc.get('LIQUID_PRESSURE', '101325.0d0'))
			
			initial_condition = {
				"name": "initial",
				"type": "LIQUID_PRESSURE HYDROSTATIC",
				"datum": datum,
				"liquid_pressure": pressure,
			}
		
		# Replace default initial condition
		agent.flow_conditions[0] = initial_condition
	
	def _parse_datum(self, datum_str: str) -> Tuple[float, float, float]:
		"""Parse DATUM string into tuple of floats."""
		parts = datum_str.split()
		if len(parts) >= 3:
			return (
				self._parse_fortran_value(parts[0]),
				self._parse_fortran_value(parts[1]),
				self._parse_fortran_value(parts[2])
			)
		return (0.0, 0.0, 5.0)
	
	def _setup_flow_conditions(self, agent, flow_conditions: Dict[str, Dict]):
		"""Setup flow conditions from planner specifications."""
		for fc_name, fc_data in flow_conditions.items():
			fc_type = fc_data.get('TYPE', '')
			
			# Skip initial conditions and boundary hydrostatic conditions
			# (they're handled separately)
			if self._should_skip_flow_condition(fc_name, fc_type):
				continue
			
			# Add flux/recharge conditions
			if fc_type == 'FLUX' or \
				fc_type == 'LIQUID_FLUX NEUMANN' \
				or 'recharge' in fc_name.lower():
				self._add_flux_condition(agent, fc_name, fc_data)
	
	def _should_skip_flow_condition(self, fc_name: str, fc_type: str) -> bool:
		"""Determine if a flow condition should be skipped during setup."""
		# Skip hydrostatic conditions (used for initial/boundary conditions)
		if fc_type == 'HYDROSTATIC':
			return True
		return False
	
	# ✅ Fix — handle both formats
	def _add_flux_condition(self, agent, fc_name, fc_data):
		fc_type = fc_data.get('TYPE', '')
	
		# Format A: LIQUID_FLUX NEUMANN with direct value
		if 'LIQUID_FLUX' in fc_data and 'FLUX_LIST' not in fc_data:
			flux_str  = str(fc_data['LIQUID_FLUX']).split()[0]  # "22.65d0 cm/y" → "22.65d0"
			flux_cm_y = self._parse_fortran_value(flux_str)
			values    = [(0.0, flux_cm_y), (10.0, flux_cm_y)]   # constant over time
	
		# Format B: FLUX_LIST with time-varying values
		else:
			flux_list  = fc_data.get('FLUX_LIST', [])
			time_units = fc_data.get('TIME_UNITS', 'y').lower()
			data_units = fc_data.get('DATA_UNITS', 'cm/y')
			values     = []
			for entry in flux_list:
				time      = self._parse_fortran_value(entry.get('TIME', 0.0))
				flux      = self._parse_fortran_value(entry.get('FLUX', 0.0))
				time_yr   = self._convert_time_to_years(time, time_units)
				flux_cm_y = self._convert_flux_to_cm_per_year(flux, data_units)
				values.append((time_yr, flux_cm_y))
	
		flow_condition = {
			"name":      fc_name,
			"type":      "LIQUID_FLUX NEUMANN",
			"flux_data": {
				"time_units": "y",
				"data_units": "cm/y",
				"values":     values,
			},
		}
		agent.add_flow_condition(flow_condition)
	
	def _convert_time_to_years(self, time: float, units: str) -> float:
		"""Convert time to years based on units."""
		if units == 'd':
			return time / 365.25
		elif units == 'y':
			return time
		elif units == 'h':
			return time / (365.25 * 24)
		elif units == 's':
			return time / (365.25 * 24 * 3600)
		return time
	
	def _convert_flux_to_cm_per_year(self, flux: float, units: str) -> float:
		"""Convert flux to cm/year based on units."""
		if units == 'm/s':
			# m/s to cm/year: multiply by 100 (m→cm) * 86400 (s→d) * 365.25 (d→y)
			return flux * 100 * 86400 * 365.25
		elif units == 'cm/y':
			return flux
		elif units == 'm/y':
			return flux * 100
		elif units == 'mm/y':
			return flux / 10
		return flux
	
	def _add_boundary_conditions(
		self,
		agent,
		coupler: Dict[str, str],
		flow_conditions: Dict[str, Dict]
	):
		"""Add boundary conditions from coupler specification."""
		# Surface recharge boundary
		surface_recharge_name = coupler.get(
			'BOUNDARY_CONDITION_SURFACE',
			'recharge_baseline'
		)
		agent.add_boundary_condition('surface_recharge', surface_recharge_name, 'top')
		
		# Deep boundary
		deep_boundary_name = coupler.get(
			'BOUNDARY_CONDITION_DEEP_BOUNDARY',
			'bottom_hydrostatic'
		)
		
		# Add deep boundary flow condition if it's hydrostatic
		deep_fc = flow_conditions.get(deep_boundary_name, {})
		if deep_fc and deep_fc.get('TYPE') == 'HYDROSTATIC':
			datum = self._parse_datum(deep_fc.get('DATUM', '0.0d0 0.0d0 0.5d0'))
			pressure = self._parse_fortran_value(deep_fc.get('LIQUID_PRESSURE', '101325.0d0'))
			
			boundary_condition = {
				"name": deep_boundary_name,
				"type": "LIQUID_PRESSURE HYDROSTATIC",
				"datum": datum,
				"liquid_pressure": pressure,
			}
			agent.add_flow_condition(boundary_condition)
		
		agent.add_boundary_condition('deep_boundary', deep_boundary_name, 'bottom')
	
	def _setup_time_parameters(
		self,
		agent,
		time_config: Dict[str, Any],
		output_config: Dict[str, Any]
	):
		"""Setup time parameters from plan."""
		# Parse time configuration
		final_time = self._parse_fortran_value(time_config.get('FINAL_TIME', '10.0d0'))
		final_time_units = time_config.get('FINAL_TIME_UNITS', 'y').lower()
		
		initial_ts = self._parse_fortran_value(time_config.get('INITIAL_TIMESTEP_SIZE', '1.0d-6'))
		initial_ts_units = time_config.get('INITIAL_TIMESTEP_SIZE_UNITS', 'd').lower()
		
		max_ts = self._parse_fortran_value(time_config.get('MAXIMUM_TIMESTEP_SIZE', '0.05d0'))
		max_ts_units = time_config.get('MAXIMUM_TIMESTEP_SIZE_UNITS', 'y').lower()
		
		# Convert final time to years if needed
		final_time_years = self._convert_time_to_years(final_time, final_time_units)
		
		# Update agent time configuration
		agent.time_config.update({
			"final_time": final_time_years,
			"final_time_units": "y",
			"initial_timestep": initial_ts,
			"initial_timestep_units": initial_ts_units,
			"maximum_timestep": max_ts,
			"maximum_timestep_units": max_ts_units
		})
		
		# Parse and convert output times
		output_times = self._parse_output_times(output_config)
		agent.output_config.update({
			"times": output_times,
			"time_units": "y",
			"format": output_config.get('FORMAT', 'TECPLOT POINT')
		})
	
	def _parse_output_times(self, output_config: Dict[str, Any]) -> List[float]:
		"""Parse and convert output times to years."""
		output_times_raw = output_config.get('TIMES', [])
		output_units = output_config.get('TIME_UNITS', 'y').lower()
		
		output_times = []
		for time_val in output_times_raw:
			time_parsed = self._parse_fortran_value(time_val)
			time_years = self._convert_time_to_years(time_parsed, output_units)
			output_times.append(time_years)
		
		return sorted(output_times)  # Ensure chronological order
	
	def prepare_cases(
		self,
		output_directory: str,
		elm_data: Optional[Any] = None
	) -> List[str]:
		"""
		Prepare all simulation cases (create directories and input files).
		
		Args:
			output_directory: Base directory for all cases
			elm_data: Optional ELMSoilData instance for material properties
			
		Returns:
			List of prepared case directories
		"""
		print(f"\n" + "="*70)
		print("PREPARING SIMULATION CASES")
		print("="*70)
		
		output_path = Path(output_directory)
		output_path.mkdir(parents=True, exist_ok=True)
		
		case_dirs = []
		for experiment in self.experiments:
			case_name = experiment['case_name']
			agent = experiment['pflotran_agent']
			
			print(f"\n📁 {experiment['scenario_name']}")
			print(f"   Initial: {experiment['initial_condition']}")
			print(f"   Recharge: {experiment['surface_recharge']}")
			print(f"   Deep BC: {experiment['deep_boundary']}")
			
			# Prepare case (creates directory and generates input file)
			case_dir = agent.prepare_case(output_dir=str(output_path), elm_data=elm_data)
			case_dirs.append(case_dir)
			experiment['case_dir'] = case_dir
			
			# Print input file location
			input_file = f"{agent.case_name}.in"
			print(f"   → {input_file} in {Path(case_dir).name}")
		
		print(f"\n✅ All {len(case_dirs)} cases prepared in {output_path}")
		return case_dirs
	
	def get_experiment_summary(self) -> Dict[str, Any]:
		"""Get summary of all experiments."""
		summary = {
			"total_experiments": len(self.experiments),
			"grid_configuration": self.grid_params,
			"experiments": []
		}
		
		for exp in self.experiments:
			summary["experiments"].append({
				"scenario_index": exp['scenario_index'],
				"scenario_name": exp['scenario_name'],
				"case_name": exp['case_name'],
				"case_directory": exp.get('case_dir', 'not_prepared_yet'),
				"initial_condition": exp['initial_condition'],
				"surface_recharge": exp['surface_recharge'],
				"deep_boundary": exp['deep_boundary']
			})
		
		return summary
	
	def print_summary(self):
		"""Print a formatted summary of all experiments."""
		summary = self.get_experiment_summary()
		
		print("\n" + "="*70)
		print("EXPERIMENT SUMMARY")
		print("="*70)
		
		grid = summary['grid_configuration']
		print(f"\n📐 Grid: {grid['nx']}×{grid['ny']}×{grid['nz']} cells, "
			  f"Depth: {grid['total_depth']:.2f} m")
		
		print(f"\n🧪 Total Experiments: {summary['total_experiments']}")
		
		for exp in summary['experiments']:
			print(f"\n   [{exp['scenario_index'] + 1}] {exp['scenario_name']}")
			print(f"       Case: {exp['case_name']}")
			print(f"       Initial: {exp['initial_condition']}")
			print(f"       Surface BC: {exp['surface_recharge']}")
			print(f"       Bottom BC: {exp['deep_boundary']}")
			if exp['case_directory'] != 'not_prepared_yet':
				print(f"       Directory: {Path(exp['case_directory']).name}")