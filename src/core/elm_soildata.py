#!/usr/bin/env python3
"""
ELM Soil Data Parser and PFLOTRAN Material Property Generator
"""
import xarray as xr
import numpy as np


def fortran_format(value, decimal_places=6):
    """
    Convert scientific notation to PFLOTRAN Fortran format.
    
    Arguments:
        value (float): The numerical value
        decimal_places (int): Number of decimal places
    
    Returns:
        str: Properly formatted Fortran string (e.g., 1.234567d-13)
    """
    # Check for zero
    if value == 0.0 or abs(value) < 1e-99:
        return "0.d0"
    
    # For integers
    if abs(value - round(value)) < 1e-12 and abs(value) >= 1.0:
        return f"{int(round(value))}.d0"
    
    # For "normal" range numbers (0.001 to 999)
    if 0.001 <= abs(value) < 1000:
        # Check if it's a simple decimal
        if abs(value - round(value, 3)) < 1e-10:
            return f"{value:.3f}d0"
        elif abs(value - round(value, 6)) < 1e-10:
            return f"{value:.6f}d0"
        else:
            return f"{value:.6f}d0"
    
    # For very small or very large numbers, use scientific notation
    exp = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10 ** exp)
    
    return f"{mantissa:.{decimal_places}f}d{exp:+d}"

class ELMSoilData:
	"""
	A class to handle extraction, computation, and writing of soil properties
	and layer thickness/interfaces into PFLOTRAN-compatible input sections.
	"""
	
	default_soil_depths = np.array([
		0.007, 0.028, 0.062, 0.119, 0.212, 0.367, 0.620, 1.038, 1.728, 2.864
	])
	
	default_layer_interfaces = np.array([
		0.000, 0.014, 0.042, 0.082, 0.155, 0.270, 0.464, 0.777, 1.300, 2.155, 3.574
	])
	
	def __init__(self, filepath):
		"""
		Initialize the ELMSoilData object by loading the file.
		
		Arguments:
			filepath (str): Path to the NetCDF surface data file.
		"""
		self.filepath = filepath
		self.ds = xr.open_dataset(filepath)
		
		# Metadata: Coordinates of the file
		self.metadata = {
			"latitude": float(self.ds.coords['lsmlat'].values),
			"longitude": float(self.ds.coords['lsmlon'].values),
			"time": self.ds.coords['time'].values,  # Time steps
		}
		
		# Initialize material properties and characteristic curves
		self.material_properties = {}
		self.characteristic_curves = {}
	
	def get_layer_thicknesses(self):
		"""
		Get the layer thicknesses from soil depth data.
		
		Returns:
			numpy array of layer thicknesses from the NetCDF file
		"""
		if 'levsoi' in self.ds.coords:
			return self.ds['levsoi'].values.copy()
		else:
			return self.default_soil_depths.copy()
	
	def extract_soil_data(self):
		"""
		Extract soil layers, textures, depths, interfaces, and thicknesses.
		Handles missing data gracefully by substituting defaults.
		
		Returns:
			Dictionary containing extracted texture, depth, interfaces, and thickness.
		"""
		sand = np.nan_to_num(self.ds['PCT_SAND'].values.squeeze(), nan=0.0)
		clay = np.nan_to_num(self.ds['PCT_CLAY'].values.squeeze(), nan=0.0)
		
		organic = (
			np.nan_to_num(self.ds['ORGANIC'].values.squeeze(), nan=0.0)
			if 'ORGANIC' in self.ds.variables else np.zeros_like(sand)
		)
		
		gravel = (
			np.nan_to_num(self.ds['PCT_GRVL'].values.squeeze(), nan=0.0)
			if 'PCT_GRVL' in self.ds.variables else np.zeros_like(sand)
		)
		
		# Use actual soil depths from NetCDF file if available, otherwise use defaults
		if 'levsoi' in self.ds.coords:
			soil_depths = self.ds['levsoi'].values
		else:
			soil_depths = self.default_soil_depths
		
		# Calculate interfaces from cumulative soil depths
		interfaces = np.concatenate([[0.0], np.cumsum(soil_depths)])
		
		return {
			'sand': sand,
			'clay': clay,
			'organic': organic,
			'gravel': gravel,
			'depths': soil_depths,  # Actual layer thicknesses
			'interfaces': interfaces,  # Cumulative depths for interface calculation
			'thickness': soil_depths   # Same as depths - actual layer thicknesses
		}
	
	def calculate_hydraulic_properties(self, sand_pct, clay_pct, organic_pct=0, gravel_pct=0):
		"""
		Calculate hydraulic properties for a layer using pedotransfer functions.
		
		Arguments:
			sand_pct (float): Percentage sand content.
			clay_pct (float): Percentage clay content.
			organic_pct (float): Percentage organic matter.
			gravel_pct (float): Percentage gravel content.
		
		Returns:
			Calculated hydraulic properties for the layer (e.g., porosity, permeability, Van Genuchten parameters, etc.).
		"""
		sand = sand_pct / 100.0
		clay = clay_pct / 100.0
		organic = organic_pct / 100.0
		
		# Porosity calculation
		porosity = 0.332 - 7.251e-4 * sand_pct + 0.1276 * np.log10(clay_pct + 1e-10)
		porosity += 0.15 * organic
		porosity *= (1 - gravel_pct / 200.0)
		porosity = np.clip(porosity, 0.2, 0.6)
		
		# Saturated hydraulic conductivity
		log_ks = -0.6 + 0.012 * sand_pct - 0.0064 * clay_pct
		ks_cm_hr = 25.4 * (10 ** log_ks)
		ks_m_s = ks_cm_hr * 2.778e-6  # Convert cm/hr to m/s
		ks_m_s = np.maximum(ks_m_s, 1e-5)
		permeability = ks_m_s * 1.0e-3 / (1000.0 * 9.81)
		
		# Van Genuchten parameters
		if sand_pct >= 50:
			alpha, m, theta_r, texture = 1e-4, 0.5, 0.065, "sandy_loam"
		elif clay_pct >= 30:
			alpha, m, theta_r, texture = 5e-5, 0.4, 0.095, "clay_loam"
		else:
			alpha, m, theta_r, texture = 1e-4, 0.5, 0.078, "loam"
		
		return {
			'porosity': porosity,
			'permeability': permeability,
			'alpha': alpha,
			'm': m,
			'n': 1 / (1 - m),
			'residual_saturation': theta_r,
			'sat_hydraulic_conductivity': ks_m_s,
			'texture_class': texture
		}
	
	def compute_properties(self):
		"""
		Compute material properties and characteristic curves for all layers.
		Note: Depths will be updated later by the main script to ensure consistency.
		"""
		soil_data = self.extract_soil_data()
		
		for i in range(len(soil_data['sand'])):
			props = self.calculate_hydraulic_properties(
				sand_pct=soil_data['sand'][i],
				clay_pct=soil_data['clay'][i],
				organic_pct=soil_data['organic'][i] * 0.01,
				gravel_pct=soil_data['gravel'][i]
			)
			
			layer_name = f"elm_layer{i+1}"
			curve_name = f"elm_cc{i+1}"
			
			# Initial depth calculation (will be overridden by main script for consistency)
			if i == 0:
				depth_bottom = 0.0
			else:
				depth_bottom = sum(soil_data['depths'][:i])
			
			depth_top = depth_bottom + soil_data['depths'][i]
			
			self.material_properties[layer_name] = {
				'depth_bottom': depth_bottom,
				'depth_top': depth_top,
				'thickness': soil_data['depths'][i],
				'porosity': props['porosity'],
				'permeability': props['permeability'],
				'id': i + 1,
				'characteristic_curve': curve_name
			}
			
			self.characteristic_curves[curve_name] = {
				'saturation_function': {
					'type': "VAN_GENUCHTEN",
					'alpha': props['alpha'],
					'm': props['m'],
					'liquid_residual_saturation': props['residual_saturation']
				},
				'permeability_function': {
					'type': "MUALEM_VG_LIQ",
					'm': props['m'],
					'liquid_residual_saturation': props['residual_saturation']
				}
			}
	
	def write_material_properties(self, output_file=None, file_handle=None):
		"""
		Write material property sections into PFLOTRAN input file.
		
		Arguments:
			output_file (str): Path to output file (if writing to new file)
			file_handle: Open file handle (if writing to existing file)
		"""
		if file_handle:
			f = file_handle
			should_close = False
		else:
			f = open(output_file, 'w')
			should_close = True
		
		f.write("#=========================== material properties ==============================\n")
		
		# Sort by ID to maintain correct order
		sorted_materials = sorted(self.material_properties.items(), 
								 key=lambda x: x[1]['id'])
		
		for layer, props in sorted_materials:
			f.write(f"MATERIAL_PROPERTY {layer}\n")
			f.write(f"  ID {props['id']}\n")
			f.write(f"  POROSITY {fortran_format(props['porosity'])}\n")
			f.write(f"  PERMEABILITY\n")
			f.write(f"    PERM_ISO {fortran_format(props['permeability'])}\n")
			f.write(f"  /\n")
			f.write(f"  CHARACTERISTIC_CURVES {props['characteristic_curve']}\n")
			f.write("END\n\n")
		
		if should_close:
			f.close()
			print(f"Material properties written to: {output_file}")
	
	def write_characteristic_curves(self, output_file=None, file_handle=None):
		"""
		Write characteristic curve sections into PFLOTRAN input file.
		
		Arguments:
			output_file (str): Path to output file (if writing to new file)
			file_handle: Open file handle (if writing to existing file)
		"""
		if file_handle:
			f = file_handle
			should_close = False
		else:
			f = open(output_file, 'w')
			should_close = True
		
		f.write("#=========================== characteristic curves ============================\n")
		
		# Sort by material ID to maintain order
		material_id_to_curve = {}
		for mat_name, mat_props in self.material_properties.items():
			curve_name = mat_props['characteristic_curve']
			material_id_to_curve[mat_props['id']] = curve_name
		
		sorted_curve_names = [material_id_to_curve[i] for i in sorted(material_id_to_curve.keys())]
		
		for curve_name in sorted_curve_names:
			if curve_name in self.characteristic_curves:
				params = self.characteristic_curves[curve_name]
				f.write(f"CHARACTERISTIC_CURVES {curve_name}\n")
				f.write(f"  SATURATION_FUNCTION {params['saturation_function']['type']}\n")
				
				# Use Fortran format for ALL values
				f.write(f"    ALPHA {fortran_format(params['saturation_function']['alpha'])}\n")
				f.write(f"    M {fortran_format(params['saturation_function']['m'])}\n")
				f.write(f"    LIQUID_RESIDUAL_SATURATION {fortran_format(params['saturation_function']['liquid_residual_saturation'])}\n")
				f.write("  /\n")
				f.write(f"  PERMEABILITY_FUNCTION {params['permeability_function']['type']}\n")
				f.write(f"    M {fortran_format(params['permeability_function']['m'])}\n")
				f.write(f"    LIQUID_RESIDUAL_SATURATION {fortran_format(params['permeability_function']['liquid_residual_saturation'])}\n")
				f.write("  /\n")
				f.write("END\n\n")
		
		if should_close:
			f.close()
			print(f"Characteristic curves written to: {output_file}")
	
	def get_planning_summary(self):
		"""
		Generate a concise summary of soil profile for experiment planning.
		
		Returns:
			Dictionary with key information needed for planning
		"""
		if not self.material_properties:
			self.compute_properties()
		
		soil_data = self.extract_soil_data()
		layer_thicknesses = self.get_layer_thicknesses()
		
		# Basic profile info
		summary = {
			"file_info": {
				"filepath": self.filepath,
				"latitude": self.metadata["latitude"],
				"longitude": self.metadata["longitude"]
			},
			"profile_structure": {
				"num_layers": len(layer_thicknesses),
				"total_depth_m": float(np.sum(layer_thicknesses)),
				"layer_thicknesses_m": layer_thicknesses.tolist(),
				"depth_resolution": "fine" if np.min(layer_thicknesses) < 0.1 else "coarse"
			}
		}
		
		# Analyze material properties if available
		if self.material_properties:
			porosities = [props['porosity'] for props in self.material_properties.values()]
			permeabilities = [props['permeability'] for props in self.material_properties.values()]
			
			summary["hydraulic_properties"] = {
				"porosity_range": [float(np.min(porosities)), float(np.max(porosities))],
				"porosity_mean": float(np.mean(porosities)),
				"permeability_range_m2": [float(np.min(permeabilities)), float(np.max(permeabilities))],
				"permeability_variation": f"{np.max(permeabilities)/np.min(permeabilities):.1e}" if np.min(permeabilities) > 0 else "N/A",
				"heterogeneity": "high" if np.max(permeabilities)/np.min(permeabilities) > 100 else "moderate"
			}
		
		# Analyze soil texture patterns
		summary["soil_characteristics"] = self._analyze_soil_texture(soil_data)
		
		return summary
	
	def _analyze_soil_texture(self, soil_data):
		"""Analyze soil texture characteristics for planning."""
		
		sand_avg = float(np.mean(soil_data['sand']))
		clay_avg = float(np.mean(soil_data['clay']))
		
		# Classify dominant texture
		if sand_avg > 70:
			texture_class = "sandy"
		elif clay_avg > 30:
			texture_class = "clayey"  
		else:
			texture_class = "loamy"
		
		return {
			"dominant_texture": texture_class,
			"sand_content_pct": [float(np.min(soil_data['sand'])), float(np.max(soil_data['sand']))],
			"clay_content_pct": [float(np.min(soil_data['clay'])), float(np.max(soil_data['clay']))],
			"texture_variability": "high" if (np.std(soil_data['sand']) > 20 or np.std(soil_data['clay']) > 15) else "low"
		}
	
	def close(self):
		"""
		Close the dataset safely to free memory.
		"""
		self.ds.close()