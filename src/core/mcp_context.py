# src/core/mcp_context.py
"""
MCPContext - Simple container for environmental data gathered via MCP servers

Travels through the agent pipeline as a single object:

    ReceptionAgent                    (Phase 3: populates it)
        └── MCPContext
              ├── PlannerAgent        (Phase 4: reads it)
              └── AnalysisReportAgent (Phase 5: reads it)

Design philosophy:
    - Flat and simple - no nested sub-classes
    - Extensible - new MCP sources just add new fields
    - Safe - agents always check .has_data before using values
    - Serializable - to_dict/from_dict for pipeline passing

Current MCP sources:
    usgs_water → streamflow_data, subsurface_data

Future MCP sources (just add fields when ready):
    weather    → weather_data
    soil_db    → soil_data
    literature → literature_data
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class MCPContext:
	# Core
	location_query: str = ""
	
	# USGS site
	usgs_site_id:   str = ""
	usgs_site_name: str = ""
	
	# usgs_water MCP
	streamflow_data: List[Dict[str, Any]] = field(default_factory=list)
	subsurface_data: List[Dict[str, Any]] = field(default_factory=list)
	
	# weather MCP
	weather_current: Dict[str, Any] = field(default_factory=dict)
	weather_climate: Dict[str, Any] = field(default_factory=dict)
	
	# geology MCP                                        ← NEW
	soil_profile:      Dict[str, Any] = field(default_factory=dict)
	pflotran_materials: Dict[str, Any] = field(default_factory=dict)
	
	# Provenance
	data_sources:  List[str] = field(default_factory=list)
	gathered_at:   str       = field(
								   default_factory=lambda:
								   datetime.utcnow().isoformat() + "Z"
							   )
	gather_errors: List[str] = field(default_factory=list)
	
	# =========================================================================
	# PROPERTIES
	# =========================================================================
	
	@property
	def has_data(self) -> bool:
		return bool(self.streamflow_data or
					self.subsurface_data  or
					self.weather_climate  or
					self.soil_profile)                  # ← updated
	
	@property
	def has_streamflow(self) -> bool:
		return bool(self.streamflow_data)
	
	@property
	def has_subsurface(self) -> bool:
		return bool(self.subsurface_data)
	
	@property
	def has_weather(self) -> bool:
		return bool(self.weather_climate)
	
	@property
	def has_geology(self) -> bool:                     # ← NEW
		return bool(self.soil_profile)
	
	@property
	def latest_streamflow(self) -> Optional[Dict[str, Any]]:
		return self.streamflow_data[-1] if self.streamflow_data else None
	
	@property
	def latest_subsurface(self) -> Optional[Dict[str, Any]]:
		return self.subsurface_data[-1] if self.subsurface_data else None
	
	# =========================================================================
	# PIPELINE SUMMARIES
	# =========================================================================
	
	def summary_for_planner(self) -> str:
		"""Compact string for injection into PlannerAgent prompts."""
		lines = ["── MCP Field Data ──────────────────────────────"]
	
		if not self.has_data:
			lines.append("  No field data available.")
			lines.append("────────────────────────────────────────────────")
			return "\n".join(lines)
	
		# Site
		if self.usgs_site_name:
			lines.append(f"  Site: {self.usgs_site_name} "
						 f"(USGS {self.usgs_site_id})")
	
		# Streamflow
		if self.has_streamflow:
			obs = self.latest_streamflow
			dt  = obs.get('datetime', 'N/A')[:10]
			val = obs.get('value', '')
			pc  = obs.get('parameter_code', 'N/A')
			lines.append(f"  Streamflow ({dt}):")
			if val:
				lines.append(f"    Value: {val} (code: {pc})")
			else:
				lines.append(f"    {len(self.streamflow_data)} observations "
							 f"(code: {pc})")
	
		# Subsurface
		if self.has_subsurface:
			obs = self.latest_subsurface
			lines.append(f"  Groundwater ({obs.get('datetime','N/A')[:10]}):")
			if obs.get('water_level_ft') is not None:
				lines.append(f"    Water level: "
							 f"{obs['water_level_ft']:.2f} ft below surface")
	
		# Weather
		if self.has_weather:
			c = self.weather_climate
			lines.append(f"  Climate ({c.get('period', 'N/A')}):")
			lines.append(f"    Precip:        "
						 f"{c.get('precip_mm_per_year','N/A')} mm/yr")
			lines.append(f"    Temp range:    "
						 f"{c.get('mean_tmin_c','N/A')} - "
						 f"{c.get('mean_tmax_c','N/A')} °C")
			lines.append(f"    Recharge flux: "
						 f"{c.get('suggested_recharge_flux_ms','N/A')} m/s")
	
		# Geology                                        ← NEW
		if self.has_geology:
			n  = self.soil_profile.get('num_layers', 0)
			layers = self.soil_profile.get('layers', [])
			lines.append(f"  Soil profile ({n} layers from SSURGO):")
			for lyr in layers:
				lines.append(
					f"    {lyr.get('depth_top_cm',0)}-"
					f"{lyr.get('depth_bot_cm','?')} cm: "
					f"{lyr.get('texture_class','unknown')} "
					f"(sand={lyr.get('sand_pct','?')}% "
					f"clay={lyr.get('clay_pct','?')}%)"
				)
			if self.pflotran_materials:
				mats = self.pflotran_materials.get('materials', [])
				lines.append(f"  PFLOTRAN materials: "
							 f"{len(mats)} layers ready to inject")
	
		if self.gather_errors:
			lines.append(f"  ⚠️  Warnings: {'; '.join(self.gather_errors)}")
	
		lines.append("────────────────────────────────────────────────")
		return "\n".join(lines)
	
	def summary_for_analyzer(self) -> str:
		"""Compact string for injection into AnalysisReportAgent prompts."""
		lines = ["── Observed Field Data ─────────────────────────"]
	
		if not self.has_data:
			lines.append("  No field observations available.")
			lines.append("────────────────────────────────────────────────")
			return "\n".join(lines)
	
		if self.usgs_site_name:
			lines.append(f"  Reference: {self.usgs_site_name} "
						 f"(USGS {self.usgs_site_id})")
	
		# Same fix in summary_for_analyzer()
		if self.has_streamflow:
			obs = self.latest_streamflow
			dt  = obs.get('datetime', 'N/A')[:10]
			val = obs.get('value', '')
			lines.append(f"  Streamflow ({dt}):")
			if val:
				lines.append(f"    {val} (code {obs.get('parameter_code','N/A')})")
			else:
				lines.append(f"    {len(self.streamflow_data)} observations available")
	
		if self.has_subsurface:
			obs = self.latest_subsurface
			lines.append(f"  Groundwater ({obs.get('datetime','N/A')[:10]}):")
			if obs.get('water_level_ft') is not None:
				lines.append(f"    {obs['water_level_ft']:.2f} ft below surface")
	
		if self.has_weather:
			c = self.weather_climate
			lines.append(f"  Climate ({c.get('period','N/A')}):")
			lines.append(f"    Annual precip: "
						 f"{c.get('precip_mm_per_year','N/A')} mm/yr")
			lines.append(f"    Recharge flux: "
						 f"{c.get('suggested_recharge_flux_ms','N/A')} m/s")
	
		if self.has_geology:                           # ← NEW
			n = self.soil_profile.get('num_layers', 0)
			lines.append(f"  Soil: {n} SSURGO layers used in simulation")
	
		lines.append(f"  Sources: {'; '.join(self.data_sources)}")
		lines.append("────────────────────────────────────────────────")
		return "\n".join(lines)
	
	# =========================================================================
	# SERIALIZATION
	# =========================================================================
	
	def to_dict(self) -> Dict[str, Any]:
		return {
			'location_query':    self.location_query,
			'usgs_site_id':      self.usgs_site_id,
			'usgs_site_name':    self.usgs_site_name,
			'streamflow_data':   self.streamflow_data,
			'subsurface_data':   self.subsurface_data,
			'weather_current':   self.weather_current,
			'weather_climate':   self.weather_climate,
			'soil_profile':      self.soil_profile,       # ← NEW
			'pflotran_materials': self.pflotran_materials, # ← NEW
			'data_sources':      self.data_sources,
			'gathered_at':       self.gathered_at,
			'gather_errors':     self.gather_errors,
		}
	
	@classmethod
	def from_dict(cls, d: Dict[str, Any]) -> MCPContext:
		return cls(
			location_query    = d.get('location_query', ''),
			usgs_site_id      = d.get('usgs_site_id', ''),
			usgs_site_name    = d.get('usgs_site_name', ''),
			streamflow_data   = d.get('streamflow_data', []),
			subsurface_data   = d.get('subsurface_data', []),
			weather_current   = d.get('weather_current', {}),
			weather_climate   = d.get('weather_climate', {}),
			soil_profile      = d.get('soil_profile', {}),       # ← NEW
			pflotran_materials = d.get('pflotran_materials', {}), # ← NEW
			data_sources      = d.get('data_sources', []),
			gathered_at       = d.get('gathered_at',
									  datetime.utcnow().isoformat() + "Z"),
			gather_errors     = d.get('gather_errors', []),
		)
	
	@classmethod
	def from_json(cls, json_str: str) -> MCPContext:
		return cls.from_dict(json.loads(json_str))
	
	@classmethod
	def empty(cls, location_query: str = "") -> MCPContext:
		return cls(
			location_query = location_query,
			gather_errors  = ["No MCP data gathered"]
		)
	
	def __repr__(self) -> str:
		return (
			f"MCPContext("
			f"location='{self.location_query}', "
			f"site='{self.usgs_site_id}', "
			f"streamflow={len(self.streamflow_data)} obs, "
			f"subsurface={len(self.subsurface_data)} obs, "
			f"weather={'yes' if self.has_weather else 'no'}, "
			f"geology={'yes' if self.has_geology else 'no'})"
		)