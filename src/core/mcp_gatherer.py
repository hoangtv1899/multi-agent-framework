# src/core/mcp_gatherer.py
"""
MCP Gatherer - Handles all regional data gathering from MCP servers

Geocoding:  Nominatim (OpenStreetMap) - free, no key
Streamflow: waterdata.usgs.gov direct API
            (waterservices.usgs.gov is blocked on Perlmutter/NERSC)
"""
import json
import requests
from datetime import datetime
from typing import Any, Dict, Optional

from .mcp_context import MCPContext

NOMINATIM_URL      = "https://nominatim.openstreetmap.org/search"
NOMINATIM_HEADERS  = {"User-Agent": "PFLOTRAN-MCP/1.0 hoang.tran@pnnl.gov"}
USGS_WATERDATA_URL = "https://waterdata.usgs.gov/nwis/iv/"


class MCPGatherer:
	"""
	Queries all available MCP servers for a location
	and returns a populated MCPContext.
	
	Note on USGS streamflow:
		fetch_usgs_data MCP tool calls waterservices.usgs.gov
		which is blocked on Perlmutter. We bypass it and call
		waterdata.usgs.gov directly via requests instead.
	"""
	
	def __init__(self, mcp_clients: Dict[str, Any]):
		self.clients = mcp_clients or {}
	
	# =========================================================================
	# MAIN ENTRY POINT
	# =========================================================================
	
	def gather(self, location: str) -> MCPContext:
		"""Query all available MCP servers for a location."""
		ctx = MCPContext(location_query=location)
	
		if 'usgs_water' in self.clients:
			ctx = self._gather_usgs(ctx)
	
		if 'weather' in self.clients:
			ctx = self._gather_weather(ctx, location)
	
		if 'geology' in self.clients:
			ctx = self._gather_geology(ctx)
	
		return ctx
	
	# =========================================================================
	# USGS WATER
	# =========================================================================
	
	def _gather_usgs(self, ctx: MCPContext) -> MCPContext:
		"""Find nearest active USGS site → fetch streamflow + groundwater."""
		client = self.clients['usgs_water']
	
		coords = self._coordinates(ctx.location_query)
		if not coords:
			ctx.gather_errors.append(
				f"Could not geocode '{ctx.location_query}'"
			)
			return ctx
	
		lat, lon = coords
	
		# Search wider bbox to find more candidate sites
		bbox = f"{lon-1.0},{lat-1.0},{lon+1.0},{lat+1.0}"
		print(f"   Searching active USGS sites near {lat:.4f},{lon:.4f}...")
	
		site_data = client.call_tool_json(
			'get_monitoring_locations',
			{'bbox': bbox, 'agency_code': 'USGS', 'limit': 20}
		)
		if not site_data:
			ctx.gather_errors.append("Could not reach USGS API")
			return ctx
	
		features = site_data.get('features', [])
		if not features:
			ctx.gather_errors.append("No USGS sites found near location")
			return ctx
	
		# Test each site - pick first one with real streamflow data
		active_site = None
		active_flow = None
	
		for feature in features:
			props   = feature.get('properties', {})
			site_id = props.get('monitoring_location_number', '')
			name    = props.get('monitoring_location_name', '')
			if not site_id:
				continue
	
			flow = self._fetch_usgs_direct(site_id, '00060')
			if flow and len(flow) > 0:
				active_site = {'id': site_id, 'name': name}
				active_flow = flow
				print(f"   ✓ Active site: {name} ({site_id}) "
					  f"- {len(flow)} obs")
				break
			else:
				print(f"   ✗ No data: {name} ({site_id})")
	
		if not active_site:
			ctx.gather_errors.append(
				"No active USGS streamflow sites found near location"
			)
			return ctx
	
		ctx.usgs_site_id   = active_site['id']
		ctx.usgs_site_name = active_site['name']
		ctx.streamflow_data = active_flow
		ctx.data_sources.append(
			f"USGS NWIS {ctx.usgs_site_id} - {ctx.usgs_site_name}"
		)
		print(f"   ✓ Streamflow: {len(ctx.streamflow_data)} obs")
	
		# Groundwater at same site
		print(f"   Fetching groundwater...")
		gw = self._fetch_usgs_direct(ctx.usgs_site_id, '72019')
		if gw is not None and len(gw) > 0:
			ctx.subsurface_data = gw
			print(f"   ✓ Groundwater: {len(ctx.subsurface_data)} obs")
		else:
			ctx.gather_errors.append(
				"No groundwater data at this site"
			)
	
		return ctx
		
	def _fetch_usgs_direct(self,
							site_id: str,
							parameter_cd: str,
							period: str = 'P7D') -> Optional[list]:
		"""
		Fetch USGS timeseries from waterdata.usgs.gov directly.
	
		Args:
			site_id:      e.g. '11467000'
			parameter_cd: '00060' discharge | '72019' groundwater depth
			period:       'P7D' = last 7 days
	
		Returns:
			List of observation dicts or None on error
		"""
		try:
			r = requests.get(
				USGS_WATERDATA_URL,
				params={
					'sites':       site_id,
					'parameterCd': parameter_cd,
					'period':      period,
					'format':      'json'
				},
				timeout=20
			)
			r.raise_for_status()
			return self._parse_usgs_timeseries(r.json())
		except Exception as e:
			print(f"   ⚠️  USGS direct fetch failed: {e}")
			return None
	
	# =========================================================================
	# WEATHER
	# =========================================================================
	
	def _gather_weather(self, ctx: MCPContext,
						 location: str) -> MCPContext:
		"""Query weather MCP: forecast + long-term climate."""
		client = self.clients['weather']
	
		coords = self._coordinates(location)
		if not coords:
			ctx.gather_errors.append(
				"Could not geocode location for weather"
			)
			return ctx
	
		lat, lon = coords
	
		# 7-day forecast
		print(f"   Fetching weather forecast...")
		forecast = client.call_tool_json(
			'get_forecast', {'lat': lat, 'lon': lon}
		)
		if forecast and 'error' not in forecast:
			ctx.weather_current = forecast
			ctx.data_sources.append("NWS 7-day forecast")
			print(f"   ✓ Forecast: "
				  f"{len(forecast.get('forecast', []))} periods")
		else:
			ctx.gather_errors.append("Weather forecast unavailable")
	
		# Long-term climate (last 20 years)
		end_year   = datetime.now().year - 1
		start_year = end_year - 19
		print(f"   Fetching climate ({start_year}-{end_year})...")
		climate = client.call_tool_json('get_climate_summary', {
			'lat': lat, 'lon': lon,
			'start_year': start_year,
			'end_year':   end_year
		})
		if climate and 'error' not in climate:
			ctx.weather_climate = climate
			ctx.data_sources.append(
				f"Open-Meteo climate {start_year}-{end_year}"
			)
			print(f"   ✓ Climate: "
				  f"{climate.get('precip_mm_per_year', 'N/A')} mm/yr")
		else:
			ctx.gather_errors.append("Climate data unavailable")
	
		return ctx
	
	# =========================================================================
	# GEOLOGY
	# =========================================================================
	
	def _gather_geology(self, ctx: MCPContext) -> MCPContext:
		"""Query geology MCP: soil profile + PFLOTRAN materials."""
		client = self.clients['geology']
	
		coords = self._coordinates(ctx.location_query)
		if not coords:
			ctx.gather_errors.append(
				"Could not geocode location for geology"
			)
			return ctx
	
		lat, lon = coords
	
		# Try original coords then offsets
		# River coords often have no SSURGO data
		offsets = [
			(0.0,   0.0),
			(0.05,  0.0),
			(0.0,   0.05),
			(-0.05, 0.0),
			(0.0,  -0.05),
			(0.05,  0.05),
		]
	
		profile     = None
		used_coords = None
	
		for dlat, dlon in offsets:
			test_lat = lat + dlat
			test_lon = lon + dlon
			result   = client.call_tool_json(
				'get_soil_profile', {'lat': test_lat, 'lon': test_lon}
			)
			if result and 'error' not in result:
				profile     = result
				used_coords = (test_lat, test_lon)
				if dlat != 0 or dlon != 0:
					print(f"   ℹ️  No SSURGO at river coords, "
						  f"using nearby land "
						  f"({test_lat:.3f},{test_lon:.3f})")
				break
	
		if not profile:
			ctx.gather_errors.append(
				"No SSURGO soil data found near location"
			)
			return ctx
	
		ctx.soil_profile = profile
		n = profile.get('num_layers', 0)
		ctx.data_sources.append(f"SSURGO soil profile ({n} layers)")
		print(f"   ✓ Soil profile: {n} layers")
	
		# PFLOTRAN materials at same coords
		print(f"   Fetching PFLOTRAN materials...")
		test_lat, test_lon = used_coords
		materials = client.call_tool_json(
			'get_pflotran_materials', {'lat': test_lat, 'lon': test_lon}
		)
		if materials and 'error' not in materials:
			ctx.pflotran_materials = materials
			print(f"   ✓ PFLOTRAN materials: "
				  f"{materials.get('num_layers', 0)} layers ready")
		else:
			ctx.gather_errors.append("PFLOTRAN materials unavailable")
	
		return ctx
	
	# =========================================================================
	# UTILITIES
	# =========================================================================
	
	def _coordinates(self, location: str) -> Optional[tuple]:
		"""Geocode location using Nominatim (OpenStreetMap)."""
		try:
			r = requests.get(
				NOMINATIM_URL,
				params={
					'q':            location,
					'format':       'json',
					'limit':        1,
					'countrycodes': 'us'
				},
				headers=NOMINATIM_HEADERS,
				timeout=10
			)
			results = r.json()
			if results:
				lat  = float(results[0]['lat'])
				lon  = float(results[0]['lon'])
				name = results[0].get('display_name', location)
				print(f"   ✓ Geocoded: {name[:60]}")
				print(f"              ({lat:.4f}, {lon:.4f})")
				return lat, lon
		except Exception as e:
			print(f"   ⚠️  Geocoding failed: {e}")
		return None
	
	def _pick_best_site(self, site_data: Dict) -> Optional[Dict]:
		"""Pick most relevant site from USGS response."""
		features = site_data.get('features', [])
		if not features:
			return None
		props = features[0].get('properties', {})
		return {
			'id':   props.get('monitoring_location_number', ''),
			'name': props.get('monitoring_location_name', '')
		}
	
	def _parse_usgs_timeseries(self, data: Dict) -> list:
		"""Parse USGS timeSeries response into flat observation list."""
		observations = []
		for series in data.get('value', {}).get('timeSeries', []):
			values = (series.get('values', [{}])[0].get('value', []))
			param  = (series.get('variable', {})
						   .get('variableCode', [{}])[0]
						   .get('value', ''))
			for v in values:
				observations.append({
					'datetime':       v.get('dateTime', ''),
					'value':          v.get('value', ''),
					'qualifier':      v.get('qualifiers', ['']),
					'parameter_code': param
				})
		if not observations and isinstance(data, list):
			observations = data
		return observations
	
	def __repr__(self):
		return f"MCPGatherer(clients={list(self.clients.keys())})"