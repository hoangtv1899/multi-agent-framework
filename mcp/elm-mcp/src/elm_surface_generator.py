#!/usr/bin/env python3
"""
ELM Surface Data Generator
src/core/elm_surface_generator.py

Generates ELM surface data NetCDF files for a given lat/lon
with soil properties from either MCP geology data (real layered
SSURGO) or synthetic uniform texture.

Key updates in this revision:
    - Parses MCP-native rich format (derivation.inputs string)
      directly, in addition to the simple {sand_pct, clay_pct, ...}
      schema. No upstream pre-conversion needed.
    - Depth-aware mapping: assigns each ELM level to the MCP
      horizon whose depth range contains the level's node depth
      (replaces the prior positional MCP-i → ELM-i mapping).
    - Backwards compatible: still accepts the simple schema if
      that's what the caller passes.

Surface file dimensions:
    nlevsoi=10   (ELM standard 10-layer soil column, ~3.4 m deep)
    lsmlat=1     (single column)
    lsmlon=1     (single column)

Soil variables modified per layer:
    PCT_SAND, PCT_CLAY, ORGANIC, PCT_GRVL

Substrate (for native mode):
    SSURGO typically covers only the top ~2 m, but ELM's hydrology
    column extends to ~3.4 m. The `substrate` parameter controls
    what fills ELM levels deeper than MCP data:

        'template'    → keep Station 2006 template values
                        (indicates "unknown"; default)
        'extrapolate' → reuse deepest MCP layer's values
        'sandy'       → coarse, high-permeability (alluvial proxy)
        'clayey'      → fine, low-permeability (aquitard proxy)

MCP-native parsing:
    Layers may arrive in either schema:

      (a) Simple format (legacy / passthrough):
          {'sand_pct': 80.0, 'clay_pct': 2.5, 'organic_pct': 3.0,
           'gravel_pct': 2.0, 'depth_top_m': 0.0, 'depth_bot_m': 0.18}

      (b) MCP-native rich format (what the geology MCP returns):
          {
            'layer_id': 1, 'texture_class': 'sandy loam',
            'depth_top_cm': 0.0, 'depth_bot_cm': 18.0,
            'derivation': {
                'method': 'Rawls & Brakensiek (1985)',
                'inputs': 'Derived from: sand=80.0% silt=17.5% '
                          'clay=2.5% db=1.65g/cm³'
            },
            ... other PFLOTRAN-style fields ...
          }

    The parser tries simple-format keys first, then falls back to
    parsing the `derivation.inputs` string.

Caching:
    Output filenames embed lat/lon + label, so repeated calls reuse
    the existing file. Pass `force=True` to bypass and regenerate
    (e.g., after fixing a parser bug — old cached files do not
    reflect new parser behavior).

Paths (overridable via env vars):
    ELM_SURFACE_TEMPLATE     — full path to template NetCDF
    ELM_SURFACE_OUTPUT_DIR   — directory for generated surfaces
"""
import hashlib
import json
import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import numpy as np
    import xarray as xr
    import netCDF4  # noqa: F401  (required by xarray engine='netcdf4')
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# PATHS — overridable via env vars
# ─────────────────────────────────────────────────────────────────────
# ELM 1d input files (originals on NERSC; copy them here on Compy)
_ELM_INPUT_FILES_DIR = os.getenv(
    "ELM_INPUT_FILES_DIR",
    "/qfs/people/tran289/IDEAS/1d_elm/input_files",
)
_DEFAULT_TEMPLATE = (
    _ELM_INPUT_FILES_DIR + "/Surfacedata_Station_2006_.nc"
)
_DEFAULT_OUTPUT_DIR = (
    _ELM_INPUT_FILES_DIR + "/surfaces"
)

SURFACE_TEMPLATE   = os.environ.get('ELM_SURFACE_TEMPLATE',   _DEFAULT_TEMPLATE)
SURFACE_OUTPUT_DIR = os.environ.get('ELM_SURFACE_OUTPUT_DIR', _DEFAULT_OUTPUT_DIR)

# Global (CONUS/world) mksurfdata-format surface file used to extract REAL
# per-location vegetation/land-cover for veg_source='conus' (see
# generate_from_mcp). Without this, every generated column's PCT_NAT_PFT /
# LAI / SAI / HEIGHT are frozen copies of SURFACE_TEMPLATE's single site.
CONUS_SURFDATA_NC = os.environ.get(
    "CONUS_SURFDATA_NC",
    "/qfs/people/tran289/IDEAS/1d_elm/conus_surfdata/"
    "surfdata_0.5x0.5_rcp3.0_simyr2015_c231113.nc",
)


# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────
N_SOIL_LEVELS = 10

SOIL_VARS = ['PCT_SAND', 'PCT_CLAY', 'ORGANIC', 'PCT_GRVL']

# Used when a horizon reports organic matter but no bulk density. A mid-range
# mineral soil; only ever scales the % -> kg/m3 conversion, never replaces a
# reported density.
_NOMINAL_BULK_DENSITY_GCC = 1.3

# Vegetation/land-cover variables that veg_source='conus' overwrites with a
# real per-location extraction instead of leaving them as frozen template
# copies. Shapes (from CONUS_SURFDATA_NC): PCT_NAT_PFT (natpft,lsmlat,lsmlon),
# PCT_CFT (cft,lsmlat,lsmlon), MONTHLY_* (time,lsmpft,lsmlat,lsmlon), the rest
# (lsmlat,lsmlon). All identical dim names/sizes to the single-point template.
VEG_VARS = [
    'PCT_NAT_PFT', 'PCT_CFT',
    'MONTHLY_LAI', 'MONTHLY_SAI', 'MONTHLY_HEIGHT_TOP', 'MONTHLY_HEIGHT_BOT',
    # LANDUNIT_PCT_VARS below MUST be extracted as a complete set from the same
    # source cell -- ELM's surfrd_get_data -> check_sums_equal_1(wt_lunit)
    # aborts the run if they do not sum to 100%. Taking only some of them from
    # CONUS while leaving the rest at the template site's values yields e.g.
    # 101.086% and a hard MPI_Abort deep in initialization.
    'PCT_NATVEG', 'PCT_CROP',
    'PCT_URBAN', 'PCT_LAKE', 'PCT_WETLAND', 'PCT_GLACIER',
]

def fold_crop_into_natveg(ds) -> float:
    """Move PCT_CROP into PCT_NATVEG in place; return the percent moved.

    elm_wrapper sets create_crop_landunit = .false., which makes the sub-grid
    [1 natveg + 15 urban] = 16 columns — EXACTLY the layout of the CONUS 1-km
    restarts, which contain zero type-2 (crop) columns across all 19.9M of them.
    That match is what lets a CONUS gridcell be subset straight into a finidat
    with no carrier file and no prior run.

    The surfdata is made to agree with the namelist rather than relying on ELM
    to reconcile them: PCT_CROP > 0 with the crop landunit disabled is the
    combination that trips initialization.

    The area is not discarded. It becomes natural vegetation at the column's
    existing PCT_NAT_PFT distribution, so the landunit partition still sums to
    100% — which the caller re-checks straight after.
    """
    if 'PCT_CROP' not in ds or 'PCT_NATVEG' not in ds:
        return 0.0
    moved = float(ds['PCT_CROP'].values.sum())
    if moved <= 0:
        return 0.0
    ds['PCT_NATVEG'].values[:] = ds['PCT_NATVEG'].values + ds['PCT_CROP'].values
    ds['PCT_CROP'].values[:] = 0.0
    return moved


# The subset of VEG_VARS that partitions the gridcell into landunits; these
# must total 100% (ELM tolerance is 1e-14 on the normalized sum).
LANDUNIT_PCT_VARS = [
    'PCT_NATVEG', 'PCT_CROP',
    'PCT_URBAN', 'PCT_LAKE', 'PCT_WETLAND', 'PCT_GLACIER',
]

# ELM standard nlevsoi=10 node depths (center of each soil cell), m.
# Used for depth-aware mapping of variable-depth MCP horizons to
# ELM's fixed 10-level grid. Source: E3SM ELM clm_varpar.F90.
ELM_LEVEL_NODE_DEPTH_M = [
    0.00710063,  # ~7 mm
    0.02791720,  # ~28 mm
    0.06225257,  # ~62 mm
    0.11886794,  # ~119 mm
    0.21222644,  # ~21 cm
    0.36605059,  # ~37 cm
    0.61975915,  # ~62 cm
    1.03802647,  # ~1.04 m
    1.72763374,  # ~1.73 m
    2.86460919,  # ~2.86 m
]

SYNTHETIC_PROFILES = {
    'sandy': {
        'PCT_SAND': 75.0, 'PCT_CLAY':  8.0,
        'ORGANIC':  13.0, 'PCT_GRVL':  5.0,   # kg/m3 (was 1.0 %)
    },
    'loamy': {
        'PCT_SAND': 40.0, 'PCT_CLAY': 20.0,
        'ORGANIC':  39.0, 'PCT_GRVL':  2.0,   # kg/m3 (was 3.0 %)
    },
    'clayey': {
        'PCT_SAND': 20.0, 'PCT_CLAY': 45.0,
        'ORGANIC':  26.0, 'PCT_GRVL':  1.0,   # kg/m3 (was 2.0 %)
    },
}

SUBSTRATE_OPTIONS = {
    'template':    "Keep Station 2006 template values (unknown substrate)",
    'extrapolate': "Reuse deepest MCP layer's values",
    'sandy':       "Coarse, high-permeability (alluvial substrate proxy)",
    'clayey':      "Fine, low-permeability (aquitard proxy)",
}

# Loam-ish fallback when a layer's percentages are missing
_FALLBACK_LAYER = {
    'PCT_SAND': 40.0,
    'PCT_CLAY': 20.0,
    'ORGANIC':  39.0,   # kg/m3, not % — see _organic_kg_m3
    'PCT_GRVL':  2.0,
}

# Regex for parsing "sand=80.0%" style entries in derivation.inputs
_DERIVATION_PCT_RE = re.compile(
    r'\b(?P<name>sand|silt|clay|organic|gravel)\s*=\s*'
    r'(?P<val>-?[\d.]+)\s*%',
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────
# CONUS/global vegetation extraction (veg_source='conus')
#
# The shipped SURFACE_TEMPLATE (Surfacedata_Station_2006_.nc) is itself a
# single-point nearest-neighbor extraction from a global 0.5x0.5 mksurfdata
# file (verified: identical PCT_NAT_PFT to CONUS_SURFDATA_NC at the
# template's own lat/lon). generate_from_mcp(..., veg_source='conus') reruns
# that same extraction for the REQUESTED lat/lon instead of always reusing
# the template's site — see single_column/create_surfdata.py for the
# original technique this replicates.
# ─────────────────────────────────────────────────────────────────────
_conus_ds_cache: Dict[str, Any] = {}


def _conus_dataset(path: str = CONUS_SURFDATA_NC):
    """
    Lazily open + coordinate-index the global surfdata file. Cached by
    path across calls. Returns None (logging a warning once) if the file
    isn't available, so callers can fall back to template vegetation
    instead of crashing.
    """
    if path in _conus_ds_cache:
        return _conus_ds_cache[path]

    if not Path(path).exists():
        logger.warning(
            f"CONUS_SURFDATA_NC not found at {path} — veg_source='conus' "
            f"will fall back to template vegetation. Set the CONUS_SURFDATA_NC "
            f"env var to a global 0.5x0.5-degree mksurfdata-format file."
        )
        _conus_ds_cache[path] = None
        return None

    ds = xr.open_dataset(path, engine='netcdf4')
    n_lat = ds.sizes['lsmlat']
    n_lon = ds.sizes['lsmlon']
    # Raw mksurfdata files carry lsmlat/lsmlon as plain dims, not coordinate
    # values -- reconstruct the regular half-degree global grid so .sel(...,
    # method='nearest') can index by real lat/lon.
    lat_vals = np.linspace(-90 + 0.25, 90 - 0.25, n_lat)
    lon_vals = np.linspace(-180 + 0.25, 180 - 0.25, n_lon)
    ds = ds.assign_coords(
        lsmlat=xr.DataArray(lat_vals, dims='lsmlat'),
        lsmlon=xr.DataArray(lon_vals, dims='lsmlon'),
    )
    _conus_ds_cache[path] = ds
    return ds


def _extract_conus_veg(lat: float, lon: float,
                       path: str = CONUS_SURFDATA_NC
                       ) -> Optional[Dict[str, Any]]:
    """
    Nearest-gridcell extraction of VEG_VARS from the global surfdata file
    for (lat, lon). Returns None if the file is unavailable.
    """
    ds = _conus_dataset(path)
    if ds is None:
        return None

    cell = ds.sel(lsmlat=[lat], lsmlon=[lon], method='nearest')
    return {var: cell[var].values for var in VEG_VARS if var in cell}


# ─────────────────────────────────────────────────────────────────────
# SURFACE GENERATOR
# ─────────────────────────────────────────────────────────────────────
# Key spellings accepted for each quantity, most specific first. The MCP emits
# `organic_matter_pct` and `gravel_pct`; older passthrough payloads use the
# short forms; SSURGO column names appear when a profile is passed through raw.
_PCT_KEYS = {
    'sand':    ('sand_pct', 'sand', 'sandtotal_r'),
    'clay':    ('clay_pct', 'clay', 'claytotal_r'),
    'organic': ('organic_pct', 'organic', 'organic_matter_pct', 'om_pct', 'om_r'),
    'gravel':  ('gravel_pct', 'gravel', 'coarse_fragment_pct', 'frag_pct'),
}


class ELMSurfaceGenerator:
    """
    Generates ELM surface data NetCDF files for single-column simulations.

    Two modes:
        generate_from_mcp()   → native layered soil from SSURGO MCP data
                                + configurable substrate below SSURGO depth
        generate_synthetic()  → uniform synthetic texture for comparison
    """

    def __init__(self,
                 template_path: str = SURFACE_TEMPLATE,
                 output_dir:    str = SURFACE_OUTPUT_DIR):
        if not XARRAY_AVAILABLE:
            raise RuntimeError(
                "xarray + netCDF4 required for ELMSurfaceGenerator. "
                "Install with: pip install netcdf4 xarray"
            )

        self.template_path = Path(template_path)
        self.output_dir    = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.template_path.exists():
            raise FileNotFoundError(
                f"Surface template not found: {self.template_path}"
            )

    # ─────────────────────────────────────────────────────────
    # MODE 1 — From MCP geology data (layered SSURGO)
    # ─────────────────────────────────────────────────────────
    def generate_from_mcp(self,
                          lat:        float,
                          lon:        float,
                          mcp_data:   Dict[str, Any],
                          substrate:   str  = 'template',
                          veg_source:  str  = 'template',
                          soil_source: str  = 'profile',
                          force:       bool = False) -> str:
        """
        Generate surface file with depth-mapped soil from MCP horizons,
        plus configurable substrate for ELM levels beyond MCP coverage.

        veg_source controls PCT_NAT_PFT/PCT_CFT/LAI/SAI/HEIGHT (vegetation
        composition), which soil-mapping above never touches:
            'template' → keep the template site's vegetation (default,
                         same behavior as before this parameter existed)
            'conus'    → real per-location extraction from CONUS_SURFDATA_NC
                         (nearest gridcell); falls back to 'template'
                         with a logged warning if that file is unavailable

        soil_source controls PCT_SAND/PCT_CLAY/ORGANIC/PCT_GRVL:
            'profile' → map the supplied horizons onto ELM's levels
            'conus'  → leave the template's own soil untouched

        Why 'conus' exists, and when it is right. A warm start hands ELM the
        CONUS restart's moisture, which is equilibrated against the CONUS
        gridcell's SOIL. Overwriting that soil with SSURGO leaves the inherited
        water inconsistent with its own hydraulics, and year one is spent
        relaxing rather than simulating: on a 14-column Naches run, five
        columns drained MORE than their annual precipitation (one at 2.98x)
        and closure residuals reached 2491 mm. Keeping the donor's soil is what
        makes a warm start actually skip spin-up.

        This costs no spatial resolution — CONUS 1 km soil varies column to
        column (37-68 % sand, 40-95 kg/m3 organic across those same 14) — it
        trades SSURGO's survey fidelity for a self-consistent initial state.
        Use 'profile' when soil is the experimental axis — a controlled sweep
        that varies clay and sand at one site, which is what tools/
        make_soil_sweep.py does. It was called 'ssurgo' because a survey query
        was its usual input, and that name made a general capability look like
        one dataset's plumbing.
        """
        if substrate not in SUBSTRATE_OPTIONS:
            raise ValueError(
                f"Unknown substrate '{substrate}'. "
                f"Choose: {sorted(SUBSTRATE_OPTIONS.keys())}"
            )
        if veg_source not in ('template', 'conus'):
            raise ValueError(
                f"Unknown veg_source '{veg_source}'. Choose: template, conus"
            )
        if soil_source not in ('profile', 'conus'):
            raise ValueError(
                f"Unknown soil_source '{soil_source}'. Choose: profile, conus"
            )

        # soil signature differentiates DIFFERENT soils at the SAME (lat,lon)
        # — e.g. a controlled soil sweep at one site (else they'd share a file).
        soil_sig = hashlib.md5(
            json.dumps(mcp_data, sort_keys=True, default=str).encode()
        ).hexdigest()[:6]
        veg_tag = '' if veg_source == 'template' else f'_veg-{veg_source}'
        # in the filename so a swept-soil and a donor-soil surface for the same
        # column never collide in the cache. 'profile' keeps the empty tag it
        # had under its old name, so existing cached files stay valid.
        soil_tag = '' if soil_source == 'profile' else f'_soil-{soil_source}'
        output_path = (
            self.output_dir /
            f"Surfacedata_{lat:.4f}_{lon:.4f}_native_{substrate}"
            f"{veg_tag}{soil_tag}_{soil_sig}.nc"
        )

        if output_path.exists() and not force:
            logger.info(f"Cached surface file: {output_path.name}")
            return str(output_path)

        if soil_source == 'conus':
            # None at every level means "keep the template's value" in
            # _write_surface — i.e. the donor gridcell's own soil, which the
            # restart was equilibrated against. Coordinates are still
            # rewritten, which is the other reason this path exists.
            mcp_layers = []
            elm_levels = [None] * N_SOIL_LEVELS
        else:
            # Stage 1: parse MCP layers into a normalized form
            mcp_layers = self._parse_mcp_layers(mcp_data)

            # Stage 2: depth-aware mapping onto ELM's 10-level grid
            elm_levels = self._map_to_elm_levels(
                mcp_layers, substrate, lat, lon)

        logger.info(
            f"Generating native surface: lat={lat:.4f}, lon={lon:.4f} "
            f"(soil={soil_source}, substrate={substrate}, "
            f"n_mcp_layers={len(mcp_layers)})"
        )
        if mcp_layers:
            depth_top = mcp_layers[ 0].get('depth_top_m')
            depth_bot = mcp_layers[-1].get('depth_bot_m')
            logger.info(
                f"   MCP depth coverage: {depth_top}–{depth_bot} m"
            )

        return self._write_surface(
            output_path = output_path,
            lat         = lat,
            lon         = lon,
            elm_levels  = elm_levels,
            label       = f'native_{substrate}',
            veg_source  = veg_source,
        )

    # ─────────────────────────────────────────────────────────
    # MODE 2 — Synthetic uniform texture
    # ─────────────────────────────────────────────────────────
    def generate_synthetic(self,
                           lat:     float,
                           lon:     float,
                           texture: str,
                           force:   bool = False) -> str:
        if texture not in SYNTHETIC_PROFILES:
            raise ValueError(
                f"Unknown texture '{texture}'. "
                f"Choose: {sorted(SYNTHETIC_PROFILES.keys())}"
            )

        output_path = (
            self.output_dir /
            f"Surfacedata_{lat:.4f}_{lon:.4f}_{texture}.nc"
        )

        if output_path.exists() and not force:
            logger.info(f"Cached surface file: {output_path.name}")
            return str(output_path)

        profile    = SYNTHETIC_PROFILES[texture]
        elm_levels = [profile for _ in range(N_SOIL_LEVELS)]

        logger.info(
            f"Generating {texture} surface: lat={lat:.4f}, lon={lon:.4f} "
            f"(sand={profile['PCT_SAND']}% clay={profile['PCT_CLAY']}%)"
        )

        return self._write_surface(
            output_path = output_path,
            lat         = lat,
            lon         = lon,
            elm_levels  = elm_levels,
            label       = texture,
        )

    # ─────────────────────────────────────────────────────────
    # PRIVATE — Depth-aware MCP → ELM level mapping
    # ─────────────────────────────────────────────────────────
    def _map_to_elm_levels(self,
                           mcp_layers: List[Dict[str, Any]],
                           substrate:  str,
                           lat:        float,
                           lon:        float
                           ) -> List[Optional[Dict]]:
        """
        For each of the N_SOIL_LEVELS ELM levels, assign an MCP horizon
        (whose depth range contains the ELM node depth) or substrate fill.

        Returns:
            List of N_SOIL_LEVELS entries. Each is either a dict with
            PCT_SAND/PCT_CLAY/ORGANIC/PCT_GRVL keys, or None (meaning
            "keep the template's value at this level").
        """
        if not mcp_layers:
            return self._substrate_only(substrate, lat, lon)

        elm_levels: List[Optional[Dict]] = []
        deepest_mcp_layer: Optional[Dict] = None

        max_mcp_bottom = max(
            (l.get('depth_bot_m') for l in mcp_layers
             if l.get('depth_bot_m') is not None),
            default=None,
        )

        for node_depth in ELM_LEVEL_NODE_DEPTH_M:
            matched: Optional[Dict] = None
            for layer in mcp_layers:
                top = layer.get('depth_top_m')
                bot = layer.get('depth_bot_m')
                if top is None or bot is None:
                    continue
                if top <= node_depth <= bot:
                    matched = layer
                    break

            if matched is not None:
                elm_levels.append(matched)
                deepest_mcp_layer = matched
                continue

            # Past the bottom of MCP coverage → apply substrate
            elm_levels.append(self._make_substrate_layer(
                substrate, deepest_mcp_layer, lat, lon))

        n_from_mcp = sum(1 for lvl in elm_levels if lvl in mcp_layers)
        logger.info(
            f"   Depth mapping: {n_from_mcp}/{N_SOIL_LEVELS} ELM levels "
            f"from MCP horizons, "
            f"{N_SOIL_LEVELS - n_from_mcp} from substrate "
            f"(deepest MCP horizon ends at "
            f"{max_mcp_bottom if max_mcp_bottom is not None else '?'} m)"
        )

        return elm_levels

    def _make_substrate_layer(self,
                              substrate:    str,
                              deepest_mcp:  Optional[Dict[str, Any]],
                              lat:          float,
                              lon:          float
                              ) -> Optional[Dict[str, Any]]:
        """Build a single layer dict for an ELM level past MCP coverage."""
        if substrate == 'extrapolate':
            if deepest_mcp is None:
                logger.warning(
                    f"substrate='extrapolate' at lat={lat:.4f}, "
                    f"lon={lon:.4f} but no MCP layer to extrapolate from "
                    f"— keeping template value at this level"
                )
                return None
            return deepest_mcp
        if substrate in SYNTHETIC_PROFILES:
            return SYNTHETIC_PROFILES[substrate]
        return None

    def _substrate_only(self,
                        substrate: str,
                        lat:       float,
                        lon:       float) -> List[Optional[Dict]]:
        """When no MCP data, fill all 10 ELM levels per substrate."""
        if substrate == 'extrapolate':
            logger.warning(
                f"substrate='extrapolate' but MCP returned 0 layers "
                f"for lat={lat:.4f}, lon={lon:.4f} — falling back "
                f"to 'template' (no extrapolation source)."
            )
            substrate = 'template'
        if substrate == 'template':
            logger.warning(
                f"No MCP layers for lat={lat:.4f}, lon={lon:.4f}. "
                f"File will have TEMPLATE soil values (Station 2006) "
                f"with only lat/lon corrected. Consider "
                f"generate_synthetic() for a controlled profile."
            )
            return [None] * N_SOIL_LEVELS
        if substrate in SYNTHETIC_PROFILES:
            return [SYNTHETIC_PROFILES[substrate]] * N_SOIL_LEVELS
        return [None] * N_SOIL_LEVELS

    # ─────────────────────────────────────────────────────────
    # PRIVATE — Write surface NetCDF
    # ─────────────────────────────────────────────────────────
    def _write_surface(self,
                       output_path: Path,
                       lat:         float,
                       lon:         float,
                       elm_levels:  List[Optional[Dict]],
                       label:       str,
                       veg_source:  str = 'template') -> str:
        """
        Open template, write per-ELM-level soil + coordinates, save out.

        elm_levels is a list of N_SOIL_LEVELS entries; None means
        "keep the template's value at this level".

        veg_source: 'template' leaves PCT_NAT_PFT/PCT_CFT/LAI/SAI/HEIGHT as
        the template's values (default); 'conus' overwrites them with a real
        nearest-gridcell extraction from CONUS_SURFDATA_NC for (lat, lon).
        """
        ds     = xr.open_dataset(str(self.template_path), engine='netcdf4')
        ds_new = ds.copy(deep=True)

        for var in SOIL_VARS:
            if var not in ds_new:
                logger.warning(f"Template missing soil variable '{var}'")
                continue
            vals = ds_new[var].values.copy()
            for i, layer in enumerate(elm_levels):
                if i >= N_SOIL_LEVELS:
                    break
                if layer is None:
                    continue   # keep template value at this level
                vals[i, 0, 0] = layer.get(var, _FALLBACK_LAYER[var])
            ds_new[var].values[:] = vals

        if veg_source == 'conus':
            conus_veg = _extract_conus_veg(lat, lon)
            if conus_veg:
                applied = set()
                for var, arr in conus_veg.items():
                    if var not in ds_new:
                        continue
                    try:
                        ds_new[var].values[:] = arr
                        applied.add(var)
                    except Exception as e:
                        logger.warning(
                            f"Could not apply CONUS '{var}' to surface file "
                            f"(shape mismatch?): {e} — keeping template value"
                        )
                # The landunit partition must be all-or-nothing: a partial
                # overwrite mixes two sites' percentages and will not total
                # 100%. Fail loudly here (seconds) rather than in ELM's
                # check_sums_equal_1 (minutes into a job).
                lu_present = [v for v in LANDUNIT_PCT_VARS if v in ds_new]
                missed = [v for v in lu_present if v not in applied]
                if missed:
                    raise RuntimeError(
                        f"CONUS extraction applied only part of the landunit "
                        f"partition (missing {missed}). Mixing sites here makes "
                        f"PCT_* sum != 100% and ELM aborts in "
                        f"surfrd_get_data/check_sums_equal_1."
                    )

                moved = fold_crop_into_natveg(ds_new)
                if moved:
                    logger.info(
                        f"   Crop: folded {moved:.2f}% PCT_CROP into "
                        f"PCT_NATVEG (create_crop_landunit=.false.)")
                total = float(sum(
                    ds_new[v].values.sum() for v in lu_present))
                if abs(total - 100.0) > 1e-6:
                    raise RuntimeError(
                        f"Landunit percentages sum to {total:.6f}%, expected "
                        f"100% (lat={lat:.4f} lon={lon:.4f}). ELM would abort "
                        f"in check_sums_equal_1. Vars: {lu_present}"
                    )
                logger.info(
                    f"   Vegetation: CONUS extraction @ lat={lat:.4f} "
                    f"lon={lon:.4f} (landunit sum {total:.3f}%)"
                )
            else:
                logger.warning(
                    "veg_source='conus' requested but CONUS data unavailable "
                    "— keeping template vegetation"
                )

        # Update location — both data variables and dimension coords
        if 'LATIXY' in ds_new:
            ds_new['LATIXY'].values[:] = lat
        if 'LONGXY' in ds_new:
            ds_new['LONGXY'].values[:] = lon
        if 'lsmlat' in ds_new.coords:
            ds_new['lsmlat'].values[:] = lat
        if 'lsmlon' in ds_new.coords:
            ds_new['lsmlon'].values[:] = lon

        ds_new.attrs['latitude']     = float(lat)
        ds_new.attrs['longitude']    = float(lon)
        ds_new.attrs['soil_texture'] = label
        ds_new.attrs['veg_source']   = veg_source
        ds_new.attrs['generated_by'] = (
            f"ELMSurfaceGenerator lat={lat:.4f} lon={lon:.4f} "
            f"label={label} veg_source={veg_source} (depth-aware mapping)"
        )

        ds_new.to_netcdf(str(output_path), engine='netcdf4')
        ds.close()
        ds_new.close()

        logger.info(f"Saved surface file: {output_path.name}")
        return str(output_path)

    # ─────────────────────────────────────────────────────────
    # PRIVATE — Parse MCP soil layers (handles both schemas)
    # ─────────────────────────────────────────────────────────
    def _parse_mcp_layers(self,
                          mcp_data: Dict[str, Any]
                          ) -> List[Dict[str, Any]]:
        """
        Extract soil layers from MCP data into a normalized form.

        Each returned dict carries:
            PCT_SAND, PCT_CLAY, ORGANIC, PCT_GRVL  (percentages)
            depth_top_m, depth_bot_m               (meters; may be None)

        Layer order is preserved (surface-down). Falls back to loam
        defaults only when neither simple keys nor the derivation
        string yields a value.
        """
        raw_layers = (
            mcp_data.get('layers') or
            mcp_data.get('soil_layers') or
            []
        )
        if not raw_layers:
            return []

        parsed = []
        fallbacks_used: Dict[tuple, list] = {}
        for i, layer in enumerate(raw_layers):
            if not isinstance(layer, dict):
                logger.warning(f"MCP layer {i} is not a dict — skipping")
                continue

            sand = self._extract_pct(layer, 'sand')
            clay = self._extract_pct(layer, 'clay')
            org  = self._organic_kg_m3(layer)
            grvl = self._extract_pct(layer, 'gravel')

            depth_top_m = self._extract_depth(layer, top=True)
            depth_bot_m = self._extract_depth(layer, top=False)

            missing = [
                name for name, val in zip(
                    ['sand', 'clay', 'organic', 'gravel'],
                    [sand, clay, org, grvl],
                ) if val is None
            ]
            if missing:
                # Collected, not logged per layer: an 8-layer profile x 14
                # columns printed ~100 identical lines, which buries anything
                # that matters. Summarised once below.
                fallbacks_used.setdefault(tuple(missing), []).append(i)

            row = {
                'PCT_SAND': sand if sand is not None
                            else _FALLBACK_LAYER['PCT_SAND'],
                'PCT_CLAY': clay if clay is not None
                            else _FALLBACK_LAYER['PCT_CLAY'],
                'ORGANIC':  org  if org  is not None
                            else _FALLBACK_LAYER['ORGANIC'],
                'PCT_GRVL': grvl if grvl is not None
                            else _FALLBACK_LAYER['PCT_GRVL'],
                'depth_top_m': depth_top_m,
                'depth_bot_m': depth_bot_m,
            }

            sand_clay = row['PCT_SAND'] + row['PCT_CLAY']
            if sand_clay > 100.0:
                logger.warning(
                    f"MCP layer {i}: sand+clay = {sand_clay:.1f}% "
                    f"(>100%). Check MCP data quality."
                )

            parsed.append(row)

        for fields, layers in fallbacks_used.items():
            logger.warning(
                f"soil: {', '.join(fields)} not reported by SSURGO for "
                f"{len(layers)} of {len(raw_layers)} layers — loam fallback "
                f"used for those fields only (measured sand/clay retained)"
            )
        return parsed

    # ─────────────────────────────────────────────────────────
    # PRIVATE — Pull a percentage from a layer dict (dual format)
    # ─────────────────────────────────────────────────────────
    def _organic_kg_m3(self, layer: Dict[str, Any]) -> Optional[float]:
        """SSURGO organic matter (% by weight) -> ELM's ORGANIC (kg/m3).

        ELM divides this field by organic_max = 130 kg/m3 to get om_frac, which
        sets porosity, saturated conductivity and retention
        (SoilStateType.F90: om_frac = organic3d/organic_max). SSURGO reports a
        PERCENT, and it was being written straight in — so a soil the CONUS
        donor describes as 57 kg/m3 (om_frac 0.44) was handed to ELM as 3.0
        (om_frac 0.023): organic soil told to behave like mineral soil, in the
        field that decides how much water it holds.

        density = mass_fraction x bulk_density. Checked against the CONUS 1 km
        product at a Naches column: 3.0 % x 1.2 g/cc -> 36 kg/m3 against its
        57 kg/m3 — same magnitude, which is the point; the two come from
        different soil products and different layer boundaries.
        """
        om = self._extract_pct(layer, 'organic')
        if om is None:
            return None
        bd = self._extract(layer, ('bulk_density_gcc', 'bulk_density',
                                   'dbthirdbar_r'))
        if bd is None or bd <= 0:
            bd = _NOMINAL_BULK_DENSITY_GCC
        return round(om / 100.0 * bd * 1000.0, 2)

    def _extract_pct(self,
                     layer:  Dict[str, Any],
                     what:   str) -> Optional[float]:
        """
        Try multiple sources in order:
          1. Direct simple keys: <what>_pct, <what>
          2. derivation.inputs string: "sand=80.0% silt=17.5% clay=2.5%"
          3. None (caller uses fallback)
        """
        # 1. Direct keys, including the names the geology MCP actually emits.
        #    'organic' used to look only for organic_pct/organic while SSURGO
        #    arrives as organic_matter_pct, so every column silently discarded
        #    a real measurement and substituted the loam constant.
        direct = self._extract(layer, _PCT_KEYS.get(what, (f'{what}_pct', what)))
        if direct is not None:
            return direct

        # 2. Parse derivation.inputs string (MCP-native format)
        deriv = layer.get('derivation')
        if isinstance(deriv, dict):
            inputs_str = deriv.get('inputs')
            if isinstance(inputs_str, str) and inputs_str:
                for m in _DERIVATION_PCT_RE.finditer(inputs_str):
                    if m.group('name').lower() == what.lower():
                        try:
                            return float(m.group('val'))
                        except ValueError:
                            continue

        return None

    # ─────────────────────────────────────────────────────────
    # PRIVATE — Pull depth (in meters) from a layer dict
    # ─────────────────────────────────────────────────────────
    def _extract_depth(self,
                       layer: Dict[str, Any],
                       top:   bool) -> Optional[float]:
        """Accept either cm or m units; return meters or None."""
        if top:
            cm_keys = ['depth_top_cm']
            m_keys  = ['depth_top_m', 'top_m', 'top']
        else:
            cm_keys = ['depth_bot_cm', 'depth_bottom_cm']
            m_keys  = ['depth_bot_m', 'depth_bottom_m',
                       'bot_m', 'bottom']

        cm_val = self._extract(layer, cm_keys)
        if cm_val is not None:
            return cm_val / 100.0

        return self._extract(layer, m_keys)

    @staticmethod
    def _extract(layer: Dict[str, Any],
                 keys:  List[str]) -> Optional[float]:
        """Return first numeric value among `keys`, or None."""
        for k in keys:
            if k in layer:
                try:
                    return float(layer[k])
                except (TypeError, ValueError):
                    return None
        return None