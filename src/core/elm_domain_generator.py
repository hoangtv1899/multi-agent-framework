#!/usr/bin/env python3
"""
ELM Domain Generator
src/core/elm_domain_generator.py

Generates single-column ELM domain files
for arbitrary lat/lon locations.

Domain file structure (confirmed):
    dims : nj=1, ni=1, nv=4
    xc   : (1,1)   ← center longitude
    yc   : (1,1)   ← center latitude
    xv   : (1,1,4) ← corner longitudes
    yv   : (1,1,4) ← corner latitudes
    mask : (1,1)   ← always 1
    frac : (1,1)   ← always 1.0
    area : (1,1)   ← cell area (keep from template)
"""
from pathlib import Path
from typing  import Optional

try:
    import numpy   as np
    import xarray  as xr
    import netCDF4
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────
DOMAIN_TEMPLATE = (
    "/global/homes/h/hvtran/RCSFA/1d_elm/"
    "input_files/Domainfile_station_2006_.nc"
)
DOMAIN_OUTPUT_DIR = (
    "/global/homes/h/hvtran/RCSFA/1d_elm/"
    "input_files/domains"
)

# ─────────────────────────────────────────────────────────────────────
# DOMAIN GENERATOR
# ─────────────────────────────────────────────────────────────────────
class ELMDomainGenerator:
    """
    Generates ELM domain files for any lat/lon.

    Uses existing domain file as template.
    Updates xc, yc, xv, yv coordinates only.
    All other fields (mask, frac, area) kept from template.

    Usage:
        gen  = ELMDomainGenerator()
        path = gen.generate(lat=46.3, lon=-119.3)
        # → .../domains/Domainfile_46.3000_-119.3000.nc
    """

    def __init__(self,
                 template_path: str = DOMAIN_TEMPLATE,
                 output_dir:    str = DOMAIN_OUTPUT_DIR):
        if not XARRAY_AVAILABLE:
            raise RuntimeError(
                "xarray + netCDF4 required.\n"
                "pip install netcdf4"
            )
        self.template_path = Path(template_path)
        self.output_dir    = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.template_path.exists():
            raise FileNotFoundError(
                f"Domain template not found: "
                f"{self.template_path}"
            )

    def generate(self,
                 lat:   float,
                 lon:   float,
                 force: bool = False) -> str:
        """
        Generate domain file for given lat/lon.

        Args:
            lat:   latitude  (-90 to 90)
            lon:   longitude (-180 to 180)
            force: regenerate even if file exists

        Returns:
            path to domain file (str)
        """
        output_path = (
            self.output_dir /
            f"Domainfile_{lat:.4f}_{lon:.4f}.nc"
        )

        # Return cached if exists
        if output_path.exists() and not force:
            print(f"   ✓ Domain file cached: "
                  f"{output_path.name}")
            return str(output_path)

        print(f"   → Generating domain file: "
              f"lat={lat:.4f}, lon={lon:.4f}")

        # Open template
        ds = xr.open_dataset(
            str(self.template_path),
            engine = 'netcdf4'
        )
        ds_new = ds.copy(deep=True)

        # Grid cell half-size (0.25 degree resolution)
        half = 0.25

        # Update center coordinates
        ds_new['xc'].values[:] = lon
        ds_new['yc'].values[:] = lat

        # Update corner coordinates
        # xv/yv shape: (nj=1, ni=1, nv=4)
        # corners: SW, SE, NE, NW
        ds_new['xv'].values[:] = np.array([
            [lon - half, lon + half,
             lon + half, lon - half]
        ]).reshape(1, 1, 4)

        ds_new['yv'].values[:] = np.array([
            [lat - half, lat - half,
             lat + half, lat + half]
        ]).reshape(1, 1, 4)

        # Update attributes
        ds_new.attrs['latitude']  = float(lat)
        ds_new.attrs['longitude'] = float(lon)
        ds_new.attrs['generated_by'] = (
            f"ELMDomainGenerator "
            f"lat={lat:.4f} lon={lon:.4f}"
        )

        # Save
        ds_new.to_netcdf(
            str(output_path),
            engine = 'netcdf4'
        )
        ds.close()
        ds_new.close()

        print(f"   ✓ Domain file saved: {output_path.name}")
        return str(output_path)