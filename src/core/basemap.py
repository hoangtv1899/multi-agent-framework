#!/usr/bin/env python3
"""One hillshade backdrop, for every map this framework draws.

MOVED HERE FROM tools/plot_sampling_design.py (2026-08-12), unchanged in
behaviour, because the comparison figures now want the same ground under them
and two tile fetchers in one tree is one too many. plot_sampling_design.py calls
this; so does the ELM MCP's compare package, through core.static_wtd's precedent
of reading framework code from the server side.

Each tile is drawn with imshow at its own lon/lat bounds rather than through
cartopy. Cartopy would be the obvious tool and was tried first: its GeoAxes
mis-places itself inside a gridspec — the panel hangs off the canvas and its
title disappears — because the fixed aspect is applied at draw time, after the
layout engine has sized the cell. Drawing the tiles directly keeps every panel
on ordinary axes and every coordinate in degrees.

Tiles are square in Mercator and we draw them as latitude rectangles, so each
one is stretched by how much sec(lat) varies across it: 0.4% over a 0.3-degree
tile at 38N, well under a pixel.

A MISSING BACKDROP IS NOT AN ERROR. Compute nodes have no network, and a figure
without a backdrop beats one that cannot be drawn. `paste` returns False and the
caller decides what to put there instead.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

# Esri's World Hillshade: relief and nothing else. Measured against the
# alternatives over this basin it is both the most detailed and the least
# coloured — mean saturation 9 against World Shaded Relief's 18, at 33% more
# contrast — so it needs no muting, and muting is what made the first attempt
# bland. World Light Gray Base is neutral but carries no relief at all, which
# for a design stratified BY elevation throws away the context that matters.
TILES = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
         "Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}")
MAX_TILES = 48

# FETCHED ONCE PER PROCESS. A comparison run draws three map panels — swe,
# water table
# and streamflow — over the same basin at the same extent, so without this the
# same twenty tiles are pulled three times from somebody else's server for one
# figure set. Keyed by the full URL, so a second basin or a different tile
# service never reads another's pixels.
_CACHE: Dict[str, object] = {}


def tile_xy(lon: float, lat: float, z: int) -> Tuple[float, float]:
    """Web Mercator tile coordinates, the slippy-map convention every XYZ
    service uses."""
    n = 2.0 ** z
    r = math.radians(lat)
    return ((lon + 180.0) / 360.0 * n,
            (1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi) / 2 * n)


def tile_lonlat(x: float, y: float, z: int) -> Tuple[float, float]:
    n = 2.0 ** z
    return (x / n * 360.0 - 180.0,
            math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n)))))


def paste(ax, extent: List[float], tiles: Optional[str] = None,
          max_tiles: Optional[int] = None, quiet: bool = False) -> bool:
    """Paste an XYZ hillshade under a map panel, in plain lon/lat.

    extent is [west, east, south, north].

    `tiles` and `max_tiles` override the module defaults for one call, so a
    figure that needs a second backdrop — a country-scale locator wants outlines
    where a basin panel wants relief — asks for it here instead of reassigning
    this module's globals around the call.

    Returns True when something was drawn.
    """
    import io
    import ssl
    import urllib.request

    import certifi
    import numpy as np
    from PIL import Image

    # This node's OpenSSL points at an empty cert directory, so the default
    # context rejects every tile and the figure comes out backdrop-less with
    # only a note on stderr. certifi's bundle is the one Python packages already
    # trust, and naming it makes the figure render the same from any shell.
    ctx = ssl.create_default_context(cafile=certifi.where())
    url_fmt = tiles or TILES
    budget = MAX_TILES if max_tiles is None else max_tiles

    for z in range(11, 5, -1):
        x0, y0 = tile_xy(extent[0], extent[3], z)
        x1, y1 = tile_xy(extent[1], extent[2], z)
        grid = [(tx, ty) for tx in range(int(x0), int(x1) + 1)
                for ty in range(int(y0), int(y1) + 1)]
        if len(grid) <= budget:
            break
    try:
        for tx, ty in grid:
            url = url_fmt.format(z=z, x=tx, y=ty)
            img = _CACHE.get(url)
            if img is None:
                with urllib.request.urlopen(url, timeout=20, context=ctx) as r:
                    img = np.asarray(Image.open(io.BytesIO(r.read()))
                                     .convert("RGB"))
                _CACHE[url] = img
            w, n = tile_lonlat(tx, ty, z)
            e, s = tile_lonlat(tx + 1, ty + 1, z)
            ax.imshow(img, extent=[w, e, s, n], origin="upper", zorder=0,
                      interpolation="bilinear")
    except Exception as ex:                                  # noqa: BLE001
        if not quiet:
            print(f"   basemap unavailable ({type(ex).__name__}: "
                  f"{str(ex)[:60]})")
        return False
    if not quiet:
        print(f"   basemap: {len(grid)} tiles at zoom {z}")
    return True
