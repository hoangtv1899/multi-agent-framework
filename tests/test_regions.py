"""Regions on a gridded run: masks, the mask record and regional statistics.

    pytest tests/test_regions.py -v

The reductions are checked against sums written out here from the fixture's
formulas (elm_gridded_fixtures.expected), not against the reducer itself, and
the polygon test against shapely where shapely is installed.
"""
import math

import numpy as np
import pytest

import elm_gridded_fixtures as fx
import gridded as G
import regions as R

Y0 = 2015
YEARS = [2015, 2016]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    fx.write_daily(d, YEARS)
    return d


@pytest.fixture(scope="module")
def stream(run):
    return G.open_stream(run, "h1")


@pytest.fixture(scope="module")
def grid(stream):
    return stream.grid


def _lonlat(grid):
    return grid.lon, grid.lat


# ── vegetated cells and elevation bands ──────────────────────────────────────
class TestVegetatedAndBands:

    def test_vegetated_needs_the_whole_cell(self):
        p = np.array([100.0, 99.99, 0.0, np.nan, 100.0000001])
        np.testing.assert_array_equal(R.vegetated(p), [True, False, False, False, True])
        np.testing.assert_array_equal(R.vegetated(p, 50.0), [True, True, False, False, True])

    def test_bands_lower_edge_in_upper_edge_out(self):
        z = np.array([999.9, 1000.0, 1499.99, 1500.0, 2000.0, 3500.0, np.nan, -10.0])
        b = R.elevation_bands(z)
        assert list(b) == ["below 1000 m", "1000-1500 m", "1500-2000 m",
                           "2000 m and above"]
        np.testing.assert_array_equal(b["below 1000 m"], [1, 0, 0, 0, 0, 0, 0, 1])
        np.testing.assert_array_equal(b["1000-1500 m"], [0, 1, 1, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(b["1500-2000 m"], [0, 0, 0, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(b["2000 m and above"], [0, 0, 0, 0, 1, 1, 0, 0])
        assert not any(m[6] for m in b.values()), "no elevation, no band"
        assert sum(m.sum() for m in b.values()) == 7

    def test_the_historys_zero_topo_is_refused(self):
        with pytest.raises(R.RegionError, match="placeholder topo"):
            R.elevation_bands(np.zeros((4, 5)))

    def test_bad_edges(self):
        with pytest.raises(R.RegionError, match="increase"):
            R.elevation_bands(np.array([1.0, 2.0]), edges=(1500, 1000))
        with pytest.raises(R.RegionError, match="missing everywhere"):
            R.elevation_bands(np.array([np.nan, np.nan]))

    def test_custom_edges_label_themselves(self):
        b = R.elevation_bands(np.array([10.0, 600.0]), edges=(500,))
        assert list(b) == ["below 500 m", "500 m and above"]


# ── polygons ─────────────────────────────────────────────────────────────────
def _square(x0, y0, x1, y1, clockwise=False):
    r = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
    return r[::-1] if clockwise else r


class TestPolygons:

    def test_cell_centres_inside_a_square(self, grid):
        lon, lat = _lonlat(grid)
        # Around the 2 x 2 block of rows 1-2, cols 1-2 (half a cell beyond).
        x0, x1 = lon[0, 1] - 0.004, lon[0, 2] + 0.004
        y0, y1 = lat[1, 0] - 0.004, lat[2, 0] + 0.004
        m, a = R.polygon_mask(grid, {"type": "Polygon",
                                     "coordinates": [_square(x0, y0, x1, y1)]})
        want = np.zeros(grid.shape, bool)
        want[1:3, 1:3] = True
        np.testing.assert_array_equal(m, want)
        assert a["n_cells"] == 4
        assert a["mask_area_km2"] == pytest.approx(grid.weight_km2[want].sum())
        assert a["polygon_inside_grid"] is True

    def test_orientation_does_not_matter_and_0_360_is_read(self, grid):
        lon, lat = _lonlat(grid)
        x0, x1 = lon[0, 0] - 0.004, lon[0, 2] + 0.004
        y0, y1 = lat[0, 0] - 0.004, lat[1, 0] + 0.004
        a = R.polygon_mask(grid, {"type": "Polygon",
                                  "coordinates": [_square(x0, y0, x1, y1)]})[0]
        b = R.polygon_mask(grid, {"type": "Polygon",
                                  "coordinates": [_square(x0 + 360, y0, x1 + 360, y1,
                                                          clockwise=True)]})[0]
        np.testing.assert_array_equal(a, b)
        assert a.sum() == 6

    def test_a_hole_is_outside(self, grid):
        lon, lat = _lonlat(grid)
        outer = _square(lon[0, 0] - 0.004, lat[0, 0] - 0.004,
                        lon[0, 4] + 0.004, lat[3, 0] + 0.004)
        hole = _square(lon[0, 2] - 0.004, lat[1, 0] - 0.004,
                       lon[0, 2] + 0.004, lat[1, 0] + 0.004, clockwise=True)
        m, a = R.polygon_mask(grid, {"type": "Polygon", "coordinates": [outer, hole]})
        assert m.sum() == 19 and not m[1, 2]
        assert a["polygon_area_km2_spherical"] < R.polygon_area_km2(
            R.polygon_rings({"type": "Polygon", "coordinates": [outer]}))

    def test_multipolygon_and_feature_collection_take_the_union(self, grid):
        lon, lat = _lonlat(grid)
        p1 = [_square(lon[0, 0] - 0.004, lat[0, 0] - 0.004,
                      lon[0, 0] + 0.004, lat[0, 0] + 0.004)]
        p2 = [_square(lon[0, 4] - 0.004, lat[3, 0] - 0.004,
                      lon[0, 4] + 0.004, lat[3, 0] + 0.004)]
        mp = {"type": "MultiPolygon", "coordinates": [p1, p2]}
        fc = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": p1}},
            {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": p2}}]}
        a, b = R.polygon_mask(grid, mp)[0], R.polygon_mask(grid, fc)[0]
        np.testing.assert_array_equal(a, b)
        assert a.sum() == 2 and a[0, 0] and a[3, 4]

    def test_a_polygon_past_the_grid_edge_is_flagged(self, grid):
        lon, lat = _lonlat(grid)
        sq = _square(lon[0, 0] - 0.05, lat[0, 0] - 0.004, lon[0, 1], lat[0, 0] + 0.004)
        _, a = R.polygon_mask(grid, {"type": "Polygon", "coordinates": [sq]})
        assert a["polygon_inside_grid"] is False

    def test_not_a_polygon(self):
        with pytest.raises(R.RegionError, match="not a polygon"):
            R.polygon_rings({"type": "Point", "coordinates": [0, 0]})

    def test_against_shapely_on_random_polygons_with_holes(self):
        shapely = pytest.importorskip("shapely")
        from shapely.geometry import Polygon
        rng = np.random.default_rng(20261022)
        px = rng.uniform(-1.2, 1.2, 4000)
        py = rng.uniform(-1.2, 1.2, 4000)
        for trial in range(40):
            # Star-shaped rings with evenly spaced, jittered vertices: no gap
            # wider than 65 degrees, so no outer edge comes within 0.4 cos(32.5)
            # = 0.34 of the centre, and the hole (radius <= 0.25) stays clear.
            n = int(rng.integers(8, 40))
            ang = (np.arange(n) + rng.uniform(-0.2, 0.2, n)) * 2 * np.pi / n
            rad = rng.uniform(0.4, 1.0, n)
            ext = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
            m = int(rng.integers(4, 12))
            a2 = (np.arange(m) + rng.uniform(-0.2, 0.2, m)) * 2 * np.pi / m
            r2 = 0.25 * rng.uniform(0.3, 1.0, m)
            hole = np.c_[r2 * np.cos(a2), r2 * np.sin(a2)]
            poly = Polygon(ext, [hole])
            assert poly.is_valid
            want = shapely.contains_xy(poly, px, py)
            got = R.points_in_polygons(px, py, [[ext, hole]])
            assert np.array_equal(got, want), f"trial {trial}: {np.sum(got != want)} differ"


class TestPolygonArea:

    def test_a_one_degree_box_is_exact(self):
        r = R.EARTH_RADIUS_KM
        ring = np.array(_square(0.0, 0.0, 1.0, 1.0))
        want = r * r * math.radians(1.0) * (math.sin(math.radians(1.0)) - 0.0)
        assert R.ring_area_km2(ring) == pytest.approx(want, rel=1e-12)
        assert R.ring_area_km2(ring[::-1]) == pytest.approx(want, rel=1e-12)

    def test_a_hole_is_subtracted(self):
        r = R.EARTH_RADIUS_KM
        outer = _square(0.0, 0.0, 1.0, 1.0)
        hole = _square(0.25, 0.25, 0.75, 0.75)
        got = R.polygon_area_km2(R.polygon_rings(
            {"type": "Polygon", "coordinates": [outer, hole]}))
        box = lambda x, y0, y1: r * r * math.radians(x) * (
            math.sin(math.radians(y1)) - math.sin(math.radians(y0)))
        assert got == pytest.approx(box(1, 0, 1) - box(0.5, 0.25, 0.75), rel=1e-12)


# ── stations ─────────────────────────────────────────────────────────────────
class TestStations:

    def _topo(self, grid):
        z = np.arange(grid.n_cells, dtype=float).reshape(grid.shape) * 10.0 + 1000.0
        return z

    def test_containing_cell_and_half_open_edges(self, grid):
        lon, lat = _lonlat(grid)
        e = R.cell_edges(grid)
        sts = [
            {"id": "centre", "lat": lat[2, 3], "lon": lon[2, 3]},
            {"id": "west_edge", "lat": lat[1, 1], "lon": e["west"][1, 1]},
            {"id": "east_edge", "lat": lat[1, 1], "lon": e["east"][1, 1]},
            {"id": "south_edge", "lat": e["south"][2, 0], "lon": lon[2, 0]},
            {"id": "outside", "lat": lat[0, 0] - 0.05, "lon": lon[0, 0]},
            {"id": "0-360", "lat": lat[3, 4], "lon": lon[3, 4] + 360.0},
        ]
        out = {r["id"]: r for r in R.station_cells(grid, sts)}
        assert (out["centre"]["containing"]["row"], out["centre"]["containing"]["col"]) == (2, 3)
        assert out["west_edge"]["containing"]["col"] == 1
        assert out["east_edge"]["containing"]["col"] == 2
        assert out["south_edge"]["containing"]["row"] == 2
        assert out["outside"]["containing"] is None and not out["outside"]["inside_grid"]
        assert out["outside"]["nearest_centre_km"] == pytest.approx(5.56, abs=0.05)
        assert out["0-360"]["containing"]["flat"] == 19
        assert out["centre"]["containing"]["distance_km"] == pytest.approx(0.0, abs=1e-6)

    def test_elevation_match_takes_the_closest_height(self, grid):
        lon, lat = _lonlat(grid)
        z = self._topo(grid)
        z[0, 0] = z[2, 2] - 37.0
        st = {"id": "s", "lat": lat[2, 2], "lon": lon[2, 2], "elevation_m": z[2, 2] - 37.0}
        r = R.station_cells(grid, [st], topo=z, radius_km=10.0)[0]
        assert r["elevation_matched"]["flat"] == 0
        assert r["elevation_matched"]["elevation_difference_m"] == pytest.approx(0.0)
        st2 = dict(st, elevation_m=z[2, 2] + 1.0)
        r2 = R.station_cells(grid, [st2], topo=z, radius_km=10.0)[0]
        assert r2["elevation_matched"]["flat"] == 12 and r2["same_cell"] is True

    @pytest.mark.parametrize("near_is_higher", [True, False])
    def test_an_elevation_tie_goes_to_the_nearer_cell(self, grid, near_is_higher):
        """|dz| = 37 m at (2, 3), 0.64 km away, and at (0, 0), 2.3 km away."""
        lon, lat = _lonlat(grid)
        z = np.full(grid.shape, 5000.0)
        z[2, 3] = 1157.0 if near_is_higher else 1083.0
        z[0, 0] = 1083.0 if near_is_higher else 1157.0
        st = {"id": "s", "lat": lat[2, 2], "lon": lon[2, 2], "elevation_m": 1120.0}
        r = R.station_cells(grid, [st], topo=z, radius_km=10.0)[0]
        assert abs(r["elevation_matched"]["elevation_difference_m"]) == 37.0
        assert r["elevation_matched"]["flat"] == 13

    def test_every_record_has_the_same_keys(self, grid):
        lon, lat = _lonlat(grid)
        out = R.station_cells(grid, [{"id": "a", "lat": lat[1, 1], "lon": lon[1, 1]},
                                     {"id": "b", "lat": lat[0, 0] - 1, "lon": lon[0, 0]}])
        assert set(out[0]) == set(out[1])
        assert out[0]["n_within_radius"] is None and out[0]["same_cell"] is None

    def test_radius_and_within_limit_the_candidates(self, grid):
        lon, lat = _lonlat(grid)
        z = self._topo(grid)
        st = {"id": "s", "lat": lat[0, 0], "lon": lon[0, 0], "elevation_m": 1190.0}
        r = R.station_cells(grid, [st], topo=z, radius_km=0.5)[0]
        assert r["n_within_radius"] == 1 and r["elevation_matched"]["flat"] == 0
        within = np.zeros(grid.shape, bool)
        within[3, 4] = True
        r = R.station_cells(grid, [st], topo=z, radius_km=10.0, within=within)[0]
        assert r["elevation_matched"]["flat"] == 19

    def test_domain_corners_give_the_same_cells(self, grid):
        e = R.cell_edges(grid)
        xv = np.stack([e["west"], e["east"], e["east"], e["west"]], axis=-1) + 360.0
        yv = np.stack([e["south"], e["south"], e["north"], e["north"]], axis=-1)
        e2 = R.cell_edges(grid, corners=(xv, yv))
        for k in e:
            np.testing.assert_allclose(e2[k], e[k], atol=1e-9)
        bent = xv.copy()
        bent[1, 1, 2] += 0.001
        with pytest.raises(R.RegionError, match="rectangles"):
            R.cell_edges(grid, corners=(bent, yv))

    def test_cell_mask(self, grid):
        m = R.cell_mask(grid, 7)
        assert m.sum() == 1 and m[1, 2]


# ── masks from other gridded files ───────────────────────────────────────────
def _field_file(path, grid, name, data, coords="latlon", shift=0.0):
    import netCDF4
    nlat, nlon = grid.shape
    with netCDF4.Dataset(path, "w") as d:
        d.createDimension("lat", nlat)
        d.createDimension("lon", nlon)
        lead = data.shape[:-2]
        for k, n in enumerate(lead):
            d.createDimension(f"t{k}", n)
        if coords == "latlon":
            d.createVariable("lat", "f8", ("lat",))[:] = grid.lat_axis + shift
            d.createVariable("lon", "f8", ("lon",))[:] = grid.lon_axis + 360.0
        elif coords == "xy":
            d.createVariable("LATIXY", "f8", ("lat", "lon"))[:] = grid.lat + shift
            d.createVariable("LONGXY", "f8", ("lat", "lon"))[:] = grid.lon + 360.0
        v = d.createVariable(name, "f8", tuple(f"t{k}" for k in range(len(lead)))
                             + ("lat", "lon"))
        v[:] = data
    return path


class TestMaskFromField:

    def test_any_year_burned(self, grid, tmp_path):
        a = np.zeros((3,) + grid.shape)
        a[0, 0, 1] = 0.2
        a[2, 3, 4] = 0.1
        a[1, 2, 3] = np.nan
        p = _field_file(tmp_path / "b.nc", grid, "burned_fraction_year", a)
        m = R.mask_from_field(p, "burned_fraction_year", grid)
        assert m.sum() == 2 and m[0, 1] and m[3, 4] and not m[2, 3]

    def test_all_dates_covered(self, grid, tmp_path):
        n = np.full((4,) + grid.shape, 3.0)
        n[2, 1, 1] = 0.0
        n[3, 2, 2] = np.nan
        p = _field_file(tmp_path / "n.nc", grid, "gpp_n", n, coords="xy")
        m = R.mask_from_field(p, "gpp_n", grid, op=">", value=0, over="all")
        assert m.sum() == 18 and not m[1, 1] and not m[2, 2]

    def test_other_cells_are_refused(self, grid, tmp_path):
        p = _field_file(tmp_path / "s.nc", grid, "x", np.ones(grid.shape), shift=0.01)
        with pytest.raises(G.GriddedReadError, match="not the same cells"):
            R.mask_from_field(p, "x", grid)
        p2 = _field_file(tmp_path / "c.nc", grid, "x", np.ones(grid.shape), coords=None)
        with pytest.raises(R.RegionError, match="no cell centres"):
            R.mask_from_field(p2, "x", grid)

    def test_bad_arguments(self, grid, tmp_path):
        p = _field_file(tmp_path / "a.nc", grid, "x", np.ones(grid.shape))
        with pytest.raises(R.RegionError, match="op must"):
            R.mask_from_field(p, "x", grid, op="~")
        with pytest.raises(R.RegionError, match="has no"):
            R.mask_from_field(p, "y", grid)


# ── the mask record ──────────────────────────────────────────────────────────
class TestMaskSet:

    def _set(self, grid):
        ms = R.MaskSet(grid)
        west = np.zeros(grid.shape, bool)
        west[:, :2] = True
        top = np.zeros(grid.shape, bool)
        top[2:, :] = True
        ms.add("west", west, "cols 0-1", ["a.geojson"])
        ms.add("top rows", top, "rows 2-3", ["surf.nc"])
        return ms, west, top

    def test_add_combine_summary(self, grid):
        ms, west, top = self._set(grid)
        c = ms.combine("west, top", ["west", "top rows"])
        assert c.n_cells == 4 and c.rule == "[west] & [top rows]"
        assert c.sources == ("a.geojson", "surf.nc")
        x = ms.combine("west, not top", ["west"], exclude=["top rows"])
        assert x.n_cells == 4 and "not [top rows]" in x.rule
        s = ms.summary()
        assert s["west"]["area_km2"] == pytest.approx(grid.weight_km2[west].sum())
        assert list(s) == ["west", "top rows", "west, top", "west, not top"]

    def test_refusals(self, grid):
        ms, west, _ = self._set(grid)
        with pytest.raises(R.RegionError, match="already defined"):
            ms.add("west", west, "again")
        with pytest.raises(R.RegionError, match="mask shape"):
            ms.add("bad", np.ones((2, 2), bool), "wrong shape")
        with pytest.raises(R.RegionError, match="no region 'nope'"):
            ms["nope"]

    def test_the_file_round_trips(self, grid, tmp_path):
        ms, west, top = self._set(grid)
        ms.add("watershed, vegetated", west & top, "w & t")
        ms.add("watershed vegetated", west | top, "w | t")     # same variable name
        p = ms.to_netcdf(tmp_path / "masks" / "regions.nc")
        back = R.MaskSet.from_netcdf(p, grid)
        assert back.names() == ms.names()
        for n in ms.names():
            np.testing.assert_array_equal(back[n], ms[n])
            assert back.mask(n).rule == ms.mask(n).rule
            assert back.mask(n).sources == ms.mask(n).sources
            assert back.mask(n).area_km2 == pytest.approx(ms.mask(n).area_km2)
        assert not (tmp_path / "masks" / "regions.nc.tmp").exists()

    def test_the_file_is_tied_to_its_grid(self, grid, tmp_path):
        ms, _, _ = self._set(grid)
        p = ms.to_netcdf(tmp_path / "r.nc")
        shifted = G.Grid(lat=grid.lat + 0.1, lon=grid.lon, area_km2=grid.area_km2,
                         landfrac=grid.landfrac, dims=grid.dims,
                         lat_axis=grid.lat_axis + 0.1, lon_axis=grid.lon_axis)
        with pytest.raises(G.GriddedReadError, match="not the same cells"):
            R.MaskSet.from_netcdf(p, shifted)


# ── reducing to regional series ──────────────────────────────────────────────
def _labels(years):
    return [(y, d) for y in years for d in range(1, 366)]


def _oracle(grid, mask, parts, scale, labels, layer=0):
    """Area-weighted NaN-aware mean, written out from the formulas."""
    nlat, nlon = grid.shape
    w = grid.weight_km2
    out = []
    for (y, d) in labels:
        num = den = 0.0
        for j in range(nlat):
            for i in range(nlon):
                if not mask[j, i]:
                    continue
                v = sum(fx.expected(p, y, d, j, i, Y0, nlon, layer=layer) for p in parts)
                if math.isfinite(v):
                    num += w[j, i] * v * scale
                    den += w[j, i]
        out.append(num / den if den > 0 else float("nan"))
    return np.array(out)


@pytest.fixture(scope="module")
def masks(grid):
    ms = R.MaskSet(grid)
    allc = np.ones(grid.shape, bool)
    allc[-1, -2] = False                       # the zero-weight cell
    ms.add("all", allc, "every weighted cell")
    row2 = np.zeros(grid.shape, bool)
    row2[2, :] = True
    ms.add("row 2", row2, "row 2")
    ms.add("cell (1,1)", R.cell_mask(grid, 6), "one cell")
    east = np.zeros(grid.shape, bool)
    east[1:3, 3:5] = True
    ms.add("east block", east, "rows 1-2, cols 3-4")
    return ms


@pytest.fixture(scope="module")
def reduced(stream, masks):
    fields = [R.Field("GPP"), R.Field("ET", ("QVEGT", "QVEGE", "QSOIL")),
              R.Field("H2OSNO"), R.Field("TSA"), R.Field("FAREA_BURNED"),
              R.Field("SOILLIQ_2", ("SOILLIQ",), layer=1)]
    return R.reduce_regions(stream, fields, masks, YEARS)


class TestReduce:

    @pytest.mark.parametrize("region", ["all", "row 2", "cell (1,1)", "east block"])
    @pytest.mark.parametrize("fld,parts,scale", [
        ("GPP", ("GPP",), 86400.0), ("ET", ("QVEGT", "QVEGE", "QSOIL"), 86400.0),
        ("H2OSNO", ("H2OSNO",), 1.0), ("TSA", ("TSA",), 1.0),
        ("FAREA_BURNED", ("FAREA_BURNED",), 86400.0)])
    def test_means_match_the_formulas(self, reduced, masks, grid, region, fld, parts, scale):
        want = _oracle(grid, masks[region], parts, scale, _labels(YEARS))
        got = reduced.series(region, fld)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=0, equal_nan=True)

    def test_a_layer_of_a_layered_field(self, reduced, masks, grid):
        want = _oracle(grid, masks["east block"], ("SOILLIQ",), 1.0, _labels(YEARS),
                       layer=1)
        np.testing.assert_allclose(reduced.series("east block", "SOILLIQ_2"), want,
                                   rtol=1e-10)

    def test_labels_units_and_table(self, reduced, masks):
        assert len(reduced.keys) == 730
        assert reduced.labels[0] == "2015-01-01" and reduced.labels[-1] == "2016-12-31"
        assert reduced.units == {"GPP": "gC/m^2/d", "ET": "mm/d", "H2OSNO": "mm",
                                 "TSA": "K", "FAREA_BURNED": "proportion/d",
                                 "SOILLIQ_2": "kg/m2"}
        assert reduced.region_table["row 2"]["n_cells"] == 5
        assert reduced.field_parts["ET"] == ["QVEGT", "QVEGE", "QSOIL"]

    def test_valid_fraction_shows_cells_without_a_value(self, reduced, grid):
        vf = reduced.valid_fraction[("row 2", "GPP")]
        w = grid.weight_km2[2]
        assert vf[49] == pytest.approx(1.0 - w[2] / w.sum())      # NaN on 19 Feb 2015
        assert np.all(vf[np.arange(730) != 49] == 1.0)
        # Cell (1,1) has no GPP on day 100: the mean is NaN, not 0.
        one = reduced.series("cell (1,1)", "GPP")
        assert np.isnan(one[99]) and np.isfinite(one[98])
        assert reduced.valid_fraction[("cell (1,1)", "GPP")][99] == 0.0
        # A cell missing one part of a sum is missing for the sum.
        et = reduced.valid_fraction[("all", "ET")]
        assert et[9] < 1.0 and et[10] == 1.0

    def test_pieces_give_the_same_numbers(self, stream, masks, reduced):
        again = R.reduce_regions(stream, [R.Field("GPP"),
                                          R.Field("ET", ("QVEGT", "QVEGE", "QSOIL"))],
                                 masks, YEARS, max_records=37)
        for r in masks.names():
            for f in ("GPP", "ET"):
                np.testing.assert_array_equal(again.series(r, f), reduced.series(r, f))

    def test_a_subset_of_regions_reads_a_window(self, stream, masks, reduced):
        only = R.reduce_regions(stream, ["TSA"], masks, [2016], regions=["east block"])
        np.testing.assert_allclose(only.series("east block", "TSA"),
                                   reduced.series("east block", "TSA")[365:], rtol=1e-12)
        assert only.regions == ["east block"]

    def test_to_frame(self, reduced):
        df = reduced.to_frame()
        assert set(df.columns) == {"label", "region", "variable", "value", "units",
                                   "valid_area_fraction"}
        assert len(df) == 730 * 4 * 6

    def test_refusals(self, stream, masks, grid):
        with pytest.raises(R.RegionError, match="layered"):
            R.reduce_regions(stream, ["SOILLIQ"], masks, [2015])
        with pytest.raises(R.RegionError, match="no levels"):
            R.reduce_regions(stream, [R.Field("GPP", layer=0)], masks, [2015])
        with pytest.raises(R.RegionError, match="different units"):
            R.reduce_regions(stream, [R.Field("MIX", ("GPP", "TSA"))], masks, [2015])
        with pytest.raises(R.RegionError, match="repeat"):
            R.reduce_regions(stream, ["GPP", "GPP"], masks, [2015])
        ms = R.MaskSet(grid)
        zero = np.zeros(grid.shape, bool)
        zero[-1, -2] = True
        ms.add("zero weight", zero, "landfrac 0")
        with pytest.raises(R.RegionError, match="zero total weight"):
            R.reduce_regions(stream, ["GPP"], ms, [2015])
        ms.add("empty", np.zeros(grid.shape, bool), "nothing")
        with pytest.raises(R.RegionError, match="no cells"):
            R.reduce_regions(stream, ["GPP"], ms, [2015], regions=["empty"])
        with pytest.raises(G.GriddedReadError, match="not on stream"):
            R.reduce_regions(stream, ["NOPE"], masks, [2015])

    def test_masks_from_another_grid_are_refused(self, stream, grid):
        other = G.Grid(lat=grid.lat + 1, lon=grid.lon, area_km2=grid.area_km2,
                       landfrac=grid.landfrac, dims=grid.dims,
                       lat_axis=grid.lat_axis + 1, lon_axis=grid.lon_axis)
        ms = R.MaskSet(other)
        ms.add("x", np.ones(grid.shape, bool), "all")
        with pytest.raises(R.RegionError, match="different grid"):
            R.reduce_regions(stream, ["GPP"], ms, [2015])

    def test_a_short_year_is_refused_or_reported(self, tmp_path, masks):
        fx.write_daily(tmp_path, YEARS, variables=("GPP",))
        (tmp_path / "synth.elm.h1.2017-01-01-00000.nc").unlink()
        s = G.open_stream(tmp_path, "h1")
        ms = R.MaskSet(s.grid)
        ms.add("all", masks["all"], "all")
        with pytest.raises(G.GriddedReadError, match="1 of 365"):
            R.reduce_regions(s, ["GPP"], ms, [2016])
        rs = R.reduce_regions(s, ["GPP"], ms, [2015, 2016], require_complete=False)
        assert len(rs.keys) == 729
        by, why = rs.by_year("all", "GPP", [2015, 2016])
        assert by[2015] is not None and by[2016] is None
        assert why[2016].startswith("1 of 365 records not read (first 2016-12-31)")


# ── yearly numbers, anomalies, timing ────────────────────────────────────────
class TestByYear:

    def test_sums_means_max_min(self, reduced, masks, grid):
        daily = _oracle(grid, masks["row 2"], ("TSA",), 1.0, _labels(YEARS))
        by, why = reduced.by_year("row 2", "TSA", YEARS, how="sum")
        assert why == {}
        assert by[2015] == pytest.approx(daily[:365].sum(), rel=1e-12)
        assert by[2016] == pytest.approx(daily[365:].sum(), rel=1e-12)
        for how, fn in (("mean", np.mean), ("max", np.max), ("min", np.min)):
            got = reduced.by_year("row 2", "TSA", [2016], how=how)[0][2016]
            assert got == pytest.approx(fn(daily[365:]), rel=1e-12)

    def test_months_and_water_years(self, reduced, masks, grid):
        daily = _oracle(grid, masks["east block"], ("H2OSNO",), 1.0, _labels(YEARS))
        labels = _labels(YEARS)
        months = [fx.doy_to_md(d)[0] for _, d in labels]
        sel = [k for k, (y, _) in enumerate(labels) if y == 2016 and months[k] in (3, 4)]
        got = reduced.by_year("east block", "H2OSNO", [2016], how="mean", months=[3, 4])[0]
        assert got[2016] == pytest.approx(daily[sel].mean(), rel=1e-12)
        wy = [k for k, (y, _) in enumerate(labels)
              if (y == 2015 and months[k] >= 10) or (y == 2016 and months[k] <= 9)]
        got = reduced.by_year("east block", "H2OSNO", [2016], how="max", water_year=True)[0]
        assert got[2016] == pytest.approx(daily[wy].max(), rel=1e-12)
        none, why = reduced.by_year("east block", "H2OSNO", [2015], water_year=True)
        assert none[2015] is None and "92 of 365 records not read" in why[2015]

    def test_a_nan_record_voids_the_total(self, reduced):
        by, why = reduced.by_year("cell (1,1)", "GPP", [2015, 2016])
        assert by[2015] is None and why[2015] == "1 of 365 records have no value"
        assert by[2016] is not None

    def test_bad_how(self, reduced):
        with pytest.raises(R.RegionError, match="how must be"):
            reduced.by_year("all", "GPP", [2015], how="median")


class TestAnomalies:
    BY = {2015: 10.0, 2016: 12.0, 2017: 8.0, 2018: 11.0, 2019: 9.0, 2020: 10.0,
          2021: 5.0}

    def test_normal_anomaly_rank(self):
        a = R.anomalies(self.BY, [2016, 2017, 2018, 2019, 2020], [2015, 2021])
        assert a["normal"] == pytest.approx(10.0)
        assert a["normal_sd"] == pytest.approx(math.sqrt(2.5))
        assert (a["normal_min"], a["normal_max"]) == (8.0, 12.0)
        y21 = a["years"][2021]
        assert y21["anomaly"] == -5.0 and y21["anomaly_pct"] == pytest.approx(-50.0)
        assert y21["rank_lowest_first"] == 1 and y21["of"] == 7
        assert y21["position"] == "below the normal-years minimum"
        y15 = a["years"][2015]
        assert y15["anomaly_pct"] == 0.0 and y15["position"] == "within the normal-years range"

    def test_percent_off_and_zero_normal(self):
        a = R.anomalies(self.BY, [2016, 2017], [2021], percent=False)
        assert a["years"][2021]["anomaly_pct"] is None
        z = R.anomalies({2016: 1e-9, 2017: -1e-9, 2021: 3.0}, [2016, 2017], [2021])
        assert z["years"][2021]["anomaly_pct"] is None

    def test_a_missing_normal_year_is_refused(self):
        by = {**self.BY, 2018: None}
        with pytest.raises(R.RegionError, match="normal years without a value"):
            R.anomalies(by, [2016, 2017, 2018], [2021])

    def test_a_missing_target_year_is_none(self):
        by = {**self.BY, 2021: None}
        a = R.anomalies(by, [2016, 2017], [2021])
        assert a["years"][2021] is None


class TestTiming:

    def test_half_of_the_period_total(self, reduced, masks, grid):
        daily = _oracle(grid, masks["all"], ("QVEGT", "QVEGE", "QSOIL"), 86400.0,
                        _labels([2016]))
        months = [fx.doy_to_md(d)[0] for d in range(1, 366)]
        sel = np.array([m <= 9 for m in months])
        x = daily[sel]
        k = int(np.argmax(np.cumsum(x) / x.sum() >= 0.5))
        got = R.centre_of_timing(reduced, "all", "ET", 2016, range(1, 10))
        assert got["day_of_year"] == k + 1
        m, d = fx.doy_to_md(k + 1)
        assert got["label"] == f"2016-{m:02d}-{d:02d}"

    def test_incomplete_period_gives_none(self, reduced):
        assert R.centre_of_timing(reduced, "cell (1,1)", "GPP", 2015, range(1, 13)) is None
        assert R.centre_of_timing(reduced, "all", "ET", 2017, range(1, 10)) is None

    def test_day_of_year_and_volume(self):
        assert R.day_of_year((2015, 3, 1, 0, 0, 0), "noleap") == 60
        assert R.day_of_year((2016, 3, 1, 0, 0, 0), "standard") == 61
        np.testing.assert_allclose(R.to_volume_m3s(np.array([1e-5]), 2434.75, "mm/s"),
                                   [1e-5 * 2434.75e3])
        np.testing.assert_allclose(R.to_volume_m3s(np.array([0.864]), 2434.75, "mm/d"),
                                   [1e-5 * 2434.75e3])
        with pytest.raises(R.RegionError, match="mm/s or mm/d"):
            R.to_volume_m3s(np.array([1.0]), 1.0, "gC/m^2/d")


# ═════════════════════════════════════════════════════════════════════════════
# Added after the independent review of 2026-10-22: each class pins a defect
# the review demonstrated, or a branch no earlier test reached.
# ═════════════════════════════════════════════════════════════════════════════
def _grid_from_axes(lat_axis, lon_axis, area=None):
    lat_axis = np.asarray(lat_axis, float)
    lon_axis = G.to_pm180(np.asarray(lon_axis, float))
    lon, lat = np.meshgrid(lon_axis, lat_axis)
    a = np.ones(lat.shape) if area is None else area
    return G.Grid(lat=lat, lon=lon, area_km2=a, landfrac=np.ones(lat.shape),
                  dims=("lat", "lon"), lat_axis=lat_axis, lon_axis=lon_axis)


class TestReduceAfterReview:

    def test_a_part_shared_by_two_fields(self, stream, grid):
        """The single-part field comes first, so its total is the cached part
        itself; scaling it in place would corrupt the second field."""
        ms = R.MaskSet(grid)
        allc = np.ones(grid.shape, bool)
        ms.add("all", allc, "all")
        fields = [R.Field("GPP"), R.Field("GPP_per_s", ("GPP",), to_per_day=False),
                  R.Field("QVEGT"), R.Field("ET", ("QVEGT", "QVEGE", "QSOIL"))]
        rs = R.reduce_regions(stream, fields, ms, [2015])
        lab = _labels([2015])
        for name, parts, scale in (("GPP", ("GPP",), 86400.0), ("GPP_per_s", ("GPP",), 1.0),
                                   ("QVEGT", ("QVEGT",), 86400.0),
                                   ("ET", ("QVEGT", "QVEGE", "QSOIL"), 86400.0)):
            np.testing.assert_allclose(rs.series("all", name),
                                       _oracle(grid, allc, parts, scale, lab),
                                       rtol=1e-10, equal_nan=True, err_msg=name)
        assert rs.units["GPP_per_s"] == "gC/m^2/s" and rs.units["GPP"] == "gC/m^2/d"
        assert rs.per == {"GPP": "day", "GPP_per_s": "second", "QVEGT": "day", "ET": "day"}

    def test_repeats_and_empty_lists_are_refused(self, stream, masks):
        with pytest.raises(R.RegionError, match="region names repeat"):
            R.reduce_regions(stream, ["TSA"], masks, [2015],
                             regions=["row 2", "east block", "row 2"])
        with pytest.raises(R.RegionError, match="years repeat"):
            R.reduce_regions(stream, ["TSA"], masks, [2015, 2016, 2015])
        with pytest.raises(R.RegionError, match="no regions"):
            R.reduce_regions(stream, ["TSA"], masks, [2015], regions=[])
        with pytest.raises(R.RegionError, match="at least 1"):
            R.reduce_regions(stream, ["TSA"], masks, [2015], max_records=0)

    def test_years_come_out_in_calendar_order(self, stream, masks, reduced):
        rs = R.reduce_regions(stream, ["TSA"], masks, [2016, 2015], regions=["row 2"])
        assert rs.labels[0] == "2015-01-01" and rs.labels[-1] == "2016-12-31"
        np.testing.assert_array_equal(rs.series("row 2", "TSA"),
                                      reduced.series("row 2", "TSA"))

    def test_the_subdaily_default_pieces(self, tmp_path):
        fx.write_daily(tmp_path, YEARS, stream="h2", steps_per_day=8)
        h2 = G.open_stream(tmp_path, "h2")
        ms = R.MaskSet(h2.grid)
        allc = np.ones(h2.grid.shape, bool)
        allc[-1, -2] = False
        ms.add("all", allc, "all")
        rs = R.reduce_regions(h2, ["TSA", "STEPCODE"], ms, [2016])
        assert len(rs.keys) == 2920 and rs.labels[-1] == "2016-12-31 21:00"
        np.testing.assert_allclose(rs.series("all", "STEPCODE"), np.arange(2921, 5841),
                                   rtol=1e-12)
        daily = _oracle(h2.grid, allc, ("TSA",), 1.0, _labels([2016]))
        np.testing.assert_allclose(rs.series("all", "TSA"), np.repeat(daily, 8), rtol=1e-10)

    def test_fill_in_landfrac_weighs_nothing(self, tmp_path):
        g = fx.grid()
        g["landfrac"][0, 1] = 1e36
        fx.write_daily(tmp_path, [2015], g=g, variables=("TSA",))
        s = G.open_stream(tmp_path, "h1")
        ms = R.MaskSet(s.grid)
        row0 = np.zeros(s.grid.shape, bool)
        row0[0] = True
        ms.add("row 0", row0, "row 0")
        rs = R.reduce_regions(s, ["TSA"], ms, [2015])
        w = s.grid.weight_km2
        assert w[0, 1] == 0.0 and w[0, 0] > 0
        np.testing.assert_allclose(rs.series("row 0", "TSA"),
                                   _oracle(s.grid, row0, ("TSA",), 1.0, _labels([2015])),
                                   rtol=1e-10)


class TestMaskInputs:

    def test_stored_masks_cannot_be_edited_in_place(self, grid):
        ms = R.MaskSet(grid)
        m = np.zeros(grid.shape, bool)
        m[1, 1] = True
        ms.add("one cell", m, "cell (1,1)")
        with pytest.raises(ValueError):
            ms["one cell"][:] = True
        ws = ms["one cell"]
        with pytest.raises(ValueError):
            ws |= True
        m[2, 2] = True                      # the caller's own array is not the region
        assert ms.mask("one cell").n_cells == 1 and ms["one cell"].sum() == 1

    def test_nan_and_fill_are_not_inside(self, grid):
        ms = R.MaskSet(grid)
        f = np.full(grid.shape, np.nan)
        f[0, :2] = 1.0
        with pytest.raises(R.RegionError, match="0/1 or True/False only; found nan"):
            ms.add("float", f, "1/NaN")
        mk = np.ma.masked_array(np.ones(grid.shape, np.int8),
                                mask=np.ones(grid.shape, bool))
        mk.mask[0, :2] = False
        with pytest.raises(R.RegionError, match="masked"):
            ms.add("masked", mk, "1/fill")
        ok = ms.add("ints", np.where(np.isnan(f), 0, 1), "0/1")
        assert ok.n_cells == 2
        plain = np.ma.masked_array(np.zeros(grid.shape, int), mask=False)
        assert ms.add("unmasked", plain, "no masked cells").n_cells == 0


class TestByYearAcrossCadences:
    """FAREA_BURNED is a constant 1e-10 (1 + cell) per second in every record,
    so whatever the cadence the year's total is 365 days of it."""

    PER_DAY = float(np.float32(1e-10 * 4)) * 86400.0          # cell 3

    def test_monthly_records_count_for_their_days(self, tmp_path):
        fx.write_monthly(tmp_path, YEARS, variables=("GPP", "TSA", "CELLID",
                                                     "FAREA_BURNED"))
        h0 = G.open_stream(tmp_path, "h0")
        ms = R.MaskSet(h0.grid)
        ms.add("cell 3", R.cell_mask(h0.grid, 3), "one cell")
        rs = R.reduce_regions(h0, ["FAREA_BURNED", "TSA"], ms, [2016])
        got = rs.by_year("cell 3", "FAREA_BURNED", [2016])[0][2016]
        assert got == pytest.approx(365 * self.PER_DAY, rel=1e-6)
        days = np.array(fx.MONTH_DAYS, float)
        mids = [sum(fx.MONTH_DAYS[:m - 1]) + 15 for m in range(1, 13)]
        tsa = np.array([fx.expected("TSA", 2016, d, 0, 3, Y0, 5) for d in mids])
        mean = rs.by_year("cell 3", "TSA", [2016], how="mean")[0][2016]
        assert mean == pytest.approx((tsa * days).sum() / days.sum(), rel=1e-12)
        assert mean != pytest.approx(tsa.mean(), rel=1e-9)
        c = R.centre_of_timing(rs, "cell 3", "FAREA_BURNED", 2016, range(1, 13))
        assert c == {"label": "2016-07", "day_of_year": 182}  # 181/365 < 0.5 <= 212/365

    def test_three_hourly_records_count_for_their_eighth_of_a_day(self, tmp_path):
        n = 8
        tb = [(-fx.DT, 0.0)] + [(k / n, (k + 1) / n) for k in range(365 * n)]
        labels = [None] + [(2015, k // n + 1, k % n) for k in range(365 * n)]
        fx._write(tmp_path / "x.elm.h2.2015-01-01-00000.nc", fx.grid(), 2015, tb, labels,
                  ["FAREA_BURNED"], steps_per_day=n)
        s = G.open_stream(tmp_path, "h2")
        ms = R.MaskSet(s.grid)
        ms.add("cell 3", R.cell_mask(s.grid, 3), "one cell")
        rs = R.reduce_regions(s, ["FAREA_BURNED"], ms, [2015])
        got = rs.by_year("cell 3", "FAREA_BURNED", [2015])[0][2015]
        assert got == pytest.approx(365 * self.PER_DAY, rel=1e-6)

    def test_a_total_of_a_rate_left_per_second_is_refused(self, stream, masks):
        rs = R.reduce_regions(stream, [R.Field("GPP_s", ("GPP",), to_per_day=False)],
                              masks, [2016], regions=["row 2"])
        with pytest.raises(R.RegionError, match="per second"):
            rs.by_year("row 2", "GPP_s", [2016])
        assert rs.by_year("row 2", "GPP_s", [2016], how="mean")[0][2016] > 0

    def test_months_within_a_water_year(self, reduced, masks, grid):
        daily = _oracle(grid, masks["east block"], ("H2OSNO",), 1.0, _labels(YEARS))
        lab = _labels(YEARS)
        mon = [fx.doy_to_md(d)[0] for _, d in lab]
        sel = [k for k, (y, _) in enumerate(lab)
               if (y == 2015 and mon[k] in (10, 11, 12)) or (y == 2016 and mon[k] in (1, 2, 3))]
        got = reduced.by_year("east block", "H2OSNO", [2016], how="mean",
                              months=[10, 11, 12, 1, 2, 3], water_year=True)[0][2016]
        assert got == pytest.approx(daily[sel].mean(), rel=1e-12)


class TestDateline:
    """A regional grid running across 180 E: 170.5-189.5 E, 60.5-67.5 N."""

    @pytest.fixture
    def g(self):
        return _grid_from_axes(np.arange(60.5, 68.0, 1.0), np.arange(170.5, 190.0, 1.0))

    def _truth(self, g):
        t = np.zeros(g.shape, bool)
        t[2:6, 6:14] = True                   # 176-184 E, 62-66 N: 32 cells
        return t

    def _box_area(self, x0, x1, y0, y1):
        r = R.EARTH_RADIUS_KM
        return r * r * math.radians(x1 - x0) * (math.sin(math.radians(y1)) -
                                                math.sin(math.radians(y0)))

    def test_a_box_written_in_0_360(self, g):
        box = {"type": "Polygon", "coordinates": [
            [[176, 62], [184, 62], [184, 66], [176, 66], [176, 62]]]}
        m, a = R.polygon_mask(g, box)
        np.testing.assert_array_equal(m, self._truth(g))
        assert a["polygon_area_km2_spherical"] == pytest.approx(
            self._box_area(176, 184, 62, 66), rel=1e-12)
        assert a["polygon_inside_grid"] is True

    def test_a_box_split_at_180_as_rfc7946_asks(self, g):
        mp = {"type": "MultiPolygon", "coordinates": [
            [[[176, 62], [180, 62], [180, 66], [176, 66], [176, 62]]],
            [[[-180, 62], [-176, 62], [-176, 66], [-180, 66], [-180, 62]]]]}
        m, a = R.polygon_mask(g, mp)
        np.testing.assert_array_equal(m, self._truth(g))
        assert a["polygon_area_km2_spherical"] == pytest.approx(
            self._box_area(176, 184, 62, 66), rel=1e-12)

    def test_stations_pair_across_180(self, g):
        sts = [{"id": "Anadyr", "lat": 64.73, "lon": 177.5},
               {"id": "just west", "lat": 64.2, "lon": 179.9},
               {"id": "just east", "lat": 64.2, "lon": -179.8},
               {"id": "east as 0-360", "lat": 64.2, "lon": 180.2}]
        e = R.cell_edges(g)
        xv = np.stack([e["west"], e["east"], e["east"], e["west"]], axis=-1) % 360.0
        yv = np.stack([e["south"], e["south"], e["north"], e["north"]], axis=-1)
        for corners in (None, (xv, yv)):
            out = {r["id"]: r["containing"] for r in R.station_cells(g, sts, corners=corners)}
            assert (out["Anadyr"]["row"], out["Anadyr"]["col"]) == (4, 7)
            assert out["just west"]["col"] == 9
            assert out["just east"]["col"] == 10 and out["east as 0-360"]["col"] == 10

    def test_a_global_grid(self):
        g = _grid_from_axes(np.arange(-89.75, 90.0, 0.5), np.arange(0.25, 360.0, 0.5))
        sts = [{"id": "Manaus", "lat": -3.10, "lon": -60.02},
               {"id": "Fiji", "lat": -17.8, "lon": 178.4},
               {"id": "at 179.9", "lat": 10.1, "lon": 179.9},
               {"id": "at 180.1", "lat": 10.1, "lon": 180.1}]
        e = R.cell_edges(g)
        xv = np.stack([e["west"], e["east"], e["east"], e["west"]], axis=-1) % 360.0
        yv = np.stack([e["south"], e["south"], e["north"], e["north"]], axis=-1)
        for corners in (None, (xv, yv)):
            for r in R.station_cells(g, sts, corners=corners):
                c = r["containing"]
                assert c is not None, r["id"]
                d = R.haversine_km(r["lat"], r["lon"], g.lat, g.lon)
                assert c["flat"] == int(np.argmin(d)), r["id"]
            # Exactly on the edge between the cells centred 299.75 and 300.25
            # E: the half-open rule gives it to the eastern one.
            on = R.station_cells(g, [{"id": "edge", "lat": -3.1, "lon": -60.0}],
                                 corners=corners)[0]["containing"]
            assert (on["row"], on["col"]) == (173, 600)


class TestPolygonsAfterReview:

    def test_nested_features_take_the_union(self, grid):
        lon, lat = _lonlat(grid)
        outer = [_square(lon[0, 0] - 0.004, lat[0, 0] - 0.004,
                         lon[0, 4] + 0.004, lat[3, 0] + 0.004)]
        inner = [_square(lon[0, 1] - 0.004, lat[1, 0] - 0.004,
                         lon[0, 3] + 0.004, lat[2, 0] + 0.004)]
        fc = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": outer}},
            {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": inner}}]}
        assert R.polygon_mask(grid, fc)[0].all()

    def test_projected_coordinates_are_refused(self):
        with pytest.raises(R.RegionError, match="not longitude/latitude"):
            R.polygon_rings({"type": "Polygon", "coordinates": [
                [[500000, 5100000], [510000, 5100000], [510000, 5110000],
                 [500000, 5100000]]]})

    def test_the_audit_percentage(self, grid):
        lon, lat = _lonlat(grid)
        x0, x1 = lon[0, 1] - 0.004, lon[0, 2] + 0.004
        y0, y1 = lat[1, 0] - 0.004, lat[2, 0] + 0.004
        _, a = R.polygon_mask(grid, {"type": "Polygon",
                                     "coordinates": [_square(x0, y0, x1, y1)]})
        r = R.EARTH_RADIUS_KM
        box = r * r * math.radians(x1 - x0) * (math.sin(math.radians(y1)) -
                                               math.sin(math.radians(y0)))
        w = grid.weight_km2[1:3, 1:3].sum()
        assert a["mask_minus_polygon_pct"] == pytest.approx(100.0 * (w - box) / box,
                                                            rel=1e-6)


class TestOneRowGrid:

    def test_polygon_mask_works_and_stations_ask_for_corners(self):
        g = _grid_from_axes([46.5], 238.40416 + 0.0083333 * np.arange(5))
        lon, lat = g.lon, g.lat
        m, a = R.polygon_mask(g, {"type": "Polygon", "coordinates": [
            _square(lon[0, 1] - 0.004, 46.49, lon[0, 2] + 0.004, 46.51)]})
        assert m.sum() == 2 and a["polygon_inside_grid"] is None
        st = [{"id": "s", "lat": 46.5, "lon": lon[0, 3]}]
        with pytest.raises(R.RegionError, match="domain file's corners"):
            R.station_cells(g, st)
        xv = np.stack([lon - 0.004, lon + 0.004, lon + 0.004, lon - 0.004], axis=-1)
        yv = np.stack([lat - 0.004, lat - 0.004, lat + 0.004, lat + 0.004], axis=-1)
        assert R.station_cells(g, st, corners=(xv, yv))[0]["containing"]["col"] == 3


class TestMaskFromFieldAfterReview:

    def test_a_field_stored_lon_lat_is_refused(self, tmp_path):
        import netCDF4
        fg = fx.grid(5, 5)
        g = _grid_from_axes(fg["lat"], fg["lon"])
        p = tmp_path / "t.nc"
        with netCDF4.Dataset(p, "w") as d:
            d.createDimension("lat", 5)
            d.createDimension("lon", 5)
            d.createVariable("lat", "f8", ("lat",))[:] = fg["lat"]
            d.createVariable("lon", "f8", ("lon",))[:] = fg["lon"]
            v = d.createVariable("burned", "f8", ("lon", "lat"))
            a = np.zeros((5, 5))
            a[3, 0] = 1.0                       # (lon 3, lat 0) = row 0, col 3
            v[:] = a
        with pytest.raises(R.RegionError, match="last dimensions must be"):
            R.mask_from_field(p, "burned", g)

    def test_packed_fill_is_found_before_unpacking(self, grid, tmp_path):
        import netCDF4
        p = tmp_path / "pk.nc"
        with netCDF4.Dataset(p, "w") as d:
            d.createDimension("year", 3)
            d.createDimension("lat", grid.shape[0])
            d.createDimension("lon", grid.shape[1])
            d.createVariable("lat", "f8", ("lat",))[:] = grid.lat_axis
            d.createVariable("lon", "f8", ("lon",))[:] = grid.lon_axis
            v = d.createVariable("bf", "i2", ("year", "lat", "lon"),
                                 fill_value=np.int16(-32767))
            v.scale_factor = 0.0001
            v.set_auto_maskandscale(False)
            raw = np.zeros((3,) + grid.shape, np.int16)
            raw[:, 0, 0] = -32767                # no data in any year
            raw[1, 2, 2] = 500                   # 5 % burned in one year
            v[:] = raw
        unburned = R.mask_from_field(p, "bf", grid, op="<", value=0.01, over="all")
        assert not unburned[0, 0] and not unburned[2, 2] and unburned.sum() == 18
        burned = R.mask_from_field(p, "bf", grid, op=">", value=0.0)
        assert burned.sum() == 1 and burned[2, 2]

    def test_nan_never_selects_unless_asked(self, grid, tmp_path):
        import netCDF4
        a = np.zeros(grid.shape)
        a[0, 1] = 0.3
        a[1, 1] = np.nan
        a[2, 2] = 1e36
        p = tmp_path / "f.nc"
        with netCDF4.Dataset(p, "w") as d:
            d.createDimension("lat", grid.shape[0])
            d.createDimension("lon", grid.shape[1])
            d.createVariable("lat", "f8", ("lat",))[:] = grid.lat_axis
            d.createVariable("lon", "f8", ("lon",))[:] = grid.lon_axis + 360.0
            d.createVariable("x", "f8", ("lat", "lon"))[:] = a
        m = R.mask_from_field(p, "x", grid, op="!=", value=0.0)
        assert m.sum() == 1 and m[0, 1]
        m2 = R.mask_from_field(p, "x", grid, op="!=", value=0.0, nan_as=True)
        assert m2.sum() == 3 and m2[1, 1] and m2[2, 2]


class TestAnomalyTies:

    def test_tied_years_share_a_rank_whatever_the_order(self):
        by = {2015: 0.0, 2016: 0.0, 2017: 0.031, 2018: 0.0, 2019: 0.0, 2020: 0.012,
              2021: 0.0}
        for d in (by, dict(reversed(list(by.items())))):
            a = R.anomalies(d, [2016, 2017, 2018, 2019, 2020], [2015, 2021], percent=False)
            assert a["years"][2015]["rank_lowest_first"] == 1
            assert a["years"][2021]["rank_lowest_first"] == 1
            assert a["years"][2015]["tied_with"] == [2016, 2018, 2019, 2021]

    def test_a_small_but_real_normal_still_gets_a_percentage(self):
        a = R.anomalies({2016: 4e-7, 2017: 6e-7, 2021: 1e-6}, [2016, 2017], [2021])
        assert a["years"][2021]["anomaly_pct"] == pytest.approx(100.0)


class TestReadSurface:

    def _surface(self, path, grid, topo, shift=0.0, with_xy=True, names=("TOPO", "PCT_NATVEG")):
        import netCDF4
        nlat, nlon = grid.shape
        with netCDF4.Dataset(path, "w") as d:
            d.createDimension("lsmlat", nlat)
            d.createDimension("lsmlon", nlon)
            if with_xy:
                d.createVariable("LATIXY", "f8", ("lsmlat", "lsmlon"))[:] = grid.lat + shift
                d.createVariable("LONGXY", "f8", ("lsmlat", "lsmlon"))[:] = grid.lon + 360.0
            if "TOPO" in names:
                d.createVariable("TOPO", "f4", ("lsmlat", "lsmlon"))[:] = topo
            if "PCT_NATVEG" in names:
                d.createVariable("PCT_NATVEG", "f8", ("lsmlat", "lsmlon"))[:] = \
                    np.full(grid.shape, 100.0)
        return path

    def test_fill_cells_and_the_coordinate_check(self, grid, tmp_path):
        topo = np.arange(grid.n_cells, dtype=float).reshape(grid.shape) * 150.0 + 300.0
        topo[1, 3] = 1e36
        got = R.read_surface(self._surface(tmp_path / "s.nc", grid, topo), grid)
        assert np.isnan(got["TOPO"][1, 3])
        t2 = topo.copy()
        t2[1, 3] = np.nan
        np.testing.assert_allclose(got["TOPO"], np.float32(t2), equal_nan=True)
        assert not any(m[1, 3] for m in R.elevation_bands(got["TOPO"]).values())
        with pytest.raises(G.GriddedReadError, match="not the same cells"):
            R.read_surface(self._surface(tmp_path / "sh.nc", grid, topo, shift=0.01), grid)
        with pytest.raises(R.RegionError, match="has no PCT_NATVEG"):
            R.read_surface(self._surface(tmp_path / "nv.nc", grid, topo, names=("TOPO",)),
                           grid)
        with pytest.raises(R.RegionError, match="no LATIXY/LONGXY"):
            R.read_surface(self._surface(tmp_path / "nx.nc", grid, topo, with_xy=False),
                           grid)


class TestCombineSources:

    def test_the_excluded_regions_sources_are_kept(self, grid):
        ms = R.MaskSet(grid)
        w = np.zeros(grid.shape, bool)
        w[:, :2] = True
        t = np.zeros(grid.shape, bool)
        t[2:, :] = True
        ms.add("west", w, "w", ["a.geojson"])
        ms.add("top", t, "t", ["surf.nc"])
        x = ms.combine("west, not top", ["west"], exclude=["top"])
        assert x.sources == ("a.geojson", "surf.nc")


class TestUnstructuredRegions:

    def test_reduce_mask_record_polygon_and_stations(self, tmp_path):
        fx.write_daily_lnd(tmp_path / "run", YEARS)
        s = G.open_stream(tmp_path / "run", "h1")
        g = s.grid
        ms = R.MaskSet(g)
        a = np.zeros(fx.NL, bool)
        a[[1, 2, 3]] = True
        b = np.zeros(fx.NL, bool)
        b[[4, 5]] = True
        ms.add("a", a, "cells 1-3")
        ms.add("b", b, "cells 4-5")
        rs = R.reduce_regions(s, ["GPP", R.Field("SL1", ("SOILLIQ",), layer=1)], ms, [2016])
        w = np.float32(fx.LND_AREA).astype(float)
        for reg, mask in (("a", a), ("b", b)):
            for fld, var, scale, layer in (("GPP", "GPP", 86400.0, 0),
                                           ("SL1", "SOILLIQ", 1.0, 1)):
                want = []
                for doy in range(1, 366):
                    vals = np.array([fx.expected(var, 2016, doy, 0, c, Y0, fx.NL,
                                                 layer=layer) for c in range(fx.NL)]) * scale
                    ok = mask & np.isfinite(vals)
                    want.append((w[ok] * vals[ok]).sum() / w[ok].sum())
                np.testing.assert_allclose(rs.series(reg, fld), want, rtol=1e-10,
                                           err_msg=f"{reg} {fld}")
        lon, lat = g.lon, g.lat
        sq = [[lon[2] - 0.003, lat[2] - 0.003], [lon[4] + 0.003, lat[2] - 0.003],
              [lon[4] + 0.003, lat[4] + 0.003], [lon[2] - 0.003, lat[4] + 0.003],
              [lon[2] - 0.003, lat[2] - 0.003]]
        m, audit = R.polygon_mask(g, {"type": "Polygon", "coordinates": [sq]})
        np.testing.assert_array_equal(m, [False, False, True, True, True, False])
        back = R.MaskSet.from_netcdf(ms.to_netcdf(tmp_path / "u.nc"), g)
        np.testing.assert_array_equal(back["a"], a)
        xv = np.stack([lon - 0.005, lon + 0.005, lon + 0.005, lon - 0.005], axis=-1)
        yv = np.stack([lat - 0.004, lat - 0.004, lat + 0.004, lat + 0.004], axis=-1)
        st = R.station_cells(g, [{"id": "s", "lat": lat[3], "lon": lon[3]}],
                             corners=(xv, yv))[0]
        assert st["containing"]["flat"] == 3
        with pytest.raises(R.RegionError, match="corners"):
            R.station_cells(g, [{"id": "s", "lat": lat[3], "lon": lon[3]}])


class TestNanCells:

    def test_cells_missing_a_value_are_counted_per_region(self, reduced):
        # GPP: cell (0,0) never, (1,1) on day 100, (2,2) on day 50 (2015).
        assert reduced.nan_cells[("cell (1,1)", "GPP")] == 1
        assert reduced.nan_cells[("row 2", "GPP")] == 1
        assert reduced.nan_cells[("all", "GPP")] == 3
        assert reduced.nan_cells[("east block", "GPP")] == 0
        # ET: (1,2) lacks QSOIL on day 10.
        assert reduced.nan_cells[("all", "ET")] == 1
        assert reduced.nan_cells[("row 2", "TSA")] == 0


class TestWeightsAndGridTolerance:
    """reduce_regions(weights=, grid_tol_deg=), which the reference gate uses
    to weight a reference run by our cells' area and accept its float32
    centres."""

    def test_weights_replace_the_streams_own(self, tmp_path):
        fx.write_daily(tmp_path, [2015, 2016])
        s = G.open_stream(tmp_path, "h1")
        ms = R.MaskSet(s.grid)
        ms.add("all", np.ones(s.grid.shape, bool), "every cell")
        w = np.zeros(s.grid.shape)
        w[1, 2] = 3.0                                   # one cell carries all the weight
        rs = R.reduce_regions(s, ["TSA"], ms, [2016], weights=w)
        want = [fx.expected("TSA", 2016, d, 1, 2, 2015, s.grid.shape[1]) for d in range(1, 366)]
        np.testing.assert_allclose(rs.series("all", "TSA"), want, rtol=1e-12)
        with pytest.raises(R.RegionError, match="weights have shape"):
            R.reduce_regions(s, ["TSA"], ms, [2016], weights=np.ones((2, 2)))

    def test_masks_from_cells_a_little_apart(self, tmp_path):
        import netCDF4
        fx.write_daily(tmp_path / "a", [2015, 2016])
        fx.write_daily(tmp_path / "b", [2015, 2016])
        for p in (tmp_path / "b").glob("*.nc"):
            with netCDF4.Dataset(p, "a") as d:
                d["lat"][:] = np.asarray(d["lat"][:], dtype=np.float64) + 5e-5
        a, b = G.open_stream(tmp_path / "a", "h1"), G.open_stream(tmp_path / "b", "h1")
        ms = R.MaskSet(a.grid)
        ms.add("all", np.ones(a.grid.shape, bool), "every cell")
        with pytest.raises(R.RegionError, match="different grid"):
            R.reduce_regions(b, ["TSA"], ms, [2016])
        rs = R.reduce_regions(b, ["TSA"], ms, [2016], grid_tol_deg=1e-4)
        np.testing.assert_allclose(rs.series("all", "TSA"),
                                   R.reduce_regions(a, ["TSA"], ms, [2016]).series("all", "TSA"),
                                   rtol=1e-9)
