"""The input-set check and the domain writer, on small files in ELM's layout.

    pytest tests/test_domain_inputs.py -v

A matching surface / domain / saved state must pass with the right numbers,
and each kind of mismatch must fail with its own reason, alone. The rules
are ELM maint-2.0's (see domain_inputs), and the subgrid layout is checked
against the real Naches and global 0.5-degree saved states in
test_run_gate_naches_oracle.py.
"""
import math

import numpy as np
import pytest

import domain_inputs as D
import elm_case_fixtures as cf
import elm_gridded_fixtures as fx


@pytest.fixture
def triple(tmp_path):
    surf = cf.write_surface(tmp_path / "surf.nc")
    rest = cf.write_restart(tmp_path / "rest.nc")
    dom = tmp_path / "domain.nc"
    D.domain_from_surface(surf, dom)
    return surf, dom, rest


def _edit(path, var, fn):
    import netCDF4
    with netCDF4.Dataset(path, "a") as d:
        v = d[var]
        v.set_auto_maskandscale(False)
        v[:] = fn(np.asarray(v[:]))


def _at(a, index, value):
    a = a.copy()
    a[index] = value
    return a


def _bump(a, flat, by):
    """a with by added at one flat position."""
    return a + np.where(np.arange(a.size).reshape(a.shape) == flat, by, 0.0)


def _swap_first_two_natural_pfts(t):
    """Swap the first two natural-vegetation PFTs of cell 1, metadata and
    weights together: a consistent file in another order."""
    for k in ("pfts1d_itypveg", "pfts1d_wtlnd", "totvegc", "leafc", "tlai"):
        t[k][[0, 1]] = t[k][[1, 0]]


def _set_set(f, **cells):
    for name, (index, value) in cells.items():
        f[name][index] = value


# Each case: how to break the matching set, and the one reason it must give.
# surface_edit changes the surface and the saved state together (a consistent
# pair on another surface); edit changes the saved state alone.
CASES = {
    "domain centre east": (dict(domain=("xc", 7, 2e-3)),
                           "ELM stops: the domain's centres sit 0.002 deg from the surface's "
                           "(surfrd_get_data allows 0.001)"),
    "domain centre north": (dict(domain=("yc", 7, 2e-3)),
                            "ELM stops: the domain's centres sit 0.002 deg from the surface's "
                            "(surfrd_get_data allows 0.001)"),
    "restart grid east": (dict(restart=("grid1d_lon", 3, 2e-4)),
                          "the saved state's cells sit 0.0002 deg from the surface's"),
    "restart grid north": (dict(restart=("grid1d_lat", 3, 2e-4)),
                           "the saved state's cells sit 0.0002 deg from the surface's"),
    "pft weight": (dict(restart=("pfts1d_wtlnd", 5, 0.01)),
                   "ELM stops: PFT weights differ from the saved state by 0.01 "
                   "(subgridRestMod allows 0.005)"),
    "lake weight": (dict(edit=lambda t: t["land1d_wtxy"].__setitem__(
        t["land1d_ityplun"] == 5, 0.2)),
        "land-unit weights differ from the saved state (lake: up to 0.1;"),
    "pft sum": (dict(surface=lambda f: f["PCT_NAT_PFT"].__setitem__((0, 2, 3),
                                                                   f["PCT_NAT_PFT"][0, 2, 3] + 0.1)),
                "ELM stops: PCT_NAT_PFT sums to 100.1 % at cell (j=2, i=3)"),
    "land-unit sum": (dict(surface=lambda f: f["PCT_NATVEG"].__setitem__((2, 2), 99.99995)),
                      "ELM stops: the land-unit percentages sum to 99.99995 % at cell (j=2, i=2)"),
    "special land units": (dict(surface_edit=lambda f: _set_set(
        f, PCT_NATVEG=((2, 2), -10.0), PCT_LAKE=((2, 2), 110.0))),
        "ELM stops: lake, wetland, urban and glacier cover 110 % at cell (j=2, i=2)"),
    "urban without a region": (dict(surface_edit=lambda f: f["URBAN_REGION_ID"].__setitem__(
        cf.URBAN_CELL, 0)), "ELM stops: 1 active cells have an urban percentage but "
                            "URBAN_REGION_ID 0 (CheckUrban)"),
    "crop": (dict(surface_edit=lambda f: _set_set(f, PCT_NATVEG=((2, 2), 90.0),
                                                  PCT_CROP=((2, 2), 10.0))),
             "the surface carries crop (PCT_CROP above 0"),
    "record order": (dict(edit=_swap_first_two_natural_pfts),
                     "the saved state is not in ELM's order (PFT 1 has vegetation type 1 in "
                     "the saved state, 0 in ELM's layout)"),
}


class TestPairing:

    def test_a_matching_set_passes_with_its_numbers(self, triple):
        r = D.check_pairing(*triple)
        assert r["pass"] and r["reasons"] == []
        c = r["checks"]
        assert c["domain_vs_surface_coord_max_diff_deg"] == 0.0
        assert c["restart_grid_max_offset_deg"] == 0.0
        assert c["pft_weight_max_abs_diff"] < 1e-12
        assert c["landunit_weight_max_abs_diff"] == {
            "natveg": 0.0, "urban_tbd": 0.0, "urban_hd": 0.0, "urban_md": 0.0, "lake": 0.0}
        assert c["layout_matches"] is True and c["layout_first_difference"] is None
        assert (c["active_cells"], c["urban_valid_cells"], c["lake_cells"]) == (20, 20, 1)
        assert c["expected_dims"] == c["restart_dims"] == {
            "gridcell": 20, "landunit": 81, "column": 321, "pft": 641}
        assert c["pct_nat_pft_sum_max_dev"] <= D.ELM_SUM_MAX_DEV
        assert c["landunit_pct_sum_max_dev"] <= D.ELM_SUM_MAX_DEV
        assert c["special_landunit_max_pct"] == 20.0         # the urban cell
        assert c["not_covered_max_pct"] == {"PCT_CROP": 0.0}
        assert c["restart_landunit_types_not_covered"] == []
        assert sum(c["soil_negative_values"].values()) == 0
        assert sum(c["soil_nonfinite_values"].values()) == 0

    @pytest.mark.parametrize("what", sorted(CASES))
    def test_each_mismatch_alone_is_named(self, tmp_path, triple, what):
        surf, dom, rest = triple
        how, reason = CASES[what]
        if "domain" in how:
            var, at, by = how["domain"]
            _edit(dom, var, lambda a: _bump(a, at, by))
        if "restart" in how:
            var, at, by = how["restart"]
            _edit(rest, var, lambda a: _bump(a, at, by))
        if "edit" in how:
            rest = cf.write_restart(tmp_path / "r2.nc", edit=how["edit"])
        if "surface" in how:              # the surface alone, the saved state as before
            surf = cf.write_surface(tmp_path / "s2.nc", edit=how["surface"])
        if "surface_edit" in how:         # both, consistently
            surf = cf.write_surface(tmp_path / "s2.nc", edit=how["surface_edit"])
            rest = cf.write_restart(tmp_path / "r2.nc", surface_edit=how["surface_edit"])
        r = D.check_pairing(surf, dom, rest)
        assert not r["pass"]
        assert len(r["reasons"]) == 1 and r["reasons"][0].startswith(reason), r["reasons"]

    def test_centres_elm_accepts_still_break_the_exact_centre_rule(self, triple):
        """ELM allows 0.001 deg (the run behind the Naches saved state had a
        domain 4.6e-5 deg off its surface); this check asks for the surface's
        own centres, and says that the rule is its own."""
        surf, dom, rest = triple
        _edit(dom, "xc", lambda a: _bump(a, 7, 5e-4))
        r = D.check_pairing(surf, dom, rest)
        assert r["reasons"] == [
            "the domain's centres sit 0.0005 deg from the surface's: this check requires the "
            "surface's own centres (to 1e-12 deg; ELM itself allows 0.001), so write the "
            "domain with domain_from_surface"]
        assert r["checks"]["domain_vs_surface_coord_max_diff_deg"] == pytest.approx(5e-4, rel=1e-6)
        elm_only = D.PairingLimits(coord_max_diff_deg=D.ELM_COORD_MAX_DIFF_DEG)
        assert D.check_pairing(surf, dom, rest, limits=elm_only)["pass"]
        _edit(dom, "xc", lambda a: _bump(a, 7, -5e-4 + 1e-13))     # rounding is not a move
        assert D.check_pairing(surf, dom, rest)["pass"]

    def test_the_same_cells_written_from_minus_180_to_180_pass(self, triple):
        surf, dom, rest = triple
        _edit(dom, "xc", lambda a: a - 360.0)
        _edit(rest, "grid1d_lon", lambda a: a - 360.0)
        r = D.check_pairing(surf, dom, rest)
        assert r["pass"], r["reasons"]
        assert r["checks"]["domain_vs_surface_coord_max_diff_deg"] < 1e-9
        assert r["checks"]["restart_grid_max_offset_deg"] < 1e-9

    def test_soil_that_is_negative_or_not_a_number(self, tmp_path, triple):
        surf, dom, rest = triple
        r = D.check_pairing(cf.write_surface(tmp_path / "neg.nc", negative_soil=True), dom, rest)
        assert r["reasons"] == ["negative soil fractions: {'PCT_SAND': 4, 'PCT_CLAY': 0, "
                                "'ORGANIC': 0}"]
        _edit(surf, "PCT_CLAY", lambda a: _at(a, (0, 1, 1), np.nan))
        r = D.check_pairing(surf, dom, rest)
        assert r["reasons"] == ["soil fractions that are not numbers: {'PCT_SAND': 0, "
                                "'PCT_CLAY': 1, 'ORGANIC': 0}"]

    def test_values_that_are_not_numbers_fail_on_their_own_reason(self, tmp_path, triple):
        """ELM's own tests (x > limit) let a NaN through, so these are not
        'ELM stops' reasons: ELM would run on the NaN."""
        surf, dom, rest = triple
        _edit(dom, "yc", lambda a: _at(a, (1, 2), np.nan))
        r = D.check_pairing(surf, dom, rest)
        assert r["reasons"] == ["the domain's or the surface's centres include values that "
                                "are not numbers"]
        assert r["checks"]["domain_vs_surface_coord_max_diff_deg"] is None
        D.domain_from_surface(surf, dom, overwrite=True)
        _edit(rest, "pfts1d_wtlnd", lambda a: _at(a, 3, np.nan))
        r = D.check_pairing(surf, dom, rest)
        assert r["reasons"] == ["the PFT weights of the saved state or the surface include "
                                "values that are not numbers"]
        bad = cf.write_surface(tmp_path / "nan.nc",
                               edit=lambda f: f["PCT_NAT_PFT"].__setitem__((2, 1, 1), np.nan))
        r = D.check_pairing(bad, dom, cf.write_restart(tmp_path / "r3.nc"))
        assert r["reasons"][0] == "PCT_NAT_PFT holds values that are not numbers at cell " \
                                  "(j=1, i=1)"

    def test_a_saved_state_with_a_crop_land_unit(self, tmp_path, triple):
        surf, dom, _ = triple
        r = D.check_pairing(surf, dom, cf.write_restart(tmp_path / "r2.nc", extra_types=(2,)))
        assert not r["pass"]
        assert r["checks"]["restart_landunit_types_not_covered"] == [2]
        assert r["reasons"][0].startswith("the saved state has land-unit types [2]")
        assert r["reasons"][1].startswith("ELM stops: the saved state's subgrid")


class TestLayout:
    """The saved state must be what ELM lays out for this surface and domain."""

    def test_a_masked_cell_is_left_out(self, tmp_path, triple):
        surf, dom, rest = triple
        _edit(dom, "mask", lambda a: _at(a, (2, 3), 0))
        r = D.check_pairing(surf, dom, rest)          # a state from an all-land domain
        assert not r["pass"] and len(r["reasons"]) == 1
        assert r["reasons"][0].startswith("ELM stops: the saved state's subgrid")
        assert r["checks"]["expected_dims"] == {"gridcell": 19, "landunit": 77, "column": 305,
                                                "pft": 609}
        active = np.ones((4, 5), bool)
        active[2, 3] = False
        r = D.check_pairing(surf, dom, cf.write_restart(tmp_path / "r19.nc", active=active))
        assert r["pass"], r["reasons"]
        assert r["checks"]["active_cells"] == 19 and r["checks"]["layout_matches"]

    def test_masked_cells_are_not_checked(self, tmp_path, triple):
        """ELM never reads a masked cell, so its surface values do not matter."""
        surf, dom, _ = triple
        active = np.ones((4, 5), bool)
        active[2, 3] = False
        bad = cf.write_surface(tmp_path / "s.nc", negative_soil=True,
                               edit=lambda f: f["PCT_NAT_PFT"].__setitem__((0, 2, 3), 50.0))
        _edit(dom, "mask", lambda a: _at(a, (2, 3), 0))
        _edit(dom, "xc", lambda a: _at(a, (2, 3), 0.0))
        r = D.check_pairing(bad, dom, cf.write_restart(tmp_path / "r.nc", active=active))
        assert r["pass"], r["reasons"]

    def test_urban_land_units_only_where_the_region_is_set(self, tmp_path, triple):
        _, dom, rest = triple

        def no_region(f):
            f["URBAN_REGION_ID"][2, 3] = 0         # no urban percentage there
        surf = cf.write_surface(tmp_path / "s.nc", edit=no_region)
        r = D.check_pairing(surf, dom, rest)       # a state with urban everywhere
        assert not r["pass"] and r["reasons"][0].startswith("ELM stops: the saved state's subgrid")
        assert r["checks"]["expected_dims"] == {"gridcell": 20, "landunit": 78, "column": 306,
                                                "pft": 626}
        r = D.check_pairing(surf, dom, cf.write_restart(tmp_path / "r.nc", surface_edit=no_region))
        assert r["pass"], r["reasons"]
        assert r["checks"]["urban_valid_cells"] == 19

    def test_wetland_and_glacier_land_units(self, tmp_path, triple):
        _, dom, rest = triple

        def wet_and_ice(f):
            _set_set(f, PCT_NATVEG=((2, 2), 95.0), PCT_WETLAND=((2, 2), 5.0))
            _set_set(f, PCT_NATVEG=((1, 3), 97.0), PCT_GLACIER=((1, 3), 3.0))
        surf = cf.write_surface(tmp_path / "s.nc", edit=wet_and_ice)
        r = D.check_pairing(surf, dom, cf.write_restart(tmp_path / "r.nc",
                                                        surface_edit=wet_and_ice))
        assert r["pass"], r["reasons"]
        c = r["checks"]
        assert c["expected_dims"] == {"gridcell": 20, "landunit": 83, "column": 323, "pft": 643}
        assert c["landunit_weight_max_abs_diff"]["wetland"] == 0.0
        assert c["landunit_weight_max_abs_diff"]["glacier"] == 0.0
        r = D.check_pairing(surf, dom, rest)       # a state without them
        assert not r["pass"] and r["reasons"][0].startswith("ELM stops: the saved state's subgrid")

    def test_the_layout_order(self):
        active = np.ones((1, 2), bool)
        lay = D.elm_layout(active, np.array([[True, False]]), np.array([[0.0, 5.0]]),
                           np.zeros((1, 2)), np.zeros((1, 2)), natpft=3)
        assert list(lay["land_type"]) == [1, 7, 8, 9, 1, 5]
        assert list(lay["land_cell"]) == [1, 1, 1, 1, 2, 2]
        assert list(lay["col_type"]) == [1] + [7] * 5 + [8] * 5 + [9] * 5 + [1, 5]
        assert list(lay["pft_type"]) == [1] * 3 + [7] * 5 + [8] * 5 + [9] * 5 + [1] * 3 + [5]
        assert list(lay["pft_veg"]) == [0, 1, 2] + [0] * 15 + [0, 1, 2, 0]

    def test_urban_hd_and_md_are_told_apart(self, tmp_path, triple):
        surf, dom, _ = triple

        def weights(hd, md):
            def edit(t):
                c0 = t["land1d_gridcell_index"] == 1
                t["land1d_wtxy"][c0 & (t["land1d_ityplun"] == 8)] = hd
                t["land1d_wtxy"][c0 & (t["land1d_ityplun"] == 9)] = md
            return edit
        hd, md = cf.URBAN_SHARES[1] / 100.0, cf.URBAN_SHARES[2] / 100.0
        good = D.check_pairing(surf, dom, cf.write_restart(tmp_path / "r1.nc",
                                                           edit=weights(hd, md)))
        assert good["pass"], good["reasons"]
        swapped = D.check_pairing(surf, dom, cf.write_restart(tmp_path / "r2.nc",
                                                              edit=weights(md, hd)))
        lu = swapped["checks"]["landunit_weight_max_abs_diff"]
        assert lu["urban_hd"] == pytest.approx(0.02) and lu["urban_md"] == pytest.approx(0.02)
        assert lu["urban_tbd"] == 0.0 and not swapped["pass"]


class TestWrongFiles:

    def test_files_that_are_not_what_they_claim(self, tmp_path, triple):
        surf, dom, rest = triple
        with pytest.raises(D.InputSetError, match="not an ELM restart"):
            D.check_pairing(surf, dom, surf)
        with pytest.raises(D.InputSetError, match="which ELM requires"):
            D.check_pairing(dom, dom, rest)
        with pytest.raises(D.InputSetError, match="not a CIME domain"):
            D.check_pairing(surf, surf, rest)

    @pytest.mark.parametrize("name", ["URBAN_REGION_ID", "PFTDATA_MASK", "PCT_WETLAND",
                                      "PCT_CROP", "PCT_SAND", "ORGANIC"])
    def test_a_surface_without_a_variable_elm_requires(self, tmp_path, triple, name):
        _, dom, rest = triple
        surf = cf.write_surface(tmp_path / "s.nc", drop=(name,))
        with pytest.raises(D.InputSetError, match=rf"lacks \['{name}'\], which ELM requires"):
            D.check_pairing(surf, dom, rest)

    def test_an_unstructured_surface(self, tmp_path, triple):
        import netCDF4
        _, dom, rest = triple
        p = tmp_path / "one_d.nc"
        with netCDF4.Dataset(p, "w") as d:
            d.createDimension("gridcell", 3)
            d.createDimension("natpft", cf.NPFT)
            for n in D.REQUIRED_SURFACE:
                d.createVariable(n, "f8", ("gridcell",))[:] = 0.0
        with pytest.raises(D.InputSetError, match="unstructured"):
            D.check_pairing(p, dom, rest)
        with pytest.raises(D.InputSetError, match="unstructured"):
            D.domain_from_surface(p, tmp_path / "d.nc")

    def test_a_domain_of_another_size(self, tmp_path, triple):
        surf, _, rest = triple
        other = cf.write_surface(tmp_path / "small.nc", g=fx.grid(3, 5))
        dom = tmp_path / "small_domain.nc"
        D.domain_from_surface(other, dom)
        r = D.check_pairing(surf, dom, rest)
        assert r["reasons"] == ["ELM stops: the domain is (3, 5) cells, the surface (4, 5)"]

    def test_indices_from_another_grid_or_pft_set_are_named(self, tmp_path, triple):
        surf, dom, _ = triple
        r = D.check_pairing(surf, dom, cf.write_restart(
            tmp_path / "r1.nc", edit=lambda t: t["grid1d_ixy"].__setitem__(0, 99)))
        assert r["reasons"] == ["the saved state is not in ELM's order (grid cell 1 has ixy 99 "
                                "in the saved state, 1 in ELM's layout); ELM reads it by "
                                "position, so its weights and state would land on other units"]
        r = D.check_pairing(surf, dom, cf.write_restart(
            tmp_path / "r2.nc", edit=lambda t: t["pfts1d_itypveg"].__setitem__(16, 17)))
        assert "PFT 17 has vegetation type 17 in the saved state, 16 in ELM's layout" in \
            r["reasons"][0]


class TestDomainFromSurface:

    def test_centres_corners_area_mask(self, tmp_path):
        import netCDF4
        surf = cf.write_surface(tmp_path / "surf.nc")
        dom = tmp_path / "d.nc"
        rec = D.domain_from_surface(surf, dom)
        assert rec["dx_deg"] == 1.0 / 120.0 and rec["dy_deg"] == 1.0 / 120.0
        assert rec["mask_from"] == "PFTDATA_MASK" and rec["frac_from"] == "LANDFRAC_PFT"
        assert rec["active_cells"] == 20
        g = fx.grid()
        lon, lat = np.meshgrid(g["lon"], g["lat"])
        h = 0.5 / 120.0
        with netCDF4.Dataset(dom) as d:
            np.testing.assert_array_equal(d["xc"][:], lon)
            np.testing.assert_array_equal(d["yc"][:], lat)
            np.testing.assert_allclose(d["xv"][1, 2], [lon[1, 2] - h, lon[1, 2] + h,
                                                       lon[1, 2] + h, lon[1, 2] - h], atol=1e-12)
            np.testing.assert_allclose(d["yv"][1, 2], [lat[1, 2] - h, lat[1, 2] - h,
                                                       lat[1, 2] + h, lat[1, 2] + h], atol=1e-12)
            want = math.radians(1 / 120) * (math.sin(math.radians(lat[2, 3] + h)) -
                                            math.sin(math.radians(lat[2, 3] - h)))
            assert d["area"][2, 3] == pytest.approx(want, rel=1e-12)
            assert d["mask"][:].dtype == np.int32 and int(d["mask"][:].min()) == 1
            total = float(np.asarray(d["area"][:]).sum())
        assert rec["box_area_km2"] == pytest.approx(total * 6371.22 ** 2, rel=1e-12)

    def test_mask_and_frac_are_the_surfaces(self, tmp_path):
        import netCDF4
        surf = cf.write_surface(tmp_path / "s.nc")
        _edit(surf, "PFTDATA_MASK", lambda a: _at(a, (2, 3), 0))
        _edit(surf, "LANDFRAC_PFT", lambda a: _at(a, (1, 4), 0.25))
        rec = D.domain_from_surface(surf, tmp_path / "d.nc")
        with netCDF4.Dataset(tmp_path / "d.nc") as d:
            mask, frac = np.asarray(d["mask"][:]), np.asarray(d["frac"][:])
        assert mask[2, 3] == 0 and mask.sum() == mask.size - 1 and rec["active_cells"] == 19
        assert frac[1, 4] == 0.25 and np.count_nonzero(frac != 1.0) == 1
        rec = D.domain_from_surface(surf, tmp_path / "d.nc", overwrite=True, mask_from="ones")
        with netCDF4.Dataset(tmp_path / "d.nc") as d:
            assert np.all(np.asarray(d["mask"][:]) == 1)
        assert rec["mask_from"] == "all ones (asked)" and rec["active_cells"] == 20
        with pytest.raises(D.InputSetError, match="mask_from"):
            D.domain_from_surface(surf, tmp_path / "d.nc", overwrite=True, mask_from="land")

    def test_without_a_mask_every_cell_is_written_active(self, tmp_path):
        import netCDF4
        surf = cf.write_surface(tmp_path / "s.nc", with_mask=False)
        rec = D.domain_from_surface(surf, tmp_path / "d.nc")
        assert rec["mask_from"] == "all ones (no PFTDATA_MASK)"
        assert rec["frac_from"] == "all ones (no LANDFRAC_PFT)"
        with netCDF4.Dataset(tmp_path / "d.nc") as d:
            assert np.all(np.asarray(d["mask"][:]) == 1)
            assert np.all(np.asarray(d["frac"][:]) == 1.0)

    def test_a_north_to_south_surface_gets_the_same_cells(self, tmp_path):
        import netCDF4
        up = cf.write_surface(tmp_path / "up.nc")
        down = cf.write_surface(tmp_path / "down.nc")
        _edit(down, "LATIXY", lambda a: a[::-1].copy())
        ra = D.domain_from_surface(up, tmp_path / "a.nc")
        rb = D.domain_from_surface(down, tmp_path / "b.nc")
        assert rb["dy_deg"] == ra["dy_deg"] == 1.0 / 120.0
        assert ra["box_area_km2"] > 0
        assert rb["box_area_km2"] == pytest.approx(ra["box_area_km2"], rel=1e-12)
        with netCDF4.Dataset(tmp_path / "a.nc") as a, netCDF4.Dataset(tmp_path / "b.nc") as b:
            np.testing.assert_allclose(np.asarray(b["area"][:]), np.asarray(a["area"][::-1]),
                                       rtol=1e-12)
            yv = np.asarray(b["yv"][0, 0])
            assert yv[0] == yv[1] < yv[2] == yv[3]     # south edge first: counterclockwise

    @pytest.mark.parametrize("lon0,seam", [(359.98, 360.0), (179.98, 180.0)])
    def test_a_row_across_0_or_180_east(self, tmp_path, lon0, seam):
        """0-360 longitudes across 0 E, and -180-180 ones across 180 E."""
        import netCDF4
        surf = cf.write_surface(tmp_path / "s.nc")
        row = lon0 + np.arange(5) / 120.0
        row = np.where(row >= seam, row - 360.0, row)
        _edit(surf, "LONGXY", lambda a: np.broadcast_to(row, a.shape).copy())
        rec = D.domain_from_surface(surf, tmp_path / "d.nc")
        assert rec["dx_deg"] == 1.0 / 120.0
        with netCDF4.Dataset(tmp_path / "d.nc") as d:
            np.testing.assert_array_equal(np.asarray(d["xc"][0]), row)

    def test_float32_centres_on_a_fine_grid(self, tmp_path):
        """At 238 E, float32 centres 1/1200 deg apart differ by up to 1.5e-5
        from one step to the next, more than 1 % of the step."""
        surf = cf.write_surface(tmp_path / "s.nc", g=fx.grid(40, 60))
        _edit(surf, "LONGXY", lambda a: (238.4 + np.arange(a.shape[1]) / 1200.0)
              .astype(np.float32).astype(np.float64)[None, :].repeat(a.shape[0], 0))
        _edit(surf, "LATIXY", lambda a: (46.4 + np.arange(a.shape[0]) / 1200.0)
              .astype(np.float32).astype(np.float64)[:, None].repeat(a.shape[1], 1))
        rec = D.domain_from_surface(surf, tmp_path / "d.nc")
        assert rec["dx_deg"] == 1.0 / 1200.0 and rec["dy_deg"] == 1.0 / 1200.0

    def test_an_irregular_sheared_or_one_row_grid_is_refused(self, tmp_path):
        surf = cf.write_surface(tmp_path / "surf.nc")
        _edit(surf, "LONGXY", lambda a: a + np.where(np.arange(a.shape[1]) == 2, 0.002, 0.0))
        with pytest.raises(D.InputSetError, match="not uniform"):
            D.domain_from_surface(surf, tmp_path / "d.nc")
        sheared = cf.write_surface(tmp_path / "sheared.nc")
        _edit(sheared, "LONGXY", lambda a: a + 0.002 * np.arange(a.shape[0])[:, None])
        with pytest.raises(D.InputSetError, match="not a regular latitude-longitude grid"):
            D.domain_from_surface(sheared, tmp_path / "d.nc")
        row = cf.write_surface(tmp_path / "row.nc", g=fx.grid(1, 5))
        with pytest.raises(D.InputSetError, match="one row only"):
            D.domain_from_surface(row, tmp_path / "d2.nc")

    def test_a_file_that_is_not_a_surface(self, tmp_path, triple):
        _, dom, rest = triple
        for p in (dom, rest):
            with pytest.raises(D.InputSetError, match="lacks LONGXY"):
                D.domain_from_surface(p, tmp_path / "d.nc")

    def test_it_will_not_overwrite_unless_told_and_never_the_surface(self, tmp_path):
        surf = cf.write_surface(tmp_path / "surf.nc")
        D.domain_from_surface(surf, tmp_path / "d.nc")
        with pytest.raises(D.InputSetError, match="exists"):
            D.domain_from_surface(surf, tmp_path / "d.nc")
        D.domain_from_surface(surf, tmp_path / "d.nc", overwrite=True)
        with pytest.raises(D.InputSetError, match="is the surface itself"):
            D.domain_from_surface(surf, surf, overwrite=True)
