"""The regional table through the one-pass reader, against the table made before it.

    pytest tests/test_gridded_block_pass_naches_oracle.py --runlocaldata -v

OPT-IN, and only where the files are on disk: the naches_suite block of
tests/local_oracle_paths.json names the reference table ("reference", a
03_results/regional.json of the Naches control for 2015-2023, made on
2026-11-03 by the field-by-field reduction) and the control's run directory
("run_dir"). The regions are rebuilt from the reference's own provenance (its
surface file and its registered polygons), so nothing else is needed.

run_values reads 2015 and 2022 through the pass, and 2021 for the December of
DJF 2022 (domain_analysis._previous_years). Every value of the two fields, by
year, month and season, must equal the reference's to the last bit: the
reduction changed how the files are read, not one number (regions.reduce_regions,
ONE PASS PER YEAR, 2026-11-04).
"""
import json
from pathlib import Path

import pytest

import domain_analysis as A
import gridded as G

pytestmark = pytest.mark.localdata

CONFIG = Path(__file__).with_name("local_oracle_paths.json")
FIELDS = ["GPP", "QRUNOFF"]          # a rate's total and, with H2OSNO below, a state's mean
YEARS = [2015, 2022]
SEASONS = [n for n, _ in A.SEASONS]


@pytest.fixture(scope="module")
def cfg():
    if not CONFIG.is_file():
        pytest.skip(f"no {CONFIG.name}")
    c = json.loads(CONFIG.read_text()).get("naches_suite") or {}
    if not (c and Path(c.get("reference", "")).is_file()
            and Path(c.get("run_dir", "")).is_dir()):
        pytest.skip("no naches_suite block, or its files are not on disk")
    return c


@pytest.fixture(scope="module")
def ref(cfg):
    r = json.loads(Path(cfg["reference"]).read_text())
    if not set(FIELDS + ["H2OSNO"]) <= set(r["runs"]["control"]["fields"]):
        pytest.skip("the reference table does not hold the fields compared here")
    return r


@pytest.fixture(scope="module")
def ours(cfg, ref):
    prov = ref["provenance"]
    reg = {n: {"polygon": p} for n, p in prov["registered_files"].items()}
    grid = G.open_stream(cfg["run_dir"], "h0").grid
    mk = A.masks(grid, prov["surface_file"], [A.VEGETATED] + sorted(reg), reg)
    assert mk["problems"] == []
    ms = mk["masks"]
    return ms, A.run_values(cfg["run_dir"], ms, FIELDS + ["H2OSNO"], YEARS)


def test_the_regions_are_the_references(ours, ref):
    ms, _ = ours
    assert ms.names() == list(ref["regions"])
    for n, r in ref["regions"].items():
        assert ms.mask(n).n_cells == r["cells"] and ms.mask(n).area_km2 == r["area_km2"], n


def test_every_year_month_and_season_to_the_last_bit(ours, ref):
    _, out = ours
    want = ref["runs"]["control"]
    assert out["problems"] == []
    n = 0
    for reg in ref["regions"]:
        for f in FIELDS + ["H2OSNO"]:
            for y in YEARS:
                assert out["values"][reg][f][str(y)] == want["values"][reg][f][str(y)], \
                    (reg, f, y)
                n += 1
                for m in range(1, 13):
                    k = A.month_key(y, m)
                    assert out["monthly"][reg][f][k] == want["monthly"][reg][f][k], (reg, f, k)
                    n += 1
                for s in SEASONS:
                    k = A.season_key(y, s)
                    if k in want["seasonal"][reg][f]:
                        assert out["seasonal"][reg][f][k] == want["seasonal"][reg][f][k], \
                            (reg, f, k)
                        n += 1
    assert n == 2 * 3 * 2 * (1 + 12) + 2 * 3 * (3 + 4)     # DJF 2015 has no December


def test_the_december_before_and_the_season_left_out(ours, ref):
    _, out = ours
    want = ref["runs"]["control"]
    for reg in ref["regions"]:
        assert out["seasonal"][reg]["GPP"]["2022-DJF"] == want["seasonal"][reg]["GPP"]["2022-DJF"]
    assert "2015-DJF" in out["seasons_left_out"] and "2015-DJF" in want["seasons_left_out"]
    assert out["seasons_left_out"]["2015-DJF"] == want["seasons_left_out"]["2015-DJF"]
    h0 = out["provenance"]["streams"]["h0"]
    assert h0["years_read"] == [2015, 2021, 2022]
    assert h0["records_per_year"] == {"2015": 12, "2021": 12, "2022": 12}


def test_how_and_units_are_the_references(ours, ref):
    _, out = ours
    want = ref["runs"]["control"]
    for f in FIELDS + ["H2OSNO"]:
        assert out["fields"][f] == want["fields"][f], f
        assert out["period_units"][f] == want["period_units"][f], f
