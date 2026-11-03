"""Months and seasons of a whole-domain experiment, against an independent read.

    pytest tests/test_domain_months_naches_oracle.py --runlocaldata -v

OPT-IN, and only where the runs are on disk: the control comes from the local
registry (mcp/elm-mcp/domain_controls.json, control "naches"), the changed run
from the naches_weather_scale block of tests/local_oracle_paths.json (its
run_dir: the control with weather_scale, +2 K and precipitation x 0.8 in
June-August 2015 over the watershed).

THE INDEPENDENT READ (2026-11-03). GPP over the watershed's vegetated cells,
the run minus the control, month by month in 2015, read from the two runs' h0
files by a script that shares no code with domain_analysis: each month's
area x landfrac weighted mean, times the month's seconds. The weather changed
in June-August only, so January to May are exactly 0, and the months after
August carry the summer's after-effect.
"""
import json
from pathlib import Path

import pytest

import domain_analysis as A

pytestmark = pytest.mark.localdata

CONFIG = Path(__file__).with_name("local_oracle_paths.json")
REGION = "watershed, vegetated"
# gC/m2, the run minus the control, to the 2 decimals the read was made to
READ = {"2015-06": -6.98, "2015-07": -7.57, "2015-08": -6.09, "2015-09": -3.07,
        "2015-10": -1.11}
YEAR = {"difference": -25.04, "control": 643.7, "run": 618.6}


@pytest.fixture(scope="module")
def table():
    if not CONFIG.is_file():
        pytest.skip(f"no {CONFIG.name}")
    c = json.loads(CONFIG.read_text()).get("naches_weather_scale") or {}
    if not c or not Path(c.get("run_dir", "")).is_dir():
        pytest.skip("no naches_weather_scale block, or its run is not on disk")
    import domain_controls as DCT
    try:
        e = DCT.get("naches")
    except DCT.ControlError as ex:
        pytest.skip(str(ex))
    return A.experiment(e, {"control": e["run_dir"], "warm": c["run_dir"]},
                        {"regions": ["watershed"], "variables": ["GPP"]}, [2015],
                        confined={"warm": "watershed"})


def test_gpp_by_month_reproduces_the_independent_read(table):
    d = table["monthly_differences"]["warm"][REGION]["GPP"]
    assert list(d) == [A.month_key(2015, m) for m in range(1, 13)]
    for m in range(1, 6):
        assert d[A.month_key(2015, m)] == 0.0, m
    for k, x in READ.items():
        assert round(d[k], 2) == x, k
    assert table["runs"]["control"]["period_units"]["GPP"]["month"] == "gC/m^2/month"


def test_the_months_add_up_to_the_year(table):
    ctl = table["runs"]["control"]
    run = table["runs"]["warm"]
    assert round(ctl["values"][REGION]["GPP"]["2015"], 1) == YEAR["control"]
    assert round(run["values"][REGION]["GPP"]["2015"], 1) == YEAR["run"]
    dy = table["differences"]["warm"][REGION]["GPP"]["2015"]
    assert round(dy, 2) == YEAR["difference"]
    for r in (ctl, run):
        assert sum(r["monthly"][REGION]["GPP"].values()) == pytest.approx(
            r["values"][REGION]["GPP"]["2015"], rel=1e-9)
    assert sum(table["monthly_differences"]["warm"][REGION]["GPP"].values()) == \
        pytest.approx(dy, rel=1e-9)


def test_seasons_and_the_left_out_winter(table):
    s = table["seasonal_differences"]["warm"][REGION]["GPP"]
    m = table["monthly_differences"]["warm"][REGION]["GPP"]
    assert s["2015-MAM"] == 0.0
    assert s["2015-JJA"] == pytest.approx(m["2015-06"] + m["2015-07"] + m["2015-08"],
                                          rel=1e-12)
    # the runs' output starts in January 2015: no December 2014, so no DJF 2015
    assert "2015-DJF" not in s
    assert "December 2014" in table["runs"]["warm"]["seasons_left_out"]["2015-DJF"]


def test_where_the_numbers_come_from(table):
    p = table["provenance"]
    assert p["regions"][REGION]["cells"] == table["regions"][REGION]["cells"]
    assert p["registered_files"]["watershed"].endswith(".geojson")
    for run in ("control", "warm"):
        h0 = p["runs"][run]["streams"]["h0"]
        assert h0["fields"] == ["GPP"] and h0["records_per_year"] == {"2015": 12}
        assert h0["files"] and all(f["bytes"] > 0 and f["modified"] for f in h0["files"])
        assert p["runs"][run]["run_dir"] == table["runs"][run]["run_dir"]
