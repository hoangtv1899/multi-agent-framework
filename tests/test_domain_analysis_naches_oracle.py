"""An experiment's regional numbers on the Naches control, against the hand analysis.

    pytest tests/test_domain_analysis_naches_oracle.py --runlocaldata -v

OPT-IN, and only where the run is on disk: the control comes from the local
registry (mcp/elm-mcp/domain_controls.json, control "naches"), the hand
analysis's records from the naches_bgc block of tests/local_oracle_paths.json.
The hand analysis summed daily means (h1); the experiment's table reads the
monthly stream (h0). The same totals from both say the time weighting is right.
"""
import json
from pathlib import Path

import pytest

import domain_analysis as A

pytestmark = pytest.mark.localdata

CONFIG = Path(__file__).with_name("local_oracle_paths.json")
YEARS = list(range(2015, 2022))
NORMAL = [2016, 2017, 2018, 2019, 2020]
BAND_NAMES = ["below 1000 m", "1000-1500 m", "1500-2000 m", "2000 m and above"]


@pytest.fixture(scope="module")
def hand():
    if not CONFIG.is_file():
        pytest.skip(f"no {CONFIG.name}")
    c = json.loads(CONFIG.read_text()).get("naches_bgc") or {}
    if not c or not Path(c.get("results_dir", "")).is_dir():
        pytest.skip("no naches_bgc block")
    return json.loads((Path(c["results_dir"]) / "model_anomalies.json").read_text())


@pytest.fixture(scope="module")
def table():
    import domain_controls as DCT
    try:
        e = DCT.get("naches")
    except DCT.ControlError as ex:
        pytest.skip(str(ex))
    return A.experiment(e, {"control": e["run_dir"]},
                        {"regions": [A.VEGETATED, A.BANDS, "watershed"],
                         "variables": ["GPP", "TLAI"], "normal_years": NORMAL}, YEARS)


def _sf(x, n):
    """Half a unit in the n-th significant figure of a recorded value, with 2 % to spare."""
    import math
    e = math.floor(math.log10(abs(x))) if x else 0
    return 0.5 * 10.0 ** (e - n + 1) * 1.02 + 1e-12


@pytest.mark.parametrize("region,hand_region", [(A.VEGETATED, "all vegetated cells")]
                         + [(b, b) for b in BAND_NAMES])
def test_gpp_by_year_and_the_event(table, hand, region, hand_region):
    ref = hand["annual"][hand_region]["gpp"]
    got = table["runs"]["control"]["values"][region]["GPP"]
    for y in YEARS:
        assert got[str(y)] == pytest.approx(ref["by_year"][str(y)], rel=1e-5), y
    a = table["event"]["anomalies"][region]["GPP"]
    assert a["normal"] == pytest.approx(ref["normal_2016_2020"], rel=1e-5)
    for y in (2015, 2021):
        assert a["years"][y]["rank_lowest_first"] == ref[str(y)]["rank_of_7_lowest_first"]
        # An anomaly of a few percent is a small difference of large numbers: the
        # 1e-5 allowed on each year's value (daily against monthly means, which
        # differ in their float32 last digits) becomes, in percent of the normal,
        # 100 x 1e-5 x (value + normal) / normal. Held to that, or to the
        # record's own 4 significant figures, whichever is looser.
        v = ref["by_year"][str(y)]
        tol = max(_sf(ref[str(y)]["anomaly_pct"], 4),
                  100 * 1e-5 * (abs(v) + abs(a["normal"])) / abs(a["normal"]))
        assert a["years"][y]["anomaly_pct"] == pytest.approx(
            ref[str(y)]["anomaly_pct"], abs=tol)


def test_the_annual_mean_lai(table, hand):
    ref = hand["annual"]["all vegetated cells"]["tlai_annual_mean"]
    got = table["runs"]["control"]["values"][A.VEGETATED]["TLAI"]
    for y in YEARS:
        assert got[str(y)] == pytest.approx(ref["by_year"][str(y)], rel=1e-5), y
    assert table["runs"]["control"]["fields"]["TLAI"]["how"] == "mean"


def test_the_watershed_and_the_observed_burned_area(table):
    ws = table["regions"]["watershed, vegetated"]
    assert ws["cells"] == 4127 and ws["area_km2"] == pytest.approx(2425.32, abs=0.01)
    ob = table["observed"]["burned_area_km2"]["values"]
    # MODIS's 2021 burn inside the box: 343.8 km2 (all cells; vegetated cells hold
    # all but a little of it), and inside the watershed somewhat less
    assert 330.0 < ob[A.VEGETATED]["2021"] <= 343.9
    assert ob["watershed, vegetated"]["2021"] < ob[A.VEGETATED]["2021"]
    assert table["problems"] == []


def test_the_report_of_the_event(table, hand, tmp_path):
    """The report rendered from these numbers, the control alone: its headline
    is the hand analysis's 2015 and 2021 anomalies, at 3 significant figures."""
    from agents.analysis import domain_report as DR
    ref = hand["annual"]["all vegetated cells"]["gpp"]
    assert round(ref["2015"]["anomaly_pct"], 2) == 5.86 and \
        round(ref["2021"]["anomaly_pct"], 1) == -18.6 and round(ref["normal_2016_2020"]) == 460
    rd = tmp_path / "run"
    rd.mkdir()
    # through JSON, as the run directory holds it (the years become strings)
    (rd / "experiment.json").write_text(json.dumps({
        "kind": "whole_domain", "regional": table,
        "experiment": {"control": {"name": "naches", "reuse": True},
                       "runs": [{"run": "control", "reused": True, "status": "reused",
                                 "changes": []}]},
        "chain": {"years_planned": YEARS, "years_passed": YEARS, "first_unpassed": None}},
        default=str))
    (rd / "strategy.json").write_text(json.dumps({"domain_experiment": {"analysis": {
        "regions": [A.VEGETATED, A.BANDS, "watershed"], "variables": ["GPP", "TLAI"],
        "normal_years": NORMAL}}}))
    out = DR.build(rd)
    assert out["headline"] == ("GPP over vegetated cells was 5.86 % above its 2016-2020 "
                               "normal (460 gC/m^2/yr) in 2015 and 18.6 % below it in 2021.")
    page = (rd / "04_analysis" / "DOMAIN_REPORT.md").read_text()
    # ranks as the hand analysis records them (test_gpp_by_year_and_the_event)
    assert "| vegetated cells | 460 | +5.86 % (rank 5 of 7) | -18.6 % (rank 1 of 7) |" in page
    for band in BAND_NAMES + ["watershed, vegetated"]:
        assert f"| {band} | " in page, band
    assert "### TLAI (dimensionless, mean over the year)" in page
    assert sorted(Path(f).name for f in out["figures"]) == ["domain_fig_GPP_anomaly.png",
                                                            "domain_fig_GPP_by_year.png"]
    assert "\u2014" not in page and "validated" not in page.lower()


def test_the_models_own_burned_area_is_the_controls_facts(table):
    """The server's model burned area per region (the mean burned fraction
    times the region's area) against the control's facts, which summed the
    same field cell by cell."""
    import domain_controls as DCT
    f = DCT.facts("naches")
    if not f:
        pytest.skip("the naches facts are not cached (refresh_domain_control_facts)")
    got = table["observed"]["burned_area_km2"]["model"]["control"][A.VEGETATED]
    for y in YEARS:
        # the facts keep one decimal: equal to within half of it
        assert got[str(y)] == pytest.approx(f["burned_area_km2"]["model"][str(y)],
                                            abs=0.05 + 1e-9), y


def test_the_analyzer_reads_the_control_offline(table, tmp_path):
    """Steps 0 and 1 of the Analyzer on the Naches control alone, no LLM: the
    regional frame is the table, the plan's targets are met, no column caveat
    appears, and the comparison holds the burned area beside MODIS's."""
    from agents.analysis import step0_context as S0
    from agents.analysis import step1_compare as S1
    from agents.analysis import step2_investigate as S2
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "reception.json").write_text(json.dumps({
        "user_request": "How did the Naches behave in the 2015 snow drought and in 2021?",
        "brief": {"design_archetype": "whole_domain"}}))
    (rd / "strategy.json").write_text(json.dumps({
        "archetype": "whole_domain", "domain_experiment": {
            "control": {"name": "naches", "reuse": True}, "runs": [], "years": YEARS,
            "analysis": {"regions": [A.VEGETATED, A.BANDS, "watershed"],
                         "variables": ["GPP", "TLAI"], "normal_years": NORMAL}}}))
    (rd / "experiment.json").write_text(json.dumps({
        "model": "elm", "kind": "whole_domain", "archetype": "whole_domain",
        "experiment": {"control": {"name": "naches", "reuse": True},
                       "runs": [{"run": "control", "reused": True, "status": "reused",
                                 "changes": []}], "check": {}},
        "chain": {"years_planned": YEARS, "years_passed": YEARS, "first_unpassed": None,
                  "runs": {}, "compute": {}},
        "columns": [{"run": "control", "year": y, "status": "reused"} for y in YEARS],
        "regional": table}, default=str))
    ctx = S0.load(rd)
    f = ctx.regional()
    g = f[(f.run == "control") & (f.region == A.VEGETATED) & (f.variable == "GPP")]
    assert dict(zip(g.year, g.value)) == pytest.approx(
        {y: table["runs"]["control"]["values"][A.VEGETATED]["GPP"][str(y)] for y in YEARS})
    assert set(f.region) == set(table["regions"]) and ctx.preflight()["blocked"] is None
    assert all(c["ok"] for c in ctx.planned_vs_actual()), ctx.planned_vs_actual()
    assert not any("column" in c["statement"].lower() for c in ctx.caveats)
    rec = S1.compare_all(ctx, rd / "04_analysis")
    (fd,) = rec["findings"]
    assert fd["n"] == len(table["regions"]) * len(YEARS)
    assert 330.0 < fd["result"]["observed"][A.VEGETATED]["2021"] <= 343.9
    brief = S2.context_brief(ctx, rec)
    assert "    watershed, vegetated: 4127 cells, " in brief and "1-D" not in brief
