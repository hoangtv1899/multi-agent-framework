"""The data catalog of the Naches control, from its own files.

    pytest tests/test_domain_catalog_naches_oracle.py --runlocaldata -v

OPT-IN, and only where the files are on disk: the naches_suite block of
tests/local_oracle_paths.json names the control's run directory ("run_dir"),
and the machine's registry (domain_controls.json) holds the control's entry
(its case, its run directory, which must be the same one, and its regions).

The catalog must render within its length, list every one of the notes' 52
fields in the fixed group order with no problem (every note names a field the
run saves, at the cadence the note says, and its units agree with the note's
how), and give each region the cells and area domain_analysis.masks gives it.

Opening a stream reads every file's time stamps (40-100 s cold), so the stream
index cache points at a temporary directory for the session: each stream is
read once, and nothing is written in the user's home.
"""
import json
import os
from pathlib import Path

import pytest

import domain_analysis as A
import domain_catalog as C
import gridded as G

pytestmark = pytest.mark.localdata

CONFIG = Path(__file__).with_name("local_oracle_paths.json")


@pytest.fixture(scope="session")
def index_cache(tmp_path_factory):
    old = os.environ.get(G.INDEX_CACHE_ENV)
    os.environ[G.INDEX_CACHE_ENV] = str(tmp_path_factory.mktemp("stream_index"))
    yield
    if old is None:
        os.environ.pop(G.INDEX_CACHE_ENV, None)
    else:
        os.environ[G.INDEX_CACHE_ENV] = old


@pytest.fixture(scope="module")
def control():
    if not CONFIG.is_file():
        pytest.skip(f"no {CONFIG.name}")
    c = json.loads(CONFIG.read_text()).get("naches_suite") or {}
    if not (c and Path(c.get("run_dir", "")).is_dir()):
        pytest.skip("no naches_suite block, or its run directory is not on disk")
    import domain_controls
    try:
        e = domain_controls.get("naches")
    except domain_controls.ControlError as err:
        pytest.skip(f"the registry has no usable naches control: {err}")
    if Path(e["run_dir"]).resolve() != Path(c["run_dir"]).resolve():
        pytest.skip("the registry's naches run is not the one naches_suite names")
    return e


@pytest.fixture(scope="module")
def cat(index_cache, control):
    return C.build(control)


def test_the_catalog_renders_within_its_length(cat):
    md = C.render(cat)
    assert 0 < len(md) <= C.MAX_CHARS
    assert cat["problems"] == []


def test_every_note_is_a_field_of_the_run_in_the_fixed_group_order(cat):
    notes = C.load_notes()
    rows = {n: e for n, e in notes.items() if C.is_row(e)}
    assert len(rows) == 52
    rank = {g: i for i, g in enumerate(C.GROUPS)}
    want = sorted(rows, key=lambda n: (rank[rows[n]["group"]], n))
    assert [r["name"] for r in cat["fields"]] == want
    for r in cat["fields"]:
        assert r["cadence"] == rows[r["name"]]["cadence"], r["name"]
        assert r["how"] == C.unit_rule(r["name"], r["units"]), r["name"]
        assert r["long_name"], r["name"]
    h2osoi = next(r for r in cat["fields"] if r["name"] == "H2OSOI")
    assert h2osoi["levels"] == [{"dim": "levgrnd", "size": 15}]
    # FIRE is no row; its note names a monthly field the run saves
    assert "FIRE" not in {r["name"] for r in cat["fields"]}
    assert cat["notes_outside_rows"]["FIRE"]["streams"] == [cat["periods"]["monthly_stream"]]
    md = C.render(cat)
    heads = [line[4:] for line in md.splitlines() if line.startswith("### ")]
    assert heads == list(C.GROUPS)


def test_the_regions_are_those_of_domain_analysis_masks(cat, control):
    grid = G.open_stream(control["run_dir"], cat["periods"]["daily_stream"]).grid
    reg = control.get("regions") or {}
    names = [A.VEGETATED] + sorted(reg) + [A.BANDS]
    # THE SUITE'S REGIONS (2026-11-04): once the control's suite is built, the
    # catalog lists its regions, the bands within each registered region among
    # them, all made by domain_analysis.masks with the control's band edges
    if cat["suite_manifest"]:
        names += [{"name": A.BANDS, "within": r} for r in sorted(reg)]
    mk = A.masks(grid, A._surface(control["case_dir"]), names, reg,
                 control.get(A.BAND_EDGES_KEY))
    assert mk["problems"] == []
    want = [(n, m["n_cells"], m["area_km2"]) for n, m in mk["masks"].summary().items()]
    got = [(r["name"], r["cells"], r["area_km2"]) for r in cat["regions"]["regions"]]
    assert got == want
    assert len(got) >= 2 + len(reg)


# THE THREE EXPERIMENT RUNS (package F1, 2026-11-05): the cases
# naches_bgc_fix_warm_dry_summer, naches_bgc_fix_nofire and naches_bgc_fix_dry,
# whose runs are warm_dry_summer, nofire and dry. Their changes are made by the
# server's own domain_changes.resolve with the settings the runs used (for
# warm_dry_summer it equals the change list in its experiment record), and the
# study record is written to the test's temporary directory, never to a run
# directory or workflow_outputs.
NACHES_RUNS = [
    ("warm_dry_summer", {"id": "weather_scale", "params": {
        "temperature_K": 2.0, "precip_factor": 0.8, "months": [6, 7, 8],
        "region": "watershed"}}),
    ("nofire", {"id": "fire_off", "params": {}}),
    ("dry", {"id": "weather_repeat_year", "params": {"map": {"2016": 2015}}})]


def test_the_catalog_fits_with_the_three_experiment_runs_and_cuts_nothing(cat, control,
                                                                         tmp_path):
    from test_domain_catalog_runs import assert_nothing_cut, server_change
    base = C.render(cat)
    assert len(C.render_parts(cat)["fixed"]) <= C.MAX_CHARS        # the control alone
    d = tmp_path / "naches_three_runs"
    (d / "01_inputs").mkdir(parents=True)
    runs = [{"run": name, "changes": [server_change(it)]} for name, it in NACHES_RUNS]
    (d / "experiment.json").write_text(json.dumps({
        "kind": "whole_domain",
        "experiment": {"runs": [{"run": "control", "reused": True, "changes": []}] + runs},
        "chain": {"years_planned": [2015, 2016], "years_passed": [2015, 2016]}}))
    (d / "01_inputs" / "domain_experiment.json").write_text(json.dumps({
        "runs": [{"name": r["run"], "changes": [{"id": r["changes"][0]["id"]}]} for r in runs],
        "analysis": {"normal_years": [2016, 2017, 2018, 2019, 2020]}}))
    s = C.build(control, study_dir=d)
    parts = C.render_parts(s)
    assert len(parts["fixed"]) <= C.MAX_CHARS
    assert len(parts["study"]) <= C.STUDY_MAX_CHARS
    assert s["problems"] == []
    assert_nothing_cut(parts, runs, base)
    print(f"\nNaches catalog: control alone {len(base):,} characters; with the three "
          f"runs fixed {len(parts['fixed']):,} + study {len(parts['study']):,}")
