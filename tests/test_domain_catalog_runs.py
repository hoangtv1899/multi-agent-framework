"""The data catalog of a study with experiment runs (package F1, 2026-11-05).

    pytest tests/test_domain_catalog_runs.py -v

The Naches catalog was 24,655 of its 25,000 characters, and each run row
quoted its changes whole (about 1,150 characters a run), so a study with
experiment runs stopped step 2a. The catalog now has two parts with their own
limits: the fixed part (the control's row and configuration, regions,
periods, fields, methods, how to load) and the study part (one compact line
per run, every "does not mean" warning whole and once, and where the full
record is read: suite.run_record). These tests render it for 0, 1, 4 and 8
synthetic runs made with the server's own change records (domain_changes),
the longest kinds included, and check that nothing is cut: every run, change,
setting, meaning and warning is there, the fixed part is the control's own
catalog line for line, and each run's full record comes back from
suite.run_record, logged as a read, never as a lookup.
"""
import json
from pathlib import Path

import pytest

import domain_access as DAC
import domain_catalog as C
import domain_changes as DCH
from test_domain_catalog import YEARS, fake          # noqa: F401 (the fake control)

# the six change kinds the server makes today, then the two longest again
ITEMS = [{"id": "weather_scale", "params": {"temperature_K": 2.0, "precip_factor": 0.8,
                                            "months": [6, 7, 8], "region": "watershed"}},
         {"id": "fire_off", "params": {}},
         {"id": "weather_repeat_year", "params": {"map": {"2016": 2015}}},
         {"id": "co2_ppmv", "params": {"ppmv": 560}},
         {"id": "ndep_year", "params": {"year": 1850}},
         {"id": "nutrient_supplement", "params": {"nitrogen": "ALL", "phosphorus": "NONE"}},
         {"id": "weather_scale", "params": {"temperature_K": -1.5, "precip_factor": 1.25,
                                            "months": [12, 1, 2], "region": "watershed"}},
         {"id": "weather_repeat_year", "params": {"map": {"2015": 2016, "2016": 2016}}}]
KEYS = ("id", "params", "summary", "meaning", "not_meaning")


def server_change(item):
    r = DCH.resolve(item)
    return {k: r[k] for k in KEYS}


def make_study(root, n, normal=(2016,)):
    """A study directory with n experiment runs, each change as the server
    writes it."""
    d = Path(root) / f"study_{n}"
    (d / "01_inputs").mkdir(parents=True)
    runs = [{"run": "control", "reused": True, "changes": []}]
    for i in range(n):
        runs.append({"run": f"r{i + 1}_{ITEMS[i]['id']}"[:24],
                     "changes": [server_change(ITEMS[i])]})
    (d / "experiment.json").write_text(json.dumps({
        "kind": "whole_domain", "experiment": {"runs": runs},
        "chain": {"years_planned": YEARS, "years_passed": YEARS}}))
    (d / "01_inputs" / "domain_experiment.json").write_text(json.dumps({
        "runs": [{"name": r["run"], "changes": [{"id": c["id"]} for c in r["changes"]]}
                 for r in runs[1:]],
        "analysis": {"regions": ["box"], "variables": ["GPP"], "normal_years": list(normal)}}))
    return d, runs[1:]


def _kept(line):
    """The lines a study changes in the fixed part: its title, its normal
    years and the no-study note."""
    return line.strip() and not (line.startswith("# Data catalog")
                                 or line.startswith("- Normal years")
                                 or line.startswith("No study"))


def assert_nothing_cut(parts, runs, base_md):
    md = parts["text"]
    for r in runs:
        assert f"| {r['run']} |" in md
        for c in r["changes"]:
            assert C._cell(f"{c['id']} {json.dumps(c['params'], sort_keys=True)}"
                           if c["params"] else c["id"]) in md
            assert C._cell(c["meaning"]).rstrip(".") in md
            assert C._cell(c["not_meaning"]).rstrip(".") in md        # whole, never cut
    if runs:
        assert C.RUN_RECORD_CALL in parts["study"]
        assert "Does not mean:" in parts["study"]
    # the fixed part is the control's own catalog, line for line: every
    # caution, the percent policy, the statuses, the vegetated-cells rule
    assert [x for x in parts["fixed"].splitlines() if _kept(x)] == \
        [x for x in base_md.splitlines() if _kept(x)]
    for must in ("PCT_NATVEG >= 100", "No percent change, by the field notes' policy",
                 "Cautions:", "## 6. How to load", "suite.run_record(run)"):
        assert must in parts["fixed"]


@pytest.mark.parametrize("n", [0, 1, 4, 8])
def test_the_catalog_fits_with_up_to_eight_runs_and_cuts_nothing(fake, tmp_path, n):
    base = C.render(C.build(fake["control"]))
    d, runs = make_study(tmp_path, n)
    cat = C.build(fake["control"], study_dir=d)
    parts = C.render_parts(cat)
    assert len(parts["fixed"]) <= C.MAX_CHARS == 25000
    assert len(parts["study"]) <= C.STUDY_MAX_CHARS == 10000
    assert [r["run"] for r in cat["runs"]] == ["control"] + [r["run"] for r in runs]
    assert_nothing_cut(parts, runs, base)
    if n == 8:
        # each distinct warning printed once, naming every run it applies to
        ws = [x for x in parts["study"].splitlines() if "Does not mean:" in x]
        assert len(ws) == len({json.dumps(r["changes"][0]["not_meaning"]) for r in runs})
        assert any(x.startswith(f"- {runs[0]['run']} (weather_scale), {runs[6]['run']} "
                                f"(weather_scale). Does not mean:") for x in ws)


def test_a_part_over_its_limit_raises_and_is_never_clipped(fake, tmp_path, monkeypatch):
    d, _ = make_study(tmp_path, 8)
    cat = C.build(fake["control"], study_dir=d)
    monkeypatch.setattr(C, "STUDY_MAX_CHARS", 1000)
    with pytest.raises(C.CatalogError, match="study part .8 experiment runs. would be"):
        C.render(cat)
    monkeypatch.setattr(C, "STUDY_MAX_CHARS", 10000)
    monkeypatch.setattr(C, "MAX_CHARS", 1000)
    with pytest.raises(C.CatalogError, match="fixed part would be"):
        C.render(cat)


def test_a_runs_full_record_is_read_on_demand_and_logged_apart(fake, tmp_path):
    d, runs = make_study(tmp_path, 4)
    log = []
    acc = DAC.SuiteAccess({"control": str(tmp_path / "no_suite_needed")},
                          gate=lambda ask: (_ for _ in ()).throw(AssertionError("gated")),
                          log=log.append, study_dir=str(d))
    for r in runs:
        rec = acc.run_record(r["run"])
        # the server's words, whole (the study record leaves out empty settings)
        assert rec["changes"] == [{k: v for k, v in c.items() if v not in (None, "", {}, [])}
                                  for c in r["changes"]]
        assert rec["years"] == YEARS and rec["years_passed"] == YEARS
        assert "Does not mean:" in rec["text"] and r["changes"][0]["summary"] in rec["text"]
    assert acc.run_record("control")["changes"] == []
    with pytest.raises(DAC.AccessError, match="no run 'nope'"):
        acc.run_record("nope")
    assert [x["call"] for x in log] == ["run_record"] * 6
    assert all(x["contract_read"] is True for x in log)       # never a lookup
    assert "refused" in log[-1] and all("sha256" in x for x in log[:-1])
