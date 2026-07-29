"""The gate between planning and compute.

It reads reception.json and strategy.json together — the only place that holds
both — and compares them before a column is materialised. It sits there because
that is where cost begins: everything upstream is one LLM call, everything
downstream is a CIME build per column and a queue slot.

Two severities, and the split is the point. A run that would produce garbage
stops. A run whose design is sound but names one unusable station loses the
pin, not the experiment — and the correction is RECORDED, because the planner
is purely LLM and nothing upstream edits it, so this is the only place a
discrepancy becomes visible.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.strategy_check import MAX_COLUMNS, check, render   # noqa: E402


def _reception(**kw):
    r = {"brief": {"design_archetype": "site",
                   "domain": {"name": "Test", "huc": "12345678",
                              "bbox": {"min_lon": -1, "min_lat": 1,
                                       "max_lon": 2, "max_lat": 3}},
                   "run_settings": {"resolved_period": {"yr_start": 1988,
                                                        "yr_end": 1988}}},
         "observations": {"streamflow": {"ok": True, "stations": [
                              {"id": "USGS-1"}, {"id": "USGS-2"}]},
                          "swe": {"ok": True, "stations": [{"triplet": "375:WA:SNTL"}]},
                          "water_table": {"ok": True, "wells": [{"id": "USGS-W1"}]}},
         "grid": {"n_in_basin": 58}}
    for k, v in kw.items():
        if v is None:
            r.pop(k, None)
        else:
            r[k] = v
    return r


def _strategy(n=19, stations=("USGS-1",)):
    return {"archetype": "site",
            "sampling": {"n_columns": n, "n_bands": 5},
            "validation": [{"variable": "streamflow", "stations": list(stations),
                            "comparison": "basin-aggregate"}]}


class TestPasses:
    def test_a_consistent_pair_is_clean(self):
        rep, _ = check(_reception(), _strategy())
        assert rep["ok"] and not rep["stop"] and not rep["corrections"]

    def test_it_reports_what_it_checked(self):
        """The run record should say what was compared, not just that it passed."""
        rep, _ = check(_reception(), _strategy())
        c = rep["checked"]
        assert c["n_columns"] == 19 and c["grid_points_in_basin"] == 58
        assert c["stations_fetched"] == 4 and c["stations_pinned"] == 1
        assert c["period"] == "1988-1988"


class TestStops:
    def test_no_bbox_stops(self):
        r = _reception(); r["brief"]["domain"].pop("bbox")
        rep, _ = check(r, _strategy())
        assert not rep["ok"] and any("bbox" in s for s in rep["stop"])

    def test_no_period_stops(self):
        r = _reception(); r["brief"]["run_settings"]["resolved_period"] = {}
        rep, _ = check(r, _strategy())
        assert not rep["ok"] and any("period" in s for s in rep["stop"])

    def test_absurd_column_count_stops(self):
        rep, _ = check(_reception(), _strategy(n=MAX_COLUMNS + 1))
        assert not rep["ok"]

    def test_more_columns_than_grid_points_stops(self):
        """A column is placed AT a grid point, so this is not a preference the
        sampler can satisfy. Only checkable because reception carries the grid."""
        rep, _ = check(_reception(grid={"n_in_basin": 12}), _strategy(n=19))
        assert not rep["ok"]
        assert any("grid points" in s for s in rep["stop"])

    def test_a_missing_column_count_stops(self):
        rep, _ = check(_reception(), {"sampling": {}})
        assert not rep["ok"]


class TestCorrections:
    def test_an_unfetched_station_is_dropped_not_fatal(self):
        rep, fixed = check(_reception(), _strategy(stations=("USGS-1", "USGS-999")))
        assert rep["ok"], "one bad station must not kill a sound design"
        assert fixed["validation"][0]["stations"] == ["USGS-1"]
        assert rep["corrections"]

    def test_losing_every_station_marks_the_target_unavailable(self):
        rep, fixed = check(_reception(), _strategy(stations=("USGS-999",)))
        assert rep["ok"]
        assert fixed["validation"][0]["stations"] == []
        assert "unavailable" in fixed["validation"][0]["comparison"]

    def test_the_original_is_not_mutated(self):
        """A caller ignoring the return value gets the original, which is safer
        than a silent edit."""
        st = _strategy(stations=("USGS-999",))
        check(_reception(), st)
        assert st["validation"][0]["stations"] == ["USGS-999"]

    def test_corrections_are_recorded_by_name(self):
        rep, _ = check(_reception(), _strategy(stations=("USGS-999",)))
        assert any("USGS-999" in m for m in rep["corrections"])

    def test_swe_triplets_count_as_known_ids(self):
        st = {"sampling": {"n_columns": 19},
              "validation": [{"variable": "swe", "stations": ["375:WA:SNTL"]}]}
        rep, fixed = check(_reception(), st)
        assert not rep["corrections"]
        assert fixed["validation"][0]["stations"] == ["375:WA:SNTL"]


class TestRender:
    def test_it_names_the_stop_reason(self):
        rep, _ = check(_reception(grid={"n_in_basin": 3}), _strategy(n=19))
        assert "STOP" in render(rep) and "grid points" in render(rep)

    def test_a_clean_check_says_so(self):
        rep, _ = check(_reception(), _strategy())
        assert "agrees with reception" in render(rep)
