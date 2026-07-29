#!/usr/bin/env python3
"""Step 0 of the Analyzer — load the run, split it three ways.

The split is not tidiness. plan is the question, data is the evidence, and
caveats bound what the evidence may support. Merged into one dict, only
convention keeps a planned number ("19 columns were requested") from being
reported as a result.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.analysis.step0_context import (  # noqa: E402
    load, BLOCKING, QUALIFY, CONTEXT)


def _run(tmp_path, experiment=None, reception=None, strategy=None):
    rd = tmp_path / "elm_run_X"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "experiment.json").write_text(json.dumps(experiment or {
        "model": "elm", "columns": [], "columns_total": 0,
        "columns_succeeded": 0}))
    (rd / "reception.json").write_text(json.dumps(reception or {}))
    (rd / "strategy.json").write_text(json.dumps(strategy or {}))
    return rd


class TestTheSplit:

    def test_no_experiment_says_what_is_missing(self, tmp_path):
        rd = tmp_path / "bare"; rd.mkdir()
        with pytest.raises(FileNotFoundError, match="has not packaged"):
            load(rd)

    def test_domain_and_period_come_from_reception(self, tmp_path):
        """The planner states neither. Reading only strategy.json would leave
        the Analyzer unable to say WHERE or WHEN the run was."""
        rd = _run(tmp_path,
                  reception={"brief": {
                      "domain": {"name": "Naches", "huc": "17030002"},
                      "run_settings": {
                          "resolved_period": {"yr_start": 1988, "yr_end": 1988},
                          "initialization": {"mode": "warm"}}}},
                  strategy={"archetype": "elevation_gradient"})
        ctx = load(rd)
        assert ctx.plan["domain"]["huc"] == "17030002"
        assert ctx.plan["period"]["yr_start"] == 1988
        assert ctx.plan["archetype"] == "elevation_gradient"

    def test_observations_come_from_reception_not_a_refetch(self, tmp_path):
        """Reception fetched them once the period was fixed. A second fetch
        can disagree with the first, and then the run's own record is not
        what was validated against."""
        rd = _run(tmp_path, reception={"observations": {
            "streamflow": {"ok": True, "stations": [{"id": "12494000"}]}}})
        assert load(rd).data["observations"]["streamflow"]["ok"] is True

    def test_plan_carries_no_results(self, tmp_path):
        """A planned count must never be reachable as a measurement."""
        rd = _run(tmp_path,
                  experiment={"columns": [], "columns_total": 3,
                              "columns_succeeded": 2},
                  strategy={"sampling": {"n_columns": 19}})
        ctx = load(rd)
        assert ctx.plan["sampling"]["n_columns"] == 19
        assert "columns_total" not in ctx.plan
        assert ctx.data["columns_succeeded"] == 2


class TestCaveats:
    """Uniform records, because that is what makes them checkable."""

    def test_limitations_are_a_dict_of_lists_not_a_list(self, tmp_path):
        """Read as a flat list it yields the KIND names as statements —
        'structural' appeared where a caveat belonged."""
        rd = _run(tmp_path, experiment={"columns": [], "limitations": {
            "structural": [{"applies_to": "runoff", "caveat": "not routed",
                            "kind": "structural"}],
            "configuration": [{"applies_to": "year one", "caveat": "warm start",
                               "kind": "configuration"}]}})
        cav = load(rd).caveats
        assert {c["statement"] for c in cav} == {"not routed", "warm start"}
        assert all(c["statement"] not in ("structural", "configuration")
                   for c in cav)

    def test_structural_limitations_block_rather_than_qualify(self, tmp_path):
        """'Not routed discharge, not comparable to a gauge' does not qualify
        a gauge comparison — it forbids the naive one."""
        rd = _run(tmp_path, experiment={"columns": [], "limitations": {
            "structural":    [{"caveat": "not routed", "applies_to": "runoff"}],
            "configuration": [{"caveat": "one year", "applies_to": "all"}]}})
        ctx = load(rd)
        sev = {c["statement"]: c["severity"] for c in ctx.caveats}
        assert sev["not routed"] == BLOCKING
        assert sev["one year"] == QUALIFY
        assert len(ctx.blocking()) == 1

    def test_assumptions_use_parameter_and_value(self, tmp_path):
        """There is no `statement` key on a ledger entry; asking for one gave
        None for every assumption."""
        rd = _run(tmp_path, experiment={"columns": [], "assumptions_ledger": [
            {"parameter": "spin-up", "value": "none — warm started",
             "source": "derived", "note": "storage inherited"}]})
        c = load(rd).caveats[0]
        assert "spin-up" in c["statement"] and "storage inherited" in c["statement"]
        assert c["severity"] == CONTEXT
        assert "derived" in c["source"]

    def test_a_bbox_sample_blocks_watershed_claims(self, tmp_path):
        rd = _run(tmp_path, experiment={"columns": [], "sampling_domain": {
            "clipped_to_watershed": False, "caveat": "bbox sample, not basin"}})
        ctx = load(rd)
        c = next(c for c in ctx.caveats if c["id"] == "bbox_not_watershed")
        assert c["severity"] == BLOCKING

    def test_a_failed_fetch_is_not_an_absence(self, tmp_path):
        """Reporting the two the same way is how 'no in-domain gauge had
        records' became a finding about a basin whose gauge reports yearly."""
        rd = _run(tmp_path, reception={"observations": {
            "streamflow": {"ok": False, "error": "HTTP 429"}}})
        c = next(c for c in load(rd).caveats
                 if c["id"] == "observations_failed_streamflow")
        assert c["severity"] == BLOCKING
        assert "429" in c["statement"] and "not evidence" in c["statement"]

    def test_every_caveat_has_the_same_five_fields(self, tmp_path):
        rd = _run(tmp_path, experiment={
            "columns": [],
            "limitations": {"structural": [{"caveat": "x", "applies_to": "y"}]},
            "assumptions_ledger": [{"parameter": "p", "value": "v"}],
            "sampling_domain": {"caveat": "z"},
            "strategy_check": {"corrections": ["dropped a station"]}})
        for c in load(rd).caveats:
            assert set(c) == {"id", "severity", "statement", "applies_to",
                              "source"}
            assert c["statement"]
            assert c["severity"] in (BLOCKING, QUALIFY, CONTEXT)


class TestSeries:
    def test_tidy_long_frame(self, tmp_path):
        pytest.importorskip("pandas")
        rd = _run(tmp_path, experiment={"columns": [{
            "case_name": "col_01",
            "variables": {"QOVER": {"daily": {
                "units": "mm/day", "values": [0.1, 0.2, 0.3],
                "dates": ["2019-01-01", "2019-01-02", "2019-01-03"]}}}}]})
        df = load(rd).series()
        assert list(df.columns) == ["date", "entity", "variable", "value",
                                    "units", "source"]
        assert len(df) == 3
        assert df["source"].unique().tolist() == ["model"]
        assert df["units"].unique().tolist() == ["mm/day"]
        assert str(df["date"].min().date()) == "2019-01-01"

    def test_no_series_returns_none_not_an_empty_frame(self, tmp_path):
        """An empty frame reads as 'measured nothing'; None reads as 'this
        run has no series', which is the true statement for a run extracted
        before daily capture existed."""
        rd = _run(tmp_path, experiment={"columns": [
            {"case_name": "col_01", "variables": {"QOVER": {"annual_mean": 1}}}]})
        assert load(rd).series() is None


class TestPlannedVsActual:
    def test_it_catches_a_design_that_was_not_delivered(self, tmp_path):
        rd = _run(tmp_path,
                  experiment={"columns_total": 19, "columns_succeeded": 19,
                              "columns": [{"case_name": "c1", "band": 1},
                                          {"case_name": "c2", "band": 2}]},
                  strategy={"sampling": {"n_columns": 19, "n_bands": 5}})
        checks = {c["claim"]: c for c in load(rd).planned_vs_actual()}
        assert checks["n_columns"]["ok"] is True
        assert checks["n_bands"]["ok"] is False
        assert checks["n_bands"]["planned"] == 5
        assert checks["n_bands"]["actual"] == 2

    def test_each_claim_names_the_file_it_was_planned_in(self, tmp_path):
        """The two upstream boxes own different halves — the planner never
        states a period, reception never states a column count."""
        rd = _run(tmp_path,
                  reception={"brief": {"run_settings": {
                      "resolved_period": {"yr_start": 2019}}}},
                  strategy={"sampling": {"n_columns": 1}})
        by = {c["claim"]: c["planned_in"] for c in load(rd).planned_vs_actual()}
        assert by["n_columns"] == "strategy.json"
        assert by["yr_start"] == "reception.json"

    def test_a_target_whose_station_was_never_fetched_fails(self, tmp_path):
        rd = _run(tmp_path,
                  reception={"observations": {"streamflow": {
                      "ok": True, "stations": [{"id": "12494000"}]}}},
                  strategy={"validation": [
                      {"variable": "streamflow",
                       "stations": ["12494000", "99999999"]}]})
        c = next(c for c in load(rd).planned_vs_actual()
                 if c["claim"] == "validation:streamflow")
        assert c["ok"] is False and c["planned"] == 2 and c["actual"] == 1
