#!/usr/bin/env python3
"""Analyzer step 2 — the deterministic half.

The LLM layer needs an API call, so what is pinned here is everything around
it: the brief it reads, and the runner that decides whether what it wrote
counts. Both are pure functions of the context, which is the point — most of
step 2 is testable offline against an archived run.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.analysis import script_runner as runner        # noqa: E402
from agents.analysis import step2_investigate as step2     # noqa: E402


class _Ctx:
    def __init__(self, cols=None, units=None, caveats=(), plan=None, sem=None):
        self._c = cols if cols is not None else [
            {"case_name": "col_01", "elevation_m": 2400.0, "band": 1,
             "soil": None, "soil_profile": {"layers": []}},
            {"case_name": "col_02", "elevation_m": 3400.0, "band": 2,
             "soil": None, "soil_profile": {"layers": []}}]
        self.plan = plan or {"question": "How does recharge behave?",
                             "goals": ["quantify recharge"],
                             "period": {"yr_start": 2019, "yr_end": 2019}}
        self.caveats = list(caveats)
        self.data = {"variable_units": units if units is not None
                     else {"QOVER": "mm/s", "H2OSNO": "mm"},
                     "field_semantics": sem or {}}

    @property
    def columns(self):
        return self._c

    def series(self):
        pd = pytest.importorskip("pandas")
        return pd.DataFrame({
            "date": ["2019-01-01", "2019-01-02"] * 2,
            "entity": ["col_01"] * 2 + ["col_02"] * 2,
            "variable": ["QOVER"] * 2 + ["H2OSNO"] * 2,
            "value": [1e-5, 2e-5, 100.0, 110.0],
            "units": ["mm/s"] * 2 + ["mm"] * 2,
            "source": ["model"] * 4})


class TestUnitsAreFixedAtTheBoundary:
    """variable_units MISLABELS the daily series, and the label is the bug.

    It says `mm/s` because that is the raw ELM variable's unit — but the series
    it labels went through the extractor's _daily(), which already resampled
    3-hourly output and converted fluxes to mm/day. The values are per-day; the
    label says per-second.

    A generated script read the label, multiplied a year of already-correct
    values by 86400, and plotted 1.1e6 mm/yr of runoff. So the fix is a
    RELABEL, not a conversion: touching the values would double the error, and
    doing exactly that is how the mislabel was finally found.
    """

    def test_per_second_fluxes_are_relabelled_not_rescaled(self):
        pytest.importorskip("pandas")
        payload = runner._payload(_Ctx())
        q = payload["df"][payload["df"]["variable"] == "QOVER"]
        assert list(q["value"]) == [1e-5, 2e-5], "values must not be touched"
        assert set(q["units"]) == {"mm/day"}

    def test_non_flux_variables_are_untouched(self):
        """H2OSNO is a storage in mm, not a rate — scaling it would be wrong."""
        pytest.importorskip("pandas")
        payload = runner._payload(_Ctx())
        s = payload["df"][payload["df"]["variable"] == "H2OSNO"]
        assert list(s["value"]) == [100.0, 110.0]
        assert set(s["units"]) == {"mm"}

    def test_the_conversion_is_reported(self):
        pytest.importorskip("pandas")
        assert runner._payload(_Ctx())["converted_to_daily"] == ["QOVER"]

    def test_the_brief_does_not_ask_for_a_second_conversion(self):
        """The runner converts, so a brief still saying 'multiply by 86400'
        would produce a double conversion. These two must not drift."""
        brief = step2.context_brief(_Ctx())
        assert "Do NOT" in brief and "86400" in brief
        assert "Multiply by 86400" not in brief


class TestTheRunnerRefusesRatherThanShrugs:
    """soil_attribution returned {} for the study's entire history and the
    figure simply did not appear. A generated script reintroduces that failure
    fresh every run, so an empty result has to be a stated refusal."""

    def _run(self, code, tmp_path):
        return runner.run(code, _Ctx(), tmp_path / "f.png")

    def test_a_result_below_the_floor_is_refused(self, tmp_path):
        pytest.importorskip("pandas")
        r = self._run('fig, ax = plt.subplots(); fig.savefig(out_path)\n'
                      'result = {"n": 0}', tmp_path)
        assert r["ok"] is False
        assert "below the floor" in r["error"]

    def test_a_result_without_n_is_refused(self, tmp_path):
        """n is how many data points the claim rests on. Without it there is
        nothing to check the claim's weight against."""
        pytest.importorskip("pandas")
        r = self._run('fig, ax = plt.subplots(); fig.savefig(out_path)\n'
                      'result = {"slope": 1.0}', tmp_path)
        assert r["ok"] is False and "no numeric `n`" in r["error"]

    def test_a_script_that_never_assigns_result_is_refused(self, tmp_path):
        pytest.importorskip("pandas")
        r = self._run('fig, ax = plt.subplots(); fig.savefig(out_path)', tmp_path)
        assert r["ok"] is False

    def test_a_raising_script_returns_a_reason_not_an_exception(self, tmp_path):
        """One bad script must not lose the four good ones alongside it."""
        pytest.importorskip("pandas")
        r = self._run("result = 1/0", tmp_path)
        assert r["ok"] is False and "ZeroDivisionError" in r["error"]

    def test_a_result_with_no_figure_is_refused(self, tmp_path):
        pytest.importorskip("pandas")
        r = self._run('result = {"n": 19}', tmp_path)
        assert r["ok"] is False and "no figure" in r["error"]

    def test_a_good_script_passes_with_its_numbers(self, tmp_path):
        pytest.importorskip("pandas")
        r = self._run('fig, ax = plt.subplots(figsize=(6,4))\n'
                      'ax.plot([1,2,3],[1,2,3]); fig.savefig(out_path, dpi=120)\n'
                      'result = {"n": 19, "slope": 1.0}', tmp_path)
        assert r["ok"] is True
        assert r["result"]["slope"] == 1.0
        assert Path(r["figure"]).exists()

    def test_the_script_is_saved_for_provenance(self, tmp_path):
        """The saved script IS the provenance for every number it produced —
        a reviewer can open it and re-run it."""
        pytest.importorskip("pandas")
        sp = tmp_path / "scripts" / "f.py"
        r = runner.run('fig, ax = plt.subplots(); ax.plot([1,2])\n'
                       'fig.savefig(out_path, dpi=120)\nresult = {"n": 5}',
                       _Ctx(), tmp_path / "f.png", script_path=sp)
        assert r["ok"] and sp.exists()
        assert "result = {\"n\": 5}" in sp.read_text()


class TestTheBriefIsGenericAndBounded:
    """Nothing here may name a basin, a variable set, or a station network —
    the framework is meant for any US watershed."""

    def test_it_carries_the_users_own_question(self):
        assert "How does recharge behave?" in step2.context_brief(_Ctx())

    def test_it_forbids_a_basin_total(self):
        """Nineteen unrouted 1-D columns do not sum to a watershed, and that is
        the sentence a model writes unless the brief forbids it."""
        b = step2.context_brief(_Ctx())
        assert "OUT OF SCOPE" in b and "routing" in b

    def test_blocking_caveats_are_quoted_with_their_scope(self):
        b = step2.context_brief(_Ctx(caveats=[
            {"id": "no_routing", "severity": "blocking",
             "statement": "columns are unrouted",
             "applies_to": "any discharge claim"}]))
        assert "BLOCKING CAVEATS" in b
        assert "no_routing" in b and "any discharge claim" in b

    def test_a_field_null_on_every_column_is_not_advertised(self):
        """`soil` is None on all 19 rows and sits beside `soil_profile` — it is
        the exact field the old soil_attribution read before returning {}."""
        b = step2.context_brief(_Ctx())
        keys = b.split("PER-COLUMN METADATA")[1].split("\n")[1]
        assert "soil_profile" in keys
        assert "soil," not in keys and not keys.strip().endswith("soil")

    def test_derived_metrics_are_shown_with_what_they_come_from(self):
        """runoff_fraction is QOVER/(QCHARGE+QOVER), not a fraction of P — I got
        that wrong myself by reading the name instead of the derivation."""
        b = step2.context_brief(_Ctx(sem={
            "runoff_fraction": {"units": "1", "from": ["QOVER", "QCHARGE"]}}))
        assert "runoff_fraction" in b and "from QOVER, QCHARGE" in b


class TestTheCeilingIsNotAQuota:

    def test_more_than_five_figures_are_cut(self):
        class _Client:
            def ask(self, messages, system_message=None):
                figs = [{"id": f"f{i}", "question": "q", "code": "pass"}
                        for i in range(9)]
                import json
                return json.dumps({"notes": "n", "figures": figs})
        spec = step2.propose(_Ctx(), client=_Client())
        assert len(spec["figures"]) == step2.MAX_PLOTS

    def test_duplicate_ids_are_dropped_not_overwritten(self):
        """Two figures with one id would write to the same PNG path, and the
        second would silently replace the first."""
        class _Client:
            def ask(self, messages, system_message=None):
                import json
                return json.dumps({"notes": "n", "figures": [
                    {"id": "same", "question": "a", "code": "pass"},
                    {"id": "same", "question": "b", "code": "pass"}]})
        spec = step2.propose(_Ctx(), client=_Client())
        assert len(spec["figures"]) == 1

    def test_a_fenced_json_reply_still_parses(self):
        class _Client:
            def ask(self, messages, system_message=None):
                return '```json\n{"notes": "n", "figures": []}\n```'
        assert step2.propose(_Ctx(), client=_Client())["figures"] == []
