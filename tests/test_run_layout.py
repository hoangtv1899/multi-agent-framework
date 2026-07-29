#!/usr/bin/env python3
"""src/core/run_layout.py — one place that knows where things live.

The point of the module is that moving a file becomes one edit instead of
nine, and that runs produced before the move stay readable. Those runs are the
only record of experiments that cost hours of compute, so "we reorganised the
directory" must never mean "the old results are unreadable".
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core import run_layout as L        # noqa: E402


def _make(tmp_path, files):
    rd = tmp_path / "elm_run_X"
    for f in files:
        p = rd / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    rd.mkdir(parents=True, exist_ok=True)
    return rd


class TestLegacyRunsStayReadable:

    def test_old_flat_layout_resolves(self, tmp_path):
        rd = _make(tmp_path, ["columns.json", "run_plan.json", "plan.json",
                              "LLM_ANALYSIS_INPUT.json"])
        assert L.resolve(rd, "columns").name == "columns.json"
        assert L.resolve(rd, "columns").parent == rd        # NOT 01_inputs
        assert L.resolve(rd, "strategy").name == "plan.json"
        assert L.resolve(rd, "experiment").name == "LLM_ANALYSIS_INPUT.json"

    def test_canonical_wins_when_both_exist(self, tmp_path):
        """A run written during the transition may have both. The new location
        is the truth; the old one is whatever was left behind."""
        rd = _make(tmp_path, ["columns.json", "01_inputs/columns.json"])
        assert L.resolve(rd, "columns").parent.name == "01_inputs"

    def test_legacy_is_detected_by_what_changed(self, tmp_path):
        old = _make(tmp_path / "a", ["columns.json"])
        new = _make(tmp_path / "b", ["experiment.json"])
        assert L.is_legacy(old)
        assert not L.is_legacy(new)


class TestRetiredNames:
    """A retired name must say WHERE the content went. Reporting it as an
    unknown key sends the reader looking for a file that was deliberately
    removed."""

    @pytest.mark.parametrize("name,into", [
        ("reception_brief", "reception"),
        ("plan",            "strategy"),
        ("run_summary",     "experiment"),
        ("analysis_report", "analysis"),
    ])
    def test_resolve_points_at_the_replacement(self, tmp_path, name, into):
        with pytest.raises(KeyError, match=into):
            L.resolve(tmp_path, name)

    def test_writing_a_retired_name_is_refused(self, tmp_path):
        with pytest.raises(KeyError, match="strategy"):
            L.write_path(tmp_path, "plan")

    def test_unknown_name_lists_what_is_known(self, tmp_path):
        with pytest.raises(KeyError, match="columns"):
            L.resolve(tmp_path, "nonsense")


class TestWriting:

    def test_write_path_is_always_canonical(self, tmp_path):
        rd = _make(tmp_path, ["columns.json"])          # legacy present
        p = L.write_path(rd, "columns")
        assert p.parent.name == "01_inputs", \
            "a new run must write the canonical location even when the " \
            "legacy file is sitting right there"

    def test_write_path_creates_the_parent(self, tmp_path):
        p = L.write_path(tmp_path / "fresh", "columns")
        assert p.parent.is_dir()


class TestSurface:

    def test_the_four_are_the_contract(self):
        assert L.BOUNDARY == ("reception", "strategy", "experiment", "analysis")

    def test_surface_distinguishes_unfinished_from_uninterpreted(self, tmp_path):
        """Missing `experiment` means the compute never finished; missing
        `analysis` means it finished and was never interpreted. Different
        problems, and worth telling apart at a glance."""
        mid  = _make(tmp_path / "a", ["reception.json", "strategy.json"])
        done = _make(tmp_path / "b", ["reception.json", "strategy.json",
                                      "experiment.json"])
        assert L.surface(mid)["experiment"] is False
        assert L.surface(done)["experiment"] is True
        assert L.surface(done)["analysis"] is False

    def test_strays_notices_the_layout_regrowing(self, tmp_path):
        rd = _make(tmp_path, ["reception.json", "strategy.json",
                              "experiment.json", "analysis.json",
                              "run.log", "something_new.json"])
        assert L.strays(rd) == ["something_new.json"]

    def test_a_clean_run_has_no_strays(self, tmp_path):
        rd = _make(tmp_path, ["reception.json", "strategy.json",
                              "experiment.json", "analysis.json",
                              "run.log", "sampling_design.png",
                              "01_inputs/columns.json"])
        assert L.strays(rd) == []
