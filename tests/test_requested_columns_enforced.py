"""The corrected column count must reach the sampler, not just the report.

tests/test_requested_columns.py already pins the DETECTION: strategy_check.check()
notices that the request asked for N and the strategy designed M, and rewrites
strategy["sampling"]["n_columns"]. That test passes, and passed throughout.

What nothing tested was whether _materialize USES the corrected value.

    912   n_total = config.get("n_columns") or exp._n_from_plan(plan)   # the PLAN
    923   config  = self.check(plan, config)                            # the STRATEGY

n_total was read at 912 from the uncorrected plan; check() rewrote the strategy
at 923; nothing re-read it. So on 2026-08-07 a request for 2 columns printed

    n_columns: the request asked for 2, the strategy designed 17
               — using the 2 that was asked for

and then built 17, warm-starting every one of them against the CONUS restarts.
Eight times the compute, with the guard reporting success.

This is the shape of test that was missing: not "does the function return the
right number" but "does the number the function returned change what runs".
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.exp_manager_base import ExperimentManagerBase          # noqa: E402

# expand_sampling lives in tools/, which is a script directory rather than a
# package — loaded by path here exactly as the framework's _load_tool does, so
# the test reads the same _n_from_plan the sampler will.
import importlib.util as _il                                     # noqa: E402
_spec = _il.spec_from_file_location(
    "expand_sampling", ROOT / "tools" / "expand_sampling.py")
exp = _il.module_from_spec(_spec)
sys.modules["expand_sampling"] = exp
_spec.loader.exec_module(exp)


def _reception(requested):
    return {"brief": {
        "domain": {"name": "Upper Gunnison", "huc": "14020002",
                   "bbox": [-107.5, 38.4, -106.5, 39.0]},
        "run_settings": {"requested_n_columns": requested,
                         "resolved_period": {"yr_start": 2020, "yr_end": 2020,
                                             "source": "user"}}}}


def _strategy(designed, bands=5):
    return {"sampling": {"n_columns": designed, "n_bands": bands}}


class TestTheCorrectionReachesTheSampler:

    def test_the_requested_count_wins_over_the_designed_one(self, tmp_path):
        """The end-to-end property, exercised through check() as _materialize
        now calls it: the config that comes back carries the corrected count
        where _n_from_plan will find it."""
        m = ExperimentManagerBase(base_output_dir=str(tmp_path))
        recep = _reception(2)
        config = {"reception": recep, "brief": recep["brief"],
                  "strategy": _strategy(17)}

        out = m.check({}, config)
        n = out.get("n_columns") or exp._n_from_plan(out.get("strategy") or {})
        assert n == 2, (
            f"the sampler would build {n} columns for a request of 2 — the "
            f"correction was recorded but not applied")

    def test_the_uncorrected_plan_is_not_what_gets_read(self, tmp_path):
        """Guards the exact regression: reading from `plan` rather than from
        the corrected strategy silently discards the correction."""
        m = ExperimentManagerBase(base_output_dir=str(tmp_path))
        recep = _reception(3)
        plan = _strategy(19)                       # the planner's own number
        config = {"reception": recep, "brief": recep["brief"],
                  "strategy": _strategy(19)}

        out = m.check(plan, config)
        from_plan = exp._n_from_plan(plan)
        from_corrected = exp._n_from_plan(out.get("strategy") or {})
        assert from_plan == 19, "fixture wrong: the plan should be untouched"
        assert from_corrected == 3, "the correction did not reach the strategy"
        assert from_plan != from_corrected, (
            "this test cannot detect the bug it exists for")

    def test_no_requested_count_leaves_the_design_alone(self, tmp_path):
        """A user who states no number gets the planner's design, unchanged."""
        m = ExperimentManagerBase(base_output_dir=str(tmp_path))
        recep = _reception(None)
        recep["brief"]["run_settings"].pop("requested_n_columns")
        config = {"reception": recep, "brief": recep["brief"],
                  "strategy": _strategy(19)}
        out = m.check({}, config)
        assert exp._n_from_plan(out.get("strategy") or {}) == 19


class TestTheOrderingItself:
    """Read the source, because the bug was an ORDER and order has no runtime
    signature once both lines have executed."""

    def test_check_runs_before_the_count_is_read(self):
        src = (ROOT / "src" / "core" / "exp_manager_base.py").read_text()
        body = src[src.index("def _materialize"):]
        body = body[:body.index("def _already_executable")]
        i_check = body.index("self.check(plan, config)")
        i_count = body.index("n_total = ")
        assert i_check < i_count, (
            "n_total is read before check() corrects the strategy — the "
            "correction will be computed, printed, and discarded")
