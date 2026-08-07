"""The extract contract must not depend on a backend's row shape.

`_as_extract` turns whatever a backend's extraction produced into the
{rows, units, extra_summary, llm_input} contract that `_package` and the
Analyzer consume. Backends disagree about how they hold rows:

    ELMResultsAnalyzer.results   Dict[str, Dict]  keyed by case name
    PFLOTRAN's                   List[Dict]

`list()` on the dict yields the case NAMES — 19 strings — and `_package` then
drops every non-dict. Job 770923 (2026-08-06): 19 columns ran clean, the ledger
recorded n_rows=19 because the COUNT was right, and experiment.json came out
with columns_total: 0.

The fix was verified at the time against a PFLOTRAN run, whose rows are a list
and therefore structurally could not expose it. That is the failure this file
exists to stop repeating: a backend must be verified on ITS OWN data shape.

Directly relevant to the ELM MCP work — see docs/ELM_MCP_PLAN.md §9 phases 1d
and 4. Both parity checks run through `_as_extract`, so a broken one is a ruler
that reads zero.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.exp_manager_base import ExperimentManagerBase      # noqa: E402


class TestRowsSurviveEitherShape:

    def _mgr(self, tmp_path):
        return ExperimentManagerBase(base_output_dir=str(tmp_path))

    def _rows(self):
        return [{"case_name": "col_01", "metrics": {"x": 1}},
                {"case_name": "col_02", "metrics": {"x": 2}}]

    def test_a_dict_of_rows_survives(self, tmp_path):
        import types
        m = self._mgr(tmp_path)
        obj = types.SimpleNamespace(
            results={r["case_name"]: r for r in self._rows()})
        got = m._as_extract(obj)
        assert [r["case_name"] for r in got["rows"]] == ["col_01", "col_02"], \
            "the dict was flattened to its keys"

    def test_a_list_of_rows_survives(self, tmp_path):
        import types
        m = self._mgr(tmp_path)
        got = m._as_extract(types.SimpleNamespace(results=self._rows()))
        assert [r["case_name"] for r in got["rows"]] == ["col_01", "col_02"]

    def test_the_packaged_file_has_the_columns_either_way(self, tmp_path):
        """The property that actually broke: a count is not evidence."""
        import types
        m = self._mgr(tmp_path)
        for shape in (self._rows(), {r["case_name"]: r for r in self._rows()}):
            pkg = m._package({}, m._as_extract(
                types.SimpleNamespace(results=shape)), {})
            assert pkg["columns_total"] == 2, f"lost rows for {type(shape).__name__}"
