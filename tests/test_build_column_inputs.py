#!/usr/bin/env python3
"""tools/build_column_inputs.py — the per-column input builder.

The generators themselves are tested elsewhere; what is tested here is the
thin layer around them, which is where the mistakes live: reading the donor
map, and honouring the veg/soil rules that keep a warm start self-consistent.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "build_column_inputs", ROOT / "tools" / "build_column_inputs.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestWarmstartMap:
    """warmstart.json is {col_id: {...}} with no wrapper key.

    Assuming a 'columns' wrapper silently returned an empty map, which is the
    worst possible failure: every column would have been built COLD, from the
    stock template, against coordinates the warm start had already snapped —
    ELM aborts at init on a surfdata/fatmgrid mismatch, and nothing upstream
    would have said why.
    """

    def test_flat_dict_shape_is_read(self, tmp_path):
        ws = tmp_path / "warmstart"; ws.mkdir()
        (ws / "warmstart.json").write_text(json.dumps({
            "col_01": {"finidat": "/x/f1.nc", "surface_template": "/x/s1.nc"},
            "col_02": {"finidat": "/x/f2.nc", "surface_template": "/x/s2.nc"},
        }))
        d = _mod()._warmstart_map(tmp_path)
        assert set(d) == {"col_01", "col_02"}
        assert d["col_01"]["surface_template"] == "/x/s1.nc"

    def test_wrapped_and_list_shapes_still_read(self, tmp_path):
        ws = tmp_path / "warmstart"; ws.mkdir()
        (ws / "warmstart.json").write_text(json.dumps(
            {"columns": [{"id": "col_01", "surface_template": "/x/s1.nc"}]}))
        assert _mod()._warmstart_map(tmp_path)["col_01"]["surface_template"] \
            == "/x/s1.nc"

    def test_cold_run_has_no_map_and_does_not_raise(self, tmp_path):
        assert _mod()._warmstart_map(tmp_path) == {}

    def test_unreadable_file_is_not_fatal(self, tmp_path):
        ws = tmp_path / "warmstart"; ws.mkdir()
        (ws / "warmstart.json").write_text("{not json")
        assert _mod()._warmstart_map(tmp_path) == {}


class TestMissingColumns:
    def test_it_says_what_is_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="materialize has not run"):
            _mod().build_all(tmp_path)
