#!/usr/bin/env python3
"""mcp/elm-mcp/src/build_column_inputs.py — the per-column input builder.

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
        "build_column_inputs", ROOT / "mcp" / "elm-mcp" / "src" / "build_column_inputs.py")
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


class TestSoilReachesTheGenerator:
    """The per-column soil key is `soil_profile`, not `soil`.

    Reading the wrong one returned None for every column, so every surface
    file was built from an EMPTY soil dict and they all collapsed to a single
    identical file — same content hash for 19 different columns. Nothing
    fails: ELM runs, every column silently carries template soil instead of
    its own, and the ensemble stops being an ensemble.

    Caught by comparing against what the working path produced for a real
    run, not by a fixture. The generators are expensive, so this test asserts
    on the ARGUMENTS instead.
    """

    def _capture(self, tmp_path, column, monkeypatch):
        m = _mod()
        seen = {}

        def fake_build_one(**kw):
            seen.update(kw)
            return {"domain": "/d.nc", "surface": "/s.nc"}

        monkeypatch.setattr(m, "build_one", fake_build_one)
        (tmp_path / "columns.json").write_text(
            json.dumps({"columns": [column]}))
        m.build_all(tmp_path, quiet=True)
        return seen

    def test_soil_profile_is_forwarded(self, tmp_path, monkeypatch):
        profile = {"source": "CONUS 1 km", "num_layers": 10,
                   "layers": [{"sand_pct": 40.2, "clay_pct": 24.1}]}
        seen = self._capture(tmp_path, {
            "id": "col_01", "lat": 38.4625, "lon": -107.3625,
            "soil_profile": profile}, monkeypatch)
        assert seen["soil"] == profile, (
            "soil_profile must reach the generator — an empty dict makes "
            "every column produce the SAME surface file")

    def test_a_column_with_no_soil_is_not_silently_normal(self, tmp_path,
                                                          monkeypatch):
        """No soil is a legitimate state (cold, no donor). It must arrive as
        falsy rather than as a plausible-looking empty profile."""
        seen = self._capture(tmp_path, {
            "id": "col_01", "lat": 38.4, "lon": -107.3}, monkeypatch)
        assert not seen["soil"]

    def test_legacy_soil_key_still_read(self, tmp_path, monkeypatch):
        seen = self._capture(tmp_path, {
            "id": "col_01", "lat": 38.4, "lon": -107.3,
            "soil": {"sand_pct": 40}}, monkeypatch)
        assert seen["soil"] == {"sand_pct": 40}
