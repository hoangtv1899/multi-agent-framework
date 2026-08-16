"""What a sweep column is CALLED, when something has to write it on an axis.

`treatment` is a nested dict. A generated plotting script told only that the key
exists does the one thing it can — prints it — and on the 2026-08-16 conceptual
sweep that put

    {'soil_texture': 5, 'prescribed_weather': {'fill': 'scale',
     'values': {'PRECTmms': 2.0}}}

on four x-axis ticks. The bars ended up in the top corner of their own canvas
with the label text running off the page. The analyzer's second round invented
its own short names and read fine, which is the tell: the model can shorten, it
just had nothing to shorten TO. So the name goes in the record before anyone
plots, and step 2 is told to use it.

These pin the label itself. Nothing here knows what ELM is or which factors
exist — the label is derived from whatever keys the design wrote.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.exp_manager_base import ExperimentManagerBase       # noqa: E402

label = ExperimentManagerBase._treatment_label


class TestTheLabelIsShortAndSaysWhatChanged:

    def test_a_scalar_factor_reads_as_a_level(self):
        assert label({"soil_texture": 5}) == "soil texture 5"

    def test_a_copied_field_says_so_rather_than_naming_a_verb(self):
        """`copy` is how the value got there, not what the column IS."""
        assert label({"prescribed_weather": "copy"}) == "weather as-is"

    def test_a_scaled_field_names_the_variable_and_the_factor(self):
        assert label({"prescribed_weather":
                      {"fill": "scale", "values": {"PRECTmms": 2.0}}}) \
            == "PRECTmms x2"

    def test_an_offset_keeps_its_sign(self):
        assert label({"prescribed_weather":
                      {"fill": "offset", "values": {"TBOT": 2.5}}}) == "TBOT +2.5"
        assert label({"prescribed_weather":
                      {"fill": "offset", "values": {"TBOT": -1}}}) == "TBOT -1"

    def test_a_set_value_reads_as_an_equality(self):
        assert label({"prescribed_weather":
                      {"fill": "set", "values": {"PRECTmms": 3}}}) == "PRECTmms = 3"

    def test_two_factors_join(self):
        assert label({"soil_texture": 55,
                      "prescribed_weather":
                          {"fill": "scale", "values": {"PRECTmms": 2.0}}}) \
            == "soil texture 55, PRECTmms x2"

    def test_the_four_columns_of_the_sweep_are_all_distinct(self):
        """The property that matters: four labels, four different strings. A
        label that collides is worse than the dict — two bars, one name."""
        sweep = [{"soil_texture": t, "prescribed_weather": w}
                 for t in (5, 55)
                 for w in ("copy", {"fill": "scale",
                                    "values": {"PRECTmms": 2.0}})]
        got = [label(t) for t in sweep]
        assert len(set(got)) == 4, got
        assert all(len(g) < 40 for g in got), "too long for an axis tick"

    def test_a_factor_nobody_has_invented_yet_still_gets_a_label(self):
        """Derived from the dict, not from a list of known factors."""
        assert label({"forcing_year": 1998}) == "forcing year 1998"

    def test_no_treatment_means_no_label(self):
        """A site run varies nothing deliberately. An absent label says "this
        column is not a treatment", which is true; inventing one would not be."""
        for empty in ({}, None, "nonsense", []):
            assert label(empty) is None


class TestTheLabelReachesTheRow:

    def test_merge_derives_it_from_the_treatment_it_just_copied(self, tmp_path):
        import json
        mgr = ExperimentManagerBase(base_output_dir=str(tmp_path))
        (mgr.run_dir / "columns.json").write_text(json.dumps({"columns": [
            {"id": "col_01", "lat": 47.0, "lon": -121.0,
             "treatment": {"soil_texture": 5, "prescribed_weather": "copy"}}]}))
        rows = [{"case_name": "col_01"}]
        mgr._merge_column_metadata(rows)
        assert rows[0]["treatment_label"] == "soil texture 5, weather as-is"

    def test_a_site_row_gains_nothing(self, tmp_path):
        import json
        mgr = ExperimentManagerBase(base_output_dir=str(tmp_path))
        (mgr.run_dir / "columns.json").write_text(json.dumps({"columns": [
            {"id": "col_01", "lat": 47.0, "lon": -121.0, "band": 2}]}))
        rows = [{"case_name": "col_01"}]
        mgr._merge_column_metadata(rows)
        assert "treatment_label" not in rows[0]
