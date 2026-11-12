#!/usr/bin/env python3
"""Coupling as a RUN of the downstream model — offline, on a fixture prior.

The old shape (step 4d: a PFLOTRAN appendix on the tail of an ELM run,
through a local deck builder that bypassed the model server) is deleted.
A coupling is the coupling archetype now: reception carries the prior
domain forward, and the PFLOTRAN manager's _build_coupled_columns turns the
prior run's RECORD into columns that carry their own boundary. What is
pinned here: the columns are the prior's verbatim; the flux, the water
table and the sentence describing them ride each column; every refusal
names its reason; the base routes the archetype; and the old path is gone.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "mcp" / "pflotran-mcp"))
sys.path.insert(0, str(ROOT / "mcp" / "elm-mcp" / "src"))

from pflotran_exp_manager import PFLOTRANExpManager     # noqa: E402
from core import exp_manager_base as base               # noqa: E402


def _prior(tmp_path, n=2, var="QDRAI", units="mm/day", wtd=(4.25, 17.8)):
    """A finished ELM run's record, as the coupling reads it."""
    d = tmp_path / "elm_run_fixture"
    (d / "03_results").mkdir(parents=True)
    cols = [{"id": f"col_{i+1:02d}", "lat": 39.9 + i * 0.01, "lon": -75.7,
             "elevation_m": 100.0 + 20 * i, "band": i + 1,
             "band_range_m": [90 + 20 * i, 110 + 20 * i], "pinned": False,
             # ELM's own extras, which must NOT travel onto a PFLOTRAN column
             "soil_summary": "loam-ish", "forcing_cell": [3, 4],
             "soil_profile": {"layers": []}, "soil_source": "donor"}
            for i in range(n)]
    (d / "columns.json").write_text(json.dumps({"columns": cols}))
    (d / "experiment.json").write_text(json.dumps({"columns": [
        {"case_name": c["id"], "metrics": {"water_table_depth_m": wtd[i]}}
        for i, c in enumerate(cols)]}))
    dates = [f"2010-01-{15+k:02d}" for k in range(10)]
    (d / "03_results" / "extracted.json").write_text(json.dumps({
        "metadata": {}, "data": {
            c["id"]: {"dates": dates,
                      "variables": {var: {"units": units,
                                          "values": [1.0 + k * 0.1 for k in range(10)]},
                                    "ZWT": {"units": "m", "values": [1] * 10}}}
            for c in cols}}))
    (d / "reception.json").write_text(json.dumps({
        "grid": {"n_in_basin": 5}, "observations": {"ok": True},
        "brief": {"domain": {"name": "Fixtureville"}}}))
    return d


def _mgr(tmp_path):
    m = PFLOTRANExpManager.__new__(PFLOTRANExpManager)
    m.run_dir = tmp_path
    m.input_dir = tmp_path / "01_inputs"
    m.input_dir.mkdir(exist_ok=True)
    return m


def _config(prior, var="QDRAI"):
    return {"brief": {"design_archetype": "coupling",
                      "coupling": {"from_model": "elm", "to_model": "pflotran",
                                   "coupling_variable": var,
                                   "prior_run_dir": str(prior)}},
            "strategy": {"archetype": "coupling"}}


class TestTheColumnsAreThePriorsVerbatim:
    def test_ids_coords_and_bands_travel_elm_extras_do_not(self, tmp_path):
        prior = _prior(tmp_path)
        out = _mgr(tmp_path)._build_coupled_columns(_config(prior))
        assert out["approach"] == "coupled" and out["n_columns"] == 2
        c = out["columns"][0]
        assert (c["id"], c["lat"], c["band"]) == ("col_01", 39.9, 1)
        for elm_only in ("soil_summary", "forcing_cell", "soil_profile", "soil_source"):
            assert elm_only not in c, elm_only

    def test_each_column_carries_its_own_boundary_and_anchor(self, tmp_path):
        prior = _prior(tmp_path)
        cols = _mgr(tmp_path)._build_coupled_columns(_config(prior))["columns"]
        assert cols[0]["water_table_m"] == 4.25 and cols[1]["water_table_m"] == 17.8
        assert cols[0]["daily_flux_mm_day"][0] == 1.0 and len(cols[0]["daily_flux_mm_day"]) == 10
        assert cols[0]["coupling_variable"] == "QDRAI"
        assert cols[0]["coupled_from"] == "elm_run_fixture"
        d = cols[0]["flux_description"]
        assert "QDRAI" in d and "already" in d and "solved" in d
        assert (cols[0]["forcing_start"], cols[0]["forcing_end"]) == (2010, 2010)

    def test_the_coupled_keys_are_named_in_the_metadata(self, tmp_path):
        keys = _mgr(tmp_path)._column_keys()
        for k in ("coupled_from", "coupling_variable", "flux_description"):
            assert k in keys.keep, k
        assert "daily_flux_mm_day" in keys.drop


class TestEveryRefusalNamesItsReason:
    def test_no_prior_dir(self, tmp_path):
        with pytest.raises(RuntimeError, match="prior run's directory"):
            _mgr(tmp_path)._build_coupled_columns(
                {"brief": {"coupling": {"prior_run_dir": str(tmp_path / "nope")}},
                 "strategy": {}})

    def test_prior_without_an_extract(self, tmp_path):
        prior = _prior(tmp_path)
        (prior / "03_results" / "extracted.json").unlink()
        with pytest.raises(RuntimeError, match="extracted.json"):
            _mgr(tmp_path)._build_coupled_columns(_config(prior))

    def test_missing_variable_names_what_is_there(self, tmp_path):
        prior = _prior(tmp_path)
        with pytest.raises(RuntimeError, match="QCHARGE.*QDRAI"):
            _mgr(tmp_path)._build_coupled_columns(_config(prior, var="QCHARGE"))

    def test_the_variable_is_read_out_of_prose(self, tmp_path):
        """The brief's coupling_variable is settled in conversation and often
        arrives as a sentence; the token that the extract holds is the one."""
        prior = _prior(tmp_path)
        cols = _mgr(tmp_path)._build_coupled_columns(_config(
            prior, var="ELM sub-surface drainage QDRAI -> PFLOTRAN top flux"))["columns"]
        assert cols[0]["coupling_variable"] == "QDRAI"

    def test_wrong_units_are_refused_not_converted(self, tmp_path):
        prior = _prior(tmp_path, units="mm/s")
        with pytest.raises(RuntimeError, match="mm/s.*mm/day"):
            _mgr(tmp_path)._build_coupled_columns(_config(prior))

    def test_a_column_with_no_solved_water_table_is_skipped_not_fatal(
            self, tmp_path):
        """The driver honestly never solved it (Naches col_04, 2026-08-25:
        ELM wrote 365 days of ZWT fill values while QDRAI was real), so the
        column is left out WITH ITS REASON ON THE RECORD — not invented an
        anchor, and not allowed to sink its 18 siblings."""
        prior = _prior(tmp_path, wtd=(4.25, None))
        res = _mgr(tmp_path)._build_coupled_columns(_config(prior))
        assert [c["id"] for c in res["columns"]] == ["col_01"]
        assert res["n_columns"] == 1
        (skip,) = res["skipped_columns"]
        assert skip["id"] == "col_02"
        assert "water_table_depth_m" in skip["reason"]

    def test_a_column_with_an_empty_forward_series_is_skipped_not_fatal(
            self, tmp_path):
        """The Naches col_04 case for the flux itself (design, section 5):
        the driver's extract holds the variable for its siblings but not
        for this column. The same skip path as the unsolved water table,
        the reason on the record; the old raise sank every column."""
        prior = _prior(tmp_path)
        ext = prior / "03_results" / "extracted.json"
        d = json.loads(ext.read_text())
        del d["data"]["col_02"]["variables"]["QDRAI"]
        ext.write_text(json.dumps(d))
        res = _mgr(tmp_path)._build_coupled_columns(_config(prior))
        assert [c["id"] for c in res["columns"]] == ["col_01"]
        assert res["n_columns"] == 1
        (skip,) = res["skipped_columns"]
        assert skip["id"] == "col_02"
        assert "wrote no QDRAI" in skip["reason"]
        assert "ZWT" in skip["reason"]                 # what it does have


class TestTheSinkIsNamedInTheRecord:
    """docs/coupling/lateral_sink_design.md, sections 6 and 7: every
    contract key classified, the outflow fields explained with their sign
    and unit, and the fixed-head sentence gone."""

    SINK_KEYS = ("sink_datum_source", "sink_datum_m", "sink_datum_applied_m",
                 "sink_datum_clipped_to_domain", "sink_datum_raised_to_band",
                 "sink_band_m", "sink_sy",
                 "sink_sy_source", "sink_tau_days", "sink_tau_source",
                 "sink_conductance_m", "sink_implied_tau_days", "hand_m",
                 "hand_path_km", "hand_stream_area_km2", "hand_threshold_km2",
                 "std_elev_m", "conus2_wtd_m")

    def test_every_contract_key_is_kept_and_optional(self, tmp_path):
        keys = _mgr(tmp_path)._column_keys()
        for k in self.SINK_KEYS:
            assert k in keys.keep and k in keys.optional, k
        for k in ("lateral_sink", "lateral_outflow"):
            assert keys.drop.get(k), k

    def test_the_semantics_say_sign_unit_and_the_three_bottoms(self):
        fs = PFLOTRANExpManager.FIELD_SEMANTICS
        day = fs["lateral_outflow_mm_day"]
        assert day["units"] == "mm/day"
        assert "positive" in day["note"] and "365.25" in day["note"]
        assert "lateral_sink Water Mass [kg/y]" in " ".join(day["from"])
        win = fs["lateral_outflow_window_mm"]
        assert win["units"] == "mm" and "last" in " ".join(win["from"])
        note = fs["water_table_m"]["note"]
        for state in ("ANCHORED", "SEALED", "LATERAL SINK"):
            assert state in note, state
        assert "fixed head" not in note.lower()
        assert "lateral outflow" in fs["_not_computed"]["note"]
        assert "drainage_flux" in fs["_not_computed"]["fields"]


class TestTheBaseRoutesTheArchetype:
    def test_materialize_takes_the_coupling_branch(self, tmp_path, monkeypatch):
        prior = _prior(tmp_path)
        m = _mgr(tmp_path)
        seen = {}
        monkeypatch.setattr(m, "_persist_columns",
                            lambda res, cols, plan, cfg: seen.update(res=res, n=len(cols)) or {"ok": True})
        cfg = dict(_config(prior))
        monkeypatch.setattr(m, "check", lambda plan, config: config)
        out = m._materialize({"anything": True}, cfg)
        assert out == {"ok": True} and seen["n"] == 2
        assert seen["res"]["approach"] == "coupled"

    def test_is_coupling_reads_either_word(self):
        assert base._is_coupling({"archetype": "coupling"}, {})
        assert base._is_coupling({}, {"design_archetype": "coupling"})
        assert base._is_coupling({"coupling": {"from_model": "elm"}}, {})
        assert not base._is_coupling({"archetype": "site"}, {})

    def test_the_old_tail_is_gone(self):
        from elm_exp_manager import ELMExpManager
        assert not hasattr(ELMExpManager, "_couple_pflotran")
        assert not hasattr(ELMExpManager, "COUPLES_TO")
        assert not hasattr(base.ExperimentManagerBase, "COUPLES_TO")
        assert not (ROOT / "tools" / "build_pflotran_cases.py").exists()
        assert not (ROOT / "tools" / "analyze_pflotran_coupled.py").exists()


# TestReceptionCarriesThePriorForward WAS HERE (eight tests of
# reception_llm._carry_prior_forward, which copied a prior run's domain,
# observations and rain into a coupling request). Reception's code for the
# studies set aside was deleted on 2026-11-11, and those tests with it.
