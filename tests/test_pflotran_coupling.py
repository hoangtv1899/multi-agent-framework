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

    def test_a_column_with_no_solved_water_table(self, tmp_path):
        prior = _prior(tmp_path, wtd=(4.25, None))
        with pytest.raises(RuntimeError, match="col_02.*water_table_depth_m"):
            _mgr(tmp_path)._build_coupled_columns(_config(prior))


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


class TestReceptionCarriesThePriorForward:
    def _brief(self, ref):
        return {"design_archetype": "coupling",
                "coupling": {"prior_experiment": ref}}

    def test_resolves_copies_and_says_so(self, tmp_path):
        from agents.reception_llm import _carry_prior_forward
        prior = _prior(tmp_path)
        (prior / "conus2_subsurface.npz").write_bytes(b"x")
        (prior / "conus2_subsurface.json").write_text("{}")
        rd = tmp_path / "pflotran_run_x"
        rd.mkdir()
        prov = []
        brief = self._brief("the run elm_run_fixture please")
        got = _carry_prior_forward(brief, str(rd), prov)
        assert got["grid"] == {"n_in_basin": 5} and got["observations"] == {"ok": True}
        assert brief["coupling"]["prior_run_dir"] == str(prior.resolve())
        assert brief["coupling"]["n_columns_prior"] == 2
        assert (rd / "conus2_subsurface.npz").exists()
        note = prov[0]["note"]
        assert "CARRIED FORWARD" in note and "wtd_conus2.tif" in note  # absent, named

    def test_an_invented_domain_is_replaced_by_the_priors(self, tmp_path):
        # The LLM resolved a basin the request never named (seen live:
        # "Naches" written onto a Brandywine chain). The prior's domain wins
        # and the correction is recorded, not silent.
        from agents.reception_llm import _carry_prior_forward
        _prior(tmp_path)
        rd = tmp_path / "elm_run_x"
        rd.mkdir()
        brief = self._brief("elm_run_fixture")
        brief["domain"] = {"name": "Naches", "bbox": {"min_lon": -121.5}}
        _carry_prior_forward(brief, str(rd), [])
        assert brief["domain"]["name"] == "Fixtureville"
        assert any("Domain overridden" in a
                   for a in brief.get("assumptions", []))

    def test_a_defaulted_period_is_replaced_a_users_is_kept(self, tmp_path):
        # An iteration must hold the forcing years fixed: an LLM default
        # (source != "user") is replaced by the prior's period; a period the
        # user stated survives — they may narrow a follow-up on purpose.
        from agents.reception_llm import _carry_prior_forward
        prior = _prior(tmp_path)
        rec = json.loads((prior / "reception.json").read_text())
        rec["brief"]["run_settings"] = {"resolved_period": {
            "yr_start": 2010, "yr_end": 2010, "source": "user"}}
        (prior / "reception.json").write_text(json.dumps(rec))
        rd = tmp_path / "elm_run_x"
        rd.mkdir()

        defaulted = self._brief("elm_run_fixture")
        defaulted["run_settings"] = {"resolved_period": {
            "yr_start": 1979, "yr_end": 1979, "source": "default"}}
        _carry_prior_forward(defaulted, str(rd), [])
        got = defaulted["run_settings"]["resolved_period"]
        assert (got["yr_start"], got["yr_end"]) == (2010, 2010)
        assert got["source"] == "carried"
        assert any("Period overridden" in a
                   for a in defaulted.get("assumptions", []))

        stated = self._brief("elm_run_fixture")
        stated["run_settings"] = {"resolved_period": {
            "yr_start": 2011, "yr_end": 2011, "source": "user"}}
        _carry_prior_forward(stated, str(rd), [])
        kept = stated["run_settings"]["resolved_period"]
        assert (kept["yr_start"], kept["source"]) == (2011, "user")

    def test_an_unresolvable_prior_raises_with_candidates(self, tmp_path):
        from agents.reception_llm import _carry_prior_forward
        rd = tmp_path / "pflotran_run_x"
        rd.mkdir()
        with pytest.raises(ValueError, match="exactly one"):
            _carry_prior_forward(self._brief("no_such_run"), str(rd), [])

    def test_a_prose_reference_resolves_to_the_first_named_run(self, tmp_path):
        # SEEN LIVE (2026-08-24): the LLM wrote "X (referenced in the canceled
        # follow-up CANCELED_Y)" into prior_experiment, the substring pass
        # matched both X and CANCELED_Y, and the run died ambiguous. To a
        # reader that reference names X: the FIRST-named run is the referent,
        # later mentions are context.
        from agents.reception_llm import _carry_prior_forward
        _prior(tmp_path)
        (tmp_path / "CANCELED_elm_run_dead").mkdir()
        rd = tmp_path / "elm_run_x"
        rd.mkdir()
        brief = self._brief("elm_run_fixture (referenced in the canceled "
                            "follow-up CANCELED_elm_run_dead)")
        _carry_prior_forward(brief, str(rd), [])
        assert brief["coupling"]["prior_run_dir"].endswith("elm_run_fixture")

    def test_a_true_tie_still_refuses(self, tmp_path):
        # Two directory names starting at the same character of the reference
        # (one a prefix of the other) is genuine ambiguity — no earliest
        # mention exists, so the refusal stands.
        from agents.reception_llm import _carry_prior_forward
        (tmp_path / "elm_run_20990101_000000").mkdir()
        (tmp_path / "elm_run_20990101_000000_2").mkdir()
        rd = tmp_path / "elm_run_x"
        rd.mkdir()
        with pytest.raises(ValueError, match="exactly one"):
            _carry_prior_forward(
                self._brief("continue elm_run_20990101_000000_2 please"),
                str(rd), [])

    def test_an_unresolvable_prior_becomes_a_question_not_a_crash(
            self, tmp_path, monkeypatch):
        # The same live failure, at the process() level: the ValueError must
        # come back as the clarify route — a question a person can answer —
        # and the minted directory must not survive ("a clarification mints
        # nothing").
        import types
        from agents.reception_llm import LLMReceptionAgent
        inst = LLMReceptionAgent.__new__(LLMReceptionAgent)  # no LLM loop
        inst._clients = {}
        inst.system = ""
        brief = {"intent": "design", "design_archetype": "coupling",
                 "coupling": {"prior_experiment": "no_such_run"}}
        inst.loop = types.SimpleNamespace(
            run=lambda system, msg: {"content": json.dumps(brief),
                                     "trace": [], "rounds": 1})
        rd = tmp_path / "elm_run_minted"

        def mint(_b):
            rd.mkdir()
            return rd

        pkg = inst.process("continue the coupling", run_dir=mint)
        assert pkg["route"]["action"] == "clarify"
        assert "exactly one" in pkg["route"]["questions"][0]
        assert not rd.exists()

    def test_a_coupling_brief_with_no_prior_raises(self, tmp_path):
        from agents.reception_llm import _carry_prior_forward
        with pytest.raises(ValueError, match="prior_experiment"):
            _carry_prior_forward({"design_archetype": "coupling"}, str(tmp_path), [])
