#!/usr/bin/env python3
"""The return leg of two-way coupling: PFLOTRAN's solved water table back
into ELM's initial state — offline.

Three pieces, each pinned: the finidat edit (set_water_table.py — ZWT and WA
mutually consistent by ELM's own relation, clamps reported, masked slots
untouched); the solved-water-table reader (the ONE public function beside the
PFLOTRAN server that owns the crossing definition); and the ELM manager's
_build_coupled_columns (the prior run's columns verbatim, each carrying its
next start and the delta an iteration watches).
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "mcp" / "elm-mcp" / "src"))

import set_water_table as swt                       # noqa: E402
from elm_exp_manager import ELMExpManager           # noqa: E402

_REAL_FINIDAT = ROOT / ("workflow_outputs/elm_run_20260813_233645/"
                        "warmstart/finidat_col_01.nc")
needs_finidat = pytest.mark.skipif(not _REAL_FINIDAT.exists(),
                                   reason="no real finidat fixture on disk")


# ── the finidat edit ─────────────────────────────────────────────────────

@needs_finidat
class TestSetWaterTable:
    def _copy(self, tmp_path):
        dst = tmp_path / "fini.nc"
        shutil.copy(_REAL_FINIDAT, dst)
        return dst

    def test_aquifer_regime_is_exact_and_consistent(self, tmp_path):
        import netCDF4
        import numpy as np
        f = self._copy(tmp_path)
        r = swt.apply(str(f), 4.291, quiet=True)
        assert r["regime"] == "aquifer" and r["exact"] and not r["clamped"]
        d = netCDF4.Dataset(f)
        z = float(np.ma.compressed(d.variables["ZWT"][:])[0])
        w = float(np.ma.compressed(d.variables["WA"][:])[0])
        d.close()
        assert z == pytest.approx(4.291)
        assert z == pytest.approx(swt.AQUIFER_MAX_M - w / swt.SY_MM_PER_M, abs=1e-9)

    def test_a_target_below_the_aquifer_is_clamped_and_says_so(self, tmp_path):
        f = self._copy(tmp_path)
        r = swt.apply(str(f), 46.8, quiet=True)
        assert r["clamped"] and r["written_m"] == swt.AQUIFER_MAX_M
        assert "CLAMPED from 46.8" in r["note"]
        assert r["new_wa_mm"] == 0.0

    def test_in_soil_is_approximate_and_says_so(self, tmp_path):
        f = self._copy(tmp_path)
        r = swt.apply(str(f), 1.2, quiet=True)
        assert r["regime"] == "in-soil" and not r["exact"]
        assert "APPROXIMATE" in r["note"] and r["new_wa_mm"] == swt.WA_MAX_MM

    def test_masked_slots_stay_masked(self, tmp_path):
        import netCDF4
        import numpy as np
        f = self._copy(tmp_path)
        before = int(np.ma.getmaskarray(
            netCDF4.Dataset(f).variables["ZWT"][:]).sum())
        swt.apply(str(f), 5.0, quiet=True)
        after = int(np.ma.getmaskarray(
            netCDF4.Dataset(f).variables["ZWT"][:]).sum())
        assert before == after and before > 0

    def test_refusals_name_their_reason(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            swt.apply(str(tmp_path / "nope.nc"), 5.0)
        f = self._copy(tmp_path)
        with pytest.raises(ValueError, match="metres below ground"):
            swt.apply(str(f), -1.0)


# ── the return-leg columns ───────────────────────────────────────────────

def _prior_pflotran(tmp_path, bottom_saturated=True):
    """A finished coupled PFLOTRAN run's record, as the return leg reads it."""
    d = tmp_path / "pflotran_run_fixture"
    (d / "03_results").mkdir(parents=True)
    cols = [{"id": "col_01", "lat": 39.9, "lon": -75.7, "elevation_m": 100.0,
             "band": 1, "pinned": False, "water_table_m": 4.25},
            {"id": "col_02", "lat": 39.95, "lon": -75.72, "elevation_m": 140.0,
             "band": 2, "pinned": False, "water_table_m": 17.0}]
    (d / "columns.json").write_text(json.dumps({"columns": cols}))
    atm = 101325.0
    depths = [0.5, 2.0, 5.0, 6.5]
    # final profile: crossing between 2.0 m (unsat) and 5.0 m (sat) -> ~4.0 m
    final1 = [atm - 5e4, atm - 2e4, atm + 1e4, atm + 2.5e4]
    final2 = ([atm - 5e4, atm - 4e4, atm - 2e4, atm + 1e4] if bottom_saturated
              else [atm - 5e4, atm - 4e4, atm - 2e4, atm - 1e4])
    (d / "03_results" / "extracted.json").write_text(json.dumps({
        "columns": {
            "col_01": {"depth_m": depths, "times_y": [0.0, 1.0],
                       "liquid_pressure_Pa": [[atm] * 4, final1],
                       "saturation": [[0.2] * 4, [0.31, 0.52, 1.0, 1.0]]},
            "col_02": {"depth_m": depths, "times_y": [0.0, 1.0],
                       "liquid_pressure_Pa": [[atm] * 4, final2]},
        }}))
    return d


def _cfg(prior):
    return {"brief": {"design_archetype": "coupling",
                      "coupling": {"from_model": "pflotran", "to_model": "elm",
                                   "coupling_variable": "water_table",
                                   "prior_run_dir": str(prior)}},
            "strategy": {"archetype": "coupling"}}


class TestTheReturnLeg:
    def test_each_column_carries_its_next_start_and_the_delta(self, tmp_path):
        out = ELMExpManager.__new__(ELMExpManager)._build_coupled_columns(
            _cfg(_prior_pflotran(tmp_path)))
        assert out["approach"] == "coupled" and out["coupling_variable"] == "water_table"
        c1 = out["columns"][0]
        # crossing between (2.0, -2e4) and (5.0, +1e4): 2 + (2/3)*3 = 4.0
        assert c1["initial_water_table_m"] == pytest.approx(4.0, abs=1e-3)
        assert c1["water_table_prior_m"] == 4.25
        assert c1["water_table_delta_m"] == pytest.approx(-0.25, abs=1e-3)
        assert c1["coupled_from"] == "pflotran_run_fixture"
        assert "water_table_m" not in c1        # the OLD anchor does not travel

    def test_a_water_table_outside_the_domain_is_refused_by_name(self, tmp_path):
        prior = _prior_pflotran(tmp_path, bottom_saturated=False)
        with pytest.raises(RuntimeError, match="col_02.*bottom cell unsaturated"):
            ELMExpManager.__new__(ELMExpManager)._build_coupled_columns(_cfg(prior))

    def test_the_coupling_keys_are_named_in_elms_metadata(self):
        keys = ELMExpManager.__new__(ELMExpManager)._column_keys()
        for k in ("coupled_from", "coupling_variable", "initial_water_table_m",
                  "initial_water_table_written_m", "initial_water_table_note",
                  "water_table_prior_m", "water_table_delta_m"):
            assert k in keys.keep and k in keys.optional, k

    def test_materialize_routes_the_elm_leg_too(self, tmp_path, monkeypatch):
        m = ELMExpManager.__new__(ELMExpManager)
        seen = {}
        monkeypatch.setattr(m, "check", lambda plan, config: config)
        monkeypatch.setattr(m, "_persist_columns",
                            lambda res, cols, plan, cfg: seen.update(n=len(cols)) or {"ok": 1})
        out = m._materialize({"x": 1}, _cfg(_prior_pflotran(tmp_path)))
        assert out == {"ok": 1} and seen["n"] == 2


# ── the one-allocation path ──────────────────────────────────────────────

class TestInAllocationRun:
    def test_inside_an_allocation_the_ensemble_runs_in_place(self, monkeypatch):
        """SLURM_JOB_ID set -> _build_cases runs the tool synchronously with
        in_allocation=True, attaches case dirs, and returns the experiments
        list (the base's polled-build convention) — never a Pending."""
        from core.exp_manager_base import Pending
        monkeypatch.setenv("SLURM_JOB_ID", "999001")
        monkeypatch.delenv("IDEAS_FORCE_SBATCH", raising=False)
        m = ELMExpManager.__new__(ELMExpManager)
        m.run_dir = Path("/tmp/x")
        m.input_dir = Path("/tmp/x/01_inputs")
        seen = {}

        class _Client:
            timeout = 300.0
            def call_tool_json(self, name, args):
                seen["tool"], seen["args"] = name, args
                return {"ran_in_allocation": True, "returncode": 0,
                        "n_cases": 2, "n_built": 2, "elapsed_s": 61.0,
                        "cases": [{"case_name": "col_01", "case_dir": "/c/1"},
                                  {"case_name": "col_02", "case_dir": "/c/2"}]}

        monkeypatch.setattr(m, "_mcp", lambda cfg: _Client())
        exps = [{"case_name": "col_01"}, {"case_name": "col_02"}]
        (m.input_dir).mkdir(parents=True, exist_ok=True)
        (m.input_dir / m.CASE_INPUTS).write_text("[]")
        out = m._build_cases(exps, {"study_walltime": "00:30:00"})
        assert not isinstance(out, Pending)
        assert out is exps and exps[0]["case_dir"] == "/c/1"
        assert seen["args"]["in_allocation"] is True

    def test_force_sbatch_overrides_the_detection(self, monkeypatch):
        """IDEAS_FORCE_SBATCH=1 restores the submit path even inside a job —
        proven by it reaching the announce step, which the in-place path
        never calls."""
        monkeypatch.setenv("SLURM_JOB_ID", "999001")
        monkeypatch.setenv("IDEAS_FORCE_SBATCH", "1")
        m = ELMExpManager.__new__(ELMExpManager)
        m.run_dir = Path("/tmp/x")
        m.input_dir = Path("/tmp/x/01_inputs")
        m.input_dir.mkdir(parents=True, exist_ok=True)
        (m.input_dir / m.CASE_INPUTS).write_text("[]")
        monkeypatch.setattr(m, "_mcp", lambda cfg: object())
        hit = {}
        monkeypatch.setattr(m, "_announce",
                            lambda *a, **k: hit.setdefault("announced", True) and "")
        with pytest.raises(Exception):
            m._build_cases([{"case_name": "c"}], {})
        assert hit.get("announced"), "the submit path (announce) was not taken"

    def test_the_server_refuses_in_allocation_outside_one(self, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        sys.path.insert(0, str(ROOT / "mcp" / "elm-mcp"))
        import main as elm_main
        r = json.loads(elm_main.run_elm_ensemble(
            run_dir=str(ROOT / "workflow_outputs" / "elm_run_20260813_233645"),
            in_allocation=True))
        assert "no SLURM_JOB_ID" in r.get("error", "")


def _delta_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "coupling_delta", ROOT / "tools" / "coupling_delta.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_convergence_is_movement_between_legs_not_an_offset_inside_one():
    """The real Brandywine chain, and the distinction the loop got wrong.

    The anchor offset (solved minus given, within one leg) sat at exactly
    +0.05 m in EVERY iteration while the state moved 46.8 -> 28.3 -> 27.8 m.
    Judged on the offset the loop called CONVERGED at iteration 0 with the
    answer 18 m away. The verdict must come from the movement.
    """
    seed = ROOT / "workflow_outputs" / "pflotran_run_20260819_074628"
    leg2 = ROOT / "workflow_outputs" / "pflotran_run_20260819_100147"
    leg3 = ROOT / "workflow_outputs" / "pflotran_run_20260819_102509"
    if not all(p.exists() for p in (seed, leg2, leg3)):
        pytest.skip("the Brandywine coupled chain is not on disk")
    mod = _delta_mod()

    # the seed leg has no predecessor: undecidable, NOT converged
    rows, prior = mod.deltas(str(seed))
    assert prior is None
    assert all(r["delta_m"] is None for r in rows)
    # ... and its anchor offset is the constant that fooled the old test
    assert all(abs(r["offset_m"]) <= 0.05 for r in rows)

    # leg 2 moved a long way from the seed — must NOT read as converged
    rows2, prior2 = mod.deltas(str(leg2))
    assert prior2 is not None and Path(prior2).name == seed.name
    assert max(abs(r["delta_m"]) for r in rows2) > 10.0

    # leg 3 is closer but still moving; the offset is unchanged at 0.05
    rows3, prior3 = mod.deltas(str(leg3))
    assert Path(prior3).name == leg2.name
    moved = max(abs(r["delta_m"]) for r in rows3)
    assert 0.1 < moved < 1.0
    assert all(abs(r["offset_m"]) <= 0.05 for r in rows3)


def test_the_previous_leg_is_walked_out_of_the_record():
    """this pflotran run -> its ELM leg -> that leg's prior pflotran run."""
    leg3 = ROOT / "workflow_outputs" / "pflotran_run_20260819_102509"
    if not leg3.exists():
        pytest.skip("the Brandywine coupled chain is not on disk")
    mod = _delta_mod()
    prev = mod.previous_leg(str(leg3))
    assert prev is not None
    assert prev.name == "pflotran_run_20260819_100147"
    # a run that was never coupled has no predecessor, and that is not an error
    assert mod.previous_leg(str(ROOT / "workflow_outputs")) is None


def test_no_ensemble_job_reads_as_in_allocation_not_as_old():
    """An in-allocation run has no job A because it never submitted one.

    The note used to blame a 2026-08-13 vintage for it, which is a different
    fact and a wrong one: nothing is missing from an ensemble that ran in
    place.
    """
    import json as _json
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from agents.analysis.step4_report import _slurm_elapsed
    import tempfile
    from pathlib import Path as _P

    with tempfile.TemporaryDirectory() as td:
        rd = _P(td)
        # ran inside an allocation: stage done, no ids at all
        (rd / "run_state.json").write_text(_json.dumps(
            {"stages": {"build_cases": {"status": "done", "n_results": 3}}}))
        assert "inside an existing allocation" in _slurm_elapsed(rd)["note"]

        # an older submitted run: job_id present, job_id_a absent
        (rd / "run_state.json").write_text(_json.dumps(
            {"stages": {"build_cases": {"status": "pending",
                                        "job_id": "770001"}}}))
        assert "predates job_id_a" in _slurm_elapsed(rd)["note"]


# ── the soil-moisture companion stamp ────────────────────────────────────

_REAL_SURFDATA = ROOT / ("workflow_outputs/elm_run_20260813_233645/"
                         "warmstart/surfdata_col_01.nc")
needs_surfdata = pytest.mark.skipif(not _REAL_SURFDATA.exists(),
                                    reason="no real surfdata fixture on disk")


def _surf(tmp_path, sand=0.0, organic=0.0, missing_sand=False):
    """A minimal ELM surface dataset: just what the pedotransfer reads."""
    import netCDF4
    p = tmp_path / "surf.nc"
    d = netCDF4.Dataset(p, "w")
    for name, n in (("nlevsoi", 10), ("lsmlat", 1), ("lsmlon", 1)):
        d.createDimension(name, n)
    if not missing_sand:
        v = d.createVariable("PCT_SAND", "f8", ("nlevsoi", "lsmlat", "lsmlon"))
        v[:] = sand
    v = d.createVariable("ORGANIC", "f8", ("nlevsoi", "lsmlat", "lsmlon"))
    v[:] = organic
    d.close()
    return p


class TestTheLayerGridAndPedotransfer:
    def test_the_grid_matches_the_measured_soil_bottom(self):
        z, dz = swt._layer_grid()
        assert len(z) == 15 and len(dz) == 15
        zi10 = 0.5 * (z[9] + z[10])
        assert zi10 == pytest.approx(swt.SOIL_BOTTOM_M, abs=0.01)

    def test_pedotransfer_is_elms_own(self, tmp_path):
        w = swt._watsat_from_surfdata(str(_surf(tmp_path, sand=30.0)))
        assert w[0] == pytest.approx(0.489 - 0.00126 * 30.0)
        w = swt._watsat_from_surfdata(str(_surf(tmp_path, organic=130.0)))
        assert w[0] == pytest.approx(0.9)


@needs_finidat
class TestApplyProfile:
    def _copy(self, tmp_path):
        dst = tmp_path / "fini.nc"
        shutil.copy(_REAL_FINIDAT, dst)
        return dst

    def _soil(self, f, row=None):
        """(liq, ice, live-rows, snow offset) for the 10 active layers."""
        import netCDF4
        import numpy as np
        d = netCDF4.Dataset(f)
        off = d.variables["H2OSOI_LIQ"].shape[1] - swt.ELM_NLEVGRND
        rows = np.where(~np.ma.getmaskarray(d.variables["ZWT"][:]))[0]
        r = rows[0] if row is None else row
        liq = np.array(d.variables["H2OSOI_LIQ"][r, :])
        ice = np.array(d.variables["H2OSOI_ICE"][r, :])
        d.close()
        return liq, ice, rows, off

    def test_liquid_is_saturation_times_elms_porosity(self, tmp_path):
        f = self._copy(tmp_path)
        _, ice, _, off = self._soil(f)
        r = swt.apply_profile(str(f), [0.0, 30.0], [0.5, 0.5],
                              str(_surf(tmp_path)), quiet=True)
        z, dz = swt._layer_grid()
        for j in range(10):
            want = max(0.5 * 0.489 * dz[j] * 1000.0 - float(ice[off + j]),
                       swt.WATMIN_KG_M2)
            assert r["new_liq_kg_m2"][j] == pytest.approx(want, rel=1e-4), j
        liq_after, _, _, _ = self._soil(f)
        assert float(liq_after[off]) == pytest.approx(r["new_liq_kg_m2"][0],
                                                      abs=1e-3)

    def test_deep_layers_and_ice_are_untouched(self, tmp_path):
        import numpy as np
        f = self._copy(tmp_path)
        liq0, ice0, _, off = self._soil(f)
        swt.apply_profile(str(f), [0.0, 30.0], [0.8, 0.8],
                          str(_surf(tmp_path)), quiet=True)
        liq1, ice1, _, _ = self._soil(f)
        assert np.allclose(liq0[off + 10:], liq1[off + 10:])   # layers 11-15
        assert np.allclose(ice0, ice1)                          # ice never moves

    def test_ice_displaces_liquid_and_is_counted(self, tmp_path):
        import netCDF4
        f = self._copy(tmp_path)
        _, _, rows, off = self._soil(f)
        with netCDF4.Dataset(f, "r+") as d:      # plant ice in layer 3
            ice = d.variables["H2OSOI_ICE"][:]
            ice[rows[0], off + 2] = 5.0
            d.variables["H2OSOI_ICE"][:] = ice
        r = swt.apply_profile(str(f), [0.0, 30.0], [0.4, 0.4],
                              str(_surf(tmp_path)), quiet=True)
        z, dz = swt._layer_grid()
        full = 0.4 * 0.489 * dz[2] * 1000.0
        assert r["new_liq_kg_m2"][2] == pytest.approx(
            max(full - 5.0, swt.WATMIN_KG_M2), abs=1e-3)
        assert r["n_ice_layers"] >= 1 and "ice left in place" in r["note"]

    def test_saturation_is_clipped_to_physical(self, tmp_path):
        f = self._copy(tmp_path)
        r = swt.apply_profile(str(f), [0.0, 30.0], [1.7, 1.7],
                              str(_surf(tmp_path)), quiet=True)
        assert max(r["saturation_at_layers"]) == pytest.approx(1.0)

    def test_refusals_name_their_reason(self, tmp_path):
        f = self._copy(tmp_path)
        with pytest.raises(ValueError, match="equal length"):
            swt.apply_profile(str(f), [0.0, 1.0, 2.0], [0.5, 0.5],
                              str(_surf(tmp_path)), quiet=True)
        with pytest.raises(KeyError, match="PCT_SAND"):
            swt.apply_profile(str(f), [0.0, 30.0], [0.5, 0.5],
                              str(_surf(tmp_path, missing_sand=True)),
                              quiet=True)


class TestTheProfileTravels:
    def test_the_profile_rides_the_column_uninterpreted(self, tmp_path):
        out = ELMExpManager.__new__(ELMExpManager)._build_coupled_columns(
            _cfg(_prior_pflotran(tmp_path)))
        c1 = out["columns"][0]
        prof = c1["initial_saturation_profile"]
        assert prof["saturation"] == [0.31, 0.52, 1.0, 1.0]   # the FINAL time
        assert prof["depth_m"] == [0.5, 2.0, 5.0, 6.5]
        assert prof["at_time_y"] == 1.0
        # col_02's block has no saturation -> no key, and that is fine
        assert "initial_saturation_profile" not in out["columns"][1]

    def test_the_new_keys_are_named_in_the_seam(self):
        keys = ELMExpManager.__new__(ELMExpManager)._column_keys()
        assert "initial_soil_moisture_note" in keys.keep
        assert "initial_soil_moisture_note" in keys.optional
        assert "initial_saturation_profile" in keys.drop
