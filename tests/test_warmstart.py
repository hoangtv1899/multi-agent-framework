"""Offline tests for the warm-start path (no ELM, no CONUS files, no LLM).

Warm start is a two-run pattern: make_warmstart edits a COMPLETED run's restart
files ("carriers") and overwrites their slow state from a CONUS/Fan prior, so
these tests cover the parts that decide WHETHER and WITH WHAT a column gets
warm-started — band selection, FINIDAT propagation, ledger honesty, and the
refusal paths. The NetCDF surgery itself needs real restarts and is exercised
by tests/test_elm_e2e_minimal.py under --runcompute.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.columns_to_plan import build_ledger, columns_to_elm_plan  # noqa: E402


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mw = _load("mw_mod", "tools/make_warmstart.py")


COLS = [{"id": f"col_{i:02d}", "lat": 46.0 + i * 0.4, "lon": -121.0,
         "fan_wtd_m": 1.0 + i, "elevation_m": 800 + 100 * i} for i in range(1, 5)]

MANIFEST = """# comment line
BAND   LAT         SIZE   DATE         PATH
lat10  43-45N      76G    2040-01-01   /fake/conus_lat10.elm.r.nc
lat11  45-47N      66G    2040-01-01   /fake/conus_lat11.elm.r.nc
lat12  47-49N      53G    2040-01-01   /fake/conus_lat12.elm.r.nc
"""


# ── CONUS band selection ─────────────────────────────────────────────────────
class TestBandSelection:
    def test_manifest_parses_bands_and_paths(self, tmp_path):
        f = tmp_path / "MANIFEST.txt"
        f.write_text(MANIFEST)
        rows = mw.read_conus_manifest(f)
        assert [r[0] for r in rows] == ["lat10", "lat11", "lat12"]
        assert rows[1] == ("lat11", 45, 47, "/fake/conus_lat11.elm.r.nc")

    def test_resolve_accepts_manifest_file_and_dir(self, tmp_path):
        f = tmp_path / "MANIFEST.txt"
        f.write_text(MANIFEST)
        assert len(mw.resolve_conus_sources(f)) == 3

        d = tmp_path / "bands"
        d.mkdir()
        (d / "conus_lat11.nc").write_text("")
        (d / "conus_lat12.nc").write_text("")
        got = mw.resolve_conus_sources(d)
        assert len(got) == 2 and all(g[1] is None for g in got)   # ranges unknown

    def test_missing_source_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            mw.resolve_conus_sources(tmp_path / "nope.txt")

    def test_latitude_picks_the_covering_band_without_opening_files(self, tmp_path):
        """A basin straddling a band edge must resolve to BOTH bands.

        This is the regression that mattered: Naches spans 45-47N and 47-49N,
        so the single-file interface silently skipped the columns outside
        whichever band was passed, and the run half-cold-started.
        """
        f = tmp_path / "MANIFEST.txt"
        f.write_text(MANIFEST)
        bs = mw.ConusBandSet(mw.resolve_conus_sources(f))
        pick = lambda lat: [s for s in bs.sources
                            if s[1] is not None and s[1] <= lat < s[2]][0][0]
        assert pick(46.74) == "lat11"
        assert pick(47.11) == "lat12"
        assert pick(44.00) == "lat10"
        assert bs._open == {}          # nothing opened — selection is metadata-only

    def test_latitude_outside_every_band_selects_nothing(self, tmp_path):
        f = tmp_path / "MANIFEST.txt"
        f.write_text(MANIFEST)
        bs = mw.ConusBandSet(mw.resolve_conus_sources(f))
        band, src = bs.for_lat(20.0)           # sub-tropical: no band, no files
        assert (band, src) == (None, None)


# ── build_warmstart refuses rather than half-working ─────────────────────────
class TestBuildWarmstartGuards:
    def test_conus_without_a_restart_spec_raises(self):
        with pytest.raises(ValueError):
            mw.build_warmstart([], {}, "/tmp/x", source="conus")

    def test_case_without_carrier_restart_is_skipped(self, tmp_path):
        case = tmp_path / "1D_ELM.abc.col_01"
        (case / "run").mkdir(parents=True)          # no *.elm.r.*.nc inside
        got = mw.build_warmstart([str(case)], {"col_01": (46.7, -121.0)},
                                 tmp_path / "out", source="fan",
                                 fan_wtd={"col_01": 2.0}, quiet=True)
        assert got == {}

    def test_fan_source_without_a_target_is_skipped(self, tmp_path):
        case = tmp_path / "1D_ELM.abc.col_01"
        (case / "run").mkdir(parents=True)
        (case / "run" / "x.elm.r.1996-01-01-00000.nc").write_text("")
        got = mw.build_warmstart([str(case)], {"col_01": (46.7, -121.0)},
                                 tmp_path / "out", source="fan",
                                 fan_wtd={}, quiet=True)
        assert got == {}


# ── FINIDAT propagation ──────────────────────────────────────────────────────
class TestFinidatPropagation:
    def test_finidat_map_lands_on_every_matching_coupler(self):
        fmap = {c["id"]: {"finidat": f"/w/{c['id']}.nc", "source": "conus"}
                for c in COLS}
        plan = columns_to_elm_plan(COLS, finidat_map=fmap)
        ccs = plan["CONDITIONS_COUPLERS"]
        assert len(ccs) == len(COLS)
        assert all(cc["FINIDAT"] == f"/w/{cc['EXPERIMENT']}.nc" for cc in ccs)

    def test_plain_string_map_entries_also_work(self):
        plan = columns_to_elm_plan(COLS, finidat_map={"col_01": "/w/a.nc"})
        by = {cc["EXPERIMENT"]: cc for cc in plan["CONDITIONS_COUPLERS"]}
        assert by["col_01"]["FINIDAT"] == "/w/a.nc"

    def test_no_map_means_no_finidat_key_at_all(self):
        plan = columns_to_elm_plan(COLS)
        assert not any("FINIDAT" in cc for cc in plan["CONDITIONS_COUPLERS"])

    def test_partial_map_leaves_the_others_cold(self):
        plan = columns_to_elm_plan(COLS, finidat_map={"col_02": "/w/b.nc"})
        warm = [cc["EXPERIMENT"] for cc in plan["CONDITIONS_COUPLERS"]
                if cc.get("FINIDAT")]
        assert warm == ["col_02"]


# ── the ledger must not overstate what was warm-started ──────────────────────
class TestLedgerHonesty:
    def _init(self, ledger):
        return next(a for a in ledger if a["parameter"] == "initialization")

    def test_cold_is_reported_as_a_default(self):
        a = self._init(build_ledger(COLS, 1995, 1995, "native", "extrapolate",
                                    "baseline"))
        assert a["source"] == "DEFAULT" and "cold start" in a["value"]

    def test_fully_warm_reports_all_columns(self):
        fmap = {c["id"]: {"finidat": "/w/x.nc", "source": "conus"} for c in COLS}
        a = self._init(build_ledger(COLS, 1995, 1995, "native", "extrapolate",
                                    "baseline", finidat_map=fmap))
        assert a["source"] == "user"
        assert f"{len(COLS)}/{len(COLS)} columns" in a["value"]
        assert "COLD" not in a["value"]

    def test_partial_warm_says_the_rest_are_cold(self):
        """The failure this guards: a 12-of-14 warm start reported as 'warm'
        would let the interpreter attribute cold-start artefacts to physics."""
        fmap = {"col_01": {"finidat": "/w/x.nc", "source": "conus"}}
        a = self._init(build_ledger(COLS, 1995, 1995, "native", "extrapolate",
                                    "baseline", finidat_map=fmap))
        assert f"1/{len(COLS)} columns" in a["value"]
        assert "the rest COLD" in a["value"]

    def test_ledger_is_attached_by_the_plan_builder(self):
        plan = columns_to_elm_plan(COLS)
        params = {a["parameter"] for a in plan["assumptions_ledger"]}
        assert {"simulation period", "spin-up", "initialization",
                "soil configuration", "N (columns)"} <= params


# ── manager step 0b: refuse loudly, cold-start safely ────────────────────────
class TestManagerWarmstartStep:
    def _mgr(self, tmp_path):
        from core.elm_exp_manager import ELMExpManager
        return ELMExpManager(base_output_dir=str(tmp_path))

    def test_no_request_means_no_warm_start(self, tmp_path):
        assert self._mgr(tmp_path)._warmstart(COLS, {}) is None

    def test_requested_without_a_carrier_cold_starts(self, tmp_path, capsys):
        got = self._mgr(tmp_path)._warmstart(COLS, {"warm_start": {"source": "conus"}})
        assert got is None
        assert "no carrier run available" in capsys.readouterr().out

    def test_carrier_without_cases_json_cold_starts(self, tmp_path):
        carrier = tmp_path / "carrier"
        carrier.mkdir()
        got = self._mgr(tmp_path)._warmstart(
            COLS, {"warm_start": {"source": "conus", "carrier_run_dir": str(carrier)}})
        assert got is None

    def test_carrier_sharing_no_columns_cold_starts(self, tmp_path, capsys):
        """A carrier from a DIFFERENT basin must not have its state transplanted
        into these columns just because both runs have a 'col_01'."""
        carrier = tmp_path / "carrier"
        carrier.mkdir()
        (carrier / "cases.json").write_text(json.dumps(
            ["/scratch/1D_ELM.abc.other_99"]))
        got = self._mgr(tmp_path)._warmstart(
            COLS, {"warm_start": {"source": "conus", "carrier_run_dir": str(carrier)}})
        assert got is None
        assert "shares no columns" in capsys.readouterr().out

    def test_string_shorthand_is_accepted(self, tmp_path, capsys):
        """config['warm_start'] = 'conus' behaves like {'source': 'conus'}."""
        assert self._mgr(tmp_path)._warmstart(COLS, {"warm_start": "conus"}) is None
        assert "no carrier run available" in capsys.readouterr().out
