"""Offline tests for the warm-start path (no ELM, no CONUS files, no LLM).

Warm start subsets one gridcell out of a CONUS 1-km restart into a standalone
single-column finidat — no carrier, no prior run. These tests cover the parts
that decide WHETHER and WITH WHAT a column gets warm-started: band selection,
FINIDAT propagation, ledger honesty, and the refusal paths.

They deliberately never touch the real CONUS files (3-79 GB each); the subset
itself is verified against them directly, and the resulting finidat is proven
by an actual ELM run.
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
fs = _load("fs_mod", "tools/make_finidat_subset.py")


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


# ── the subsetter refuses rather than half-working ───────────────────────────
class TestSubsetGuards:
    def test_donor_with_a_lake_or_glacier_landunit_is_rejected(self):
        """Those gridcells have 17+ columns; our cases build 16, and ELM's
        check_dim aborts on the mismatch. ~6% of CONUS gridcells."""
        assert set(fs.EXPECTED_ITYPLUN) == {1, 7, 8, 9}
        assert fs.EXTRA_LANDUNITS == {3, 4, 5, 6}

    def test_expected_layout_is_natveg_plus_three_urban(self):
        assert fs.EXPECTED_ITYPLUN == [1] + [7] * 5 + [8] * 5 + [9] * 5
        assert len(fs.EXPECTED_ITYPLUN) == 16

    def test_every_renumbered_vector_has_a_declared_rule(self):
        """A vector renumbered to the wrong target silently corrupts the
        initial state, so the rule table is asserted rather than assumed."""
        assert fs.RENUMBER["cols1d_landunit_index"] == "landunit"
        assert fs.RENUMBER["pfts1d_column_index"] == "column"
        assert fs.RENUMBER["pfts1d_landunit_index"] == "landunit"
        for v in ("grid1d_ixy", "grid1d_jxy", "cols1d_gridcell_index",
                  "pfts1d_gridcell_index", "land1d_gridcell_index"):
            assert fs.RENUMBER[v] is None          # -> collapses to gridcell 1
        assert set(fs.RENUMBER) >= {"cols1d_ixy", "cols1d_jxy",
                                    "pfts1d_ixy", "pfts1d_jxy"}

    def test_validate_flags_a_wrong_layout(self, tmp_path):
        pytest.importorskip("netCDF4")
        import netCDF4
        import numpy as np
        f = tmp_path / "bad.nc"
        with netCDF4.Dataset(f, "w") as d:
            d.createDimension("gridcell", 1); d.createDimension("landunit", 4)
            d.createDimension("column", 18); d.createDimension("pft", 32)
            v = d.createVariable("cols1d_ityplun", "i4", ("column",))
            v[:] = np.array([1, 2, 2] + [7] * 5 + [8] * 5 + [9] * 5)
            g = d.createVariable("grid1d_lon", "f8", ("gridcell",)); g[:] = -120.0
        problems = fs.validate(str(f))
        assert any("column=18" in p for p in problems)
        assert any("cols1d_ityplun" in p for p in problems)

    def test_validate_flags_unconverted_longitude(self, tmp_path):
        """CONUS stores 0-360; the domain files use -180..180. Disagreement
        aborts ELM at init on a surfdata/fatmgrid mismatch."""
        pytest.importorskip("netCDF4")
        import netCDF4
        import numpy as np
        f = tmp_path / "lon.nc"
        with netCDF4.Dataset(f, "w") as d:
            d.createDimension("gridcell", 1); d.createDimension("landunit", 4)
            d.createDimension("column", 16); d.createDimension("pft", 32)
            v = d.createVariable("cols1d_ityplun", "i4", ("column",))
            v[:] = np.array(fs.EXPECTED_ITYPLUN)
            g = d.createVariable("grid1d_lon", "f8", ("gridcell",)); g[:] = 239.16
        assert any("0-360" in p for p in fs.validate(str(f)))


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


# ── manager step 0b ─────────────────────────────────────────────────────────
class TestManagerWarmstartStep:
    def _mgr(self, tmp_path):
        from core.elm_exp_manager import ELMExpManager
        return ELMExpManager(base_output_dir=str(tmp_path))

    def test_a_missing_request_still_warm_starts(self, tmp_path):
        """Warm start is no longer opt-in.

        It used to return None when the config said nothing, so a caller that
        simply forgot the flag got a silently different experiment. Now the
        only two outcomes are a manifest or an exception — never a quiet None,
        which is the one result that let a cold ensemble pass for a warm one.
        """
        try:
            got = self._mgr(tmp_path)._warmstart(COLS, {})
        except Exception:
            return                      # attempted and failed loudly: correct
        assert got, "returned a falsy manifest instead of warm starting or raising"

    def test_an_unreadable_conus_source_raises(self, tmp_path):
        """It used to cold start and print a warning.

        That reads as forgiving and is not. The start type is decided PER
        COLUMN downstream, so this path produced ensembles where some columns
        were warm on the donor's CONUS soil and others cold on a different soil
        dataset, inside one run, signalled only by a "(cold)" in a print. Every
        cross-column comparison then spanned two experiments.
        """
        with pytest.raises(RuntimeError, match="warm start failed"):
            self._mgr(tmp_path)._warmstart(
                COLS, {"warm_start": {"conus_restart": str(tmp_path / "missing.txt")}})

    def test_a_band_naming_a_nonexistent_restart_raises(self, tmp_path):
        """Manifest parses, but the restart it names does not exist."""
        man = tmp_path / "MANIFEST.txt"
        man.write_text(MANIFEST)
        with pytest.raises(RuntimeError):
            self._mgr(tmp_path)._warmstart(
                COLS, {"warm_start": {"conus_restart": str(man)}})

    def test_the_failure_names_what_went_wrong(self, tmp_path):
        """A hard stop is only an improvement if it says why it stopped."""
        with pytest.raises(RuntimeError) as e:
            self._mgr(tmp_path)._warmstart(
                COLS, {"warm_start": {"conus_restart": str(tmp_path / "nope.txt")}})
        assert "warm start" in str(e.value).lower()


# ─────────────────────────────────────────────────────────────────────────────
# Index-vector reuse — why a warm start was slow
# ─────────────────────────────────────────────────────────────────────────────
class _CountingVar:
    def __init__(self, arr, owner): self.arr, self.owner = arr, owner
    def __getitem__(self, k):
        self.owner.reads += 1
        return self.arr[k]


class _FakeRestart:
    """Stands in for a 70 GB CONUS band; counts whole-vector reads."""
    def __init__(self, path, n=6):
        import numpy as np
        self.path, self.reads = path, 0
        self.variables = {
            "grid1d_lat": _CountingVar(np.arange(n, dtype=float), self),
            "grid1d_lon": _CountingVar(np.full(n, -121.0), self),
            "cols1d_gridcell_index": _CountingVar(
                np.repeat(np.arange(1, n + 1), 2), self),
        }
    def filepath(self): return self.path


@pytest.fixture(autouse=True)
def _clear_idx_cache():
    fs._IDX_CACHE.update(path=None, vars={})
    yield
    fs._IDX_CACHE.update(path=None, vars={})


class TestIndexVectorCache:
    """plan_subset needs six whole index vectors to locate ONE gridcell. They
    describe the band, not the column, so re-reading them per column was the
    dominant cost of a warm start: measured on a real run, 11 columns across two
    bands took 15 min 11 s — about 83 s per column — to extract 0.2 MB each."""

    def test_one_read_per_vector_per_band(self):
        d = _FakeRestart("/band/lat11.nc")
        for _ in range(11):
            fs._idx(d, "grid1d_lat")
            fs._idx(d, "grid1d_lon")
            fs._idx(d, "cols1d_gridcell_index")
        assert d.reads == 3, "vectors must be read once per band, not per column"

    def test_the_cached_values_are_correct_not_just_few(self):
        """A cache that returns the wrong array is worse than no cache."""
        import numpy as np
        d = _FakeRestart("/band/lat11.nc")
        first = fs._idx(d, "grid1d_lat").copy()
        assert np.array_equal(fs._idx(d, "grid1d_lat"), first)
        assert np.array_equal(first, np.arange(6, dtype=float))

    def test_switching_band_evicts_the_previous_one(self):
        """Only one band is held: these arrays are hundreds of MB and the
        loop visits bands in order, so a single slot is enough."""
        a = _FakeRestart("/band/lat11.nc")
        b = _FakeRestart("/band/lat12.nc")
        fs._idx(a, "grid1d_lat")
        fs._idx(b, "grid1d_lat")
        assert fs._IDX_CACHE["path"] == "/band/lat12.nc"
        assert len(fs._IDX_CACHE["vars"]) == 1

    def test_a_second_band_does_not_serve_stale_data(self):
        import numpy as np
        a = _FakeRestart("/band/lat11.nc", n=6)
        b = _FakeRestart("/band/lat12.nc", n=9)
        assert len(fs._idx(a, "grid1d_lat")) == 6
        assert len(fs._idx(b, "grid1d_lat")) == 9


class TestBandLookupWithoutOpening:
    """build_finidats needs the FILE PATH, not ConusSource's index vectors —
    it does its own finer-grained lookup. Constructing ConusSource anyway read
    four whole vectors per band for nothing."""

    def test_declared_ranges_answer_without_opening_anything(self):
        bs = mw.ConusBandSet([("lat11", 46.0, 47.0, "/x/lat11.nc"),
                              ("lat12", 47.0, 48.0, "/x/lat12.nc")])
        assert bs.path_for_lat(46.5) == ("lat11", "/x/lat11.nc")
        assert bs.path_for_lat(47.5) == ("lat12", "/x/lat12.nc")
        assert bs._open == {}, "no restart may be opened to answer this"

    def test_a_latitude_outside_every_band_opens_nothing(self):
        bs = mw.ConusBandSet([("lat11", 46.0, 47.0, "/x/lat11.nc")])
        band, path = bs.path_for_lat(20.0)
        assert path is None
        assert bs._open == {}


class TestParallelSubset:
    """write_subset copies ~216 variables, each a separate seek into a 70 GB
    file, so a column is bound by I/O LATENCY and barely uses CPU. Overlapping
    columns hides the seeks: measured 53.4 s/column serial against 5.4 s/column
    with three workers, and 14 columns end-to-end from 941 s to 27 s with
    byte-identical donor gridcells."""

    def test_worker_reports_failure_instead_of_killing_the_pool(self, tmp_path):
        """A bad column must cold-start alone, not take the ensemble down —
        an exception crossing a process boundary would abort every column."""
        cid, info, reason = fs._subset_one(
            ("col_99", 46.7, -121.0, "lat11",
             str(tmp_path / "does_not_exist.nc"), str(tmp_path)))
        assert cid == "col_99"
        assert info is None
        assert reason and isinstance(reason, str)

    def test_worker_count_comes_from_the_environment(self):
        """IDEAS_WARMSTART_WORKERS lets a busy login node dial this down
        without editing code."""
        src = (ROOT / "tools" / "make_finidat_subset.py").read_text()
        assert 'os.environ.get("IDEAS_WARMSTART_WORKERS"' in src
        assert fs.DEFAULT_WORKERS >= 1

    def test_build_finidats_accepts_a_workers_argument(self):
        import inspect
        assert "workers" in inspect.signature(fs.build_finidats).parameters

    def test_a_single_column_does_not_spawn_a_pool(self, tmp_path, monkeypatch):
        """One column has nothing to overlap; paying process startup for it
        would make the common single-column case slower."""
        called = {"pool": False}

        class _Boom:
            def __init__(self, *a, **k): called["pool"] = True
            def __enter__(self): raise AssertionError("pool must not be used")
            def __exit__(self, *a): return False

        import concurrent.futures as cf
        monkeypatch.setattr(cf, "ProcessPoolExecutor", _Boom)
        bands = mw.ConusBandSet([("lat11", 46.0, 47.0, str(tmp_path / "nope.nc"))])
        # the column resolves to a band but the file is absent -> skipped, and
        # crucially the serial path is taken without touching the pool
        out = fs.build_finidats([{"id": "col_01", "lat": 46.5, "lon": -121.0}],
                                tmp_path, bands, quiet=True)
        assert out == {}
        assert called["pool"] is False
