"""Each history file's index on disk: a second open reads no file, and gives
the same stream.

    pytest tests/test_gridded_index_cache.py -v

gridded keeps what it reads of a history file to open a stream (the grid, the
time bounds, the first nstep, the calendar and time units, the variables) in a
cache outside the run, so a new process opens a stream without reading a
record (gridded, THE INDEX ON DISK, 2026-11-04). The checks here, on small
files in ELM's own layout (elm_gridded_fixtures):

  * a stream from the cache is the stream from the files, attribute by
    attribute and array by array, for every layout the reader handles;
  * a second open opens no file at all;
  * a file grown, touched, or rewritten in place with its mtime put back is
    read again;
  * an entry that is corrupt, cut short, of another file, of another format,
    or holding the wrong shapes is ignored and made again, never trusted;
  * off means off, the default location is ~/.cache/elm-mcp/stream_index/,
    nothing is written outside the cache directory, a cache directory inside
    the run is not used, and a failed write leaves neither an entry nor a
    temporary file;
  * the entry format is raised whenever the code that makes an index
    changes, so no entry made by older code is trusted;
  * the numbers check imports neither gridded nor anything that reads the
    cache.

tests/conftest.py switches the cache off for every other test; each test here
points it at a directory of its own.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import elm_gridded_fixtures as fx
import gridded as G

ENV = G.INDEX_CACHE_ENV
REPO = Path(__file__).resolve().parents[1]
YEARS = [2015, 2016]


@pytest.fixture
def fresh(monkeypatch):
    """A new process's view: nothing indexed in memory."""
    monkeypatch.setattr(G, "_FILE_INDEX", {})
    return G._FILE_INDEX


@pytest.fixture
def run(tmp_path):
    d = tmp_path / "run"
    fx.write_daily(d, YEARS)
    fx.write_daily(d, YEARS, stream="h2", steps_per_day=8)
    fx.write_monthly(d, YEARS, variables=("GPP", "TSA", "CELLID", "FAREA_BURNED"))
    return d


@pytest.fixture
def cache(tmp_path, monkeypatch, fresh):
    d = tmp_path / "cache"
    monkeypatch.setenv(ENV, str(d))
    return d


def _entries(d: Path):
    return sorted(p for p in d.glob("*.npz")) if d.is_dir() else []


class Opens:
    """netCDF4.Dataset, counted (and refused when `refuse`)."""

    def __init__(self, monkeypatch, refuse=False):
        import netCDF4
        self.paths, self.refuse, real = [], refuse, netCDF4.Dataset

        def counting(path, *a, **k):
            self.paths.append(Path(str(path)).name)
            if self.refuse:
                raise AssertionError(f"a file was opened: {path}")
            return real(path, *a, **k)
        monkeypatch.setattr(netCDF4, "Dataset", counting)


def _from_files(monkeypatch, run_dir, stream):
    """The stream as the files make it: cache off, nothing in memory."""
    with monkeypatch.context() as m:
        m.setenv(ENV, "off")
        m.setattr(G, "_FILE_INDEX", {})
        s = G.open_stream(run_dir, stream)
        index = dict(G._FILE_INDEX)
    return s, index


def _same_bits(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def assert_same_stream(a: G.Stream, b: G.Stream) -> None:
    """Everything a Stream holds, and everything it answers, field by field."""
    assert a.name == b.name and a.files == b.files
    assert (a.cadence, a.steps_per_day, a.calendar) == (b.cadence, b.steps_per_day, b.calendar)
    assert len(a.records) == len(b.records)
    for x, y in zip(a.records, b.records):
        assert (x.key, x.file, x.index) == (y.key, y.file, y.index)
        assert (x.start, x.end) == (y.start, y.end)
        assert (type(x.start), str(x.start), str(x.end)) == (type(y.start), str(y.start), str(y.end))
    assert a.dropped == b.dropped and a.extended == b.extended
    assert a._vars == b._vars
    for v, d in b._vars.items():                         # tuples stay tuples
        assert type(d["dims"]) is tuple and type(d["shape"]) is tuple
        assert all(type(n) is int for n in d["shape"])
    assert a.variables == b.variables and a.metadata_variables == b.metadata_variables
    for v in a.variables:
        assert (a.units(v), a.per_second(v), a.levels(v)) == (b.units(v), b.per_second(v),
                                                             b.levels(v))
    for f in ("lat", "lon", "area_km2", "landfrac", "lat_axis", "lon_axis", "weight_km2"):
        assert _same_bits(getattr(a.grid, f), getattr(b.grid, f)), f
    assert a.grid.dims == b.grid.dims and type(b.grid.dims) is tuple
    assert a.years() == b.years() and a.complete_years() == b.complete_years()
    assert a.labels() == b.labels()
    assert a.describe() == b.describe()
    for y in a.years():
        assert a.year_records(y)[1] == b.year_records(y)[1]


def assert_same_index(a: dict, b: dict) -> None:
    """Two _FILE_INDEX contents, file by file and item by item."""
    assert sorted(a) == sorted(b)
    for k in a:
        x, y = a[k], b[k]
        assert set(x) == set(y)
        assert _same_bits(x["tb"], y["tb"]) and not y["tb"].flags.writeable
        for f in ("time_units", "calendar", "first_nstep", "variables", "metadata"):
            assert x[f] == y[f] and type(x[f]) is type(y[f]), f
        for f in ("lat", "lon", "area_km2", "landfrac", "lat_axis", "lon_axis"):
            assert _same_bits(getattr(x["grid"], f), getattr(y["grid"], f)), f
            arr = getattr(y["grid"], f)
            assert arr is None or not arr.flags.writeable, f


# ── layouts: the stream from the cache is the stream from the files ──────────
def _restarted_segment(d):
    """MidA_fdrought's layout: one file per month, and a later segment whose
    January starts one step early, inside December."""
    g = fx.grid()
    fx._write(d / "s.elm.h0.2018-12.nc", g, 2018, [(334.0, 365.0)], [(2018, 349, 0)],
              ["DOYCODE", "GPP"])
    for m in range(1, 13):
        a = sum(fx.MONTH_DAYS[:m - 1]) - (fx.DT if m == 1 else 0.0)
        fx._write(d / f"s.elm.h0.2019-{m:02d}.nc", g, 2019,
                  [(a, float(sum(fx.MONTH_DAYS[:m])))],
                  [(2019, sum(fx.MONTH_DAYS[:m - 1]) + 15, 0)], ["DOYCODE", "GPP"])
    return "h0"


def _fill_bounds_and_no_nstep(d):
    """A record whose bounds were never written (1e36: dropped, with the
    reason), and a file without nstep (first_nstep None)."""
    import netCDF4
    p = d / "n.elm.h1.2015-01-01-00000.nc"
    fx._write(p, fx.grid(), 2015, [(k, k + 1.0) for k in range(6)],
              [(2015, k + 1, 0) for k in range(6)], ["GPP", "TSA"], nstep=False)
    with netCDF4.Dataset(p, "a") as ds:
        ds["time_bounds"][3] = [1e36, 1e36]
    return "h1"


def _unstructured(d):
    fx.write_daily_lnd(d, YEARS)
    return "h1"


def _short_year(d):
    fx.write_daily(d, YEARS, last_file=False)
    return "h1"


LAYOUTS = {
    "daily": lambda d: fx.write_daily(d, YEARS) and "h1",
    "three-hourly": lambda d: fx.write_daily(d, YEARS, stream="h2", steps_per_day=8) and "h2",
    "monthly": lambda d: fx.write_monthly(d, YEARS) and "h0",
    "restarted segment": _restarted_segment,
    "fill bounds, no nstep": _fill_bounds_and_no_nstep,
    "unstructured": _unstructured,
    "short year": _short_year,
}


class TestTheStreamIsTheFiles:

    @pytest.mark.parametrize("layout", sorted(LAYOUTS))
    def test_built_from_the_cache(self, tmp_path, cache, monkeypatch, layout):
        d = tmp_path / "run"
        d.mkdir()
        stream = LAYOUTS[layout](d)
        want, want_index = _from_files(monkeypatch, d, stream)
        # first open with the cache on: read from the files, and written down
        first = G.open_stream(d, stream)
        assert_same_stream(want, first)
        assert len(_entries(cache)) == len(first.files)
        # a new process: from the cache, and no file opened
        G._FILE_INDEX.clear()
        with monkeypatch.context() as m:
            Opens(m, refuse=True)
            got = G.open_stream(d, stream)
        assert_same_stream(want, got)
        assert_same_index(want_index, G._FILE_INDEX)
        if layout == "fill bounds, no nstep":
            assert [x["record"] for x in got.dropped] == [3]
            assert all(i["first_nstep"] is None for i in G._FILE_INDEX.values())
        if layout == "restarted segment":
            assert got.extended and "overlaps_previous_days" in got.extended[0]

    def test_reading_after_an_open_from_the_cache(self, run, cache, monkeypatch):
        """The values read through a stream opened from the cache are the
        values read through one opened from the files."""
        want, _ = _from_files(monkeypatch, run, "h1")
        G.open_stream(run, "h1")
        G._FILE_INDEX.clear()
        got = G.open_stream(run, "h1")
        for v in ("GPP", "SOILLIQ", "DOYCODE"):
            assert _same_bits(want.read(v, 2016).data, got.read(v, 2016).data)


# ── a changed file is read again ─────────────────────────────────────────────
class TestAChangedFileIsReadAgain:

    def _warm(self, run):
        G.open_stream(run, "h1")
        G._FILE_INDEX.clear()

    def _reopen(self, run, monkeypatch):
        with monkeypatch.context() as m:
            opens = Opens(m)
            s = G.open_stream(run, "h1")
        G._FILE_INDEX.clear()
        return s, opens.paths

    def test_a_file_grown_by_a_record(self, run, cache, monkeypatch):
        import netCDF4
        self._warm(run)
        last = run / "synth.elm.h1.2017-01-01-00000.nc"
        size = last.stat().st_size
        with netCDF4.Dataset(last, "a") as ds:
            n = len(ds.dimensions["time"])
            ds["time_bounds"][n] = [730.0, 731.0]
            ds["time"][n] = 731.0
        assert last.stat().st_size > size
        got, opened = self._reopen(run, monkeypatch)
        assert opened == [last.name]
        want, _ = _from_files(monkeypatch, run, "h1")
        assert_same_stream(want, got)
        assert got.labels()[-1] == "2017-01-01"

    def test_a_touched_file(self, run, cache, monkeypatch):
        self._warm(run)
        f = run / "synth.elm.h1.2016-01-01-00000.nc"
        st = f.stat()
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 7_000_000_000))
        _, opened = self._reopen(run, monkeypatch)
        assert opened == [f.name]
        _, opened = self._reopen(run, monkeypatch)            # and kept again
        assert opened == []

    def test_rewritten_in_place_with_the_same_size_and_mtime(self, run, cache, monkeypatch):
        """Lustre keeps mtime in whole seconds, and every full-year file of a
        stream has one size: a rewrite with the mtime put back is still caught,
        by its ctime."""
        import netCDF4
        self._warm(run)
        f = run / "synth.elm.h1.2016-01-01-00000.nc"
        st = f.stat()
        time.sleep(0.05)                     # let the clock move past the last write
        with netCDF4.Dataset(f, "a") as ds:
            tb = ds["time_bounds"]
            tb[5] = [1e36, 1e36]             # a record's bounds unwritten: dropped
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
        st2 = f.stat()
        assert (st2.st_size, st2.st_mtime_ns, st2.st_ino) == (st.st_size, st.st_mtime_ns, st.st_ino)
        assert st2.st_ctime_ns != st.st_ctime_ns
        got, opened = self._reopen(run, monkeypatch)
        assert opened == [f.name]
        want, _ = _from_files(monkeypatch, run, "h1")
        assert_same_stream(want, got)
        assert [d["record"] for d in got.dropped if d["file"] == f.name] == [5]

    def test_a_file_that_changes_while_it_is_read_is_not_kept(self, run, cache, monkeypatch):
        real = G._index_from_file

        def changing(path):
            info = real(path)
            st = path.stat()
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
            return info
        monkeypatch.setattr(G, "_index_from_file", changing)
        G.open_stream(run, "h1")
        assert _entries(cache) == []


# ── an entry that cannot be trusted is made again ────────────────────────────
def _entry_of(cache, path):
    return G._index_entry(str(Path(path).resolve()))


def _rewrite_entry(entry, change):
    """Load an entry's arrays and meta, change them, write the entry back."""
    with np.load(entry, allow_pickle=False) as z:
        arrays = {n: z[n] for n in z.files}
    meta = json.loads(arrays["meta"].tobytes().decode())
    arrays, meta = change(arrays, meta)
    if meta is not None:
        arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    with open(entry, "wb") as f:
        np.savez_compressed(f, **arrays)


def _swap(name, value):
    def change(arrays, meta):
        arrays[name] = value(arrays[name])
        return arrays, meta
    return change


def _meta(field, value):
    def change(arrays, meta):
        meta[field] = value(meta[field]) if callable(value) else value
        return arrays, meta
    return change


SPOILED = {
    "garbage": lambda e, other: e.write_bytes(os.urandom(4096)),
    "cut short": lambda e, other: e.write_bytes(e.read_bytes()[: e.stat().st_size // 2]),
    "empty": lambda e, other: e.write_bytes(b""),
    "another file's entry": lambda e, other: shutil.copyfile(other, e),
    "another format": lambda e, other: _rewrite_entry(e, _meta("format", 999)),
    "a key one second older": lambda e, other: _rewrite_entry(
        e, _meta("key", lambda k: k[:3] + [k[3] - 1_000_000_000] + k[4:])),
    "a key one byte bigger": lambda e, other: _rewrite_entry(
        e, _meta("key", lambda k: k[:2] + [k[2] + 1] + k[3:])),
    "bounds with three columns": lambda e, other: _rewrite_entry(
        e, _swap("tb", lambda a: np.zeros((a.shape[0], 3)))),
    "bounds as float32": lambda e, other: _rewrite_entry(
        e, _swap("tb", lambda a: a.astype(np.float32))),
    "a grid array of another shape": lambda e, other: _rewrite_entry(
        e, _swap("area_km2", lambda a: a[:-1])),
    "an array too many": lambda e, other: _rewrite_entry(
        e, lambda arrays, meta: ({**arrays, "extra": np.zeros(3)}, meta)),
    "an array missing": lambda e, other: _rewrite_entry(
        e, lambda arrays, meta: ({k: v for k, v in arrays.items() if k != "lon_axis"}, meta)),
    "a shape that is not whole numbers": lambda e, other: _rewrite_entry(
        e, _meta("variables", lambda vs: {**vs, "GPP": {**vs["GPP"], "shape": [4.0, 5]}})),
    "a first nstep that is text": lambda e, other: _rewrite_entry(e, _meta("first_nstep", "0")),
    "meta that is not JSON": lambda e, other: _rewrite_entry(
        e, lambda arrays, meta: ({**arrays, "meta": np.frombuffer(b"{not json", np.uint8)}, None)),
}


class TestAnEntryThatCannotBeTrusted:

    @pytest.mark.parametrize("how", sorted(SPOILED))
    def test_is_made_again(self, run, cache, monkeypatch, how):
        s = G.open_stream(run, "h1")
        f, other = s.files[1], s.files[0]
        e = _entry_of(cache, f)
        SPOILED[how](e, _entry_of(cache, other))
        G._FILE_INDEX.clear()
        with monkeypatch.context() as m:
            opens = Opens(m)
            got = G.open_stream(run, "h1")
        assert opens.paths == [f.name], "the spoiled entry was trusted"
        want, _ = _from_files(monkeypatch, run, "h1")
        assert_same_stream(want, got)
        # and the entry is whole again: the next open reads no file
        G._FILE_INDEX.clear()
        with monkeypatch.context() as m:
            Opens(m, refuse=True)
            assert_same_stream(want, G.open_stream(run, "h1"))


# ── where it writes, and when it does not ────────────────────────────────────
def _tree(root: Path):
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*"))}


class TestWhereItWrites:

    @pytest.mark.parametrize("off", ["off", "OFF", " off ", "0", "false", "no", "none"])
    def test_off_means_off(self, run, tmp_path, monkeypatch, fresh, off):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv(ENV, off)
        assert G.index_cache_dir() is None
        before = _tree(tmp_path)
        G.open_stream(run, "h1")
        G._FILE_INDEX.clear()
        with monkeypatch.context() as m:
            opens = Opens(m)
            G.open_stream(run, "h1")
        assert len(opens.paths) == 3                     # every file read again
        assert _tree(tmp_path) == before

    @pytest.mark.parametrize("unset", [None, "", "  "])
    def test_the_default_location(self, run, tmp_path, monkeypatch, fresh, unset):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        if unset is None:
            monkeypatch.delenv(ENV, raising=False)
        else:
            monkeypatch.setenv(ENV, unset)
        want = home / ".cache" / "elm-mcp" / "stream_index"
        assert G.index_cache_dir() == want
        s = G.open_stream(run, "h1")
        assert len(_entries(want)) == len(s.files)

    def test_a_cache_directory_given_with_a_tilde(self, run, tmp_path, monkeypatch, fresh):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv(ENV, "~/idx")
        G.open_stream(run, "h0")
        assert len(_entries(home / "idx")) == 2

    def test_nothing_is_written_outside_the_cache_directory(self, tmp_path, run, cache,
                                                            monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        before = _tree(tmp_path)
        for st in ("h0", "h1", "h2"):
            G.open_stream(run, st)
        G._FILE_INDEX.clear()
        for st in ("h0", "h1", "h2"):
            G.open_stream(run, st)
        after = _tree(tmp_path)
        new = sorted(set(after) - set(before))
        changed = sorted(k for k in before if after.get(k) != before[k])
        assert changed == [] and new
        assert all(k == "cache" or k.startswith("cache/") for k in new), new
        assert not any(Path(k).name.startswith(".tmp-") for k in new)
        assert len(_entries(cache)) == 3 + 3 + 2

    @pytest.mark.parametrize("inside", [".", "idx", "deeper/idx"])
    def test_a_cache_directory_in_the_run_is_not_used(self, run, monkeypatch, fresh, inside):
        """Never in a run or case directory: a cache at or under the history's
        own directory is not used at all, neither written nor read."""
        monkeypatch.setenv(ENV, str(run / inside))
        before = _tree(run)
        G.open_stream(run, "h1")
        G._FILE_INDEX.clear()
        with monkeypatch.context() as m:
            opens = Opens(m)
            G.open_stream(run, "h1")
        assert len(opens.paths) == 3
        assert _tree(run) == before

    @pytest.mark.parametrize("relative", ["idx", "off-ish", "./idx", "../idx"])
    def test_a_relative_cache_directory_is_not_used(self, run, tmp_path, monkeypatch, fresh,
                                                    relative):
        """Where a relative path lands depends on where each process runs (a
        case directory among them), so none is used: nothing is written there
        or anywhere else, and every open reads its files."""
        work = tmp_path / "work" / "here"
        work.mkdir(parents=True)
        monkeypatch.chdir(work)
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv(ENV, relative)
        assert G.index_cache_dir() is None
        before = _tree(tmp_path)
        G.open_stream(run, "h1")
        G._FILE_INDEX.clear()
        with monkeypatch.context() as m:
            opens = Opens(m)
            G.open_stream(run, "h1")
        assert len(opens.paths) == 3
        assert _tree(tmp_path) == before

    def test_a_cache_directory_that_cannot_be_made(self, run, tmp_path, monkeypatch, fresh):
        blocker = tmp_path / "a file"
        blocker.write_text("not a directory")
        monkeypatch.setenv(ENV, str(blocker / "idx"))
        want, _ = _from_files(monkeypatch, run, "h1")
        assert_same_stream(want, G.open_stream(run, "h1"))
        assert blocker.read_text() == "not a directory"

    @pytest.mark.parametrize("fails", ["the rename", "the write"])
    def test_a_failed_write_leaves_nothing(self, run, cache, monkeypatch, fails):
        """Written whole or not at all: no entry, and no temporary file left."""
        if fails == "the rename":
            def boom(*a, **k):
                raise OSError("rename refused")
            monkeypatch.setattr(G.os, "replace", boom)
        else:
            def half(f, **arrays):
                f.write(b"PK\x03\x04 half an entry")
                raise OSError("disk full")
            monkeypatch.setattr(np, "savez_compressed", half)
        want, _ = _from_files(monkeypatch, run, "h1")
        assert_same_stream(want, G.open_stream(run, "h1"))
        assert list(cache.iterdir()) == []

    def test_processes_opening_together(self, run, cache):
        """Four processes open the same streams at once; every entry left is
        whole, and a fifth reads them all without opening a file."""
        code = ("import sys; sys.path.insert(0, %r); import gridded as G\n"
                "for s in ('h0', 'h1', 'h2'):\n"
                "    G.open_stream(%r, s)\n") % (str(REPO / "mcp" / "elm-mcp" / "src"), str(run))
        env = {**os.environ, ENV: str(cache)}
        procs = [subprocess.Popen([sys.executable, "-c", code], env=env) for _ in range(4)]
        assert [p.wait(timeout=300) for p in procs] == [0] * 4
        assert len(_entries(cache)) == 8
        assert not [p for p in cache.iterdir() if p.name.startswith(".tmp-")]
        for e in _entries(cache):
            with np.load(e, allow_pickle=False) as z:
                assert "meta" in z.files
        G._FILE_INDEX.clear()
        for st in ("h0", "h1", "h2"):
            with pytest.MonkeyPatch.context() as m:
                Opens(m, refuse=True)
                got = G.open_stream(run, st)
            with pytest.MonkeyPatch.context() as m:
                want, _ = _from_files(m, run, st)
            assert_same_stream(want, got)


# ── the format moves with the code that makes an index ───────────────────────
# An entry is trusted while its format number matches, so a change to how an
# index is made (or encoded) must raise gridded._INDEX_FORMAT, or entries made
# by the old code are read as if made by the new. The source of every function
# that makes, encodes or checks an index is held to the hash recorded with the
# format: change one, and this fails until the format is raised and the hash
# recorded beside it.
INDEX_MAKERS = ("_index_from_file", "_grid_from", "read_clean", "_calendar", "to_pm180",
                "_index_save", "_index_checked", "_index_load")
FORMATS = {1: "55f1adf2c6c573ff"}


def test_the_format_moves_with_the_code_that_makes_an_index():
    import hashlib
    import inspect
    src = "".join(inspect.getsource(getattr(G, n)) for n in INDEX_MAKERS) \
        + repr(G._CALENDAR_ALIASES) + repr(G.FILL_ABS)
    h = hashlib.sha256(src.encode()).hexdigest()[:16]
    assert FORMATS.get(G._INDEX_FORMAT) == h, (
        f"the making of a stream index changed (hash {h}): raise gridded._INDEX_FORMAT "
        f"to {max(FORMATS) + 1} and record {max(FORMATS) + 1}: {h!r} in FORMATS")


# ── the numbers check reads its own history ──────────────────────────────────
def test_the_numbers_check_does_not_read_the_cache():
    """numbers_check makes every number again from the history with its own
    code; it imports neither gridded (where the cache lives) nor the modules
    built on it, and names no cache."""
    code = ("import sys; sys.path[:0] = [%r, %r]\n"
            "import agents.analysis.numbers_check as N\n"
            "print(sorted(m for m in ('gridded', 'regions', 'domain_analysis', 'run_gate')"
            " if m in sys.modules))\n") % (str(REPO / "src"), str(REPO / "mcp" / "elm-mcp" / "src"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=300, cwd=str(REPO))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
    src = (REPO / "src" / "agents" / "analysis" / "numbers_check.py").read_text()
    assert "import gridded" not in src and "from gridded" not in src
    assert ENV not in src and "stream_index" not in src
