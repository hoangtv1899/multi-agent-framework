"""The read-ahead of a one-pass read: it may change the time a pass takes, and
nothing else.

    pytest tests/test_gridded_read_ahead.py -v

While Stream.read_block_pass walks a year, a thread reads the year's records
straight through so that the pass's small reads find them in the page cache
(gridded, READ-AHEAD, 2026-11-04). It is a hint. So every check here holds the
pass with the thread to the pass without it, bit for bit, on small files in
ELM's own layout (elm_gridded_fixtures): a daily year across two files, a
3-hourly and a monthly stream, a year with a record missing, a layer and a
window. Then the thread itself: it reads the bytes the year's records occupy
and no others, it is gone when the pass returns and when it raises, it stops
when the pass ends rather than reading on, and when it fails, cannot start or
has nothing to read, the pass goes on as if it had never been asked for.
"""
import shutil
import threading
import time

import numpy as np
import pytest

import elm_gridded_fixtures as fx
import gridded as G

YEARS = [2015, 2016]
DAILY = ["DOYCODE", "CELLID", "GPP", "QVEGT", "QSOIL", "H2OSNO", "FAREA_BURNED", "TSA"]
NAME = "gridded-read-ahead"


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    fx.write_daily(d, YEARS)                               # h1: 2015, 2016, 2017
    fx.write_daily(d, YEARS, stream="h2", steps_per_day=8)
    fx.write_monthly(d, YEARS, variables=("GPP", "TSA", "CELLID", "FAREA_BURNED"))
    return d


def _threads():
    return [t for t in threading.enumerate() if t.name == NAME]


def _same_bits(a, b) -> bool:
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _pass(stream, names, year, **kw):
    got = []
    out = stream.read_block_pass(names, year, reduce=got.append, **kw)
    return got, out


def _assert_same_blocks(a, b):
    """Two passes handed over the same blocks: labels, units, flags and every
    value to the last bit."""
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert (x.year, x.first, x.keys, x.labels, x.missing) == \
            (y.year, y.first, y.keys, y.labels, y.missing)
        assert x.units == y.units and x.per_second == y.per_second
        assert list(x.data) == list(y.data)
        for v in x.data:
            assert _same_bits(x.data[v], y.data[v]), v


def _assert_on_is_off(stream, names, year, **kw):
    """The pass with the thread, the pass without it, and read(): one answer."""
    on, out_on = _pass(stream, names, year, read_ahead=True, **kw)
    off, out_off = _pass(stream, names, year, read_ahead=False, **kw)
    assert out_on["read_ahead"]["on"] is True and out_on["read_ahead"]["error"] is None
    assert out_off["read_ahead"] == {"on": False, "bytes_planned": 0, "bytes_read": 0,
                                     "error": None}
    assert {k: v for k, v in out_on.items() if k != "read_ahead"} == \
        {k: v for k, v in out_off.items() if k != "read_ahead"}
    _assert_same_blocks(on, off)
    for v in names:
        want = stream.read(v, year, **{k: w for k, w in kw.items()
                                       if k in ("window", "layer", "require_complete")})
        got = np.concatenate([b.data[v] for b in on], axis=0)
        assert _same_bits(got, want.data), v
    return out_on


# ── the values do not change ─────────────────────────────────────────────────
class TestTheValuesDoNotChange:

    @pytest.mark.parametrize("block_records", [1, 7, 30, 400])
    def test_a_daily_year_across_two_files(self, run, block_records):
        """31 December is the first record of the next year's file; the run's
        start step opens the first file and is left out."""
        s = G.open_stream(run, "h1")
        for y in YEARS:
            out = _assert_on_is_off(s, DAILY, y, block_records=block_records)
            assert out["records"] == 365
        assert len({r.file for r in s.year_records(2016)[0]}) == 2

    @pytest.mark.parametrize("block_records", [1, 30, 3000])
    def test_a_three_hourly_stream(self, run, block_records):
        _assert_on_is_off(G.open_stream(run, "h2"), ["STEPCODE", "CELLID", "TSA"], 2016,
                          block_records=block_records)

    @pytest.mark.parametrize("block_records", [1, 5, 12])
    def test_a_monthly_stream_forced_on(self, run, block_records):
        s = G.open_stream(run, "h0")
        for y in YEARS:
            _assert_on_is_off(s, ["GPP", "TSA", "CELLID", "FAREA_BURNED", "DOYCODE"], y,
                              block_records=block_records)

    def test_a_year_with_a_record_missing(self, run, tmp_path):
        """Without the 2017 file, 2016 lacks 31 December: read without it, the
        absent label listed on the first block, with the thread or without."""
        d = tmp_path / "short"
        shutil.copytree(run, d)
        (d / "synth.elm.h1.2017-01-01-00000.nc").unlink()
        s = G.open_stream(d, "h1")
        for n in (1, 30, 400):
            out = _assert_on_is_off(s, ["GPP", "TSA"], 2016, block_records=n,
                                    require_complete=False)
            assert out["missing"] == ["2016-12-31"] and out["records"] == 364
        with pytest.raises(G.GriddedReadError, match="1 of 365"):
            s.read_block_pass(["GPP"], 2016, reduce=print, read_ahead=True)
        assert _threads() == []

    def test_a_layer_and_a_window(self, run):
        s = G.open_stream(run, "h1")
        _assert_on_is_off(s, ["SOILLIQ"], 2016, block_records=7, layer=2)
        _assert_on_is_off(s, ["CELLID", "GPP", "SOILLIQ"], 2016, block_records=30,
                          window=(1, 3, 2, 5))

    def test_an_unstructured_grid(self, tmp_path):
        fx.write_daily_lnd(tmp_path, YEARS)
        _assert_on_is_off(G.open_stream(tmp_path, "h1"), ["GPP", "TSA", "SOILLIQ"], 2016,
                          block_records=30)


class TestTheDefault:
    """On for a daily or sub-daily stream, off for a monthly one (measured on
    the Naches control: Stream.read_block_pass)."""

    def test_by_cadence(self, run):
        for stream, names, on in (("h1", ["GPP"], True), ("h2", ["TSA"], True),
                                  ("h0", ["GPP"], False)):
            s = G.open_stream(run, stream)
            assert s.read_ahead_default() is on, stream
            _, out = _pass(s, names, 2016)
            assert out["read_ahead"]["on"] is on, stream

    def test_forced_either_way(self, run):
        _, out = _pass(G.open_stream(run, "h1"), ["GPP"], 2016, read_ahead=False)
        assert out["read_ahead"]["on"] is False
        _, out = _pass(G.open_stream(run, "h0"), ["GPP"], 2016, read_ahead=True)
        assert out["read_ahead"]["on"] is True


# ── what the thread reads ────────────────────────────────────────────────────
def _raw_offsets(path, values_be: bytes):
    raw = path.read_bytes()
    at = raw.find(values_be)
    assert at >= 0 and raw.find(values_be, at + 1) == -1, "not found once"
    return at


class TestTheSpans:

    def test_each_record_is_inside_its_span(self, run):
        """DOYCODE is different at every record, so its bytes can be found in
        the file: each record's lie inside the span record_span gives it, and
        the spans of consecutive records tile the record section."""
        import netCDF4
        p = run / "synth.elm.h1.2016-01-01-00000.nc"
        with netCDF4.Dataset(p) as ds:
            ds.set_auto_maskandscale(False)
            n = len(ds.dimensions["time"])
            spans = [G.record_span(ds, p, r, r) for r in range(n)]
            for r in (0, 1, 100, n - 1):
                be = np.asarray(ds["DOYCODE"][r]).astype(">f4").tobytes()
                at = _raw_offsets(p, be)
                assert spans[r][0] <= at and at + len(be) <= spans[r][1], r
            assert G.record_span(ds, p, 0, n - 1) == (spans[0][0], spans[-1][1])
        assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
        assert spans[-1][1] == p.stat().st_size

    def test_padding_and_the_one_record_variable(self, tmp_path):
        """Each record variable's share of a record is padded to 4 bytes; with
        one record variable there is no padding at all (the classic format,
        held here byte by byte)."""
        import netCDF4
        many = tmp_path / "many.nc"
        with netCDF4.Dataset(many, "w", format="NETCDF3_64BIT_OFFSET") as d:
            d.createDimension("time", None)
            d.createDimension("three", 3)
            d.createVariable("big", "f4", ("three", "three"))[:] = 7.0
            c = d.createVariable("c", "S1", ("time", "three"))
            x = d.createVariable("x", "i2", ("time",))
            for r in range(5):
                c[r] = np.array([b"r", b"%d" % r, b"!"], dtype="S1")
                x[r] = 1000 + r
        raw = many.read_bytes()
        with netCDF4.Dataset(many) as ds:
            s = [G.record_span(ds, many, r, r) for r in range(5)]
        for r, (a, b) in enumerate(s):
            assert b - a == 8                                # 3 bytes -> 4, 2 bytes -> 4
            assert raw[a:a + 3] == b"r%d!" % r
            assert raw[a + 4:a + 6] == np.array([1000 + r], dtype=">i2").tobytes()
        assert s[-1][1] == len(raw)
        for fmt, vals in (("NETCDF3_CLASSIC", np.arange(4000, 4007, dtype="i2")),
                          ("NETCDF3_64BIT_OFFSET", np.arange(7, dtype="f4") + 0.5)):
            one = tmp_path / f"one_{vals.dtype}.nc"
            with netCDF4.Dataset(one, "w", format=fmt) as d:
                d.createDimension("time", None)
                d.createVariable("y", vals.dtype, ("time",))[:] = vals
            raw = one.read_bytes()
            with netCDF4.Dataset(one) as ds:
                s = [G.record_span(ds, one, r, r) for r in range(7)]
            for r, (a, b) in enumerate(s):
                assert b - a == vals.dtype.itemsize
                assert raw[a:b] == vals[r:r + 1].astype(vals.dtype.newbyteorder(">")).tobytes()
            assert s[-1][1] == len(raw)

    def test_no_span_where_there_is_no_classic_layout(self, tmp_path, run):
        import netCDF4
        p = tmp_path / "nc4.nc"
        with netCDF4.Dataset(p, "w", format="NETCDF4") as d:
            d.createDimension("time", None)
            d.createVariable("t", "f8", ("time",))[:] = np.arange(3.0)
        with netCDF4.Dataset(p) as ds:
            assert G.record_span(ds, p, 0, 2) is None
        q = run / "synth.elm.h1.2016-01-01-00000.nc"
        with netCDF4.Dataset(q) as ds:
            n = len(ds.dimensions["time"])
            assert G.record_span(ds, q, 0, n) is None           # past the last record
            assert G.record_span(ds, q, 3, 2) is None

    def test_the_pass_reads_the_years_records_in_order(self, run, monkeypatch):
        """2016: records 1-364 of its own file, then record 0 of the 2017 file
        (31 December), and nothing else."""
        import netCDF4
        seen = []
        real = G._ReadAhead.__init__

        def spying(self, spans, chunk=None):
            seen.append(list(spans))
            real(self, spans, chunk)
        monkeypatch.setattr(G._ReadAhead, "__init__", spying)
        s = G.open_stream(run, "h1")
        _, out = _pass(s, ["GPP"], 2016, read_ahead=True)
        f16, f17 = run / "synth.elm.h1.2016-01-01-00000.nc", run / "synth.elm.h1.2017-01-01-00000.nc"
        with netCDF4.Dataset(f16) as a, netCDF4.Dataset(f17) as b:
            want = [(str(f16),) + G.record_span(a, f16, 1, 364),
                    (str(f17),) + G.record_span(b, f17, 0, 0)]
        assert seen == [want]
        assert out["read_ahead"]["bytes_planned"] == sum(e - s for _, s, e in want)
        assert out["read_ahead"]["bytes_read"] == out["read_ahead"]["bytes_planned"]


# ── the thread ───────────────────────────────────────────────────────────────
@pytest.fixture
def slow(monkeypatch):
    """A read-ahead that reads 64 bytes at a time, 20 ms per read: still
    running whenever the pass looks."""
    calls = []

    def read(f, view):
        calls.append(time.monotonic())
        time.sleep(0.02)
        return f.readinto(view[:64])
    monkeypatch.setattr(G._ReadAhead, "_read", staticmethod(read))
    return calls


class TestTheThread:

    def test_it_is_gone_after_the_pass_returns(self, run, slow):
        s = G.open_stream(run, "h1")
        alive = []
        out = s.read_block_pass(["GPP"], 2016, block_records=30, read_ahead=True,
                                reduce=lambda b: alive.append(len(_threads())))
        assert alive == [1] * 13, "the thread did not run beside every block"
        assert _threads() == []
        assert out["read_ahead"]["on"] and out["read_ahead"]["error"] is None

    def test_it_is_gone_after_the_pass_raises(self, run, slow):
        s = G.open_stream(run, "h1")

        def boom(b):
            assert len(_threads()) == 1
            raise RuntimeError("the callback failed")
        with pytest.raises(RuntimeError, match="the callback failed"):
            s.read_block_pass(["GPP", "TSA"], 2016, reduce=boom, read_ahead=True)
        assert _threads() == []

    def test_it_stops_when_the_pass_ends(self, run, slow):
        """Stopped part way, not left to read on: no read starts once the pass
        has returned."""
        s = G.open_stream(run, "h1")
        _, out = _pass(s, ["GPP"], 2016, block_records=400, read_ahead=True)
        ra = out["read_ahead"]
        assert ra["on"] and 0 < ra["bytes_read"] < ra["bytes_planned"]
        n, t_end = len(slow), time.monotonic()
        time.sleep(0.1)
        assert len(slow) == n and all(t < t_end for t in slow)

    def test_a_failing_read_does_not_fail_the_pass(self, run, monkeypatch):
        def bad(f, view):
            raise OSError(5, "Input/output error")
        monkeypatch.setattr(G._ReadAhead, "_read", staticmethod(bad))
        s = G.open_stream(run, "h1")
        on, out = _pass(s, DAILY, 2016, read_ahead=True)
        off, _ = _pass(s, DAILY, 2016, read_ahead=False)
        _assert_same_blocks(on, off)
        assert out["read_ahead"]["on"] is True and out["read_ahead"]["bytes_read"] == 0
        assert out["read_ahead"]["error"].startswith("OSError")
        assert _threads() == []

    def test_a_file_gone_from_under_it(self, run, monkeypatch):
        """The thread opens its own handle; a path it cannot open is its error,
        not the pass's."""
        real = G._ReadAhead.__init__

        def elsewhere(self, spans, chunk=None):
            real(self, [(p + ".gone", a, b) for p, a, b in spans], chunk)
        monkeypatch.setattr(G._ReadAhead, "__init__", elsewhere)
        s = G.open_stream(run, "h1")
        on, out = _pass(s, ["GPP", "TSA"], 2016, read_ahead=True)
        off, _ = _pass(s, ["GPP", "TSA"], 2016, read_ahead=False)
        _assert_same_blocks(on, off)
        assert "FileNotFoundError" in out["read_ahead"]["error"]

    def test_a_thread_that_cannot_start(self, run, monkeypatch):
        def no(self):
            raise RuntimeError("can't start new thread")
        monkeypatch.setattr(G._ReadAhead, "start", no)
        s = G.open_stream(run, "h1")
        on, out = _pass(s, ["GPP"], 2016, read_ahead=True)
        off, _ = _pass(s, ["GPP"], 2016, read_ahead=False)
        _assert_same_blocks(on, off)
        assert out["read_ahead"]["on"] is False
        assert "can't start new thread" in out["read_ahead"]["error"]
        assert _threads() == []

    def test_spans_that_cannot_be_made(self, run, monkeypatch):
        s = G.open_stream(run, "h1")
        off, _ = _pass(s, ["GPP"], 2016, read_ahead=False)

        def broken(*a, **k):
            raise ValueError("a header this reader does not know")
        monkeypatch.setattr(G, "record_span", broken)
        on, out = _pass(s, ["GPP"], 2016, read_ahead=True)
        _assert_same_blocks(on, off)
        assert out["read_ahead"]["on"] is False and "ValueError" in out["read_ahead"]["error"]
        monkeypatch.setattr(G, "record_span", lambda *a, **k: None)
        on, out = _pass(s, ["GPP"], 2016, read_ahead=True)
        _assert_same_blocks(on, off)
        assert out["read_ahead"]["on"] is False and "not classic" in out["read_ahead"]["error"]

    def test_no_thread_starts_for_a_refusal(self, run, tmp_path, monkeypatch):
        """The refusals come before the thread: a refused pass starts none,
        down to a field that a later file lacks."""
        started = []
        real = G._ReadAhead.start

        def counted(self):
            started.append(self.spans)
            real(self)
        monkeypatch.setattr(G._ReadAhead, "start", counted)
        s = G.open_stream(run, "h1")
        for names, kw in ((["GPP", "NOPE"], {}), (["SOILLIQ"], {"layer": 9}),
                          (["GPP"], {"block_records": 0}), (["GPP"], {"reduce": None})):
            with pytest.raises(G.GriddedReadError):
                s.read_block_pass(names, 2016, **{"reduce": print, "read_ahead": True, **kw})
        fx.write_daily(tmp_path, [2015], variables=("GPP", "TSA"))
        (tmp_path / "synth.elm.h1.2016-01-01-00000.nc").unlink()
        fx._write(tmp_path / "synth.elm.h1.2016-01-01-00000.nc", fx.grid(), 2015,
                  [(364.0, 365.0)], [(2015, 365, 0)], ["GPP"])
        with pytest.raises(G.GriddedReadError, match="missing from"):
            G.open_stream(tmp_path, "h1").read_block_pass(["GPP", "TSA"], 2015, reduce=print,
                                                          read_ahead=True)
        assert started == [] and _threads() == []
        s.read_block_pass(["GPP"], 2016, reduce=lambda b: None, read_ahead=True)
        assert len(started) == 1                         # and an accepted one does

    def test_the_thread_never_calls_netcdf(self, run, monkeypatch):
        """netCDF4 is not safe to call from two threads; the read-ahead reads
        through a plain file handle only."""
        import netCDF4
        others = []
        real = netCDF4.Dataset

        def watched(*a, **k):
            if threading.current_thread() is not threading.main_thread():
                others.append(threading.current_thread().name)
            return real(*a, **k)
        monkeypatch.setattr(netCDF4, "Dataset", watched)
        _, out = _pass(G.open_stream(run, "h1"), DAILY, 2016, read_ahead=True)
        assert out["read_ahead"]["on"] and others == []
