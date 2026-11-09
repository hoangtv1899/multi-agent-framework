"""The stream index cache and the read-ahead's record spans, on the Naches control.

    pytest tests/test_gridded_index_cache_naches_oracle.py --runlocaldata -v

OPT-IN, and only where the files are on disk: the naches_suite block of
tests/local_oracle_paths.json names the control's run directory ("run_dir").
Nothing is written beside the run: the cache goes to a temporary directory of
the test's own.

  * h0 and h1 opened from the cache are the streams opened from the files,
    attribute by attribute and array by array, and the open from the cache
    opens no file (gridded, THE INDEX ON DISK, 2026-11-04). h2 is left out:
    its 29,200 time bounds take minutes to read from a cold cache.
  * The read-ahead reads a record where the record is (gridded, READ-AHEAD):
    the last h1 file's last record (31 December of the last year, alone in
    its file) is read raw, and four fields' values, the first and the last
    record variable among them, are found inside the span record_span gives
    it, the last one ending where the span ends. A full year file's records
    have the same size and end where the file ends.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import gridded as G
from test_gridded_index_cache import Opens, assert_same_index, assert_same_stream

pytestmark = pytest.mark.localdata

CONFIG = Path(__file__).with_name("local_oracle_paths.json")


@pytest.fixture(scope="module")
def run_dir():
    if not CONFIG.is_file():
        pytest.skip(f"no {CONFIG.name}")
    c = json.loads(CONFIG.read_text()).get("naches_suite") or {}
    if not (c and Path(c.get("run_dir", "")).is_dir()):
        pytest.skip("no naches_suite block, or its run directory is not on disk")
    return Path(c["run_dir"])


@pytest.mark.parametrize("stream", ["h0", "h1"])
def test_the_stream_from_the_cache_is_the_files(run_dir, stream, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv(G.INDEX_CACHE_ENV, str(cache))
    monkeypatch.setattr(G, "_FILE_INDEX", {})
    want = G.open_stream(run_dir, stream)                    # from the files, written down
    want_index = dict(G._FILE_INDEX)
    assert len(list(cache.glob("*.npz"))) == len(want.files)
    G._FILE_INDEX.clear()
    with monkeypatch.context() as m:
        Opens(m, refuse=True)
        got = G.open_stream(run_dir, stream)                 # from the cache, no file opened
    assert_same_stream(want, got)
    assert_same_index(want_index, G._FILE_INDEX)
    assert got.complete_years() == list(range(2015, 2024))
    assert not any(p.name.startswith(".tmp-") for p in cache.iterdir())


def test_a_record_span_is_where_the_record_is(run_dir):
    import netCDF4
    files = G.discover(run_dir)["h1"]
    last, full = files[-1], files[1]
    with netCDF4.Dataset(last) as ds:
        n = len(ds.dimensions["time"])
        a, b = G.record_span(ds, last, n - 1, n - 1)
        with open(last, "rb") as f:
            f.seek(a)
            raw = f.read(b - a)
        record_vars = [v for v, x in ds.variables.items() if x.dimensions[:1] == ("time",)]
        for v in (record_vars[0], "GPP", "H2OSOI", record_vars[-1]):
            x = ds[v]
            x.set_auto_maskandscale(False)
            want = np.asarray(x[n - 1]).astype(x.dtype.newbyteorder(">")).tobytes()
            at = raw.find(want)
            assert at >= 0, v
            if v == record_vars[0]:
                assert at == 0, v                            # the record opens with it
            if v == record_vars[-1]:
                assert at + len(want) == len(raw), v         # and closes with the last
    assert b == last.stat().st_size
    with netCDF4.Dataset(full) as ds:
        m = len(ds.dimensions["time"])
        a0, b0 = G.record_span(ds, full, 0, 0)
        assert G.record_span(ds, full, 0, m - 1) == (a0, full.stat().st_size)
    assert b0 - a0 == b - a                                  # one record size, both files
