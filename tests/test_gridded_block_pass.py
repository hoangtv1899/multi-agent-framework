"""The one-pass reader against read(), and the regional numbers it now makes.

    pytest tests/test_gridded_block_pass.py -v

Stream.read_block_pass walks a year's records in blocks and takes every field
from a block before the next (gridded, ONE PASS OVER A YEAR, 2026-11-04). Each
check here sets the pass beside read() on small files in ELM's own layout
(elm_gridded_fixtures), record for record and bit for bit, so anything read()
gets right the pass must get right too: the year assembled across two files,
the run's start step left out, a short year, monthly and 3-hourly streams,
fill, rates per second, layers and windows, and every refusal. The values
themselves are checked against the fixture's formulas by
tests/test_gridded_reader.py.

The regional numbers (regions.reduce_regions, ONE PASS PER YEAR) are set
beside the reduction as it was before, written out below: each part read over
a whole piece of the year with read_records, field after field, then the same
matrix products. They must agree to the last bit, not to a tolerance.
"""
import shutil
import weakref

import numpy as np
import pytest

import domain_analysis as A
import elm_case_fixtures as cf
import elm_gridded_fixtures as fx
import gridded as G
import regions as R

Y0 = 2015
YEARS = [2015, 2016]
DAILY = ["DOYCODE", "CELLID", "GPP", "QVEGT", "QVEGE", "QSOIL", "H2OSNO",
         "FAREA_BURNED", "TSA"]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    fx.write_daily(d, YEARS)                               # h1: 2015, 2016, 2017
    fx.write_daily(d, YEARS, stream="h2", steps_per_day=8)
    fx.write_monthly(d, YEARS, variables=("GPP", "TSA", "CELLID", "FAREA_BURNED"))
    return d


def _same_bits(a, b) -> bool:
    """Equal to the last bit: same shape, same dtype, same bytes (NaN included)."""
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _pass(stream, names, year, **kw):
    """Every block the pass hands over, and what it returns."""
    got = []
    out = stream.read_block_pass(names, year, reduce=got.append, **kw)
    return got, out


def _assert_pass_is_read(stream, names, year, block_records, **kw):
    blocks, out = _pass(stream, names, year, block_records=block_records, **kw)
    n_rec = len(stream.year_records(year)[0])
    sizes = [len(b.keys) for b in blocks]
    assert sizes == [min(block_records, n_rec - a) for a in range(0, n_rec, block_records)]
    assert [b.first for b in blocks] == list(range(0, n_rec, block_records))
    assert out["records"] == n_rec and out["blocks"] == len(blocks) and out["year"] == year
    for v in names:
        want = stream.read(v, year, **kw)
        pieces = [b.block(v) for b in blocks]
        got = np.concatenate([p.data for p in pieces], axis=0)
        assert got.dtype == np.float64
        assert _same_bits(got, want.data), v
        assert [k for p in pieces for k in p.keys] == want.keys
        assert [x for p in pieces for x in p.labels] == want.labels
        assert all(p.units == want.units and p.per_second is want.per_second
                   for p in pieces), v
        assert [m for p in pieces for m in p.missing] == want.missing
        assert pieces[0].missing == want.missing and out["missing"] == want.missing
        assert list(blocks[0].data) == list(names)
    return blocks


# ── the pass gives what read() gives ─────────────────────────────────────────
class TestThePassIsRead:

    @pytest.mark.parametrize("block_records", [1, 7, 400])
    def test_a_daily_year_across_two_files(self, run, block_records):
        """31 December of each year is the first record of the next year's
        file; the run's 30-minute start step opens the first file."""
        s = G.open_stream(run, "h1")
        for y in YEARS:
            _assert_pass_is_read(s, DAILY, y, block_records)
        have, _ = s.year_records(2015)
        assert len({r.file for r in have}) == 2

    @pytest.mark.parametrize("block_records", [1, 7, 13])
    def test_a_monthly_stream(self, run, block_records):
        """January 2015 holds the run's first step and is kept (Stream.extended)."""
        s = G.open_stream(run, "h0")
        assert [e["month"] for e in s.extended] == ["2015-01"]
        for y in YEARS:
            _assert_pass_is_read(s, ["GPP", "TSA", "CELLID", "FAREA_BURNED", "DOYCODE"],
                                 y, block_records)

    @pytest.mark.parametrize("block_records", [1, 7, 3000])
    def test_a_three_hourly_stream(self, run, block_records):
        s = G.open_stream(run, "h2")
        _assert_pass_is_read(s, ["STEPCODE", "CELLID", "TSA"], 2016, block_records)

    def test_the_start_step_never_appears(self, run):
        blocks, _ = _pass(G.open_stream(run, "h1"), ["DOYCODE"], 2015, block_records=30)
        first = blocks[0].data["DOYCODE"]
        assert first[0, 0, 0] == 1.0 and np.all(first != -999.0)

    def test_fill_and_nan_are_nan_where_read_has_them(self, run):
        s = G.open_stream(run, "h1")
        blocks, _ = _pass(s, ["GPP", "QSOIL"], 2015, block_records=30)
        gpp = np.concatenate([b.data["GPP"] for b in blocks])
        assert np.all(np.isnan(gpp[:, 0, 0]))              # a cell GPP never covers
        assert np.isnan(gpp[99, 1, 1]) and np.isnan(gpp[49, 2, 2])    # 1e36, and NaN
        assert int(np.isnan(gpp).sum()) == 365 + 2
        qsoil = np.concatenate([b.data["QSOIL"] for b in blocks])
        assert np.isnan(qsoil[9, 1, 2]) and int(np.isnan(qsoil).sum()) == 1

    def test_units_and_rates_per_second(self, run):
        """FAREA_BURNED says 'proportion' and is a rate per second
        (gridded.PER_SECOND_DESPITE_UNITS); the pass says so as read() does."""
        blocks, _ = _pass(G.open_stream(run, "h1"), ["FAREA_BURNED", "GPP", "TSA"], 2016,
                          block_records=100)
        b = blocks[0]
        assert b.units == {"FAREA_BURNED": "proportion", "GPP": "gC/m^2/s", "TSA": "K"}
        assert b.per_second == {"FAREA_BURNED": True, "GPP": True, "TSA": False}

    @pytest.mark.parametrize("layer", [None, 0, 2, -1])
    def test_a_layered_field_all_levels_or_one(self, run, layer):
        s = G.open_stream(run, "h1")
        blocks = _assert_pass_is_read(s, ["SOILLIQ"], 2016, 7, layer=layer)
        shape = blocks[0].data["SOILLIQ"].shape
        assert shape == ((7, fx.NLEV, 4, 5) if layer is None else (7, 4, 5))

    def test_a_window(self, run):
        _assert_pass_is_read(G.open_stream(run, "h1"), ["CELLID", "GPP", "SOILLIQ"], 2016, 30,
                             window=(1, 3, 2, 5))

    def test_a_short_year(self, run, tmp_path):
        """Without the 2017 file, 2016 lacks 31 December: refused as read()
        refuses it, or read without it, the absent label listed once."""
        d = tmp_path / "short"
        shutil.copytree(run, d)
        (d / "synth.elm.h1.2017-01-01-00000.nc").unlink()
        s = G.open_stream(d, "h1")
        with pytest.raises(G.GriddedReadError) as want:
            s.read("GPP", 2016)
        got = []
        with pytest.raises(G.GriddedReadError) as e:
            s.read_block_pass(["GPP", "TSA"], 2016, reduce=got.append)
        assert str(e.value) == str(want.value) and got == []
        for n in (1, 7, 400):
            blocks = _assert_pass_is_read(s, ["GPP", "TSA"], 2016, n, require_complete=False)
            assert blocks[0].missing == ["2016-12-31"] and all(b.missing == [] for b in blocks[1:])
            assert sum(len(b.keys) for b in blocks) == 364

    def test_a_restarted_segment(self, tmp_path):
        """MidA_fdrought's layout: one file per month, and a new segment whose
        January starts one step early, inside December (Stream.extended)."""
        g = fx.grid()
        fx._write(tmp_path / "s.elm.h0.2018-12.nc", g, 2018, [(334.0, 365.0)],
                  [(2018, 349, 0)], ["DOYCODE", "GPP"])
        for m in range(1, 13):
            a = sum(fx.MONTH_DAYS[:m - 1]) - (fx.DT if m == 1 else 0.0)
            fx._write(tmp_path / f"s.elm.h0.2019-{m:02d}.nc", g, 2019,
                      [(a, float(sum(fx.MONTH_DAYS[:m])))],
                      [(2019, sum(fx.MONTH_DAYS[:m - 1]) + 15, 0)], ["DOYCODE", "GPP"])
        s = G.open_stream(tmp_path, "h0")
        assert s.extended[0]["overlaps_previous_days"] == round(fx.DT, 6)
        for n in (1, 5, 12):
            _assert_pass_is_read(s, ["DOYCODE", "GPP"], 2019, n)
        _assert_pass_is_read(s, ["DOYCODE"], 2018, 3, require_complete=False)

    def test_an_unstructured_grid(self, tmp_path):
        fx.write_daily_lnd(tmp_path, YEARS)
        s = G.open_stream(tmp_path, "h1")
        _assert_pass_is_read(s, ["GPP", "TSA", "SOILLIQ"], 2016, 30)
        _assert_pass_is_read(s, ["SOILLIQ"], 2016, 30, layer=1)
        with pytest.raises(G.GriddedReadError, match="regular"):
            s.read_block_pass(["GPP"], 2016, reduce=print, window=(0, 1, 0, 1))


class TestWhatThePassKeeps:

    def test_nothing_the_callback_lets_go(self, run):
        """Each block's arrays are gone by the time the next one arrives."""
        s = G.open_stream(run, "h1")
        refs = []

        def take(b):
            assert all(r() is None for r in refs), "an earlier block was kept"
            refs[:] = [weakref.ref(a) for a in b.data.values()]
        out = s.read_block_pass(DAILY, 2016, block_records=30, reduce=take)
        assert all(r() is None for r in refs)
        assert set(out) == {"year", "records", "blocks", "missing", "read_ahead"}
        assert set(out["read_ahead"]) == {"on", "bytes_planned", "bytes_read", "error"}

    def test_each_file_is_opened_once(self, run, monkeypatch):
        """A year of 13 blocks and nine fields opens its two files once each,
        where read_records opens a file for every field."""
        import netCDF4
        s = G.open_stream(run, "h1")
        opened = []
        real = netCDF4.Dataset

        def counting(*a, **k):
            opened.append(str(a[0]))
            return real(*a, **k)
        monkeypatch.setattr(netCDF4, "Dataset", counting)
        s.read_block_pass(DAILY, 2016, block_records=30, reduce=lambda b: None)
        assert sorted(p.rsplit("/", 1)[-1] for p in opened) == [
            "synth.elm.h1.2016-01-01-00000.nc", "synth.elm.h1.2017-01-01-00000.nc"]


class TestRefusals:
    """Everything read() refuses is refused before the first block."""

    @pytest.fixture
    def calls(self):
        return []

    def _refused(self, s, calls, match, names, year=2016, **kw):
        with pytest.raises(G.GriddedReadError, match=match):
            s.read_block_pass(names, year, reduce=calls.append, **kw)
        assert calls == [], "a block was handed over before the refusal"

    def test_what_read_refuses(self, run, calls):
        s = G.open_stream(run, "h1")
        self._refused(s, calls, "not on stream h1", ["GPP", "NOPE"])
        self._refused(s, calls, "per-record value", ["GPP", "nstep"])
        self._refused(s, calls, "no levels", ["SOILLIQ", "GPP"], layer=1)
        self._refused(s, calls, "not one of them", ["SOILLIQ"], layer=fx.NLEV)
        self._refused(s, calls, "no records in 2030", ["GPP"], year=2030,
                      require_complete=False)

    def test_its_own_arguments(self, run, calls):
        s = G.open_stream(run, "h1")
        self._refused(s, calls, "at least 1", ["GPP"], block_records=0)
        self._refused(s, calls, "repeat", ["GPP", "TSA", "GPP"])
        self._refused(s, calls, "at least one field", [])
        with pytest.raises(G.GriddedReadError, match="reduce callback"):
            s.read_block_pass(["GPP"], 2016)

    def test_one_field_by_name(self, run):
        s = G.open_stream(run, "h1")
        blocks, _ = _pass(s, "TSA", 2016, block_records=400)
        assert list(blocks[0].data) == ["TSA"]

    def test_an_instantaneous_field_on_a_daily_stream(self, tmp_path, calls):
        import netCDF4
        fx.write_daily(tmp_path, [2015], variables=("TSA", "GPP"))
        for p in tmp_path.glob("*.nc"):
            with netCDF4.Dataset(p, "a") as ds:
                ds["TSA"].cell_methods = "time: point"
        self._refused(G.open_stream(tmp_path, "h1"), calls, "instantaneous", ["GPP", "TSA"],
                      year=2015)

    def test_a_field_a_later_file_lacks(self, tmp_path, calls):
        fx.write_daily(tmp_path, [2015], variables=("GPP", "TSA"))
        (tmp_path / "synth.elm.h1.2016-01-01-00000.nc").unlink()
        fx._write(tmp_path / "synth.elm.h1.2016-01-01-00000.nc", fx.grid(), 2015,
                  [(364.0, 365.0)], [(2015, 365, 0)], ["GPP"])
        s = G.open_stream(tmp_path, "h1")
        with pytest.raises(G.GriddedReadError) as want:
            s.read("TSA", 2015)
        with pytest.raises(G.GriddedReadError) as e:
            s.read_block_pass(["GPP", "TSA"], 2015, reduce=calls.append)
        assert str(e.value) == str(want.value) and "missing from" in str(e.value)
        assert calls == []


# ═════════════════════════════════════════════════════════════════════════════
# THE REGIONAL NUMBERS, BIT FOR BIT
# ═════════════════════════════════════════════════════════════════════════════
def _field_by_field(stream, fields, masks, years, regions=None, require_complete=True,
                    max_records=None):
    """reduce_regions as it read before 2026-11-04 (HEAD e96fcbf): for each
    piece of each year, each part read over the whole piece with read_records,
    summed, scaled and reduced by the same matrix products. The oracle the
    one-pass reduction must equal to the last bit."""
    flds = [f if isinstance(f, R.Field) else R.Field(str(f)) for f in fields]
    regs = masks.names() if regions is None else list(regions)
    g = stream.grid
    w = g.weight_km2
    union = np.zeros(g.shape, dtype=bool)
    for r in regs:
        union |= masks[r]
    window = None
    if g.regular:
        rows, cols = np.flatnonzero(union.any(axis=1)), np.flatnonzero(union.any(axis=0))
        window = (int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)
        r0, r1, c0, c1 = window
        cells = union[r0:r1, c0:c1].ravel()
        sub_w = w[r0:r1, c0:c1].ravel()[cells]
        sub = {r: masks[r][r0:r1, c0:c1].ravel()[cells] for r in regs}
    else:
        cells = union.ravel()
        sub_w = w.ravel()[cells]
        sub = {r: masks[r].ravel()[cells] for r in regs}
    W = np.stack([np.where(sub[r], sub_w, 0.0) for r in regs], axis=1)
    wtot = W.sum(axis=0)
    if max_records is None:
        max_records = 31 * (stream.steps_per_day or 1) if stream.cadence == "subdaily" \
            else 10 ** 9
    vals = {(r, f.name): [] for r in regs for f in flds}
    vfrac = {(r, f.name): [] for r in regs for f in flds}
    keys = []
    for y in sorted(int(v) for v in years):
        have, missing = stream.year_records(y)
        assert have and not (missing and require_complete)
        for a in range(0, len(have), int(max_records)):
            part = have[a:a + int(max_records)]
            for f in flds:
                rate = stream.per_second(f.sources[0]) and f.to_per_day
                total = None
                for p in f.sources:
                    x = stream.read_records(p, part, window=window, layer=f.layer)
                    x = x.reshape(len(part), -1)[:, cells]
                    total = x.copy() if total is None else total + x
                total *= G.SECONDS_PER_DAY if rate else 1.0
                ok = np.isfinite(total)
                num = np.where(ok, total, 0.0) @ W
                den = ok.astype(np.float64) @ W
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
                for j, r in enumerate(regs):
                    vals[(r, f.name)].append(mean[:, j])
                    vfrac[(r, f.name)].append(den[:, j] / wtot[j])
            keys += [rec.key for rec in part]
    return (keys, {k: np.concatenate(v) for k, v in vals.items()},
            {k: np.concatenate(v) for k, v in vfrac.items()})


def _assert_same_as_field_by_field(rs, stream, fields, masks, years, **kw):
    keys, vals, vfrac = _field_by_field(stream, fields, masks, years, **kw)
    assert rs.keys == keys
    assert set(rs.values) == set(vals)
    for k in vals:
        assert _same_bits(rs.values[k], vals[k]), k
        assert _same_bits(rs.valid_fraction[k], vfrac[k]), k


@pytest.fixture(scope="module")
def masks(run):
    g = G.open_stream(run, "h1").grid
    ms = R.MaskSet(g)
    allc = np.ones(g.shape, bool)
    allc[-1, -2] = False                       # the zero-weight cell
    ms.add("all", allc, "every weighted cell")
    row2 = np.zeros(g.shape, bool)
    row2[2, :] = True
    ms.add("row 2", row2, "row 2")
    ms.add("cell (1,1)", R.cell_mask(g, 6), "one cell")
    east = np.zeros(g.shape, bool)
    east[1:3, 3:5] = True
    ms.add("east block", east, "rows 1-2, cols 3-4")
    return ms


DAILY_FIELDS = [R.Field("GPP"), R.Field("ET", ("QVEGT", "QVEGE", "QSOIL")),
                R.Field("H2OSNO"), R.Field("TSA"), R.Field("FAREA_BURNED"),
                R.Field("SOILLIQ_2", ("SOILLIQ",), layer=1)]


class TestTheRegionalNumbersDoNotChange:

    @pytest.mark.parametrize("max_records", [None, 37, 1])
    def test_a_daily_stream(self, run, masks, max_records):
        s = G.open_stream(run, "h1")
        rs = R.reduce_regions(s, DAILY_FIELDS, masks, YEARS, max_records=max_records)
        _assert_same_as_field_by_field(rs, s, DAILY_FIELDS, masks, YEARS,
                                       max_records=max_records)
        assert len(rs.keys) == 730

    def test_a_window_of_some_regions(self, run, masks):
        s = G.open_stream(run, "h1")
        rs = R.reduce_regions(s, ["TSA", "GPP"], masks, [2016], regions=["east block"])
        _assert_same_as_field_by_field(rs, s, ["TSA", "GPP"], masks, [2016],
                                       regions=["east block"])

    def test_a_monthly_stream(self, run, masks):
        s = G.open_stream(run, "h0")
        flds = ["GPP", "TSA", "FAREA_BURNED", R.Field("GPP_s", ("GPP",), to_per_day=False)]
        rs = R.reduce_regions(s, flds, masks, YEARS)
        _assert_same_as_field_by_field(rs, s, flds, masks, YEARS)

    def test_a_three_hourly_stream_in_its_pieces(self, run, masks):
        """31 days of steps per piece (248 records) while the pass reads 30 at
        a time: pieces and blocks do not line up, and nothing moves."""
        s = G.open_stream(run, "h2")
        rs = R.reduce_regions(s, ["TSA", "CELLID"], masks, [2016])
        _assert_same_as_field_by_field(rs, s, ["TSA", "CELLID"], masks, [2016])
        assert len(rs.keys) == 2920

    def test_one_part_at_two_layers(self, run, masks):
        """Two fields from one layered variable: one pass per layer."""
        s = G.open_stream(run, "h1")
        flds = [R.Field("L0", ("SOILLIQ",), layer=0), R.Field("GPP"),
                R.Field("L2", ("SOILLIQ",), layer=2)]
        rs = R.reduce_regions(s, flds, masks, [2015])
        _assert_same_as_field_by_field(rs, s, flds, masks, [2015])

    def test_a_short_year(self, run, masks, tmp_path):
        d = tmp_path / "short"
        shutil.copytree(run, d)
        (d / "synth.elm.h1.2017-01-01-00000.nc").unlink()
        s = G.open_stream(d, "h1")
        with pytest.raises(G.GriddedReadError, match="1 of 365"):
            R.reduce_regions(s, ["GPP"], masks, [2016])
        rs = R.reduce_regions(s, ["GPP", "TSA"], masks, YEARS, require_complete=False)
        _assert_same_as_field_by_field(rs, s, ["GPP", "TSA"], masks, YEARS,
                                       require_complete=False)
        assert len(rs.keys) == 729

    def test_nan_cells_are_counted_as_before(self, run, masks):
        s = G.open_stream(run, "h1")
        rs = R.reduce_regions(s, DAILY_FIELDS, masks, YEARS, max_records=37)
        assert rs.nan_cells[("all", "GPP")] == 3        # never covered, day 100, day 50
        assert rs.nan_cells[("all", "ET")] == 1         # QSOIL on day 10
        assert rs.nan_cells[("all", "TSA")] == 0

    def test_a_layered_field_is_refused(self, run, masks):
        s = G.open_stream(run, "h1")
        with pytest.raises(R.RegionError, match="layered"):
            R.reduce_regions(s, ["GPP", "SOILLIQ"], masks, [2015])


class TestTheRegionalTable:
    """domain_analysis.run_values through the pass: every reduction it makes
    (the years asked, and the year before for December) equals the
    field-by-field reduction to the last bit, so its table does too."""

    def _checked(self, monkeypatch):
        real = R.reduce_regions
        calls = []

        def checked(stream, fields, masks, years, **kw):
            rs = real(stream, fields, masks, years, **kw)
            _assert_same_as_field_by_field(rs, stream, fields, masks, years,
                                           require_complete=kw.get("require_complete", True))
            calls.append((stream.name, [f.name for f in fields], sorted(years)))
            return rs
        monkeypatch.setattr(R, "reduce_regions", checked)
        return calls

    def test_the_december_before_and_the_layered_field(self, tmp_path, monkeypatch):
        years = [2015, 2016, 2017]
        c = cf.write_case(tmp_path / "x", years=tuple(years))
        # the case's own h0 and h1 files, rewritten with the fields asked for
        fx.write_monthly(c["run_dir"], years, case="synth",
                         variables=("GPP", "TSA", "CELLID", "FAREA_BURNED"))
        fx.write_daily(c["run_dir"], years, case="synth", stream="h1",
                       variables=("GPP", "TSA", "SOILLIQ", "H2OSNO"))
        calls = self._checked(monkeypatch)
        grid = G.open_stream(c["run_dir"], "h0").grid
        ms = A.masks(grid, c["surface"], [A.VEGETATED, A.BANDS])["masks"]
        out = A.run_values(c["run_dir"], ms, ["GPP", "TSA", "FAREA_BURNED", "H2OSNO",
                                              "SOILLIQ"], [2016, 2017])
        # 2015 is read for its December only, the first month of DJF 2016
        assert calls == [("h0", ["GPP", "TSA", "FAREA_BURNED"], years),
                         ("h1", ["H2OSNO"], years)]
        sea = out["seasonal"][A.VEGETATED]
        assert sea["GPP"]["2016-DJF"] is not None and sea["H2OSNO"]["2016-DJF"] is not None
        assert "2015-12" not in out["monthly"][A.VEGETATED]["GPP"]
        assert set(out["values"][A.VEGETATED]["GPP"]) == {"2016", "2017"}
        assert out["fields"]["H2OSNO"]["stream"] == "h1"
        # a layered field still has no regional number
        assert "SOILLIQ" not in out["fields"]
        assert any(p.startswith("SOILLIQ has a value at each of its levels")
                   for p in out["problems"])
