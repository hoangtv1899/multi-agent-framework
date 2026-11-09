"""The suite's timing.nc: the timing measures of domain_methods, per cell and
year, every setting, made from the suite's own passes.

    pytest tests/test_domain_suite_timing.py -v

TIMING.NC (2026-11-05). On the synthetic run of tests/test_domain_suite.py
(its daily stream holds GPP, so photosynthesis_onset and growth_peak are made
and the other four are recorded as not built): what the build writes equals
domain_methods.timing_run on the same run, value for value; a suite without
timing.nc gets it from a pass of its own and nothing else is touched; a build
after that is a no-op; a changed method version makes timing.nc again and only
it. The passes themselves are counted in tests/test_domain_suite.py (one per
stream and year).
"""
import json

import netCDF4
import numpy as np
import pytest

import domain_methods as M
import domain_suite as S
import gridded as G
from test_domain_suite import FIELDS, make_control

YEARS = [2015, 2016]


def _mtimes(folder):
    return {p.name: p.stat().st_mtime_ns for p in folder.iterdir()}


def _read(path):
    with netCDF4.Dataset(path) as d:
        d.set_auto_mask(False)
        return {v: np.asarray(d[v][:]) for v in d.variables}, \
            {a: getattr(d, a) for a in d.ncattrs()}


@pytest.fixture()
def built(tmp_path):
    control, _ = make_control(tmp_path, YEARS)
    out = tmp_path / "suite"
    res = S.build(control, out, fields=FIELDS)
    return {"control": control, "out": out, "res": res, "man": S.load(out)}


def test_build_writes_timing_equal_to_the_methods(built):
    out, man = built["out"], built["man"]
    t = man["timing"]
    assert t["file"] == S.TIMING_FILE and (out / S.TIMING_FILE).is_file()
    assert sorted(t["measures"]) == ["growth_peak", "photosynthesis_onset"]
    assert sorted(t["not_built"]) == ["half_runoff", "leaf_green_up", "snow_gone", "snow_peak"]
    assert t["years"] == YEARS and t["stream"] == "h1" and "own passes" in t["made_by"]
    assert man["fields"]["GPP"]["products"] == list(S.PRODUCTS)       # a product of its own
    assert S.TIMING_FILE in man["files"]
    got, attrs = _read(out / S.TIMING_FILE)
    assert attrs["calendar"] == "noleap" and json.loads(attrs["reasons"])["0"] == "a date"
    want = M.timing_run(built["control"]["run_dir"], YEARS,
                        measures=("photosynthesis_onset", "growth_peak"))
    shape = got["lat"].shape + got["lon"].shape
    for mid in ("photosynthesis_onset", "growth_peak"):
        for lab, st in M.METHODS[mid].variants():
            base = S._timing_name(mid, lab)
            assert t["measures"][mid]["variants"][lab] == st
            for i, y in enumerate(YEARS):
                w = want[mid][y]["variants"][lab]
                np.testing.assert_array_equal(got[base + "__date"][..., i],
                                              w["date"].reshape(shape).astype(np.float32))
                assert (got[base + "__reason"][..., i] == w["reason"].reshape(shape)).all()
                assert (got[base + "__flags"][..., i] == w["flags"].reshape(shape)).all()
            assert got[base + "__date"].dtype == np.float32 and got[base + "__reason"].dtype == np.int8
    assert got["year"].tolist() == YEARS


def test_a_suite_without_timing_gets_it_and_nothing_else_moves(built, monkeypatch):
    out = built["out"]
    first, _ = _read(out / S.TIMING_FILE)
    (out / S.TIMING_FILE).unlink()
    man = json.loads((out / S.MANIFEST).read_text())
    man.pop("timing")
    (out / S.MANIFEST).write_text(json.dumps(man))
    before = {k: v for k, v in _mtimes(out).items() if k != S.MANIFEST}
    calls = []
    real = G.Stream.read_block_pass

    def counted(self, vars, year, *a, **k):
        calls.append((self.name, int(year), tuple(sorted(vars))))
        return real(self, vars, year, *a, **k)
    monkeypatch.setattr(G.Stream, "read_block_pass", counted)
    res = S.build(built["control"], out, fields=FIELDS)
    assert res["action"] == "reused" and S.TIMING_FILE in res["why"]
    assert sorted(calls) == [("h1", 2015, ("GPP",)), ("h1", 2016, ("GPP",))]
    after = _mtimes(out)
    assert {k: after[k] for k in before} == before                       # nothing else rewritten
    again, _ = _read(out / S.TIMING_FILE)
    for k, a in first.items():
        np.testing.assert_array_equal(again[k], a, err_msg=k)
    t = S.load(out)["timing"]
    assert "a pass of its own" in t["made_by"] and t["years"] == YEARS
    calls.clear()
    now = _mtimes(out)
    assert S.build(built["control"], out, fields=FIELDS)["action"] == "reused"
    assert calls == [] and _mtimes(out) == now                           # then a no-op


def test_a_changed_method_version_makes_timing_again_and_only_it(built):
    out = built["out"]
    man = json.loads((out / S.MANIFEST).read_text())
    man["timing"]["measures"]["growth_peak"]["version"] = 0
    (out / S.MANIFEST).write_text(json.dumps(man))
    before = {k: v for k, v in _mtimes(out).items() if k not in (S.MANIFEST, S.TIMING_FILE)}
    res = S.build(built["control"], out, fields=FIELDS)
    assert res["action"] == "reused" and "versions" in res["why"]
    after = _mtimes(out)
    assert {k: after[k] for k in before} == before
    assert S.load(out)["timing"]["measures"]["growth_peak"]["version"] == \
        M.METHODS["growth_peak"].version
