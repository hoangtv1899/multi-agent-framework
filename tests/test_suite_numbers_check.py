"""The numbers a study's claims cite, made again from the history (package F2).

    pytest tests/test_suite_numbers_check.py -v

THREE KINDS OF TEST, NO MODEL.

1. Known answers. Each family the check makes again (the six timing measures
   with their reasons and flags, the area-weighted quantile, the overlap's
   correlation, equal-area quarters and shared-sign rule, the elevation
   split's bands, contributions, shares, concentration and its empty and
   small bands, the screen's scores, flags, statuses and event marks) is run
   on small series and maps whose answers are worked out by hand here.
2. A study on small fake runs. A control and one experiment run are written as
   history files (elm_case_fixtures, test_domain_suite.make_control), their
   suites built, and the lookups made through the real `suite` (one of each
   method family) and logged by the framework's own log. The check, reading
   only the history, agrees with every logged number; then one number at a
   time is planted wrong in lookups.jsonl and the check names the record, the
   value logged and the value made again.
3. Independence. The check imports none of domain_methods, domain_suite,
   domain_access, gridded, regions or domain_analysis.
"""
import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pytest

import domain_access as DAC
import domain_suite as S
from test_domain_suite import FIELDS, make_control

from agents.analysis import analysis_plan as AP          # noqa: E402
from agents.analysis import numbers_check as NC          # noqa: E402
from agents.analysis import suite_check as SC            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
YEARS = [2015, 2016, 2017]
NORMAL = [2015, 2016, 2017]
REGION = "box, vegetated"
EDGES = [1000]
NAN = np.nan


# ── 1. known answers ────────────────────────────────────────────────────────

def test_the_area_weighted_quantile_is_the_averaged_inverted_cdf():
    assert SC.wquantile([1, 2, 3, 4], [1, 1, 1, 1], 0.5) == 2.5
    assert SC.wquantile([1, 2, 3], [1, 1, 2], 0.5) == 2.5
    assert SC.wquantile([1, 2, 3], [1, 1, 2], 0.25) == 1.5
    assert SC.wquantile([3, 1, 2], [1, 5, 1], 0.5) == 1.0
    assert SC.wquantile([NAN, 2.0], [1, 1], 0.5) == 2.0
    assert SC.wquantile([], [], 0.5) is None


def _cols(*series):
    return np.array(series, dtype=np.float64).T


def test_snow_peak_dates_reasons_and_a_water_year_without_its_start():
    x = _cols([0, 2, 9, 50, 7, 1, 0, 0, 0, 0], [0.5] * 10,
              [0, 2, 9, 50, 7, NAN, 0, 0, 0, 0])
    r = SC.m_snow_peak(x, -91, 0, threshold_mm=1.0)
    assert np.array_equal(r["date"], [-88, NAN, NAN], equal_nan=True)
    assert list(r["reason"]) == [0, SC.REASON["no_snow"], SC.REASON["cell_values_missing"]]
    r = SC.m_snow_peak(x, -91, 2, threshold_mm=1.0)
    assert set(r["reason"]) == {SC.REASON["before_record"]} and np.isnan(r["date"]).all()


def test_snow_gone_from_the_peak_returns_follow_up_persists_and_partial():
    x = _cols([0, 5, 9, 4, 0.5, 0.2, 0.1, 0, 0, 0, 0, 0],         # melt-out on day 4
              [9, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0, 0],               # melts, then snow returns
              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0],              # melting in the last days
              [5] * 12,                                           # snow persists
              [0.5] * 12)                                         # no snow
    r = SC.m_snow_gone(x, 1, 0, threshold_mm=1.0, days=3)
    assert np.array_equal(r["date"], [5, 2, NAN, NAN, NAN], equal_nan=True)
    R = SC.REASON
    assert list(r["reason"]) == [0, 0, R["insufficient_follow_up"], R["snow_persists"],
                                 R["no_snow"]]
    assert list(r["flags"]) == [0, SC.FLAG["snow_returns"], 0, 0, 0]
    lead = np.vstack([np.full((2, 5), NAN), x])
    p = SC.m_snow_gone(lead, -1, 2, threshold_mm=1.0, days=3)
    assert np.array_equal(p["date"], r["date"] + 0, equal_nan=True)   # searched from the record
    assert all(f & SC.FLAG["partial_record"] for f in p["flags"])


def test_leaf_green_up_from_the_low_before_the_peak_with_its_marks():
    x = _cols([1, 0.5, 0.2, 1, 2, 3, 4, 5, 3], [4, 4.2, 4.5, 4.8, 5, 4.9, 4.5, 4.4, 4.2],
              [0] * 9)
    r = SC.m_leaf_green_up(x, 1, 0, fraction=0.2, rise_mark=0.5)
    assert np.array_equal(r["date"], [5, NAN, NAN], equal_nan=True)
    assert list(r["reason"]) == [0, SC.REASON["little_leaf_change"], SC.REASON["no_leaves"]]
    r = SC.m_leaf_green_up(x, 1, 0, fraction=0.2, rise_mark=0.1)   # the rise mark is a rule
    assert r["date"][1] == 2


def test_photosynthesis_onset_after_the_low_of_the_running_mean():
    x = _cols([0, 0, 0, 0, 1, 2, 3, 4, 5, 6], [5] * 10, [0] * 10)
    r = SC.m_photosynthesis_onset(x, 1, 0, fraction=0.2, window=3)
    assert np.array_equal(r["date"], [5, NAN, NAN], equal_nan=True)
    assert list(r["reason"]) == [0, SC.REASON["no_dormant_season"],
                                 SC.REASON["no_photosynthesis"]]


def test_growth_peak_and_its_two_peaks_flag():
    x = _cols([0, 1, 5, 1, 0, 0, 1, 4.9, 1, 0], [0, 1, 5, 1, 0, 0, 0, 0, 0, 0])
    r = SC.m_growth_peak(x, 1, 0, window=3, near=0.1, dip=0.1)
    assert list(r["date"]) == [3, 3]
    assert list(r["flags"]) == [SC.FLAG["two_peaks"], 0]


def test_half_runoff_first_crossing_and_falling_runoff():
    x = _cols([1, 1, 1, 1], [2, -1, 1, 2], [0, 0, 0, 0], [-1, -1, 0, 0])
    r = SC.m_half_runoff(x, -91, 0, fraction=0.5)
    assert np.array_equal(r["date"], [-90, -91, NAN, NAN], equal_nan=True)
    assert list(r["reason"])[2:] == [SC.REASON["no_runoff"]] * 2
    assert list(r["flags"]) == [0, SC.FLAG["runoff_falls"], 0, SC.FLAG["runoff_falls"]]
    assert set(SC.m_half_runoff(x, -91, 3, fraction=0.5)["reason"]) == {SC.REASON["before_record"]}


class _Ctx:
    """The parts of the check's context a map method reads."""

    def __init__(self, w, masks, bands=None, others=()):
        self.w, self._m, self._b, self._o = np.asarray(w, float), masks, bands, list(others)

    def mask(self, name):
        if name.startswith("not "):
            return ~self._m[name[4:]]
        return np.asarray(self._m[name], dtype=bool)

    def band_names(self):
        return self._b

    def other_regions(self):
        return self._o


def test_overlap_correlation_quarters_and_the_shared_sign_rule():
    a = np.arange(1.0, 9.0)
    ctx = _Ctx(np.ones(8), {"r": np.ones(8, bool)})
    o = SC.overlap(ctx, a, a.copy(), {"region": "r"}, {"sign_share_limit": 0.8})
    assert o["correlation"] == pytest.approx(1.0) and o["covered_share"] == 1.0
    assert [o[f"quarters_by_second_map.{k}.first_map_mean"] for k in range(4)] == \
        [1.5, 3.5, 5.5, 7.5]
    assert o["same_sign_share"] is None                     # one sign covers all the area
    a = np.array([-1, -2, 3, 4, -5, 6, -7, 8.0])
    b = np.array([1, -1, 1, -1, 1, -1, 1, -1.0])
    o = SC.overlap(ctx, a, b, {"region": "r"}, {"sign_share_limit": 0.8})
    assert o["same_sign_share"] == pytest.approx(np.mean(np.sign(a) == np.sign(b)))
    assert o["sign_shares.first_positive"] == 0.5
    b2 = b.copy()
    b2[0] = NAN                                             # one cell without the second map
    o = SC.overlap(ctx, a, b2, {"region": "r"}, {"sign_share_limit": 0.8})
    assert o["cells"] == 7 and o["missing_cell_values.second.cells"] == 1
    assert o["covered_share"] == pytest.approx(7 / 8)


def test_elevation_split_bands_contributions_shares_concentration_and_flags():
    x = np.array([1.0, -1.0, 2.0, 0.0, 3.0])
    w = np.array([1.0, 1.0, 1.0, 1.0, 0.001])
    m = {"r": np.array([1, 1, 1, 1, 1], bool), "A": np.array([1, 1, 0, 0, 0], bool),
         "B": np.array([0, 0, 1, 1, 0], bool), "C": np.zeros(5, bool),
         "D": np.array([0, 0, 0, 0, 1], bool), "fire": np.array([1, 0, 1, 0, 0], bool)}
    ctx = _Ctx(w, m, bands=["A", "B", "C", "D"], others=["fire"])
    e = SC.elevation_split(ctx, x, {"region": "r"}, {"small_share": 0.01}, {})
    tot = 1 + 1 + 2 + 0 + 0.003
    assert e["total_abs_change_x_area"] == pytest.approx(tot)
    assert e["parts.A.mean"] == 0.0 and e["parts.A.positive_x_area"] == 1.0
    assert e["parts.A.negative_x_area"] == -1.0
    assert e["parts.A.abs_change_share"] == pytest.approx(2 / tot)
    assert e["parts.B.mean"] == 1.0 and e["parts.B.area_share"] == pytest.approx(2 / 4.001)
    assert e["flagged.C"] == "empty" and e["flagged.D"] == "small"
    assert "flagged.A" not in e and e["parts.C.area_share"] == 0.0
    assert e["other_regions.fire.mean"] == 1.5 and e["other_regions.not fire.mean"] == \
        pytest.approx((-1 + 0 + 0.003) / 2.001)
    # half of the total absolute change: the small cell of 3, then part of the cell of 2
    assert e["concentration"] == pytest.approx((0.001 + (tot / 2 - 0.003) / 2) / 4.001)


def test_screen_rows_scores_flags_statuses_and_event_marks(monkeypatch):
    vals = {"2015": 10.0, "2016": 14.0, "2017": 12.0}
    seas = {"DJF": 1.0, "MAM": 5.0, "JJA": 9.0, "SON": 3.0}

    def getter(ctx, run, field, region):
        def get(label):
            k, y, p = SC.parse_period(label)
            if field == "FAREA_BURNED":
                return {2015: 0.01, 2016: 0.05, 2017: 0.0}[y], 1.0
            if field == "FLAT":
                return (5.0 + (0.001 if y == 2016 else 0.0) if k == "year" else seas[p]), 1.0
            v = vals[label] if k == "year" else seas[p] + (y - 2015) * 0.1
            return (v + (1.0 if run == "warm" else 0.0)), 1.0
        return get
    monkeypatch.setattr(SC, "_getter", getter)

    class C:
        w = np.array([2.0, 3.0])
        notes = {"GPP": {"type": "flux", "percent": {"allowed": True}},
                 "TOTVEGC": {"type": "store", "percent": {"allowed": True}},
                 "FLAT": {"type": "state", "percent": {"allowed": False}},
                 "FAREA_BURNED": {"type": "event"}}

        def mask(self, name):
            return np.array([True, True])

        def note(self, f):
            return self.notes.get(f, {})
    ctx = C()
    r = SC.screen_row(ctx, "control", None, "GPP", "2016", "r", NORMAL, 0.01)
    assert r["status"] == "scored" and r["reference"] == 12.0 and r["change"] == 2.0
    assert r["score"] == pytest.approx(1.0) and r["n_normal"] == 3
    assert r["percent"] == pytest.approx(100 * 2 / 12) and r["near_constant"] is False
    assert SC.screen_row(ctx, "control", None, "TOTVEGC", "2016", "r", NORMAL,
                         0.01)["status"] == "not scored yet"
    flat = SC.screen_row(ctx, "control", None, "FLAT", "2016", "r", NORMAL, 0.01)
    assert flat["status"] == "near-constant normal" and flat["score"] is None
    assert flat["percent"] is None
    assert SC.screen_row(ctx, "control", None, "GPP", "2016", "r", [2015],
                         0.01)["status"] == "insufficient support"
    ev = SC.screen_row(ctx, "control", None, "FAREA_BURNED", "2016", "r", [2015, 2017], 0.01)
    assert ev["value"] == pytest.approx(0.05 * 5) and ev["mark"] == "above_normal_maximum"
    assert ev["reference"] == pytest.approx(0.025) and ev["score"] is None
    ev = SC.screen_row(ctx, "control", None, "FAREA_BURNED", "2016", "r", NORMAL, 0.01)
    assert ev["status"] == "within the normal years' range" and ev["mark"] is None
    d = SC.screen_row(ctx, "warm", "control", "GPP", "2016", "r", NORMAL, 0.01)
    assert d["change"] == 1.0 and d["score"] == pytest.approx(1.0 / 2.0)


# ── 2. a study on small fake runs ───────────────────────────────────────────

def _scale_gpp(run_dir, factor):
    for p in sorted(Path(run_dir).glob("*.elm.h[01].*.nc")):
        with netCDF4.Dataset(p, "a") as d:
            if "GPP" in d.variables:
                d["GPP"][:] = (np.asarray(d["GPP"][:]) * factor).astype(np.float32)


def _row(res, field, period):
    for k, v in res.items():
        if k.startswith("rows.") and k.endswith(".field") and v == field and \
                res.get(k[:-5] + "period") == period:
            return k[:-5]
    raise AssertionError(f"no row {field} {period}")


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("f2")
    control, _ = make_control(root / "ctl", YEARS)
    control["elevation_band_edges"] = EDGES
    S.build(control, root / "suite", fields=FIELDS)
    warm, _ = make_control(root / "warm", YEARS)
    _scale_gpp(warm["run_dir"], 1.1)
    rd = root / "study"
    S.build(control, rd / "03_results" / "suite" / "warm", run_dir=warm["run_dir"],
            fields=FIELDS)
    (rd / "01_inputs").mkdir(parents=True)
    (rd / "01_inputs" / "domain_control_facts.json").write_text(json.dumps(
        {"control": {"name": "synth", "run_dir": control["run_dir"],
                     "case_dir": control["case_dir"]}}))
    (rd / "domain_case" / "warm" / "checks").mkdir(parents=True)
    (rd / "domain_case" / "warm" / "checks" / "y2015.json").write_text(
        json.dumps({"run_dir": warm["run_dir"]}))
    out = rd / "04_analysis"
    out.mkdir()
    (out / AP.PLAN_FILE).write_text(json.dumps({"baseline": {"normal_years": NORMAL}}))
    acc = DAC.SuiteAccess({"control": str(root / "suite"),
                           "warm": str(rd / "03_results" / "suite" / "warm")},
                          gate=lambda ask: {"planned": True}, log=AP.LookupLog(out, "fig"),
                          normal_years=NORMAL)
    acc.anomaly("GPP", "2016", region=REGION)                                   # L1
    acc.anomaly("H2OSOI", "2017-DJF", region=REGION)                            # L2
    acc.difference("warm", "GPP", "2016-JJA", region=REGION)                    # L3
    g = acc.get_map("control", "TOTECOSYSC", "2016")                            # L4
    an = acc.anomaly("GPP", "2016")                                             # L5
    acc.overlap(an["anomaly"], g, REGION, by_band=True)                         # L6
    acc.elevation_split(an["anomaly"], "vegetated cells")                       # L7
    acc.growth_peak(2016, REGION)                                               # L8
    acc.photosynthesis_onset(2016, REGION, run="warm", against="control")       # L9
    acc.screen(2016, REGION)                                                    # L10
    acc.screen(2016, REGION, run="warm")                                        # L11
    acc.overlap(an["anomaly"] * 2.0, g, REGION)                                 # L12
    lines = [json.loads(x) for x in (out / "lookups.jsonl").read_text().splitlines()]
    assert len(lines) == 12 and all("sha256" in x for x in lines)
    res = {i + 1: x.get("result") or {} for i, x in enumerate(lines)}
    rows = {"gpp": _row(res[10], "GPP", "2016"), "tot": _row(res[10], "TOTECOSYSC", "2016-JJA"),
            "fire": _row(res[10], "COL_FIRE_CLOSS", "2016"),
            "warm": _row(res[11], "GPP", "2016-JJA")}
    claims = [{"claim": "c", "evidence": [f"L{i}"],
               "evidence_trace": {"matched": [f"L{i}:{p}" for p in paths]}}
              for i, paths in ((1, ["anomaly", "percent"]), (2, ["anomaly"]),
                               (3, ["difference", "percent"]), (6, ["correlation"]),
                               (7, ["parts.below 1000 m.abs_change_share"]),
                               (8, ["variants.default.distribution.median"]),
                               (9, ["variants.default.distribution.reasons.no_dormant_season"]),
                               (10, [rows["gpp"] + "score", rows["tot"] + "change",
                                     rows["fire"] + "value"]),
                               (11, [rows["warm"] + "score"]))]
    claims.append({"claim": "maps", "evidence": ["L4", "L5"]})
    entry = dict(control)
    return {"rd": rd, "out": out, "entry": entry, "claims": claims, "res": res, "rows": rows}


def _planted(study, tmp_path, line, path, new):
    rd = tmp_path / "planted"
    (rd / "04_analysis").mkdir(parents=True)
    for d in ("01_inputs", "domain_case"):
        shutil.copytree(study["rd"] / d, rd / d)
    shutil.copy(study["out"] / AP.PLAN_FILE, rd / "04_analysis" / AP.PLAN_FILE)
    lines = (study["out"] / "lookups.jsonl").read_text().splitlines()
    rec = json.loads(lines[line - 1])
    assert path in rec["result"], (path, sorted(rec["result"])[:30])
    rec["result"][path] = new
    lines[line - 1] = json.dumps(rec)
    (rd / "04_analysis" / "lookups.jsonl").write_text("\n".join(lines) + "\n")
    return rd


def test_the_valid_fake_study_reproduces_every_cited_number(study):
    rec = NC.check_evidence(study["rd"], study["claims"], registry=study["entry"])
    assert rec["pass"], rec.get("error") or rec["mismatches"][:5]
    assert rec["n_compared"] > 100, rec["n_compared"]
    assert set(rec["records"]) >= {f"L{i}" for i in range(1, 12)}
    assert (study["out"] / NC.EVIDENCE_JSON).is_file()
    assert "map a: line L5 (anomaly, its anomaly)" in rec["records"]["L6"]["notes"]
    # it read only what the cited records name: GPP, H2OSOI, TOTECOSYSC (the
    # map and the screen's JJA row) and COL_FIRE_CLOSS (the screen's fire row)
    read = {r.split(": ")[1].split(" from ")[0] for r in rec["sources"]["reads"]}
    assert read == {"GPP", "H2OSOI", "TOTECOSYSC", "COL_FIRE_CLOSS"}, read
    line = NC.evidence_page_line(rec)
    assert line.startswith("**Numbers check: passed.**")


FAULTS = [("region value", 1, "value", "*", 1.001),
          ("anomaly", 1, "anomaly", "*", 1.0001),
          ("percent", 1, "percent", "*", 0.9999),
          ("difference", 3, "difference", "*", 1.0001),
          ("layered DJF anomaly", 2, "anomaly", "*", 1.001),
          ("normal year missing", 2, "normal_missing", "=", []),
          ("coverage share", 1, "coverage.least_area_share", "*", 0.98),
          ("support", 1, "support.sd", "*", 1.001),
          ("screen score", 10, "{gpp}score", "*", 1.0001),
          ("screen change", 10, "{tot}change", "*", 1.0001),
          ("event value", 10, "{fire}value", "*", 1.001),
          ("screen status", 10, "{gpp}status", "=", "near-constant normal"),
          ("experiment screen score", 11, "{warm}score", "*", 1.0001),
          ("overlap correlation", 6, "correlation", "*", 0.999),
          ("overlap quarter", 6, "quarters_by_second_map.2.first_map_mean", "*", 1.0001),
          ("overlap band", 6, "by_band.below 1000 m.cells", "+", 1),
          ("band share", 7, "parts.below 1000 m.abs_change_share", "*", 1.0001),
          ("concentration", 7, "concentration", "*", 1.0001),
          ("timing date", 8, "variants.default.distribution.median", "+", 1),
          ("eligibility count", 9, "variants.default.shift.eligible.both.cells", "+", 1),
          ("reason count", 9, "variants.default.distribution.reasons.no_dormant_season", "+", 1),
          ("timing coverage", 8, "variants.default.distribution.inputs_area_share", "*", 0.99),
          # THE RANGE ACROSS SETTINGS (the user's decisions on G part 1, 2026-11-08)
          ("shift on the comparable cells", 8, "across_settings.evaluated.1.median_shift", "+",
           0.5),
          ("comparable cells", 8, "across_settings.comparable.cells", "+", 1),
          ("comparable share", 8, "across_settings.comparable.area_share", "*", 0.99),
          ("range spread", 8, "across_settings.spread_days", "+", 0.5),
          ("range largest", 8, "across_settings.maximum.median_shift", "+", 0.5),
          ("range smallest's setting", 8, "across_settings.minimum.variant", "=", "window=99"),
          ("range on the control", 9, "across_settings.comparable.cells", "+", 1)]


def plant(old, op, x):
    return x if op == "=" else old + x if op == "+" else old * x


@pytest.mark.parametrize("what,line,path,op,x", FAULTS, ids=[f[0] for f in FAULTS])
def test_a_number_planted_wrong_in_its_record_is_caught(study, tmp_path, what, line, path, op,
                                                        x):
    path = path.format(**study["rows"])
    old = study["res"][line][path]
    new = plant(old, op, x)
    assert new != old, (old, new)
    rd = _planted(study, tmp_path, line, path, new)
    rec = NC.check_evidence(rd, study["claims"], registry=study["entry"])
    assert not rec["pass"]
    hits = [m for m in rec["mismatches"] if m["record"] == f"L{line}" and m["path"] == path]
    assert hits and hits[0]["logged"] == new, rec["mismatches"][:3]
    assert len(rec["mismatches"]) == 1, rec["mismatches"]
    why = NC.evidence_reason(rec)
    assert f"L{line}" in why and path in why


@pytest.mark.parametrize("flag", ["range_required", "direction_reversal"])
def test_a_range_flag_planted_wrong_is_caught(study, tmp_path, flag):
    """The two flags of the range across settings (the user's decisions on G
    part 1, 2026-11-08), each made again by the check's own rule: flipped in
    the record, the check finds it."""
    path = f"across_settings.{flag}"
    for line in (8, 9):
        old = study["res"][line][path]
        assert isinstance(old, bool), (line, path, old)
        rd = _planted(study, tmp_path / f"L{line}", line, path, not old)
        rec = NC.check_evidence(rd, study["claims"], registry=study["entry"], write=False)
        hits = [m for m in rec["mismatches"] if m["record"] == f"L{line}" and m["path"] == path]
        assert not rec["pass"] and hits and len(rec["mismatches"]) == 1, rec["mismatches"][:3]


def test_the_range_across_settings_is_made_again_in_full(study):
    """Every number and flag of the range across settings is made again: none
    is left among the numbers the check does not make."""
    rec = NC.check_evidence(study["rd"], study["claims"], registry=study["entry"], write=False)
    assert rec["pass"], rec["mismatches"][:3]
    left = [p for x in rec["not_recomputed"] for p in x.get("paths") or []
            if p.startswith("across_settings.")]
    assert not left, left
    for line in (8, 9):
        assert any(p.startswith("across_settings.") for p in study["res"][line])


def test_a_map_made_by_a_scripts_own_arithmetic_cannot_be_made_again(study):
    rec = NC.check_evidence(study["rd"], [{"claim": "x", "evidence": ["L12"]}],
                            registry=study["entry"], write=False)
    assert not rec["pass"]
    assert "no array a logged lookup returned" in rec["mismatches"][0]["why"]


def test_a_run_without_a_managers_record_is_not_read_from_anywhere_else(study, tmp_path):
    rd = _planted(study, tmp_path, 3, "difference", study["res"][3]["difference"])
    shutil.rmtree(rd / "domain_case")
    rec = NC.check_evidence(rd, study["claims"][2:3], registry=study["entry"], write=False)
    assert not rec["pass"] and "no record of the manager names run warm" in \
        rec["mismatches"][0]["why"]


def test_the_withheld_page_names_the_record_the_logged_and_the_remade_value(study, tmp_path):
    from agents.analysis.claim_trace import evidence_withheld_markdown
    rd = _planted(study, tmp_path, 1, "anomaly", study["res"][1]["anomaly"] * 1.01)
    rec = NC.check_evidence(rd, study["claims"], registry=study["entry"], write=False)
    page = evidence_withheld_markdown({
        "question": "q", "headline": "h", "claims": [],
        "evidence_numbers": rec,
        "evidence_trace": {"withheld_because": NC.evidence_reason(rec), "records": {}}})
    assert page.startswith("# WITHHELD: 1 of the numbers its claims cite did not reproduce")
    assert "## The numbers made again from the model output" in page
    assert "| L1 (anomaly, control, GPP, 2016, box, vegetated) | anomaly |" in page


# ── 3. independence ─────────────────────────────────────────────────────────

FORBIDDEN = {"domain_methods", "domain_suite", "domain_access", "gridded", "regions",
             "domain_analysis"}


def test_the_check_imports_none_of_the_code_it_checks():
    for name in ("suite_check.py", "numbers_check.py"):
        tree = ast.parse((ROOT / "src" / "agents" / "analysis" / name).read_text())
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[-1] for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                mods |= {(n.module or "").split(".")[-1]} | {a.name for a in n.names}
        assert not mods & FORBIDDEN, (name, mods & FORBIDDEN)
    code = ("import sys; sys.path[:0] = [{src!r}, {elm!r}]; "
            "from agents.analysis import suite_check; "
            "print(sorted(m for m in sys.modules if m.split('.')[-1] in {bad!r}))").format(
        src=str(ROOT / "src"), elm=str(ROOT / "mcp" / "elm-mcp" / "src"), bad=sorted(FORBIDDEN))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(ROOT), timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "[]", out.stdout
