"""Bars a reader can read (the user's decision of 2026-11-09), synthetic.

    pytest tests/test_figure_bars.py -v

A panel of one anomaly drew one bar that filled the panel, since the axes fit
themselves to it, so every lone bar looked alike and its value could be read
only off the ticks. A bar is now narrow and carries its value (a true minus
sign, beyond the bar's end, inside the panel), and an anomaly's bar stands on
its normal years' range, read from its own record, on the scale the bar is
drawn on, which the figure's legend names. Plotting only: values, units,
records and caveats are as before, and a figure without bars is laid out as
before.
"""
import numpy as np
import pytest
from matplotlib.figure import Figure

from agents.analysis import figure_spec as FS
from agents.analysis import plan_contract as PC

REGION = "watershed, vegetated"
MINUS = "−"


def _ref(ref, kind, value, line=1, outcome=PC.PRODUCED, out=None, **kw):
    return {"ref": ref, "kind": kind, "made": True, "line": line, "value": value,
            "outcome": outcome, "why": None, "out": out or {"kind": kind, "region": REGION},
            **kw}


def _anom(a, normal, lo, hi, units="gC/m^2/season", n=5):
    return {"anomaly": a, "percent": 100.0 * a / normal, "change_units": units,
            "units": units, "normal": normal, "value": normal + a,
            "support": {"n": n, "years": [2016, 2017, 2018, 2019, 2020][:n], "min": lo,
                        "max": hi, "sd": 1.0}}


# a summer GPP anomaly with its normal years, made up
GPP = _anom(-36.9, 310.0, 284.0, 332.0)
CHANGE = "grey band: the range of the normal years 2016 to 2020, each minus their mean"
PERCENT = ("grey band: the range of the normal years 2016 to 2020, each as a percent "
           "change from their mean")
TIMING = {"variants": {"default": {"shift": {"median": -3.0}}},
          "across_settings": {"minimum": {"median_shift": -4.0},
                              "maximum": {"median_shift": -2.0}}}


def _spec(panels):
    return {"id": "f", "panels": panels}


def _measure(fig):
    """Each panel's box, its value labels (white box included), bars and tick
    labels, on the page, as laid out when the figure is saved."""
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        out.append({"box": ax.get_window_extent(rnd).frozen(),
                    "labels": [(t.get_text(), FS._label_extent(t, rnd).frozen())
                               for t in ax.texts
                               if t.get_gid() in (FS.BAR_LABEL_H, FS.BAR_LABEL_V)],
                    "bars": [b.get_window_extent(rnd).frozen() for b in _bars(ax)],
                    "ticks": [x.get_window_extent(rnd).frozen()
                              for x in ax.get_xticklabels() + ax.get_yticklabels()
                              if x.get_text()]})
    return out


@pytest.fixture
def saved(monkeypatch):
    """Each figure as it is saved, laid out, measured just before saving."""
    figs = []
    orig = Figure.savefig

    def save(self, *a, **k):
        self.measured = _measure(self)
        figs.append(self)
        return orig(self, *a, **k)
    monkeypatch.setattr(Figure, "savefig", save)
    return figs


def _labels(ax, gid=FS.BAR_LABEL_V):
    return [t for t in ax.texts if t.get_gid() == gid]


def _bars(ax):
    return [p for p in ax.patches if p.get_zorder() == 2]


def _bands(ax):
    return [p for p in ax.patches if p.get_zorder() == 1]


def test_a_value_is_written_as_a_reader_reads_it():
    assert FS.value_text(-36.86214) == f"{MINUS}36.9"
    assert FS.value_text(6.4) == "6.4"
    assert FS.value_text(-215.42) == f"{MINUS}215"            # whole from 100 up
    assert FS.value_text(3518.4) == "3,518"
    assert FS.value_text(-0.0047152) == f"{MINUS}0.00472"
    assert FS.value_text(float("nan")) == "no value" and FS.value_text(None) == "no value"
    assert "-" not in FS.value_text(-2.74)


def test_an_anomalys_normal_range_is_on_the_scale_of_its_bar():
    r = _ref("P2.O1", "anomaly", GPP)
    band = FS.normal_range(r, {})
    assert band["low"] == pytest.approx(284.0 - 310.0)
    assert band["high"] == pytest.approx(332.0 - 310.0)
    assert band["label"] == CHANGE
    pct = FS.normal_range(r, {"value": "percent"})
    assert pct["low"] == pytest.approx(100 * (284.0 / 310.0 - 1))
    assert pct["high"] == pytest.approx(100 * (332.0 / 310.0 - 1))
    assert pct["label"] == PERCENT
    # a negative mean turns the percents round; the range keeps its order
    neg = FS.normal_range(_ref("x", "anomaly", _anom(1.0, -10.0, -12.0, -8.0)),
                          {"value": "percent"})
    assert (neg["low"], neg["high"]) == (pytest.approx(-20.0), pytest.approx(20.0))


def test_no_range_where_the_record_holds_none():
    diff = _ref("d", "difference", {"difference": 2.0, "change_units": "mm"})
    assert FS.normal_range(diff, {}) is None
    one = dict(GPP, support=dict(GPP["support"], n=1, years=[2016]))
    assert FS.normal_range(_ref("a", "anomaly", one), {}) is None
    assert FS.normal_range(_ref("a", "anomaly", dict(GPP, support=None)), {}) is None
    assert FS.normal_range(_ref("a", "anomaly", dict(GPP, percent=None)),
                           {"value": "percent"}) is None
    per_cell = dict(GPP, normal=np.ones(4), support={"n": 5, "years": [2016],
                                                     "min": np.zeros(4), "max": np.ones(4)})
    assert FS.normal_range(_ref("a", "anomaly", per_cell), {}) is None
    assert FS.normal_range(_ref("t", "growth_peak", TIMING), {}) is None
    # only an anomaly's record holds its normal years: any other kind, even one
    # whose record had such fields, draws no range
    assert FS.normal_range(_ref("d", "difference", dict(GPP)), {}) is None


def test_a_lone_bar_is_narrow_carries_its_value_and_stands_on_its_normal_range(tmp_path,
                                                                             saved):
    o = FS.render_figure(_spec([{"kind": "bars", "outputs": ["P2.O1"]}]),
                         {"P2.O1": _ref("P2.O1", "anomaly", GPP, line=7)}, tmp_path / "f.png")
    assert o["ok"] and o["panels"][0]["records"] == ["L7"]          # its record kept
    fig = saved[-1]
    ax = fig.axes[0]
    bar, band = _bars(ax)[0], _bands(ax)[0]
    lo, hi = ax.get_xlim()
    assert bar.get_width() == FS.BAR_WIDTH and bar.get_width() / (hi - lo) <= 0.25
    assert band.get_width() > bar.get_width()
    assert band.get_y() == pytest.approx(284.0 - 310.0)
    assert band.get_y() + band.get_height() == pytest.approx(332.0 -
                                                            310.0)
    assert ax.get_ylabel() == "gC/m^2/season"                       # its unit kept
    assert [t.get_text() for t in _labels(ax)] == [f"{MINUS}36.9"]
    assert [t.get_text() for t in fig.legends[0].get_texts()] == [CHANGE]


def test_a_negative_value_reads_below_its_bar_inside_the_panel(tmp_path, saved):
    FS.render_figure(_spec([{"kind": "bars", "outputs": ["P2.O1"]}]),
                     {"P2.O1": _ref("P2.O1", "anomaly", GPP)}, tmp_path / "f.png")
    m = saved[-1].measured[0]
    (text, e), = m["labels"]
    box, bar = m["box"], m["bars"][0]
    assert text.startswith(MINUS)
    assert e.y1 <= bar.y0 + 0.5                                       # beyond its end
    assert box.x0 < e.x0 and e.x1 < box.x1 and box.y0 < e.y0 and e.y1 < box.y1


def test_a_difference_carries_its_value_and_no_range(tmp_path, saved):
    d = _ref("P3.O1", "difference", {"difference": 12.5, "change_units": "mm",
                                      "units": "mm"})
    o = FS.render_figure(_spec([{"kind": "bars", "outputs": ["P3.O1"]}]), {"P3.O1": d},
                         tmp_path / "f.png")
    assert o["ok"]
    fig = saved[-1]
    ax = fig.axes[0]
    assert not _bands(ax) and not fig.legends
    assert [t.get_text() for t in _labels(ax)] == ["12.5"] and ax.get_ylabel() == "mm"


def test_several_bars_each_carry_their_value_and_range_and_each_scale_is_named_once(
        tmp_path, saved):
    refs = {"P1.O1": _ref("P1.O1", "anomaly", GPP),
            "P1.O2": _ref("P1.O2", "anomaly", _anom(10.0, 300.0, 280.0, 320.0))}
    o = FS.render_figure(_spec([{"kind": "bars", "outputs": ["P1.O1", "P1.O2"]},
                                {"kind": "bars", "outputs": ["P1.O1", "P1.O2"],
                                 "value": "percent"}]), refs, tmp_path / "f.png")
    assert o["ok"]
    fig = saved[-1]
    a0, a1 = fig.axes[:2]
    assert [t.get_text() for t in _labels(a0)] == [f"{MINUS}36.9", "10"]
    assert [t.get_text() for t in _labels(a1)] == [f"{MINUS}11.9", "3.33"]
    assert len(_bands(a0)) == len(_bands(a1)) == 2
    assert a1.get_ylabel() == "percent"
    assert [t.get_text() for t in fig.legends[0].get_texts()] == [CHANGE, PERCENT]


def test_a_timing_bars_range_is_named_as_the_range_across_settings(tmp_path, saved):
    o = FS.render_figure(_spec([{"kind": "bars", "outputs": ["P4.O1"]}]),
                         {"P4.O1": _ref("P4.O1", "growth_peak", TIMING)}, tmp_path / "f.png")
    assert o["ok"]
    fig = saved[-1]
    assert [t.get_text() for t in fig.legends[0].get_texts()] == [FS.SETTINGS_RANGE]
    assert [t.get_text() for t in _labels(fig.axes[0])] == [f"{MINUS}3"]
    assert fig.axes[0].get_ylabel() == "days"


def test_band_means_of_long_negative_bars_keep_their_labels_inside_the_panel(tmp_path,
                                                                          saved):
    parts = {"below 1000 m": {"mean": 11.2}, "1000-1500 m": {"mean": -52.3},
             "1500-2000 m": {"mean": -57.8}, "2000 m and above": {"mean": -38.1}}
    es = _ref("P4.O3", "elevation_split", {"parts": parts},
              out={"kind": "elevation_split", "region": REGION, "inputs": ["P4.O1"]},
              input_units={"P4.O1": "gC/m^2/yr"})
    o = FS.render_figure(_spec([{"kind": "bars", "outputs": ["P4.O3"]}]), {"P4.O3": es},
                         tmp_path / "f.png")
    assert o["ok"]
    m = saved[-1].measured[0]
    assert [t for t, _ in m["labels"]] == ["11.2", f"{MINUS}52.3", f"{MINUS}57.8",
                                           f"{MINUS}38.1"]
    box = m["box"]
    for t, e in m["labels"]:
        assert box.x0 + 2 <= e.x0 and e.x1 <= box.x1 - 2, t
        assert not any(e.x0 < n.x1 and n.x0 < e.x1 and e.y0 < n.y1 and n.y0 < e.y1
                       for n in m["ticks"]), t


def test_a_figure_without_bars_is_laid_out_as_before(tmp_path, monkeypatch):
    def fail(fig, tries=6):
        raise AssertionError("labels fitted in a figure without bars")
    monkeypatch.setattr(FS, "_fit_labels", fail)
    lab = ["2016-01-01", "2016-01-02"]
    s = _ref("P3.O1", "series", {"labels": lab, "values": np.array([1.0, 2.0]),
                                 "units": "mm", "step": "day"})
    o = FS.render_figure(_spec([{"kind": "series", "outputs": ["P3.O1"]}]), {"P3.O1": s},
                         tmp_path / "f.png")
    assert o["ok"]
