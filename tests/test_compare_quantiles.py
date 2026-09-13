"""The ELM water-table distribution's quartiles interpolate (2026-09-13).

v[n // 2] is the upper-middle VALUE for an even n. The Naches run of
2026-08-14 reported a median water table of 12.087 m over 18 columns; that
is the 10th sorted value, and the median of the 18 is 10.68 m. The number
was cited, audited as cited, re-worded twice, and would have reached a
slide. No audit of citations catches a computation; only a recomputation
does, which is what this test is.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _elm_water_table():
    spec = importlib.util.spec_from_file_location(
        "elm_compare_water_table",
        ROOT / "mcp" / "elm-mcp" / "src" / "compare" / "water_table.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_an_even_count_median_is_the_mean_of_the_middle_pair():
    wt = _elm_water_table()
    naches = [4.391, 4.428, 4.773, 5.002, 5.466, 5.610, 5.694, 7.791, 9.278,
              12.087, 21.460, 25.782, 39.620, 45.331, 58.915, 65.490, 66.755,
              70.238]
    q = wt._quantiles(naches)
    assert q["n"] == 18 and q["median"] == 10.683      # (9.278 + 12.087) / 2, not 12.087
    assert q["min"] == 4.391 and q["max"] == 70.238
    assert q["p25"] < q["median"] < q["p75"]


def test_an_odd_count_median_is_the_middle_value_and_none_is_skipped():
    wt = _elm_water_table()
    q = wt._quantiles([3.0, None, 1.0, 2.0])
    assert (q["n"], q["min"], q["median"], q["max"]) == (3, 1.0, 2.0, 3.0)
    assert wt._quantiles([None]) is None
