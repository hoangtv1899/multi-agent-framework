"""What the framework TELLS the analyzer about its own initialization.

The analyzer is bound by the limitations catalogue and the assumptions ledger —
it may not argue past them, by design. So when both asserted "no multi-year
spin-up: slow-state fluxes are transient" on a run that had warm-started from a
spun-up CONUS restart precisely to avoid spin-up, the analyzer dutifully
recommended three to five spin-up years. It was not being stubborn; it was
repeating what it had been told.

Initialization is ONE state, not a checklist. These pin that the caveat
describes the run that actually happened.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.columns_to_plan import build_ledger          # noqa: E402
from core.limitations import select_limitations        # noqa: E402


def _init_caveats(**kw):
    cfg = select_limitations(**kw)["configuration"]
    return [c["caveat"] for c in cfg]


def _has(caveats, *words):
    return any(all(w in c for w in words) for c in caveats)


class TestLimitationsFollowTheState:
    def test_cold_start_still_warns_about_spin_up(self):
        c = _init_caveats(warm_start=False)
        assert _has(c, "non-equilibrated storage")

    def test_warm_conus_with_donor_soil_does_not(self):
        """The whole point of the warm start. Telling the analyzer this run has
        non-equilibrated storage is what produced the spin-up recommendation."""
        c = _init_caveats(warm_start=True, warm_source="conus", soil_source="conus")
        assert not _has(c, "non-equilibrated storage")
        assert _has(c, "inherited from the spun-up CONUS")

    def test_it_still_says_the_state_is_not_free(self):
        """Not equilibrated to THIS period — the honest residual, with a way to
        size it rather than a blanket warning."""
        c = _init_caveats(warm_start=True, warm_source="conus", soil_source="conus")
        assert _has(c, "closure")

    def test_overriding_the_donor_soil_gets_the_strong_caveat(self):
        """This is the configuration that produced drainage above precipitation
        — it must NOT be described as consistent."""
        c = _init_caveats(warm_start=True, warm_source="conus", soil_source="ssurgo")
        assert _has(c, "OVERRIDES the donor's soil")
        assert _has(c, "unusable")

    def test_fan_warm_start_keeps_its_own_caveat(self):
        c = _init_caveats(warm_start=True, warm_source="fan", soil_source="ssurgo")
        assert _has(c, "Fan (2013)")

    def test_a_genuinely_spun_up_run_gets_no_initialization_caveat(self):
        c = _init_caveats(warm_start=True, warm_source="conus",
                          soil_source="conus", spinup_years=5)
        assert not _has(c, "non-equilibrated storage")
        assert not _has(c, "inherited from the spun-up CONUS")

    def test_the_two_states_are_never_asserted_together(self):
        """'no spin-up' and 'warm started from a spun-up restart' cannot both
        be true; emitting both is what made the recommendation incoherent."""
        for kw in (dict(warm_start=False),
                   dict(warm_start=True, warm_source="conus", soil_source="conus"),
                   dict(warm_start=True, warm_source="conus", soil_source="ssurgo")):
            c = _init_caveats(**kw)
            assert not (_has(c, "non-equilibrated storage")
                        and _has(c, "inherited from the spun-up CONUS"))


class TestLedgerFollowsTheState:
    COLS = [{"id": "col_01"}, {"id": "col_02"}]

    def _spinup_row(self, finidat_map):
        rows = build_ledger(self.COLS, 1995, 1995, "native", "template",
                            "1995-1995", finidat_map=finidat_map)
        return next(r for r in rows if r["parameter"] == "spin-up")

    def test_warm_run_is_not_told_its_terms_are_transient(self):
        row = self._spinup_row({"col_01": {"source": "conus"},
                                "col_02": {"source": "conus"}})
        assert ">=3 spin-up years" not in row["note"]
        assert "inherited from the CONUS spin-up" in row["note"]

    def test_cold_run_still_is(self):
        row = self._spinup_row(None)
        assert ">=3 spin-up years" in row["note"]

    def test_the_value_says_which_it_was(self):
        assert "warm" in self._spinup_row(
            {"col_01": {"source": "conus"}})["value"]
        assert self._spinup_row(None)["value"] == "none"


class TestColumnsJsonMatchesTheRun:
    """columns.json is what analyze_run and validate_run read for coordinates
    and soil. The warm start SNAPS each column to its donor gridcell and hands
    it that cell's soil, so a file written before step 0b describes a plan the
    run then departed from — every column of the 2026-07-28 run sat 200–700 m
    from where columns.json claimed, and the spatial maps and nearest-station
    matching inherited that error."""

    def test_it_is_written_after_the_warm_start(self):
        src = (ROOT / "src" / "core" / "elm_exp_manager.py").read_text()
        warm = src.index("finidat_map = self._warmstart")
        soil = src.index("self._attach_donor_soil")
        write = src.index('"columns.json").write_text')
        assert warm < write, "columns.json written before coordinates are snapped"
        assert soil < write, "columns.json written before the run's soil is known"

    def test_and_before_the_design_figure_and_the_plan(self):
        src = (ROOT / "src" / "core" / "elm_exp_manager.py").read_text()
        write = src.index('"columns.json").write_text')
        assert write < src.index("png = exp.plot_columns")
        assert write < src.index("executable = columns_to_elm_plan")

    def test_it_is_only_written_once(self):
        """Two write sites would let one of them drift back before step 0b."""
        src = (ROOT / "src" / "core" / "elm_exp_manager.py").read_text()
        assert src.count('"columns.json").write_text') == 2   # input_dir + run_dir
