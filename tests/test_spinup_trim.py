"""How much of the front of a run is thrown away, and on what grounds.

The extractor drops the start-up transient before any statistic, and for a long
time it dropped exactly fourteen days from everything. Fourteen days is the
right number for a WARM start: the column inherits roughly correct storage from
the CONUS spin-up and spends a fortnight correcting it.

It is the wrong number for a COLD start by more than an order of magnitude. The
model builds storage from its own defaults, which is a filling rather than a
correction. Measured on the 4-column 1995-2004 Naches conceptual sweep, QDRAI
was exactly 0.0 for the whole of 1995 and reached a steady 445.9 mm/yr only from
1996, with the water table still travelling 8.80 -> 5.87 -> 4.09 m across the
first three years. A fourteen-day trim leaves every bit of that inside the
averages, and nothing in the record said so.

These pin the one rule that now covers both: the trim is read off how the run
was STARTED, the same way for a site study and a conceptual sweep, and a trim
that could not be applied is reported rather than silently skipped.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp" / "elm-mcp" / "src"))

from extract import (COLD_START_DAYS, MIN_KEPT_DAYS,        # noqa: E402
                     SPINUP_DAYS, _drop_spinup,
                     initialisation, resolve_spinup)
from column_rows import spinup_dropped                      # noqa: E402


def _run(tmp_path, cases, columns=None):
    """A run directory with just enough in it to be classified."""
    (tmp_path / "01_inputs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "01_inputs" / "case_inputs.json").write_text(json.dumps(cases))
    if columns is not None:
        (tmp_path / "columns.json").write_text(json.dumps(columns))
    return tmp_path


def _warm(name="col_01"):
    return {"case_name": name,
            "runtime_config": {"FINIDAT": f"/warmstart/finidat_{name}.nc"}}


def _cold(name="col_01"):
    return {"case_name": name, "runtime_config": {"FSURDAT": "/surf.nc"}}


class TestTheTrimFollowsHowTheRunStarted:

    def test_a_warm_run_keeps_the_fortnight(self, tmp_path):
        days, basis, _ = resolve_spinup(_run(tmp_path, [_warm()]))
        assert (days, basis) == (SPINUP_DAYS, "warm")

    def test_a_cold_run_gets_a_year(self, tmp_path):
        """The regression this file exists for: 14 days on a cold start put an
        entirely transient 1995 into the averages of a 10-year sweep."""
        days, basis, _ = resolve_spinup(_run(tmp_path, [_cold()]))
        assert (days, basis) == (COLD_START_DAYS, "cold")
        assert days > SPINUP_DAYS * 20

    def test_one_rule_covers_the_site_study_and_the_sweep(self, tmp_path):
        """Nothing here reads the archetype. A conceptual sweep that warm-starts
        gets the warm trim, and a site study that cold-starts gets the cold one
        — because the question the trim answers is about initialisation, not
        about what kind of study the column belongs to."""
        site = _run(tmp_path / "site", [_cold("col_01")])
        sweep = _run(tmp_path / "sweep", [_warm("col_01")])
        assert resolve_spinup(site)[0] == COLD_START_DAYS
        assert resolve_spinup(sweep)[0] == SPINUP_DAYS

    def test_a_mixed_ensemble_takes_the_longer_cut(self, tmp_path):
        """Trimming the warm columns by a year costs record. Leaving the cold
        ones untrimmed puts a filling transient into numbers that are then
        compared column against column, which is worse."""
        days, basis, why = resolve_spinup(
            _run(tmp_path, [_warm("col_01"), _cold("col_02")]))
        assert (days, basis) == (COLD_START_DAYS, "mixed")
        assert "comparable" in why

    def test_the_caller_still_wins(self, tmp_path):
        """Someone who has measured the transient on the run in front of them
        is not overruled by a default."""
        assert resolve_spinup(_run(tmp_path, [_cold()]), 30)[0] == 30

    def test_an_unreadable_run_says_so_rather_than_guessing_cold(self, tmp_path):
        """Assuming cold would throw away a year of a warm run on no evidence.
        The reason string is what makes the assumption visible."""
        days, basis, why = resolve_spinup(tmp_path)
        assert (days, basis) == (SPINUP_DAYS, "unknown")
        assert "assumed" in why


class TestWhatTheCaseRanBeatsWhatTheDesignAsked:

    def test_finidat_decides_not_the_flag(self, tmp_path):
        """columns.json carries the REQUEST; case_inputs.json carries what was
        built. They were out of step for a whole afternoon on 2026-08-15 — the
        cold-start flag never crossed the MCP wire, so every record said cold
        while every case carried a restart file. Reading the flag would have
        trimmed 14 days off runs that needed a year, and agreed with itself."""
        rd = _run(tmp_path, [_warm()], columns=[{"id": "col_01",
                                                 "warm_start": False}])
        assert initialisation(rd) == "warm"

    def test_the_flag_is_the_fallback_when_no_case_was_built(self, tmp_path):
        (tmp_path / "columns.json").write_text(
            json.dumps([{"id": "col_01", "warm_start": False}]))
        assert initialisation(tmp_path) == "cold"


class TestTheTrimNeverConsumesTheRun:
    """Refusing only the case that leaves literally nothing was not enough once
    the cold trim went to a year. The 4-column 1995 fixture runs 1 Jan to 1 Jan:
    a 365-day window left the single timestep at 1996-01-01, the partial-day
    drop removed that, and four columns that had run perfectly well came back as
    `no requested variable had values`. Measured, not imagined — it is what the
    first re-extraction under the new rule produced.
    """

    def _ds(self, n_days, start="1995-01-01"):
        xr = pytest.importorskip("xarray")
        np_ = pytest.importorskip("numpy")
        pd = pytest.importorskip("pandas")
        # 3-hourly, the native history cadence.
        t = pd.date_range(start, periods=n_days * 8, freq="3h")
        return xr.Dataset({"QOVER": ("time", np_.ones(len(t)))},
                          coords={"time": t})

    def test_a_year_and_a_day_is_not_trimmed_to_a_stub(self):
        """The exact fixture shape. One extra timestep past the window is not
        a run, and trading a whole year of record for it is not equilibration."""
        ds, dropped = _drop_spinup(self._ds(365) , 365)
        assert dropped == 0
        assert ds.sizes["time"] == 365 * 8, "the record was consumed"

    def test_a_long_enough_remainder_is_still_trimmed(self):
        ds, dropped = _drop_spinup(self._ds(365 + MIN_KEPT_DAYS + 5), 365)
        assert dropped == 365 * 8
        assert ds.sizes["time"] == (MIN_KEPT_DAYS + 5) * 8

    def test_the_warm_case_is_untouched_by_the_guard(self):
        """A site run trims a fortnight off a year and keeps 351 days. The
        guard must not have quietly changed what those runs do."""
        ds, dropped = _drop_spinup(self._ds(365), SPINUP_DAYS)
        assert dropped == SPINUP_DAYS * 8
        assert ds.sizes["time"] == (365 - SPINUP_DAYS) * 8


class TestATrimThatDidNotFitIsAFinding:
    """`_drop_spinup` keeps a record shorter than the window rather than
    returning an empty series — a short run is a finding, an empty one is a bug.
    That made "0 timesteps dropped" ambiguous between "no trim was wanted" and
    "a year was wanted, the run is 351 days long, and the whole transient is
    still in here". The second is the one-year cold run, and it read as the
    first: spinup_dropped returned {} and the package said nothing at all.
    """

    def _extracted(self, tmp_path, days, dropped, requested):
        (tmp_path / "03_results").mkdir(parents=True, exist_ok=True)
        (tmp_path / "03_results" / "extracted.json").write_text(json.dumps({
            "metadata": {
                "spinup_days": days,
                "spinup_basis": "cold",
                "spinup_reason": "cold-start transient",
                "columns": {"col_01": {
                    "status": "ok",
                    "record_start": "1995-01-01",
                    "date_range": ["1995-01-15", "1995-12-17"],
                    "n_timesteps_dropped_spinup": dropped,
                    "spinup_days_requested": requested}}},
            "data": {}}))
        return tmp_path

    def test_a_window_longer_than_the_record_is_reported(self, tmp_path):
        got = spinup_dropped(self._extracted(tmp_path, 365, 0, 365))
        assert got, "the run that needed the trim most reported nothing"
        assert got["applied"] is False
        assert got["not_trimmed"] == ["col_01"]
        assert "still in the series" in got["note"]

    def test_a_trim_that_worked_says_applied(self, tmp_path):
        got = spinup_dropped(self._extracted(tmp_path, 365, 365, 365))
        assert got["applied"] is True
        assert "not_trimmed" not in got

    def test_the_reason_comes_from_the_run_not_a_constant(self, tmp_path):
        """It used to be a hardcoded sentence about CONUS warm starts, printed
        under cold-start records that had never seen a restart file."""
        got = spinup_dropped(self._extracted(tmp_path, 365, 365, 365))
        assert got["basis"] == "cold"
        assert "CONUS" not in got["reason"]
