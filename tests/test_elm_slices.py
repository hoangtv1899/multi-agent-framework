#!/usr/bin/env python3
"""The slice seam: continue a built ELM case in place, window by window.

    pytest tests/test_elm_slices.py -v

The walk-through-the-year coupling runs ONE case per column for the whole
year, advancing it a window at a time (CONTINUE_RUN + a day-count stop) and
stamping the restart between windows. These tests cover the seam's plain
code against a FAKE case directory — a stub ./xmlchange that records its
arguments, and hand-written rpointer files — because the real xmlchange
needs a CIME checkout and the real run needs a compute node. The
slices-match-the-straight-run proof is a separate one-column SLURM job,
submitted only with permission.
"""
import os
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "mcp" / "elm-mcp" / "src"))

import elm_wrapper as ew                               # noqa: E402


def _fake_case(tmp_path):
    """A case dir whose ./xmlchange appends each call to xml.log and whose
    ./preview_namelists leaves a marker — the regeneration step that turns
    CONTINUE_RUN into the run-dir namelists the executable actually reads."""
    case = tmp_path / "case"
    (case / "run").mkdir(parents=True)
    stub = case / "xmlchange"
    stub.write_text("#!/bin/bash\necho \"$@\" >> xml.log\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    pv = case / "preview_namelists"
    pv.write_text("#!/bin/bash\ntouch namelists_regenerated\n")
    pv.chmod(pv.stat().st_mode | stat.S_IEXEC)
    return case


class TestConfigureContinuation:
    def test_the_five_keys_in_order(self, tmp_path):
        case = _fake_case(tmp_path)
        ew.configure_continuation(case, stop_n=7)
        got = (case / "xml.log").read_text().splitlines()
        assert got == ["CONTINUE_RUN=TRUE", "STOP_N=7", "STOP_OPTION=ndays",
                       "REST_N=7", "REST_OPTION=ndays"]

    def test_the_namelists_are_regenerated(self, tmp_path):
        """xmlchange edits env_run.xml only; the executable reads the run-dir
        namelists. Without regeneration the next slice silently re-runs the
        previous window (measured: job 774959)."""
        case = _fake_case(tmp_path)
        ew.configure_continuation(case, stop_n=7)
        assert (case / "namelists_regenerated").exists()

    def test_a_monthly_window(self, tmp_path):
        case = _fake_case(tmp_path)
        ew.configure_continuation(case, stop_n=1, stop_option="nmonths")
        got = (case / "xml.log").read_text()
        assert "STOP_OPTION=nmonths" in got and "REST_OPTION=nmonths" in got

    def test_a_non_integer_window_is_refused(self, tmp_path):
        with pytest.raises(ValueError):
            ew.configure_continuation(_fake_case(tmp_path), stop_n="a week")


class TestLatestRestart:
    def test_follows_the_pointer_not_the_newest_file(self, tmp_path):
        case = _fake_case(tmp_path)
        rd = case / "run"
        old = rd / "case.elm.r.1981-01-08-00000.nc"
        new = rd / "case.elm.r.1981-01-15-00000.nc"
        old.write_bytes(b"old")
        new.write_bytes(b"new")
        # the pointer says OLD — perhaps a slice was rerun — and the pointer
        # wins, because it is what ELM itself will read
        (rd / "rpointer.lnd").write_text(old.name + "\n")
        assert ew.latest_restart(case) == old

    def test_no_pointer_is_refused_with_the_reason(self, tmp_path):
        with pytest.raises(ValueError, match="rpointer.lnd"):
            ew.latest_restart(_fake_case(tmp_path))

    def test_a_dangling_pointer_names_the_missing_file(self, tmp_path):
        case = _fake_case(tmp_path)
        (case / "run" / "rpointer.lnd").write_text("gone.elm.r.nc\n")
        with pytest.raises(ValueError, match="gone.elm.r.nc"):
            ew.latest_restart(case)


class TestTheBuilderCarriesTheWindow:
    def test_coupler_stop_option_reaches_runtime_config(self):
        """A coupler entry carrying STOP_OPTION/REST_* wins over the config
        defaults — read from the source, since building a real case needs
        CIME. The assembly is three dict lookups; this pins that they exist
        and prefer the coupler."""
        import inspect
        import elm_experiment_builder as eb
        src = inspect.getsource(eb.ELMExperimentBuilder._build_one) \
            if hasattr(eb, "ELMExperimentBuilder") else \
            inspect.getsource(eb)
        for key in ("STOP_OPTION", "REST_N", "REST_OPTION"):
            assert f"coupler.get(\n                '{key}'" in src \
                or f"coupler.get('{key}'" in src, key
