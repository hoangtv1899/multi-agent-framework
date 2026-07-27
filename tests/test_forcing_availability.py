"""Offline tests for forcing availability (no filesystem beyond tmp_path).

The forcing window is the one hard constraint on a request: without forcing
there is no run at all, whereas a missing observation only costs a comparison.
It used to be a sentence in the Reception prompt, and that sentence had drifted
five years from the filesystem. These tests pin the behaviours that make the
disk the authority — especially the two failure modes that would put a user
back where they started: inventing a window when the tree is unreadable, and
offering a range with a hole in it.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.forcing_availability import (clamp, contiguous_span,  # noqa: E402
                                       month_counts, nldas_dir,
                                       render_forcing_facts, scan_nldas)


def _tree(root: Path, years, months=12, strays=()):
    """Build a fake NLDAS tree: root/atm/datm7/NLDAS/clmforc.nldas.YYYY-MM.nc"""
    d = root / "atm" / "datm7" / "NLDAS"
    d.mkdir(parents=True, exist_ok=True)
    for y in years:
        for m in range(1, months + 1):
            (d / f"clmforc.nldas.{y}-{m:02d}.nc").touch()
    for name in strays:
        (d / name).touch()
    return d


# ── pure arithmetic, no disk ────────────────────────────────────────────────
class TestContiguousSpan:
    def test_unbroken_run_is_the_whole_range(self):
        assert contiguous_span([1979, 1980, 1981]) == [1979, 1980, 1981]

    def test_a_gap_year_truncates_the_offer(self):
        """A hole mid-record aborts the simulation partway rather than at
        submit time, so first..last is not a safe thing to advertise."""
        assert contiguous_span([1979, 1980, 1990, 1991, 1992]) == [1990, 1991, 1992]

    def test_longest_run_wins_not_the_earliest(self):
        assert contiguous_span([1970, 1980, 1981, 1982]) == [1980, 1981, 1982]

    def test_duplicates_and_disorder_do_not_matter(self):
        assert contiguous_span([1981, 1979, 1980, 1981]) == [1979, 1980, 1981]

    def test_empty_and_single(self):
        assert contiguous_span([]) == []
        assert contiguous_span([1995]) == [1995]


# ── reading the tree ────────────────────────────────────────────────────────
class TestScan:
    def test_counts_months_per_year(self, tmp_path):
        _tree(tmp_path, [1979, 1980])
        assert month_counts(nldas_dir(str(tmp_path))) == {1979: 12, 1980: 12}

    def test_a_partial_year_is_not_runnable(self, tmp_path):
        d = _tree(tmp_path, [1979])
        for m in range(1, 9):                       # 2024: 8 of 12 months
            (d / f"clmforc.nldas.2024-{m:02d}.nc").touch()
        w = scan_nldas(str(tmp_path))
        assert w["yr_last"] == 1979
        assert w["partial_years"] == {2024: 8}

    def test_stray_filenames_do_not_invent_years(self, tmp_path):
        """The real tree carries clmforc.nldas.0016-06.nc; parsed naively that
        is a year 16 AD and the advertised window starts two millennia early."""
        _tree(tmp_path, [1979, 1980],
              strays=["README", "backup", "clmforc.nldas.nc", "other.1995-01.nc"])
        w = scan_nldas(str(tmp_path))
        assert (w["yr_first"], w["yr_last"]) == (1979, 1980)

    def test_missing_tree_reports_absence_not_an_empty_range(self, tmp_path):
        w = scan_nldas(str(tmp_path / "nothing-here"))
        assert w["exists"] is False
        assert w["yr_first"] is None and w["n_years"] == 0

    def test_env_var_overrides_the_default_root(self, tmp_path, monkeypatch):
        _tree(tmp_path, [2000])
        monkeypatch.setenv("DIN_LOC_ROOT", str(tmp_path))
        assert scan_nldas()["yr_first"] == 2000


# ── what Reception is told ──────────────────────────────────────────────────
class TestRendering:
    def test_the_window_reaches_the_prompt(self, tmp_path):
        _tree(tmp_path, range(1979, 1985))
        txt = render_forcing_facts(str(tmp_path))
        assert "1979-1984" in txt
        assert "6 complete years" in txt

    def test_no_tree_means_no_years_are_claimed(self, tmp_path):
        """Inventing a range is the one error here that costs a queue slot."""
        txt = render_forcing_facts(str(tmp_path / "absent"))
        assert "UNKNOWN" in txt
        assert "1979" not in txt and "2023" not in txt

    def test_the_dead_qian_fallback_is_not_offered(self, tmp_path):
        """DATM_MODE is pinned to CLMMOSARTTEST in elm_wrapper, so a promised
        Qian fallback is forcing the code cannot produce."""
        _tree(tmp_path, [1995])
        txt = render_forcing_facts(str(tmp_path))
        assert "Qian" not in txt
        assert "no fallback" in txt.lower()


# ── clamping a request ──────────────────────────────────────────────────────
class TestClamp:
    @pytest.fixture
    def root(self, tmp_path):
        _tree(tmp_path, range(1990, 2001))           # 1990-2000
        return str(tmp_path)

    def test_a_period_inside_the_window_is_untouched(self, root):
        assert clamp(1995, 1996, root) == (1995, 1996, None)

    def test_an_overhanging_period_is_trimmed_with_a_note(self, root):
        s, e, note = clamp(1985, 1995, root)
        assert (s, e) == (1990, 1995)
        assert "clamped" in note

    def test_a_wholly_outside_period_is_refused_not_relocated(self, root):
        """Silently moving a 1970s study into the 1990s answers a different
        question than the one asked."""
        s, e, note = clamp(1970, 1975, root)
        assert s is None and e is None
        assert "cannot run" in note

    def test_unverifiable_forcing_passes_through_with_a_warning(self, tmp_path):
        s, e, note = clamp(1995, 1995, str(tmp_path / "absent"))
        assert (s, e) == (1995, 1995)
        assert "unverified" in note
