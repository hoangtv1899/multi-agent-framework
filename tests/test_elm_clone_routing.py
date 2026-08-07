"""Smoke test for the cloning wrapper. Doesn't actually run case.build —
just verifies the wrapper accepts ref_case_dir parameter and routes
to _clone_case() correctly."""
import sys
from unittest.mock import patch, MagicMock
sys.path.insert(0, "src")

from elm_wrapper import GeneratedELMAgent

def test_prepare_case_no_ref_calls_create():
    """Without ref_case_dir, prepare_case takes the fresh-build path."""
    w = GeneratedELMAgent.__new__(GeneratedELMAgent)  # bypass __init__
    w.runtime_config = {}
    w.case_suffix = "test"
    w.case_dir = None
    w.case_name = None

    with patch.object(w, '_create_case') as create, \
         patch.object(w, '_configure_case') as cfg, \
         patch.object(w, '_write_namelists'), \
         patch.object(w, '_build_case') as build:
        w.case_dir = "/tmp/fake_case"
        w.prepare_case()
        create.assert_called_once()
        build.assert_called_once()
        cfg.assert_called_once_with()  # no runtime_only kwarg

def test_prepare_case_with_ref_calls_clone():
    """With ref_case_dir, prepare_case takes the clone path."""
    w = GeneratedELMAgent.__new__(GeneratedELMAgent)
    w.runtime_config = {}
    w.case_suffix = "test"
    w.case_dir = None
    w.case_name = None

    with patch.object(w, '_clone_case') as clone, \
         patch.object(w, '_configure_case') as cfg, \
         patch.object(w, '_write_namelists'), \
         patch.object(w, '_setup_case'), \
         patch.object(w, '_build_case') as build:
        w.case_dir = "/tmp/fake_case"
        w.prepare_case(ref_case_dir="/tmp/ref_case")
        clone.assert_called_once_with("/tmp/ref_case")
        cfg.assert_called_once_with(runtime_only=True)
        build.assert_not_called()

# ─────────────────────────────────────────────────────────────────────
# RUNDIR inheritance — the 2026-07-25..08-03 clobber
# ─────────────────────────────────────────────────────────────────────
# create_clone copies env_run.xml verbatim and builds the clone's namelists
# before prepare_case() reaches _configure_case. Whatever RUNDIR the reference
# left in that file is where the clone writes during that window. An absolute
# literal sent it into the REFERENCE's run dir; these pin the two properties
# that keep it out.

def _configure(runtime_only=False, case_dir="/scratch/E3SMv3/CASE_A"):
    """Run _configure_case against a recording _xmlchange."""
    from pathlib import Path
    w = GeneratedELMAgent.__new__(GeneratedELMAgent)
    w.runtime_config = {}
    w.case_dir = Path(case_dir)
    calls = {}
    with patch.object(w, '_xmlchange', side_effect=lambda k, v: calls.__setitem__(k, v)):
        w._configure_case(runtime_only=runtime_only)
    return calls


def test_rundir_is_symbolic_not_an_absolute_case_path():
    """RUNDIR must survive being copied into another case.

    The failing value was str(case_dir / "run") — correct for the case that
    wrote it and wrong for every clone that inherits it.
    """
    for runtime_only in (False, True):
        rundir = _configure(runtime_only=runtime_only)['RUNDIR']
        assert '$CASEROOT' in rundir, (
            f"RUNDIR={rundir!r} — a clone inherits this verbatim and would "
            f"write its namelists into the reference's run directory")
        assert 'CASE_A' not in rundir


def test_inherited_rundir_resolves_to_the_clone_not_the_reference():
    """The bug, stated as the property it broke.

    Mimics what create_clone does: copy the reference's raw RUNDIR into a new
    case, then resolve $CASEROOT against THAT case.
    """
    raw = _configure(case_dir="/scratch/E3SMv3/REF")['RUNDIR']
    resolved = raw.replace('$CASEROOT', '/scratch/E3SMv3/CLONE')
    assert resolved == '/scratch/E3SMv3/CLONE/run'
    assert '/REF/' not in resolved


def test_exeroot_stays_absolute_so_keepexe_clones_find_the_exe():
    """The deliberate asymmetry: EXEROOT must NOT be made symbolic.

    --keepexe works because the clone inherits an absolute path back to the
    reference's build. $CASEROOT/build would send it to a directory that was
    never compiled.
    """
    calls = _configure()
    assert calls['EXEROOT'] == '/scratch/E3SMv3/CASE_A/build'
    assert '$' not in calls['EXEROOT']
    # Clones must not touch it at all — env_build.xml is inherited.
    assert 'EXEROOT' not in _configure(runtime_only=True)
