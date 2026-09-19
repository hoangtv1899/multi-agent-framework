"""mcp/elm-mcp/src/_vendor/ must stay byte-identical to the framework.

elm-mcp is published as its own repository, and it carries verbatim copies of
the six self-contained framework modules its server needs so that a standalone
checkout works with nothing beside it.  Inside this repository those copies
never execute: framework_path puts the framework ahead of them on sys.path.

The usual objection to a vendored copy is that it forks and then drifts and
nobody notices.  That is what this file is for.  Edit src/core/keyset.py and
this test fails, naming the file and the command that refreshes it.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ELM = ROOT / "mcp" / "elm-mcp"
SYNC = ELM / "scripts" / "sync_vendor.py"
VENDOR = ELM / "src" / "_vendor"

pytestmark = pytest.mark.skipif(
    not SYNC.is_file(), reason="elm-mcp is not present in this checkout"
)


def _pairs():
    """(framework file, vendored copy) for everything sync_vendor tracks."""
    sys.path.insert(0, str(ELM / "scripts"))
    import importlib.util

    spec = importlib.util.spec_from_file_location("sync_vendor", SYNC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [(ROOT / src, VENDOR / dest) for src, dest in mod.FILES.items()]


def test_vendor_directory_exists():
    """Without it, a standalone elm-mcp checkout cannot import anything."""
    assert VENDOR.is_dir(), (
        f"{VENDOR} is missing. Restore it with:\n"
        f"    python3 {SYNC.relative_to(ROOT)} --write"
    )


@pytest.mark.parametrize(
    "src,dest", _pairs(), ids=lambda p: p.name if isinstance(p, Path) else str(p)
)
def test_vendored_copy_matches_framework(src, dest):
    assert src.is_file(), f"the framework no longer has {src.relative_to(ROOT)}"
    assert dest.is_file(), (
        f"{dest.relative_to(ROOT)} is not vendored. Refresh with:\n"
        f"    python3 {SYNC.relative_to(ROOT)} --write"
    )
    assert dest.read_bytes() == src.read_bytes(), (
        f"{dest.relative_to(ROOT)} has drifted from {src.relative_to(ROOT)}.\n"
        f"The vendored copy is what a standalone elm-mcp checkout runs, so a "
        f"divergence means the published server behaves differently from this "
        f"one.\nRefresh with:\n    python3 {SYNC.relative_to(ROOT)} --write"
    )


def test_sync_check_passes():
    """The script's own --check agrees, so CI can call it directly."""
    r = subprocess.run(
        [sys.executable, str(SYNC), "--check", "--framework", str(ROOT)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"sync_vendor --check failed:\n{r.stdout}\n{r.stderr}"


def test_exp_manager_base_is_not_vendored():
    """The base class ELM and PFLOTRAN share must never be copied.

    A second copy would fork execute_plan, the stage names and the run-state
    contract between the two models, and nothing would report the divergence.
    Code that needs it calls ensure_on_path(require=True) instead.
    """
    assert not (VENDOR / "core" / "exp_manager_base.py").exists(), (
        "core/exp_manager_base.py has been vendored into elm-mcp. It is the "
        "base class ELM and PFLOTRAN share; vendoring forks a contract two "
        "models depend on. Remove it and use ensure_on_path(require=True)."
    )
