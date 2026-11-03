"""The checkout's local changes as one patch a person can apply (2026-11-03).

    pytest tests/test_source_patch.py -v

The design page says the local changes are saved as a patch to apply to the
commit it names. The patch used to hold each submodule's diff with paths
inside the submodule, and no untracked file, so `git apply` stopped at the
first submodule file and the rebuilt code had a different fingerprint. Here
a checkout with a submodule edit, a top-level edit and untracked files is
written as a patch, applied to a clean clone with its submodules checked
out, and the two fingerprints compared. Git is whatever the machine has
(Compy's login node: 1.8.3).
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp" / "elm-mcp" / "src"))

import domain_case as DC                                    # noqa: E402


def _git(cwd, *a, ok=(0,)):
    p = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t",
                        "-c", "protocol.file.allow=always", *a],
                       cwd=str(cwd), capture_output=True, text=True)
    assert p.returncode in ok, (a, p.stdout, p.stderr)
    return p.stdout


@pytest.fixture
def checkout(tmp_path):
    """An E3SM-like checkout: a top-level file and a cime submodule, both
    committed, then edited, with an untracked source file, an untracked
    binary file and an untracked empty file beside them."""
    sub = tmp_path / "cime_origin"
    sub.mkdir()
    _git(sub, "init", "-q")
    (sub / "config.xml").write_text("<machine>compy</machine>\n")
    _git(sub, "add", "config.xml")
    _git(sub, "commit", "-q", "-m", "cime")
    top = tmp_path / "e3sm"
    top.mkdir()
    _git(top, "init", "-q")
    (top / "Mod.F90").write_text("x = 1\n")
    _git(top, "add", "Mod.F90")
    _git(top, "submodule", "add", "-q", str(sub), "cime")
    _git(top, "commit", "-q", "-m", "e3sm")
    # the local changes a build would compile
    (top / "Mod.F90").write_text("x = 2\n")
    (top / "cime" / "config.xml").write_text("<machine>compy</machine>\n<edited/>\n")
    (top / "src" / "new dir").mkdir(parents=True)
    (top / "src" / "new dir" / "Extra.F90").write_text("y = 3\n")
    (top / "table.bin").write_bytes(bytes(range(256)) * 4)
    (top / "empty.F90").write_bytes(b"")
    return top


def _clean_clone(checkout, tmp_path):
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(checkout), str(clone))
    _git(clone, "submodule", "update", "--init", "-q")
    return clone


class TestThePatchAppliesAtTheRoot:

    def test_a_clean_clone_with_the_patch_builds_the_same_code(self, checkout, tmp_path):
        before = DC.source_state(checkout)
        assert "cime" in " ".join(before["changed"]) and before["untracked"]
        text, left_out = DC.source_patch_parts(checkout)
        assert left_out == []
        assert text.splitlines()[0].startswith(f"# E3SM checkout {checkout} at commit ")
        assert "Entering" not in text and "Subproject commit" not in text
        assert "a/cime/config.xml" in text and "b/src/new dir/Extra.F90" in text
        patch = tmp_path / "source_changes.patch"
        patch.write_text(text)
        clone = _clean_clone(checkout, tmp_path)
        assert DC.source_state(clone)["fingerprint"] != before["fingerprint"]
        _git(clone, "apply", str(patch))
        after = DC.source_state(clone)
        assert after["untracked"] == before["untracked"]
        assert after["changed"] == before["changed"]
        assert after["fingerprint"] == before["fingerprint"]
        assert (clone / "table.bin").read_bytes() == (checkout / "table.bin").read_bytes()
        assert DC.source_patch(checkout) == text

    def test_a_file_too_large_to_carry_is_listed_not_patched(self, checkout, tmp_path,
                                                             monkeypatch):
        monkeypatch.setattr(DC, "_UNTRACKED_HASH_MAX", 100)
        text, left_out = DC.source_patch_parts(checkout)
        assert left_out == ["table.bin"]
        assert "#   table.bin\n" in text
        assert "b/table.bin" not in text and "b/src/new dir/Extra.F90" in text
        patch = tmp_path / "p.patch"
        patch.write_text(text)
        clone = _clean_clone(checkout, tmp_path)
        _git(clone, "apply", str(patch))
        assert not (clone / "table.bin").exists()
        assert (clone / "src" / "new dir" / "Extra.F90").read_text() == "y = 3\n"
