"""The bulk-data paths are overridable, and say where they came from.

Phase 1a measured that this server reaches the CONUS restart and surfdata under
the five variables an MCP client forwards — and the reason it passes is that
every data root is hardcoded. Making them configurable must not give that back.

Hence four layers with a file among them: an environment variable alone works
under the framework's MCPManager (mcp_client.py copies os.environ) and fails
silently under a standard client, so the only override that works everywhere is
one the server reads off disk itself.

These tests pin the precedence, the reporting, and the two things that would
otherwise fail quietly: a manifest whose restarts have vanished, and a
paths.json with a typo in it.
"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp" / "elm-mcp" / "src"))

import paths as P                                               # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """No inherited override, and paths.json pointed somewhere disposable."""
    for _, (_, env_var, _) in P.DATA_PATHS.items():
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(P, "PATHS_FILE", tmp_path / "paths.json")


def _write_paths_file(content):
    P.PATHS_FILE.write_text(content if isinstance(content, str)
                            else json.dumps(content))


class TestPrecedence:

    def test_default_when_nothing_is_set(self):
        p, src = P.resolve("conus_surfdata")
        assert src == "default"
        assert str(p) == P.DATA_PATHS["conus_surfdata"][0]

    def test_file_beats_default(self):
        _write_paths_file({"conus_surfdata": "/tmp/from-file"})
        p, src = P.resolve("conus_surfdata")
        assert (str(p), src) == ("/tmp/from-file", "file:paths.json")

    def test_env_beats_file(self, monkeypatch):
        _write_paths_file({"conus_surfdata": "/tmp/from-file"})
        monkeypatch.setenv("IDEAS_CONUS_SURFDATA", "/tmp/from-env")
        p, src = P.resolve("conus_surfdata")
        assert (str(p), src) == ("/tmp/from-env", "env:IDEAS_CONUS_SURFDATA")

    def test_argument_beats_everything(self, monkeypatch):
        _write_paths_file({"conus_surfdata": "/tmp/from-file"})
        monkeypatch.setenv("IDEAS_CONUS_SURFDATA", "/tmp/from-env")
        p, src = P.resolve("conus_surfdata", "/tmp/from-arg")
        assert (str(p), src) == ("/tmp/from-arg", "argument")

    def test_an_unknown_name_is_an_error_not_a_default(self):
        with pytest.raises(KeyError):
            P.resolve("conus_sufrdata")          # typo, deliberately


class TestItFailsLoudly:

    def test_a_malformed_paths_file_is_reported_not_swallowed(self, capfd):
        _write_paths_file("{ not json")
        p, src = P.resolve("conus_surfdata")
        assert src == "default", "a typo must not be indistinguishable from absence"
        assert "paths.json is unreadable" in capfd.readouterr().out

    def test_a_missing_path_says_so(self, monkeypatch):
        monkeypatch.setenv("IDEAS_CONUS_SURFDATA", "/nope/not/here")
        d = P.describe("conus_surfdata")
        assert d["present"] is False
        assert d["error"] == "does not exist"

    def test_a_manifest_whose_restarts_are_gone_is_not_present(
            self, monkeypatch, tmp_path):
        """The failure this exists to catch: the manifest is fine, the 43 GB
        of restarts it points at are not, and nothing notices until a CIME
        build dies eight minutes in."""
        m = tmp_path / "MANIFEST.txt"
        m.write_text("lat7 38-40N x y /gone/conus_lat7.elm.r.nc\n")
        monkeypatch.setenv("IDEAS_CONUS_RESTART", str(m))
        d = P.describe("conus_restart_manifest")
        assert d["bands_total"] == 1
        assert d["bands_resolving"] == 0
        assert d["present"] is False, "an empty manifest must not read as ready"
        assert "no restart file that exists" in d["error"]

    def test_partially_resolving_bands_warn(self, monkeypatch, tmp_path):
        real = tmp_path / "band.nc"
        real.write_text("x")
        m = tmp_path / "MANIFEST.txt"
        m.write_text(f"lat7 38-40N x y {real}\n"
                     f"lat8 40-42N x y /gone/other.nc\n")
        monkeypatch.setenv("IDEAS_CONUS_RESTART", str(m))
        d = P.describe("conus_restart_manifest")
        assert d["present"] is True
        assert (d["bands_resolving"], d["bands_total"]) == (1, 2)
        assert any("1 of 2 bands resolve" in w for w in d["warnings"])


class TestReporting:

    def test_describe_names_the_source_and_the_override_routes(self):
        d = P.describe("conus_surfdata")
        assert d["source"] == "default" and d["is_default"] is True
        assert d["override_with"]["env"] == "IDEAS_CONUS_SURFDATA"
        assert "paths.json" in d["override_with"]["file"]

    def test_foreign_ownership_is_warned_about(self, monkeypatch, tmp_path):
        f = tmp_path / "someone_elses.nc"
        f.write_text("x")
        monkeypatch.setenv("IDEAS_CONUS_SURFDATA", str(f))
        monkeypatch.setenv("USER", "not-the-owner")
        d = P.describe("conus_surfdata")
        assert any("not by you" in w for w in d.get("warnings", []))

    def test_provenance_carries_value_and_source(self, monkeypatch):
        monkeypatch.setenv("IDEAS_CONUS_SURFDATA", "/tmp/from-env")
        pr = P.provenance()
        assert pr["conus_surfdata"] == "/tmp/from-env"
        assert pr["conus_surfdata_source"] == "env:IDEAS_CONUS_SURFDATA"
        # every path is represented, so a run record cannot omit one silently
        for name in P.DATA_PATHS:
            assert name in pr and f"{name}_source" in pr


class TestRealDataOnThisMachine:
    """Not a unit test — a statement about Compy today, which is what the
    warning exists to make visible."""

    def test_the_conus_restarts_resolve_and_are_flagged_as_borrowed(self):
        d = P.describe("conus_restart_manifest")
        if not d["present"]:
            pytest.skip("CONUS manifest not available on this machine")
        assert d["bands_resolving"] == d["bands_total"]
        assert d.get("restart_owner"), "who owns the restarts must be reported"
