#!/usr/bin/env python3
"""The ELM MCP server — Phase 4.

What is pinned here is the two things that make a server usable by someone who
did not write it: that it starts in a stranger's environment, and that it does
not claim tools it does not have.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SERVER = ROOT / "mcp" / "elm-mcp" / "main.py"


def _load():
    spec = importlib.util.spec_from_file_location("elm_mcp_server", SERVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def srv():
    return _load()


class TestItStartsInAStrangersEnvironment:
    """mcp.client.stdio.get_default_environment() forwards ONLY HOME, LOGNAME,
    PATH, SHELL and USER. Our MCPManager happens to forward os.environ.copy();
    a standard client does not.

    That difference is why every LAMBDA tool in the reaction MCP worked through
    this framework and failed under Claude Code — same server, same machine,
    different launcher. This is the test that would have caught it.
    """

    def test_every_path_resolves_with_only_the_five_forwarded_vars(self):
        out = subprocess.run(
            [sys.executable, "-c",
             f"import importlib.util, json;"
             f"s=importlib.util.spec_from_file_location('e', {str(SERVER)!r});"
             f"m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
             f"print(m.describe_elm_capabilities())"],
            env={"HOME": os.path.expanduser("~"),
                 "LOGNAME": os.environ.get("LOGNAME", "u"),
                 "USER": os.environ.get("USER", "u"),
                 "SHELL": "/bin/bash",
                 "PATH": "/usr/bin:/bin"},
            capture_output=True, text=True, timeout=180)
        assert out.returncode == 0, out.stderr[-2000:]
        d = json.loads(out.stdout)
        assert not d["broken_imports"], (
            f"the server cannot import what it fronts when the environment is "
            f"not inherited: {d['broken_imports']}")
        assert not d["missing_requirements"], (
            f"a path was inherited rather than resolved: "
            f"{d['missing_requirements']}")

    def test_the_locale_is_set_here_not_per_subprocess(self, srv):
        """CIME's python dies on the C locale. Set at import so a tool added
        later cannot forget it — elm_wrapper learned this the hard way."""
        assert os.environ.get("LC_ALL", "").lower().startswith("en_us")


class TestItDoesNotClaimWhatItCannotDo:

    def test_available_now_matches_the_registered_tools(self, srv):
        """A capabilities tool that lists planned work as available is worse
        than none — the reaction MCP has 40 tools and it cost real time to
        find which were usable."""
        d = json.loads(srv.describe_elm_capabilities())
        planned = {s["tool"] for s in d["workflow"]
                   if s["status"] != "available"}
        assert not (set(d["available_now"]) & planned), \
            "a tool is listed as both available and planned"
        for name in d["available_now"]:
            assert callable(getattr(srv, name, None)), \
                f"{name} is advertised but not defined"

    def test_requirements_are_checked_not_asserted(self, srv):
        """A list of assumptions is worth nothing on a machine where one is
        false, which is exactly the case this exists to diagnose."""
        d = json.loads(srv.describe_elm_capabilities())
        assert d["requirements"], "no requirements reported"
        for name, r in d["requirements"].items():
            assert set(r) == {"path", "present"}, name
            assert isinstance(r["present"], bool)
            # present must reflect the filesystem, not a hard-coded true
            if r["present"] and r["path"]:
                assert Path(r["path"]).exists(), \
                    f"{name} claims present but {r['path']} does not exist"

    def test_it_says_what_it_is_not(self, srv):
        """An ensemble is not a study. Someone driving these tools directly
        gets no sampling design, no strategy gate and no experiment.json."""
        d = json.loads(srv.describe_elm_capabilities())
        joined = " ".join(d["does_not"]).lower()
        for missing in ("sampling", "strategy", "caveat", "experiment.json"):
            assert missing in joined, f"does_not never mentions {missing}"

    def test_ready_is_false_when_something_is_missing(self, srv, monkeypatch):
        """`ready` must be derived, not declared."""
        monkeypatch.setenv("E3SM_SRC_DIR", "/no/such/tree")
        d = json.loads(srv.describe_elm_capabilities())
        assert d["ready"] is False
        assert "e3sm_source" in d["missing_requirements"]


class TestItIsRegistered:

    def test_mcp_config_has_elm(self):
        cfg = json.loads((ROOT / "mcp_config.json").read_text())
        assert "elm" in cfg["mcp_servers"], \
            "the framework cannot use a server it does not know about"
        entry = cfg["mcp_servers"]["elm"]
        assert Path(entry["args"][0]).is_file()

    def test_the_timeout_allows_for_netcdf_reads(self):
        """prepare and run SUBMIT rather than block, but build generates
        per-column surfaces and collect reads NetCDF, and neither is instant
        for 19 columns."""
        cfg = json.loads((ROOT / "mcp_config.json").read_text())
        assert cfg["mcp_servers"]["elm"].get("timeout", 0) >= 300
