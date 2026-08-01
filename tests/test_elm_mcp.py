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


# ─────────────────────────────────────────────────────────────────────
# Step 6 — the framework driving the server
# ─────────────────────────────────────────────────────────────────────
class _FakeClient:
    """Stands in for an MCP client. Records what was asked and replies."""

    def __init__(self, **replies):
        self.timeout = 30.0
        self.calls = []
        self.replies = replies

    def call_tool_json(self, tool, args):
        self.calls.append((tool, args))
        r = self.replies.get(tool)
        return r(args) if callable(r) else r


def _mgr(tmp_path):
    from core.elm_exp_manager import ELMExpManager
    return ELMExpManager(base_output_dir=str(tmp_path))


class TestTheMCPIsTheDefaultForELM:
    """D5. The MCP path runs whenever an `elm` client is registered."""

    def test_registered_means_used(self, tmp_path):
        m = _mgr(tmp_path)
        c = _FakeClient()
        assert m._mcp({"mcp_clients": {"elm": c}}) is c

    def test_no_client_still_runs_locally(self, tmp_path):
        """Not a silent demotion (d61eaed): the choice is made by what is
        CONFIGURED, not by a failure."""
        m = _mgr(tmp_path)
        assert m._mcp({"mcp_clients": {}}) is None
        assert m._mcp({}) is None

    def test_it_can_be_turned_off(self, tmp_path):
        m = _mgr(tmp_path)
        c = _FakeClient()
        assert m._mcp({"mcp_clients": {"elm": c},
                       "run_via_mcp": False}) is None

    def test_a_timeout_is_an_error_not_an_empty_result(self, tmp_path):
        """call_tool_json returns None on timeout. Reading that as 'no cases'
        would report a healthy ensemble as zero columns."""
        m = _mgr(tmp_path)
        c = _FakeClient(build_elm_cases=None)
        with pytest.raises(RuntimeError, match="did not answer"):
            m._mcp_call(c, "build_elm_cases", {})

    def test_a_server_side_error_is_raised(self, tmp_path):
        m = _mgr(tmp_path)
        c = _FakeClient(build_elm_cases={"error": "no run_plan.json"})
        with pytest.raises(RuntimeError, match="no run_plan.json"):
            m._mcp_call(c, "build_elm_cases", {})

    def test_the_call_budget_is_raised_then_restored(self, tmp_path):
        """A client ceiling 15x too small is how the reaction MCP failed
        before 19fffc3 — and it must not stay raised afterwards."""
        m = _mgr(tmp_path)
        seen = {}
        c = _FakeClient()
        c.replies["x"] = lambda a: seen.setdefault("timeout", c.timeout) or {}
        m._mcp_call(c, "x", {}, budget=1800)
        assert seen["timeout"] == 1800
        assert c.timeout == 30.0, "the raise must be scoped to the one call"


class TestEachStageGoesThroughTheServer:

    def test_prepare_hands_back_a_job_not_case_dirs(self, tmp_path):
        """D1: the CIME build is sbatch'd, so _prepare returns a Pending."""
        from core.exp_manager_base import Pending
        m = _mgr(tmp_path)
        c = _FakeClient(prepare_elm_cases={"job_id": "880001",
                                           "log_path": "/x/prepare.log"})
        out = m._prepare([{"case_name": "col_01"}],
                         {"mcp_clients": {"elm": c}})
        assert isinstance(out, Pending)
        assert out.job_id == "880001"

    def test_run_hands_back_a_job(self, tmp_path):
        from core.exp_manager_base import Pending
        m = _mgr(tmp_path)
        c = _FakeClient(submit_elm_ensemble={"job_id": "770693", "n_cases": 2})
        out = m._run([{"case_name": "c1", "case_dir": str(tmp_path)}],
                     {"mcp_clients": {"elm": c}})
        assert isinstance(out, Pending) and out.job_id == "770693"

    def test_submitting_without_cases_is_refused(self, tmp_path):
        """An ensemble cannot run before its cases are built, and submitting
        an empty job list would queue a job that does nothing and report it
        as progress."""
        m = _mgr(tmp_path)
        c = _FakeClient(submit_elm_ensemble={"job_id": "1"})
        with pytest.raises(RuntimeError, match="prepare must run"):
            m._run([{"case_name": "c1"}], {"mcp_clients": {"elm": c}})

    def test_poll_dispatches_on_the_stage(self, tmp_path):
        """Phase 3b added `stage` to the record for exactly this: a finished
        CIME build hands back case dirs, a finished ensemble hands back
        outcomes, and a job id does not say which."""
        m = _mgr(tmp_path)
        c = _FakeClient(
            check_elm_job={"active": False, "state": "COMPLETED"},
            collect_prepared_cases={"ok": True, "n_ok": 1, "n_total": 1,
                                    "cases": [{"case_name": "col_01",
                                               "case_dir": "/scratch/col_01"}]})
        exps = [{"case_name": "col_01"}]
        out = m._poll({"stage": "prepare", "job_id": "880001"}, exps,
                      {"mcp_clients": {"elm": c}})
        assert out is exps
        assert exps[0]["case_dir"] == "/scratch/col_01"
        assert "collect_prepared_cases" in [t for t, _ in c.calls]

    def test_an_active_job_collects_nothing(self, tmp_path):
        m = _mgr(tmp_path)
        c = _FakeClient(check_elm_job={"active": True, "state": "RUNNING"})
        assert m._poll({"stage": "run", "job_id": "770693"}, [],
                       {"mcp_clients": {"elm": c}}) is None
        assert "collect_prepared_cases" not in [t for t, _ in c.calls]

    def test_a_failed_build_raises_rather_than_running_short(self, tmp_path):
        """A case that did not compile is a failure to report, not an ensemble
        to run one column short."""
        m = _mgr(tmp_path)
        c = _FakeClient(check_elm_job={"active": False, "state": "FAILED"},
                        collect_prepared_cases={"ok": False,
                                                "error": "CIME build failed",
                                                "log_tail": "..."})
        with pytest.raises(RuntimeError, match="case build failed"):
            m._poll({"stage": "prepare", "job_id": "880001"},
                    [{"case_name": "c1"}], {"mcp_clients": {"elm": c}})


class TestTheExtractionAdapter:
    """_extract must return something _package and _save_llm_input can use,
    built from the payload on disk rather than from the MCP response."""

    def test_it_exposes_what_the_later_stages_touch(self):
        from core.elm_exp_manager import _MCPExtraction
        payload = {"experiments": [{"case_name": "c1", "metrics": {"a": 1.0}}],
                   "units": {"QCHARGE": "mm/s"}}
        x = _MCPExtraction(payload)
        assert x.results == payload["experiments"]
        assert x.units["QCHARGE"] == "mm/s"
        assert x.summary["units"], "_package reads .summary too"
        assert x.get_llm_analysis_input() is payload
        assert x.extra_summary == {}, \
            "an attribute that sometimes exists is worse than one that is " \
            "sometimes empty"

    def test_package_accepts_it(self, tmp_path):
        from core.elm_exp_manager import _MCPExtraction
        m = _mgr(tmp_path)
        x = _MCPExtraction({"experiments": [
            {"case_name": "c1", "status": "ok", "metrics": {"a": 1.0}},
            {"case_name": "c2", "status": "failed", "metrics": {}}]})
        m._package({}, x, {})
        pkg = json.loads((m.run_dir / "experiment.json").read_text())
        assert pkg["columns_total"] == 2
        assert pkg["columns_succeeded"] == 1
