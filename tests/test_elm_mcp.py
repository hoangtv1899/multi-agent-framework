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

    def test_every_advertised_tool_exists(self, srv):
        """A capabilities tool that lists work it cannot do is worse than none
        — the reaction MCP has 40 tools and it cost real time to find which
        were usable."""
        d = json.loads(srv.describe_elm_capabilities())
        assert d["workflow"], "no workflow advertised"
        for step in d["workflow"]:
            assert callable(getattr(srv, step["tool"], None)), \
                f"{step['tool']} is advertised but not defined"

    def test_the_surface_is_only_compile_and_run(self, srv):
        """The rule the whole design follows: this server compiles and runs
        the model, and the caller decides what to run and reads what came
        out. A tool that generated inputs or read results would break it."""
        d = json.loads(srv.describe_elm_capabilities())
        advertised = {step["tool"] for step in d["workflow"]}
        assert advertised == {"build_elm_cases", "submit_elm_ensemble",
                              "check_elm_job"}, advertised

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
        for missing in ("sampling", "warm-start", "surface", "history files",
                        "caveat", "experiment.json"):
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

    def test_the_case_build_hands_back_a_job_not_case_dirs(self, tmp_path):
        """D1: the CIME build is sbatch'd, so _build_cases returns a Pending."""
        from core.exp_manager_base import Pending
        m = _mgr(tmp_path)
        (m.input_dir / m.CASE_INPUTS).write_text(json.dumps(
            [{"case_name": "col_01", "runtime_config": {"FSURDAT": "/x.nc"}}]))
        c = _FakeClient(build_elm_cases={"job_id": "880001",
                                         "log_path": "/x/build.log"})
        out = m._build_cases([{"case_name": "col_01"}],
                         {"mcp_clients": {"elm": c}})
        assert isinstance(out, Pending)
        assert out.job_id == "880001"

    def test_the_case_build_refuses_without_inputs(self, tmp_path):
        """The server does not generate inputs. Submitting a build job with
        nothing for it to read would burn a queue slot to fail."""
        m = _mgr(tmp_path)
        c = _FakeClient(build_elm_cases={"job_id": "880001"})
        with pytest.raises(RuntimeError, match="case_inputs.json"):
            m._build_cases([{"case_name": "col_01"}], {"mcp_clients": {"elm": c}})

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
        with pytest.raises(RuntimeError, match="build_cases must run"):
            m._run([{"case_name": "c1"}], {"mcp_clients": {"elm": c}})

    def test_poll_dispatches_on_the_stage(self, tmp_path):
        """Phase 3b added `stage` to the record for exactly this: a finished
        CIME build hands back case dirs, a finished ensemble hands back
        outcomes, and a job id does not say which.

        The case dirs come back from check_elm_job itself — a few short
        strings travel inline, which is why there is no separate collect tool
        for the build."""
        m = _mgr(tmp_path)
        c = _FakeClient(check_elm_job={
            "active": False, "state": "COMPLETED", "ok": True,
            "n_ok": 1, "n_total": 1,
            "cases": [{"case_name": "col_01", "case_dir": "/scratch/col_01"}]})
        exps = [{"case_name": "col_01"}]
        out = m._poll({"stage": "build_cases", "job_id": "880001"}, exps,
                      {"mcp_clients": {"elm": c}})
        assert out is exps
        assert exps[0]["case_dir"] == "/scratch/col_01"
        assert [t for t, _ in c.calls] == ["check_elm_job"], \
            "one call: the answer rides back with the status"

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
        c = _FakeClient(check_elm_job={"active": False, "state": "FAILED",
                                       "ok": False,
                                       "error": "CIME build failed",
                                       "log_tail": "..."})
        with pytest.raises(RuntimeError, match="case build failed"):
            m._poll({"stage": "build_cases", "job_id": "880001"},
                    [{"case_name": "c1"}], {"mcp_clients": {"elm": c}})



class TestExtractionStaysWithTheCaller:
    """_extract went back to the framework: reading history files needs no
    scheduler, no long wait and no login-node CPU — none of the reasons the
    boundary exists. PFLOTRAN's never left, and now the two match."""

    def test_the_manager_does_not_route_extract_through_the_server(self):
        src = (ROOT / "src" / "core" / "elm_exp_manager.py").read_text()
        i = src.index("def _extract(")
        body = src[i:i + 1500]
        assert "_mcp(" not in body, "extract must not reach for a client"
        assert "ELMResultsAnalyzer" in body

    def test_the_server_advertises_no_result_reading(self, srv):
        d = json.loads(srv.describe_elm_capabilities())
        assert not any("collect" in t["tool"] for t in d["workflow"])


class TestTheCaseListCarriesWhatTheBuildNeeds:
    """The whole split rests on this file. The server never opens a surfdata
    file, so anything the warm start decided has to cross as data."""

    def test_the_adapter_is_replaced_by_what_built_it(self, tmp_path):
        from core.elm_input_agent import ELMAgentAdapter
        m = _mgr(tmp_path)
        a = ELMAgentAdapter(case_name="col_01",
                            runtime_config={"FSURDAT": "/s.nc",
                                            "FINIDAT": "/w.nc",
                                            "STOP_N": "1"})
        row = m._serialise_case_inputs([{"case_name": "col_01", "lat": 38.6,
                                   "elm_agent": a}])[0]
        assert row["runtime_config"]["FSURDAT"] == "/s.nc"
        assert row["runtime_config"]["FINIDAT"] == "/w.nc"
        assert "elm_agent" not in row

    def test_no_object_is_written_as_its_repr(self, tmp_path):
        """json.dumps(default=str) turns an adapter into a string that is
        truthy, attribute-free and useless — the Phase 3 failure, one layer
        further out."""
        from core.elm_input_agent import ELMAgentAdapter
        m = _mgr(tmp_path)
        m._save_case_inputs([{"case_name": "col_01",
                        "elm_agent": ELMAgentAdapter(
                            case_name="col_01",
                            runtime_config={"STOP_N": "1"})}])
        raw = (m.input_dir / m.CASE_INPUTS).read_text()
        assert "ELMAgentAdapter" not in raw and "object at 0x" not in raw

    def test_the_default_serialiser_is_a_no_op(self, tmp_path):
        """PFLOTRAN's experiments are pure data — it must not need a hook."""
        from core.pflotran_exp_manager import PFLOTRANExpManager
        m = PFLOTRANExpManager(base_output_dir=str(tmp_path))
        exps = [{"id": "col_01", "n_cells": 9}]
        assert m._serialise_case_inputs(exps) == exps

    def test_a_case_list_with_no_runtime_config_is_caught_before_a_queue_slot(
            self, tmp_path):
        """It happened: the case list was written without runtime_config, the
        file looked plausible, and it named nothing the build needs. A job was
        submitted against it before anyone noticed."""
        m = _mgr(tmp_path)
        (m.input_dir / m.CASE_INPUTS).write_text(
            json.dumps([{"case_name": "col_01"}]))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "bcj", ROOT / "mcp" / "elm-mcp" / "build_cases_job.py")
        job = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(job)
        rc = job.main(str(m.run_dir))
        assert rc == 1
        out = json.loads((m.input_dir / "built_cases.json").read_text())
        assert out["ok"] is False
        assert "runtime_config" in out["error"]
