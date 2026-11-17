"""Batch settings per machine (mcp/elm-mcp/src/machine_settings.py), and the job
scripts and sbatch lines that follow them.

Compy's row must leave every job as it was before the table existed. Perlmutter's
must give a QOS, the CPU constraint, 128 tasks per node and no assumed account.
Project names here are made up.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import domain_case as DC
import machine_settings as MS

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "mcp" / "elm-mcp" / "main.py"
MACHINE_VARS = ("IDEAS_MACHINE", "NERSC_HOST", "IDEAS_SLURM_ACCOUNT", "IDEAS_SLURM_QUEUE")


@pytest.fixture
def env(monkeypatch):
    """No machine, account or queue set: the process looks like Compy."""
    for k in MACHINE_VARS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _recipe(root, mach="compy", ntasks="40", project="madeupproj"):
    return DC.Recipe(case="synth", case_root=str(root), srcroot="/src/E3SM",
                     compset="2000_DATM%QIA_ELM%CNPRDCTCBC_SICE_SOCN_SROF_SGLC_SWAV",
                     mach=mach, project=project,
                     xml=[("NTASKS", ntasks), ("RUN_STARTDATE", "2015-01-01")],
                     start="2015-01-01")


class TestTheTable:

    def test_compy_keeps_the_values_every_earlier_job_used(self, env):
        s = MS.settings("compy")
        assert (s["tasks_per_node"], s["queue_option"], s["default_queue"],
                s["header"], s["default_account"]) == (40, "-p", "short", [], "e3sm")
        assert MS.queue_args("compy") == ["-p", "short"]
        assert MS.account(machine="compy") == "e3sm"

    def test_perlmutter_asks_for_a_qos_and_cpu_nodes_and_assumes_no_account(self, env):
        assert MS.queue_args("pm-cpu") == ["-q", "regular"]
        assert MS.queue_args("pm-cpu", "debug") == ["-q", "debug"]
        assert MS.header_lines("pm-cpu") == ["#SBATCH --constraint=cpu"]
        assert MS.settings("pm-cpu")["tasks_per_node"] == 128
        with pytest.raises(MS.MachineError, match="IDEAS_SLURM_ACCOUNT"):
            MS.account(machine="pm-cpu")

    def test_the_machine_this_process_runs_on(self, env):
        assert MS.current() == "compy"
        env.setenv("NERSC_HOST", "perlmutter")
        assert MS.current() == "pm-cpu"
        env.setenv("IDEAS_MACHINE", "compy")              # named, it wins
        assert MS.current() == "compy"

    def test_a_machine_the_table_does_not_hold_is_refused_by_name(self, env):
        with pytest.raises(MS.MachineError, match="'nowhere'"):
            MS.settings("nowhere")
        env.setenv("IDEAS_MACHINE", "nowhere")
        with pytest.raises(MS.MachineError, match="'nowhere'"):
            MS.default_queue()

    def test_the_account_given_then_the_environment_then_the_default(self, env):
        env.setenv("IDEAS_SLURM_ACCOUNT", "  ")           # empty counts as unset
        assert MS.account(machine="compy") == "e3sm"
        with pytest.raises(MS.MachineError):
            MS.account(machine="pm-cpu")
        env.setenv("IDEAS_SLURM_ACCOUNT", "envproj")
        assert MS.account(machine="pm-cpu") == "envproj"
        assert MS.account("givenproj", machine="pm-cpu") == "givenproj"


class TestTheChain:

    def test_a_compy_script_is_as_before(self, tmp_path):
        text = DC.write_stage_script(_recipe(tmp_path), tmp_path / "rec").read_text()
        assert text.splitlines()[:5] == [
            "#!/bin/bash", "#SBATCH --account=madeupproj", "#SBATCH --nodes=1",
            "#SBATCH --exclusive", f"#SBATCH --output={tmp_path / 'rec'}/logs/%x.%j.out"]
        assert "#   usage: sbatch -J <name> -p <partition> -t <wall> run_stage.sh" in text
        assert "constraint" not in text

    def test_a_perlmutter_script_asks_for_cpu_nodes(self, tmp_path):
        r = _recipe(tmp_path, mach="pm-cpu", ntasks="128")
        text = DC.write_stage_script(r, tmp_path / "rec").read_text()
        assert text.splitlines()[1:6] == [
            "#SBATCH --account=madeupproj", "#SBATCH --nodes=1", "#SBATCH --exclusive",
            "#SBATCH --constraint=cpu", f"#SBATCH --output={tmp_path / 'rec'}/logs/%x.%j.out"]
        assert "#   usage: sbatch -J <name> -q <qos> -t <wall> run_stage.sh" in text

    def test_the_node_count_follows_the_machine(self, tmp_path):
        assert DC._nodes(_recipe(tmp_path, ntasks="128")) == 4          # 40 a node
        assert DC._nodes(_recipe(tmp_path, mach="pm-cpu", ntasks="128")) == 1
        assert DC._nodes(_recipe(tmp_path, mach="pm-cpu", ntasks="256")) == 2
        assert DC._nodes(_recipe(tmp_path, mach="pm-cpu", ntasks="-3")) == 3

    def test_the_default_queue_is_the_machines(self, tmp_path):
        def queues(r, **kw):
            return {j.partition for j in DC.plan_chain(r, [2015], **kw)}
        assert queues(_recipe(tmp_path)) == {"short"}
        assert queues(_recipe(tmp_path, mach="pm-cpu")) == {"regular"}
        assert queues(_recipe(tmp_path, mach="pm-cpu"), partition="debug") == {"debug"}

    @pytest.mark.parametrize("mach,want", [("compy", ["-p", "short"]),
                                           ("pm-cpu", ["-q", "regular"])])
    def test_sbatch_is_told_the_queue_the_machines_way(self, tmp_path, monkeypatch, mach, want):
        monkeypatch.setattr(DC, "node_local", lambda path: None)
        r, rec, calls = _recipe(tmp_path, mach=mach), tmp_path / "rec", []
        DC.write_stage_script(r, rec)

        def slurm(argv, timeout=120):
            calls.append(list(argv))
            return 0, f"{100 + len(calls)}\n", ""
        seg = DC.submit_chain(DC.plan_chain(r, [2015]), rec, run=slurm, machine=mach)
        assert seg["ok"] and len(calls) == 4                  # case, smoke, y2015, report
        assert all(c[:4] == ["sbatch", "--parsable"] + want for c in calls), calls

    def test_a_machine_the_table_does_not_hold_stops_the_chain_before_any_job(
            self, tmp_path, monkeypatch):
        r = _recipe(tmp_path, mach="nowhere")
        with pytest.raises(DC.DomainCaseError, match="nowhere"):
            DC.write_stage_script(r, tmp_path / "rec")
        with pytest.raises(DC.DomainCaseError, match="nowhere"):
            DC.plan_chain(r, [2015])
        monkeypatch.setattr(DC, "node_local", lambda path: None)
        DC.write_stage_script(_recipe(tmp_path), tmp_path / "rec")
        calls = []
        seg = DC.submit_chain(DC.plan_chain(_recipe(tmp_path), [2015]), tmp_path / "rec",
                              run=lambda argv, timeout=120: calls.append(argv) or (0, "1\n", ""),
                              machine="nowhere")
        assert not seg["ok"] and "nowhere" in seg["error"] and calls == []


class TestTheFinalizeJob:

    @pytest.fixture
    def mgr(self, tmp_path):
        from elm_exp_manager import ELMExpManager
        m = ELMExpManager.__new__(ELMExpManager)
        m.run_dir = tmp_path
        return m

    def _script(self, mgr, monkeypatch, config):
        import shutil
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")

        def check_output(argv, **kw):
            if argv[0] == "df":
                return "Filesystem 1K-blocks Used Available Capacity Mounted\nnfs:/x 1 1 1 1% /x\n"
            return "4242\n"
        monkeypatch.setattr(subprocess, "check_output", check_output)
        assert mgr._submit_job_b("101:102", config, "") == "4242"
        return (mgr.run_dir / "ensemble_B.sbatch").read_text()

    def test_on_compy_job_b_is_as_before(self, env, mgr, monkeypatch):
        text = self._script(mgr, monkeypatch, {})
        assert "#SBATCH -p short\n#SBATCH -A e3sm\n#SBATCH -t 00:30:00\n" in text
        assert "constraint" not in text

    def test_on_perlmutter_job_b_asks_for_a_qos_and_cpu_nodes(self, env, mgr, monkeypatch):
        env.setenv("NERSC_HOST", "perlmutter")
        env.setenv("IDEAS_SLURM_ACCOUNT", "madeupproj")
        text = self._script(mgr, monkeypatch, {})
        assert ("#SBATCH -q regular\n#SBATCH -A madeupproj\n#SBATCH --constraint=cpu\n"
                "#SBATCH -t 00:30:00\n") in text

    def test_a_study_with_no_account_on_perlmutter_queues_nothing(self, env, mgr, monkeypatch):
        env.setenv("NERSC_HOST", "perlmutter")

        def no_job(*a, **k):
            pytest.fail("a job was submitted, or the server asked to submit one")
        monkeypatch.setattr(mgr, "_executed", lambda experiments: [{"run": "r1"}])
        monkeypatch.setattr(mgr, "_mcp_call", no_job)
        monkeypatch.setattr(mgr, "_domain_client", no_job)
        with pytest.raises(RuntimeError, match="nothing was submitted"):
            mgr._submit_experiment([{"run": "r1"}], {})
        monkeypatch.setattr(mgr, "_is_experiment_rows", lambda experiments: False)
        with pytest.raises(RuntimeError, match="IDEAS_SLURM_ACCOUNT"):
            mgr._submit_domain([], {})


class TestTheServerOnPerlmutter:

    def test_it_assumes_none_of_compys_settings(self, tmp_path):
        """In a stranger's environment on Perlmutter (the five variables an MCP
        client forwards, and NERSC_HOST), the server sets the QOS and nothing of
        Compy's, and its column route refuses by name."""
        code = (
            "import importlib.util, json, os;"
            f"s=importlib.util.spec_from_file_location('e', {str(SERVER)!r});"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "print(json.dumps({'env': {k: os.environ.get(k) for k in ("
            "'PSCRATCH', 'E3SM_SRC_DIR', 'MODULEPATH', 'MODULESHOME', "
            "'IDEAS_SLURM_ACCOUNT', 'IDEAS_SLURM_QUEUE')},"
            f" 'column': json.loads(m.run_elm_ensemble(run_dir={str(tmp_path)!r})),"
            " 'caps': json.loads(m.describe_elm_capabilities())}))")
        out = subprocess.run(
            [sys.executable, "-c", code],
            env={"HOME": os.path.expanduser("~"), "LOGNAME": os.environ.get("LOGNAME", "u"),
                 "USER": os.environ.get("USER", "u"), "SHELL": "/bin/bash",
                 "PATH": "/usr/bin:/bin", "NERSC_HOST": "perlmutter"},
            capture_output=True, text=True, timeout=180)
        assert out.returncode == 0, out.stderr[-2000:]
        d = json.loads(out.stdout.strip().splitlines()[-1])
        assert d["env"] == {"PSCRATCH": None, "E3SM_SRC_DIR": None, "MODULEPATH": None,
                            "MODULESHOME": None, "IDEAS_SLURM_ACCOUNT": None,
                            "IDEAS_SLURM_QUEUE": "regular"}
        assert "Compy only" in d["column"]["error"]
        reqs = d["caps"]["requirements"] if "requirements" in d["caps"] else d["caps"]
        assert "E3SM_SRC_DIR is not set" in json.dumps(reqs)
