"""Run a PFLOTRAN ensemble as a SCHEDULER JOB instead of inside the call.

tools/job_runner.py

WHY THIS EXISTS, GIVEN run_simulation ALREADY WORKS
---------------------------------------------------
Not speed. A framework column solves in ~0.3 s, and for an ensemble of those
the inline path is simply better — a job would spend longer in the queue than
in the solver.

It is about WHERE the work runs and HOW LONG the caller is held:

  * Inline, the simulations execute in the MCP server's own process tree. On a
    cluster that server lives on a LOGIN NODE, so `max_parallel` concurrent
    mpirun processes are login-node load. That is antisocial at 4 wide and
    unacceptable at 40.

  * An MCP client opens a fresh stdio session per call and tearing it down
    kills the server AND ITS CHILDREN. Every inline second is a second the
    whole ensemble can be destroyed in. A three-hour reactive-transport run
    cannot be attempted this way at all, whatever the timeout is set to.

So this module adds a second mode, deliberately NOT a replacement:

    submit  -> a job id, immediately
    check   -> is the scheduler still busy with it?
    collect -> the same result dict run_simulation returns

The caller chooses. Use the inline path for the many-small case, which is most
of what this server does; submit when the run is long, wide, or when the server
is somewhere it should not be spending CPU.

WHAT IS NOT NEGOTIABLE
----------------------
`results_by_input`. The aggregate exit_codes list is in COMPLETION order, so
exit_codes[i] does not belong to input_files[i]; only the map says which deck
produced which outcome. A batch path that dropped it would run the ensemble
correctly and make it impossible to attribute — which is worse than not running
it, because the numbers look fine.

Scheduler: SLURM. sbatch/squeue/sacct are looked up at call time and their
absence is reported rather than raised, so this module imports cleanly on a
laptop.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

JOB_DIR_NAME = ".pflotran_job"
SPEC_NAME = "spec.json"
RESULT_NAME = "result.json"
SCRIPT_NAME = "job.sbatch"
LOG_NAME = "job.log"

# The payload the job runs. A FILE, not a heredoc: it can be run by hand when a
# job fails, read without untangling shell quoting, and a syntax error in it is
# found by importing it rather than after a queue wait.
PAYLOAD = Path(__file__).resolve().parent / "pflotran_job_payload.py"

# States in which the scheduler still owns the job. Anything else — COMPLETED,
# FAILED, TIMEOUT, CANCELLED — means it is finished with it, whatever it did.
ACTIVE_STATES = {
    "PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED",
    "RESIZING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "SIGNALING",
    "STAGE_OUT", "RESV_DEL_HOLD", "STOPPED",
}


def job_dir(output_dir: Union[str, Path]) -> Path:
    return Path(output_dir) / JOB_DIR_NAME


def slurm_state(job_id: Any) -> Optional[str]:
    """What SLURM says the job is doing, or None if it will not say.

    squeue first — cheap, and the only one that sees a job that has not
    started. Then sacct, the only one that remembers a job that has left the
    queue.

    None means NO ANSWER, never "finished". A squeue that times out or a
    cluster without sacct must not be read as a completed ensemble; the caller
    decides what other evidence it trusts.
    """
    jid = str(job_id).split("_")[0].split(".")[0]
    if not jid.isdigit():
        return None

    def _ask(cmd) -> Optional[str]:
        if not shutil.which(cmd[0]):
            return None
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=60)
        except Exception:                                       # noqa: BLE001
            return None
        lines = (out.stdout or "").strip().splitlines()
        return lines[0].strip() if lines and lines[0].strip() else None

    st = _ask(["squeue", "-h", "-j", jid, "-o", "%T"])
    if st:
        return st.upper()
    # -X so a job's STEPS do not shadow the job: step lines come back first and
    # a step can read COMPLETED while the job itself is still going.
    st = _ask(["sacct", "-n", "-X", "-j", jid, "-o", "State"])
    if st:
        return st.split()[0].upper()      # sacct spells it "CANCELLED by 1234"
    return None


def submit(input_files: List[str],
           output_dir: Union[str, Path],
           executable: str,
           num_cores: int = 1,
           mpi_command: str = "mpirun",
           max_parallel: Optional[int] = None,
           timeout: Optional[float] = None,
           queue: Optional[str] = None,
           walltime: str = "01:00:00",
           account: Optional[str] = None,
           nodes: int = 1) -> Dict[str, Any]:
    """Write a job for this ensemble and submit it. Returns a job id.

    Everything the job needs is written to disk first, because the compute node
    shares none of this process's memory and none of its arguments.
    """
    if not shutil.which("sbatch"):
        return {"error": "sbatch is not available — use run_pflotran_simulation "
                         "for the inline path"}
    if not PAYLOAD.is_file():
        return {"error": f"missing job payload: {PAYLOAD}"}

    decks = [str(Path(f).resolve()) for f in (input_files or [])]
    missing = [d for d in decks if not Path(d).is_file()]
    if not decks:
        return {"error": "no input files given"}
    if missing:
        return {"error": f"input files do not exist: {missing[:5]}"}

    jd = job_dir(output_dir)
    jd.mkdir(parents=True, exist_ok=True)
    # A stale result from an earlier attempt would be read as THIS job's answer
    # the moment it is polled.
    (jd / RESULT_NAME).unlink(missing_ok=True)

    spec = {
        "input_files": decks,
        "executable": executable,
        "num_cores": int(num_cores),
        "mpi_command": mpi_command,
        "max_parallel": int(max_parallel) if max_parallel else None,
        "timeout": float(timeout) if timeout else None,
        "output_dir": str(Path(output_dir).resolve()),
    }
    (jd / SPEC_NAME).write_text(json.dumps(spec, indent=2))

    q = queue or os.environ.get("IDEAS_SLURM_QUEUE", "short")
    acct = account or os.environ.get("IDEAS_SLURM_ACCOUNT", "")
    acct_line = f"#SBATCH -A {acct}" if acct else ""
    script = jd / SCRIPT_NAME
    # sys.executable, not "python": the job gets a login shell with no
    # environment from here, and the payload imports this package.
    script.write_text(f"""#!/bin/bash
#SBATCH -J pflotran_ens
#SBATCH -N {int(nodes)}
#SBATCH -p {q}
{acct_line}
#SBATCH -t {walltime}
#SBATCH -o {jd / LOG_NAME}
export PFLOTRAN_EXECUTABLE={executable}
export MPI_COMMAND={mpi_command}
cd {Path(__file__).resolve().parents[1]}
echo "running {len(decks)} deck(s) on $(hostname)"
{sys.executable} {PAYLOAD} {jd / SPEC_NAME}
""")

    try:
        jid = subprocess.check_output(["sbatch", "--parsable", str(script)],
                                      text=True, timeout=120).strip()
    except Exception as e:                                      # noqa: BLE001
        return {"error": f"sbatch failed: {e}"}

    return {
        "job_id": jid.split(";")[0],
        "n_decks": len(decks),
        "queue": q,
        "walltime": walltime,
        "job_dir": str(jd),
        "log_path": str(jd / LOG_NAME),
        "result_path": str(jd / RESULT_NAME),
    }


def check(job_id: Any, output_dir: Union[str, Path] = "") -> Dict[str, Any]:
    """Is the scheduler still busy with this job?

    `active` is the field to branch on, and it is TRUE when the scheduler will
    not answer. "No answer" is not "finished": collecting on a squeue timeout
    would read an ensemble still in flight as a finished one.
    """
    state = slurm_state(job_id)
    known = state is not None
    out = {
        "job_id": str(job_id),
        "state": state,
        "active": (state in ACTIVE_STATES) if known else True,
        "scheduler_answered": known,
    }
    if not known:
        out["note"] = ("the scheduler will not say — treated as still running "
                       "unless a result file says otherwise")
    if output_dir:
        jd = job_dir(output_dir)
        done = (jd / RESULT_NAME).is_file()
        out["result_written"] = done
        # DISTINCT FROM `active`. The scheduler finishing and the result
        # becoming readable here are different moments — see collect(). A
        # caller that branches on `active` alone can arrive before the file
        # does; one that waits for `ready` cannot.
        out["ready"] = done
        if not known and done:
            out["active"] = False
            out["note"] = "gone from the scheduler, but the result file exists"
        log = jd / LOG_NAME
        if log.is_file():
            out["log_tail"] = "\n".join(
                log.read_text(errors="replace").strip().splitlines()[-10:])
    return out


def collect(output_dir: Union[str, Path],
            wait_s: float = 30.0) -> Dict[str, Any]:
    """What the job produced — the same dict run_simulation returns.

    Same shape ON PURPOSE. A caller that switches between the inline and the
    batch path should not have to parse two result formats, and the one field
    that matters most, results_by_input, is what makes either usable at all.

    WAITS BRIEFLY FOR THE FILE. "The scheduler has finished" and "the result is
    readable from here" are not the same instant: the job writes to a parallel
    filesystem from a compute node and the submitting host's view of it lags.

    Measured, not guessed — job 770699 ran three decks correctly, logged
    ENSEMBLE_DONE 3/3, and wrote result.json at 16:06:28.961 against a job end
    time of 16:06:28. A collect fired the moment sacct said COMPLETED found
    nothing and reported an ensemble of three successful columns as producing
    no results at all.

    Bounded, because an unbounded wait would hang forever on a job that really
    did die without writing. wait_s=0 skips the wait for a caller that has
    already established the file is there.
    """
    import time

    jd = job_dir(output_dir)
    rp = jd / RESULT_NAME

    deadline = time.time() + max(0.0, float(wait_s))
    while not rp.is_file() and time.time() < deadline:
        time.sleep(2.0)

    if not rp.is_file():
        log = jd / LOG_NAME
        return {
            "status": "incomplete",
            "error": f"no result file after waiting {wait_s:.0f}s — the job "
                     f"has not finished, or it died before writing one",
            "log_tail": ("\n".join(log.read_text(errors="replace")
                                   .strip().splitlines()[-20:])
                         if log.is_file() else ""),
        }
    try:
        return json.loads(rp.read_text())
    except Exception as e:                                      # noqa: BLE001
        # A partially-flushed file parses as broken JSON rather than as absent,
        # so this path is the same race one step later. One retry, then report.
        time.sleep(3.0)
        try:
            return json.loads(rp.read_text())
        except Exception:                                       # noqa: BLE001
            return {"status": "failed", "error": f"unreadable result: {e}"}
