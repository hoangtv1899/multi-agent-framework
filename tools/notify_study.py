#!/usr/bin/env python3
"""What actually happened in a study, as an email body — and as an exit code.

    python tools/notify_study.py <run_dir> [--mail <addr>]
    echo $?                                         # 0 = the study is usable

Why this exists: Slurm's own mail carries a SUBJECT and nothing else —
"Job_id=770816 Name=elm_study Ended, COMPLETED, ExitCode 0". For job 770816
every word of that was true of the job script and false of the study: the
package stage had failed and no analysis existed. The scheduler reports
whether the SCRIPT ran, and the script ran fine.

So the exit status here is the study's, not the script's, and the body says
which stages did what. In the unattended flow this mail is the only thing the
user sees, and it must not claim more than the run delivered.
"""
import json
import sys
from pathlib import Path

# Stages whose failure means the study did not deliver what it promised.
# `analyze` counts: the unattended flow exists to produce an interpretation,
# and a run that stops at experiment.json has not produced one.
LOAD_BEARING = ("build_cases", "run", "extract", "package", "analyze")


def _read(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:                                           # noqa: BLE001
        return {}


def build(run_dir: str):
    rd = Path(run_dir)
    state = _read(rd / "run_state.json")
    stages = state.get("stages") or {}
    summary = _read(rd / "RUN_SUMMARY.json")

    lines = [f"study : {rd.name}", f"model : {state.get('model', '?')}",
             f"path  : {rd}", ""]

    bad = []
    for name in ExpectedOrder(stages):
        st = stages.get(name) or {}
        status = st.get("status", "—")
        extra = ", ".join(f"{k}={st[k]}" for k in
                          ("n_ok", "n_results", "n_rows", "n_experiments")
                          if k in st)
        note = st.get("error") or st.get("reason") or ""
        lines.append(f"  {name:18s} {status:8s} {extra}"
                     + (f"   {note[:80]}" if note else ""))
        if name in LOAD_BEARING and status not in ("done", "skipped"):
            bad.append(f"{name} ({status})")

    n_ok = (summary.get("n_successful") if isinstance(summary, dict) else None)
    if n_ok is not None:
        lines += ["", f"columns: {n_ok}/{summary.get('n_experiments', '?')}"]

    figs = sorted((rd / "04_analysis").glob("*.png")) if (rd / "04_analysis").is_dir() else []
    docs = sorted((rd / "04_analysis").glob("*.json")) if (rd / "04_analysis").is_dir() else []
    lines += ["", f"04_analysis: {len(docs)} json, {len(figs)} figures"]

    if bad:
        lines = [f"INCOMPLETE — {', '.join(bad)}", ""] + lines
        lines += ["", "The model output is on disk. Finish or inspect with:",
                  f"    python workflow.py --resume {rd}"]
    else:
        lines = ["OK — the analysis is written", ""] + lines
        lines += ["", f"Read: {rd / '04_analysis'}"]

    return "\n".join(lines), (1 if bad else 0)


def ExpectedOrder(stages):
    """Ledger order, with any unknown stage appended rather than dropped."""
    known = ("materialize", "build_case_inputs", "build_cases", "run",
             "extract", "package", "analyze")
    return [s for s in known if s in stages] + \
           [s for s in stages if s not in known]


REPORT_NAME = "STUDY_REPORT.txt"


def send(addr: str, subject: str, body: str) -> str:
    """Mail the summary from a COMPUTE NODE.

    Not via `mail`: that command is not installed on Compy's nodes (verified
    2026-08-03, job 770819 — "NO mail on node"). What IS there is a local
    Postfix on localhost:25 (job 770820: "220 n0002.local ESMTP Postfix") and
    /usr/sbin/sendmail. Slurm's own notifications leave by the same route, and
    those are known to arrive, so this is the path with evidence behind it
    rather than the one that looks conventional.
    """
    import smtplib, socket, getpass
    from email.message import EmailMessage
    msg = EmailMessage()
    msg["From"] = f"{getpass.getuser()}@{socket.getfqdn()}"
    msg["To"] = addr
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP("localhost", 25, timeout=30) as smtp:
            smtp.send_message(msg)
        return f"mailed {addr}"
    except Exception as e:                                      # noqa: BLE001
        # Never fatal: the report is on disk either way, and losing the run
        # over a mail failure would be absurd.
        return f"could NOT mail {addr} ({type(e).__name__}: {e})"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: notify_study.py <run_dir> [--mail <addr>]")
    body, rc = build(sys.argv[1])
    print(body)
    # Also written to the run directory. The compute node has no mail
    # transport (verified 2026-08-03, job 770819: "NO mail on node"), so the
    # body cannot always be delivered — but it can always be LEFT somewhere
    # the user can read without reconstructing it from the ledger.
    try:
        (Path(sys.argv[1]) / REPORT_NAME).write_text(body + "\n")
    except Exception as e:                                      # noqa: BLE001
        print(f"(could not write {REPORT_NAME}: {e})")
    if "--mail" in sys.argv:
        addr = sys.argv[sys.argv.index("--mail") + 1]
        tag = "OK" if rc == 0 else "INCOMPLETE"
        print(send(addr, f"[IDEAS] {tag}: {Path(sys.argv[1]).name}", body))
    sys.exit(rc)
