#!/usr/bin/env python3
"""What actually happened in a study, as an email body — and as an exit code.

    python tools/notify_study.py <run_dir>
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
    # NO mail from here. A compute node has no /bin/mail (job 770819) and its
    # local Postfix ACCEPTS messages it cannot relay — smtplib returned success
    # and nothing was ever delivered (jobs 770821, 770905). Delivery belongs to
    # Slurm, whose MailProg runs on the controller and demonstrably works; this
    # exit status is what makes its subject line meaningful.
    sys.exit(rc)
