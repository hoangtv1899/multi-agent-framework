#!/usr/bin/env python3
"""The batch settings of each machine this server submits jobs on.

A whole-domain chain takes its machine from its recipe (CIME's MACH, read from
the control's case), so its jobs ask for what that machine needs. The finalize
job (job B) and the server's own defaults take the machine this process runs
on, from current().

Compy's row holds the values every job used before this table existed, so
nothing changes there. A machine missing from the table is refused by name,
never given Compy's values: a wrong queue is refused by SLURM, but a wrong
account can be charged.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

# CIME machine name -> its batch settings.
#   tasks_per_node   MPI tasks one node holds (CIME's MAX_MPITASKS_PER_NODE),
#                    for the node count of a job
#   queue_option     the sbatch option that names the queue
#   default_queue    the queue a job uses when none is given
#   header           extra #SBATCH options every job script of the machine carries
#   default_account  the account charged when none is given; None where no
#                    account may be assumed
MACHINES: Dict[str, Dict] = {
    "compy": {"tasks_per_node": 40, "queue_option": "-p", "default_queue": "short",
              "header": [], "default_account": "e3sm"},
    # Perlmutter's CPU nodes. Every job names a QOS with -q and asks for CPU
    # nodes. No account is assumed: E3SM's own default for this machine is
    # another project's allocation.
    "pm-cpu": {"tasks_per_node": 128, "queue_option": "-q", "default_queue": "regular",
               "header": ["--constraint=cpu"], "default_account": None},
}
DEFAULT_MACHINE = "compy"


class MachineError(Exception):
    """A machine or an account this server cannot submit for, and why."""


def current() -> str:
    """The machine this process runs on. IDEAS_MACHINE when it is set; pm-cpu
    when NERSC says this is Perlmutter; otherwise Compy, where every earlier
    run was made."""
    m = (os.environ.get("IDEAS_MACHINE") or "").strip()
    if m:
        return m
    if (os.environ.get("NERSC_HOST") or "").strip() == "perlmutter":
        return "pm-cpu"
    return DEFAULT_MACHINE


def settings(machine: Optional[str] = None) -> Dict:
    """The row of a machine (this one when none is named). Raises for a machine
    the table does not hold."""
    m = machine or current()
    if m not in MACHINES:
        raise MachineError(f"no batch settings for the machine {m!r}: add its row to "
                           f"machine_settings.MACHINES (known: {', '.join(sorted(MACHINES))})")
    return MACHINES[m]


def default_queue(machine: Optional[str] = None) -> str:
    return settings(machine)["default_queue"]


def queue_args(machine: Optional[str] = None, queue: Optional[str] = None) -> List[str]:
    """The sbatch options that name the queue, such as ['-p', 'short']."""
    s = settings(machine)
    return [s["queue_option"], str(queue or s["default_queue"])]


def header_lines(machine: Optional[str] = None) -> List[str]:
    """The machine's extra #SBATCH lines, in the order they are written."""
    return [f"#SBATCH {h}" for h in settings(machine)["header"]]


def account(explicit: Optional[str] = None, machine: Optional[str] = None) -> str:
    """The account to charge. A value given, then IDEAS_SLURM_ACCOUNT, then
    the machine's default. An empty value counts as none. Raises where the
    machine assumes no account."""
    for a in (explicit, os.environ.get("IDEAS_SLURM_ACCOUNT")):
        if a is not None and str(a).strip():
            return str(a).strip()
    d = settings(machine)["default_account"]
    if d:
        return d
    raise MachineError(f"no account to charge on {machine or current()}: set "
                       f"IDEAS_SLURM_ACCOUNT to the account of the project the jobs "
                       f"should be charged to")
