#!/usr/bin/env python3
"""Which runs on disk could be continued — and what each one was about.

src/core/resumable.py

The discovery half of resumability. Phases 1-3 gave a run the ability to record
what it has done and to hand back a job id instead of blocking; this is what
finds those runs again afterwards.

It is DELIBERATELY NOT AN AGENT. "Is job 770603 finished?" is a mechanical
question that squeue answers exactly, and a model guessing at it adds a failure
mode where none is needed. So the scan is plain code, always correct, and
reception's job is to pick from what it returns rather than to produce a run
directory of its own. A hallucinated path either crashes or — worse — resumes
the wrong study and reports it as the one that was asked about.

Which is also why `request` is in every record. Run directories are named
elm_run_20260731_112621: timestamps, not topics. Two studies of the same
watershed an hour apart are indistinguishable by name, and "resume my Gunnison
run" matches neither directory unless the scan says what each run was for. The
discriminating power lives in this file, not in the caller's cleverness.
"""
import json
import sys
from datetime import datetime
from pathlib  import Path
from typing   import Any, Dict, List, Optional

sys.path.insert(0, "src")

from core.exp_manager_base import ExperimentManagerBase        # noqa: E402

STATE_FILE = ExperimentManagerBase.STATE_FILE
STAGES     = ExperimentManagerBase.STAGES
TERMINAL   = ExperimentManagerBase.TERMINAL_STAGE


def _read_json(p: Path) -> Optional[Dict[str, Any]]:
	"""Never raises. A directory being scanned is not a directory under our
	control — half-written files and hand-edited JSON are both ordinary."""
	try:
		if p.exists():
			d = json.loads(p.read_text())
			return d if isinstance(d, dict) else None
	except Exception:                                           # noqa: BLE001
		pass
	return None


def _age(iso: Optional[str]) -> Optional[str]:
	"""'14 min', '2 h', '3 d' — how long ago, in the largest honest unit."""
	if not iso:
		return None
	try:
		dt = datetime.fromisoformat(str(iso))
	except Exception:                                           # noqa: BLE001
		return None
	s = (datetime.now() - dt).total_seconds()
	if s < 90:
		return f"{int(s)} s"
	if s < 5400:
		return f"{int(s / 60)} min"
	if s < 172800:
		return f"{int(s / 3600)} h"
	return f"{int(s / 86400)} d"


def _mtime(run_dir: Path) -> Optional[str]:
	"""When the run dir was last touched — the only clock a run with no
	run_state.json has. Reads the directory, not a walk of it: a 19-column ELM study has
	thousands of files and this runs on every candidate."""
	try:
		return datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat()
	except Exception:                                           # noqa: BLE001
		return None


def _model_from_name(name: str) -> Optional[str]:
	"""'elm_run_20260731_112621' → 'elm'. Only for runs with no run state, where
	the directory name is the sole surviving statement of what ran — which is
	exactly why run dirs stopped being called elm_run_* for every backend."""
	head = name.split("_run_")[0]
	return head.replace("_", "-") if head and head != name else None


def _request_text(run_dir: Path) -> Optional[str]:
	"""What the user actually asked for, from reception.json.

	The one field that makes two same-day runs of the same watershed tellable
	apart. Falls back through the brief, because reception's shape has moved
	once already and an older run should still be identifiable.
	"""
	d = _read_json(run_dir / "reception.json") or {}
	for key in ("user_request", "request", "question"):
		v = d.get(key)
		if isinstance(v, str) and v.strip():
			return v.strip()
	brief = d.get("brief")
	if isinstance(brief, dict):
		v = brief.get("user_request")
		if isinstance(v, str) and v.strip():
			return v.strip()
	return None


def _manager_for(model: str):
	"""The manager class for a backend name, or None if there is not one.

	The last of what backends.get() did, and deliberately not a table: two
	names, two imports, and a new model adds one line here rather than an
	entry in a registry that also has to be kept in step with the servers.
	"""
	name = (model or "").strip().lower()
	# BOTH MANAGERS LIVE BESIDE THEIR SERVER, under mcp/<server>-mcp/, and are
	# imported by path: src/core/ names no model (ELM moved 2026-08-10,
	# PFLOTRAN 2026-08-18). A new model adds one branch here.
	import sys
	from pathlib import Path as _P
	mcp_root = _P(__file__).resolve().parents[2] / "mcp"
	if name == "pflotran":
		d = mcp_root / "pflotran-mcp"
		if d.is_dir() and str(d) not in sys.path:
			sys.path.append(str(d))
		from pflotran_exp_manager import PFLOTRANExpManager
		return PFLOTRANExpManager
	if name == "elm":
		d = mcp_root / "elm-mcp" / "src"
		if d.is_dir() and str(d) not in sys.path:
			sys.path.append(str(d))
		from elm_exp_manager import ELMExpManager
		return ELMExpManager
	return None


def _stages_for(model: Optional[str]) -> tuple:
	"""The stages this backend actually runs.

	PFLOTRAN declares NEEDS_CASE_BUILD = False — deck generation IS its build — so
	`build_cases` is never recorded for a PFLOTRAN run and is not missing when it
	is absent. Reading the run state without asking the backend reports `build_cases`
	as the outstanding stage of every interrupted PFLOTRAN study, forever.
	"""
	if not model:
		return STAGES
	# ASKED OF THE MANAGER, BY NAME (2026-08-17). This went through
	# core/backends.py, which was deleted on 2026-08-16 — so the lookup raised,
	# the except swallowed it, and every backend fell back to "assume all
	# stages". A PFLOTRAN run that had finished its decks reported `build_cases`
	# as outstanding, which is the exact failure this function exists to
	# prevent, arrived at by the fallback rather than by the bug.
	try:
		cls = _manager_for(model)
	except Exception:                                           # noqa: BLE001
		return STAGES                                # unknown backend: assume all
	if cls is None:
		return STAGES
	return tuple(s for s in STAGES
				 if s != "build_cases" or getattr(cls, "NEEDS_CASE_BUILD", True))


def _next_stage(stages: Dict[str, Any], model: Optional[str] = None
				) -> Optional[str]:
	"""The first stage this backend runs that is not recorded done."""
	for s in _stages_for(model):
		if (stages.get(s) or {}).get("status") != "done":
			return s
	return None


def inspect_run(run_dir: Path, check_jobs: bool = False) -> Dict[str, Any]:
	"""One directory, described. Always returns a record, never raises.

	`resumable` is the verdict and `why` is the reason for it, in both
	directions — a caller that finds nothing to resume should be able to say
	WHY rather than just showing an empty list.
	"""
	run_dir = Path(run_dir)
	rec: Dict[str, Any] = {
		"run_dir":   str(run_dir),
		"name":      run_dir.name,
		"model":     None,
		"request":   _request_text(run_dir),
		"stage":     None,
		"status":    None,
		"job_id":    None,
		"job_state": None,
		"updated":   None,
		"age":       None,
		"resumable": False,
		"why":       "",
	}

	state = _read_json(run_dir / STATE_FILE)
	if state is None:
		# Every run made before the run state existed lands here. Re-entering one
		# with resume=True would find an empty run state and redo everything,
		# which is not resuming — so say so rather than offer it.
		#
		# Still worth describing properly: these are most of what is on disk
		# right now, and a person choosing between them needs to know which
		# ones finished. The name and the mtime are all there is to go on, and
		# both are honest.
		rec["model"] = _model_from_name(run_dir.name)
		rec["age"]   = _age(_mtime(run_dir))
		if (run_dir / "experiment.json").exists():
			rec.update(stage=TERMINAL, status="done")
			rec["why"] = "complete (experiment.json was written), no run state"
		else:
			rec["why"] = ("incomplete AND predates the run state — nothing "
						  "records what it finished, so it would have to be "
						  "re-run rather than resumed")
		return rec

	stages = state.get("stages") or {}
	rec["model"]   = state.get("model")
	rec["updated"] = state.get("updated")
	rec["age"]     = _age(state.get("updated"))

	# ANY stage may be waiting on a job, not just `run`. Since D1 the CIME case
	# build is sbatch'd too, and a study parked at `build_cases` looks identical in
	# the run state — but "your cases are being built" and "your ensemble is
	# simulating" are hours apart in what happens next, so the stage is named.
	for name in _stages_for(rec["model"]):
		entry = stages.get(name) or {}
		if entry.get("status") != "pending":
			continue
		jid = entry.get("job_id")
		rec.update(stage=name, status="pending", job_id=jid, resumable=True)
		rec["why"] = f"{name}: job {jid} was submitted and has not been collected"
		if check_jobs and jid:
			st = ExperimentManagerBase._slurm_state(jid)
			rec["job_state"] = st
			if st is None:
				rec["why"] += " (the scheduler will not say what it is doing)"
			elif st in ExperimentManagerBase.ACTIVE_JOB_STATES:
				rec["why"] = f"{name}: job {jid} is {st}"
			else:
				rec["why"] = (f"{name}: job {jid} finished ({st}) — ready to "
							  f"collect")
		return rec

	nxt = _next_stage(stages, rec["model"])
	if nxt is None or (stages.get(TERMINAL) or {}).get("status") == "done":
		# _package wrote experiment.json, which is what the run is FOR. A
		# failed analyze leaves a finished study with no report, and re-running
		# the whole pipeline is not the way to get one.
		rec.update(stage=TERMINAL, status="done")
		rec["why"] = "complete (experiment.json was written)"
		return rec

	if not stages:
		rec["why"] = "run state is empty — nothing has been recorded as done"
		return rec

	done = [s for s in _stages_for(rec["model"])
			if (stages.get(s) or {}).get("status") == "done"]
	rec.update(stage=done[-1] if done else None, status="done",
			   resumable=True)
	rec["next"] = nxt
	rec["why"]  = (f"stopped after {done[-1]}; {nxt} has not run"
				   if done else f"{nxt} has not run")
	return rec


def find_resumable(output_dir: str = "./workflow_outputs",
				   check_jobs: bool = False,
				   include_all: bool = False) -> List[Dict[str, Any]]:
	"""Every run under output_dir that a later session could continue.

	Newest first. `include_all` returns the complete ones and those with no run state
	too, each carrying its own `why` — which is what lets a caller print
	"4 runs, none resumable, all predate the run state" instead of nothing.

	check_jobs costs one squeue (and possibly one sacct) per pending run, so it
	is opt-in; it is worth paying interactively and not worth paying inside a
	loop.
	"""
	base = Path(output_dir)
	if not base.is_dir():
		return []

	out = []
	for d in sorted(base.iterdir(), reverse=True):
		if not d.is_dir():
			continue
		# A run directory is one that left one of the three things only a run
		# leaves. Skips the eval outputs, probe scripts and scratch dirs that
		# share this parent without being runs.
		if not any((d / f).exists() for f in
				   (STATE_FILE, "reception.json", "RUN_SUMMARY.json",
					"experiment.json")):
			continue
		rec = inspect_run(d, check_jobs=check_jobs)
		if rec["resumable"] or include_all:
			out.append(rec)
	return out


def describe(rec: Dict[str, Any], width: int = 68) -> str:
	"""One run, as two or three lines a person can choose between.

	The request text is the point — see the module docstring. Truncated rather
	than wrapped so a list of runs stays a scannable column.
	"""
	head = f"{rec['name']}  [{rec.get('model') or '?'}]"
	if rec.get("age"):
		head += f"  {rec['age']} ago"
	bits = []
	why = rec.get("why") or ""
	# The job bit only when `why` does not already name the job — with
	# check_jobs on it usually does, and printing both read as
	# "job 770680 RUNNING; job 770680 is RUNNING".
	if rec.get("job_id") and str(rec["job_id"]) not in why:
		bits.append(f"job {rec['job_id']}"
					+ (f" {rec['job_state']}" if rec.get("job_state") else ""))
	if why:
		bits.append(why)
	req = rec.get("request")
	if req:
		req = req if len(req) <= width else req[:width - 1].rstrip() + "…"
	return (f"{head}\n"
			f"    {'; '.join(bits)}\n"
			+ (f"    “{req}”\n" if req else ""))


def print_listing(rows: List[Dict[str, Any]], output_dir: str = "") -> None:
	"""The runs on disk, numbered, each with the reason it is or is not
	resumable — so the choice is the user's and it is an informed one.

	Prints the NOT-resumable ones too, greyed out by position rather than
	hidden. An empty list on its own is indistinguishable from a broken scan,
	and "3 runs, all complete" is a different answer from "no runs at all".
	"""
	live = [r for r in rows if r.get("resumable")]
	rest = [r for r in rows if not r.get("resumable")]

	if not rows:
		print(f"\nNo runs found under {output_dir or 'the output directory'}.\n")
		return

	print(f"\n{'=' * 72}")
	print(f"RESUMABLE RUNS in {output_dir}" if output_dir else "RESUMABLE RUNS")
	print("=" * 72)

	if live:
		for i, r in enumerate(live, 1):
			print(f"\n[{i}] {describe(r)}", end="")
		print(f"\nResume one with:")
		print(f"    python workflow.py --resume {live[0]['run_dir']}")
	else:
		print("\nNothing can be resumed right now.")

	if rest:
		print(f"\n{'-' * 72}")
		print(f"{len(rest)} other run(s), not resumable:")
		for r in rest:
			why = r.get("why", "")
			age = f"  {r['age']} ago" if r.get("age") else ""
			print(f"    {r['name']}{age}\n        {why}")
	print()
