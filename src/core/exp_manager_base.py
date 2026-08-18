#!/usr/bin/env python3
"""
Experiment Manager — the model-agnostic half
src/core/exp_manager_base.py

The third box. It takes reception.json and strategy.json and turns them into
compute, then hands a results package to the Analyzer.

What lives HERE is everything true of any land-surface or subsurface model:

    check         compare the strategy against reception before spending
                  anything — the last moment both are in hand
    materialize   strategy → concrete columns (coordinates, elevation bands,
                  soil, water table) and the sampling-design figure
    package       run summary + the LLM input payload

What lives in a BACKEND subclass is everything that knows a model's file
formats and job model: how to warm-start it, how to build a case, how to
submit it, and how to read its history files.

The split matters because ELM and PFLOTRAN differ in every one of those and in
none of the first set. A PFLOTRAN study of the same watershed samples the same
columns from the same DEM against the same strategy; only the case directory
and the input deck differ. Sharing the sampler is also what makes coupling
meaningful — a PFLOTRAN run over columns ELM already produced is only a
comparison if both used one materialisation.

Two ordering constraints are structural, not stylistic, and the base enforces
them so no backend can get them wrong:

  1. _refine_columns() runs BEFORE columns.json is written. A warm start snaps
     each column to its donor gridcell, so a file written earlier describes a
     plan that does not match the run — every column of the 2026-07-28 run was
     200-700 m from where columns.json claimed, and both the spatial maps and
     the nearest-station matching read that file.
  2. The sampling-design figure is drawn AFTER the refinement, for the same
     reason: it was showing a design that had already been superseded.

Subclasses implement:
    _refine_columns(columns, config) -> Dict   backend edits to the columns
                                               (warm start, donor soil); the
                                               returned dict is passed on to
                                               _to_run_plan. Default: no-op.
    _to_run_plan(plan, columns, config, refine) -> Dict   columns → the
                                               backend's executable plan.
    _build_case_inputs / _build_cases / _run / _extract        the compute stages.
"""
import json
import os
import sys
from pathlib  import Path
from datetime import datetime
from typing   import Any, Dict, List, Optional

sys.path.insert(0, "src")

# The key lists below are KeySets, not tuples: they raise when a producer
# writes a key they neither keep nor drop, instead of dropping it in silence.
#
# IMPORTED AS `core.keyset` EVERYWHERE, including from the MCP. The same file
# reached by two path roots is two module objects with two distinct exception
# classes, so `except UnclassifiedKey` in one importer does not catch the one
# raised in the other. One spelling, one identity.
from core import keyset                                         # noqa: E402
from core.keyset import KeySet                                  # noqa: E402

# Used only when neither the caller nor the strategy says. The planner
# always emits n_bands, so reaching this means the plan was hand-written.
DEFAULT_BANDS = 4

# The sampling approaches the manager can materialise. `elevation_bands` is the
# spatial sampler; `factor_sweep` asks the backend for columns instead.
FACTOR_SWEEP = "factor_sweep"


def _is_factor_sweep(strategy: Dict[str, Any],
					 brief: Optional[Dict[str, Any]] = None) -> bool:
	"""Is this a controlled sweep rather than a spatial sample?

	TWO SOURCES, EITHER SUFFICIENT. `sampling.approach` is the planner's word
	and `design_archetype` is reception's, and they are cross-checked in
	strategy_check rather than here. Requiring both to agree would make a
	materialize fail on a disagreement the gate has already reported and
	corrected; requiring neither would send a conceptual design to a sampler
	that needs a basin.

	Defaults to False on anything unrecognised, so a malformed strategy takes
	the path that has always existed rather than a new one.
	"""
	sampling = (strategy or {}).get("sampling") or {}
	if str(sampling.get("approach") or "").strip().lower() == FACTOR_SWEEP:
		return True
	arche = ((strategy or {}).get("archetype")
			 or (brief or {}).get("design_archetype") or "")
	return str(arche).strip().lower() == "conceptual"


class Pending:
	"""What _run returns when it SUBMITTED the ensemble instead of finishing it.

	A backend that queues work has two honest things it can do at the end of
	_run: block until the scheduler is done, or say "here is the job id". The
	first is what ELM did, and it is why a 40-minute queue wait held a Python
	process — and a session — hostage. This is the second.

	It is a distinct TYPE rather than a dict with a status key because _run's
	success shapes are already loose (ELM returns {case: bool}, PFLOTRAN a list
	of dicts), and a dict carrying "status": "pending" would be indistinguishable
	from a backend that happened to key its results that way. isinstance is not
	guessable.

	  job_id   what the scheduler called it — the one thing a later session
	           needs, and what gets written into the run state
	  detail   anything else the backend wants back when it is polled; it is
	           stored verbatim in the run-state entry and handed to _poll
	"""

	def __init__(self, job_id: Any, **detail: Any):
		self.job_id = str(job_id)
		self.detail = detail

	def __repr__(self) -> str:                                  # pragma: no cover
		return f"Pending(job_id={self.job_id!r}, {self.detail!r})"


class ExperimentManagerBase:
	"""Sampling, the strategy gate, and packaging — no model in sight."""

	MODEL = "experiment"          # backends override; names the run dir

	def __init__(self,
				 base_output_dir: str = "./workflow_outputs",
				 run_dir: str = None):
		"""
		run_dir  an EXISTING directory to run in. The coordinator creates it and
		         writes reception.json and strategy.json into it as each stage
		         finishes, so every file is written by whatever produced it and
		         this stage only ever READS its inputs. It also means a crash
		         here leaves those two intact.

		         Omitted (standalone / CLI use), one is minted as before.
		"""
		self.base_output_dir = Path(base_output_dir)
		if run_dir:
			self.run_dir = Path(run_dir)
		else:
			timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
			self.run_dir = self.base_output_dir / f"{self.MODEL}_run_{timestamp}"
		self.run_dir.mkdir(parents=True, exist_ok=True)

		# Four numbered subdirs — the same names for every backend, so a
		# coupled study has one directory layout rather than two.
		self.input_dir       = self.run_dir / "01_inputs"
		self.setup_plots_dir = self.run_dir / "02_setup_plots"
		self.results_dir     = self.run_dir / "03_results"
		self.analysis_dir    = self.run_dir / "04_analysis"

		for d in [self.input_dir, self.setup_plots_dir,
				  self.results_dir, self.analysis_dir]:
			d.mkdir(exist_ok=True)

		self.strategy_report = None

	# ─────────────────────────────────────────────────────────
	# STEP 0a — THE GATE
	# ─────────────────────────────────────────────────────────
	def check(self,
			  plan:   Dict[str, Any],
			  config: Dict[str, Any]) -> Dict[str, Any]:
		"""Compare strategy against reception. Returns a possibly-updated config.

		Here rather than at the planner hand-off because this is where cost
		begins: everything upstream is one LLM call, everything downstream is a
		case build and a queue slot per column. STOP conditions raise;
		corrections are recorded in the run and the run continues.
		"""
		from core.strategy_check import check as _check, render as _render
		reception = config.get("reception") or {"brief": config.get("brief") or {}}
		report, fixed = _check(reception, config.get("strategy") or plan)
		print("🔎 STEP 0: Checking strategy against reception")
		print(_render(report))
		if not report["ok"]:
			raise ValueError("strategy does not agree with reception: "
							 + "; ".join(report["stop"]))
		if report["corrections"]:
			config = {**config, "strategy": fixed}
		self.strategy_report = report
		return config

	# ─────────────────────────────────────────────────────────
	# BACKEND HOOKS
	# ─────────────────────────────────────────────────────────
	# ─────────────────────────────────────────────────────────
	# THE STAGE SEQUENCE — one implementation, all backends
	# ─────────────────────────────────────────────────────────
	# Backends declare what they need rather than stubbing stages they do not
	# have. A no-op _build_cases() reads as "built no cases, successfully", which
	# is the silent-success pattern this codebase keeps rediscovering; a
	# declaration reads as "this model has no such stage".
	NEEDS_CASE_BUILD   = True    # ELM compiles CIME cases; PFLOTRAN writes decks
	NEEDS_SCHEDULER = True    # ELM submits to SLURM; PFLOTRAN runs in 0.3 s
	COUPLES_TO      = None    # backend name this one hands its output to

	# ─────────────────────────────────────────────────────────
	# THE RUN STATE — what this study has already finished, on disk
	# ─────────────────────────────────────────────────────────
	#
	# One small file per study. A checklist: each stage is ticked off as it
	# finishes, with the job id it was waiting on and the artifacts it left
	# behind. It is A HANDOFF NOTE, not a progress bar — nothing reads it while
	# a study is running in one process; it exists so the NEXT process can pick
	# the study up.
	#
	# Written for one reason: an ELM ensemble is ~40 minutes in a queue, and a
	# process that has to sit and block for it is a process that cannot be
	# interrupted, resumed, or moved between sessions. The run state is what
	# lets a later invocation know that materialize and build are already done
	# and the only thing outstanding is a job id.
	#
	# (Called "the ledger" until 2026-08-10. Renamed because that word already
	# names a DIFFERENT object here — the assumptions ledger in assumptions.json,
	# which records the choices a study made rather than the steps it finished.
	# One word for two files is one word too few.)
	#
	# IT IS LOAD-BEARING NOW — the comment here said "phase 1 writes it and
	# nothing reads it", which stopped being true with jobs A and B. The
	# framework submits both and EXITS; job B starts hours later, on another
	# node, in another process, and this file is the only thing carrying the job
	# id and the finished-stage record across that gap. A one-job study needs it
	# MORE than the five-stage pipeline it was built for, not less
	# (docs/EXP_MANAGER_ELM.md §8).
	STATE_FILE = "run_state.json"

	# The stage sequence, in order, as execute_plan runs it. Named here so the
	# run state's reader and its writer cannot disagree about what a stage is
	# called or which one comes next — a scan that thought "extract" preceded
	# "run" would report the wrong thing as outstanding.
	STAGES = ("materialize", "build_case_inputs", "build_cases", "run",
			  "extract", "package", "analyze")

	# The stage that means "this study produced its product". _package writes
	# experiment.json, which is what the Analyzer reads and what the run is
	# for; analyze is non-fatal and a run whose report failed is still finished.
	TERMINAL_STAGE = "package"

	def _state_path(self) -> Path:
		return self.run_dir / self.STATE_FILE

	def _load_state(self) -> Dict[str, Any]:
		"""The run state, or an empty one. Never raises — a corrupt or absent
		run state must degrade to "nothing is known to be done", which is the
		same as a fresh run, rather than taking the run down."""
		p = self._state_path()
		if not p.exists():
			return {"model": self.MODEL, "run_dir": str(self.run_dir),
					"stages": {}}
		try:
			d = json.loads(p.read_text())
			if isinstance(d, dict) and isinstance(d.get("stages"), dict):
				return d
		except Exception:                                       # noqa: BLE001
			pass
		print(f"   ⚠️  {self.STATE_FILE} unreadable — treating this run as fresh")
		return {"model": self.MODEL, "run_dir": str(self.run_dir), "stages": {}}

	def _mark(self, stage: str, status: str = "done", **fields) -> None:
		"""Record one stage. Never raises: the run state is bookkeeping, and a
		failure to write it must not lose a stage that actually completed."""
		try:
			st = self._load_state()
			st["model"] = self.MODEL
			st["run_dir"] = str(self.run_dir)
			entry = dict(st["stages"].get(stage) or {})
			entry.update(status=status, at=datetime.now().isoformat(), **fields)
			st["stages"][stage] = entry
			st["updated"] = entry["at"]
			self._state_path().write_text(json.dumps(st, indent=2, default=str))
		except Exception as e:                                  # noqa: BLE001
			print(f"   ⚠️  could not record stage '{stage}' ({e})")

	def _artifacts(self, *names: str) -> List[str]:
		"""Of the named files, the ones that actually exist. Recording a file
		that was never written would make the run state a claim rather than a
		record, and a later resume would trust it."""
		return [n for n in names if (self.run_dir / n).exists()]

	# ─────────────────────────────────────────────────────────
	# REHYDRATION — a finished stage's output, read back off disk
	# ─────────────────────────────────────────────────────────
	#
	# Written HERE and not per backend. ELM's cases.json is a list of case
	# DIRECTORY PATHS while _build_case_inputs returns a list of dicts, so reconstructing
	# from it would be lossy and backend-specific; the base already holds
	# _build_case_inputs's return value, so persisting that is both uniform and exact.
	CASE_INPUTS = "case_inputs.json"

	def _serialise_case_inputs(self, experiments: List[Dict]) -> List[Dict]:
		"""The experiments as PLAIN DATA, for the case-inputs file.

		The mirror of _rehydrate_handles: that drops the dead reprs on the way
		back in, and this stops them being written on the way out. A backend
		whose experiments hold a live object must override, or json.dumps'
		default=str turns it into its repr — truthy, attribute-free, and
		useless to anything that reads the file.

		Not academic: ELM keeps its whole runtime_config inside an adapter
		object, so the default here wrote a case list with no FSURDAT, no
		FINIDAT and no domain paths. The file existed, looked plausible, and
		named nothing the case build needs.
		"""
		return experiments

	def _save_case_inputs(self, experiments: List[Dict]) -> None:
		try:
			(self.input_dir / self.CASE_INPUTS).write_text(
				json.dumps(self._serialise_case_inputs(experiments),
						   indent=2, default=str))
		except Exception as e:                                  # noqa: BLE001
			print(f"   ⚠️  could not persist the case inputs ({e}) — this "
				  f"run cannot be resumed past _build_case_inputs")

	def _rehydrate_materialize(self) -> Optional[Dict[str, Any]]:
		"""The executable plan _materialize produced (run_plan.json)."""
		p = self.run_dir / "run_plan.json"
		return json.loads(p.read_text()) if p.exists() else None

	def _rehydrate_case_inputs(self) -> Optional[List[Dict]]:
		"""_build_case_inputs's experiments. Paths come back as STRINGS, which every
		consumer already tolerates — _run and _extract both wrap them in
		Path() rather than assuming the type."""
		p = self.input_dir / self.CASE_INPUTS
		if not p.exists():
			return None
		d = json.loads(p.read_text())
		return d if isinstance(d, list) else None

	def _rehydrate_handles(self, experiments: List[Dict],
						   plan: Dict[str, Any], config: Dict[str, Any]) -> None:
		"""Restore the live objects a rehydrated build is missing.

		The manifest is JSON, so anything in an experiment dict that was an
		OBJECT comes back as its repr — truthy, attribute-free, and happy to be
		passed around until something calls a method on it. A backend that put
		a model handle in there gets a hook to fix that up (or to drop it) here,
		before any stage can mistake the string for the thing.

		Default: nothing to restore. PFLOTRAN's experiments are pure data.
		"""
		return None

	# ─────────────────────────────────────────────────────────
	# THE EXTRACT CONTRACT
	# ─────────────────────────────────────────────────────────
	# A stage's output is DATA, not a live object. This one was the last
	# holdout: _extract handed back an ELMResultsAnalyzer and the base read it
	# with getattr, which looks like coupling and mostly was not — the Analyzer
	# ignores the object entirely (it reads experiment.json), and the resume
	# path has always rebuilt a SimpleNamespace from that same file. A plain
	# dict was already standing in every time a run was resumed.
	#
	# Saying so in one place makes the stage boundary crossable: a dict can go
	# through a tool call, an object cannot.
	#
	# `llm_input` LEFT THE CONTRACT 2026-08-13. It carried a prompt payload the
	# ELM results object packed for the deleted report agent; a stage boundary
	# is a place to hand over MEASUREMENTS, and nothing else consumed it.
	EXTRACT_KEYS = ("rows", "units", "extra_summary", "spinup_dropped")

	@staticmethod
	def _as_extract(obj: Any) -> Dict[str, Any]:
		"""Whatever a backend built → the stage's JSON contract.

		Accepts a dict already in the contract, or an analyzer-shaped object,
		so a backend can be converted without its internals changing on the
		same day the contract does.
		"""
		if isinstance(obj, dict) and "rows" in obj:
			return {k: obj.get(k) for k in
					ExperimentManagerBase.EXTRACT_KEYS} | {
					k: v for k, v in obj.items()
					if k not in ExperimentManagerBase.EXTRACT_KEYS}
		units: Dict[str, Any] = {}
		# ELMResultsAnalyzer puts units in summary['units']; a bare namespace
		# exposes .units. Reading only the second gave an EMPTY units block on
		# the 2019 Gunnison run — a package that states no units.
		for src in (getattr(obj, "units", None),
					(getattr(obj, "summary", None) or {}).get("units")):
			if isinstance(src, dict) and src:
				units = dict(src)
				break
		# ELMResultsAnalyzer.results is Dict[str, Dict] KEYED BY CASE NAME;
		# PFLOTRAN's is a list. list() on the dict yields the case NAMES, and
		# _package then drops every non-dict — job 770923 packaged
		# columns_total: 0 from 19 clean columns, while the run state recorded
		# n_rows=19 because the COUNT was right. _extract_rows was fixed for
		# exactly this and the fix was not carried to its sibling here.
		return {"rows": ExperimentManagerBase._extract_rows(obj),
				"units": units,
				"extra_summary": getattr(obj, "extra_summary", None) or {},
				"spinup_dropped": getattr(obj, "spinup_dropped", None) or {}}

	@staticmethod
	def _extract_rows(res: Any) -> List[Dict]:
		"""The rows, from the contract or from a legacy object.

		A backend may hand back rows as a LIST or as a dict keyed by case name
		— _package has accepted both since PFLOTRAN landed. An earlier version
		of this helper called list() on the dict, which yields the case NAMES
		and reported a full ensemble as zero columns.
		"""
		raw = (res.get("rows") if isinstance(res, dict) and "rows" in res
			   else res if isinstance(res, dict)
			   else getattr(res, "results", None))
		if isinstance(raw, dict):
			raw = list(raw.values())
		return list(raw or [])

	def _rehydrate_extract(self):
		"""_extract's contract, from experiment.json — which _package already
		writes and which carries the rows and their units verbatim.

		This used to build a SimpleNamespace to imitate the live analyzer. It
		no longer has to imitate anything: a resumed run and a fresh one now
		hand the same dict downstream, which is what made the object look like
		a boundary when it never was one.
		"""
		p = self.run_dir / "experiment.json"
		if not p.exists():
			return None
		d = json.loads(p.read_text())
		rows = d.get("columns")
		if not isinstance(rows, list):
			return None
		return {"rows": rows,
				"units": dict(d.get("variable_units") or {}),
				"extra_summary": d.get("extra_summary") or {},
				"spinup_dropped": d.get("spinup_dropped") or {}}

	def execute_plan(self,
					 experiment_plan: Dict[str, Any],
					 config:          Dict[str, Any]
					 ) -> Dict[str, Any]:
		"""materialize → build_case_inputs → [build_cases] → run → extract → package → analyze.

		Lifted out of ELMExpManager so a second backend cannot re-implement it.
		Two orderings here are structural and the reason this is a template
		method rather than something each backend writes:

		  * _package BEFORE the Analyzer. experiment.json is this box's product
		    and the Analyzer's input. Ordered the other way, a crash in the
		    Analyzer took the results package with it — the ensemble was
		    computed and nothing on disk said so.
		  * every stage from _package onward is NON-FATAL. By then the compute
		    has succeeded and _extract has written its summary; a reporting bug
		    must not discard an ensemble that cost a queue slot and an hour.

		Backends supply _build_case_inputs/_run/_extract, and _build_cases only if they have
		one. Nothing below knows what model is running.
		"""
		start_time = datetime.now()
		model = self.MODEL.upper()

		# RESUME IS OPT-IN. A caller re-running execute_plan against a
		# directory usually means "do it again"; only a caller that has been
		# told to continue an interrupted run means "skip what is done". The
		# cost of guessing wrong in one direction is a wasted ensemble, and in
		# the other a study that silently reuses stale compute — so it is
		# asked for explicitly rather than inferred from the directory.
		resume = bool(config.get("resume"))
		state = self._load_state() if resume else {"stages": {}}

		def _done(stage: str) -> bool:
			return (state["stages"].get(stage) or {}).get("status") == "done"

		def _reuse(stage: str, what: str) -> None:
			print(f"↻ {stage}: already done — reusing {what}")

		try:
			# Step 0 — strategy → concrete columns → executable plan.
			rehydrated = self._rehydrate_materialize() if _done("materialize") else None
			if rehydrated is not None:
				_reuse("materialize", "run_plan.json")
				experiment_plan = rehydrated
			else:
				experiment_plan = self._materialize(experiment_plan, config)
			self._mark("materialize", artifacts=self._artifacts(
				"columns.json", "run_plan.json", "plan.json",
				"assumptions.json", "reception.json", "reception_brief.json"))

			print(f"📋 STEP 1: Building Case Inputs")
			print("-" * 40)
			experiments = self._rehydrate_case_inputs() if _done("build_case_inputs") else None
			if experiments is not None:
				_reuse("build_case_inputs", f"{len(experiments)} experiment(s) from "
							   f"{self.CASE_INPUTS}")
				self._rehydrate_handles(experiments, experiment_plan, config)
			else:
				experiments = self._build_case_inputs(experiment_plan, config)
				self._save_case_inputs(experiments)
			self._mark("build_case_inputs", n_experiments=len(experiments or []),
					   artifacts=self._artifacts("cases.json", "cases_all.json"))

			if self.NEEDS_CASE_BUILD:
				print("\n⚙️  STEP 2: Building Cases")
				print("-" * 40)
				if _done("build_cases"):
					_reuse("build_cases", "the cases already on disk")
				else:
					# Job-shaped too, since D1: an 8-10 minute CIME build is
					# sbatch'd rather than run on a login node, so _build_cases may
					# hand back a job id exactly as _run does.
					out = self._advance(
						"build_cases", lambda: self._build_cases(experiments, config),
						state=state, resume=resume,
						experiments=experiments, config=config,
						n_experiments=len(experiments or []))
					if out is self.STOP:
						return self._pending_summary(
							experiment_plan, experiments,
							self._load_state()["stages"]["build_cases"], start_time,
							stage="build_cases")
					# _build_cases mutates `experiments` in place (it sets case_dir),
					# so its return value is not the carrier — but a POLLED
					# build_cases came back from a different session and has to
					# re-attach the case dirs it found.
					if isinstance(out, list) and out:
						experiments = out
						self._save_case_inputs(experiments)

			print(f"\n🌿 STEP 3: Running Simulations")
			print("-" * 40)
			# Three ways to arrive here: nothing has run; the ensemble already
			# ran; or an earlier session SUBMITTED it and left. The third is the
			# whole point of the run state.
			if _done("run"):
				_reuse("run", "the completed ensemble")
				results = experiments
			else:
				results = self._advance(
					"run", lambda: self._run(experiments, config),
					state=state, resume=resume,
					experiments=experiments, config=config)
				if results is self.STOP:
					return self._pending_summary(
						experiment_plan, experiments,
						self._load_state()["stages"]["run"], start_time)

			# Reading the model's own output format is the backend's job;
			# saying what the numbers MEAN is not, and everything past this
			# line belongs to the Analyzer.
			print("\n📊 STEP 4: Extracting Results")
			print("-" * 40)
			analyzer = self._rehydrate_extract() if _done("extract") else None
			if analyzer is not None:
				_reuse("extract", f"{len(self._extract_rows(analyzer))} row(s) "
								  f"from experiment.json")
			else:
				analyzer = self._as_extract(self._extract(
					experiments, plan=experiment_plan, config=config))
			self._mark("extract", n_rows=len(self._extract_rows(analyzer)))

			print("\n📦 STEP 4b: Packaging Results")
			print("-" * 40)
			try:
				self._package(experiment_plan, analyzer, config)
				self._mark("package",
						   artifacts=self._artifacts("experiment.json"))
			except Exception as e:                              # noqa: BLE001
				print(f"   ✗ PACKAGING FAILED ({e}) — the results are still "
					  f"in 04_analysis/")
				self._mark("package", status="failed", error=str(e)[:200])

			print("\n🔭 STEP 4c: Analyzer")
			print("-" * 40)
			# NOTHING SUCCEEDED — there is nothing to interpret.
			#
			# The Analyzer is four LLM calls; on a 0/2 ensemble it spent 140 s
			# and ~22 k tokens to conclude, correctly, that it had no data. It
			# was doing its job: steps 2 and 3 are built to withhold claims when
			# the evidence is absent, so they withhold, at full price.
			#
			# _package still runs (above): experiment.json is the record that
			# the run failed, and skipping THAT would lose the only account of
			# what happened. Only the interpretation is skipped, and the run state
			# says why so a reader does not think the Analyzer crashed.
			n_ok = sum(1 for v in self._outcome_map(results).values() if v)
			if experiments and not n_ok:
				print(f"   ⏭  skipped — 0/{len(experiments)} columns produced "
					  f"output, so there is nothing to interpret")
				self._mark("analyze", status="skipped",
						   reason="no successful columns")
			else:
				try:
					from agents.analyzer import Analyzer
					# THE RETURN VALUE IS THE ANSWER. Analyzer.run() reports a
					# failed step by RETURNING {"error": ...}, not by raising —
					# so discarding it marked a run "analyze: done" whose
					# 04_analysis held one partial file and whose step 0 had
					# printed "❌ context failed" to a console nobody was
					# watching. Harmless while a human sat at the terminal; not
					# harmless once the unattended flow mails "your analysis is
					# ready" off the back of this run-state entry.
					st = Analyzer(str(self.run_dir)).run(results=analyzer,
														 config=config) or {}
					steps = st.get("steps") if isinstance(st, dict) else None
					# A STEP THAT FAILED IS A FAILURE, even when the run
					# carried on. Reading only status["error"] was not enough:
					# steps 2-3 report their own collapse by setting
					# steps["investigate"]=False and nothing else, so job
					# 770905 lost its entire interpretation to a 503 from the
					# gateway — 0 LLM calls, verdict None — and was recorded
					# "analyze: done", mailed to the user as "OK — the analysis
					# is written". Exactly the failure the error check was
					# added to prevent, one level finer.
					failed_steps = [k for k, v in (steps or {}).items()
									if v is False]
					if isinstance(st, dict) and (st.get("error") or failed_steps):
						why = st.get("error") or \
							("these steps failed: " + ", ".join(failed_steps))
						print(f"   ⚠️  analyzer incomplete: {why} — "
							  f"experiment.json stands")
						self._mark("analyze", status="failed",
								   error=str(why)[:200], steps=steps,
								   failed_steps=failed_steps or None)
					else:
						self._mark("analyze", steps=steps)
				except Exception as e:                          # noqa: BLE001
					print(f"   ⚠️  analyzer failed ({e}) — experiment.json "
						  f"stands")
					self._mark("analyze", status="failed", error=str(e)[:200])

			# Step 4d — hand off to a downstream model, when the backend
			# declares one and the plan asks for it. Non-fatal: this study
			# stands on its own.
			if self.COUPLES_TO:
				try:
					self._couple(experiment_plan, config)
				except Exception as e:                          # noqa: BLE001
					print(f"   ⚠️  {self.COUPLES_TO} coupling failed ({e}) — "
						  f"the {self.MODEL} run stands")

			# STEP 5 IS GONE (2026-08-13). It wrote LLM_ANALYSIS_INPUT.json, an
			# alias holding a prompt payload for the report agent that this
			# framework no longer has. experiment.json is the product, and
			# 04_analysis/analysis.json is the Analyzer's boundary file.

			# NON-FATAL, like every stage since _package — and this one was
			# not, which is the one place the rule was written down and not
			# followed. A shape mismatch here raised AFTER the compute, after
			# experiment.json, and after the whole Analyzer had run: everything
			# of value was already on disk and the run still ended in a
			# traceback, with the caller getting an exception instead of the
			# summary of a study that had in fact succeeded.
			end_time = datetime.now()
			try:
				run_summary = self._create_run_summary(
					experiment_plan, experiments, results, start_time, end_time)
				self._save_run_summary(run_summary)
			except Exception as e:                              # noqa: BLE001
				print(f"   ⚠️  RUN_SUMMARY.json failed ({e}) — the ensemble and "
					  f"experiment.json stand")
				run_summary = {
					'run_directory':         str(self.run_dir),
					'model_type':            self.MODEL,
					'experiments_total':     len(experiments),
					'experiments_success':   len(self._outcome_map(results)),
					'total_runtime_seconds': (end_time - start_time).total_seconds(),
					'summary_error':         str(e),
				}

			n_ok  = run_summary['experiments_success']
			n_tot = run_summary['experiments_total']
			rt    = run_summary['total_runtime_seconds']
			print(f"\n{'=' * 60}")
			print(f"{model} COMPLETE: {n_ok}/{n_tot} | {rt:.1f}s")
			print(f"Output: {self.run_dir}")
			print(f"{'=' * 60}\n")
			return run_summary

		except Exception as e:
			self._save_error(e, start_time)
			raise

	# Stages a backend must supply. Raising by default rather than no-op'ing:
	# a manager missing its compute stage should fail loudly at the first call,
	# not report an empty successful run.
	def _build_case_inputs(self, plan, config):
		raise NotImplementedError(f"{type(self).__name__} must implement _build_case_inputs")

	def _build_cases(self, experiments, config=None):
		"""config is passed so a backend can see mcp_clients here, the same as
		_build_case_inputs and _run do. Defaulted so a caller that predates it still works."""
		raise NotImplementedError(
			f"{type(self).__name__} declares NEEDS_CASE_BUILD but has no _build_cases")

	def _run(self, experiments, config):
		raise NotImplementedError(f"{type(self).__name__} must implement _run")

	def _poll(self, record: Dict[str, Any], experiments, config):
		"""Has the job in `record` finished?

		Return what the stage would have returned had it waited, or None if the
		job is still queued or running. `record` is the run state's entry for that
		stage, so it carries job_id, whatever the Pending marker's detail held,
		and — since build_cases became job-shaped too — **which stage it is**, under
		the key `stage`. A backend answers differently for a CIME build than for
		an ensemble, and it cannot tell them apart from a job id.

		Only a backend whose stages can return Pending needs this — hence
		raising rather than returning None, which would read as "still running,
		forever".
		"""
		raise NotImplementedError(
			f"{type(self).__name__}.{record.get('stage', 'a stage')} returned "
			f"Pending (job {record.get('job_id')}) but the class has no _poll "
			f"to ask whether that job has finished")

	# Returned by _advance when the run cannot continue in this session because
	# the stage's work is in a queue. A sentinel rather than None: None is a
	# legitimate return from _build_cases, and conflating the two would end every
	# ELM run at the build_cases stage.
	STOP = object()

	# ─────────────────────────────────────────────────────────
	# TALKING TO A MODEL SERVER
	# ─────────────────────────────────────────────────────────
	# Plumbing, not model knowledge: which client, how to call it, where to
	# mail the result. It lived on ELMExpManager because that is where the
	# first MCP call was written, and PFLOTRAN then grew its own way of doing
	# the same job — two backends, two conventions, and a third would have
	# invented a third. None of it knows what a column is.
	MCP_NAME: Optional[str] = None      # the server this backend drives
	# _pinning_rules AND CAPABILITIES_TOOL DELETED 2026-08-18.
	#
	# The sampler used to ask the model server what a column may be pinned to
	# — a second fetch of the block workflow.py had already fetched for the
	# planner. Two fetches of one rule can disagree (a server updated between
	# plan and run, a resume a week later, a manager naming a different server
	# than the brief — all of which happened), and CAPABILITIES_TOOL existed
	# only to make that second fetch. workflow.py now writes the block the
	# planner was SHOWN into strategy.json beside the design, and
	# expand_sampling._pinned_from_plan reads it from there. Sampling is a
	# function of reception.json and strategy.json, and touches no server.

	def _mcp(self, config: Dict[str, Any]):
		"""The elm MCP client, or None to run locally."""
		if not (config or {}).get("run_via_mcp", True):
			return None
		return ((config or {}).get("mcp_clients") or {}).get(self.MCP_NAME)

	@staticmethod
	def _mcp_call(client, tool: str, args: Dict[str, Any],
				  budget: Optional[float] = None) -> Dict[str, Any]:
		"""One MCP call, with the client's timeout raised for its duration.

		The per-server timeout is sized for the quick tools. build_elm_cases
		generates per-column surfaces and collect_elm_results reads NetCDF, and
		neither is quick for 19 columns — a client ceiling that was 15x too
		small is exactly how the reaction MCP failed before commit 19fffc3.
		"""
		prev = getattr(client, "timeout", None)
		try:
			if budget and prev is not None and prev < budget:
				client.timeout = budget
			out = client.call_tool_json(tool, args)
		finally:
			if prev is not None:
				client.timeout = prev
		if out is None:
			# DOES NOT NAME A SERVER (2026-08-17). It said "the elm MCP"
			# whatever it had called — written when there was one — so a
			# PFLOTRAN tool that returned nothing reported an ELM timeout and
			# sent a reader to the wrong logs. This is a @staticmethod and has
			# no backend to ask, so it names the TOOL, which is unambiguous.
			raise RuntimeError(
				f"the MCP server did not answer {tool} within its timeout, "
				f"or returned no JSON")
		# `error` means the CALL could not be made. A payload carrying `ok` is
		# reporting an OUTCOME — a build that failed, an ensemble with no
		# results — and the caller has more to say about that than this does,
		# including the job's log. Raising here would make those messages dead
		# code and replace them with a one-liner.
		if isinstance(out, dict) and out.get("error") and "ok" not in out:
			raise RuntimeError(f"elm MCP {tool}: {out['error']}")
		return out


	@staticmethod
	def _notify_email(config: Dict[str, Any]) -> str:
		"""Where Slurm should send the END,FAIL mail. Empty = do not ask for one.

		Never guessed from the username: a wrong address means the one signal
		the unattended flow depends on goes silently nowhere.
		"""
		from core.notify_prefs import remembered_email
		return str((config or {}).get("notify_email")
				   or os.environ.get("IDEAS_NOTIFY_EMAIL", "")
				   or remembered_email() or "").strip()


	def _advance(self, stage: str, fn, *, state: Dict[str, Any], resume: bool,
				 experiments, config, **done_fields):
		"""Run one stage that may hand back a job id instead of finishing.

		Three paths, and the run state is what tells them apart:

		  * an earlier session submitted this stage and left → poll it. Still
		    running, and the run stops again (STOP); finished, and its result
		    is returned as though the stage had just run.
		  * the stage runs and returns a Pending → record the id and STOP.
		  * the stage runs and returns a result → record it and carry on.

		Written once and used by both build_cases and run. The two differ in what
		they return and in nothing else that matters here, which is the whole
		reason this is a method rather than the same fifteen lines twice.
		"""
		record = (state["stages"].get(stage) or {}) if resume else {}

		if record.get("status") == "pending":
			print(f"↻ {stage}: job {record.get('job_id')} was submitted "
				  f"{record.get('at', '?')} — checking whether it landed")
			out = self._poll({**record, "stage": stage}, experiments, config)
			if out is None:
				return self.STOP
			self._mark(stage, n_results=self._n_results(out),
					   job_id=record.get("job_id"), **done_fields)
			return out

		out = fn()
		if isinstance(out, Pending):
			# Submitted, not finished. Everything downstream needs results that
			# do not exist yet, so the run stops here and says so — rather than
			# carrying on and reporting 0/19 as though the science had failed.
			#
			# Warn, do not raise: the job is ALREADY in the queue by the time we
			# get here, so refusing to continue would throw away the one thing
			# that makes it recoverable — its id.
			if type(self)._poll is ExperimentManagerBase._poll:
				print(f"   ⚠️  {type(self).__name__} submits but has no _poll — "
					  f"job {out.job_id} is recorded in {self.STATE_FILE} and "
					  f"will have to be collected by hand")
			self._mark(stage, status="pending", job_id=out.job_id, **out.detail)
			return self.STOP

		self._mark(stage, n_results=self._n_results(out), **done_fields)
		return out

	def adopt_completed_run(self) -> Dict[str, Any]:
		"""Record the job-shaped stages as done from what is ON DISK.

		For the in-job finalize: one batch job builds the cases, runs the
		ensemble, and then runs the pipeline tail itself. At that last step the
		normal resume path cannot be used as-is, because it would poll the
		SLURM job that produced the output — and that job is US. `squeue` says
		RUNNING, `_poll` returns None, and the run stops one line before the
		analysis it was submitted to produce.

		So the evidence is taken from the filesystem instead of the scheduler:
		the output either exists or it does not, which is the same question
		_poll was asking and the only one that actually matters here.

		Returns {stage: n} for what it adopted. The base adopts `run` only;
		a backend with its own artifacts (ELM's built case dirs) extends it.
		"""
		adopted: Dict[str, Any] = {}
		experiments = self._rehydrate_case_inputs() or []
		if not hasattr(self, "_outcomes_from_disk"):
			return adopted
		outcomes = self._outcomes_from_disk(experiments)
		self._mark("run", n_results=len(outcomes),
				   n_ok=sum(1 for v in outcomes.values() if v),
				   adopted_from="disk")
		adopted["run"] = len(outcomes)
		return adopted

	# States in which SLURM still owns the job. Anything else — COMPLETED,
	# FAILED, TIMEOUT, CANCELLED, NODE_FAIL — means the scheduler is finished
	# with it, whatever it did, and the backend should go look at the output.
	ACTIVE_JOB_STATES = {
		"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED",
		"RESIZING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "SIGNALING",
		"STAGE_OUT", "RESV_DEL_HOLD", "STOPPED",
	}

	@staticmethod
	def _slurm_state(job_id: Any) -> Optional[str]:
		"""SLURM's word for what job_id is doing, or None if it will not say.

		squeue first — it is cheap and it is the only one that sees a job that
		has not started. Then sacct, which is the only one that remembers a job
		that has already left the queue.

		None means NO ANSWER, not "finished". A squeue that times out or a
		cluster without sacct must not be read as a completed ensemble; the
		caller decides what other evidence it trusts.
		"""
		import shutil, subprocess
		jid = str(job_id).split("_")[0].split(".")[0]
		if not jid.isdigit():
			return None

		def _run(cmd) -> Optional[str]:
			if not shutil.which(cmd[0]):
				return None
			try:
				out = subprocess.run(cmd, capture_output=True, text=True,
									 timeout=60)
			except Exception:                                   # noqa: BLE001
				return None
			line = (out.stdout or "").strip().splitlines()
			return line[0].strip() if line and line[0].strip() else None

		st = _run(["squeue", "-h", "-j", jid, "-o", "%T"])
		if st:
			return st.upper()
		# -X so a job's steps do not shadow the job itself; the step lines come
		# back first and a step can read COMPLETED while the job is still going.
		st = _run(["sacct", "-n", "-X", "-j", jid, "-o", "State"])
		if st:
			# sacct spells cancellation "CANCELLED by 12345"
			return st.split()[0].upper()
		return None

	def _extract(self, experiments, plan=None, config=None):
		raise NotImplementedError(f"{type(self).__name__} must implement _extract")

	def _couple(self, plan, config):
		raise NotImplementedError(
			f"{type(self).__name__} declares COUPLES_TO={self.COUPLES_TO} "
			f"but has no _couple")


	def _refine_columns(self, columns, config: Dict[str, Any]) -> Dict[str, Any]:
		"""Backend edits to the sampled columns, before they are persisted.

		ELM warm-starts here, which snaps coordinates and adopts the donor
		cell's soil. A backend with nothing to add returns {}.
		"""
		return {}

	def _build_sweep_columns(self, design: Dict[str, Any],
							 config: Dict[str, Any]) -> Dict[str, Any]:
		"""Columns for a CONTROLLED SWEEP, built by the backend that runs them.

		The sibling of _refine_columns, and here for the same reason: what a
		clay level BECOMES is model knowledge. The base knows a sweep is one
		column per level combination; only the server knows that a level of 27
		means a synthetic surface dataset with those percentages in it, and only
		the server can say whether 27 is buildable at all.

		Returns the same payload expand_sampling.expand() returns — `columns`
		plus whatever provenance the backend wants to persist beside them —
		because everything after this point treats the two identically.

		A backend with no sweep support raises, which is the correct answer:
		it means this model cannot run the study that was designed, and
		producing a partial one instead would be worse.
		"""
		raise NotImplementedError(
			f"{type(self).__name__} cannot build a controlled sweep. The "
			f"strategy asked for a factor sweep, which needs this backend to "
			f"turn factor levels into columns; only the model server knows "
			f"what a level means.")

	def _draw_design(self, res: Dict[str, Any], config: Dict[str, Any]) -> None:
		"""Draw the sampling-design figure, if this backend has one.

		A NO-OP HERE ON PURPOSE. The figure is model knowledge: ELM's shows the
		initial soil water read out of each column's finidat and the donor
		gridcell's soil profile, neither of which the base can know about. It
		used to be drawn here from tools/expand_sampling.plot_columns, which is
		why the base was importing a plotter and why the figure described the
		columns as SAMPLED rather than as RUN.

		Called after columns.json is persisted, so a backend can read it.
		"""
		return None

	def _to_run_plan(self, plan, columns, config, refine) -> Dict[str, Any]:
		raise NotImplementedError(
			f"{type(self).__name__} must turn columns into an executable plan")

	# ─────────────────────────────────────────────────────────
	# STEP 0b — MATERIALIZE SAMPLING (strategy → concrete columns)
	# ─────────────────────────────────────────────────────────
	def _materialize(self,
					 plan:   Dict[str, Any],
					 config: Dict[str, Any]) -> Dict[str, Any]:
		"""Turn a planner strategy into concrete columns, then into a run plan.

		No-op when the plan already carries an executable payload (the legacy
		single-site shape), so old plans keep working unchanged.

		Needs, via config:
			brief        — reception brief (domain bbox + optional HUC)
			reception    — reception.json: the grid, the polygon, the stations
			strategy     — strategy.json: the design and its pinning block
			mcp_clients  — live MCP clients; terrain is used only for the point
			               elevation at pinned stations, and is optional
		Optional: yr_start, yr_end, soil_config, substrate, n_columns, n_bands.
		"""
		if self._already_executable(plan):
			return plan

		exp     = _load_tool("expand_sampling")
		brief   = config.get("brief") or {}
		clients = config.get("mcp_clients") or {}

		# WHICH KIND OF STUDY, read before anything is required of the inputs.
		#
		# The two demands below — a bounding box and the terrain client — are
		# needs of SPATIAL SAMPLING, not of materialising columns. A controlled
		# sweep has no basin to clip and no elevation grid to sample, so both
		# would refuse a design that is complete and correct. Everything from
		# here to the shared tail is the site path, untouched.
		strategy_in = config.get("strategy") or plan
		is_sweep = _is_factor_sweep(strategy_in, brief)

		bbox = None if is_sweep else exp._bbox_from_brief(brief)
		if not is_sweep:
			if not bbox:
				raise ValueError(
					"Cannot materialize sampling: no domain bbox in the reception "
					"brief, and the plan has no executable payload. Pass "
					"config['brief'] with domain.bbox, or supply an explicit plan.")
			# THE TERRAIN CLIENT IS NO LONGER REQUIRED HERE (2026-08-18). The
			# grid and the basin polygon come from reception.json — fetched and
			# clipped once, by the only component that reaches outside — and
			# the sampler reads them. Terrain is used for one thing now: the
			# point elevation AT each pinned station, which has a fallback.
			if not clients.get("terrain"):
				print("   ⚠️  no 'terrain' client — pinned columns will take the "
					  "station's reported elevation or the nearest grid point, "
					  "and say which")

		# CHECK FIRST, then read the counts. The order is the whole point.
		#
		# check() corrects strategy["sampling"]["n_columns"] when the request
		# asked for a different number than the planner designed, and returns a
		# config carrying the corrected strategy. Reading n_total before this
		# call took the number from the UNCORRECTED plan, so the correction was
		# computed, printed — "using the 2 that was asked for" — recorded in the
		# run, and then ignored. Live from 5e0cfd5 until 2026-08-07, when a
		# request for 2 columns built 17 and warm-started every one of them.
		#
		# 5e0cfd5 fixed the DETECTION. This is the enforcement.
		config = self.check(plan, config)

		# ── the sweep path, and it rejoins below at _refine_columns ─────────
		#
		# DELIBERATELY AFTER check(). The gate is what corrects the column count
		# to the factorial the levels imply, refuses a factor with one level,
		# and stops a design that names no coordinates. Building first and
		# checking after would spend the build on a design the gate would have
		# refused — which is the ordering mistake the comment above records for
		# the site path, made once already.
		if is_sweep:
			design = ((config.get("strategy") or plan).get("sampling") or {})
			print("\n🗺️  STEP 0: Materializing a Controlled Sweep")
			print("-" * 40)
			# A SWEEP STARTS COLD, and this is the line that makes it true.
			# warm_start defaults to True for the site path, and the sweep path
			# inherited that default silently: every conceptual run so far began
			# from the CONUS restart, so a study designed to depend on no real
			# place was handed a real gridcell's water content, and the only
			# trace was a caveat nobody had chosen. A controlled sweep either
			# holds the initial state constant or it is not controlled.
			#
			# THE COST IS ACCEPTED, NOT HIDDEN: a cold column takes years to
			# forget ELM's defaults, so a short run reports the initialisation
			# as much as the soil. That is recorded as a caveat on the finding
			# rather than fixed with a spin-up.
			config = dict(config)
			config["warm_start"] = False
			names = ", ".join(str(f.get("name"))
							  for f in (design.get("factors") or []))
			print(f"   factors: {names or '(none)'}  "
				  f"N={design.get('n_columns')}  no basin, no sampling")
			res = self._build_sweep_columns(design, config) or {}
			columns = res.get("columns") or []
			if not columns:
				raise RuntimeError(
					"The sweep design produced no columns. The backend "
					"accepted the design and returned nothing, which is a "
					"failure rather than an empty result.")
			return self._persist_columns(res, columns, plan, config)

		# From the corrected strategy first, since that is what check() rewrote;
		# `plan` remains the fallback for a caller that passes no strategy.
		n_total = (config.get("n_columns")
				   or exp._n_from_plan(config.get("strategy") or plan))
		if not n_total:
			raise ValueError(
				"Cannot materialize sampling: no column count in the plan "
				"and none in config.")
		# The planner's band count, not reception's list length. See
		# expand_sampling._n_bands_from_plan for what that fallback got wrong.
		bands = (config.get("n_bands")
				 or exp._n_bands_from_plan(config.get("strategy") or plan)
				 or DEFAULT_BANDS)
		strategy = config.get("strategy") or plan
		per_band = config.get("per_band") or exp._per_band_from_plan(strategy)

		# The stations the strategy wants a column AT. Resolved against what
		# reception actually fetched, filtered on the pinning block the strategy
		# carries, and raising when plan and observations disagree — see
		# expand_sampling._pinned_from_plan for why a miss is not survivable.
		reception = config.get("reception") or {}
		pinned = (exp._pinned_from_plan(strategy, reception)
				  if reception else [])
		if not reception and (strategy.get("validation") or []):
			print("   ⚠️  the strategy names validation stations but no reception "
				  "was passed — NO column will be pinned, and no variable can be "
				  "compared at a station. Pass config['reception'].")

		print("\n🗺️  STEP 0: Materializing Sampling")
		print("-" * 40)
		print(f"   bbox={bbox} N={n_total} bands={bands} per_band={per_band} "
			  f"pinned={len(pinned)}")

		# THE GRID AND THE POLYGON ARE RECEPTION'S, READ AS WRITTEN. gather_grid
		# fetched the DEM at the sampler's own resolution, retried at higher
		# density when too few points landed in the basin, dropped every point
		# outside the WBD polygon, and wrote both to reception.json — its own
		# comment says "kept so a re-plot needs no refetch". This stage used to
		# fetch the polygon AGAIN from terrain and clip AGAIN with a different
		# predicate (largest ring, against reception's all-rings), fifty-eight
		# points in and fifty-eight out on every run. data_gather already
		# records why the fetch moved there: "the Experiment Manager used to
		# make this call itself, at which point the grid it clipped and the
		# grid reception described were two different things."
		rec_grid = reception.get("grid") or {}
		if not rec_grid.get("points"):
			raise RuntimeError(
				"Cannot materialize sampling: reception.json carries no grid — "
				"gather_grid did not run, or returned nothing. The sampler reads "
				"reception's grid and fetches none of its own; fix reception "
				"rather than sampling on a grid it never described.")
		res = exp.expand(rec_grid, n_total, bands, per_band=per_band,
						 pinned=pinned, terrain=clients.get("terrain"))
		if res.get("error"):
			raise RuntimeError(f"Sampling expansion failed: {res['error']}")
		columns = res.get("columns", [])
		if not columns:
			raise RuntimeError("Sampling expansion produced no columns.")

		# WHETHER THIS IS A WATERSHED SAMPLE OR A BBOX FALLBACK is reception's
		# fact too — it is the one that had or lacked the polygon. Without one
		# the columns are the raw bbox and can lie OUTSIDE the basin: the run
		# completes, but the ensemble no longer represents the watershed that
		# was asked about. A scientific difference, said loudly.
		huc = (brief.get("domain") or {}).get("huc")
		clipped = bool(rec_grid.get("clipped_to_watershed")) and bool(res.get("boundary"))
		if not clipped:
			why = ("no HUC in the reception brief" if not huc
				   else f"reception fetched no boundary for HUC {huc}")
			name = (brief.get("domain") or {}).get("name") or "the watershed"
			print(f"   ⚠️  WARNING: the grid is the RAW BBOX, not the watershed — {why}.")
			print(f"   ⚠️  Columns may fall outside the basin; the ensemble is "
				  f"a bounding-box sample, not '{name}'.")
		# bbox FIRST, as expand() used to write it, so columns.json keeps its
		# historical key order and an archived fixture still hashes the same.
		res = {"bbox": bbox, **res}
		res["sampling_domain"] = {
			"clipped_to_watershed": clipped,
			"huc": huc,
			"name": (brief.get("domain") or {}).get("name"),
			"bbox": bbox,
			"caveat": None if clipped else (
				"Columns were sampled from the bounding box, NOT clipped to the "
				"watershed boundary (reception had no polygon). Some columns may "
				"lie outside the basin; treat the ensemble as a bbox sample."),
		}

		return self._persist_columns(res, columns, plan, config)

	def _persist_columns(self, res: Dict[str, Any], columns,
						 plan: Dict[str, Any],
						 config: Dict[str, Any]) -> Dict[str, Any]:
		"""Refine, write, draw, and turn columns into an executable plan.

		EXTRACTED SO THE TWO ARCHETYPES SHARE ONE TAIL. Sampled columns and
		swept columns differ in how they were CHOSEN and in nothing after that:
		both need the backend's refinement, both are persisted to the same two
		files, both get the design figure, and both become a run plan the same
		way. A second copy of this for sweeps would be the place the two paths
		silently drifted apart.

		Everything below this line is unchanged from when it lived inline.
		"""
		brief = config.get("brief") or {}
		yr_start = int(config.get("yr_start", 1995))
		yr_end   = int(config.get("yr_end",   yr_start))

		# Backend refinement BEFORE persisting — see the module docstring. The
		# ELM warm start snaps every column to its donor gridcell.
		refine = self._refine_columns(columns, config) or {}
		res["columns"] = columns

		(self.input_dir / "columns.json").write_text(json.dumps(res, indent=2))
		(self.run_dir  / "columns.json").write_text(json.dumps(res, indent=2))

		# The design figure is the BACKEND's — see _draw_design. It runs here,
		# after columns.json is persisted, so the backend can read what was
		# written rather than be handed it.
		try:
			self._draw_design(res, config)
		except Exception as e:                                  # noqa: BLE001
			# NON-FATAL, and it names the exception TYPE. A figure is not worth
			# a finished materialize, but "failed (foo)" reads as a data
			# problem whatever went wrong — and a swallowed NameError cost a
			# whole payload earlier today.
			print(f"   ⚠️  sampling_design.png failed "
				  f"({type(e).__name__}: {e}) — non-fatal")

		executable = self._to_run_plan(plan, columns, config, refine)
		merged     = {**plan, **executable}

		# The honesty payload reads this back at extraction; without it an
		# integrated run shipped an empty run state.
		(self.run_dir / "assumptions.json").write_text(
			json.dumps(executable.get("assumptions_ledger", []), indent=2))
		(self.run_dir / "run_plan.json").write_text(json.dumps(merged, indent=2))
		# Make the run dir self-describing, with the SAME filenames the
		# standalone tools expect. validate_run.build_validation() needs
		# reception_brief.json (domain bbox) and interpret_run.interpret()
		# needs plan.json (goals + feasibility verdict). Writing them here
		# means an integrated run is consumable by every existing tool.
		if not (self.run_dir / "reception_brief.json").exists():
			(self.run_dir / "reception_brief.json").write_text(
				json.dumps(brief, indent=2))

		# reception.json is normally the coordinator's to write, and when it
		# already exists that copy wins. But a manager driven directly — by a
		# tool, a test, or a single-model run — has no coordinator, and
		# step0_context takes the USER'S QUESTION from this file and nowhere
		# else. Without it the Analyzer ran the entire box against
		# `question: None`: it still chose figures and still reported success,
		# so nothing failed and nothing recorded that the one input step 2
		# exists to serve had gone missing.
		recep = config.get("reception")
		if recep and not (self.run_dir / "reception.json").exists():
			(self.run_dir / "reception.json").write_text(
				json.dumps(recep, indent=2))
		(self.run_dir / "plan.json").write_text(json.dumps(plan, indent=2))
		print(f"✓ {len(columns)} column(s) materialized ({yr_start}-{yr_end})")
		return merged

	def _already_executable(self, plan: Dict[str, Any]) -> bool:
		"""Does this plan already carry a backend payload? Backends override."""
		return False


	# ─────────────────────────────────────────────────────────
	# PACKAGE — the one file the Analyzer reads
	# ─────────────────────────────────────────────────────────
	#
	# FIELD_SEMANTICS is the point of this stage. `precip_mm_yr` once held
	# RAIN alone, so every fraction computed against it was inflated and a
	# run was reported as draining more water than fell on it — a semantic
	# bug, not a structural one, and no schema check could have caught it.
	# Naming the source variables next to the number is what makes the next
	# one visible.
	#
	# EMPTY HERE, AND DECLARED PER BACKEND. This dict used to hold ELM's
	# metrics, which meant PFLOTRAN inherited them: its experiment.json
	# described QOVER-derived runoff fractions and a RAIN+SNOW precipitation
	# total for a run that computes none of those. That is worse than saying
	# nothing, because step 2 hands field_semantics to an LLM as the
	# authority on what the run's numbers mean — a saturation study came
	# annotated with a surface water budget it never had.
	#
	# The base cannot supply a default, because the failure mode of a wrong
	# default is exactly the one this dict exists to prevent.
	FIELD_SEMANTICS: Dict[str, Any] = {}

	def _variable_units(self, results: Any) -> Dict[str, str]:
		"""Units for the raw model variables, wherever the backend keeps them.

		ELMResultsAnalyzer puts them in summary['units']; a bare namespace may
		expose .units. Reading only the second gave an EMPTY units block on
		the 2019 Gunnison run — a results package that states no units is
		exactly the failure FIELD_SEMANTICS exists to prevent.
		"""
		# The contract states them outright; the fallbacks below exist for the
		# analyzer-shaped objects that predate it.
		if isinstance(results, dict) and isinstance(results.get("units"), dict) \
				and results["units"]:
			return dict(results["units"])
		# hydro_summary.json was the third source here and is no longer written
		# (2026-08-13). It was never reached in practice: the contract above
		# always carries units, and both object shapes below predate it.
		for src in (getattr(results, "units", None),
					(getattr(results, "summary", None) or {}).get("units")):
			if isinstance(src, dict) and src:
				return src
		return {}

	# Fields the extraction cannot know, because they describe how the column
	# was CHOSEN AND BUILT rather than what the model did with it. columns.json
	# is the one seam: the sampler writes where a column is, and each model's
	# build call writes what it learned about the column onto the same list —
	# ELM its donor soil and forcing cell after the warm start, PFLOTRAN the
	# water table, the forcing and the deck's own warnings after the join.
	# This list is what carries that seam onto the packaged rows.
	#
	# TWO PARTS, ONE ASSERTION (2026-08-18). The framework's own keys are
	# here — what the SAMPLER writes, the same for every model. What a BUILD
	# CALL adds is that model's knowledge, so each manager declares it beside
	# its server as COLUMN_METADATA_EXTRA; _column_keys() composes the two
	# once and the merge asserts against the union. Before this the base list
	# named ELM's soil keys itself, and PFLOTRAN's build facts travelled by a
	# second, unchecked list in its extractor — where `warning` ("this column
	# is essentially saturated") vanished on the way to experiment.json.
	#
	# Each key is here because something downstream reads it:
	#
	#   band, band_range_m   area-weighting the ensemble. The sampler allocates
	#                        >=1 column per elevation band regardless of band
	#                        size, so an unweighted mean over-weights small
	#                        bands. Without this the weighting silently
	#                        degrades to a plain average.
	#   lat, lon, elevation_m  normally come from the extraction, which reads
	#                        them off the coupler. Listed here too because
	#                        columns.json is the AUTHORITY on where a column
	#                        is, and an extraction invoked without them
	#                        produced 19 rows with elevation_m absent — which
	#                        silently flattens every elevation figure and
	#                        every gradient claim to a single point.
	#   pinned, station_id, station_variable
	#                        which station the sampler put this column on. Off
	#                        the list, the join dropped them, every row reached
	#                        the Analyzer with pinned=None, and the comparison
	#                        re-derived the pairing geometrically — guessing at
	#                        an answer the sampler had already recorded.
	#   forcing_start, forcing_end
	#                        the years THIS column ran, written by the build
	#                        call of either model. A forcing_year sweep varies
	#                        them between columns, and a PFLOTRAN column with
	#                        no rain series has none, so a run-level period
	#                        would describe neither correctly.
	#   treatment, weather   written by a CONTROLLED SWEEP, absent on a site
	#                        run: WHY THIS COLUMN DIFFERS, which on a sweep is
	#                        the only thing that makes it a column rather than
	#                        a repeat, and which weather it was given. The
	#                        sweep is the framework's archetype — the base
	#                        derives treatment_label from `treatment` below —
	#                        so the keys are the framework's, whichever server
	#                        builds the sweep.
	#
	# A NAME ON THIS LIST THAT NOTHING PRODUCES IS NOT FREE: it asks for a
	# capability the pipeline may no longer have, and a key that is absent
	# looks exactly like a key that is null this time. That is what report()
	# below exists to catch.
	COLUMN_METADATA = KeySet(
		"COLUMN_METADATA",
		keep = ("lat", "lon", "elevation_m",
				"band", "band_range_m",
				"pinned", "station_id", "station_variable",
				"forcing_start", "forcing_end",
				"treatment", "weather"),
		# A run that pinned nothing has no column carrying these, and that is
		# a fact about the design rather than a gap in the list. The years are
		# optional for the same reason: a column with no series has none. And
		# a site run varies nothing deliberately, so it has no treatment and
		# no written weather — neither absence is a hole.
		optional = ("station_id", "station_variable",
					"forcing_start", "forcing_end",
					"treatment", "weather"),
		drop = {
			"id": "the join key — _merge_column_metadata matches on it, so "
				  "carrying it onto the row would restate the row's own name",
			"station_name": "the pinned station's label. station_id is the "
							"identifier a reader can look up; a name cannot "
							"be resolved back to a record",
			"station_elevation_m": "consumed at sampling time — "
								   "expand_sampling falls back to it when the "
								   "terrain server has no elevation for the "
								   "station, and records the result as "
								   "elevation_m",
			"in_basin": "reception's inside-the-divide tag, copied here with "
						"the rest of the station record. The comparison "
						"applies it from the observations, which is where it "
						"was measured",
			"elevation_source": "WRITTEN BY TWO PRODUCERS AND READ BY NONE "
								"(inputs.py:260, expand_sampling.py:486). "
								"Dropped rather than carried because a field "
								"nobody reads is not provenance, it is weight",
			"fan_wtd_m": "THE ASSERTION'S FIRST CATCH, and it caught a fossil. "
						 "Fan's water table left the sampler on 2026-08-07 and "
						 "was renamed wtd_prior_m on 2026-08-12; the only file "
						 "on disk still carrying it is one archived Gunnison "
						 "PFLOTRAN study. Dropped, not kept, because reviving "
						 "the old name would give the row two spellings of one "
						 "number — which is how the name went stale the first "
						 "time",
		},
		source = "columns.json -> columns[*]",
		where  = "src/core/exp_manager_base.py :: COLUMN_METADATA",
	)

	# WHAT THIS MODEL'S BUILD CALL ADDS TO A COLUMN — declared by the manager
	# beside its server, None for a backend whose build adds nothing. Composed
	# with COLUMN_METADATA above by _column_keys(); a key on columns.json that
	# neither part names raises at package time, whichever producer wrote it.
	COLUMN_METADATA_EXTRA: Optional[KeySet] = None

	def _column_keys(self) -> KeySet:
		"""COLUMN_METADATA plus this backend's additions, composed once."""
		ks = getattr(self, "_column_keys_cache", None)
		if ks is not None:
			return ks
		base, extra = self.COLUMN_METADATA, self.COLUMN_METADATA_EXTRA
		if extra is None:
			ks = base
		else:
			both = set(base.keep) & set(extra.keep)
			if both:
				raise ValueError(
					f"{extra.name}: {sorted(both)} already kept by "
					f"COLUMN_METADATA — declare a key once")
			ks = KeySet(
				f"COLUMN_METADATA[{self.MODEL}]",
				keep = base.keep + extra.keep,
				optional = tuple(base.optional | extra.optional),
				drop = {**base.drop, **extra.drop},
				source = base.source,
				where = f"{base.where} + {extra.where or extra.name}")
		self._column_keys_cache = ks
		return ks

	@staticmethod
	def _treatment_label(treatment: Any) -> Optional[str]:
		"""A sweep column's treatment in a few words, fit for an axis tick.

		WHY THE RECORD CARRIES THIS RATHER THAN THE FIGURE DERIVING IT. The
		treatment is a nested dict, and a generated plotting script handed one
		with no better option does the only thing it can: prints it. On the
		2026-08-16 sweep that put

		    {'soil_texture': 5, 'prescribed_weather': {'fill': 'scale',
		     'values': {'PRECTmms': 2.0}}}

		on four x-axis ticks, and the bars ended up in the top corner of their
		own canvas with the labels running off the page. The second round wrote
		its own short names and read fine, so the fix is not to teach the model
		to shorten — it is to have the name in the record before anyone plots.

		GENERIC OVER THE DICT, not a list of known factors. It reads whatever
		keys the design put there, so a sweep over a factor nobody has invented
		yet still gets a label rather than a fallback.
		"""
		if not isinstance(treatment, dict) or not treatment:
			return None

		def num(x):
			try:
				f = float(x)
			except (TypeError, ValueError):
				return str(x)
			return str(int(f)) if f == int(f) else f"{f:g}"

		def one(key, val):
			# `prescribed_` says how the value got there, which the reader of a
			# tick label does not need; the factor's name is the rest.
			name = str(key).replace("prescribed_", "").replace("_", " ")
			if isinstance(val, dict):
				fill = str(val.get("fill") or "").strip().lower()
				vals = val.get("values") or {}
				if fill == "scale" and vals:
					return ", ".join(f"{k} x{num(v)}" for k, v in vals.items())
				if fill == "offset" and vals:
					return ", ".join(
						f"{k} {'+' if float(v) >= 0 else ''}{num(v)}"
						for k, v in vals.items())
				if fill in ("set", "constant", "uniform") and vals:
					return ", ".join(f"{k} = {num(v)}" for k, v in vals.items())
				if fill == "copy":
					return f"{name} as-is"
				return f"{name} {fill}".strip()
			if isinstance(val, str):
				return (f"{name} as-is" if val.strip().lower() == "copy"
						else f"{name} {val}")
			return f"{name} {num(val)}"

		parts = [p for p in (one(k, v) for k, v in treatment.items()) if p]
		return ", ".join(parts) or None

	def _merge_column_metadata(self, rows: List[Dict[str, Any]]) -> None:
		"""Join the sampling metadata onto the extracted rows, in place.

		The extraction reads history files and knows nothing about bands or
		priors; columns.json holds those and nothing else does. Keeping them
		in two files is what forced the Analyzer to open both, and the row's
		own `soil` field comes back null without this.

		Joined on the column id. A row with no matching column keeps whatever
		it has — a mismatch here means the run directory is inconsistent, and
		dropping data is not the way to report that.
		"""
		f = _resolve_columns(self.run_dir)
		if not f:
			return
		try:
			cols = (json.loads(f.read_text()) or {}).get("columns") or []
		except Exception as e:
			print(f"   ⚠️  could not read column metadata ({e})")
			return
		by_id = {c.get("id"): c for c in cols if c.get("id")}
		if not by_id:
			return

		missed = 0
		for r in rows:
			src = by_id.get(r.get("case_name") or r.get("scenario_name"))
			if not src:
				missed += 1
				continue
			# ASSERT, THEN COPY. check() raises on a key columns.json carries
			# that this list neither keeps nor drops — the sampler adding a
			# field is a decision someone has to make here, not a value that
			# disappears on the way through. It also records which declared
			# keys actually turned up, which is what report() reads.
			keys = self._column_keys()
			keys.check(src)
			for k in keys.keep:
				if src.get(k) is not None and r.get(k) is None:
					r[k] = src[k]
			# DERIVED, not copied, so it is not a COLUMN_METADATA key: nothing
			# in columns.json carries it. A site run varies nothing and gets no
			# treatment, so it gets no label either — an absent label means
			# "this column is not a treatment", which is the truth.
			label = self._treatment_label(r.get("treatment"))
			if label:
				r["treatment_label"] = label
		if missed:
			print(f"   ⚠️  {missed} extracted column(s) had no sampling "
				  f"metadata — experiment.json cannot be area-weighted")

	def _ensemble_blocks(self, results: Any) -> Dict[str, Any]:
		"""Whole-ensemble products, plus the honesty payload.

		Read from the extraction the backend just wrote rather than from the
		results object, because that file is the authoritative record of what
		was extracted and it exists by the time this stage runs. Missing
		blocks are omitted rather than nulled — a key that is absent says
		"not computed", where a null says "computed as nothing".
		"""
		out: Dict[str, Any] = {}
		# hydro_summary.json used to be read here as a fallback for the honesty
		# payload. It is no longer written, and the fallback never fired: the
		# manager sets extra_summary on the results object in _extract, which is
		# the branch below.
		# No derived blocks at all. comparisons, soil_attribution,
		# spatial_summary and driver_matrix are the Analyzer's to compute
		# (src/agents/drivers.py): every one of them is a claim about the
		# ensemble, and the package carries evidence. Three of the four were
		# also silently empty here, because they filtered on a `soil` field
		# extraction never wrote.

		# The honesty payload: what this run cannot support, and what was
		# assumed to make it runnable. It reached the written report but not
		# the package, so an Analyzer reading only this file would have
		# stated conclusions with none of the caveats attached.
		# WHAT THE START-UP TRIM REMOVED. A series that does not start where
		# the simulation did must say so — a reader comparing this to a gauge
		# record needs to know the opening stretch is missing. It was computed
		# at extraction and then went nowhere: step 4 has always read
		# `ctx.data["spinup_dropped"]` and always found None, because nothing
		# put it in the package. Wired 2026-08-13; both runs that day dropped
		# 14 days per column and neither report said so. How long the trim is
		# depends on how the run started — 14 days warm, a year cold — and it
		# carries `basis` so the reader knows which of the two they are holding.
		drop = (results.get("spinup_dropped") if isinstance(results, dict)
				else getattr(results, "spinup_dropped", None)) or {}
		if drop:
			out["spinup_dropped"] = drop

		extra = (results.get("extra_summary") if isinstance(results, dict)
				 else getattr(results, "extra_summary", None)) or {}
		for k in ("limitations", "assumptions_ledger"):
			v = extra.get(k)
			if v:
				out[k] = v
		if "assumptions_ledger" not in out:
			af = self.run_dir / "assumptions.json"
			if af.exists():
				try:
					led = json.loads(af.read_text())
					if led:
						out["assumptions_ledger"] = led
				except Exception:
					pass
		return out

	def _sampling_blocks(self) -> Dict[str, Any]:
		"""Basin outline and how the columns were drawn.

		sampling_domain records whether the sample was clipped to the
		watershed or fell back to the raw bounding box. That is not
		decoration: a bbox sample does not represent the basin that was asked
		about, and an Analyzer that cannot see the difference will describe
		one as the other.
		"""
		out: Dict[str, Any] = {}
		f = _resolve_columns(self.run_dir)
		if not f:
			return out
		try:
			cj = json.loads(f.read_text()) or {}
		except Exception:
			return out
		for k in ("boundary", "sampling_domain", "grid", "bands"):
			if cj.get(k):
				out[k] = cj[k]
		return out

	def _package(self,
				 plan:    Dict[str, Any],
				 results: Any,
				 config:  Dict[str, Any] = None) -> Dict[str, Any]:
		"""experiment.json — everything the Analyzer needs, and nothing else.

		Deliberately simple. The expensive stages are the build and the run;
		this one is cheap to revise, so it carries what is known now rather
		than waiting for a schema the Analyzer has not yet asked for.

		The per-column rows come straight from the backend's extraction, so
		this stage stays model-agnostic: it labels and assembles, it does not
		compute.
		"""
		config = config or {}
		rows   = self._extract_rows(results)
		if isinstance(rows, dict):
			rows = list(rows.values())
		rows = [dict(r) for r in rows if isinstance(r, dict)]
		self._merge_column_metadata(rows)

		brief  = config.get("brief") or {}
		period = ((brief.get("run_settings") or {}).get("resolved_period")) or {}
		# Enumerate FAILURE, not success. Backends spell success differently
		# ("ok", "success", "completed") and a new spelling must not silently
		# report a good run as zero columns; a column with metrics and no
		# failure status produced numbers, which is the thing being counted.
		FAILED = {"failed", "error", "timeout", "cancelled"}
		ok = [r for r in rows
			  if str((r or {}).get("status", "")).lower() not in FAILED
			  and (r or {}).get("metrics")]

		pkg = {
			"model":   self.MODEL,
			"run_dir": str(self.run_dir),
			"created": datetime.now().isoformat(timespec="seconds"),
			# WHICH KIND OF STUDY THIS WAS. Carried because several stages
			# downstream have to behave differently and were deciding by
			# looking for a basin — an inference that reads a fetch failure as
			# a sweep. step1_compare skips on it; the caveat set is chosen by
			# it. Absent here, the Analyzer compared a controlled sweep against
			# four observation sets nobody had gathered.
			"archetype": (str((config.get("strategy") or {}).get("archetype")
							  or brief.get("design_archetype") or "").strip()
						  .lower() or None),
			"domain": {
				"name": (brief.get("domain") or {}).get("name"),
				"huc":  (brief.get("domain") or {}).get("huc"),
				"bbox": (brief.get("domain") or {}).get("bbox"),
			},
			"period": {"yr_start": period.get("yr_start") or config.get("yr_start"),
					   "yr_end":   period.get("yr_end")   or config.get("yr_end"),
					   "source":   period.get("source")},
			"columns_total":     len(rows),
			"columns_succeeded": len(ok),
			# What the numbers MEAN — see FIELD_SEMANTICS above.
			"field_semantics":   self.FIELD_SEMANTICS,
			"variable_units":    self._variable_units(results),
			"strategy_check":    self.strategy_report,
			"goals":             plan.get("goals"),
			"columns":           rows,

			# ── ENSEMBLE-LEVEL PRODUCTS ──────────────────────────────
			# The first cut of this stage took the per-column rows and left
			# these behind, so hydro_summary.json carried MORE than the
			# package that was supposed to replace it and the Analyzer went
			# on reading five files. Each of these is consumed by something:
			# soil_attribution by the soil_control figure, driver_matrix and
			# comparisons by controls, spatial_summary by the spatial map.
			**self._ensemble_blocks(results),

			# ── WHERE THE COLUMNS CAME FROM ──────────────────────────
			# boundary and sampling_domain draw the basin outline on the
			# spatial maps, and say whether the ensemble is a watershed
			# sample or a bbox fallback — which changes what the figures
			# are entitled to claim.
			**self._sampling_blocks(),

			"artifacts": {
				"sampling_design": "sampling_design.png",
				"extracted":       "03_results/extracted.json",
				"note": "extracted.json is the record of what was READ from the "
						"history files; this file is what the Analyzer reads, "
						"and everything derived is computed from that record",
			},
		}
		# COMPACT, not indented. This file carries ~188k daily values for a
		# 19-column run, and indent=2 spends about seven characters of
		# whitespace on each of them: 6.4 MB indented against 2.3 MB compact,
		# for identical content. At 48 columns that is the difference between
		# 16 MB and 6 MB.
		#
		# Nothing reads those arrays by eye, and the file stays valid JSON —
		# `python -m json.tool` or jq renders it readably on demand. Rounding
		# the values is the smaller lever by far (6.6 -> 6.4 MB); the
		# serialisation is where the size actually is.
		(self.run_dir / "experiment.json").write_text(
			json.dumps(pkg, separators=(",", ":"), default=str))
		print(f"✓ experiment.json — {len(ok)}/{len(rows)} column(s) "
			  f"→ the Analyzer's only input")
		# THE OTHER HALF OF THE ASSERTION. take() raises when a producer writes
		# a key no list names; this reports the mirror case — a list naming a
		# key no producer wrote. It cannot raise: `station_id` is legitimately
		# absent on every unpinned column. But a name that NEVER appears, on
		# any column of the run, is asking for a capability the pipeline does
		# not have, and `wtd_prior_m` did exactly that for a week in silence.
		keyset.report()
		return pkg


	# ─────────────────────────────────────────────────────────
	# PACKAGE — run summary + the payload the Analyzer reads
	# ─────────────────────────────────────────────────────────

	# ─────────────────────────────────────────────────────────
	# _save_llm_input DELETED 2026-08-13
	# ─────────────────────────────────────────────────────────
	# It wrote LLM_ANALYSIS_INPUT.json: experiment.json's rows plus a
	# `focus_variables` hint block, packed for AnalysisReportAgent. That
	# agent is deleted — the Analyzer's steps 3 and 4 interpret, over the
	# comparison and the caveats and the figures that this payload never
	# carried.
	#
	# FOUR OF ITS FIELDS WERE ALREADY DEAD when it went. It attached
	# `limitations` and `assumptions_ledger` off extra_summary, plus
	# `validation` from 04_analysis/validation.json and `interpretation_md`
	# from interpretation.md. Nothing fills extra_summary any more; nothing
	# writes validation.json at all; and interpretation.md is written only by
	# the standalone analyze_agentic.py, which runs long after this point. So
	# all four were absent on every pipeline run. The comment above them said
	# the written report could be held to the same standard as the
	# interpretation — it had not been able to for some time, and nothing said
	# so.
	# ─────────────────────────────────────────────────────────
	# RUN SUMMARY
	# ─────────────────────────────────────────────────────────

	# ─────────────────────────────────────────────────────────
	# RUN SUMMARY
	# ─────────────────────────────────────────────────────────
	@staticmethod
	def _outcome_map(results: Any) -> Dict[str, bool]:
		"""{case name -> did it succeed}, from either shape a backend returns.

		ELM's _run returns {case_name: bool}. PFLOTRAN's returns a LIST of
		per-case dicts, which carries more (runtime, return code, reason) and
		is the better shape. This function used to assume the dict and called
		.values() on it, so a PFLOTRAN run raised AttributeError at the very
		last stage — after the compute, after experiment.json, after the whole
		Analyzer had run. Everything of value was already on disk and the run
		still ended in a traceback with no RUN_SUMMARY.json.
		"""
		if isinstance(results, dict):
			return {str(k): bool(v) for k, v in results.items()}
		out: Dict[str, bool] = {}
		for r in (results or []):
			if isinstance(r, dict):
				name = r.get('case_name') or r.get('id')
				if name:
					out[str(name)] = (r.get('status') == 'completed'
									  if 'status' in r else True)
		return out

	@staticmethod
	def _n_results(results: Any) -> Optional[int]:
		"""How many results a backend returned, for the run state. None when the
		shape has no length — a count is bookkeeping, not worth a raise."""
		return len(results) if hasattr(results, "__len__") else None

	def _pending_summary(self,
						 plan:        Dict[str, Any],
						 experiments: List[Dict],
						 record:      Dict[str, Any],
						 start_time:  datetime,
						 stage:       str = "run") -> Dict[str, Any]:
		"""The run summary for an ensemble that is still in a queue.

		THE SAME SHAPE as a finished run's, with status="pending" and the job
		id added. Deliberately not a smaller dict: every existing caller reads
		experiments_success and experiments_total off this, and handing them a
		different shape would turn "your job is queued" into an AttributeError
		three call frames away.
		"""
		end_time = datetime.now()
		summary  = self._create_run_summary(plan, experiments, {},
											start_time, end_time, pending=True)
		summary.update({
			'status':          'pending',
			'job_id':          record.get('job_id'),
			# BOTH IDS, because they mean different things and the caller needs
			# to say which is which. `job_id` is the one to POLL, which is B
			# when a job B was chained; A is the one actually simulating. Only
			# `job_id` used to travel, so every reader downstream could name
			# the analysis job and none could name the ensemble.
			'job_id_a':        record.get('job_id_a'),
			'job_id_b':        record.get('job_id_b'),
			# WHICH stage is waiting. With build_cases job-shaped as well as run,
			# "job 770603 is queued" no longer says whether the cases are being
			# built or the ensemble is being simulated — and those are hours
			# apart in what happens next.
			'pending_stage':   record.get('stage') or stage,
			'submitted_at':    record.get('at') or start_time.isoformat(),
			'resume_command':  f"python workflow.py --resume {self.run_dir}",
		})
		try:
			self._save_run_summary(summary)
		except Exception as e:                                  # noqa: BLE001
			print(f"   ⚠️  RUN_SUMMARY.json failed ({e}) — the job is still "
				  f"recorded in {self.STATE_FILE}")

		# WHICH JOB DOES WHICH. `record['job_id']` is what the framework POLLS,
		# and when a job B was chained that is B — the analysis tail, not the
		# ensemble. So this line read "19 experiment(s) queued as job 773439"
		# for a run whose 19 columns were job 773438, and a user checking
		# squeue for the number they were given saw a 7-minute job where they
		# expected an 18-minute one. Both ids are on the record; name both.
		n = summary['experiments_pending']
		jid_a = record.get('job_id_a')
		jid_b = record.get('job_id_b')
		print(f"\n{'=' * 60}")
		print(f"{self.MODEL.upper()} SUBMITTED — {n} column(s) queued. "
			  f"NOTHING HAS RUN YET.")
		if jid_a and jid_b and jid_a != jid_b:
			print(f"  job {jid_a}   runs the {n} columns")
			print(f"  job {jid_b}   extracts, packages and analyses them "
				  f"when {jid_a} finishes")
		else:
			print(f"  job {record.get('job_id')}   runs the {n} columns")
		print(f"Output:  {self.run_dir}")
		print(f"Answer:  {self.run_dir}/04_analysis/analysis.json  "
			  f"(when the analysis job finishes)")
		print(f"Watch:   squeue -u $USER")
		print(f"If the analysis job never runs: {summary['resume_command']}")
		print(f"{'=' * 60}\n")
		return summary

	def _create_run_summary(self,
							plan:        Dict[str, Any],
							experiments: List[Dict],
							results:     Any,
							start_time:  datetime,
							end_time:    datetime,
							pending:     bool = False
							) -> Dict[str, Any]:
		n_total   = len(experiments)
		ok        = self._outcome_map(results)
		n_success = sum(1 for v in ok.values() if v)

		# .get(), not [], deliberately. This is the LAST step: the ensemble is
		# already computed and experiment.json already written, so a missing
		# key here would throw away a finished run over a summary field.
		exp_details = [
			{
				'name':              e.get('scenario_name') or e.get('case_name')
									 or e.get('id'),
				# `id` as well as `case_name`: PFLOTRAN's experiments are keyed
				# by id, so keying only on case_name marked every one of them
				# failed in a summary written after they had all succeeded.
				'case_name':         e.get('case_name') or e.get('id'),
				# A queued column has not failed — it has not been asked yet.
				# Calling it 'failed' would put 19 failures in the summary of a
				# run whose job is sitting healthily in the queue.
				'status':            'completed'
									 if ok.get(str(e.get('case_name')
													or e.get('id')))
									 else ('pending' if pending else 'failed'),
				'forcing_period':    e.get('forcing_period'),
				'forcing_start':     e.get('forcing_start'),
				'forcing_end':       e.get('forcing_end'),
				'model_type':        self.MODEL,
				# The measured time when the backend recorded one. Hardcoding 0
				# made step 4 report a per-column runtime of zero for every
				# column of every run — a number that looked measured and was
				# not.
				'runtime_seconds':   e.get('runtime_seconds') or 0,
				'timesteps':         0,
				'newton_iterations': 0,
			}
			for e in experiments
		]

		return {
			'run_directory':         str(self.run_dir),
			'start_time':            start_time.isoformat(),
			'end_time':              end_time.isoformat(),
			'total_runtime_seconds': (
				end_time - start_time
			).total_seconds(),
			'experiments_total':     n_total,
			'experiments_success':   n_success,
			# Unfinished ≠ failed. Both keys are always present so a consumer
			# can add them up without knowing which kind of run it is reading.
			'experiments_failed':    0 if pending else n_total - n_success,
			'experiments_pending':   n_total - n_success if pending else 0,
			'status':                'pending' if pending else 'completed',
			'experiments':           exp_details,
			'convergence_warnings':  [],
			'output_files': {
				'inputs':             str(self.input_dir),
				'setup_plots':        str(self.setup_plots_dir),
				'results':            str(self.results_dir),
				'analysis':           str(self.analysis_dir),
				'experiment_summary': str(
					self.input_dir / "experiment_summary.json"),
				'execution_report':   str(
					self.results_dir / "execution_report.txt"),
				'results_csv':        str(
					self.results_dir / "results_summary.csv"),
				'extracted':          str(
					self.results_dir / "extracted.json"),
				# experiment.json is the manager's product. `llm_input` used to
				# name an alias beside it, written for the report agent; both
				# are gone (2026-08-13).
				'experiment':         str(self.run_dir / "experiment.json"),
				'analysis_report':    str(
					self.analysis_dir / "analysis.json"),
			},
			'model_type': self.MODEL,
		}


	def _save_run_summary(self,
						  run_summary: Dict[str, Any]) -> None:
		summary_file = self.run_dir / "RUN_SUMMARY.json"
		with open(summary_file, 'w') as f:
			json.dump(run_summary, f, indent=2, default=str)
		print(f"✓ RUN_SUMMARY.json saved")


	def _save_error(self,
					error:      Exception,
					start_time: datetime) -> None:
		import traceback
		error_log = {
			'run_directory': str(self.run_dir),
			'start_time':    start_time.isoformat(),
			'end_time':      datetime.now().isoformat(),
			'status':        'failed',
			'error':         str(error),
			'traceback':     traceback.format_exc(),
			'model_type':    self.MODEL,
		}
		error_file = self.run_dir / "ERROR_LOG.json"
		with open(error_file, 'w') as f:
			json.dump(error_log, f, indent=2)


def _resolve_columns(run_dir: Path) -> Optional[Path]:
	"""columns.json, canonical location first, then the legacy top level."""
	for p in (run_dir / "01_inputs" / "columns.json", run_dir / "columns.json"):
		if p.exists():
			return p
	return None


def _load_tool(name: str):
	"""Import tools/<name>.py by path — tools/ is a script dir, not a package."""
	import importlib.util
	if name in sys.modules:
		return sys.modules[name]
	path = Path(__file__).resolve().parents[2] / "tools" / f"{name}.py"
	spec = importlib.util.spec_from_file_location(name, path)
	mod  = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	spec.loader.exec_module(mod)
	return mod
