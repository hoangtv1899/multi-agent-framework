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
import sys
from pathlib  import Path
from datetime import datetime
from typing   import Any, Dict, List, Optional

sys.path.insert(0, "src")

# Used only when neither the caller nor the strategy says. The planner
# always emits n_bands, so reaching this means the plan was hand-written.
DEFAULT_BANDS = 4


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
	           needs, and what gets written into the ledger
	  detail   anything else the backend wants back when it is polled; it is
	           stored verbatim in the ledger entry and handed to _poll
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
	# THE STAGE LEDGER — what has already been done, on disk
	# ─────────────────────────────────────────────────────────
	#
	# run_state.json records each stage as it finishes, with the artifacts it
	# left behind. Written for one reason: an ELM ensemble is ~40 minutes in a
	# queue, and a process that has to sit and block for it is a process that
	# cannot be interrupted, resumed, or moved between sessions. The ledger is
	# what lets a later invocation know that materialize and build are already
	# done and the only thing outstanding is a job id.
	#
	# PHASE 1 WRITES IT AND NOTHING READS IT. That is deliberate: the file can
	# be checked against runs that already exist before any control flow
	# depends on it, so a bug in the ledger cannot break a working pipeline.
	STATE_FILE = "run_state.json"

	# The stage sequence, in order, as execute_plan runs it. Named here so the
	# ledger's reader and its writer cannot disagree about what a stage is
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
		"""The ledger, or an empty one. Never raises — a corrupt or absent
		ledger must degrade to "nothing is known to be done", which is the
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
		"""Record one stage. Never raises: the ledger is bookkeeping, and a
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
		that was never written would make the ledger a claim rather than a
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

	def _rehydrate_extract(self):
		"""_extract's rows, from experiment.json — which _package already
		writes and which carries the rows and their units verbatim."""
		p = self.run_dir / "experiment.json"
		if not p.exists():
			return None
		d = json.loads(p.read_text())
		rows = d.get("columns")
		if not isinstance(rows, list):
			return None
		import types
		units = d.get("variable_units") or {}
		ns = types.SimpleNamespace(results=rows, units=dict(units),
								   summary={"units": dict(units)})
		return ns

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
			# whole point of the ledger.
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
				_reuse("extract", f"{len(analyzer.results)} row(s) from "
								  f"experiment.json")
			else:
				analyzer = self._extract(
					experiments, plan=experiment_plan, config=config)
			self._mark("extract",
					   n_rows=len(getattr(analyzer, "results", []) or []))

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
			# what happened. Only the interpretation is skipped, and the ledger
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
					Analyzer(str(self.run_dir)).run(results=analyzer,
													config=config)
					self._mark("analyze")
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

			print("\n📦 STEP 5: Packaging LLM Input")
			print("-" * 40)
			self._save_llm_input(experiment_plan, analyzer)

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
		job is still queued or running. `record` is the ledger's entry for that
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

	def _advance(self, stage: str, fn, *, state: Dict[str, Any], resume: bool,
				 experiments, config, **done_fields):
		"""Run one stage that may hand back a job id instead of finishing.

		Three paths, and the ledger is what tells them apart:

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
			mcp_clients  — dict of live MCP clients (terrain/fan_wtd/geology)
		Optional: yr_start, yr_end, soil_config, substrate, n_columns, n_bands.
		"""
		if self._already_executable(plan):
			return plan

		exp     = _load_tool("expand_sampling")
		brief   = config.get("brief") or {}
		clients = config.get("mcp_clients") or {}

		bbox = exp._bbox_from_brief(brief)
		if not bbox:
			raise ValueError(
				"Cannot materialize sampling: no domain bbox in the reception "
				"brief, and the plan has no executable payload. Pass "
				"config['brief'] with domain.bbox, or supply an explicit plan.")
		if not clients.get("terrain"):
			raise ValueError(
				"Cannot materialize sampling: the 'terrain' MCP client is "
				"required. Pass config['mcp_clients'].")

		n_total = config.get("n_columns") or exp._n_from_plan(plan)
		if not n_total:
			raise ValueError(
				"Cannot materialize sampling: no column count in the plan "
				"and none in config.")
		# The planner's band count, not reception's list length. See
		# expand_sampling._n_bands_from_plan for what that fallback got wrong.
		bands = (config.get("n_bands")
				 or exp._n_bands_from_plan(config.get("strategy") or plan)
				 or DEFAULT_BANDS)

		config = self.check(plan, config)

		print("\n🗺️  STEP 0: Materializing Sampling")
		print("-" * 40)
		print(f"   bbox={bbox} N={n_total} bands={bands}")

		# Clip the rectangular DEM sample to the real basin when we know the HUC.
		# Without a boundary the sample is the raw bbox, so columns can land
		# OUTSIDE the basin — the run still completes, but the ensemble no
		# longer represents the watershed that was asked about. That is a
		# scientific difference, so say so loudly rather than degrade silently.
		boundary = None
		huc = (brief.get("domain") or {}).get("huc")
		if huc:
			try:
				b = clients["terrain"].call_tool_json(
					"get_watershed_boundary",
					{"huc": huc, "huc_level": len(str(huc))}) or {}
				boundary = b.get("rings")
			except Exception as e:
				print(f"   ⚠️  boundary lookup FAILED for HUC {huc} ({e})")
		if not boundary:
			why = ("no HUC in the reception brief"
				   if not huc else f"boundary lookup returned nothing for HUC {huc}")
			name = (brief.get("domain") or {}).get("name") or "the watershed"
			print(f"   ⚠️  WARNING: sampling the RAW BBOX, not the watershed — {why}.")
			print(f"   ⚠️  Columns may fall outside the basin; the ensemble is "
				  f"a bounding-box sample, not '{name}'.")
			print(f"   ⚠️  Fix: ensure reception resolves a HUC, or pass "
				  f"config['boundary'] explicitly.")
		boundary = config.get("boundary", boundary)

		res = exp.expand(clients, bbox, n_total, bands, boundary=boundary)
		if res.get("error"):
			raise RuntimeError(f"Sampling expansion failed: {res['error']}")
		columns = res.get("columns", [])
		if not columns:
			raise RuntimeError("Sampling expansion produced no columns.")

		# Keep the WBD polygon in the result: plot_columns() draws it as the
		# basin outline in panel (a), and it makes columns.json self-contained
		# (grid + boundary) so a later re-plot needs no MCP fetch.
		if boundary:
			res["boundary"] = boundary

		# Provenance: was this a true watershed sample or a bbox fallback?
		res["sampling_domain"] = {
			"clipped_to_watershed": bool(boundary),
			"huc": huc,
			"name": (brief.get("domain") or {}).get("name"),
			"bbox": bbox,
			"caveat": None if boundary else (
				"Columns were sampled from the bounding box, NOT clipped to the "
				"watershed boundary (no HUC resolved). Some columns may lie "
				"outside the basin; treat the ensemble as a bbox sample."),
		}

		yr_start = int(config.get("yr_start", 1995))
		yr_end   = int(config.get("yr_end",   yr_start))

		# Backend refinement BEFORE persisting — see the module docstring. The
		# ELM warm start snaps every column to its donor gridcell.
		refine = self._refine_columns(columns, config) or {}
		res["columns"] = columns

		(self.input_dir / "columns.json").write_text(json.dumps(res, indent=2))
		(self.run_dir  / "columns.json").write_text(json.dumps(res, indent=2))

		# The sampling-design figure: domain map + watershed outline,
		# hypsometry with band edges, soil configs, Fan WTD vs elevation,
		# per-band allocation, NLDAS precip gradient. Same function
		# tools/expand_sampling.py --plot uses; forcing_year populates the
		# precip-vs-elevation panel that is otherwise blank.
		try:
			png = exp.plot_columns(
				res, str(self.run_dir / "sampling_design.png"),
				forcing_year = yr_start)
			print(f"✓ sampling design → {Path(png).name}")
		except Exception as e:
			print(f"   ⚠️  sampling_design.png failed ({e}) — non-fatal")

		executable = self._to_run_plan(plan, columns, config, refine)
		merged     = {**plan, **executable}

		# The honesty payload reads this back at extraction; without it an
		# integrated run shipped an empty ledger.
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
		for src in (getattr(results, "units", None),
					(getattr(results, "summary", None) or {}).get("units"),
					(self.analysis_dir / "hydro_summary.json")):
			if isinstance(src, dict) and src:
				return src
			if isinstance(src, Path) and src.exists():
				try:
					u = (json.loads(src.read_text()) or {}).get("units")
					if u:
						return u
				except Exception:
					pass
		return {}

	# Fields the extraction cannot know, because they describe how the column
	# was CHOSEN rather than what the model did with it. Each is here because
	# something downstream reads it:
	#
	#   band, band_range_m   area-weighting the ensemble. The sampler allocates
	#                        >=1 column per elevation band regardless of band
	#                        size, so an unweighted mean over-weights small
	#                        bands. Without this the weighting silently
	#                        degrades to a plain average.
	#   fan_wtd_m            the prior water-table depth, compared against
	#                        modelled ZWT and plotted per column.
	#   soil_*               which soil this column actually got, and from
	#                        where — the soil-attribution figure and any claim
	#                        that soil explains a gradient rest on it.
	#   lat, lon, elevation_m  normally come from the extraction, which reads
	#                        them off the coupler. Listed here too because
	#                        columns.json is the AUTHORITY on where a column
	#                        is, and an extraction invoked without them
	#                        produced 19 rows with elevation_m absent — which
	#                        silently flattens every elevation figure and
	#                        every gradient claim to a single point.
	COLUMN_METADATA = ("lat", "lon", "elevation_m",
					   "band", "band_range_m", "fan_wtd_m", "soil_top_texture",
					   "soil_layers", "soil_source", "soil_profile")

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
			for k in self.COLUMN_METADATA:
				if src.get(k) is not None and r.get(k) is None:
					r[k] = src[k]
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
		hs: Dict[str, Any] = {}
		f = self.analysis_dir / "hydro_summary.json"
		if f.exists():
			try:
				hs = json.loads(f.read_text()) or {}
			except Exception as e:
				print(f"   ⚠️  could not read the extraction ({e})")
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
		extra = getattr(results, "extra_summary", None) or {}
		for k in ("limitations", "assumptions_ledger"):
			v = extra.get(k) or hs.get(k)
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
		rows   = getattr(results, "results", None) or []
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
				"extracted":       "04_analysis/hydro_summary.json",
				"note": "hydro_summary.json holds the same per-column rows; "
						"this file is the one the Analyzer reads",
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
		return pkg


	# ─────────────────────────────────────────────────────────
	# PACKAGE — run summary + the payload the Analyzer reads
	# ─────────────────────────────────────────────────────────

	# ─────────────────────────────────────────────────────────
	# STEP 5 — PACKAGE LLM INPUT (top level)
	# ─────────────────────────────────────────────────────────
	def _save_llm_input(self,
						plan:     Dict[str, Any],
						analyzer: Any) -> None:
		"""Save LLM_ANALYSIS_INPUT.json at the top level of run_dir.

		An ALIAS. experiment.json is the manager's real product; this exists
		because the standalone report tools open it by this name.

		Non-fatal, and that matters: it runs AFTER the compute has succeeded,
		so anything raising here throws away a finished ensemble over a
		convenience file. get_llm_analysis_input() is also specific to ELM's
		results object — a backend without it must still finish its run.
		"""
		try:
			llm_input = analyzer.get_llm_analysis_input()
		except Exception as e:
			print(f"   ⚠️  LLM_ANALYSIS_INPUT.json skipped ({e}) — "
				  f"experiment.json is unaffected")
			return
		llm_input['experiment_plan'] = plan
		llm_input['run_directory']   = str(self.run_dir)

		# get_llm_analysis_input() omits extra_summary, so the honesty payload
		# and the observation verdicts never reached the report agent. Attach
		# them here so the written report can be held to the same standard as
		# 04_analysis/interpretation.md.
		if getattr(analyzer, 'extra_summary', None):
			llm_input['limitations'] = analyzer.extra_summary.get('limitations')
			llm_input['assumptions_ledger'] = \
				analyzer.extra_summary.get('assumptions_ledger')
		vp = self.analysis_dir / "validation.json"
		if vp.exists():
			try:
				val = json.loads(vp.read_text())
				llm_input['validation'] = {
					"targets": val.get("targets"),
					"observation_inventory": val.get("observation_inventory"),
					"hydrograph_metrics": {
						k: v for k, v in (val.get("hydrograph") or {}).items()
						if k not in ("days", "obs", "mod")},
				}
			except Exception as e:
				print(f"   ⚠️  could not attach validation to LLM input: {e}")
		ip = self.analysis_dir / "interpretation.md"
		if ip.exists():
			llm_input['interpretation_md'] = ip.read_text()

		try:
			llm_file = self.run_dir / "LLM_ANALYSIS_INPUT.json"
			with open(llm_file, 'w') as f:
				json.dump(llm_input, f, indent=2, default=str)
			print(f"✓ LLM_ANALYSIS_INPUT.json saved")
		except Exception as e:
			print(f"   ⚠️  LLM_ANALYSIS_INPUT.json not written ({e})")

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
		"""How many results a backend returned, for the ledger. None when the
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

		n = summary['experiments_pending']
		print(f"\n{'=' * 60}")
		print(f"{self.MODEL.upper()} SUBMITTED: {n} experiment(s) queued as job "
			  f"{record.get('job_id')}")
		print(f"Output: {self.run_dir}")
		print(f"Resume: {summary['resume_command']}")
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
				'hydro_summary':      str(
					self.analysis_dir / "hydro_summary.json"),
				# experiment.json is the product; llm_input is the alias the
				# standalone report tools open by name.
				'experiment':         str(self.run_dir / "experiment.json"),
				'llm_input':          str(
					self.run_dir / "LLM_ANALYSIS_INPUT.json"),
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
