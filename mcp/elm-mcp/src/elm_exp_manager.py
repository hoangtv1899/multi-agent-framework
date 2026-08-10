#!/usr/bin/env python3
"""
ELM Experiment Manager
src/core/elm_exp_manager.py

Single responsibility: orchestrate the ELM execution pipeline — the
Experiment Manager box of the framework (see ARCHITECTURE.md).

Output directory structure:

    elm_run_YYYYMMDD_HHMMSS/
        ├── columns.json  run_plan.json  plan.json  reception_brief.json
        ├── assumptions.json                        (the honesty ledger)
        ├── sampling_design.png                     (step 0, materialize)
        ├── warmstart/                              (step 0b, only if warm)
        │   ├── Warmstart_<col>.nc
        │   └── warmstart.json
        ├── 01_inputs/
        │   └── experiment_summary.json
        ├── 02_setup_plots/
        │   └── column_surfaces.png                 (real FSURDAT per column)
        ├── 03_results/
        │   ├── execution_report.txt
        │   └── results_summary.csv
        ├── 04_analysis/
        │   ├── hydro_summary.json  validation.json  interpretation.md
        │   ├── partitioning.png  controls.png  soil_control.png
        │   ├── wtd_columns.png
        │   └── validation_{hydrograph,yield,water_table,swe,context}.png
        ├── 05_pflotran/                            (step 4d, only if coupled)
        │   ├── <col>/<col>.in + outputs
        │   ├── pflotran_summary_coupled.json  pflotran_study.json
        │   └── pflotran_coupled.png
        ├── ANALYSIS_REPORT.json
        ├── LLM_ANALYSIS_INPUT.json
        └── RUN_SUMMARY.json

ELM cases live at: $PSCRATCH/E3SMv3/1D_ELM.*/
"""
import csv
import glob
import os
import json
import sys
from pathlib  import Path
from datetime import datetime
from typing   import Dict, Any, List, Optional

sys.path.insert(0, "src")

from core.exp_manager_base import ExperimentManagerBase, Pending
from elm_results_analyzer  import ELMResultsAnalyzer


# parents[3]: this file is mcp/elm-mcp/src/. _ROOT is the FRAMEWORK root — the
# only thing left that needs it is _load_tool, for the PFLOTRAN coupling CLIs.
_ROOT = Path(__file__).resolve().parents[3]



def _load_tool(name: str):
	"""Import tools/<name>.py by path — tools/ is a script dir, not a package.

	Down to one user: the PFLOTRAN coupling (step 4d, build_pflotran_cases +
	analyze_pflotran_coupled), so that stage and its standalone CLIs share one
	implementation. Everything else that used this now lives beside this file
	or behind the MCP.
	"""
	import importlib.util
	if name in sys.modules:
		return sys.modules[name]
	path = _ROOT / "tools" / f"{name}.py"
	spec = importlib.util.spec_from_file_location(name, path)
	mod  = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	spec.loader.exec_module(mod)
	return mod


# ─────────────────────────────────────────────────────────────────────
# ELM EXPERIMENT MANAGER
# ─────────────────────────────────────────────────────────────────────
class ELMExpManager(ExperimentManagerBase):
	"""
	Executes ELM experiment plans. EVERY ELM COMPUTATION IS THE MCP'S.

	What is left here is the four things that are not ELM knowledge: WHEN to
	call the server (the stage order is the base's), WHERE the run directory
	is, WHAT the derived fields mean (FIELD_SEMANTICS), and arranging the
	framework's own follow-up job. The stages read:

	    materialize        one MCP call — warm start, donor soil, surfaces,
	                       case_inputs.json, the run plan  (_refine_columns)
	    build_case_inputs  read back what that call wrote
	    build_cases        submit JOB A (build + run), submit JOB B, stop
	    run                collect from disk — A already ran the columns
	    extract            read the history NetCDFs
	    couple             optional one-way handoff to PFLOTRAN

	THERE IS NO LOCAL PATH. It was deleted 2026-08-10: _refine_columns requires
	the `elm` client, so every branch that ran ELM by import was unreachable
	from the first stage onward and only looked like a fallback. With it went
	_run_batch, _submit_via_mcp, the ELMExperimentBuilder handle, and the last
	use of tools/submit_cases.sh from this side of the boundary.

	Figures, observation validation and interpretation are NOT here — they are
	the Analyzer (src/agents/analyzer.py), which execute_plan invokes after
	extraction.
	"""
	MODEL = "elm"

	def __init__(self,
				 base_output_dir: str = "./workflow_outputs",
				 run_dir: str = None):
		super().__init__(base_output_dir, run_dir)
		print(f"\n{'=' * 60}")
		print(f"ELM Experiment Manager")
		print(f"Run dir : {self.run_dir}")
		print(f"Cases   : $PSCRATCH/E3SMv3/")
		print(f"{'=' * 60}\n")

	# ─────────────────────────────────────────────────────────
	# BACKEND HOOKS — what the base cannot know about ELM
	# ─────────────────────────────────────────────────────────
	def _already_executable(self, plan: Dict[str, Any]) -> bool:
		"""CONDITIONS_COUPLERS is ELM's executable payload (the legacy shape)."""
		return bool(plan.get("CONDITIONS_COUPLERS"))

	def _refine_columns(self, columns, config: Dict[str, Any]) -> Dict[str, Any]:
		"""ONE MCP call: warm start, donor soil, surfaces, case_inputs.json.

		Runs inside the base's materialize, BEFORE columns.json is written and
		before the design figure is drawn — the warm start snaps each column to
		its donor gridcell, so anything persisted earlier describes a plan the
		run will not follow.

		WHY THE WHOLE INPUT BUILD HAPPENS HERE, in a hook named for refinement.
		The MCP does all six input steps in one call and the manager used to
		split them across two stages, with the columns.json write wedged
		between. That split is why routing this through the MCP was impossible
		for a week: calling the tool from the later stage re-ran the warm start,
		and calling it from here could not work while the tool's only input was
		a file that does not exist yet at this moment. The tool now takes the
		columns as DATA, so the call lands here — the one moment that satisfies
		the ordering constraint — and the two stages after it have nothing left
		to compute.

		THE COLUMNS ARE REPLACED IN PLACE. The base persists the same list
		object it handed us, so returning a new one would silently persist the
		pre-snap coordinates.
		"""
		client = self._mcp(config)
		if client is None:
			raise RuntimeError(
				"the elm MCP is required to build ELM inputs — the local "
				"by-import path was deleted 2026-08-10 (docs/EXP_MANAGER_ELM.md "
				"§4). Register an `elm` client in mcp_config.json.")
		yr_start = int((config or {}).get("yr_start", 1995))
		ws = (config or {}).get("warm_start")
		out = self._mcp_call(client, "build_elm_inputs_from_location", {
			"run_dir":       str(self.run_dir),
			"columns":       columns,
			"yr_start":      yr_start,
			"yr_end":        int((config or {}).get("yr_end", yr_start)),
			"soil_config":   str((config or {}).get("soil_config", "native")),
			"substrate":     str((config or {}).get("substrate", "extrapolate")),
			"conus_restart": str((ws or {}).get("conus_restart", "")
								 if isinstance(ws, dict) else ""),
		}, budget=900)
		if not out.get("ok"):
			raise RuntimeError(
				f"build_elm_inputs_from_location failed: {out.get('error')}")
		columns[:] = out["columns"]          # snapped — see the docstring
		self._mcp_inputs = out
		print(f"✓ {out['n_cases']} case input(s) built via the elm MCP")
		return {"mcp_inputs": out}

	def _to_run_plan(self, plan, columns, config, refine) -> Dict[str, Any]:
		"""The plan the MCP already built, handed back.

		Rebuilding it here would be the same computation twice with two chances
		to disagree; the base still persists it as run_plan.json and resumes
		from it, so it is returned by the call rather than recomputed.
		"""
		out = (refine or {}).get("mcp_inputs") or getattr(self, "_mcp_inputs", None)
		if not out or "run_plan" not in out:
			raise RuntimeError(
				"no run plan from the elm MCP — _refine_columns must run first")
		return out["run_plan"]

	# ─────────────────────────────────────────────────────────
	# MAIN ENTRY POINT
	# ─────────────────────────────────────────────────────────
	# execute_plan now lives in ExperimentManagerBase. ELM keeps the stages
	# and the declarations; the sequence and its ordering guarantees are the
	# base's, so a second backend cannot re-implement them differently.
	NEEDS_CASE_BUILD   = True          # CIME case build, ~8 min for the first
	NEEDS_SCHEDULER = True          # sbatch + wait
	COUPLES_TO      = "pflotran"    # one-way QINFL handoff, when asked for

	# What ELM's derived metrics MEAN. Lives here, not on the base, because
	# every entry names an ELM history variable: on the base it was silently
	# inherited by PFLOTRAN, whose experiment.json then advertised a surface
	# water budget it never computed. See ExperimentManagerBase.FIELD_SEMANTICS.
	FIELD_SEMANTICS = {
		"precip_mm_yr":            {"units": "mm/yr", "from": ["RAIN", "SNOW"],
									"note": "TOTAL precipitation: rain + snow"},
		"rainfall_mm_yr":          {"units": "mm/yr", "from": ["RAIN"]},
		"snowfall_mm_yr":          {"units": "mm/yr", "from": ["SNOW"]},
		"annual_runoff_mm_yr":     {"units": "mm/yr", "from": ["QOVER"],
									"note": "SURFACE runoff only. QDRAI "
											"(subsurface drainage) is NOT "
											"included, so this is not the "
											"streamflow-comparable total"},
		"annual_recharge_mm_yr":   {"units": "mm/yr", "from": ["QCHARGE"]},
		# NOT fractions of precipitation. Both denominators are
		# (QCHARGE + QOVER): these say how the drainage SPLITS between
		# recharge and runoff, and they sum to 1 by construction even when
		# both terms are ~0. Reading them as runoff/P is wrong by a factor
		# of P/(QCHARGE+QOVER) — on the 2019 Gunnison run that made columns
		# draining ~0 mm/yr report recharge_fraction = 1.00.
		"runoff_fraction":         {"units": "1",
									"from": ["QOVER", "QCHARGE"],
									"note": "QOVER / (QCHARGE + QOVER) — the "
											"recharge-vs-runoff SPLIT, not a "
											"fraction of precipitation"},
		"recharge_fraction":       {"units": "1",
									"from": ["QCHARGE", "QOVER"],
									"note": "QCHARGE / (QCHARGE + QOVER) — the "
											"recharge-vs-runoff SPLIT, not a "
											"fraction of precipitation"},
		"recharge_to_runoff_ratio": {"units": "1", "from": ["QCHARGE", "QOVER"]},
		"precip_total_mm_yr":      {"units": "mm/yr", "from": ["RAIN", "SNOW"],
									"note": "water-budget total; "
											"precip_mm_yr is the same sum"},
		"water_budget":            {"units": "mm/yr", "from": ["RAIN", "SNOW",
															   "QOVER", "QDRAI",
															   "QCHARGE", "TWS"]},
		"water_table_depth_m":     {"units": "m", "from": ["ZWT"],
									"note": "positive downward from the surface"},
		"peak_swe_mm":             {"units": "mm", "from": ["H2OSNO"]},
		"tws_seasonal_range_mm":   {"units": "mm", "from": ["TWS"]},
	}

	def _couple(self, plan, config):
		"""Base calls this when COUPLES_TO is set; delegate to the existing
		implementation so the coupling code has one home."""
		return self._couple_pflotran(plan, config)

	# ─────────────────────────────────────────────────────────
	# THE MCP — the only path there is
	# ─────────────────────────────────────────────────────────
	# Not a default with a fallback behind it: _refine_columns raises without a
	# client, so a study that reaches any stage below has one. run_via_mcp=False
	# now fails at the first stage instead of quietly running a second
	# implementation of ELM that nobody has exercised since the move.
	MCP_NAME = "elm"

	def _announce(self, experiments, config) -> str:
		"""Say what is about to happen. PRINTS ONLY — it never asks.

		Asking belongs to the coordinator: it owns every other question put to
		the user (interactive_reception, the --resume prompt) and it owns the
		TTY guard that keeps those from hanging a scripted run. A backend that
		called input() here would be a compute stage holding the terminal open,
		and it only ended up here because the run directory and the column
		count happened to be in scope.

		The address is whatever the coordinator, the environment, or the
		remembered preference already settled on.
		"""
		n = len(experiments or [])
		email = self._notify_email(config)
		print()
		print(f"⚙️  This ELM study runs unattended.")
		print(f"    {n} column(s) · build + run + analysis, ~25-40 min in the queue.")
		print(f"    Nothing to watch. Results will appear in:")
		print(f"      {self.analysis_dir}")
		print(f"    ✉  {email}" if email else
			  "    (no notification address — nothing will tell you when it lands)")
		print()
		return email

	# ─────────────────────────────────────────────────────────
	# STEP 2 — JOBS A AND B  (the whole of the compute)
	# ─────────────────────────────────────────────────────────
	def _build_cases(self, experiments: List[Dict], config: Dict[str, Any] = None):
		"""JOB A through the MCP, then JOB B by dependency. Returns a Pending.

		Named build_cases because that is the base's slot for "a stage that may
		hand back a job id"; job A does the build AND the run, so _run has
		nothing left to submit by the time it is reached. ONE JOB, ALWAYS — the
		split flow cost three invocations for a 25-40 minute study (submit the
		build, come back and submit the run, come back and analyze), which made
		the person the scheduler.

		A builds the cases and runs every column, and stops. B is submitted HERE,
		with --dependency=afterany on A's id, and does the analysis. Then this
		returns and the framework exits — SLURM starts B by itself.

		WHY B IS SUBMITTED BY THE FRAMEWORK AND NOT BY A. The tool this replaces,
		run_elm_study, ended its job by running `workflow.py --finalize`: the
		server's job executing the client's code, which is the dependency the
		boundary rule forbids and the reason the two could not be separated.
		Submitting B from this side inverts it — the framework asks for the model
		run and arranges its own follow-up, and nothing in the server calls back.

		afterany, NEVER afterok. With afterok a failed ensemble means B never runs
		and no mail is ever sent — the silent failure of 2026-08-06, twice. afterany
		means B always runs and always reports, including "the ensemble failed".
		"""
		config = config or {}
		client = self._mcp(config)
		if client is None:
			raise RuntimeError(
				"the elm MCP is required to build and run ELM cases — the "
				"local by-import path was deleted 2026-08-10 "
				"(docs/EXP_MANAGER_ELM.md §4). Register an `elm` client in "
				"mcp_config.json.")
		# The server reads 01_inputs/case_inputs.json and nothing else. It does
		# not generate inputs and never opens a surfdata file: FSURDAT and
		# FINIDAT — the warm start's two products — crossed the boundary as
		# data when _refine_columns wrote this file.
		src = self.input_dir / self.CASE_INPUTS
		if not src.is_file():
			raise RuntimeError(
				f"{src} is missing — _refine_columns should have written it "
				f"via the elm MCP")
		email = self._announce(experiments, config)
		out = self._mcp_call(client, "run_elm_ensemble", {
			"run_dir":  str(self.run_dir),
			"queue":    str(config.get("queue", "")),
			"walltime": str(config.get("study_walltime", "02:00:00")),
		}, budget=600)
		if out.get("error"):
			raise RuntimeError(f"run_elm_ensemble: {out['error']}")
		job_a = str(out["job_id"])
		job_b = self._submit_job_b(job_a, config, email)
		if email:
			print(f"   ✉  {email} will be mailed when job B lands")
		return Pending(job_b or job_a, n_cases=len(experiments or []),
					   log=out.get("log_path"), via="mcp", scope="study",
					   notify=email or None)

	def _submit_job_b(self, job_a: str, config: Dict[str, Any],
					  email: str = "") -> Optional[str]:
		"""JOB B — the analysis, held until A ends. Returns its id, or None.

		Non-fatal: A is already queued and its model output does not depend on B.
		Losing B costs the analysis, which can be run by hand; raising here would
		strand a running ensemble with nothing recording that it exists.
		"""
		import shutil, subprocess
		if not shutil.which("sbatch"):
			print("   ⚠️  no sbatch — job B not submitted; analyse by hand")
			return None
		# REFUSE A NODE-LOCAL RUN DIRECTORY. A compute node cannot see this node's
		# /tmp: the first A/B attempt (770938/770939) put its scripts there and both
		# jobs died in two seconds with completely empty logs, which is
		# indistinguishable from a scheduler fault.
		#
		# It also makes this method safe to reach from a test. pytest's tmp_path is
		# under /tmp, and `sbatch` exists on a login node, so a test that got past
		# the mocked client would otherwise submit a real job into the queue.
		try:
			fs = subprocess.check_output(
				["df", "-P", str(self.run_dir)], text=True,
				timeout=20).splitlines()[-1].split()[0]
		except Exception:                                       # noqa: BLE001
			fs = ""
		if fs.startswith("/dev/"):
			print(f"   ⚠️  {self.run_dir} is on {fs}, a node-local filesystem — "
				  f"job B not submitted. Put the run directory on $PSCRATCH.")
			return None
		fw = Path(__file__).resolve().parents[3]
		sb = self.run_dir / "ensemble_B.sbatch"
		mail = (f"#SBATCH --mail-user={email}\n#SBATCH --mail-type=END,FAIL"
				if email else "")
		sb.write_text(f"""#!/bin/bash
#SBATCH -J elm_B
#SBATCH -N 1
#SBATCH -p {config.get("queue") or os.environ.get("IDEAS_SLURM_QUEUE", "short")}
#SBATCH -A {config.get("account") or os.environ.get("IDEAS_SLURM_ACCOUNT", "e3sm")}
#SBATCH -t {config.get("analysis_walltime", "00:30:00")}
#SBATCH -o {self.run_dir}/ensemble_B.log
{mail}
cd {fw}
# The analysis is DEFERRED while the Analyzer is redesigned; job B reports what
# the ensemble did and stops. Set IDEAS_RUN_ANALYSIS=1 to run the tail here.
if [ "${{IDEAS_RUN_ANALYSIS:-0}}" = "1" ]; then
  {sys.executable} workflow.py --resume {self.run_dir}
  {sys.executable} tools/notify_study.py {self.run_dir}
else
  {sys.executable} tools/notify_study.py {self.run_dir} --deferred
fi
exit $?
""")
		try:
			jid = subprocess.check_output(
				["sbatch", "--parsable", f"--dependency=afterany:{job_a}", str(sb)],
				text=True, timeout=120).strip().split(";")[0]
		except Exception as e:                                  # noqa: BLE001
			print(f"   ⚠️  job B not submitted ({e}) — A is running; analyse by hand")
			return None
		print(f"   job A {job_a} (build+run) → job B {jid} (afterany, reports)")
		return jid

	def _poll(self, record: Dict[str, Any], experiments, config):
		"""Has job A landed? The case directories if so, None if not.

		ONE SHAPE, because there is one job. build_cases is the only stage that
		returns a Pending now, so the dispatch on record['stage'] is gone with
		the split flow — and with it the way that dispatch went wrong: polling a
		build+run job down the `build_cases` branch handed back case dirs and let
		execute_plan walk into _run, which submitted the ensemble A HAD ALREADY
		RUN. Every column would have run twice, and the second pass would have
		overwritten the history files of the first.

		The case directories come back from check_elm_job itself. A stage that
		hands back a job id needs somewhere to hand back its ANSWER, and a few
		short strings can travel inline — unlike results, which is why reading
		those stayed on this side.
		"""
		client = self._mcp(config or {})
		if client is None:
			raise RuntimeError(
				f"job {record.get('job_id')} is recorded as pending but there "
				f"is no `elm` client to ask about it — register one in "
				f"mcp_config.json, or check it by hand with squeue")
		st = self._mcp_call(client, "check_elm_job",
							{"job_id": str(record.get("job_id")),
							 "run_dir": str(self.run_dir)})
		if st.get("active", True):
			print(f"   job {record.get('job_id')} is "
				  f"{st.get('state') or 'unanswered'} — nothing to collect yet")
			return None
		if not st.get("ok"):
			raise RuntimeError(
				f"the case build failed: {st.get('error')}"
				+ (f"\n{st.get('log_tail')}" if st.get("log_tail") else ""))
		by_name = {c.get("case_name"): c.get("case_dir")
				   for c in (st.get("cases") or [])}
		for e in experiments:
			if by_name.get(e.get("case_name")):
				e["case_dir"] = by_name[e["case_name"]]
		print(f"   ✓ {st.get('n_ok')}/{st.get('n_total')} case(s) built")
		return experiments

	def _build_case_inputs(self,
			   plan:   Dict[str, Any],
			   config: Dict[str, Any]) -> List[Dict]:
		"""Read what the MCP already wrote. Computes nothing.

		_refine_columns made the one call that produced case_inputs.json, so
		this stage exists only because the base's pipeline has a slot for it.
		The rows come back as PLAIN DATA with no live ELMAgent, and that is now
		the ONLY state they are ever in — not a resume-only special case. The
		--keepexe builder handle went with the local path; the job side does
		the cloning, on the compute node, out of case_inputs.json.
		"""
		rows = self._rehydrate_case_inputs()
		if not rows:
			raise RuntimeError(
				f"{self.input_dir / self.CASE_INPUTS} is missing or empty — "
				f"_refine_columns should have written it via the elm MCP")
		return rows

	# 02_setup_plots/column_surfaces.png — the soil each column ACTUALLY got —
	# is drawn by the MCP's build job (mcp/elm-mcp/scripts/ensemble_job.py). It
	# has to be: it reads each case's generated FSURDAT through its `run/lnd_in`,
	# which does not exist until the case is built, and the build happens on the
	# compute node. Drawing it from here was left behind by the move to jobs A+B
	# and produced nothing for two studies.

	BUILT_CASES = "built_cases.json"

	def adopt_completed_run(self) -> Dict[str, Any]:
		"""ELM also adopts the case build, and re-attaches where each landed.

		built_cases.json is the one thing the finalize cannot work out for
		itself: case directories carry a creation timestamp, so they cannot be
		derived from the case names. Without re-attaching them, _extract globs
		`{case_dir}/run/*.elm.h0.*.nc` against a case_dir of None and every
		column reports zero history files — a finished ensemble read as a
		total failure.
		"""
		experiments = self._rehydrate_case_inputs() or []
		src = self.input_dir / self.BUILT_CASES
		adopted: Dict[str, Any] = {}
		if src.is_file() and experiments:
			try:
				built = json.loads(src.read_text())
				by_name = {c.get("case_name"): c.get("case_dir")
						   for c in (built.get("cases") or [])}
				n = 0
				for e in experiments:
					if by_name.get(e.get("case_name")):
						e["case_dir"] = by_name[e["case_name"]]
						n += 1
				self._save_case_inputs(experiments)
				self._mark("build_cases", n_experiments=len(experiments),
						   n_ok=n, adopted_from=self.BUILT_CASES)
				adopted["build_cases"] = n
			except Exception as e:                              # noqa: BLE001
				print(f"   ⚠️  could not adopt {self.BUILT_CASES} ({e}) — "
					  f"the case directories will be missing")
		adopted.update(super().adopt_completed_run())
		return adopted

	# ─────────────────────────────────────────────────────────
	# STEP 3 — RUN (writes to 03_results/)
	# ─────────────────────────────────────────────────────────
	def _run(self,
			 experiments: List[Dict],
			 config:      Dict[str, Any]):
		"""COLLECT. Job A already ran the columns; this never submits anything.

		The stage is still here because the base's sequence has a slot for it
		and because everything it does after the columns stop — the run
		summaries, execution_report.txt, results_summary.csv — still has to
		happen. What it does NOT do is run ELM: by the time execute_plan reaches
		this line, either job A has just been polled (the case dirs came back
		from _poll) or the ledger says build_cases is done, and both mean the
		ensemble has already been through the queue.

		SUBMITTING HERE WAS A BUG, not a simplification removed. A --resume of a
		landed A+B study walked out of _poll into this stage and re-submitted
		every column, over the top of the history files the first pass wrote.
		Nothing had exercised it only because the analysis is deferred.

		Success is what is ON DISK: a column succeeded if ELM wrote it a history
		file. The scheduler being finished is what ends the wait; the files
		decide the outcome.
		"""
		return self._collect(experiments, self._outcomes_from_disk(experiments))

	def _collect(self,
				 experiments: List[Dict],
				 results:     Dict[str, bool]) -> Dict[str, bool]:
		"""Everything that happens once the columns have stopped running."""
		for exp in experiments:
			try:
				exp['run_summary'] = self._run_summary_for(exp)
			except Exception as e:                              # noqa: BLE001
				print(f"   ⚠️  run summary failed for {exp['case_name']}: {e}")

		n_ok   = sum(1 for v in results.values() if v)
		n_fail = len(results) - n_ok
		print(f"✓ {n_ok} succeeded, {n_fail} failed")

		# Post-run reporting → 03_results/
		self._write_execution_report(experiments, results)
		self._write_results_csv(experiments, results)

		return results

	@staticmethod
	def _run_summary_for(exp: Dict) -> Dict[str, Any]:
		"""The case's run summary. Off disk, always.

		There is no live agent to ask any more: the cases are built inside job
		A, and what comes back to this side is JSON. Asking the string repr of
		an ELMAgent for a summary is how this once reported zero history files
		for cases that had plenty — the branch that did it is gone rather than
		guarded, so there is nothing left to fall back FROM.
		"""
		cd = exp.get('case_dir')
		return {'history_files': sorted(glob.glob(f"{cd}/run/*.elm.h0.*.nc"))
								 if cd else []}

	def _outcomes_from_disk(self, experiments: List[Dict]) -> Dict[str, bool]:
		"""Which columns produced output — the one definition of success: a
		column succeeded if ELM wrote it a history file."""
		return {e['case_name']: bool(e.get('case_dir') and
									 glob.glob(f"{e['case_dir']}/run/*.elm.h0.*.nc"))
				for e in experiments}

	# ONE definition, in inputs.py, because two lists that must agree are two
	# lists that will not. A key added there and forgotten here would drop a
	# field from case_inputs.json silently — the file would exist, parse, and be
	# missing something the build needs.
	@property
	def CASE_KEYS(self):                                    # noqa: N802
		import inputs
		return inputs.CASE_KEYS

	def _serialise_case_inputs(self, experiments):
		"""Delegates to inputs.serialise_case_inputs — see there."""
		import inputs
		return inputs.serialise_case_inputs(experiments)

	def _rehydrate_handles(self, experiments, plan, config) -> None:
		"""Drop the ELMAgent string reprs a reloaded case_inputs.json carries.

		An agent is an object and the manifest is JSON, so what comes back is a
		string: truthy, with no methods, and therefore a thing that fails at the
		call site rather than here. Nothing on this side calls one any more —
		the build happens inside job A — so they are dropped and the key stays
		absent.
		"""
		stale = [e for e in experiments if isinstance(e.get('elm_agent'), str)]
		for e in stale:
			e.pop('elm_agent', None)
		if stale:
			print(f"   ↻ {len(stale)} case handle(s) are not JSON — working "
				  f"from the case directories on disk")

	def _write_execution_report(self,
								experiments: List[Dict],
								results:     Dict[str, bool]) -> None:
		"""Write a plain-text run report to 03_results/execution_report.txt."""
		report_file = self.results_dir / "execution_report.txt"

		n_total   = len(experiments)
		n_success = sum(results.values())

		lines = []
		lines.append("=" * 60)
		lines.append("ELM EXECUTION REPORT")
		lines.append("=" * 60)
		lines.append(f"Generated:         {datetime.now().isoformat()}")
		lines.append(f"Run directory:     {self.run_dir}")
		lines.append(f"Total experiments: {n_total}")
		lines.append(f"Successful:        {n_success}")
		lines.append(f"Failed:            {n_total - n_success}")
		lines.append("")

		for exp in experiments:
			case_name   = exp['case_name']
			status      = ('completed' if results.get(case_name)
						   else 'failed')
			run_summary = exp.get('run_summary', {}) or {}
			n_hist      = len(run_summary.get('history_files', []))

			lines.append(f"── {case_name}")
			lines.append(f"   status:           {status}")
			lines.append(f"   scenario:         {exp.get('scenario_name', '?')}")
			lines.append(f"   forcing_period:   {exp.get('forcing_period', '?')}")
			lines.append(f"   forcing_years:    "
						 f"{exp.get('forcing_start', '?')}-"
						 f"{exp.get('forcing_end', '?')}")
			lines.append(f"   soil_config:      {exp.get('soil_config', '?')}")
			if exp.get('substrate'):
				lines.append(f"   substrate:        {exp['substrate']}")
			lines.append(f"   case_dir:         {exp.get('case_dir', '?')}")
			lines.append(f"   history_files:    {n_hist}")
			lines.append("")

		report_file.write_text('\n'.join(lines))
		print(f"✓ execution_report.txt saved")

	def _write_results_csv(self,
						   experiments: List[Dict],
						   results:     Dict[str, bool]) -> None:
		"""Write a tabular summary to 03_results/results_summary.csv."""
		csv_file = self.results_dir / "results_summary.csv"
		fields = ['case_name', 'scenario_name', 'forcing_period',
				  'forcing_start', 'forcing_end', 'soil_config',
				  'substrate', 'status', 'history_file_count',
				  'case_dir']

		with open(csv_file, 'w', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=fields)
			writer.writeheader()
			for exp in experiments:
				case_name   = exp['case_name']
				run_summary = exp.get('run_summary', {}) or {}
				writer.writerow({
					'case_name':          case_name,
					'scenario_name':      exp.get('scenario_name', ''),
					'forcing_period':     exp.get('forcing_period', ''),
					'forcing_start':      exp.get('forcing_start', ''),
					'forcing_end':        exp.get('forcing_end', ''),
					'soil_config':        exp.get('soil_config', ''),
					'substrate':          exp.get('substrate', '') or '',
					'status':             ('completed'
										   if results.get(case_name)
										   else 'failed'),
					'history_file_count': len(run_summary.get(
						'history_files', [])),
					'case_dir':           exp.get('case_dir', ''),
				})
		print(f"✓ results_summary.csv saved")

	# ─────────────────────────────────────────────────────────
	# STEP 4 — ANALYZE (writes to 04_analysis/)
	# ─────────────────────────────────────────────────────────
	def _extract(self,
				 experiments,
				 plan:   Dict[str, Any] = None,
				 config: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Read the ELM history NetCDFs and pull the numbers out → 04_analysis/.

		This is extraction, not analysis: it knows ELM's output format and
		nothing about what the numbers mean. Figures, observation comparison
		and interpretation belong to the Analyzer (src/agents/analyzer.py) and
		are no longer reachable from here.

		Also attaches the honesty payload (structural + configuration
		limitations, assumptions ledger). ELMResultsAnalyzer already computes
		the confounding notes, fit r2 and soil attribution; only
		tools/analyze_run.py used to add `extra_summary`, so runs driven
		through this manager silently lost the caveats.
		"""
		analyzer = ELMResultsAnalyzer(
			experiments  = experiments,
			analysis_dir = str(self.analysis_dir),
		)

		cfg  = config or {}
		plan = plan or {}
		try:
			from core.limitations import select_limitations
			couplers = plan.get("CONDITIONS_COUPLERS") or [{}]
			y0 = int(couplers[0].get("DATM_CLMNCEP_YR_START", cfg.get("yr_start", 1995)))
			y1 = int(couplers[0].get("DATM_CLMNCEP_YR_END",   cfg.get("yr_end", y0)))
			analyzer.extra_summary = {
				"limitations": select_limitations(
					n_years       = max(1, y1 - y0 + 1),
					warm_start    = bool(couplers[0].get("FINIDAT")),
					forcing       = cfg.get("forcing", "nldas"),
					spinup_years  = int(cfg.get("spinup_years", 0)),
					# WHICH warm start, and whether the soil was kept with it.
					# Without these the caveat cannot tell an equilibrated,
					# self-consistent state from the mismatch that made year
					# one a relaxation — and it defaulted to warning about both.
					warm_source   = ((cfg.get("warm_start") or {}).get("source")
									 if isinstance(cfg.get("warm_start"), dict)
									 else cfg.get("warm_start")) or "conus",
					soil_source   = "conus",   # warm start is required; the
					# donor's surfdata is always what ELM runs on
				),
				"assumptions_ledger": (
					json.loads((self.run_dir / "assumptions.json").read_text())
					if (self.run_dir / "assumptions.json").exists() else []),
			}
		except Exception as e:
			print(f"   ⚠️  limitations payload unavailable ({e})")

		analyzer.extract_all()

		# The stage's output is DATA. ELMResultsAnalyzer stays as the thing
		# that COMPUTES the rows; it just no longer crosses the boundary.
		return self._as_extract(analyzer)

	# ─────────────────────────────────────────────────────────
	# STEP 4d — ONE-WAY ELM → PFLOTRAN COUPLING
	# ─────────────────────────────────────────────────────────
	def _couple_pflotran(self,
						 plan:   Dict[str, Any],
						 config: Dict[str, Any]) -> bool:
		"""Drive a 1-D PFLOTRAN column per ELM column with that column's own
		daily QINFL -> 05_pflotran/. Non-fatal.

		Runs when the planner emitted a coupling_design (the archetype the
		Reception and Planner prompts already speak), or when config forces it
		with pflotran={'run': True}. Both prompts have described this coupling
		for months; nothing executed it, so "one-way coupling" was a design the
		framework could plan but never perform.

		The columns are the SAME ones ELM just ran — same lat/lon, same SSURGO
		profile, same Fan water table — so the two models answer one question
		jointly: ELM partitions the surface water, PFLOTRAN carries it down.
		1-D Richards columns take seconds, so this runs serially in-process
		with no batch queue.
		"""
		cfg = (config or {}).get("pflotran") or {}
		coupling = plan.get("coupling_design") or {}
		archetype = ((plan.get("model_choice") or {}).get("design_archetype") or "")
		wanted = bool(cfg.get("run")) or bool(coupling) or archetype == "coupling"
		if not wanted:
			return False

		print("\n🪨 STEP 4d: Coupling ELM → PFLOTRAN")
		print("-" * 40)
		if coupling.get("driver"):
			print(f"   driver: {coupling['driver']}")

		# Checked before the try so a missing file reports what is missing,
		# not a bare errno from json.loads.
		cols_f = self.run_dir / "columns.json"
		cols = []
		if cols_f.exists():
			cols = json.loads(cols_f.read_text())
			cols = cols.get("columns", cols) if isinstance(cols, dict) else cols
		if not cols:
			print("   ⚠️  no columns.json in this run — skipping coupling")
			return False

		try:
			bp  = _load_tool("build_pflotran_cases")
			out = self.run_dir / "05_pflotran"

			res = bp.build_ensemble(
				cols, out,
				flux_from  = str(self.run_dir),      # this run's QINFL
				run        = cfg.get("run", True),
				timeout    = int(cfg.get("timeout", 1800)),
				spin_years = float(cfg.get("spin_years", 10.0)),
				depth_cap  = float(cfg.get("depth_cap", 50.0)),
				bottom     = cfg.get("bottom", "fan"),
			)
			cases = res.get("cases") or []
			if not cases:
				print("   ⚠️  no PFLOTRAN decks built (no ELM flux?) — skipping")
				return False

			n_ok = sum(1 for m in cases if m.get("run_ok"))
			print(f"✓ {n_ok}/{len(cases)} PFLOTRAN column(s) ran → 05_pflotran/")
			if not n_ok:
				return False

			ap = _load_tool("analyze_pflotran_coupled")
			ap.analyze_coupled(out, quiet=True)
			print("✓ lag + attenuation → 05_pflotran/pflotran_summary_coupled.json")
			return True
		except Exception as e:
			print(f"   ⚠️  coupling failed ({e}) — the ELM study stands")
			return False
