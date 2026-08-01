#!/usr/bin/env python3
"""
IDEAS workflow coordinator — the four agents of the framework, in order.

    User request
      → Reception          brief   (MCP tool loop: what/where/when)
      → Planner            plan    (sampling strategy + feasibility verdict)
      → Experiment Manager run     (materialize → build → prepare → run)
      → Analyzer           report  (metrics → validation → interpretation)

WHICH MODEL runs is `--model`, resolved through core/backends.py: `elm`,
`pflotran`, or `lambda-pflotran`. The choice is the Experiment Manager CLASS,
because the backends differ in the stages they have, not only in the code
inside them. All three write the same experiment.json and are read by the same
Analyzer.

(This file used to say PFLOTRAN was not driven from here. It is, since the
backend table landed — the standalone tools/build_pflotran_cases.py still
works and builds byte-identical decks.)
"""
import sys
import traceback
import json
from pathlib import Path
from typing  import Optional
sys.path.insert(0, "src")

class _NothingToReportOn(Exception):
	"""Not a failure: the run finished and produced nothing to interpret.

	Its own type so the skip reads as a skip. Folded into the generic handler
	it would print "written report failed", which describes a broken reporter
	rather than an empty ensemble.
	"""


from agents.planner               import Planner
from agents.analysis_report_agent import AnalysisReportAgent
from core.mcp_manager             import MCPManager

class WorkflowCoordinator:
	"""Wires Reception → Planner → Experiment Manager → Analyzer."""

	# WHICH MODEL, as a class attribute so it has a value even on an instance
	# built without __init__ — the test harness does exactly that, and so does
	# anything that reconstructs a coordinator to inspect it. __init__ replaces
	# it with the validated name.
	model = "elm"

	def __init__(self,
				 reception_model:      str = "claude-opus-4-8-project",
				 planner_model:        str = "claude-opus-4-8-project",
				 analyzer_model:       str = "claude-opus-4-8-project",
				 default_output_dir:   str = "./workflow_outputs",
				 mcp_config_file:      str = "mcp_config.json",
				 interactive_reception: bool = False,
				 model:                str = None):

		# WHICH MODEL THIS SESSION RUNS. Validated here, at construction,
		# rather than at _execute — that is minutes of reception and planning
		# later, and a typo'd name should not cost an LLM call to discover.
		from core import backends
		self.model = (model or backends.DEFAULT).strip().lower()
		backends.get(self.model)          # raises on an unknown name

		# ── MCP Manager ───────────────────────────────────────
		print("\n" + "=" * 70)
		print("Initializing MCP Tools")
		print("=" * 70)
		try:
			self.mcp_manager = MCPManager(mcp_config_file)
			mcp_clients      = self.mcp_manager.get_all_clients()
			if mcp_clients:
				print(f"✓ Loaded {len(mcp_clients)} MCP server(s)")
				for name in mcp_clients:
					print(f"  - {name}")
				# The servers' own INFO logging would otherwise interleave with
				# the reception trace between every prompt. Say where it went.
				from core.mcp_client import mcp_errlog
				_log = getattr(mcp_errlog(), "name", "<stderr>")
				if _log != "<stderr>":
					print(f"  server logs \u2192 {_log}  "
					      f"(IDEAS_MCP_LOG=stderr to show them here)")
			else:
				print("⚠️  No MCP servers configured")
		except Exception as e:
			print(f"⚠️  MCP initialization failed: {e}")
			mcp_clients = {}
		print("=" * 70 + "\n")
	
		# Keep the live clients: the ELM manager's materialize stage needs
		# terrain/fan_wtd/geology to turn a sampling strategy into columns.
		self.mcp_clients = mcp_clients

		# ── Agents ────────────────────────────────────────────
		# Reception is an LLM-driven tool loop over all MCP servers. It emits
		# `domain: {name, huc, bbox}`, which the Experiment Manager's
		# materialize stage needs to turn a strategy into columns.
		from agents.reception_llm import LLMReceptionAgent
		self.reception = LLMReceptionAgent(
			model       = reception_model,
			mcp_clients = mcp_clients,
			interactive = interactive_reception,
		)
		self.planner  = Planner(model=planner_model)
		self.analyzer = AnalysisReportAgent(model=analyzer_model, model_type="elm")

		# ── Defaults ──────────────────────────────────────────
		self.default_output_dir   = default_output_dir
	
		# ── Conversation state ────────────────────────────────
		self.conversation_context = {
			'last_run_dir':  None,
			'last_plan':     None,
			'last_analysis': None,
			'last_focus':    None,
		}
	
	# ═════════════════════════════════════════════════════════
	# MAIN ENTRY POINT
	# ═════════════════════════════════════════════════════════
	def _reception_context(self) -> Optional[dict]:
		"""What reception should know about earlier turns — and nothing more.

		The context is JSON-dumped verbatim into reception's prompt, so this
		is a token budget, not just a convenience. conversation_context holds
		last_analysis and last_plan in FULL; sending those would put an entire
		analysis report and strategy into every subsequent request.

		What reception actually needs is enough to resolve a follow-up: which
		run to look at for "analyze that again", and what was being studied so
		"the same basin" resolves. Returns None on the first turn so no
		CONVERSATION CONTEXT block is emitted at all.
		"""
		cc = self.conversation_context or {}
		ctx = {}
		if cc.get('last_run_dir'):
			ctx['prior_run_dir'] = str(cc['last_run_dir'])
		if cc.get('last_focus'):
			ctx['prior_focus'] = cc['last_focus']
		plan = cc.get('last_plan') or {}
		if isinstance(plan, dict):
			samp = plan.get('sampling') or {}
			if samp.get('n_columns'):
				ctx['prior_n_columns'] = samp['n_columns']
			if plan.get('archetype'):
				ctx['prior_archetype'] = plan['archetype']
		return ctx or None

	def process_request(self,
						user_request: str,
						output_dir:   Optional[str] = None
						) -> str:
		print("\n" + "=" * 70)
		print("COORDINATOR")
		print("=" * 70)
		print(f"Request: {user_request[:80]}...")
		print("=" * 70 + "\n")
	
		# Step 1 — Reception
		result = self.reception.process(
			user_request = user_request,
			context      = self._reception_context(),
		)
		# Reception returns the whole package now: route (dispatch), brief
		# (science), observations + grid (what was fetched), provenance.
		# The adapter that used to flatten this into a dataclass is gone — it
		# forwarded only `brief`, which is how observations, grid and
		# provenance were being dropped before anything downstream saw them.
		result["user_request"] = user_request
		action = (result.get("route") or {}).get("action", "clarify")
		print(f"🧠 Route: {action}\n")

		# Step 2 — Route
		if action == 'clarify':
			return self._workflow_clarification(result)
		elif action == 'analyze_existing':
			return self._workflow_analyze_existing(result)
		elif action == 'design':
			return self._workflow_design_and_run(
				result     = result,
				output_dir = output_dir or self.default_output_dir,
			)
		else:
			return f"❌ Unknown route: {action}"
	
	# ═════════════════════════════════════════════════════════
	# INTERACTIVE MODE
	# ═════════════════════════════════════════════════════════
	def run_interactive(self):
		print("\n" + "=" * 70)
		print("COORDINATOR - INTERACTIVE MODE")
		print("=" * 70)
		print("\nCommands:")
		print("  - Type your request naturally")
		print("  - 'quit' or 'exit' → stop")
		print("  - 'status'         → show context")
		print("  - 'clear'          → clear history")
		print("=" * 70 + "\n")
	
		while True:
			try:
				user_input = input("You: ").strip()
				if not user_input:
					continue
				if user_input.lower() in ['quit', 'exit', 'q']:
					print("\n👋 Goodbye!\n")
					break
				if user_input.lower() == 'status':
					self._print_status()
					continue
				if user_input.lower() == 'clear':
					self._clear_context()
					print("✓ Conversation history cleared\n")
					continue
				response = self.process_request(user_input)
				print(f"\n{response}\n")
			except KeyboardInterrupt:
				print("\n\n👋 Interrupted. Goodbye!\n")
				break
			except Exception as e:
				print(f"\n❌ Error: {e}\n")
				traceback.print_exc()
	
	# ═════════════════════════════════════════════════════════
	# WORKFLOW 1 — CLARIFICATION
	# ═════════════════════════════════════════════════════════
	def _workflow_clarification(self, result) -> str:
		print("💬 WORKFLOW: Clarification Needed\n")
		lines = ["I need some clarification:\n"]
		for i, q in enumerate((result.get('route') or {}).get('questions') or [], 1):
			lines.append(f"{i}. {q}")
		return "\n".join(lines)
	
	# ═════════════════════════════════════════════════════════
	# WORKFLOW 2 — ANALYZE EXISTING
	# ═════════════════════════════════════════════════════════
	def _workflow_analyze_existing(self, result) -> str:
		print("📊 WORKFLOW: Analyze Existing Results\n")
		run_dir = (
			(result.get('route') or {}).get('prior_run_dir') or
			self.conversation_context.get('last_run_dir')
		)
		if not run_dir or not Path(run_dir).exists():
			return ("❌ Please specify a valid run directory or "
					"ensure a previous run exists.")
	
		print(f"📂 Analyzing: {run_dir}\n")
		llm_input_file = Path(run_dir) / "LLM_ANALYSIS_INPUT.json"
		if not llm_input_file.exists():
			return f"❌ Analysis input not found in {run_dir}"
	
		try:
			analysis = self.analyzer.generate_analysis_report(
				user_request   = result.get('user_request', ''),
				llm_input_file = str(llm_input_file),
				output_file    = str(
					Path(run_dir) / "ANALYSIS_REPORT.json"
				),
			)
			self.conversation_context['last_analysis'] = analysis
			self.conversation_context['last_run_dir']  = run_dir
			return self._format_analysis_response(analysis)
		except Exception as e:
			return f"❌ Analysis failed: {e}"
	
	# ═════════════════════════════════════════════════════════
	# WORKFLOW 3 — DESIGN & RUN
	# ═════════════════════════════════════════════════════════
	def _workflow_design_and_run(self, result, output_dir: str) -> str:
		print("🚀 WORKFLOW: Design & Run\n")
		try:
			# The COORDINATOR owns the run directory, and each stage's file is
			# written when that stage finishes. Reception and the planner then
			# only ever produce, and the Experiment Manager only ever reads —
			# which also means a failure downstream leaves both intact.
			from datetime import datetime as _dt
			# NAMED FOR THE MODEL THAT RAN. Every run directory used to be
			# elm_run_*, which was accurate while ELM was the only backend and
			# becomes a mislabel the moment it is not — the directory name is
			# the first thing anyone reads, and archived PFLOTRAN studies would
			# all claim to be ELM.
			run_dir = Path(output_dir) / f"{self.model}_run_{_dt.now():%Y%m%d_%H%M%S}"
			run_dir.mkdir(parents=True, exist_ok=True)
			(run_dir / "reception.json").write_text(
				json.dumps({k: v for k, v in result.items()
							if k not in ("trace", "raw")}, indent=2, default=str))
			# alias for the standalone tools that open it by this name
			(run_dir / "reception_brief.json").write_text(
				json.dumps(result.get("brief") or {}, indent=2, default=str))

			# Step 1 — Plan
			print("📋 STEP 1: Planning Experiments")
			print("-" * 50)
			plan = self.planner.plan(result)
			(run_dir / "strategy.json").write_text(
				json.dumps(plan, indent=2, default=str))
			# The capability-aware planner emits a STRATEGY, never
			# CONDITIONS_COUPLERS — those are materialized against real data in
			# the Experiment Manager's step 0. Counting them here printed
			# "0 experiments" on every successful plan, which reads as a failure.
			strat = plan.get('sampling') or {}
			n_req = strat.get('n_columns')
			verdict = (plan.get('feasibility') or {}).get('verdict', '?')
			print(f"✓ Strategy: {strat.get('approach', 'n/a')}, "
				  f"N={n_req if n_req is not None else 'expander decides'}, "
				  f"feasibility={verdict}")
			print("  (columns are materialized from real data in step 0)\n")
	
			self.conversation_context['last_plan']  = plan
			self.conversation_context['last_focus'] = (
				((result.get('brief') or {}).get('scientific_framing') or {}).get('goals', [None])[0]
			)
	
			# Step 2 — Execute
			print("⚙️  STEP 2: Executing Experiments")
			print("-" * 50)
			run_summary = self._execute(
				plan         = plan,
				output_dir   = output_dir,
				run_dir      = run_dir,
				reception    = result,
				brief        = result.get('brief') or {},
				period       = ((result.get('brief') or {}).get('run_settings') or {}).get('resolved_period'),
				initialization = ((result.get('brief') or {}).get('run_settings') or {}).get('initialization'),
			)
			print(f"✓ Execution: "
				  f"{run_summary['experiments_success']}/"
				  f"{run_summary['experiments_total']} "
				  f"succeeded\n")
	
			self.conversation_context['last_run_dir'] = (
				run_summary['run_directory']
			)
	
			# Step 3 — Analyze
			print("📊 STEP 3: Analyzing Results")
			print("-" * 50)
			llm_input_file = (
				Path(run_summary['run_directory']) /
				"LLM_ANALYSIS_INPUT.json"
			)
			# NON-FATAL. The ensemble is computed and experiment.json is
			# written by the time we get here, so a report failure — a dead
			# gateway, a missing alias file — must not be reported as a failed
			# pipeline. It reads as though the run was lost, and it was not.
			analysis = None
			try:
				# Nothing succeeded — same reasoning as the Analyzer skip in
				# execute_plan. A written report over zero columns costs an LLM
				# call to say it has no data, which RUN_SUMMARY.json already
				# says for free.
				if not run_summary.get('experiments_success'):
					raise _NothingToReportOn(
						f"0/{run_summary['experiments_total']} columns produced "
						f"output")
				if not llm_input_file.exists():
					raise FileNotFoundError(
						f"{llm_input_file.name} was not written; "
						f"experiment.json holds the results")
				analysis = self.analyzer.generate_analysis_report(
					user_request    = result.get('user_request', ''),
					experiment_plan = plan,
					llm_input_file  = str(llm_input_file),
					output_file     = str(
						Path(run_summary['run_directory']) /
						"ANALYSIS_REPORT.json"
					),
				)
				print("✓ Analysis complete\n")
				self.conversation_context['last_analysis'] = analysis
			except _NothingToReportOn as e:
				print(f"   ⏭  report skipped — {e}\n")
			except Exception as e:                              # noqa: BLE001
				print(f"   ⚠️  written report failed ({e}) — the run and its "
					  f"results are intact\n")
	
			if analysis is None:
				return (f"✅ Run complete: {run_summary['experiments_success']}"
						f"/{run_summary['experiments_total']} columns → "
						f"{run_summary['run_directory']}\n"
						f"   (the written report was not produced; "
						f"experiment.json holds the results)")
			return self._format_full_pipeline_response(
				run_summary, analysis
			)
	
		except Exception as e:
			return (f"❌ Pipeline failed: {e}\n\n"
					f"{traceback.format_exc()}")
	
	# ═════════════════════════════════════════════════════════
	# RESUME — continue a run that was interrupted or submitted
	# ═════════════════════════════════════════════════════════
	def resume_run(self, run_dir: str) -> str:
		"""Re-enter an existing run directory and carry on from its ledger.

		Everything needed is already ON DISK, written by the stage that
		produced it: run_state.json says what is done and which model ran,
		reception.json and strategy.json are the inputs, run_plan.json is the
		materialized plan. Nothing is reconstructed from conversation, which is
		what makes this work across sessions — and across machines.

		THE MODEL COMES FROM THE LEDGER, not from --model. Resuming an ELM run
		with the default backend would build PFLOTRAN decks in an ELM
		directory; the run itself is the authority on what it is.
		"""
		from core import backends
		# _read_json rather than json.loads: these files were written by an
		# earlier session and a half-written one must degrade to "no reception
		# package", not take the resume down.
		from core.resumable import inspect_run, describe, _read_json

		rd = Path(run_dir)
		if not rd.is_dir():
			return f"❌ No such run directory: {rd}"

		rec = inspect_run(rd, check_jobs=True)
		print("\n" + "=" * 70)
		print("RESUMING")
		print("=" * 70)
		print(describe(rec))

		if not rec["resumable"]:
			return (f"❌ {rd.name} cannot be resumed: {rec['why']}\n"
					f"   Run `python workflow.py --resume` to see what can.")

		model = rec.get("model") or self.model
		try:
			backends.get(model)
		except Exception as e:                                  # noqa: BLE001
			return (f"❌ The ledger says this run used model '{model}', which "
					f"this build does not have ({e})")
		if model != self.model:
			print(f"   model: {model} (from the ledger, not --model)")
			self.model = model

		reception = _read_json(rd / "reception.json") or {}
		strategy  = _read_json(rd / "strategy.json")  or {}
		brief     = reception.get("brief") or {}
		settings  = brief.get("run_settings") or {}

		try:
			summary = self._execute(
				plan           = strategy,
				output_dir     = str(rd.parent),
				run_dir        = rd,
				reception      = reception,
				brief          = brief,
				period         = settings.get("resolved_period"),
				initialization = settings.get("initialization"),
				resume         = True,
			)
		except Exception as e:                                  # noqa: BLE001
			return f"❌ Resume failed: {e}\n\n{traceback.format_exc()}"

		self.conversation_context['last_run_dir'] = summary['run_directory']

		# Still queued is a NORMAL outcome, not a failure — that is the whole
		# point of the pending state, and it must not read as a broken run.
		if summary.get("status") == "pending":
			return (f"⏳ Job {summary.get('job_id')} is still running — "
					f"{summary['experiments_pending']} column(s) queued.\n"
					f"   Check again: {summary['resume_command']}")
		return (f"✅ Resumed: {summary['experiments_success']}"
				f"/{summary['experiments_total']} columns → "
				f"{summary['run_directory']}")

	def _execute(self,
				 plan:           dict,
				 output_dir:     str,
				 run_dir:        Path,
				 reception:      dict,
				 brief:          dict = None,
				 period:         dict = None,
				 initialization: dict = None,
				 resume:         bool = False) -> dict:
		"""Hand the plan to the Experiment Manager.

		run_dir and reception are PASSED IN, not rediscovered. The coordinator
		created the directory and holds the reception package; reaching for
		either as a free variable is how this method came to reference two
		names that only existed in its caller.
		"""
		# WHICH MODEL, resolved through the one table every caller shares. The
		# class is the choice: the backends differ in the STAGES they have, and
		# the base's execute_plan reads those declarations off the class.
		from core import backends
		Manager = backends.get(self.model)
		executor = Manager(base_output_dir=output_dir, run_dir=str(run_dir))
		# brief + mcp_clients feed the manager's materialize stage, which turns
		# the planner's sampling_strategy into the backend's own run plan.
		cfg = backends.config_for(
			self.model,
			{
				'brief':       brief or {},
				'reception':   reception,
				'strategy':    plan,
				'mcp_clients': self.mcp_clients,
				# Warm start edits a completed run's restart files, so the
				# carrier is whatever this session ran last. That makes "now
				# warm-start it" work as a plain follow-up, with no paths for
				# the user to supply.
				'last_run_dir': self.conversation_context.get('last_run_dir'),
			},
			period=period,
			initialization=initialization)
		# Opt-in, and only ever set by resume_run. A fresh run must not inherit
		# it: re-entering a directory usually means "do it again", and guessing
		# wrong that way silently reuses stale compute.
		if resume:
			cfg['resume'] = True
		return executor.execute_plan(plan, cfg)
	
	# ═════════════════════════════════════════════════════════
	# UTILITIES
	# ═════════════════════════════════════════════════════════
	def _print_status(self):
		print("\n" + "-" * 70)
		print("CONVERSATION STATUS")
		print("-" * 70)
		print(f"Last Run:     "
			  f"{self.conversation_context.get('last_run_dir', 'None')}")
		print(f"Last Focus:   "
			  f"{self.conversation_context.get('last_focus', 'None')}")
		print(f"Has Plan:     "
			  f"{'Yes' if self.conversation_context.get('last_plan') else 'No'}")
		print(f"Has Analysis: "
			  f"{'Yes' if self.conversation_context.get('last_analysis') else 'No'}")
		print("-" * 70 + "\n")
	
	def _clear_context(self):
		self.conversation_context = {
			'last_run_dir':  None,
			'last_plan':     None,
			'last_analysis': None,
			'last_focus':    None,
		}
	
	def _format_analysis_response(self, analysis: dict) -> str:
		lines = ["=" * 70, "ANALYSIS RESULTS", "=" * 70, ""]
		lines += ["📌 ANSWER:", "-" * 70,
				  analysis.get('answer_to_user_question', 'N/A'),
				  ""]
		findings = analysis.get('key_findings', [])[:3]
		if findings:
			lines += ["🔍 KEY FINDINGS:", "-" * 70]
			for i, f in enumerate(findings, 1):
				lines.append(f"{i}. {f.get('finding', 'N/A')}")
			lines.append("")
		lines += [
			"=" * 70,
			f"Full report: "
			f"{self.conversation_context.get('last_run_dir')}/"
			f"ANALYSIS_REPORT.txt",
			"=" * 70,
		]
		return "\n".join(lines)
	
	def _format_full_pipeline_response(self,
										run_summary: dict,
										analysis:    dict) -> str:
		lines = ["=" * 70, "WORKFLOW COMPLETE", "=" * 70, ""]
		lines += ["📌 ANSWER:", "-" * 70,
				  analysis.get('answer_to_user_question', 'N/A'),
				  ""]
		lines += ["⚙️  EXECUTION:", "-" * 70,
				  f"• Experiments: "
				  f"{run_summary['experiments_success']}/"
				  f"{run_summary['experiments_total']} succeeded",
				  f"• Runtime:     "
				  f"{run_summary['total_runtime_seconds']:.1f}s",
				  f"• Output:      "
				  f"{run_summary['run_directory']}", ""]
		findings = analysis.get('key_findings', [])[:3]
		if findings:
			lines += ["🔍 KEY FINDINGS:", "-" * 70]
			for i, f in enumerate(findings, 1):
				lines.append(f"{i}. {f.get('finding', 'N/A')}")
			lines.append("")
		recs = analysis.get('recommendations', [])
		if recs:
			lines += ["💡 RECOMMENDATIONS:", "-" * 70]
			for rec in recs[:2]:
				lines.append(f"  • {rec}")
			lines.append("")
		lines += ["=" * 70,
				  f"Full details: {run_summary['run_directory']}",
				  "=" * 70]
		return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="IDEAS workflow: Reception → Planner → "
                    "Experiment Manager → Analyzer"
    )
    parser.add_argument(
        '--interactive', '-i',
        action = 'store_true',
        help   = 'Run in interactive mode'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default = './workflow_outputs',
        help    = 'Output directory'
    )
    parser.add_argument(
        '--mcp-config', '-m',
        default = 'mcp_config.json',
        help    = 'MCP configuration file'
    )
    from core import backends
    parser.add_argument(
        '--model',
        choices = backends.names(),
        default = backends.DEFAULT,
        help    = f'which model to run (default: {backends.DEFAULT}). '
                  f'elm: land-surface columns, warm-started from the CONUS '
                  f'restarts. pflotran: standalone 1-D subsurface flow over '
                  f'20 y, initialised at the Fan 2013 water table. '
                  f'lambda-pflotran: the same flow plus the LAMBDA '
                  f'organic-matter reaction sandbox, capped at 1 y (the timestep '
                  f'collapses above 1 y on sampled columns) — a demonstration of '
                  f'reactive transport, not a calibration.'
    )
    parser.add_argument(
        '--resume',
        nargs   = '?',
        const   = '',            # bare --resume: list what can be resumed
        default = None,          # absent: not a resume at all
        metavar = 'RUN_DIR',
        help    = 'continue an interrupted or submitted run. With a run '
                  'directory, resumes that one; on its own, lists the runs '
                  'that can be resumed and why, and asks which.'
    )
    parser.add_argument(
        '--no-ask',
        action = 'store_true',
        help   = 'do NOT let reception ask clarifying questions — it resolves '
                 'the period and other gaps itself and records them as '
                 'conflicts (the old default)'
    )
    parser.add_argument(
        '--ask', '-a',
        action = 'store_true',
        help   = argparse.SUPPRESS       # implied by --interactive; kept so
                                         # existing commands keep working
    )
    args = parser.parse_args()

    # ── --resume, before anything expensive ──────────────────────────
    # Listing is a filesystem scan and needs no LLM, no MCP servers and no
    # reception. Building the coordinator first would spend all three to print
    # a directory listing — and would fail outright if the gateway were down,
    # which is precisely a moment someone might be trying to recover a run.
    if args.resume is not None:
        from core.resumable import find_resumable, print_listing
        target = args.resume
        if not target:
            rows = find_resumable(args.output_dir, check_jobs=True,
                                  include_all=True)
            print_listing(rows, args.output_dir)
            live = [r for r in rows if r.get("resumable")]
            if not live:
                return
            if sys.stdin.isatty():
                # THE CHOICE IS THE USER'S, including when there is only one.
                # Auto-resuming a lone candidate looks helpful and is not: a
                # bare --resume is often someone asking what is outstanding,
                # and answering it by starting work is not what was asked.
                # Enter means "I was only looking".
                try:
                    pick = input(f"Which one? [1-{len(live)}, or Enter to "
                                 f"quit] ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return
                if not pick.isdigit() or not 1 <= int(pick) <= len(live):
                    print("Nothing resumed.")
                    return
                target = live[int(pick) - 1]["run_dir"]
            else:
                # No TTY: print the commands rather than guessing.
                print("Pick one and pass it explicitly:")
                for r in live:
                    print(f"    python workflow.py --resume {r['run_dir']}")
                return

        coordinator = WorkflowCoordinator(
            default_output_dir    = args.output_dir,
            mcp_config_file       = args.mcp_config,
            interactive_reception = False,
            model                 = args.model,
        )
        print(coordinator.resume_run(target))
        return

    # --interactive means the user is sitting at a TTY answering prompts, so
    # reception may ask them things. Requiring a second --ask flag to enable
    # that made "interactive" mode silently non-interactive: it resolved the
    # simulation period on its own and went straight to the planner, which is
    # the one decision a user most wants to make.
    coordinator = WorkflowCoordinator(
        default_output_dir    = args.output_dir,
        mcp_config_file       = args.mcp_config,
        interactive_reception = (args.interactive or args.ask) and not args.no_ask,
        model                 = args.model,
    )

    if args.interactive:
        coordinator.run_interactive()
    else:
        print("=" * 70)
        print("Run with --interactive for interactive mode")
        print("\nExamples:")
        print("  python workflow.py --interactive")
        print("  python workflow.py --interactive --model pflotran")
        print("  python workflow.py --interactive --ask")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()