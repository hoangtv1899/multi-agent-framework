#!/usr/bin/env python3
"""
IDEAS workflow coordinator — the four agents of the framework, in order.

    User request
      → Reception          brief   (MCP tool loop: what/where/when)
      → Planner            plan    (sampling strategy + feasibility verdict)
      → Experiment Manager run     (materialize → build → prepare → run)
      → Analyzer           report  (metrics → validation → interpretation)

PFLOTRAN is not driven from here. The legacy in-process PFLOTRAN manager was
removed; reactive-transport runs go through tools/build_pflotran_cases.py and
are documented in docs/PFLOTRAN_PLAN.md.
"""
import sys
import traceback
from pathlib import Path
from typing  import Optional
sys.path.insert(0, "src")
from agents.planner               import Planner
from agents.analysis_report_agent import AnalysisReportAgent
from core.mcp_manager             import MCPManager

class WorkflowCoordinator:
	"""Wires Reception → Planner → Experiment Manager → Analyzer."""

	def __init__(self,
				 reception_model:      str = "claude-opus-4-8-project",
				 planner_model:        str = "claude-opus-4-8-project",
				 analyzer_model:       str = "claude-opus-4-8-project",
				 default_output_dir:   str = "./workflow_outputs",
				 mcp_config_file:      str = "mcp_config.json",
				 interactive_reception: bool = False):
	
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
			user_request         = user_request,
			conversation_context = self.conversation_context,
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
			# Step 1 — Plan
			print("📋 STEP 1: Planning Experiments")
			print("-" * 50)
			plan  = self.planner.create_plan(
				brief = result.get('brief') or {}
			)
			# The capability-aware planner emits a STRATEGY, never
			# CONDITIONS_COUPLERS — those are materialized against real data in
			# the Experiment Manager's step 0. Counting them here printed
			# "0 experiments" on every successful plan, which reads as a failure.
			strat = plan.get('sampling_strategy') or {}
			n_req = strat.get('n_exploratory')
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
	
			return self._format_full_pipeline_response(
				run_summary, analysis
			)
	
		except Exception as e:
			return (f"❌ Pipeline failed: {e}\n\n"
					f"{traceback.format_exc()}")
	
	def _execute(self,
				 plan:           dict,
				 output_dir:     str,
				 brief:          dict = None,
				 period:         dict = None,
				 initialization: dict = None) -> dict:
		"""Hand the plan to the Experiment Manager."""
		from core.elm_exp_manager import ELMExpManager
		executor = ELMExpManager(base_output_dir=output_dir)
		# brief + mcp_clients feed the manager's materialize stage, which
		# turns the planner's sampling_strategy into CONDITIONS_COUPLERS.
		cfg = {
			'brief':       brief or {},
			'reception':   result,
			'mcp_clients': self.mcp_clients,
			# Warm start edits a completed run's restart files, so the carrier
			# is whatever this session ran last. That makes "now warm-start it"
			# work as a plain follow-up, with no paths for the user to supply.
			'last_run_dir': self.conversation_context.get('last_run_dir'),
		}
		# WARM IS THE DEFAULT. A cold single-column year starts from ELM's
		# generic state and spends the run relaxing out of it -- measured on
		# this framework, recharge came out -0.18 mm/yr cold against 309 warm
		# on the SAME column. Subsetting the CONUS restart costs ~2 s per
		# column, so there is no reason to pay that price by default. Cold is
		# now an explicit opt-out, not what you get by saying nothing.
		if (initialization or {}).get('mode') != 'cold':
			cfg['warm_start'] = {
				'source': ((initialization or {}).get('source') or 'conus'),
			}
		# Honour the period reception resolved, instead of silently
		# defaulting to 1995 inside the manager.
		if period:
			if period.get('yr_start'):
				cfg['yr_start'] = int(period['yr_start'])
			cfg['yr_end'] = int(period.get('yr_end')
								or period.get('yr_start') or 1995)
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

    # --interactive means the user is sitting at a TTY answering prompts, so
    # reception may ask them things. Requiring a second --ask flag to enable
    # that made "interactive" mode silently non-interactive: it resolved the
    # simulation period on its own and went straight to the planner, which is
    # the one decision a user most wants to make.
    coordinator = WorkflowCoordinator(
        default_output_dir    = args.output_dir,
        mcp_config_file       = args.mcp_config,
        interactive_reception = (args.interactive or args.ask) and not args.no_ask,
    )

    if args.interactive:
        coordinator.run_interactive()
    else:
        print("=" * 70)
        print("Run with --interactive for interactive mode")
        print("\nExamples:")
        print("  python workflow.py --interactive")
        print("  python workflow.py --interactive --ask")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()