# workflow.py — full updated file
# Changes marked with ← NEW

#!/usr/bin/env python3
"""
PFLOTRAN/ELM Coordinator
Orchestrates: Reception → Planner → Execute → Analyze
"""
import sys
import traceback
from pathlib import Path
from typing  import Optional
sys.path.insert(0, "src")
from agents.reception_agent       import ReceptionAgent
from agents.planner_agent         import PlannerAgent
from agents.analysis_report_agent import AnalysisReportAgent
from core.exp_manager             import ExpManager
from core.mcp_manager             import MCPManager

class PFLOTRANCoordinator:
	"""
	Coordinates PFLOTRAN and ELM workflows.
	Pipeline:
	User Request
	→ ReceptionAgent.process()   (MCP + intent + brief)
	→ PlannerAgent.create_plan() (design + validate)
	→ ExpManager.execute_plan()  (build + run)        ← PFLOTRAN
	  OR ELMExpManager.execute_plan()                  ← ELM
	→ AnalysisReportAgent        (interpret results)
	"""
	
	def __init__(self,
				 reception_model:      str = "gpt-5.5-project",
				 planner_model:        str = "claude-opus-4-8-project",
				 analyzer_model:       str = "claude-opus-4-8-project",
				 default_pflotran_exe: str = "pflotran",
				 default_output_dir:   str = "./workflow_outputs",
				 mcp_config_file:      str = "mcp_config.json",
				 model_type:           str = "pflotran"):  # ← NEW
	
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
			else:
				print("⚠️  No MCP servers configured")
		except Exception as e:
			print(f"⚠️  MCP initialization failed: {e}")
			mcp_clients = {}
		print("=" * 70 + "\n")
	
		# ── Model type ────────────────────────────────────────
		self.model_type = model_type.lower()  # ← NEW
		print(f"🔧 Model type: {self.model_type.upper()}\n")
	
		# ── Agents ────────────────────────────────────────────
		self.reception = ReceptionAgent(
			model       = reception_model,
			mcp_clients = mcp_clients,
			model_type  = self.model_type,
		)
		self.planner = PlannerAgent(
			model      = planner_model,
			model_type = self.model_type,   # ← NEW
		)
		self.analyzer = AnalysisReportAgent(
			model      = analyzer_model,
			model_type = self.model_type,   # ← NEW
		)
	
		# ── Defaults ──────────────────────────────────────────
		self.default_pflotran_exe = default_pflotran_exe
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
						pflotran_exe: Optional[str] = None,
						output_dir:   Optional[str] = None
						) -> str:
		print("\n" + "=" * 70)
		print("COORDINATOR")
		print("=" * 70)
		print(f"Model  : {self.model_type.upper()}")
		print(f"Request: {user_request[:80]}...")
		print("=" * 70 + "\n")
	
		# Step 1 — Reception
		result = self.reception.process(
			user_request         = user_request,
			conversation_context = self.conversation_context,
		)
		print(f"🧠 Intent: {result.intent} ({result.confidence})\n")
	
		# Step 2 — Route by intent
		if result.intent == 'clarification_needed':
			return self._workflow_clarification(result)
		elif result.intent == 'analyze_existing':
			return self._workflow_analyze_existing(result)
		elif result.intent == 'design_and_run':
			return self._workflow_design_and_run(
				result       = result,
				pflotran_exe = pflotran_exe or self.default_pflotran_exe,
				output_dir   = output_dir   or self.default_output_dir,
			)
		else:
			return f"❌ Unknown intent: {result.intent}"
	
	# ═════════════════════════════════════════════════════════
	# INTERACTIVE MODE
	# ═════════════════════════════════════════════════════════
	def run_interactive(self):
		print("\n" + "=" * 70)
		print(f"COORDINATOR - INTERACTIVE MODE "
			  f"[{self.model_type.upper()}]")
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
		for i, q in enumerate(result.clarification_questions, 1):
			lines.append(f"{i}. {q}")
		return "\n".join(lines)
	
	# ═════════════════════════════════════════════════════════
	# WORKFLOW 2 — ANALYZE EXISTING
	# ═════════════════════════════════════════════════════════
	def _workflow_analyze_existing(self, result) -> str:
		print("📊 WORKFLOW: Analyze Existing Results\n")
		run_dir = (
			result.parameters.get('existing_run_dir') or
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
				user_request   = result.user_request,
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
	def _workflow_design_and_run(self,
								  result,
								  pflotran_exe: str,
								  output_dir:   str) -> str:
		print("🚀 WORKFLOW: Design & Run\n")
		try:
			# Step 1 — Plan
			print("📋 STEP 1: Planning Experiments")
			print("-" * 50)
			plan  = self.planner.create_plan(
				brief = result.to_planner_brief()
			)
			n_exp = len(plan.get('CONDITIONS_COUPLERS', []))
			print(f"✓ Plan created: {n_exp} experiments\n")
	
			self.conversation_context['last_plan']  = plan
			self.conversation_context['last_focus'] = (
				result.parameters.get('experiment_focus')
			)
	
			# Step 2 — Execute (route by model_type)
			print("⚙️  STEP 2: Executing Experiments")
			print("-" * 50)
			run_summary = self._execute(
				plan         = plan,
				pflotran_exe = pflotran_exe,
				output_dir   = output_dir,
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
				user_request    = result.user_request,
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
				 plan:         dict,
				 pflotran_exe: str,
				 output_dir:   str) -> dict:
		"""
		Route execution to correct model manager.
		PFLOTRAN → ExpManager
		ELM      → ELMExpManager
		"""
		if self.model_type == 'elm':                    # ← NEW
			from core.elm_exp_manager import ELMExpManager
			executor = ELMExpManager(
				base_output_dir = output_dir
			)
			return executor.execute_plan(
				plan,
				{}
			)
		else:
			executor = ExpManager(
				base_output_dir = output_dir
			)
			return executor.execute_plan(plan, {
				'pflotran_exe':  pflotran_exe,
				'time_indices':  [0, 1, 2, 3, 4, 5],
				'skip_plotting': False,
			})
	
	# ═════════════════════════════════════════════════════════
	# UTILITIES
	# ═════════════════════════════════════════════════════════
	def _print_status(self):
		print("\n" + "-" * 70)
		print("CONVERSATION STATUS")
		print("-" * 70)
		print(f"Model:        {self.model_type.upper()}")
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
				  f"• Model:       {self.model_type.upper()}",
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
        description="PFLOTRAN/ELM Coordinator"
    )
    parser.add_argument(
        '--interactive', '-i',
        action = 'store_true',
        help   = 'Run in interactive mode'
    )
    parser.add_argument(
        '--model', '-M',
        default = 'pflotran',
        choices = ['pflotran', 'elm'],   # ← NEW
        help    = 'Model to use (pflotran or elm)'
    )
    parser.add_argument(
        '--pflotran-exe', '-p',
        default = 'pflotran',
        help    = 'Path to PFLOTRAN executable'
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
    args = parser.parse_args()

    coordinator = PFLOTRANCoordinator(
        default_pflotran_exe = args.pflotran_exe,
        default_output_dir   = args.output_dir,
        mcp_config_file      = args.mcp_config,
        model_type           = args.model,       # ← NEW
    )

    if args.interactive:
        coordinator.run_interactive()
    else:
        print("=" * 70)
        print("Run with --interactive for interactive mode")
        print("\nExamples:")
        print("  python workflow.py --interactive")
        print("  python workflow.py --interactive --model elm")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()