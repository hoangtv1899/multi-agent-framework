#!/usr/bin/env python3
"""
Analysis Report Agent
Two LLM calls:
    1. Analysis   → interprets simulation results
    2. Validation → checks and fixes the analysis
"""
import json
from pathlib import Path
from typing  import Dict, Any, Optional

from agents.llm_agent import LLMAgent
from agents.prompts   import load_prompt


class AnalysisReportAgent(LLMAgent):
	"""
	Interprets PFLOTRAN simulation results.
	
	Receives:
		- User's original question
		- Experiment plan (from Planner)
		- LLM_ANALYSIS_INPUT.json (from ExpManager)
	
	Produces:
		- Direct answer to user question
		- Key findings with numbers + physics
		- Solver performance summary
		- Specific next-step recommendations
	"""
	
	def __init__(self,
				 model:       str  = "claude-sonnet-4-5-20250929-v1-project",
				 model_type:  str  = "pflotran",
				 mcp_clients: Dict = None):
	
		if model_type == "elm":
			self.prompt_analysis   = load_prompt("analyzer_system_elm")
			self.prompt_validation = load_prompt("analyzer_validation")
		else:
			self.prompt_analysis   = load_prompt("analyzer_system")
			self.prompt_validation = load_prompt("analyzer_validation")
	
		super().__init__("analyzer", self.prompt_analysis, model)
	
	# ─────────────────────────────────────────────────────────────────
	# MAIN ENTRY POINT
	# ─────────────────────────────────────────────────────────────────
	
	def generate_analysis_report(self,
								  user_request:    str,
								  llm_input_file:  str,
								  output_file:     str,
								  experiment_plan: Dict  = None,
								  **kwargs                        # absorbs old params
								  ) -> Dict[str, Any]:
		"""
		Generate analysis report from simulation results.
	
		Args:
			user_request:    Original user question
			llm_input_file:  Path to LLM_ANALYSIS_INPUT.json
			output_file:     Path to save analysis report
			experiment_plan: Planner JSON (optional but recommended)
	
		Returns:
			Analysis dict with answer, findings, performance, recommendations
		"""
		# ── Load simulation data ──────────────────────────────────
		print("\n📊 Loading simulation data...")
		data = self._load_data(llm_input_file, experiment_plan)
	
		# ── Call 1: Analyze ───────────────────────────────────────
		print("\n🔬 Analyzing results...")
		analysis = self._analyze(user_request, data)
	
		# ── Call 2: Validate + fix ────────────────────────────────
		print("\n✅ Validating analysis...")
		analysis = self._validate_and_fix(analysis, data)
	
		# ── Save report ───────────────────────────────────────────
		self._save_report(analysis, output_file)
	
		return analysis
	
	# ─────────────────────────────────────────────────────────────────
	# LOAD DATA
	# ─────────────────────────────────────────────────────────────────
	
	def _load_data(self,
				   llm_input_file:  str,
				   experiment_plan: Optional[Dict]) -> Dict[str, Any]:
		"""
		Load and merge simulation data + experiment plan.
		Python only — no LLM involved.
		"""
		data = {}
	
		# Load ExpManager output
		path = Path(llm_input_file)
		if path.exists():
			with open(path) as f:
				data["simulation"] = json.load(f)
			print(f"   ✓ Loaded simulation data from {path.name}")
		else:
			print(f"   ⚠️  Simulation data not found: {llm_input_file}")
			data["simulation"] = {}
	
		# Merge experiment plan
		if experiment_plan:
			data["experiment_plan"] = {
				"parameter_space":    experiment_plan.get("parameter_space", {}),
				"conditions_count":   len(experiment_plan.get("CONDITIONS_COUPLERS", [])),
				"experiments":        [
					{
						"name":    c.get("EXPERIMENT"),
						"initial": c.get("INITIAL_CONDITION"),
						"surface": c.get("BOUNDARY_CONDITION_SURFACE"),
					}
					for c in experiment_plan.get("CONDITIONS_COUPLERS", [])
				]
			}
	
		return data
	
	# ─────────────────────────────────────────────────────────────────
	# CALL 1: ANALYZE
	# ─────────────────────────────────────────────────────────────────
	
	def _analyze(self,
				 user_request: str,
				 data:         Dict) -> Dict:
		"""One heavyweight LLM call — full analysis."""
		prompt = (
			f"Analyze these PFLOTRAN simulation results.\n\n"
			f"USER QUESTION:\n{user_request}\n\n"
			f"SIMULATION DATA:\n{json.dumps(data, indent=2)}"
		)
		try:
			response = self.ask_with_system(
				user_message   = prompt,
				system_message = self.prompt_analysis
			)
			return self.parse_json(response)
		except Exception as e:
			print(f"   ⚠️  Analysis failed: {e}")
			return self._default_analysis(user_request)
	
	# ─────────────────────────────────────────────────────────────────
	# CALL 2: VALIDATE + FIX
	# ─────────────────────────────────────────────────────────────────
	
	def _validate_and_fix(self,
						  analysis: Dict,
						  data:     Dict) -> Dict:
		"""One LLM call — checks and fixes the analysis."""
		prompt = (
			f"Review this hydrogeology analysis report:\n\n"
			f"ANALYSIS:\n{json.dumps(analysis, indent=2)}\n\n"
			f"SIMULATION DATA (for reference):\n"
			f"{json.dumps(data.get('simulation', {}), indent=2)}"
		)
		try:
			response = self.ask_with_system(
				user_message   = prompt,
				system_message = self.prompt_validation
			)
			result = self.parse_json(response)
	
			# Report issues
			for issue in result.get("issues", []):
				icon = ("❌" if issue.get("severity") == "critical"
						else "⚠️")
				print(f"   {icon} [{issue.get('check')}] "
					  f"{issue.get('issue')}")
	
			# Use corrected analysis if provided
			corrected = result.get("corrected_analysis")
			if corrected:
				print("   ✓ Analysis corrected by validator")
				return corrected
	
			print("   ✓ All checks passed")
			return analysis
	
		except Exception as e:
			print(f"   ⚠️  Validation failed: {e} — using original")
			return analysis
	
	# ─────────────────────────────────────────────────────────────────
	# SAVE REPORT
	# ─────────────────────────────────────────────────────────────────
	
	def _save_report(self,
					 analysis:    Dict,
					 output_file: str):
		"""Save analysis as JSON + readable text summary."""
		output_path = Path(output_file)
		output_path.parent.mkdir(parents=True, exist_ok=True)
	
		# Save JSON
		with open(output_path, "w") as f:
			json.dump(analysis, f, indent=2)
	
		# Save readable text summary
		txt_path = output_path.with_suffix(".txt")
		with open(txt_path, "w") as f:
			f.write("=" * 70 + "\n")
			f.write("PFLOTRAN ANALYSIS REPORT\n")
			f.write("=" * 70 + "\n\n")
	
			f.write("ANSWER\n")
			f.write("-" * 70 + "\n")
			f.write(analysis.get("answer_to_user_question", "N/A"))
			f.write("\n\n")
	
			f.write("KEY FINDINGS\n")
			f.write("-" * 70 + "\n")
			for i, finding in enumerate(
				analysis.get("key_findings", []), 1
			):
				f.write(f"{i}. {finding.get('finding', 'N/A')}\n")
				f.write(f"   Why: {finding.get('physics', 'N/A')}\n\n")
	
			perf = analysis.get("performance", {})
			if perf:
				f.write("SOLVER PERFORMANCE\n")
				f.write("-" * 70 + "\n")
				f.write(f"Behavior:   {perf.get('solver_behavior', 'N/A')}\n")
				f.write(f"Newton:     {perf.get('newton_range', 'N/A')}\n")
				f.write(f"TS Cuts:    {perf.get('timestep_cuts', 'N/A')}\n")
				if perf.get("notes"):
					f.write(f"Notes:      {perf['notes']}\n")
				f.write("\n")
	
			recs = analysis.get("recommendations", [])
			if recs:
				f.write("RECOMMENDATIONS\n")
				f.write("-" * 70 + "\n")
				for rec in recs:
					f.write(f"• {rec}\n")
	
		print(f"   ✓ Report saved to {output_path.name}")
	
	# ─────────────────────────────────────────────────────────────────
	# FALLBACK
	# ─────────────────────────────────────────────────────────────────
	
	@staticmethod
	def _default_analysis(user_request: str) -> Dict:
		"""Safe fallback when analysis LLM call fails."""
		return {
			"answer_to_user_question": (
				f"Analysis could not be completed for: {user_request}"
			),
			"key_findings":   [],
			"performance":    {
				"solver_behavior":    "unknown",
				"newton_range":       "unknown",
				"timestep_cuts":      "unknown",
				"wall_clock_seconds": "unknown",
				"notes":              "Analysis failed — check logs"
			},
			"recommendations": [
				"Review LLM_ANALYSIS_INPUT.json manually"
			]
		}
	
	def __repr__(self):
		return f"AnalysisReportAgent(model={self.llm.model})"