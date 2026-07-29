#!/usr/bin/env python3
"""
The Analyzer
src/agents/analyzer.py

The fourth box. It reads what the Experiment Manager produced and says what it
means: draws the science figures, compares the run against observations, and
writes the interpretation.

These three stages used to live inside the Experiment Manager, which made the
boundary between the boxes fictional — the manager both ran the model and
judged it, so "the experiment succeeded" and "the experiment showed something"
were decided by the same code. They are different questions with different
failure modes. A run whose columns all completed is a manager success even when
every column drains twice its precipitation; only the Analyzer can say the
second thing, and it has to be able to say it about a run the manager considers
finished.

What stays with the manager is EXTRACTION — pulling numbers out of ELM history
NetCDFs. That is reading the model's own output format, which is a property of
the backend, not of the analysis. The Analyzer consumes the extracted shape.

Every stage here is non-fatal. A run stands as a run without figures, without
observations, and without prose; losing the interpretation must never cost the
compute that produced it.
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parents[2]


def _load_tool(name: str):
	"""Import tools/<name>.py by path — tools/ is a script dir, not a package.

	Kept identical to the manager's loader so the Analyzer and the standalone
	CLIs (analyze_run, validate_run, analyze_agentic, interpret_run) share one
	implementation of each stage rather than drifting apart.
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


class Analyzer:
	"""Figures, validation, interpretation — over one finished run directory."""

	def __init__(self, run_dir: str, verbose: bool = True):
		self.run_dir      = Path(run_dir)
		self.analysis_dir = self.run_dir / "04_analysis"
		self.analysis_dir.mkdir(parents=True, exist_ok=True)
		self.verbose      = verbose

	def _say(self, msg: str) -> None:
		if self.verbose:
			print(msg)

	# ─────────────────────────────────────────────────────────
	# The whole box, in the order the stages depend on each other
	# ─────────────────────────────────────────────────────────
	def run(self,
			results: Any = None,
			config:  Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
		"""Figures → validation → interpretation.

		`results` is the manager's extracted results object when the Analyzer
		runs in-process straight after a run; the figures need the ensemble in
		memory. Omitted, the figure stage is skipped and the rest still runs
		off what is on disk — which is how the Analyzer is invoked against an
		older run directory.
		"""
		config = config or {}
		return {
			"figures":        self.figures(results) if results is not None else False,
			"validation":     self.validate(config),
			"interpretation": self.interpret(config),
		}

	# ─────────────────────────────────────────────────────────
	# FIGURES
	# ─────────────────────────────────────────────────────────
	def figures(self, analyzer: Any) -> bool:
		"""04_analysis/ figures — the same five tools/analyze_run.py --plot draws.

		These are the science figures: partitioning (fractions of P per column),
		controls (fractions vs drivers, confound shown), soil_control (forcing
		held constant) and wtd_columns. They read the ensemble, so they answer
		questions about the ensemble.

		They replaced ELMResultsAnalyzer.plot_all(), which drew a 2-panel
		<case>_overview.png per column plus one comparison figure. That set had
		no elevation gradient, no soil attribution, no water-budget closure and
		no WTD panel — everything the study is actually about — and it meant an
		integrated run produced a strictly weaker figure set than the same run
		analysed from the command line. Non-fatal.
		"""
		try:
			ar   = _load_tool("analyze_run")
			from agents.analysis import step2_derive as _drv
			rows = getattr(analyzer, "results", None) or []
			if isinstance(rows, dict):
				rows = list(rows.values())
			# computed here, not read off the extraction object: the old call
			# returned {} on every run, so soil_control.png was never drawn
			soil = _drv.soil_attribution(rows)
			if not soil.get("available"):
				soil = None

			n = 0
			ar.plot_partitioning(analyzer.results,
								 self.analysis_dir / "partitioning.png"); n += 1
			ar.plot_controls(analyzer.results,
							 self.analysis_dir / "controls.png"); n += 1
			ar.plot_spatial(analyzer.results, self.run_dir,
							self.analysis_dir / "spatial.png"); n += 1
			ar.plot_wtd(analyzer.results, self.run_dir,
						self.analysis_dir / "wtd_columns.png"); n += 1
			if soil:
				ar.plot_soil(soil, self.analysis_dir / "soil_control.png"); n += 1
			self._say(f"✓ {n} analysis figure(s) → 04_analysis/")
			return True
		except Exception as e:
			self._say(f"   ⚠️  analysis plots failed: {e}")
			return False

	# ─────────────────────────────────────────────────────────
	# VALIDATE AGAINST OBSERVATIONS
	# ─────────────────────────────────────────────────────────
	def validate(self, config: Dict[str, Any]) -> bool:
		"""Compare the run against in-domain observations (USGS wells and
		gauges, SNOTEL SWE) -> 04_analysis/validation.json + validation.png.

		Reuses tools/validate_run.build_validation() rather than
		reimplementing it. Non-fatal: needs live MCP + network, and a run is
		still useful without it.
		"""
		clients = (config or {}).get("mcp_clients") or {}
		if not clients:
			self._say("   ⚠️  no MCP clients — skipping observation validation")
			return False
		try:
			vr  = _load_tool("validate_run")
			val = vr.build_validation(self.run_dir, clients)
			(self.analysis_dir / "validation.json").write_text(
				json.dumps(val, indent=2, default=str))
			try:
				vr.plot_validation(val, self.analysis_dir / "validation.png")
			except Exception as e:
				self._say(f"   ⚠️  validation plot failed: {e}")
			n = sum(1 for t in (val.get("targets") or [])
					if t.get("status") == "compared")
			self._say(f"✓ validation: {n} target(s) compared "
					  f"→ 04_analysis/validation.json")
			return True
		except Exception as e:
			self._say(f"   ⚠️  observation validation failed ({e}) — continuing")
			return False

	# ─────────────────────────────────────────────────────────
	# INTERPRET (numbers + feasibility + validation)
	# ─────────────────────────────────────────────────────────
	def interpret(self, config: Dict[str, Any]) -> bool:
		"""Choose figures, render, LOOK at them, interpret.

		Agentic by default — it picks which figures answer the question rather
		than emitting a fixed set, renders them from the vetted registry, and
		reviews each rendering by sight before writing the interpretation.
		config['agentic_analyzer']=False falls back to the one-shot interpreter.

		Flexible in what it explores; bound in what it may claim. Numbers must
		trace to the JSON, every figure records its provenance, and the
		validation verdicts (context-only, the domain-match and impossible-ratio
		refusals) are not negotiable. Non-fatal either way.
		"""
		cfg   = config or {}
		model = cfg.get("interpreter_model", "claude-opus-4-8-project")
		if cfg.get("agentic_analyzer", True):
			try:
				ag = _load_tool("analyze_agentic")
				import sys as _sys
				argv = _sys.argv
				_sys.argv = ["analyze_agentic", "--run-dir", str(self.run_dir),
							 "--model", model]
				if not cfg.get("analyzer_vision", True):
					_sys.argv.append("--no-vision")
				try:
					ag.main()
				finally:
					_sys.argv = argv
				self._say("✓ analysis → 04_analysis/{analysis_plan,figure_captions}"
						  ".json + interpretation.md")
				return True
			except SystemExit as e:
				self._say(f"   ⚠️  agentic analyzer stopped ({e}) — falling back")
			except Exception as e:
				self._say(f"   ⚠️  agentic analyzer failed ({e}) — falling back")
		try:
			ir = _load_tool("interpret_run")
			ir.interpret(self.run_dir, model=model, quiet=True)
			self._say("✓ interpretation → 04_analysis/interpretation.md")
			return True
		except Exception as e:
			self._say(f"   ⚠️  interpretation failed ({e}) — continuing")
			return False
