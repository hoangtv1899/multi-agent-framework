#!/usr/bin/env python3
"""
The Analyzer
src/agents/analyzer.py

The fourth box. It reads what the Experiment Manager produced and says what it
means, in five steps that each hand the next one a file:

    step 0  context      the three boundary files -> plan / data / caveats
    step 1  compare      SWE, streamflow, WTD against observations
                         -> comparison.json + 5 figures
    step 2  investigate  an LLM decides which figures answer the user's
                         question and writes the code; pandas computes
                         -> investigation.json + up to 5 figures
    step 3  interpret    an LLM concludes and code audits the conclusion;
                         may send step 2 back once -> interpretation.json
    step 4  report       assemble -> analysis.json, the box's boundary file

These stages used to live inside the Experiment Manager, which made the
boundary between the boxes fictional: the manager both ran the model and judged
it, so "the experiment succeeded" and "the experiment showed something" were
decided by the same code. They are different questions with different failure
modes. A run whose columns all completed is a manager success even when every
column drains twice its precipitation, and only the Analyzer can say the second
thing — about a run the manager considers finished.

WHY EACH STEP WRITES A FILE. Any step can be re-run against an archived study
without repeating the ones before it. That is not tidiness: step 2 costs an LLM
call plus a subprocess per figure, and step 3 could not be developed at all
while its input existed only in memory. It is also what lets a person open
04_analysis/ and see what the machine saw.

EVERY STEP IS NON-FATAL. A step that fails records why and the rest continue.
An Analyzer that aborted on a failed figure would discard the four that worked
and the comparison that preceded them — and this box exists to report what a
run shows, including that part of it could not be shown.
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class Analyzer:
    """steps 0-4 over one finished run directory."""

    def __init__(self, run_dir: str, verbose: bool = True):
        self.run_dir = Path(run_dir)
        self.analysis_dir = self.run_dir / "04_analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ─────────────────────────────────────────────────────────
    # The whole box, in the order the steps depend on each other
    # ─────────────────────────────────────────────────────────
    def run(self,
            results: Any = None,
            config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """step 0 -> 1 -> (2 <-> 3) -> 4.

        `results` is accepted for the manager's call signature and no longer
        used: step 0 reads the packaged run off disk, which is what lets the
        same code serve a live run and an archived one identically. Passing the
        in-memory object would have made those two paths different.
        """
        config = config or {}
        t0 = time.time()
        status: Dict[str, Any] = {"steps": {}}

        # ── step 0 ───────────────────────────────────────────────────────
        try:
            from agents.analysis import step0_context
            ctx = step0_context.load(self.run_dir)
            n_block = sum(1 for c in (ctx.caveats or [])
                          if c.get("severity") == "blocking")
            self._say(f"✓ step 0  context: {len(ctx.columns)} columns, "
                      f"{len(ctx.caveats or [])} caveats ({n_block} blocking)")
            status["steps"]["context"] = True
        except Exception as e:
            # Without step 0 there is nothing for any later step to read, so
            # this is the one failure that ends the box.
            self._say(f"❌ step 0  context failed: {e}")
            status["steps"]["context"] = False
            status["error"] = str(e)
            return status

        # NOTHING TO ANALYSE IS NOT A REASON TO CALL AN LLM. A run directory
        # with no columns — empty, half-written, or one whose extraction failed
        # — used to fall straight through into steps 2 and 3, which spent real
        # API calls to discover there was no data. Checked here because it is
        # the first point that knows.
        if not ctx.columns:
            self._say("   ⚠️  no columns in this run — nothing to analyse")
            status["steps"]["compare"] = False
            status["steps"]["investigate"] = False
            status["steps"]["interpret"] = False
            status["steps"]["report"] = False
            status["error"] = "no columns"
            status["seconds"] = round(time.time() - t0, 1)
            return status

        # ── step 1 ───────────────────────────────────────────────────────
        comparison = {}
        try:
            from agents.analysis import step1_compare
            comparison = step1_compare.compare_all(ctx, self.analysis_dir)
            self._say(f"✓ step 1  compare: {len(comparison.get('figures') or {})} "
                      f"figures, {len(comparison.get('caveats') or [])} caveats "
                      f"→ comparison.json")
            status["steps"]["compare"] = True
        except Exception as e:
            self._say(f"   ⚠️  step 1 compare failed: {e}")
            status["steps"]["compare"] = False

        # ── steps 2 and 3, as a bounded loop ─────────────────────────────
        investigation = interpretation = None
        rounds, stopped = [], None
        try:
            from agents.analysis import step3_interpret
            loop = step3_interpret.investigate_and_interpret(
                ctx, self.analysis_dir, comparison=comparison,
                model=config.get("analysis_model",
                                 step3_interpret.DEFAULT_MODEL),
                with_images=config.get("analysis_with_images", True),
                max_rounds=config.get("analysis_max_rounds",
                                      step3_interpret.MAX_ROUNDS))
            investigation = loop["investigation"]
            interpretation = loop["interpretation"]
            rounds, stopped = loop["rounds"], loop["stopped_because"]
            a = interpretation.get("audit") or {}
            self._say(f"✓ step 2  investigate: "
                      f"{investigation['n_succeeded']}/{investigation['n_proposed']} "
                      f"figures over {loop['n_rounds']} round(s) "
                      f"→ investigation.json")
            self._say(f"✓ step 3  interpret: {interpretation['verdict']}, "
                      f"{a.get('n_claims', 0) - a.get('n_struck', 0)} claims kept "
                      f"of {a.get('n_claims', 0)} ({stopped}) "
                      f"→ interpretation.json")
            status["steps"]["investigate"] = True
            status["steps"]["interpret"] = True
        except Exception as e:
            self._say(f"   ⚠️  steps 2-3 failed: {e}")
            status["steps"]["investigate"] = status["steps"]["interpret"] = False

        # ── step 4 ───────────────────────────────────────────────────────
        try:
            from agents.analysis import step4_report
            report = step4_report.build(
                ctx, comparison, investigation or {}, interpretation or {},
                self.run_dir, rounds=rounds, stopped_because=stopped)
            path = step4_report.write(report, self.analysis_dir)
            self._say("✓ step 4  report → " + str(Path(path).name))
            if self.verbose:
                self._say("\n" + step4_report.summary(report))
            status["steps"]["report"] = True
            status["report"] = path
            status["verdict"] = report.get("verdict")
        except Exception as e:
            self._say(f"   ⚠️  step 4 report failed: {e}")
            status["steps"]["report"] = False

        status["seconds"] = round(time.time() - t0, 1)
        return status
