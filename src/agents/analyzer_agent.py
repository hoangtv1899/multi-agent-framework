#!/usr/bin/env python3
"""The Analyzer, agentic: choose the figures, render them, LOOK at them, revise.

The other three agents reason; this one used to render a fixed set and hand the
numbers to a one-shot interpreter. It now runs the loop a person runs — decide
which figures answer the question, draw them, look at whether they actually do,
redraw if not, then interpret.

Why this stage is allowed to be flexible when the Planner is not: a planner
error is expensive and silent (a CIME build per column, inherited by every later
stage, visible only once the compute is spent), while an analyzer error is cheap
and visible (the data is unchanged, regeneration takes seconds, and the failure
lands in front of a human). Different risk, different constraint.

The three things it is NOT free to do:

  1. INVENT NUMBERS. Every value the interpretation states comes from
     experiment.json / validation.json. Vision judges whether a figure is
     readable and on-point; it never reads a measurement off an image.
  2. HIDE PROVENANCE. Each figure records whether a vetted registry function or
     analyzer-supplied code drew it.
  3. UPGRADE THE EVIDENCE. It may plot and propose anything, but it cannot turn
     a `context-only` comparison into support, nor contradict the assumptions
     ledger, the limitations catalogue, or the refusals validation already
     issued (domain mismatch, impossible runoff ratio, temporal mismatch). A
     fluent paragraph explaining why a bad comparison is fine reads better than
     a refusal, which is exactly why the refusal has to bind.
"""
import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, "src")

from agents.llm_agent import SimpleLLMClient
from core.figure_registry import (REGISTRY, available, catalogue,
                                  detect_capabilities)

SELECT_SYSTEM = """You are the ANALYSIS agent of a hydrologic simulation
framework. A study has finished; deterministic analysis and observation
validation have already run. Your job right now is ONLY to choose which figures
answer the user's question.

Rules:
- Choose from the catalogue you are given. Nothing else exists.
- Choose the FEWEST figures that answer the question. Three good figures beat
  eight. A figure you cannot say a purpose for should not be chosen.
- Order them the way a reader should meet them.
- If the question needs something the catalogue cannot draw, say so in
  `missing_capabilities` — that is a real finding, not a failure.

Return JSON only:
{"figures": [{"name": "<from the catalogue>", "why": "<what THIS figure
contributes to answering the question, one line>"}],
 "missing_capabilities": ["<a figure the question needs that does not exist>"]}"""

REVIEW_SYSTEM = """You are reviewing a rendered scientific figure for whether it
does its job. You are looking at the IMAGE.

Judge only what an image can settle:
- Is it readable? Overlapping labels, colliding annotations, an empty panel, an
  unreadable legend, a colourbar drawn over an axis.
- Does it show what it was chosen to show?

Do NOT read data values off the plot and do NOT judge whether the science is
right — the numbers live in JSON and are authoritative. If the figure is fine,
say so plainly.

Return JSON only:
{"usable": true|false,
 "problems": ["<specific, actionable rendering problem>"],
 "shows_what_was_asked": true|false}"""

INTERPRET_SYSTEM = """You are the analysis agent of a hydrologic simulation
framework. Write the interpretation of a finished study.

THE BINDING RULES — these are not style guidance:
- Every number you state must appear in the payload. Never compute a new one,
  never estimate, never read one off a figure.
- Validation verdicts are FINAL. A comparison marked `context-only` is NOT
  evidence and you may not argue it into evidence. If validation refused a
  comparison (domain mismatch, an impossible runoff ratio, a temporal
  mismatch), that refusal stands and you say so.
- The assumptions ledger and limitations catalogue bind you. If initialization
  is DEFAULT-sourced and the run is short, that is a caveat you must carry.
- If the evidence does not answer the question, say that. An honest "this run
  cannot answer that, and here is what would" is the correct output.

Output MARKDOWN, UNDER 220 WORDS, exactly these sections:
**Answer** — 1-2 sentences, leading with the number that answers the question.
**Why** — 3-4 bullets, each driver -> response with its number and a short
mechanism.
**Trust** — what the observations do and do not support, naming any refused
comparison, plus the single most consequential caveat.
**Next** — 1-2 experiments, each tied to a limitation actually present."""


def _b64(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


class AnalyzerAgent:
    """Selects figures, renders them, reviews them by sight, then interprets."""

    def __init__(self, model: str = "claude-opus-4-8-project",
                 vision: bool = True, max_review_rounds: int = 1,
                 verbose: bool = True):
        self.llm = SimpleLLMClient(model=model)
        self.vision = vision
        self.max_review_rounds = max_review_rounds
        self.verbose = verbose

    def _say(self, *a):
        if self.verbose:
            print(*a)

    # ── 1. choose ────────────────────────────────────────────────────────
    def select_figures(self, question: str, caps: List[str],
                       brief: Dict = None, plan: Dict = None) -> Dict:
        menu = catalogue(caps)
        msg = (f"USER QUESTION:\n{question}\n\n"
               f"FIGURES AVAILABLE FOR THIS RUN:\n{menu}\n\n"
               f"DOMAIN: {json.dumps((brief or {}).get('domain') or {})}\n"
               f"FEASIBILITY: {json.dumps((plan or {}).get('feasibility') or {})}\n\n"
               "Choose the fewest figures that answer the question.")
        sel = _parse_json(self._chat(SELECT_SYSTEM, msg)) or {}
        names = [f["name"] for f in (sel.get("figures") or [])
                 if f.get("name") in REGISTRY and f["name"] in available(caps)]
        # never return nothing: fall back to whatever the run supports
        if not names:
            names = available(caps)[:4]
            sel = {"figures": [{"name": n, "why": "default selection"}
                               for n in names]}
        sel["figures"] = [f for f in sel["figures"] if f["name"] in names]
        return sel

    # ── 2. look ──────────────────────────────────────────────────────────
    def review_figure(self, path: Path, purpose: str) -> Dict:
        """Ask the model to LOOK at the rendered figure. Layout only."""
        if not self.vision or not Path(path).exists():
            return {"usable": True, "problems": [], "skipped": True}
        try:
            r = self.llm.client.chat.completions.create(
                model=self.llm.model, max_tokens=400,
                messages=[{"role": "system", "content": REVIEW_SYSTEM},
                          {"role": "user", "content": [
                              {"type": "text",
                               "text": f"This figure was chosen to show: {purpose}"},
                              {"type": "image_url", "image_url": {
                                  "url": f"data:image/png;base64,{_b64(path)}"}}]}])
            out = _parse_json(r.choices[0].message.content or "") or {}
            out.setdefault("usable", True)
            out.setdefault("problems", [])
            return out
        except Exception as e:
            return {"usable": True, "problems": [], "error": str(e)[:120]}

    # ── 3. interpret ─────────────────────────────────────────────────────
    def interpret(self, question: str, payload: Dict) -> str:
        return self._chat(INTERPRET_SYSTEM,
                          f"QUESTION:\n{question}\n\nEVIDENCE:\n"
                          f"{json.dumps(payload, indent=1, default=str)}",
                          max_tokens=1600)

    def _chat(self, system: str, user: str, max_tokens: int = 1200) -> str:
        r = self.llm.client.chat.completions.create(
            model=self.llm.model, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        return r.choices[0].message.content or ""


def _parse_json(text: str):
    """Tolerant JSON extraction — reuses the repair cascade the other agents use."""
    try:
        from agents.llm_agent import LLMAgent
        return LLMAgent.parse_json(LLMAgent, text)          # unbound is fine
    except Exception:
        pass
    import re
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None
