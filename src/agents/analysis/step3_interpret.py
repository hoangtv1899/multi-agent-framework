#!/usr/bin/env python3
"""
Analyzer step 3 — interpret the findings, and decide whether they suffice
src/agents/analysis/step3_interpret.py

    in   ctx + step 1's comparison record + step 2's investigation record
    out  04_analysis/interpretation.json  {claims, verdict, audit, feedback}

STEP 2 CONCLUDES NOTHING; STEP 3 COMPUTES NOTHING. That inversion is the whole
design. Step 2 decided what to compute and let pandas do it; step 3 reads what
came back, looks at the figures, and says what it means — but may not produce a
number of its own, because a number with no finding behind it cannot be checked
against anything.

THE LLM JUDGES; CODE AUDITS THE JUDGEMENT. Three checks, mechanical rather than
prompted, because a prompt cannot be verified after the fact and this is where
every caveat raised upstream finally has to bind:

    cites       every claim names a finding_id that exists. A sentence with no
                citation is an assertion with no evidence.
    no_new      every number in a claim appears in the cited finding's result.
                This is what stops a rounded, restated or invented figure —
                and rounding is not a nicety here: "recharge is about 30 mm/yr"
                cannot be traced back to 31.4 by anything downstream.
    respects    a claim whose cited finding is inside a blocking caveat's scope
                must carry that caveat's id, or it is struck.

A struck claim is not deleted quietly — it is recorded with the reason, because
the audit's own findings are evidence about the run.

IT LOOKS AT THE FIGURES. The gateway serves vision, and a reviewer that reads
only the numbers cannot see an unreadable axis, a scale that hides the data, or
a plot whose shape contradicts its caption. The units bug earlier in this
project was found by LOOKING at a figure whose numbers had passed every guard.

THE LOOP IS BOUNDED AT MAX_ROUNDS. Step 3 may send step 2 back once to fix what
it judged insufficient. Two rounds, not more: each costs an LLM call plus a
subprocess per figure, and a reviewer allowed to keep asking will keep asking.
If round 2 is still judged insufficient, that verdict IS the result — an
experiment that cannot answer the question is a finding, not a failure to
retry harder.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

FILENAME = "interpretation.json"

# At most one revision. See the module docstring: the second round's verdict
# stands whatever it says.
MAX_ROUNDS = 2

DEFAULT_MODEL = "claude-opus-5-project"

# Numbers below this are structural (indices, counts, years) rather than
# measurements, and demanding provenance for "2019" or "19 columns" would
# make the no-new-numbers audit fire constantly on prose that is fine.
_YEARLIKE = re.compile(r"^(19|20)\d{2}$")


def _numbers(text: str) -> List[str]:
    return re.findall(r"-?\d+(?:\.\d+)?", str(text))


def _flatten_numbers(obj) -> set:
    """Every number anywhere in a finding's result, as strings.

    Compared as strings deliberately: 31.4 and 31.40 are the same measurement,
    but a claim saying 31 when the finding says 31.4 has rounded, and rounding
    breaks the trace back to the data. The audit should notice that.
    """
    out = set()
    if isinstance(obj, dict):
        for v in obj.values():
            out |= _flatten_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out |= _flatten_numbers(v)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out.add(str(obj))
        out.add(str(round(float(obj), 3)).rstrip("0").rstrip("."))
        out.add(str(int(obj)) if float(obj).is_integer() else str(obj))
    elif isinstance(obj, str):
        out |= set(_numbers(obj))
    return out


def audit(claims: List[Dict[str, Any]], investigation: Dict[str, Any],
          caveats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check each claim against the evidence it cites. Pure function.

    Returns {kept, struck} where every struck claim carries its reason. Testable
    with no API call, which is the point: the audit is the part that must not
    depend on a model behaving well.
    """
    by_id = {f["id"]: f for f in (investigation.get("findings") or [])}
    blocking = {c.get("id"): c for c in (caveats or [])
                if c.get("severity") == "blocking"}

    kept, struck = [], []
    for c in claims:
        if not isinstance(c, dict):
            continue
        fid = c.get("finding_id")
        text = str(c.get("claim") or "")

        if fid not in by_id:
            struck.append(dict(c, struck_because=(
                f"cites finding_id {fid!r}, which step 2 did not produce. A "
                f"claim with no evidence behind it cannot be checked.")))
            continue

        have = _flatten_numbers(by_id[fid].get("result"))
        invented = [n for n in _numbers(text)
                    if n not in have and not _YEARLIKE.match(n)]
        if invented:
            struck.append(dict(c, struck_because=(
                f"states {', '.join(invented[:4])}, which does not appear in "
                f"finding {fid}. Step 3 may not produce numbers — a rounded or "
                f"restated figure cannot be traced back to the data.")))
            continue

        cited = set(c.get("caveats") or [])
        missing = [cid for cid in blocking
                   if cid in (by_id[fid].get("blocked_by") or [])
                   and cid not in cited]
        if missing:
            struck.append(dict(c, struck_because=(
                f"falls inside blocking caveat(s) {', '.join(missing)} without "
                f"carrying them. An unrespected caveat looks exactly like a "
                f"respected one, which is why they are records.")))
            continue

        kept.append(c)

    return {"kept": kept, "struck": struck,
            "n_claims": len(claims), "n_struck": len(struck)}


# ─────────────────────────────────────────────────────────────────────
# THE PROMPT
# ─────────────────────────────────────────────────────────────────────
def review_brief(ctx, comparison: Dict[str, Any],
                 investigation: Dict[str, Any]) -> str:
    plan = ctx.plan or {}
    blocking = [c for c in (ctx.caveats or []) if c.get("severity") == "blocking"]
    blocking += [c for c in (comparison.get("caveats") or [])
                 if c.get("severity") == "blocking"]

    lines = [
        "THE USER ASKED:",
        f"    {plan.get('question')}",
        "",
        "BLOCKING CAVEATS — a claim of these kinds must not be made, or must",
        "carry the caveat id explicitly:",
    ]
    for c in blocking:
        lines.append(f"    [{c.get('id')}] applies to: {c.get('applies_to')}")
        lines.append(f"        {str(c.get('statement'))[:240]}")

    lines += ["", "WHAT STEP 2 INVESTIGATED:",
              f"    {investigation.get('notes')}", "", "FINDINGS:"]
    for f in (investigation.get("findings") or []):
        lines.append(f"    id: {f['id']}   (n={f.get('n')}, scale={f.get('scale')})")
        lines.append(f"        question: {f.get('question')}")
        lines.append(f"        result:   {json.dumps(f.get('result'), default=str)[:700]}")
    if investigation.get("caveats"):
        lines.append("")
        lines.append("FIGURES THAT FAILED TO PRODUCE A RESULT:")
        for c in investigation["caveats"]:
            lines.append(f"    {str(c.get('statement'))[:200]}")
    return "\n".join(lines)


TASK = f"""\
You are reviewing this analysis. The figures are attached as images — look at
them, not only at the numbers: an unreadable scale or a plot whose shape
contradicts its stated question is exactly what this review is for.

You may NOT produce a number of your own. Every figure you state must appear in
the cited finding's `result`, exactly as it appears there — do not round, do
not restate, do not convert. A number that cannot be traced back to a finding
is struck automatically.

Judge whether the findings answer what the user asked. Decide:
  "sufficient"    the question is answered as well as this experiment allows
  "insufficient"  a revised set of figures would materially improve the answer

"insufficient" is not for wishing the experiment were different. If the
experiment cannot address the question, that is a SUFFICIENT answer of the form
"this cannot be determined from this run, because ...". Only ask for a revision
if step 2 could plausibly do better with the same data.

Return ONLY JSON:
{{"claims": [
    {{"claim": "<one sentence, every number traceable to the finding>",
      "finding_id": "<the finding it rests on>",
      "caveats": ["<ids of any blocking caveat this claim falls under>"]}}
  ],
  "answer": "<2-4 sentences answering the user's question, or saying plainly "
            "that this experiment cannot answer it and why>",
  "verdict": "sufficient" | "insufficient",
  "feedback": "<if insufficient: what step 2 should do differently. Be "
              "specific about which figure and what is wrong with it. Empty "
              "if sufficient.>"}}
"""


def _parse(reply: str) -> Dict[str, Any]:
    m = re.search(r"```(?:json)?\s*(.+?)```", reply, re.S)
    text = m.group(1) if m else reply
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j < 0:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[i:j + 1])


def _content(brief: str, figures: List[str], with_images: bool):
    """The user turn: the brief, then each figure as an image block."""
    parts: List[Dict[str, Any]] = [{"type": "text", "text": brief + "\n" + TASK}]
    if not with_images:
        return brief + "\n" + TASK
    import base64
    for p in figures:
        try:
            b = base64.b64encode(Path(p).read_bytes()).decode()
            parts.append({"type": "text", "text": f"figure: {Path(p).stem}"})
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:image/png;base64,{b}"}})
        except Exception:
            continue
    return parts


def interpret(ctx, comparison, investigation, out_dir,
              model: str = DEFAULT_MODEL, client=None,
              with_images: bool = True) -> Dict[str, Any]:
    """One review pass: judge, then audit the judgement."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if client is None:
        from agents.llm_agent import SimpleLLMClient
        client = SimpleLLMClient(model=model)

    brief = review_brief(ctx, comparison or {}, investigation or {})
    content = _content(brief, investigation.get("figures") or [], with_images)
    spec = _parse(client.ask([{"role": "user", "content": content}]))

    caveats = list(ctx.caveats or []) + list((comparison or {}).get("caveats") or [])
    result = audit(spec.get("claims") or [], investigation, caveats)

    verdict = spec.get("verdict")
    if verdict not in ("sufficient", "insufficient"):
        verdict = "insufficient"          # an unparseable verdict is not a pass

    out = {"answer": spec.get("answer"), "verdict": verdict,
           "feedback": spec.get("feedback") or None,
           "claims": result["kept"], "struck": result["struck"],
           "audit": {k: result[k] for k in ("n_claims", "n_struck")},
           "round": investigation.get("round", 1)}
    (out_dir / FILENAME).write_text(json.dumps(out, indent=2, default=str))
    return out


# ─────────────────────────────────────────────────────────────────────
# THE BOUNDED LOOP
# ─────────────────────────────────────────────────────────────────────
def investigate_and_interpret(ctx, out_dir, comparison=None,
                              model: str = DEFAULT_MODEL,
                              client=None, with_images: bool = True,
                              max_rounds: int = MAX_ROUNDS) -> Dict[str, Any]:
    """step 2 -> step 3, with at most `max_rounds` passes.

    Step 3 may send step 2 back once with specific feedback. It does not get to
    keep asking: each round costs an LLM call plus a subprocess per figure, and
    a reviewer with an unbounded budget will always find something. If the last
    round is still judged insufficient, that verdict stands and is reported —
    "this experiment cannot answer the question" is a result.
    """
    from agents.analysis import step2_investigate as _step2

    rounds: List[Dict[str, Any]] = []
    feedback = None
    investigation = interpretation = None

    for r in range(1, max(1, max_rounds) + 1):
        investigation = _step2.investigate(
            ctx, out_dir, step1=comparison, model=model,
            feedback=feedback, round_no=r)
        interpretation = interpret(ctx, comparison, investigation, out_dir,
                                   model=model, client=client,
                                   with_images=with_images)
        rounds.append({"round": r,
                       "verdict": interpretation["verdict"],
                       "n_findings": investigation["n_succeeded"],
                       "n_claims": interpretation["audit"]["n_claims"],
                       "n_struck": interpretation["audit"]["n_struck"],
                       "feedback": interpretation.get("feedback")})
        if interpretation["verdict"] == "sufficient":
            break
        feedback = interpretation.get("feedback")
        if not feedback:                  # insufficient with nothing actionable
            break                         # to say is not worth another round

    return {"investigation": investigation, "interpretation": interpretation,
            "rounds": rounds, "n_rounds": len(rounds),
            "stopped_because": ("sufficient"
                                if interpretation["verdict"] == "sufficient"
                                else f"exhausted {len(rounds)} of {max_rounds} rounds")}
