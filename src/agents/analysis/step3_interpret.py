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

# A declared value that is a year is a date, not a measurement.
_YEARLIKE = re.compile(r"^(19|20)\d{2}$")

_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?")


def _numbers(text: str) -> List[str]:
    return _NUMBER.findall(str(text))


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


# Words too generic to identify a caveat's subject. "any", "claim", "at" and
# friends appear in most scopes and would match most sentences.
_STOP = {"any", "the", "and", "or", "of", "to", "at", "in", "on", "for", "a",
         "claim", "claims", "these", "this", "that", "it", "its", "with",
         "without", "not", "no", "is", "are", "be", "been", "validation",
         "partitioning", "stations", "onset", "first", "measured", "modelled"}


def _scope_terms(caveat: Dict[str, Any]):
    """(variables, words) a caveat's `applies_to` is about.

    Variables are the uppercase tokens (QOVER, ZWT, SWE) and are matched
    case-sensitively, because SNOW the variable is not "snow" the word.
    Words are the rest, lowercased, minus the ones too generic to identify a
    subject.
    """
    scope = str(caveat.get("applies_to") or "")
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]*", scope)
    variables = {t for t in tokens if t.isupper() and len(t) > 2}
    words = {t.lower() for t in tokens
             if t not in variables and len(t) > 2 and t.lower() not in _STOP}
    return variables, words


def required_caveats(claim_text: str, candidates: List[str],
                     caveats: List[Dict[str, Any]]) -> List[str]:
    """Which of a finding's candidate caveats this CLAIM actually falls under.

    THE FIX FOR THE AUDIT'S WORST FALSE POSITIVE. _blocked_by tags a finding
    with every caveat scoped to any variable that figure used, so a
    five-variable figure inherits five caveats. Enforcing all of them on every
    claim citing it struck "no observational validation of the water table is
    possible" for not carrying limitation_structural_1 — scoped to RUNOFF. The
    claim was correct and the caveat was irrelevant to it.

    So the figure bounds what is POSSIBLE and the claim decides what APPLIES: a
    caveat is required only if the claim's own text is about its subject. A
    claim that never mentions runoff does not owe the runoff caveat, however
    many variables its figure happened to plot.
    """
    text = str(claim_text or "")
    lower = text.lower()
    by_id = {c.get("id"): c for c in (caveats or [])}
    out = []
    for cid in (candidates or []):
        c = by_id.get(cid)
        if not c:
            continue
        variables, words = _scope_terms(c)
        if any(re.search(rf"\b{re.escape(v)}\b", text) for v in variables) \
           or any(re.search(rf"\b{re.escape(w)}\b", lower) for w in words):
            out.append(cid)
    return out


def run_facts(ctx) -> set:
    """Numbers that describe the RUN rather than any one finding.

    The column count and the simulated years are facts of record in ctx, and a
    claim is entitled to state them. A live run struck a correct sentence —
    "across the 19 sampled columns, precipitation ranges..." — because 19 was
    not in the cited finding's result, only in ctx. The check consulted the
    finding alone, so run-level facts had nowhere to be true.

    Every finding's own `n` is included for the same reason: a claim saying how
    many points it rests on is quoting the record, not inventing.
    """
    out = set()
    try:
        out.add(str(len(ctx.columns)))
        p = (ctx.plan or {}).get("period") or {}
        for k in ("yr_start", "yr_end"):
            if p.get(k) is not None:
                out.add(str(p[k]))
    except Exception:
        pass
    return out


# ─────────────────────────────────────────────────────────────────────
# THE THREE RULES
# ─────────────────────────────────────────────────────────────────────
# NAMED IN CODE, NOT ONLY IN PROSE (2026-08-14). `cites`, `no_new` and
# `respects` were labels that appeared in the documentation and nowhere in the
# source — a reviewer sent to find the function implementing `no_new` found
# three anonymous blocks inside audit() and no such name anywhere. Docs that
# point at symbols which do not exist are worse than docs with no symbols: the
# reader concludes the code is elsewhere rather than that the name is fiction.
#
# `no_new` is also gone as a NAME, because it described the rule this one
# replaced: a scan of the sentence for any number not in the finding. That
# needed six exemptions in a row and still struck correct claims. The rule now
# checks only what the claim DECLARES as measured, which is a different thing
# and deserves a different word.
#
# Each returns the strike reason, or None to pass. First non-None wins, so the
# order below is the order a claim is judged in.
AUDIT_RULES = ("cites", "declared_values", "respects")


def rule_cites(claim: Dict[str, Any],
               by_id: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """The claim must name a finding that exists.

    PROVES: there is evidence behind the sentence, and it can be located.
    DOES NOT PROVE: that the finding supports the sentence.
    """
    fid = claim.get("finding_id")
    if fid in by_id:
        return None
    return (f"cites finding_id {fid!r}, which step 2 did not produce. A "
            f"claim with no evidence behind it cannot be checked.")


def rule_declared_values(claim: Dict[str, Any],
                         finding: Dict[str, Any],
                         allowed: set) -> Optional[str]:
    """Every value the claim declares as measured must be in that finding.

    THE CLAIM DECLARES ITS MEASUREMENTS; prose numbers are not audited. This
    replaced a scan of the sentence text, which needed six exemptions in a row
    — years, identifiers, labels, approximations, run facts, subset counts —
    each added after it struck a correct claim. There is no reliable way to
    tell "31.4 mm/yr" from "16 of 19 columns" or "band 2" by looking at the
    text. Six patches on one rule is the rule being wrong.

    `allowed` widens "in that finding" to facts of record: the column count,
    the simulated years, and every finding's own `n`. A claim quoting those is
    quoting the run, not inventing.

    PROVES: no fabricated measurement survives.
    DOES NOT PROVE: that the number means what the sentence says it means, or
    that it is the right number for the claim. A value used with the wrong
    variable, unit, period or sign passes this rule.
    """
    have = _flatten_numbers(finding.get("result"))
    declared = [str(v) for v in (claim.get("values") or [])]
    invented = [v for v in declared
                if v not in have and v not in allowed
                and not _YEARLIKE.match(v)]
    if not invented:
        return None
    return (f"declares {', '.join(invented[:4])} as measured, but "
            f"finding {claim.get('finding_id')} does not contain it. A value "
            f"that cannot be traced to the data it came from is not a "
            f"measurement.")


def rule_respects(claim: Dict[str, Any],
                  finding: Dict[str, Any],
                  caveats: List[Dict[str, Any]],
                  blocking: Dict[str, Any]) -> Optional[str]:
    """A claim about a blocking caveat's subject must carry that caveat's id.

    The FIGURE bounds what is possible (`finding["blocked_by"]`) and the CLAIM
    decides what applies — see required_caveats(). Only `blocking` severity is
    enforced; `qualify` and `context` travel to the report and bind nothing.

    PROVES: a claim inside a forbidden scope acknowledges it explicitly.
    DOES NOT PROVE: that the acknowledgement is meaningful, or that the claim
    is one the caveat permits at all — carrying the id satisfies this rule.
    """
    missing = [cid for cid in required_caveats(
                   str(claim.get("claim") or ""),
                   finding.get("blocked_by"), caveats)
               if cid in blocking and cid not in set(claim.get("caveats") or [])]
    if not missing:
        return None
    return (f"is about {', '.join(missing)}'s subject but does not carry "
            f"it. An unrespected caveat looks exactly like a respected "
            f"one, which is why they are records.")


def audit(claims: List[Dict[str, Any]], investigation: Dict[str, Any],
          caveats: List[Dict[str, Any]], facts: Optional[set] = None
          ) -> Dict[str, Any]:
    """Check each claim against the evidence it cites. Pure function.

    Returns {kept, struck} where every struck claim carries its reason AND the
    name of the rule that struck it. Testable with no API call, which is the
    point: the audit is the part that must not depend on a model behaving well.

    The three rules are `rule_cites`, `rule_declared_values` and
    `rule_respects`, applied in that order; the first to object wins, and a
    claim that passes all three is kept.
    """
    by_id = {f["id"]: f for f in (investigation.get("findings") or [])}
    allowed = set(facts or set())
    for f in (investigation.get("findings") or []):
        if f.get("n") is not None:
            allowed.add(str(f["n"]))
    blocking = {c.get("id"): c for c in (caveats or [])
                if c.get("severity") == "blocking"}

    kept, struck = [], []
    for c in claims:
        if not isinstance(c, dict):
            continue

        why = rule_cites(c, by_id)
        rule = "cites"
        if why is None:
            finding = by_id[c["finding_id"]]
            why, rule = rule_declared_values(c, finding, allowed), "declared_values"
        if why is None:
            why, rule = rule_respects(c, finding, caveats, blocking), "respects"

        if why is None:
            kept.append(c)
        else:
            # WHICH RULE, not only why. The reason is prose meant for a reader;
            # `struck_by` is what a test, a dashboard or a run-to-run
            # comparison can group on without parsing English.
            struck.append(dict(c, struck_because=why, struck_by=rule))

    return {"kept": kept, "struck": struck,
            "n_claims": len(claims), "n_struck": len(struck)}


# ─────────────────────────────────────────────────────────────────────
# THE PROMPT
# ─────────────────────────────────────────────────────────────────────
# A finding's result, whole if it fits. The old cap was 700 characters, and it
# was silent — the model was shown a fifth of a result and asked to quote from
# it exactly. Measured on the 2026-08-13 runs: `partition_fractions_vs_elevation`
# held 324 numbers and 292 of them were behind the cut; five of the ten findings
# across the two studies were truncated.
#
# 6000 is generous rather than principled: the largest result measured was 4244
# characters, so nothing real is cut today, and a runaway result still cannot
# swallow the prompt. When it DOES cut, it says so and forbids quoting from
# that finding — a model asked to copy exactly from evidence it cannot see is
# being set up to fail the audit.
RESULT_CAP = 6000


def _render_result(result) -> str:
    body = json.dumps(result, default=str)
    if len(body) <= RESULT_CAP:
        return body
    return (body[:RESULT_CAP] +
            f"  …CUT at {RESULT_CAP} of {len(body)} chars — DO NOT quote "
            f"numbers from this finding; cite one you can read in full")


PLAN_CAP = 8000


def _plan_text(investigation: Dict[str, Any]) -> str:
    """Step 2's plan document, or "" when there is none to read.

    Capped like a finding is, and for the same reason: a brief that silently
    truncates evidence is what taught a model to quote from text it could only
    partly see. When it cuts, it says so.
    """
    p = investigation.get("plan_md")
    if not p:
        return ""
    try:
        body = Path(p).read_text()
    except OSError:
        return ""
    if len(body) <= PLAN_CAP:
        return body
    return body[:PLAN_CAP] + f"\n…CUT at {PLAN_CAP} of {len(body)} chars."


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

    # ONE LIST, AND IT IS THE LIST THE AUDIT CHECKS AGAINST (2026-08-13).
    #
    # Step 1's comparison findings and step 2's investigation findings are
    # equally citable — `evidence["findings"]` in interpret() is the
    # concatenation of both — but this brief used to render them in two
    # different shapes: step 2's under `FINDINGS:` with an id, a question and a
    # result, step 1's as a separate prose block headed "WHAT THE COMPARISON
    # MEASURED". A model writing about snow then cited the step-2 finding whose
    # SUBJECT matched, `swe_snotel_validation`, while quoting numbers that live
    # in `compare_swe`.
    #
    # Measured on the 2026-08-13 verification runs: of 15 struck values, 8 sat
    # in a compare_* finding and 6 in a different step-2 finding. Every one of
    # them was a real measurement in this run, cited under the wrong id. The
    # audit was right every time; the prompt was what made the mistake easy.
    #
    # So the rule is now structural: whatever the audit will accept is listed
    # here, in one shape, under the id that has to be named.
    findings = list(investigation.get("findings") or []) + \
        list(comparison.get("findings") or [])

    # STEP 2'S PLAN, IN FULL. This used to be the single `notes` line, which
    # said what step 2 could not address and nothing about what it chose to do
    # or why. Judging whether a figure set answers the question means knowing
    # what it was trying to answer — a thin set is a different judgement when
    # the run is thin than when the plan was.
    #
    # Read from the FILE step 2 wrote, not passed through memory, so step 3 can
    # be re-run alone against an archived study and see the same thing. Absent
    # is not an error: a study analysed before 2026-08-14 has no plan document,
    # and the `notes` line is the fallback it always was.
    plan_md = _plan_text(investigation)
    if plan_md:
        lines += ["", "WHAT STEP 2 PLANNED AND WHY — its own account of how it",
                  "mapped the question onto the model's variables, with the",
                  "outcome of each step measured rather than claimed:",
                  "", plan_md, ""]
    else:
        lines += ["", "WHAT STEP 2 INVESTIGATED:",
                  f"    {investigation.get('notes')}", ""]

    lines += ["FINDINGS — every citable number in this run is below, under the",
              "id you must name. A value not in the finding you cite is struck,",
              "even when it is a real measurement from somewhere else:"]
    for f in findings:
        lines.append(f"    id: {f['id']}   (n={f.get('n')}, scale={f.get('scale')})")
        lines.append(f"        question: {f.get('question')}")
        lines.append(f"        result:   {_render_result(f.get('result'))}")
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

List in `values` every MEASURED quantity your claim asserts, copied exactly
from the cited finding's result — do not round, restate or convert. Counts you
made by reading a table, band or column labels, and thresholds you chose to
describe a pattern are NOT measurements and do not belong there. A declared
value absent from the finding is struck automatically.

CITE THE FINDING THE NUMBER IS IN, not the one whose subject matches. These
are different, and confusing them is the single commonest way a true sentence
gets struck here. Observation metrics — bias, RMSE, NSE, KGE, station counts —
live in the `compare_<observable>` findings; the step-2 findings hold what the
generated scripts computed from the model's own series. If your sentence pairs
one of each, split it into two claims, each citing where its numbers came from.
Before you list a value, find it in the finding you are about to name.

Judge whether the findings answer what the user asked. Decide:
  "sufficient"    the question is answered as well as this experiment allows
  "insufficient"  a revised set of figures would materially improve the answer

"insufficient" is not for wishing the experiment were different. If the
experiment cannot address the question, that is a SUFFICIENT answer of the form
"this cannot be determined from this run, because ...". Only ask for a revision
if step 2 could plausibly do better with the same data.

Return ONLY JSON:
{{"claims": [
    {{"claim": "<one sentence>",
      "finding_id": "<the finding it rests on>",
      "values": [<every MEASURED number this claim asserts, copied exactly from
                  the finding's result. NOT counts you made by reading a table,
                  NOT band or column labels, NOT thresholds you chose. Use []
                  if the claim asserts no measurement.>],
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
    blob = text[i:j + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # The `code` field carries generated Python — raw newlines inside
        # string literals, echoed comments, trailing commas — which naive
        # json.loads rejects. A live run died here on a reply that was
        # otherwise fine. llm_agent already solved this for the planner, so
        # reuse its DETERMINISTIC repairs rather than write a third parser.
        # Not parse_json_resilient: that adds an LLM self-repair round, and a
        # step that silently spends another call to fix its own output is a
        # step whose cost accounting lies.
        from agents.llm_agent import LLMAgent
        repaired = LLMAgent._escape_raw_newlines(
            LLMAgent._strip_json_comments(blob))
        return json.loads(repaired)


# The longest side a figure is sent at. Vision models resize anything larger
# before they look at it, so pixels above this are paid for and then discarded —
# and the comparison figures are 2700 px wide and up to 1.9 MB each, which at
# ten attachments is a 25 MB request that buys no extra detail.
MAX_IMAGE_PX = 1568


def _encoded(path: Path) -> Optional[tuple]:
    """(mime, base64) for one figure, downscaled if it is oversized.

    Falls back to the file as it stands when Pillow is absent: sending a large
    PNG costs bandwidth, and sending nothing costs the review its eyes.
    """
    import base64
    raw = path.read_bytes()
    try:
        import io
        from PIL import Image
        im = Image.open(io.BytesIO(raw))
        if max(im.size) > MAX_IMAGE_PX:
            scale = MAX_IMAGE_PX / max(im.size)
            im = im.resize((max(1, round(im.width * scale)),
                            max(1, round(im.height * scale))),
                           Image.LANCZOS)
        # White, not transparent: matplotlib saves RGBA, and a transparent
        # background composites to black in some viewers — an unreadable
        # figure the reviewer would report as a plotting bug.
        if im.mode in ("RGBA", "LA", "P"):
            bg = Image.new("RGB", im.size, "white")
            im = im.convert("RGBA")
            bg.paste(im, mask=im.split()[-1])
            im = bg
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=88, optimize=True)
        return "image/jpeg", base64.b64encode(buf.getvalue()).decode()
    except Exception:                                           # noqa: BLE001
        return "image/png", base64.b64encode(raw).decode()


def _content(brief: str, figures: List[str], with_images: bool):
    """The user turn: the brief, then each figure as an image block."""
    parts: List[Dict[str, Any]] = [{"type": "text", "text": brief + "\n" + TASK}]
    if not with_images:
        return brief + "\n" + TASK
    for p in figures:
        try:
            enc = _encoded(Path(p))
            if not enc:
                continue
            mime, b = enc
            parts.append({"type": "text", "text": f"figure: {Path(p).stem}"})
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:{mime};base64,{b}"}})
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
        client.label = "step3_interpret"      # so step 4 can attribute the spend

    comparison = comparison or {}
    brief = review_brief(ctx, comparison, investigation or {})

    # THE EVIDENCE IS BOTH STEPS'. Step 2's figures answer the user's question;
    # step 1's put the model beside an observation, which nothing step 2 draws
    # can do. The audit reads the same merged list, so a claim citing
    # `compare_water_table` is checked against the comparison record exactly as
    # citing a step-2 figure is checked against its result.
    evidence = dict(investigation)
    evidence["findings"] = list(investigation.get("findings") or []) + \
        list(comparison.get("findings") or [])

    figures = list(investigation.get("figures") or [])
    figures += [p for p in (comparison.get("figures") or {}).values()
                if isinstance(p, str) and Path(p).is_file()]
    content = _content(brief, figures, with_images)
    reply = client.ask([{"role": "user", "content": content}])
    spec = _parse(reply)

    # THE BRIEF, NOT `content`. `content` is the brief plus every figure
    # base64-encoded — tens of megabytes of image bytes that say nothing a
    # reader can use. The text is the part that decides what the model can
    # cite, and the part a replayed test needs.
    from agents.analysis import script_runner as _runner
    exchange = _runner.save_exchange(
        out_dir, "step3", investigation.get("round", 1) if investigation else 1,
        brief, reply)

    caveats = list(ctx.caveats or []) + list(comparison.get("caveats") or [])
    result = audit(spec.get("claims") or [], evidence, caveats,
                   facts=run_facts(ctx))

    verdict = spec.get("verdict")
    if verdict not in ("sufficient", "insufficient"):
        verdict = "insufficient"          # an unparseable verdict is not a pass

    out = {"answer": spec.get("answer"), "verdict": verdict,
           "feedback": spec.get("feedback") or None,
           "claims": result["kept"], "struck": result["struck"],
           "audit": {k: result[k] for k in ("n_claims", "n_struck")},
           "exchange": exchange or None,
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
        # WHAT THIS ROUND ACTUALLY PRODUCED, by name. A later round REPLACES
        # the investigation wholesale — `investigation` is reassigned and only
        # the last one reaches step 4 — so a round 2 that drops three of round
        # 1's five findings loses them with nothing recording it. The prompt
        # tells the model "keep what worked"; whether it did was unknowable.
        ids = sorted(f["id"] for f in (investigation.get("findings") or []))
        kept_claims = (interpretation["audit"]["n_claims"]
                       - interpretation["audit"]["n_struck"])
        entry = {"round": r,
                 "verdict": interpretation["verdict"],
                 "n_findings": investigation["n_succeeded"],
                 "finding_ids": ids,
                 "n_claims": interpretation["audit"]["n_claims"],
                 "n_struck": interpretation["audit"]["n_struck"],
                 "n_claims_kept": kept_claims,
                 "feedback": interpretation.get("feedback")}
        if rounds:
            prev = rounds[-1]
            dropped = sorted(set(prev["finding_ids"]) - set(ids))
            if dropped:
                entry["findings_dropped_from_previous_round"] = dropped
            # A REVISION THAT MADE IT WORSE. Not corrected here — the reviewer
            # judged this round's set and that judgement stands — but a reader
            # comparing two runs needs to know the extra round cost claims
            # rather than earning them.
            if kept_claims < prev["n_claims_kept"]:
                entry["regressed"] = (
                    f"round {r} kept {kept_claims} claims against round "
                    f"{prev['round']}'s {prev['n_claims_kept']}")
        rounds.append(entry)

        if interpretation["verdict"] == "sufficient":
            break
        feedback = interpretation.get("feedback")
        if not feedback:                  # insufficient with nothing actionable
            break                         # to say is not worth another round

    regressed = [r_["round"] for r_ in rounds if r_.get("regressed")]
    return {"investigation": investigation, "interpretation": interpretation,
            "rounds": rounds, "n_rounds": len(rounds),
            "regressed_rounds": regressed or None,
            "stopped_because": ("sufficient"
                                if interpretation["verdict"] == "sufficient"
                                else f"exhausted {len(rounds)} of {max_rounds} rounds")}
