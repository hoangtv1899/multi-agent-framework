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
    percent     (a study with a suite, 2026-11-08) a claim giving a percent of a
                field whose notes give none is struck, by the evidence trace's
                own rule (claim_trace.percent_problems).

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

from agents.analysis.dashes import strip_dashes, strip_dashes_in

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
    if getattr(ctx, "is_domain", False):
        return _domain_facts(ctx)
    try:
        out.add(str(len(ctx.columns)))
        p = (ctx.plan or {}).get("period") or {}
        for k in ("yr_start", "yr_end"):
            if p.get(k) is not None:
                out.add(str(p[k]))
    except Exception:
        pass
    return out


def _domain_facts(ctx) -> set:
    """A whole-domain experiment's facts of record: each region's cell count
    and area, the model years, the normal years, and how many regions and
    runs there are. NOT the number of run-year rows in ctx.columns: that is a
    count of records, not a fact about the domain, and a claim quoting it as
    one would be describing the bookkeeping."""
    out = set()
    try:
        t = (ctx.data or {}).get("regional") or {}
        for info in (t.get("regions") or {}).values():
            out |= _flatten_numbers({k: (info or {}).get(k) for k in ("cells", "area_km2")})
        out |= _flatten_numbers(t.get("years") or [])
        out |= _flatten_numbers((t.get("event") or {}).get("normal_years") or [])
        out.add(str(len(t.get("regions") or {})))
        out.add(str(len(t.get("runs") or {})))
    except Exception:                                           # noqa: BLE001
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
AUDIT_RULES = ("cites", "declared_values", "respects", "percent")


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


def rule_percent(claim: Dict[str, Any], finding: Dict[str, Any],
                 records: Optional[Dict[str, Dict[str, Any]]]) -> Optional[str]:
    """A claim giving a percent of a field whose notes give none is struck
    (fix 5 of G part 2, 2026-11-08): question 1's audit kept "BTRAN 20.2
    percent lower", the script's own arithmetic on a bounded index. The rule
    is the evidence trace's own (claim_trace.percent_problems), judged on the
    records the claim names, or, when it names none, on the records of its
    finding whose field its text names (by code or by the field note's plain
    name). A study with no evidence records (no suite) is not judged here.

    PROVES: a percent of a field the notes forbid one for is not printed.
    DOES NOT PROVE: that a percent of an allowed field is the right one.
    """
    if not records:
        return None
    from agents.analysis import claim_trace as _ct
    text = str(claim.get("claim") or "")
    ids = [str(x) for x in (claim.get("evidence") or [])]
    if ids:
        recs = [records[i] for i in ids if i in records]
    else:
        notes = _ct._elm_modules()[1].load_notes()
        named = _ct.field_mentions(text, notes)
        mine = [str(x.get("id")) for x in finding.get("lookups") or [] if isinstance(x, dict)]
        recs = [records[i] for i in mine if i in records and records[i].get("field") in named]
    probs = _ct.percent_problems(text, recs)
    if not probs:
        return None
    return (f"gives {probs[0]}. A percent the field's notes forbid is struck here as it fails "
            f"the evidence trace.")


def _tracer(ctx, records):
    """The evidence trace for one claim of a study with a suite, on the inputs
    step 4's gate uses; None for any other study, or when it cannot be set up
    (the gate then decides, as before)."""
    if records is None:
        return None
    try:
        from agents.analysis import analysis_plan as _plan, claim_trace as _ct
        from agents.analysis.step4_report import suite_regions
        return _ct.claim_tracer(_plan.analysis_dir(ctx), study_dir=ctx.run_dir,
                                regions=suite_regions(ctx))
    except Exception:                                               # noqa: BLE001
        return None


def rule_trace(claim: Dict[str, Any], tracer) -> Optional[str]:
    """A claim the evidence trace fails is struck, with the trace's reasons
    (claim_trace.claim_tracer; 2026-11-10). None without a tracer: a study
    with no suite has no evidence records to trace to."""
    if tracer is None:
        return None
    try:
        problems = tracer(claim)
    except Exception as e:                                          # noqa: BLE001
        problems = [f"it could not be traced ({type(e).__name__}: {e})"]
    return ("it fails the evidence trace: " + "; ".join(str(p) for p in problems)) \
        if problems else None


def audit(claims: List[Dict[str, Any]], investigation: Dict[str, Any],
          caveats: List[Dict[str, Any]], facts: Optional[set] = None,
          records: Optional[Dict[str, Dict[str, Any]]] = None,
          tracer=None) -> Dict[str, Any]:
    """Check each claim against the evidence it cites. Pure function.

    Returns {kept, struck} where every struck claim carries its reason AND the
    name of the rule that struck it. Testable with no API call, which is the
    point: the audit is the part that must not depend on a model behaving well.

    The rules are `rule_cites`, `rule_declared_values`, `rule_respects` and,
    for a study with a suite (`records`, its evidence records), `rule_percent`
    and `rule_trace` (`tracer`), applied in that order; the first to object
    wins, and a claim that passes them all is kept.
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
            why, rule = rule_percent(c, finding, records), "percent"
        if why is None:
            why, rule = rule_trace(c, tracer), "trace"

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


def domain_reading(ctx) -> List[str]:
    """For a whole-domain experiment: which fields are stocks averaged over
    the year and which are totals over it, read off the regional table, and
    what a difference in each may be called. [] for any other study.

    ADDED 2026-10-25 after the fire-off twin's headline said the model's fire
    "removes" 23.2 gC/m2 of vegetation carbon in 2016: a gap between two
    yearly-mean stocks, which carries 2015's fire too, while the fire took
    7.92 that year. The audit proves every number is real; it cannot tell a
    stock from a flux, so the brief says which is which."""
    if not getattr(ctx, "is_domain", False):
        return []
    t = (ctx.data or {}).get("regional") or {}
    how: Dict[str, str] = {}
    for r in (t.get("runs") or {}).values():
        for v, meta in ((r or {}).get("fields") or {}).items():
            how.setdefault(v, (meta or {}).get("how") or "")
    for name in (t.get("observed") or {}):
        how.setdefault(name, "sum")
    stocks = sorted(v for v, h in how.items() if h == "mean")
    totals = sorted(v for v, h in how.items() if h == "sum")
    if not (stocks or totals):
        return []
    out = ["", "THE MODEL'S OUTPUT ONLY: every number of this experiment is the model's.",
           "    Never use an observation to explain, support or rule out a result. A",
           "    caveat that says the run does not contain an observed event is said as",
           "    that, and nothing is inferred from it.",
           "", "HOW TO READ A NUMBER OF THIS EXPERIMENT:"]
    if stocks:
        out += [f"    stocks, averaged over the year: {', '.join(stocks)}",
                "        A difference in one, between runs or between a year and the normal,",
                "        is how much larger or smaller the stock is. Never call it an amount",
                "        added, removed, emitted or lost in that year: a gap that grows from",
                "        one year to the next can carry the earlier years too."]
    if totals:
        out += [f"    totals over the year: {', '.join(totals)}",
                "        An amount moved in that year."]
    # MONTHS AND SEASONS (2026-11-03): the same rule over a month or a season
    if any((r or {}).get("monthly") for r in (t.get("runs") or {}).values()):
        out += ["    over a month or a season, the same rule: a stock's value is its mean",
                "        over that month or season, a total's the amount moved in it. A claim",
                "        about a month or a season is read from that month or season, never",
                "        from the year's value."]
    return out


def suite_records(ctx) -> Optional[Dict[str, Dict[str, Any]]]:
    """The evidence records of a study with a suite (claim_trace.
    evidence_records of its own lookups log, the records the trace checks),
    or None for any other study."""
    from agents.analysis import analysis_plan as _plan
    if not (getattr(ctx, "is_domain", False) and _plan.suite_bearing(ctx)):
        return None
    from agents.analysis import claim_trace as _ct
    return _ct.evidence_records(_plan.analysis_dir(ctx))


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
    lines += domain_reading(ctx)

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
    # A STUDY WITH A SUITE: EACH FINDING'S EVIDENCE RECORDS (fix 1 of G part 2,
    # 2026-11-08), the ones the evidence trace checks, so a claim can name
    # them; other studies' briefs are unchanged byte for byte.
    records = suite_records(ctx)
    for f in findings:
        lines.append(f"    id: {f['id']}   (n={f.get('n')}, scale={f.get('scale')})")
        lines.append(f"        question: {f.get('question')}")
        lines.append(f"        result:   {_render_result(f.get('result'))}")
        if records is not None:
            from agents.analysis import claim_trace as _ct
            lines += _ct.finding_evidence_lines(f, records)
    if records is not None:
        # THE PLAN'S EXPECTATIONS (2026-11-08): what a tested claim names
        from agents.analysis import analysis_plan as _ap
        from agents.analysis import claim_trace as _ct
        lines += _ct.expectation_lines(_ap.load_plan(_ap.analysis_dir(ctx)), records)
    if investigation.get("caveats"):
        lines.append("")
        lines.append("FIGURES THAT FAILED TO PRODUCE A RESULT:")
        for c in investigation["caveats"]:
            lines.append(f"    {str(c.get('statement'))[:200]}")
    return "\n".join(lines)


TASK_BODY = f"""\
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

WRITE FOR A READER WHO WILL PUT THIS ON A SLIDE. Every sentence you return
is read aloud to people who do not know this model. The rules, checked by code
after you answer, and a sentence that breaks one is sent back once:
  - Sentences of at most 30 words. One idea per sentence.
  - Plain words. Not "parameterisation", "bimodal", "diagnosed",
    "quasi-steady"; say what happens instead ("the model's drainage rule",
    "two groups of columns", "the water table sits below the soil").
  - A variable code is never bare. The first time it appears, gloss it:
    "sub-surface drainage (QDRAI)", "aquifer recharge (QCHARGE)", "water
    table depth (ZWT)". After that the plain words alone are fine.
  - No column ids ("col_11"). Say what distinguishes the column: "the highest
    column, at 1913 m", "10 of 19 columns".
  - Numbers in a SENTENCE are rounded to 2 or 3 significant figures with a
    unit or a percent sign ("37 percent of precipitation", "805 mm"). The
    exact measured value goes in `values`, unrounded; the audit reads
    `values`, the reader reads the sentence.
  - No dashes as punctuation. Use a comma, a colon or a new sentence.
  - Every figure you write with a unit or a percent sign is one of that
    claim's `values`, rounded; a percent is a value times 100. "96 to 100
    percent" over values of 0.991 to 0.999 is a wrong sentence, and the gate
    will send it back. Design labels ("the 5 m column", "500 mm of rain")
    are not values and are fine as integers.
  - Say "compared with observations", never "validated" or "validation":
    a 1-D column is compared with observations, never validated by them.
  - `headline`: ONE sentence of at most 25 words that answers the question
    with its key number, the sentence a slide title could carry. If the
    experiment cannot answer, the headline says so plainly.
  - THE HEADLINE SAYS NO MORE THAN ONE CLAIM SAYS. It restates one of your
    audited claims with that claim's scope: the same count of columns, the
    same numbers, the same qualifier. A claim about four columns does not
    become a headline about every column; "reached the water table in four
    of 17 columns" may not become "reached the water table". If no single
    claim carries the answer, the headline names the strongest claim and
    stops there.
  - CAUSES REST ON NUMBERS. A cause ("because", "so", "drove", "led to", or
    a mechanism such as leaves closing their pores or a shorter growing
    season) is written only where a field in the numbers you were given
    shows it, for that region and year. A cause no field shows is left out,
    or said only as possible ("may").
  - `answer`: 2 to 4 further sentences, same rules, in this order: what was
    found, why as far as the numbers show it, what it does not show. Each
    sentence restates a claim or a caveat you listed, with its scope; a
    sentence that says more than any claim says is not allowed. The reader
    will paste the headline and the answer onto a slide unread, so the
    audit of the claims must cover them.

"""


# THE REPLY TEMPLATE OF A STUDY WITHOUT A SUITE, unchanged byte for byte
# (2026-11-08): a study with a suite gets claim_trace's rendered template
# instead (suite_task), and this one stays the columns', PFLOTRAN's and a
# whole-domain study's without a suite.
TASK_REPLY = """\
Return ONLY JSON:
{"claims": [
    {"claim": "<one sentence>",
      "finding_id": "<the finding it rests on>",
      "values": [<every MEASURED number this claim asserts, copied exactly from
                  the finding's result. NOT counts you made by reading a table,
                  NOT band or column labels, NOT thresholds you chose. Use []
                  if the claim asserts no measurement.>],
      "caveats": ["<ids of any blocking caveat this claim falls under>"]}
  ],
  "headline": "<ONE plain sentence, at most 25 words, with the key number, "
              "saying no more than one of the claims above says>",
  "answer": "<2-4 plain sentences: what was found, why, what it does not show; "
            "or saying plainly that this experiment cannot answer it and why>",
  "verdict": "sufficient" | "insufficient",
  "feedback": "<if insufficient: what step 2 should do differently. Be "
              "specific about which figure and what is wrong with it. Empty "
              "if sufficient.>"}
"""
TASK = TASK_BODY + TASK_REPLY


# A STUDY WITH A SUITE (package F1, 2026-11-05): each claim names its
# evidence records, its class and the warnings it carries, and the report is
# withheld when any claim fails the evidence trace (claim_trace.trace_evidence).
# ITS TEMPLATE IS RENDERED (the user's decision on G part 2, 2026-11-08): the
# field list SUITE_CLAIMS used to give sat after a generic "Return ONLY JSON"
# block that named only claim, finding_id, values and caveats, and question 1's
# model followed the block. suite_task() is TASK_BODY, these rules, the
# classes, the warning codes and the reply template, the last three rendered
# from claim_trace's definitions.
SUITE_RULES = """
THIS STUDY HAS A SUITE. Under each finding above are its evidence records (the
calls its figure made on `suite`, each with an id such as "L7") and its result
numbers, split into values of those records and numbers the script made
itself. Every claim also gives "evidence" (the records it rests on),
"evidence_class" and "warnings" (every warning code its records carry; a claim
that leaves one out fails), as the template at the end shows.
Each number a claim states is a value of its records; a number the script made
itself is a value of no record, and a claim stating it fails. A year, month,
season or region a claim names is one its records name. A timing shift whose
record says "range rule: applies" is stated with the default, the smallest and
the largest shift across the settings, each with its setting (as "... days at
the default fraction 0.2, ... days at fraction 0.3"). No causal word except "in
the model, the change caused" for in-model attribution; no "caused by" in any
form; no percent of a field whose notes give none (each record says whether
its field allows one). THE HEADLINE IS NOT YOURS TO WRITE IN THIS STUDY, and the
`headline` rules above do not apply: give "headline_rank", every claim's number
once, the claim that answers the question most directly first. The report's
code makes its headline the text of the first of them that traces and that the
question type allows: for a hypothesis question a tested expectation resting
on no lookup found by screening, not planned or made after the close; for an
exploratory one any claim, labelled exploratory by the report. If none is
allowed, and the report is otherwise published, its headline says plainly that
the run is inconclusive. When a cause stays open, give "open_causes": the run
that would settle it, from the server's list below.
"""


def suite_task() -> str:
    """Step 3's task for a study with a suite: TASK_BODY, SUITE_RULES, the
    evidence classes, the warning codes and the reply template, the last
    three rendered from claim_trace (CLASS_NEEDS and CLASS_WORDING, WARNINGS,
    CLAIM_PARTS and REPLY_PARTS)."""
    from agents.analysis import claim_trace as _ct
    return (TASK_BODY + SUITE_RULES + "\n"
            + "\n".join(_ct.class_lines() + [""] + _ct.warning_lines() + [""]
                        + _ct.run_lines() + [""] + _ct.claims_reply_lines()) + "\n")


# ─────────────────────────────────────────────────────────────────────
# READABILITY GATE (2026-09-12)
# ─────────────────────────────────────────────────────────────────────
# The audit above proves a claim's numbers are real. It says nothing about
# whether a person can read the sentence, and the answer paragraph is not
# audited at all. A live consequence: an answer of one 120-word sentence,
# stuffed with variable codes, column ids and "parameterisation", was pasted
# onto a slide and presented as it stood. The numbers were right and the room
# understood none of it.
#
# So the wording is gated the way the numbers are: by code, not by hoping the
# prompt was read. The checks are mechanical and listed below; a text that
# fails is sent back ONCE with the exact problems, wording only, `values` and
# `finding_id` untouched, and then audited exactly as before. A text that
# still fails after the rewrite is kept and its problems are recorded on the
# interpretation, so the report can say so rather than pretend.
MAX_SENTENCE_WORDS = 30
MAX_HEADLINE_WORDS = 25
# Acronyms a reader of this field knows, and proper names. Not variable codes.
PLAIN_ALLOWED = {
    "ELM", "PFLOTRAN", "USGS", "SNOTEL", "NLDAS", "CONUS", "CONUS2", "HUC",
    "HUC8", "DEM", "SSURGO", "GLHYMPS", "MODIS", "NSE", "KGE", "RMSE", "MAE",
    "SWE", "ET", "WTD", "LLM", "MCP", "SLURM", "USA", "WA", "CO", "PA", "DE",
    "ID", "JSON", "CSV", "MM", "NA", "N", "S", "E", "W", "II", "III", "IV",
    "AND", "OR", "NOT", "CUT", "DO", "PA",
}
JARGON = ("parameterisation", "parameterization", "bimodal", "diagnosed",
          "quasi-steady", "heteroscedastic", "ergodic", "stochastically",
          "ansatz")
_CODE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
# NUMBERS IN A SENTENCE MUST BE THE CLAIM'S OWN VALUES (2026-09-13). The
# audit reads `values`; it never reads the sentence. A live case: a claim
# said "96 to 100 percent" over values of 0.991 to 0.999 and passed. So a
# percent figure, and any decimal figure with a unit, must match one of the
# claim's values (a percent may match a value times 100) at the precision
# of whichever is coarser. Integers with a unit are left alone: "the 5 m
# column" and "500 mm of rain" are design labels, not measurements, and the
# task already tells the model that labels do not belong in `values`. The
# headline and the answer are held to the union of every claim's values.
_UNIT = (r"(?:mm per day|mm/day|square kilometres|saturation units|percent|%|"
         r"km2|km|mm|cm|m|days?|years?)")
_NUM = r"-?\d+(?:\.\d+)?"
_NUMBERS_WITH_UNIT = re.compile(
    rf"((?:{_NUM}\s*(?:,|to|and|or)\s*)*{_NUM})\s*({_UNIT})(?![A-Za-z0-9])")
_BANNED = (
    (re.compile(r"\bvalidat\w*", re.I),
     'say "compared with observations" or "comparison"; a 1-D column is '
     'compared with observations, never validated by them'),
)


def _decimals(x) -> int:
    """Digits after the point as written; an integral value has none."""
    t = x if isinstance(x, str) else repr(float(x))
    if "e" in t or "." not in t:
        return 0
    frac = t.split(".")[1].rstrip("0")
    return len(frac)


def same_number(written, measured) -> bool:
    """`written` (as it appears in a sentence) states `measured` at the
    precision of whichever is coarser: 37 states 0.368 as a percent, 296.3
    states 296, 0.011 states 0.0105; 96 does not state 0.991."""
    a, b = abs(float(written)), abs(float(measured))
    for coarse in (_decimals(written), _decimals(measured)):
        if abs(a - b) <= 0.5 * 10.0 ** (-coarse) + 1e-9:
            return True
    return False


def number_problems(text: str, values, where: str, any_claim: bool = False):
    """Every figure in `text` that no value backs, as a sentence to act on."""
    vals = []
    for v in values or []:
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    problems = []
    for group, unit in _NUMBERS_WITH_UNIT.findall(str(text or "")):
        percent = unit in ("percent", "%")
        for num in re.findall(_NUM, group):
            if not percent and "." not in num:
                continue                                  # a label, not a value
            cands = vals + ([v * 100.0 for v in vals] if percent else [])
            if any(same_number(num, v) for v in cands):
                continue
            shown = ", ".join(f"{v:g}" for v in vals) or "none"
            problems.append(
                f"{where}: \"{num} {unit}\" is not "
                + ("among the values of any claim; the headline and the "
                   "answer state only what a claim states"
                   if any_claim else
                   f"one of this claim's values ({shown}); state the measured "
                   f"value, rounded to 2 or 3 significant figures"))
    for pat, advice in _BANNED:
        for w in sorted(set(m.group(0) for m in pat.finditer(str(text or "")))):
            problems.append(f"{where}: \"{w}\": {advice}")
    return problems
_COL_ID = re.compile(r"\bcol_\d+\b")
_LONG_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])-?\d*\.\d{5,}\b")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"])")


# MOVED TO agents.analysis.dashes (2026-11-08), one helper for every record
# that keeps the model's words; the old name stays.
_strip_dashes = strip_dashes


def _sentences(text: str):
    return [x.strip() for x in _SENTENCE_END.split(str(text or "").strip())
            if x.strip()]


def glossed_codes(texts) -> set:
    """Variable codes that appear in a plain-language gloss, "(CODE)", anywhere."""
    out = set()
    for t in texts:
        out |= set(re.findall(r"\(([A-Z][A-Z0-9_]{2,})\)", str(t or "")))
    return out


def readability_problems(text: str, glossed: set, where: str,
                         max_words: int = MAX_SENTENCE_WORDS,
                         needs_number: bool = False):
    """Every way `text` fails the slide-reader rules, as sentences a model can act on."""
    t = str(text or "")
    problems = []
    if not t.strip():
        return [f"{where}: empty"]
    for sen in _sentences(t):
        n = len(sen.split())
        if n > max_words:
            problems.append(f"{where}: a {n}-word sentence; the limit is "
                            f"{max_words}. Split it: \"{sen[:70]}...\"")
    for code in sorted(set(_CODE.findall(t))):
        if code in PLAIN_ALLOWED or code in glossed:
            continue
        problems.append(f"{where}: {code} appears without a plain-language "
                        f"gloss; write it once as \"<plain words> ({code})\"")
    for cid in sorted(set(_COL_ID.findall(t))):
        problems.append(f"{where}: refers to {cid} by id; say what "
                        f"distinguishes that column (its elevation, how wet "
                        f"it is, its water table) or how many columns")
    for w in JARGON:
        if re.search(rf"\b{re.escape(w)}\b", t, re.I):
            problems.append(f"{where}: \"{w}\" is jargon; say what it means")
    for num in _LONG_NUMBER.findall(t):
        problems.append(f"{where}: {num} has too many digits for a sentence; "
                        f"round to 2 or 3 significant figures (the exact value "
                        f"belongs in `values`)")
    if needs_number and not re.search(r"\d", t):
        problems.append(f"{where}: carries no number; the headline states "
                        f"the key result with its value and unit")
    return problems


def check_readability(spec: Dict[str, Any], headline: bool = True):
    """All problems in a parsed reply's wording: headline, answer, every
    claim. A study with a suite writes no headline (`headline` False): its
    report's headline is a claim's own text, gated as that claim."""
    claims = [c for c in (spec.get("claims") or []) if isinstance(c, dict)]
    texts = [spec.get("headline"), spec.get("answer")] + \
        [c.get("claim") for c in claims]
    glossed = glossed_codes(texts)
    cannot = bool(re.search(r"cannot (be )?(answer|determin)",
                            str(spec.get("headline") or "") +
                            str(spec.get("answer") or ""), re.I))
    problems = readability_problems(spec.get("headline"), glossed, "headline",
                                    max_words=MAX_HEADLINE_WORDS,
                                    needs_number=not cannot) if headline else []
    problems += readability_problems(spec.get("answer"), glossed, "answer")
    for i, c in enumerate(claims, 1):
        problems += readability_problems(c.get("claim"), glossed,
                                         f"claim {i}")
    every = [v for c in claims for v in (c.get("values") or [])]
    if headline:
        problems += number_problems(spec.get("headline"), every, "headline", True)
    problems += number_problems(spec.get("answer"), every, "answer", True)
    for i, c in enumerate(claims, 1):
        problems += number_problems(c.get("claim"), c.get("values"),
                                    f"claim {i}")
    return problems


REWRITE = """\
The wording below failed the slide-reader rules listed after it. Rewrite ONLY
the wording. Do not change, add, remove or re-round any number that carries
meaning; do not add a measurement; keep every claim's meaning and its order.
The one exception: where a PROBLEM says a figure is not among the values,
replace that figure with the value the problem names, rounded to 2 or 3
significant figures, or drop the sentence that carries it.
A variable code stays present but glossed once, as "<plain words> (CODE)".
The headline and every sentence of the answer say no more than one of the
claims says: the same count of columns, the same numbers, the same qualifier.
Say "compared with observations", never "validated".

PROBLEMS:
{problems}

TEXT:
{text}

Return ONLY JSON: {{"headline": "...", "answer": "...",
                    "claims": ["<claim 1 wording>", "<claim 2 wording>", ...]}}
with exactly {n} claims in the same order.
"""


def rewrite_wording(spec: Dict[str, Any], problems, client,
                    headline: bool = True) -> Dict[str, Any]:
    """One rewrite of the wording, `values` and `finding_id` untouched; a
    study with a suite (`headline` False) keeps no headline the rewrite
    gives."""
    claims = [c for c in (spec.get("claims") or []) if isinstance(c, dict)]
    text = json.dumps({"headline": spec.get("headline"),
                       "answer": spec.get("answer"),
                       "claims": [c.get("claim") for c in claims]}, indent=1)
    msg = REWRITE.format(problems="\n".join(f"  - {p}" for p in problems),
                         text=text, n=len(claims))
    reply = client.ask([{"role": "user", "content": msg}])
    new = _parse(reply)
    words = new.get("claims") or []
    out = dict(spec)
    out["headline"] = (_strip_dashes(new.get("headline") or spec.get("headline"))
                       if headline else spec.get("headline"))
    out["answer"] = _strip_dashes(new.get("answer") or spec.get("answer"))
    out["claims"] = [dict(c, claim=_strip_dashes(words[i]))
                     if i < len(words) and words[i] else c
                     for i, c in enumerate(claims)]
    return out


def gate_wording(spec: Dict[str, Any], client, headline: bool = True) -> Dict[str, Any]:
    """Dashes stripped, then the checks, then at most one rewrite.

    Returns the spec to audit plus a `readability` record: what was wrong
    before, whether a rewrite was asked for, and what is still wrong after.
    A study with a suite (`headline` False) has no headline to check.
    """
    spec = dict(spec)
    spec["headline"] = _strip_dashes(spec.get("headline")) if headline else None
    spec["answer"] = _strip_dashes(spec.get("answer"))
    spec["claims"] = [dict(c, claim=_strip_dashes(c.get("claim")))
                      if isinstance(c, dict) else c
                      for c in (spec.get("claims") or [])]
    before = check_readability(spec, headline)
    record = {"problems_before": before, "rewritten": False,
              "problems_after": before}
    if not before:
        return spec, record
    try:
        spec2 = rewrite_wording(spec, before, client, headline)
    except Exception as e:                                       # noqa: BLE001
        record["rewrite_error"] = str(e)[:200]
        return spec, record
    record["rewritten"] = True
    record["problems_after"] = check_readability(spec2, headline)
    return spec2, record


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


def _content(brief: str, figures: List[str], with_images: bool, task: str = TASK):
    """The user turn: the brief and the task, then each figure as an image
    block. `task` is TASK for every study without a suite, suite_task() for
    one with a suite."""
    parts: List[Dict[str, Any]] = [{"type": "text", "text": brief + "\n" + task}]
    if not with_images:
        return brief + "\n" + task
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


VERDICT_ASK = """\
Your reply above gave no verdict. Judge it as the instructions asked and return
ONLY JSON: {"verdict": "sufficient" | "insufficient",
            "feedback": "<if insufficient: what step 2 should do differently>"}
"""


def _ask_verdict(client, content, reply) -> Dict[str, Any]:
    """The verdict alone, asked once after a reply that gave none.

    Returns the fields to merge into the spec: `verdict_asked_again` always,
    and `verdict` with its `feedback` only when the answer is one of the two
    allowed. Any failure returns just the marker, so the caller's default
    ("insufficient") stands.
    """
    out: Dict[str, Any] = {"verdict_asked_again": True}
    try:
        again = _parse(client.ask([{"role": "user", "content": content},
                                   {"role": "assistant", "content": reply},
                                   {"role": "user", "content": VERDICT_ASK}]))
    except Exception:                                            # noqa: BLE001
        return out
    if again.get("verdict") in ("sufficient", "insufficient"):
        out["verdict"] = again["verdict"]
        out["feedback"] = again.get("feedback") or None
    return out


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
    # A STUDY WITH A SUITE: its records under each finding (in the brief) and
    # its own task, whose template asks for evidence, evidence_class,
    # expectation, warnings and headline (fix 1 of G part 2, 2026-11-08)
    records = suite_records(ctx)
    task = suite_task() if records is not None else TASK
    if records is not None:
        # WHAT THE RECIPE'S OUTPUTS ARE FOR (analysis_recipes, 2026-11-09): only
        # for the parts the plan carried out, after the task
        from agents.analysis import analysis_plan as _plan, analysis_recipes as _ar
        said = _ar.report_lines((ctx.plan or {}).get("question"),
                                _plan.load_plan(_plan.analysis_dir(ctx)))
        task += ("\n" + "\n".join(said) + "\n") if said else ""

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
    content = _content(brief, figures, with_images, task)
    reply = client.ask([{"role": "user", "content": content}])
    # A REPLY THAT CANNOT BE READ IS SAVED FIRST (2026-11-08), cut off at its
    # output limit or not JSON, then the step fails with what happened
    try:
        spec = _parse(reply)
    except (ValueError, json.JSONDecodeError) as e:
        from agents.analysis import script_runner as _sr
        saved = _sr.save_exchange(out_dir, "step3", investigation.get("round", 1)
                                  if investigation else 1, brief, reply)
        cut = getattr(client, "last_finish_reason", None) == "length"
        raise ValueError(("step 3's reply was cut off at its output limit" if cut else
                          f"step 3's reply is not readable JSON ({e})")
                         + (f"; it is saved as "
                            f"{', '.join(sorted(Path(v).name for v in saved.values()))}"
                            if saved else "")) from None
    # A REPLY WITH NO VERDICT IS ASKED ONCE MORE, FOR THE VERDICT ALONE
    # (2026-10-27). A live whole-domain run returned a complete answer whose
    # every claim the audit kept, but no `verdict` key. Read as "insufficient",
    # it reached the user labelled insufficient, with no feedback to drive a
    # second round. One short follow-up now asks for the verdict on the reply
    # just given; a reply that still gives none stays "insufficient" below.
    if spec.get("verdict") not in ("sufficient", "insufficient"):
        spec = dict(spec, **_ask_verdict(client, content, reply))
    # THE WORDING IS GATED LIKE THE NUMBERS. Deterministic checks, one
    # bounded rewrite of the wording alone, then the audit as before.
    spec, readability = gate_wording(spec, client, headline=records is None)

    # THE BRIEF, NOT `content`. `content` is the brief plus every figure
    # base64-encoded — tens of megabytes of image bytes that say nothing a
    # reader can use. The text is the part that decides what the model can
    # cite, and the part a replayed test needs.
    #
    # A study with a suite keeps its own task beside the brief too: it holds
    # the template the model was asked to follow, which the replay of a
    # recorded study needs. Every other study's record is the brief alone, as
    # it was.
    from agents.analysis import script_runner as _runner
    exchange = _runner.save_exchange(
        out_dir, "step3", investigation.get("round", 1) if investigation else 1,
        brief if records is None else brief + "\n" + task, reply)

    caveats = list(ctx.caveats or []) + list(comparison.get("caveats") or [])
    result = audit(spec.get("claims") or [], evidence, caveats,
                   facts=run_facts(ctx), records=records, tracer=_tracer(ctx, records))

    verdict = spec.get("verdict")
    if verdict not in ("sufficient", "insufficient"):
        verdict = "insufficient"          # an unparseable verdict is not a pass

    # THE HEADLINE RANKING, ON THE CLAIMS THE AUDIT KEPT (2026-11-08): a
    # study with a suite writes no headline; its ranking, given by the
    # reply's claim numbers, is carried onto the kept claims' numbers
    ranked = None
    if records is not None:
        given = spec.get("headline_rank") if isinstance(spec.get("headline_rank"), list) else []
        said = [c for c in (spec.get("claims") or [])]
        kept_at = {id(c): j for j, c in enumerate(result["kept"], 1)}
        ranked = [kept_at[id(said[k - 1])] for k in given
                  if isinstance(k, int) and not isinstance(k, bool) and 1 <= k <= len(said)
                  and id(said[k - 1]) in kept_at]
    out = {"headline": spec.get("headline") if records is None else None,
           **({"headline_rank": ranked,
               "headline_rank_as_given": spec.get("headline_rank")}
              if records is not None else {}),
           "answer": spec.get("answer"), "verdict": verdict,
           "verdict_asked_again": bool(spec.get("verdict_asked_again")),
           "readability": readability,
           # the model's words kept in the record carry no dashes (fix 6)
           "feedback": strip_dashes(spec.get("feedback")) or None,
           "open_causes": strip_dashes_in(spec.get("open_causes") or []),
           "claims": result["kept"], "struck": result["struck"],
           "audit": {k: result[k] for k in ("n_claims", "n_struck")},
           "exchange": exchange or None,
           "round": investigation.get("round", 1)}
    (out_dir / FILENAME).write_text(json.dumps(out, indent=2, default=str))
    return out


# ─────────────────────────────────────────────────────────────────────
# THE BOUNDED LOOP
# ─────────────────────────────────────────────────────────────────────
# THE PLAN'S REVIEW (package E, 2026-11-05). For a suite-bearing study, step
# 3 reviews the recorded analysis plan once, before any lookup: does it follow
# from what was asked? One short model call, and code's own findings on the
# plan (a field, region, method or setting that does not exist, a timing
# setting timing.nc does not hold, a screen out of the question type's order)
# fail it too. Its record is tied to the plan file's hash. This automated
# review approves the analysis plan only: it never authorizes a simulation,
# and running or submitting any model run still needs the user's own
# authorization (the design review's yes, and the user's standing rule to ask
# before any SLURM job).
PLAN_REVIEW_LEAD = """\
Below are a question a person asked of a model study, the question type the
study's Planner recorded, and the ANALYSIS PLAN written for it before any
result was seen. Judge one thing: does the plan follow from what was asked?
It follows when its comparisons, fields, regions, periods and methods address
the question as asked, its expectations (if any) are the question's own and
not new claims, and its competing explanations are relevant. Do not judge
whether the answer will be interesting; no result exists yet.
"""


def _review_json_line() -> str:
    """PLAN_REVIEW_ASK's reply line, rendered from analysis_plan.REVIEW_REPLY,
    the definition rule_plan_follows_question reads (2026-11-08)."""
    from agents.analysis import analysis_plan as _plan
    parts = [f"{json.dumps(p.name)}: {_plan._shown(p)}" for p in _plan.REVIEW_REPLY]
    first = "Return ONLY JSON: {" + ", ".join(parts[:-1]) + ","
    return first + "\n" + parts[-1] + "}\n"


PLAN_REVIEW_ASK = PLAN_REVIEW_LEAD + "\n" + _review_json_line()


def read_review_reply(got: Dict[str, Any]) -> Dict[str, Any]:
    """The review's reply as REVIEW_REPLY defines it: `follows` counts only
    when it is true itself, `why` is words, and `problems` a list of words (a
    single string is one problem, never its letters)."""
    from agents.analysis import analysis_plan as _plan
    names = [p.name for p in _plan.REVIEW_REPLY]
    follows, why, problems = (got.get(n) for n in names)
    if isinstance(problems, str):
        problems = [problems]
    return {"follows": follows is True,
            "why": str(why or ""),
            "problems": [str(x) for x in (problems if isinstance(problems, list) else [])
                         if str(x).strip()]}


def rule_plan_follows_question(ctx, out_dir, client=None,
                               model: str = DEFAULT_MODEL, label: Optional[str] = None
                               ) -> Dict[str, Any]:
    """Step 3's review of 04_analysis/analysis_plan.json, recorded once in
    analysis_plan_review.json: passed only when the model says the plan
    follows from the question AND code found no problem in the plan. The
    prompt is the same for a rewritten plan (it is not told it reviews a
    rewrite); `label` names the client for the spend (step3_plan_review, or
    step3_plan_review_rewrite for the rewrite's review), and the exchange is
    saved under the plan's attempt (round 1 or 2)."""
    from agents.analysis import analysis_plan as _plan
    from agents.analysis import script_runner as _runner
    out_dir = Path(out_dir)
    done = _plan.load_review(out_dir)
    if done:
        return done
    plan = _plan.load_plan(out_dir)
    if not plan:
        return _plan.record_review(out_dir, False, "there is no analysis plan to review",
                                   reviewer="code")
    attempt = int(plan.get("attempt") or 1)
    if client is None:
        from agents.llm_agent import SimpleLLMClient
        client = SimpleLLMClient(model=model)
        client.label = label or (_plan.REVIEW_LABEL if attempt == 1 else
                                 _plan.REWRITE_REVIEW_LABEL)
    shown = {k: plan.get(k) for k in ("question_type",) + _plan.PLAN_KEYS}
    prompt = (f"THE QUESTION:\n    {plan.get('question')}\n\nTHE PLAN:\n"
              + json.dumps(shown, indent=1, default=str) + "\n\n" + PLAN_REVIEW_ASK)
    reply = client.ask([{"role": "user", "content": prompt}])
    unreadable = None
    try:
        got = _parse(reply)
    except (ValueError, json.JSONDecodeError) as e:
        got, unreadable = {}, f"the review's reply was not readable: {e}"
    exchange = _runner.save_exchange(out_dir, "step3_plan_review", attempt, prompt, reply)
    r = read_review_reply(got)
    model_problems = r["problems"] + ([unreadable] if unreadable else [])
    code_problems = list(plan.get("problems") or [])
    problems = model_problems + [f"code: {p}" for p in code_problems]
    passed = r["follows"] and not code_problems and not unreadable
    return _plan.record_review(out_dir, passed, r["why"] or (unreadable or ""), problems,
                               reviewer="step 3's model review (rule_plan_follows_question)",
                               exchange=exchange or None,
                               follows=False if unreadable else r["follows"],
                               model_problems=model_problems, code_problems=code_problems)


def skipped_interpretation(out_dir, why: str, round_no: int = 1) -> Dict[str, Any]:
    """Step 3 not asked, and why, written to interpretation.json in place of a
    reply: a study whose analysis plan did not pass its review made no figure
    and no lookup, so there is nothing to interpret and the call is saved (the
    user's decision on G part 2, 2026-11-08: question 2 still paid a model
    call to read "no results were produced")."""
    out = {"skipped": True, "skipped_because": why, "headline": None, "answer": None,
           "verdict": None, "verdict_asked_again": False, "readability": None,
           "feedback": None, "open_causes": [], "claims": [], "struck": [],
           "audit": {"n_claims": 0, "n_struck": 0}, "exchange": None, "round": round_no}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / FILENAME).write_text(json.dumps(out, indent=2, default=str))
    return out


def _given(clients: Optional[Dict[str, Any]], label: str) -> Dict[str, Any]:
    """{"client": the client for `label`} when the caller gave one, else {}:
    a step then makes its own, labelled for the spend."""
    c = (clients or {}).get(label)
    return {"client": c} if c is not None else {}


def investigate_and_interpret(ctx, out_dir, comparison=None,
                              model: str = DEFAULT_MODEL,
                              client=None, with_images: bool = True,
                              max_rounds: int = MAX_ROUNDS,
                              clients: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """step 2 -> step 3, with at most `max_rounds` passes.

    Step 3 may send step 2 back once with specific feedback. It does not get to
    keep asking: each round costs an LLM call plus a subprocess per figure, and
    a reviewer with an unbounded budget will always find something. If the last
    round is still judged insufficient, that verdict stands and is reported —
    "this experiment cannot answer the question" is a result.

    `client` is step 3's (as it always was); `clients` maps a step's label
    (step2a_plan, step3_plan_review, step2a_plan_rewrite,
    step3_plan_review_rewrite, step2_investigate, step3_interpret) to the
    client that step uses, for a replay with no model call. A step with no
    client given makes its own.
    """
    from agents.analysis import step2_investigate as _step2
    from agents.analysis import analysis_plan as _plan

    rounds: List[Dict[str, Any]] = []
    feedback = None
    investigation = interpretation = None
    step3_client = client if client is not None else (clients or {}).get(
        "step3_interpret")

    # STEP 2a, THEN STEP 3'S REVIEW OF THE PLAN, before any round (package E,
    # 2026-11-05), only for a suite-bearing study. A plan that fails is
    # rewritten ONCE, before any outcome is looked up, and the rewrite gets the
    # same check and the same review (the user's decision on G part 2,
    # 2026-11-08). A plan that still did not pass makes one round: step 2
    # proposes nothing and says why, and step 3 is not asked.
    plan_review = None
    on_suite = _plan.suite_bearing(ctx)
    plan_ok = True
    if on_suite:
        _plan.write_plan(ctx, out_dir, model=model, **_given(clients, _plan.PLAN_LABEL))
        plan_review = rule_plan_follows_question(ctx, out_dir, model=model,
                                                 **_given(clients, _plan.REVIEW_LABEL))
        if (plan_review or {}).get("passed") is False and _plan.can_rewrite(out_dir)[0]:
            _plan.rewrite_plan(ctx, out_dir, model=model,
                               **_given(clients, _plan.REWRITE_LABEL))
            plan_review = rule_plan_follows_question(
                ctx, out_dir, model=model, label=_plan.REWRITE_REVIEW_LABEL,
                **_given(clients, _plan.REWRITE_REVIEW_LABEL))
        plan_ok = _plan.plan_passed(out_dir)[0]
        if not plan_ok:
            max_rounds = 1

    figures: List[Dict[str, Any]] = []         # every round's figures, for the close
    skipped: Dict[str, str] = {}               # step 2b's own reasons, by analysis
    for r in range(1, max(1, max_rounds) + 1):
        investigation = _step2.investigate(
            ctx, out_dir, step1=comparison, model=model,
            feedback=feedback, round_no=r,
            **({"history": {"figures": list(figures), "skipped": dict(skipped)}}
               if on_suite and plan_ok else {}),
            **_given(clients, "step2_investigate"))
        figures += list(investigation.get("figure_outcomes") or [])
        skipped.update({x.get("analysis"): x.get("why")
                        for x in investigation.get("skipped") or [] if isinstance(x, dict)})
        if on_suite and not plan_ok:
            interpretation = skipped_interpretation(
                out_dir, _plan.plan_passed(out_dir)[1], round_no=r)
        else:
            interpretation = interpret(ctx, comparison, investigation, out_dir,
                                       model=model, client=step3_client,
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
        if interpretation.get("skipped"):
            entry["step3_skipped"] = interpretation.get("skipped_because")
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

    # THE CLOSE OF THE PLANNED PHASE, if no round closed it (the user's
    # decision on G part 2, 2026-11-08): in a hypothesis study whose screen
    # never ran, and in every exploratory study (bookkeeping: nothing is
    # required), the framework closes it after the last round, before step 4.
    if on_suite and plan_ok and not _plan.load_planned_phase(out_dir):
        qt = _plan.load_plan(out_dir).get("question_type")
        _plan.close_planned_phase(
            out_dir, when="after the last round",
            by=("the framework, after the last round of step 2b: the screen never ran"
                if qt == "hypothesis" else
                "the framework, after the last round of step 2b: an exploratory question, "
                "so the close is bookkeeping and no analysis is required"),
            figures=figures, skipped=skipped)

    regressed = [r_["round"] for r_ in rounds if r_.get("regressed")]
    return {"investigation": investigation, "interpretation": interpretation,
            **({"plan_review": plan_review} if plan_review is not None else {}),
            "rounds": rounds, "n_rounds": len(rounds),
            "regressed_rounds": regressed or None,
            "stopped_because": ("the analysis plan did not pass its review, so step 3 was "
                                "not asked" if interpretation.get("skipped") else
                                "sufficient"
                                if interpretation["verdict"] == "sufficient"
                                else f"exhausted {len(rounds)} of {max_rounds} rounds")}
