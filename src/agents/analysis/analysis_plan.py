#!/usr/bin/env python3
"""
The recorded analysis plan, its review, its amendments and the lookups log
src/agents/analysis/analysis_plan.py

    04_analysis/analysis_plan.json              step 2a, written once (or rewritten
                                                once, before any lookup)
    04_analysis/analysis_plan_review.json       step 3's review of it, once
    04_analysis/analysis_plan.v1.json           the first plan, when it failed its
    04_analysis/analysis_plan_review.v1.json    review and was rewritten, and that
                                                review, kept as they were
    04_analysis/analysis_plan_amendments.jsonl  amendments, appended only
    04_analysis/lookups.jsonl                   every call on `suite`, appended
    04_analysis/planned_phase.json              the close of the planned analyses:
                                                each one completed, failed or
                                                skipped, with its reason, once
    04_analysis/screen_record.json              the plan's screen: scope and rules
    04_analysis/screen_L<n>.csv                 a screen's whole table (lookup L<n>)
    04_analysis/contract_reads.jsonl            method contracts read (no lookup)

ONLY ON THE WHOLE-DOMAIN PATH WITH A SUITE (package E, 2026-11-05). A study is
suite-bearing when it is a whole-domain experiment and its control's results
store (domain_suite) is built and is the store of the output this study
reads: the study's own 03_results/suite/control, or, for a reused control,
the registry's suite_dir whose manifest names the same run directory as the
study's record of the control. Every other study (columns, PFLOTRAN, a
whole-domain study without a suite) takes the paths it took before.

THE THREE SAFEGUARDS.

  1. The method contracts. A script reaches the methods only through `suite`
     (domain_access.SuiteAccess), whose calls run domain_methods with the
     settings settings_for allows: the default, a named alternative, or a
     setting this plan (or an amendment) names. A timing setting is answered
     only from what timing.nc holds. suite.method_contract(method_id) gives a
     method's full contract at any time: no outcome, so it is not a lookup.
  2. The analysis plan. Step 2a writes it from the question, the
     configuration (the experiment record and the server's change list) and
     the catalogs, and nothing that carries an outcome (WITHHELD). Step 3
     reviews it (rule_plan_follows_question): code's check (plan_problems)
     and a model's review. A plan that fails is written once more, from the
     same inputs plus the first plan and what its review found, before any
     outcome is looked up, and the rewrite gets exactly the same check and
     review; the first version and its review are kept (the user's decision
     on G part 2, 2026-11-08). That automated review approves the analysis
     plan only: it never authorizes a simulation, and nothing here can
     submit or build one. Running or submitting any model run still needs
     the user's own authorization (the design review's yes, and the user's
     standing rule to ask before any SLURM job). An amendment is appended
     with a time stamp, never rewritten, linked to the lookups made under
     it, and never counts as planned.
  3. The lookups log and its gate (check_lookup, closing package E,
     2026-11-05). No call before the plan has passed that review and the plan
     file still hashes to what was reviewed. Then a lookup (get_map,
     get_series, get_timing, a method call) is refused unless one of the
     reviewed plan's analyses declares its field, region, period (or series
     step), method and setting, or an amendment appended before it does; an
     amended lookup is marked not planned. The question type sets the order
     (plan section 3.5): for a hypothesis, the screen is refused until the
     framework has closed the planned-analysis phase (planned_phase.json:
     each analysis of the plan completed, failed or skipped, with a reason;
     the user's decision on G part 2, 2026-11-08), every lookup after that
     close is exploratory and never planned, and what the screen turns up
     is "found by screening, not tested"; for an exploratory question, every
     lookup but the screen is refused until the plan's screen has run and
     its scope, scoring rule and display rule are recorded
     (screen_record.json). Every call, refused or answered, is logged with
     its keys, time, method and settings, its marks, and a hash of what it
     returned.

This is workflow enforcement, not a sandbox: script_runner runs code as the
user, so a script could still open a file. The claim trace (package F) is what
would show it: a number that did not come through a logged lookup fails.
"""
import hashlib
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from agents.analysis import plan_contract as PC
from agents.analysis.dashes import strip_dashes, strip_dashes_in

_ROOT = Path(__file__).resolve().parents[3]
_ELM_SRC = _ROOT / "mcp" / "elm-mcp" / "src"

PLAN_FILE = "analysis_plan.json"
REVIEW_FILE = "analysis_plan_review.json"
# THE FIRST VERSION, KEPT (the user's decision on G part 2, 2026-11-08): when
# the first plan fails its review it is rewritten once, before any outcome is
# looked up; the first plan and its review are moved to these names unchanged,
# so the first review's hash still ties it to the first plan.
PLAN_V1_FILE = "analysis_plan.v1.json"
REVIEW_V1_FILE = "analysis_plan_review.v1.json"
AMENDMENTS_FILE = "analysis_plan_amendments.jsonl"
LOOKUPS_FILE = "lookups.jsonl"
PLANNED_PHASE_FILE = "planned_phase.json"
# THE FRAMEWORK'S OWN CALLS OF THE REQUIRED OUTPUTS (2026-11-08): logged under
# this script name, recorded in REQUIRED_FILE, their results kept for the
# figures of a later round in REQUIRED_VALUES
REQUIRED_SCRIPT = "required_outputs"
REQUIRED_FILE = "required_outputs.json"
REQUIRED_VALUES = "required_outputs.pkl"
SUITE_MANIFEST = "suite.json"
RULE = "rule_plan_follows_question"
QUESTION_TYPES = ("hypothesis", "exploratory")
PLAN_VERSION = 1
# The labels of the model calls of this module and of step 3's review of it,
# which step 4 counts in the Analyzer's spend (step4_report.ANALYZER_LABELS).
PLAN_LABEL = "step2a_plan"
REWRITE_LABEL = "step2a_plan_rewrite"
REVIEW_LABEL = "step3_plan_review"
REWRITE_REVIEW_LABEL = "step3_plan_review_rewrite"
# Set in a figure script's environment by script_runner: the close of the
# planned phase is the framework's, never a script's.
IN_SCRIPT_ENV = "ANALYZER_FIGURE_SCRIPT"
# What step 2a never sees, and what the runner never binds for a study with a
# suite, before or after the review (package F1, 2026-11-05): the Analyzer
# reaches values only through the scoped, logged suite calls.
OUTCOME_TABLES = ("reg", "reg_monthly", "reg_seasonal")
# the suite call that answers what each table held, named in the refusal
TABLE_CALLS = {"reg": 'suite.get_series(run, field, region, "year")',
               "reg_monthly": 'suite.get_series(run, field, region, "month")',
               "reg_seasonal": 'suite.get_series(run, field, region, "season")'}
WITHHELD = ("the regional tables (reg, reg_monthly, reg_seasonal)",
            "the code-only report (DOMAIN_REPORT.md) and its figures",
            "the comparisons with observations")
SEEN = ("the question", "the question type (the Planner's)",
        "the configuration (the experiment record and the server's change list)",
        "the data catalog", "the methods (the compact view of docs/ANALYSIS_METHODS.md)")
SEEN_RECIPE = "the analysis recipe the question calls for (guidance, analysis_recipes)"
# calls that read no outcome: what the grid is, not what the run did
SUPPORT_CALLS = ("region_mask", "cell_area", "lat_lon", "find_fields")
SCREEN_FILE = "screen_record.json"
# THE SCREEN'S WHOLE TABLE (the user's decision 4 on G part 1, 2026-11-08):
# every answered screen's table, every field and period with every column,
# written as a CSV beside the lookups log and named by its lookup; the line
# records its file, row count and hash, and the report links it beside the
# display rule ("The whole table is the CSV beside this rule").
SCREEN_CSV = "screen_L{n}.csv"
CONTRACT_READS_FILE = "contract_reads.jsonl"
SCREENING_LABEL = "found by screening, not tested"
EXPLORATORY_LABEL = "exploratory"
SCREEN_WHEN = {"hypothesis": "last", "exploratory": "first"}
SERIES_STEPS = ("day", "month", "season", "year", "wateryear")
ARRAY_METHODS = ("overlap", "elevation_split")    # take maps already looked up
SCOPE_KEYS = ("fields", "regions", "periods", "methods", "steps")
APPROVES = ("step 3's automated review approves this analysis plan only; it never "
            "authorizes a simulation. Running or submitting any model run still needs the "
            "user's own authorization: the design review's yes, and the user's standing rule "
            "to ask before any SLURM job")
SIGNS = ("lower", "higher", "earlier", "later", "none")
# A QUESTION ABOUT TIMING NEEDS A TIMING MEASURE (the user's decision after the
# framework's third acceptance batch, 2026-11-08): question 1 asked whether an
# early snow loss accompanied the summer GPP decline, and both versions of its
# plan named no timing measure, so its review passed a plan that could not
# test the question as asked. The words that make a question one about timing,
# whole words in any case, a hyphen optional; never "when", which most
# questions use for other things. A word that only dates a period ("early
# summer", "late 2015", "in later years", "a year later") is not one.
TIMING_SIGNS = ("earlier", "later")
TIMING_QUESTION_WORDS = ("early", "earlier", "earliest", "late", "later", "latest", "sooner",
                         "onset", "timing", "delay", "delayed", "melt-out", "green-up")
TIMING_PERIOD_WORDS = ("spring", "summer", "autumn", "fall", "winter", "season", "month",
                       "months", "year", "years", "decade", "decades", "january", "february",
                       "march", "april", "may", "june", "july", "august", "september",
                       "october", "november", "december")
_TIMING_WORD = re.compile(
    r"(?<!\w)(?<!year )(?<!years )("
    + "|".join(re.escape(w).replace(r"\-", "[- ]?") for w in TIMING_QUESTION_WORDS)
    + r")(?!\w)(?![- ](?:" + "|".join(TIMING_PERIOD_WORDS) + r"|\d{4})(?!\w))",
    re.IGNORECASE)
# ... THE MEASURES OF WHAT IT IS ABOUT, NOT ANY MEASURE, AND NO DIRECTION (the
# user's decision of 2026-11-08): a question about an early snow loss needs a
# snow measure, not a growth one, for the year it names, and the rule never
# asks the plan to state "earlier" or "later" when the question does not.
# What a timing word is about: the words naming what each measure dates,
# found within TIMING_WINDOW words of it in its own clause, else anywhere in
# its clause, else anywhere in the question (a clause ends at a comma, a
# semicolon, "and", "or", "but", "while" or the sentence's end). The most
# specific wins: each word found counts for each measure that lists it one
# over the number of measures listing it, and the measures with the highest
# count are the ones the word is about ("snow melt-out": snow_gone; "early
# snow loss": either snow measure). A timing word about nothing these
# measures date needs none of them.
TIMING_TOPICS = {
    "snow_peak": ("snow", "snowpack", "swe", "peak"),
    "snow_gone": ("snow", "snowpack", "snowmelt", "melt", "melt-out", "snow-free",
                  "snow loss"),
    "leaf_green_up": ("leaf", "leaves", "green-up", "leaf-out", "lai", "canopy"),
    "photosynthesis_onset": ("photosynthesis", "gpp", "uptake", "growth", "productivity",
                             "onset"),
    "growth_peak": ("photosynthesis", "gpp", "uptake", "growth", "productivity", "peak"),
    "half_runoff": ("runoff", "streamflow", "discharge"),
}
TIMING_WINDOW = 4
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")
_CLAUSE_END = re.compile(r"[,;:.?!]|\b(?:and|or|but|while)\b", re.IGNORECASE)


def _topic_re(w: str) -> "re.Pattern":
    return re.compile(r"(?<![\w-])" + re.escape(w).replace(r"\-", "[- ]?") + r"(?![\w-])",
                      re.IGNORECASE)


_TOPIC_RES = {mid: [(w, _topic_re(w)) for w in ws] for mid, ws in TIMING_TOPICS.items()}


def timing_words(question: Any) -> List[str]:
    """The words of `question` that ask about timing (TIMING_QUESTION_WORDS,
    not those that only date a period), in the order they come."""
    return [m.group(1) for m in _TIMING_WORD.finditer(str(question or ""))]


_SHARED = {}
for _ws in TIMING_TOPICS.values():
    for _w in _ws:
        _SHARED[_w] = _SHARED.get(_w, 0) + 1


def _topics(text: str) -> Tuple[List[str], List[str]]:
    """(the topic words found, the measures they are most specific to) in
    `text`: each word counts 1/(the measures listing it) for each of them,
    and the measures with the highest count are kept, in TIMING_TOPICS'
    order."""
    found, score = {}, {}
    for mid, pats in _TOPIC_RES.items():
        for w, rx in pats:
            m = rx.search(text)
            if m:
                score[mid] = score.get(mid, 0.0) + 1.0 / _SHARED[w]
                found.setdefault(w, m.start())
    if not score:
        return [], []
    top = max(score.values())
    words = [w for w in sorted(found, key=lambda w: (found[w], -len(w)))
             if not any(v != w and found[v] == found[w] and w in v for v in found)]
    return words, [mid for mid in TIMING_TOPICS if abs(score.get(mid, -1) - top) < 1e-9]


def timing_needs(question: Any) -> List[Dict[str, Any]]:
    """For each timing word of `question`, what it is about and the timing
    measures that date it: {"word", "topics", "measures"}, from the words
    within TIMING_WINDOW of it in its clause, else its clause, else the whole
    question; a word whose topics name no measure is left out, and a need
    already listed is not listed again."""
    q = str(question or "")
    ends = [0] + [b.start() for b in _CLAUSE_END.finditer(q)] + [len(q)]
    out: List[Dict[str, Any]] = []
    for m in _TIMING_WORD.finditer(q):
        lo = max(e for e in ends if e <= m.start())
        hi = min(e for e in ends if e >= m.end())
        clause = q[lo:hi]
        toks = list(_TOKEN.finditer(clause))
        i = next((k for k, t in enumerate(toks) if lo + t.end() > m.start()), len(toks))
        near = " ".join(t.group(0) for t in toks[max(0, i - TIMING_WINDOW):
                                                 i + TIMING_WINDOW + 1])
        topics, ms = _topics(near)
        if not ms:
            topics, ms = _topics(clause)
        if not ms:
            topics, ms = _topics(q)
        if ms and all(x["measures"] != ms for x in out):
            out.append({"word": m.group(1), "topics": topics, "measures": ms})
    return out


def _or(xs: Sequence[str]) -> str:
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " or " + xs[-1]


AGAINST = ("normal years", "control")


# ─────────────────────────────────────────────────────────────────────
# THE PLAN'S SHAPE AND RULES, DEFINED ONCE
# ─────────────────────────────────────────────────────────────────────
# ONE DEFINITION, TWO READERS (the user's decision on G part 2, 2026-11-08).
# Question 2 listed the screen as an analysis and question 3 wrote a setting
# not keyed by a method; code rejected both, while the task the model read
# allowed the first and never said how the second was keyed. The prompt and
# the check were two hand-written copies that had drifted apart. Now
# PLAN_PARTS says what each key of the plan holds and PLAN_RULES what the
# check refuses, each rule with its words and its check: plan_problems runs
# the checks and plan_task() renders the words, the parts and each method's
# settings (read from domain_methods), so neither can say what the other
# does not.
@dataclass(frozen=True)
class Part:
    """One part of a model-facing reply: its name, what it holds and what it
    means, in plain words, and the example the prompt shows. `holds` is
    "name" (an id or a name), "words" (prose, kept without dashes), "list"
    (of names), "years", "year", "settings" (by method, then setting) or
    "choice" (one of `choices`); a "list" with `choices` holds only those."""
    name: str
    holds: str
    means: str
    example: Any = "..."
    choices: Tuple[Any, ...] = ()


PLAN_PARTS: Dict[str, Dict[str, Any]] = {
    "comparisons": {
        "shape": "list",
        "means": "what is compared with what (a run against the control, or the control's "
                 "years against its normal years), over which years",
        "item": (Part("id", "name", "its id", "C1"),
                 Part("what", "words", "what is compared with what"),
                 Part("run", "name", "the run: control, or a run's name", "control"),
                 Part("against", "choice", "what the run is compared against",
                      choices=AGAINST),
                 Part("years", "years", "the years compared", [2015]))},
    "baseline": {
        "shape": "object",
        "means": "the normal years (the Planner's, given above, unless the question needs "
                 "none)",
        "item": (Part("normal_years", "years", "the normal years", "[...]"),
                 Part("why", "words", "why these years"))},
    "analyses": {
        "shape": "list",
        "means": "each a planned lookup: what a script may look up, and why",
        "item": (Part("id", "name", "its id", "P1"),
                 Part("question", "words", "the question it answers"),
                 Part("fields", "list", "the catalog fields it reads", ["GPP"]),
                 Part("regions", "list", "the regions it reads", ["..."]),
                 Part("periods", "list", "the periods it reads", ["2015"]),
                 Part("steps", "list", "the series steps it reads a region's series at, "
                      "if any", [], choices=SERIES_STEPS),
                 Part("methods", "list", "the methods it runs", ["anomaly"]),
                 Part("settings", "settings", "a method's settings other than its defaults, "
                      "by method id, then setting name; {} keeps every default", {}),
                 # ITS REQUIRED OUTPUTS, APART FROM ITS SCOPE (the user's decision
                 # after G part 2's fixes, 2026-11-08): the scope above says what a
                 # script may look up, these what it must produce (plan_contract)
                 Part("outputs", "outputs", "the outputs it must produce to count as "
                      "completed, apart from its scope: each one suite call named exactly "
                      "(see THE REQUIRED OUTPUTS)",
                      [{"id": "O1", "kind": "anomaly", "field": "GPP", "period": "2015",
                        "region": "..."}]),
                 Part("why", "words", "why the question needs it"))},
    "expectations": {
        "shape": "list",
        "means": "for a hypothesis question, what the question expects (field, sign, timing), "
                 "each tied to an analysis; none for an exploratory question",
        "item": (Part("id", "name", "its id", "E1"),
                 Part("analysis", "name", "the analysis that tests it", "P1"),
                 Part("field", "name", "the field it is about", "GPP"),
                 Part("sign", "choice", "the direction the question expects", choices=SIGNS),
                 Part("text", "words", "the expectation, in words"))},
    "competing": {
        "shape": "list",
        "means": "other mechanisms that could give the same pattern, each with the fields "
                 "that would tell them apart (the methods view lists candidates)",
        "item": (Part("text", "words", "the mechanism, in words"),
                 Part("fields", "list", "the fields that would tell it apart", ["..."]),
                 Part("analysis", "name", "the analysis that reads them", "P2"))},
    "screen": {
        "shape": "object",
        "means": "when the screen runs (last for a hypothesis question, first for an "
                 "exploratory one), its scope (region, year), and that its scores and its "
                 "display follow the screen's own rule",
        "item": (Part("when", "choice", "when it runs: last for a hypothesis question, first "
                      "for an exploratory one",
                      choices=tuple(dict.fromkeys(SCREEN_WHEN.values()))),
                 Part("region", "name", "its region", "..."),
                 Part("year", "year", "its year", 2015),
                 Part("why", "words", "why it runs then, over that scope"))},
}
PLAN_KEYS = tuple(PLAN_PARTS)
# what the plan file holds besides the model's parts: written by the framework
# (the question type is the Planner's, never the model's)
PLAN_RECORD_KEYS = ("plan_version", "written_at", "step", "question", "question_type",
                    "question_type_from", "attempt", "rewrite_of", "seen", "withheld",
                    "problems")
# THE PLAN REVIEW'S REPLY (step 3's rule_plan_follows_question), defined once:
# the review reads it and PLAN_REVIEW_ASK's JSON line is rendered from it.
REVIEW_REPLY = (Part("follows", "choice", "does the plan follow from what was asked",
                     choices=(True, False)),
                Part("why", "words", "one or two sentences", "<one or two sentences>"),
                Part("problems", "list", "each thing that does not follow",
                     ["<each thing that does not follow>"]))


def _shape_default(key: str):
    return {} if PLAN_PARTS[key]["shape"] == "object" else []


def _shown(p: Part) -> str:
    """A part's example as the prompt shows it: a choice's values joined by
    "|", any other value as JSON (a placeholder list "[...]" as itself)."""
    if p.holds == "choice":
        return " | ".join(json.dumps(v) for v in p.choices)
    if p.holds == "settings":
        return '{} | {"<method id>": {"<setting name>": <value>}}'
    if isinstance(p.example, str) and p.example.startswith("["):
        return p.example
    return json.dumps(p.example)


def _part_means(p: Part) -> str:
    """A part's meaning with the values it may take, as the prompt says it."""
    if p.choices:
        return (f"{p.means}, one of: " if p.holds == "choice" else f"{p.means}, any of: ") + \
            ", ".join(json.dumps(v) if isinstance(v, bool) else str(v) for v in p.choices)
    return p.means


def reply_json(parts: Dict[str, Dict[str, Any]]) -> List[str]:
    """The lines of a reply's JSON template, rendered from its parts: one
    line per key, each item's parts with their examples."""
    lines = []
    keys = list(parts)
    for i, k in enumerate(keys):
        d = parts[k]
        body = ", ".join(f"{json.dumps(p.name)}: {_shown(p)}" for p in d["item"])
        body = f"[{{{body}}}]" if d["shape"] == "list" else f"{{{body}}}"
        line = ("{" if i == 0 else " ") + f"{json.dumps(k)}: {body}" + \
            ("}" if i == len(keys) - 1 else ",")
        lines += textwrap.wrap(line, 79, subsequent_indent="   ", break_long_words=False,
                               break_on_hyphens=False)
    return lines


def _wrapped(text: str, first: str, rest: str) -> List[str]:
    return textwrap.wrap(text, 79, initial_indent=first, subsequent_indent=rest,
                         break_long_words=False, break_on_hyphens=False)


@dataclass(frozen=True)
class PlanRule:
    """One rule of the plan check: what it is about, whether it judges each
    analysis or the plan as a whole, the words the prompt shows, and the
    check that returns its problems."""
    id: str
    scope: str                                    # "analysis" or "plan"
    words: Any                                    # the words, or what renders them
    check: Callable[[Dict[str, Any], Dict[str, Any]], List[str]]

    def text(self) -> str:
        """The words the prompt shows: rendered when they read definitions
        this module imports only when needed (the timing rule's)."""
        return self.words() if callable(self.words) else self.words


def _check_fields(a, env):
    return [f"{a.get('id')}: no catalog field {f!r}" for f in a.get("fields") or []
            if f not in env["fields"]]


def _check_regions(a, env):
    if not env["regions"]:
        return []
    return [f"{a.get('id')}: no region {r!r}" for r in a.get("regions") or []
            if r not in env["regions"]]


def _check_methods(a, env):
    out = []
    for m in a.get("methods") or []:
        if m == "screen":
            out.append(f"{a.get('id')}: the screen is declared in the plan's screen entry, not "
                       f"as an analysis")
        elif m not in env["M"].METHODS:
            out.append(f"{a.get('id')}: no method {m!r}")
    return out


def _check_steps(a, env):
    return [f"{a.get('id')}: no series step {x!r} ({', '.join(SERIES_STEPS)})"
            for x in a.get("steps") or [] if x not in SERIES_STEPS]


def _check_settings(a, env):
    M, aid, out = env["M"], a.get("id"), []
    for m, st in (a.get("settings") or {}).items():
        if m not in M.METHODS:
            out.append(f"{aid}: settings for no method {m!r}")
            continue
        for k, v in (st or {}).items():
            if k not in M.METHODS[m].defaults():
                out.append(f"{aid}: {m} has no setting {k!r}")
            elif m in M.TIMING_IDS:
                # TIMING SETTINGS ARE STORED, NEVER COMPUTED ON THE FLY
                s = M.METHODS[m].setting(k)
                for x in (v if isinstance(v, list) else [v]):
                    if x != s.default and x not in s.alternatives:
                        out.append(
                            f"{aid}: {m} {k}={x!r} is not a setting timing.nc holds (it "
                            f"holds the default {s.default!r} and the contract's "
                            f"alternatives {list(s.alternatives)}); a timing measure is "
                            f"never computed on the fly")
    return out


def _check_periods(a, env):
    M, out = env["M"], []
    for p in a.get("periods") or []:
        s = str(p)
        if s.startswith("normal"):
            if s[7:].isdigit():
                out.append(f"{a.get('id')}: {s}: month maps are not in the suite")
            continue
        try:
            M.parse_period(s)        # a month is a region series, never a map
        except M.MethodError as e:
            out.append(f"{a.get('id')}: {e}")
    return out


def _check_outputs(a, env):
    """The required outputs (plan_contract.output_problems), each asked of
    the gate's own rule whether this analysis's scope allows its call."""
    # its scope as the gate sees it: the analysis's keys and the settings it
    # names, read safely (a malformed settings entry is _check_settings's)
    it = {k: list(a.get(k) or []) for k in SCOPE_KEYS}
    st = a.get("settings") if isinstance(a.get("settings"), dict) else {}
    it["settings"] = {m: {k: (v if isinstance(v, list) else [v]) for k, v in kv.items()}
                      for m, kv in st.items() if isinstance(kv, dict)}
    return PC.output_problems(a, env["M"].TIMING_IDS, lambda lk: _not_declared(it, lk),
                              env.get("outputs"), defaults=_method_defaults) \
        + _month_map_problems(a, env["M"])


def _month_map_problems(a, M) -> List[str]:
    """A per-cell output of a month (a map, or an anomaly or a difference
    with region null): the suite holds no month maps, so the call would be
    refused by the data after the plan had passed."""
    out = []
    for o in a.get("outputs") or []:
        if not isinstance(o, dict) or o.get("kind") not in ("map",) + PC.NULL_REGION:
            continue
        if o.get("kind") != "map" and o.get("region") is not None:
            continue
        try:
            kind = M.parse_period(str(o.get("period")))[0]
        except M.MethodError:
            continue                                  # _check_periods names it
        if kind == "month":
            out.append(f"{a.get('id')}: output {o.get('id')!r}: month maps are not in the "
                       f"suite: a month ({o.get('period')}) is a region's value only, in an "
                       f"anomaly or a difference that names a region")
    return out


def _method_defaults(kind: str) -> Dict[str, Any]:
    """A method's settings at their defaults (domain_methods), {} for a kind
    that is no method."""
    _elm()
    import domain_methods as M                                      # noqa: N812
    return M.METHODS[kind].defaults() if kind in M.METHODS else {}


def produce_required(ctx, out_dir, round_no: int = 1) -> Dict[str, Any]:
    """THE FRAMEWORK MAKES THE REQUIRED OUTPUTS (the user's decision of
    2026-11-08: step 2b's output size, and plotting independence). Once per
    study, before step 2b's model call: in an exploratory study the plan's
    screen first (the gate refuses every other lookup until it has run),
    then every required output's call exactly as plan_contract.call_spec
    writes it, through the gated, logged `suite` like any script's, under
    the script name REQUIRED_SCRIPT. An input map comes from the output it
    names. A call the gate or the data refuses is logged as refused and
    judged at the close; a call that cannot be made (its input map was not
    made) is recorded with why. Returns {"values": {label: result},
    "record": REQUIRED_FILE's content}; a second call reads them back."""
    import pickle
    out = Path(out_dir)
    rec_p, val_p = out / REQUIRED_FILE, out / REQUIRED_VALUES
    if rec_p.is_file() and val_p.is_file():
        with open(val_p, "rb") as f:
            return {"values": pickle.load(f), "record": _read_json(rec_p)}
    check_gate(out)
    _elm()
    import domain_methods as M                                      # noqa: N812
    plan = load_plan(out)
    index = PC.plan_index(plan)
    acc = make_access(access_spec(ctx), script=REQUIRED_SCRIPT)
    first = _lines(out / LOOKUPS_FILE) + 1
    values: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    sc = plan.get("screen") or {}
    if plan.get("question_type") == "exploratory" and not (out / SCREEN_FILE).is_file() \
            and sc.get("region") and str(sc.get("year")).isdigit():
        try:
            values["screen"] = acc.screen(int(sc["year"]), sc["region"])
            rows.append({"label": "screen", "line": _lines(out / LOOKUPS_FILE), "ok": True})
        except Exception as e:                                      # noqa: BLE001
            rows.append({"label": "screen", "line": _lines(out / LOOKUPS_FILE), "ok": False,
                         "error": f"{type(e).__name__}: {e}"[:300]})
    for c in PC.required_calls(plan, M.TIMING_IDS, _method_defaults):
        cs, label = c["call"], c["label"]
        args, missing = [], None
        for v in cs["args"]:
            if isinstance(v, dict) and "input" in v:
                io = index.get(v["input"]) or {}
                src = values.get(v["input"])
                path = PC.MAP_PATH.get(io.get("kind"))
                m = src if path is None or src is None else (src or {}).get(path)
                # A DEPENDENCY NOT MADE (2026-11-08, before the third batch): an
                # input whose call was refused, or whose map is empty because the
                # method found it not estimable, is no map; the call that takes it
                # is not made, and the record says which input and why
                if m is None:
                    why = (src or {}).get("why") if isinstance(src, dict) else None
                    missing = v["input"] + (f" ({why})" if why else "")
                    break
                args.append(m)
            else:
                args.append(v)
        if missing:
            rows.append({"label": label, "line": None, "ok": False,
                         "error": f"not called: its input map {missing} was not made"})
            continue
        try:
            values[label] = getattr(acc, cs["fn"])(*args, **cs["kwargs"])
            rows.append({"label": label, "line": _lines(out / LOOKUPS_FILE), "ok": True})
        except Exception as e:                                      # noqa: BLE001
            rows.append({"label": label, "line": _lines(out / LOOKUPS_FILE), "ok": False,
                         "error": f"{type(e).__name__}: {e}"[:300]})
    # each call as the brief wrote it, beside what it came to
    analyses = {a.get("id"): a for a in plan.get("analyses") or [] if isinstance(a, dict)}
    written = {c["label"]: PC.call_text(c["output"], analyses.get(c["analysis"]) or {}, index,
                                        M.TIMING_IDS, _method_defaults)
               for c in PC.required_calls(plan, M.TIMING_IDS, _method_defaults)}
    record = {"file": REQUIRED_FILE, "at": _now(), "round": round_no,
              "script": REQUIRED_SCRIPT, "lines": [first, _lines(out / LOOKUPS_FILE)],
              "calls": [dict(r, call=written.get(r["label"], "suite.screen (the plan's)"))
                        for r in rows],
              "values_file": REQUIRED_VALUES}
    with open(val_p, "wb") as f:
        pickle.dump(values, f)
    rec_p.write_text(json.dumps(record, indent=1, default=str))
    return {"values": values, "record": record}


def required_attempt(record: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    """The framework's calls as one attempt of every analysis, for the close
    (plan_contract.evaluate's history): its lines, and what could not be
    called."""
    errs = [f"{r['label']}: {r['error']}" for r in (record or {}).get("calls") or []
            if not r.get("ok") and r.get("label") != "screen"]
    return {"id": REQUIRED_SCRIPT, "what": "the framework's required calls",
            "plan_items": [a.get("id") for a in plan.get("analyses") or [] if isinstance(a, dict)],
            "ok": not errs, "error": "; ".join(errs) or None,
            "round": (record or {}).get("round"), "lines": (record or {}).get("lines")}


def required_findings(out_dir, record: Dict[str, Any], plan: Dict[str, Any]
                      ) -> List[Dict[str, Any]]:
    """The framework's calls as findings step 3 reads, one per analysis: its
    question, each required output's result (the logged record's own flat
    result, so a claim's values are the record's) and the evidence records
    (claim_trace.lookups_since). Independent of any figure."""
    from agents.analysis import claim_trace as CT
    out = Path(out_dir)
    log = lookups(out)
    lo, hi = (record or {}).get("lines") or [1, 0]
    recs = {x["id"]: x for x in CT.lookups_since(out, lo - 1) if x["id"] in
            {f"L{i}" for i in range(lo, hi + 1)}}
    by_label = {r["label"]: r for r in (record or {}).get("calls") or []}
    found = []
    for a in plan.get("analyses") or []:
        if not isinstance(a, dict):
            continue
        aid, res, ids = a.get("id"), {}, []
        for o in a.get("outputs") or []:
            lab = f"{aid}.{o.get('id')}"
            r = by_label.get(lab) or {}
            ln = log[r["line"] - 1] if r.get("ok") and r.get("line") else None
            if ln is not None and "sha256" in ln:
                # ITS CITABLE VALUES ONLY (2026-11-08): the labels a result
                # holds (its normal years, its settings) are no numbers a
                # script made, and step 3's brief listed them as such
                res[lab] = CT.citable(ln.get("result")) or \
                    {"returned": "a map (no numbers of its own)"}
                ids.append(f"L{r['line']}")
            else:
                res[lab] = {"not made": (ln or {}).get("refused") or r.get("error") or "no call"}
        if not res:
            continue
        found.append({"id": f"required_{aid}".lower(), "question": a.get("question"),
                      "scale": "regions" if len(a.get("regions") or []) > 1 else "overall",
                      "variables": list(a.get("fields") or []), "result": res,
                      "n": len(ids), "figure": None, "script": None,
                      "made_by": "the framework (the plan's required calls)",
                      "plan_items": [aid], "amendment": None,
                      "lookups": [recs[i] for i in ids if i in recs]})
    return found


def output_calls(plan: Dict[str, Any], units: Optional[Dict[str, str]] = None) -> List[str]:
    """Step 2b's list of the plan's required outputs, each with the exact
    call that produces it (plan_contract.call_lines), and with `units` the
    unit each is drawn in."""
    _elm()
    import domain_methods as M                                      # noqa: N812
    return PC.call_lines(plan, M.TIMING_IDS, _method_defaults, units=units)


def output_units(out_dir, plan: Dict[str, Any]) -> Dict[str, str]:
    """THE UNIT EACH REQUIRED OUTPUT IS DRAWN IN (2026-11-08), by the rule the
    figure check splits panels by (figure_spec.output_unit), read from the
    framework's own results (REQUIRED_VALUES): only the unit, never a value.
    Only for an output a bars or series panel can draw; {} before the
    framework's calls are made."""
    import pickle
    from agents.analysis import figure_spec as FS
    p = Path(out_dir) / REQUIRED_VALUES
    if not p.is_file():
        return {}
    with open(p, "rb") as fh:
        values = pickle.load(fh)
    index = PC.plan_index(plan)
    out: Dict[str, str] = {}
    for lab, o in index.items():
        if o.get("kind") in PC.NULL_REGION and o.get("region") is None:
            continue                                     # a map: drawn by a map panel
        ins = list(o.get("inputs") or [])
        r = {"kind": o.get("kind"), "out": o, "value": values.get(lab),
             "input_units": {i: ((values.get(i) or {}).get("change_units")
                                 or (values.get(i) or {}).get("units"))
                             if isinstance(values.get(i), dict) else None for i in ins}}
        u = FS.output_unit(r, {})
        if u and o.get("kind") != "overlap":
            out[lab] = u
    return out


def _check_baseline(plan, env):
    ny = [int(y) for y in ((plan.get("baseline") or {}).get("normal_years") or [])]
    py = env["planner_years"]
    if py and ny and sorted(ny) != sorted(int(y) for y in py):
        return [f"baseline: normal years {ny} are not the Planner's {list(py)}"]
    return []


def _check_screen(plan, env):
    qt, rs, out = env["question_type"], env["regions"], []
    sc = plan.get("screen")
    if not isinstance(sc, dict) or not sc:
        return ["screen: not declared (plan section 3.5: the screen runs last for a "
                "hypothesis question and first for an exploratory one)"]
    if qt in SCREEN_WHEN and sc.get("when") != SCREEN_WHEN[qt]:
        out.append(f"screen: when is {sc.get('when')!r}, but a {qt} question runs it "
                   f"{SCREEN_WHEN[qt]}")
    if rs and sc.get("region") not in rs:
        out.append(f"screen: no region {sc.get('region')!r}")
    if not str(sc.get("year")).isdigit():
        out.append(f"screen: year {sc.get('year')!r} is not a year")
    return out


def _check_timing(plan, env):
    """A question about timing has, for each timing word, a timing measure of
    what the word is about (timing_needs) among its required outputs, for a
    year the question names (any year when it names none); an expectation
    whose sign is earlier or later is tested by an analysis with a timing
    measure of the expectation's own field among its outputs. It asks for no
    expectation: a direction comes from the plan, never from this rule."""
    from agents.analysis import claim_trace as _ct
    M = env["M"]
    ids = tuple(M.TIMING_IDS)
    analyses = [a for a in plan.get("analyses") or [] if isinstance(a, dict)]
    outs = {str(a.get("id")): [o for o in a.get("outputs") or [] if isinstance(o, dict)
                               and o.get("kind") in ids] for a in analyses}
    years = sorted(int(y) for y in _ct.years_named(str(plan.get("question") or "")))
    out = []
    for need in timing_needs(plan.get("question")):
        if not any(o.get("kind") in need["measures"]
                   and (not years or PC._year(o.get("period")) in years)
                   for os_ in outs.values() for o in os_):
            out.append(f"timing: the question's {need['word']!r} is about "
                       f"{' and '.join(need['topics'])}, but no analysis has "
                       f"{_or(need['measures'])} among its outputs"
                       + (f" for {_or([str(y) for y in years])}" if years else ""))
    dated = {}
    for k in ids:
        dated.setdefault(M.TIMING_FIELD[k], []).append(k)
    for e in plan.get("expectations") or []:
        if not isinstance(e, dict) or e.get("sign") not in TIMING_SIGNS:
            continue
        f, aid = e.get("field"), str(e.get("analysis"))
        if f not in dated:
            out.append(f"timing: {e.get('id')} expects {e.get('sign')!r} for {f}, which no "
                       f"timing measure dates (" + "; ".join(
                           f"{x}: {', '.join(ms)}" for x, ms in dated.items()) + ")")
        elif not any(o.get("kind") in dated[f] for o in outs.get(aid) or []):
            out.append(f"timing: {e.get('id')} expects {e.get('sign')!r} for {f}, but its "
                       f"analysis {aid} has no {_or(dated[f])} among its outputs")
    return out


def _timing_rule_words() -> str:
    """The timing rule as the plan task states it, from the definitions the
    check reads (TIMING_QUESTION_WORDS, TIMING_TOPICS, domain_methods'
    TIMING_FIELD)."""
    _elm()
    import domain_methods as M                                      # noqa: N812
    dated = {}
    for k in M.TIMING_IDS:
        dated.setdefault(M.TIMING_FIELD[k], []).append(k)
    return ("timing: a question that asks about timing (its words: "
            + ", ".join(TIMING_QUESTION_WORDS) + "; not a word that only dates a period, "
            "such as early summer, late 2015 or later years) has, for each such word, a timing "
            "measure of what the word is about among its outputs, for a year the question "
            f"names. What a word is about is read from the words within {TIMING_WINDOW} of "
            "it in its clause, else its clause, else the whole question: " + "; ".join(
                f"{k} dates {', '.join(ws)}" for k, ws in TIMING_TOPICS.items())
            + "; the measure those words are most specific to is the one it needs (a word "
            "several measures list counts for each a share; a tie allows any of the tied "
            "measures), and a word about none of these needs no measure. An expectation "
            "whose sign is "
            + " or ".join(TIMING_SIGNS) + " is tested by an analysis with a timing measure of "
            "the expectation's own field among its outputs (" + "; ".join(
                f"{f}: {', '.join(ms)}" for f, ms in dated.items())
            + "). The rule asks for the measures and never for an expectation or a direction")


# In the order plan_problems applies them: each analysis's rules in turn, then
# the plan's, so its problems come in the order they always came.
PLAN_RULES = (
    PlanRule("fields", "analysis", "fields: catalog fields only (the rows of the data catalog "
             "above)", _check_fields),
    PlanRule("regions", "analysis", "regions: the catalog's regions only", _check_regions),
    PlanRule("methods", "analysis", "methods: the methods above only. The screen is never an "
             "analysis's method: it is declared once, in the plan's own \"screen\" entry, and "
             "an analysis that lists \"screen\" fails the check", _check_methods),
    PlanRule("steps", "analysis", f"steps: the series steps only ({', '.join(SERIES_STEPS)})",
             _check_steps),
    PlanRule("settings", "analysis", "settings: keyed by method id, then setting name, as in "
             "{\"snow_gone\": {\"threshold_mm\": 5.0}}; only a method's own settings, listed "
             "below; leave them out to use the default and the contract's alternatives, which "
             "come with every result, and name another value only with a reason; a timing "
             "measure takes only its default or one of its contract's alternatives, the "
             "settings timing.nc holds; a call's own arguments (overlap's by_band, a timing "
             "measure's against) are not settings: the script passes them at the call",
             _check_settings),
    # A MONTH FOR A REGION'S VALUE ONLY (2026-11-09): the words said "no
    # months", though a region's anomaly or difference of a month has always
    # been a method's answer; the check now refuses the per-cell ones by name
    PlanRule("periods", "analysis", "periods: 2015, WY2016, 2015-JJA; a month (2015-07) only "
             "for a region's value, in an anomaly or a difference that names a region: month "
             "maps are not in the suite; a normal years' mean is a period of its own: normal, "
             "normal-WY, normal-JJA, never of a month", _check_periods),
    PlanRule("outputs", "analysis", PC.RULE_WORDS, _check_outputs),
    PlanRule("baseline", "plan", "baseline: the normal years are the Planner's, given above",
             _check_baseline),
    PlanRule("screen", "plan", "screen: declared once, in this entry; it runs last for a "
             "hypothesis question and first for an exploratory one; its region is one of the "
             "catalog's regions and its year a year", _check_screen),
    PlanRule("timing", "plan", _timing_rule_words, _check_timing),
)


def settings_lines() -> List[str]:
    """Each method's settings as the plan may write them, read from
    domain_methods: name, default, the contract's alternatives, units, and
    PROVISIONAL where the setting is approved for integration only as a
    provisional value. A timing measure is marked: it takes only these."""
    _elm()
    import domain_methods as M                                      # noqa: N812
    lines = _wrapped(f"EACH METHOD'S SETTINGS, from its contract: the only settings "
                     f"\"settings\" may name, each under its method's id. A timing measure "
                     f"({', '.join(M.TIMING_IDS)}) takes only its default or an alternative "
                     f"listed here, because timing.nc holds only those; any other method may "
                     f"take another value, named with a reason.", "", "")
    for mid, m in M.METHODS.items():
        head = f"{mid}, a timing measure (only these values)" if mid in M.TIMING_IDS else mid
        each = []
        for s in m.settings:
            alts = (("alternative " if len(s.alternatives) == 1 else "alternatives ")
                    + ", ".join(repr(x) for x in s.alternatives)
                    if s.alternatives else "no alternative")
            each.append(f"{s.name}, default {s.default!r}, {alts} ({s.units})"
                        + (", PROVISIONAL" if s.status == M.PROVISIONAL else ""))
        lines += _wrapped(f"{head}: " + ("; ".join(each) if each else "no settings") + ".",
                          "  - ", "    ")
    return lines


class PlanNotApproved(Exception):
    """The plan has not passed step 3's review: no lookup is made."""


def _elm():
    if str(_ELM_SRC) not in sys.path:
        sys.path.insert(0, str(_ELM_SRC))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(p) -> Dict[str, Any]:
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return {}


# ─────────────────────────────────────────────────────────────────────
# WHICH STUDIES, AND WHICH SUITES
# ─────────────────────────────────────────────────────────────────────
def _registry_entry(name: Optional[str]) -> Dict[str, Any]:
    """The control's registry entry, through the server's own loader."""
    if not name:
        return {}
    _elm()
    try:
        import domain_controls as DCT                               # noqa: N812
        return (DCT.load().get("controls") or {}).get(name) or {}
    except Exception:                                               # noqa: BLE001
        return {}


def _same_dir(a, b) -> bool:
    try:
        return Path(str(a)).resolve() == Path(str(b)).resolve()
    except OSError:
        return False


def suite_dirs(ctx) -> Dict[str, str]:
    """{run: suite directory} of a suite-bearing study, {} for any other.

    The control's suite is the study's own 03_results/suite/control, or for a
    reused control the registry's suite_dir, accepted only when its manifest
    names the run directory the study's record of the control names (01_inputs/
    domain_control_facts.json): a suite of other output would answer for a run
    this study did not read. A run's suite is its 03_results/suite/<run>."""
    if not getattr(ctx, "is_domain", False):
        return {}
    de = (getattr(ctx, "plan", None) or {}).get("experiment") or {}
    ctl = de.get("control") or {}
    rd = Path(str(getattr(ctx, "run_dir", "") or "."))
    own = rd / "03_results" / "suite"
    out: Dict[str, str] = {}
    if (own / "control" / SUITE_MANIFEST).is_file():
        out["control"] = str(own / "control")
    elif ctl.get("reuse"):
        sd = _registry_entry(ctl.get("name")).get("suite_dir")
        man = _read_json(Path(str(sd)) / SUITE_MANIFEST) if sd else {}
        facts = _read_json(rd / "01_inputs" / "domain_control_facts.json").get("control") or {}
        if man and facts.get("run_dir") and _same_dir(man.get("run_dir"), facts["run_dir"]):
            out["control"] = str(sd)
    if "control" not in out:
        return {}
    for r in de.get("runs") or []:
        name = (r or {}).get("run") or (r or {}).get("name")
        if name and name != "control" and (own / name / SUITE_MANIFEST).is_file():
            out[name] = str(own / name)
    return out


def suite_bearing(ctx) -> bool:
    return bool(suite_dirs(ctx))


def question_type(ctx) -> Dict[str, str]:
    """The Planner's question type (analysis.question_type), never the
    Analyzer's: chosen from the question before any data is seen. A study
    whose Planner recorded none is read as exploratory, which claims no
    tested expectation, and says so."""
    an = ((getattr(ctx, "plan", None) or {}).get("experiment") or {}).get("analysis") or {}
    qt = str(an.get("question_type") or "").strip().lower()
    if qt in QUESTION_TYPES:
        return {"type": qt, "from": "the Planner"}
    return {"type": "exploratory",
            "from": "the Planner recorded none, so it is read as exploratory (no tested "
                    "expectation can be claimed)"}


def planner_normal_years(ctx) -> List[int]:
    """The study's normal years: the Planner's, or, where it named none, the
    control's own from the registry (normal_years_from says which).

    THE CONTROL'S OWN WHEN THE PLANNER NAMES NONE (2026-11-10). A what-if
    study compares a run with its control in one year, so its Planner names
    no normal years; the screen then had no yardstick, every row came back
    with insufficient support and, in an exploratory study where the screen
    runs first, no planned figure could be drawn (the live study of
    2026-11-09). The control's registered normal years are that yardstick."""
    de = (getattr(ctx, "plan", None) or {}).get("experiment") or {}
    ny = [int(y) for y in ((de.get("analysis") or {}).get("normal_years") or [])]
    if ny:
        return ny
    own = _registry_entry((de.get("control") or {}).get("name")).get("normal_years") or []
    return [int(y) for y in own]


def normal_years_from(ctx) -> str:
    de = (getattr(ctx, "plan", None) or {}).get("experiment") or {}
    if (de.get("analysis") or {}).get("normal_years"):
        return "the Planner"
    return ("the control's own (the Planner named none)" if planner_normal_years(ctx)
            else "none named")


# ─────────────────────────────────────────────────────────────────────
# WHAT STEP 2a SEES
# ─────────────────────────────────────────────────────────────────────
def compact_methods() -> str:
    """The methods as the Analyzer reads them: tools/make_methods_doc.py's
    compact view, made from the module now, so it cannot be stale."""
    tools = _ROOT / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    _elm()
    import make_methods_doc                                         # noqa: E402
    return make_methods_doc.build_compact()


def catalog_text(ctx) -> str:
    """The data catalog of the study's control, with the study's runs and
    normal years: coverage, never an outcome (domain_catalog's rule)."""
    _elm()
    import domain_catalog as DC                                     # noqa: N812
    de = (ctx.plan or {}).get("experiment") or {}
    entry = dict(_registry_entry((de.get("control") or {}).get("name")))
    dirs = suite_dirs(ctx)
    if not entry.get("run_dir"):
        man = _read_json(Path(dirs.get("control", "")) / SUITE_MANIFEST)
        entry.update(name=(de.get("control") or {}).get("name"), run_dir=man.get("run_dir"))
    cat = DC.build(entry, study_dir=ctx.run_dir, suite_dir=dirs.get("control"))
    return DC.render(cat)


def configuration(ctx) -> Dict[str, Any]:
    """The study's design: the control, each run's changes with what each
    means and does not mean (the server's change list, as packaged), the
    years and the Planner's analysis. The question's half, never evidence."""
    plan = getattr(ctx, "plan", None) or {}
    de = plan.get("experiment") or {}
    runs = []
    for r in de.get("runs") or []:
        if not isinstance(r, dict):
            continue
        runs.append({"run": r.get("run") or r.get("name"), "reused": r.get("reused"),
                     "changes": [{k: c.get(k) for k in ("id", "params", "meaning",
                                                       "not_meaning")}
                                 for c in (r.get("changes") or []) if isinstance(c, dict)]})
    return {"control": de.get("control"), "runs": runs, "years": de.get("years"),
            "planner_analysis": de.get("analysis") or {}, "goals": plan.get("goals") or [],
            "domain": plan.get("domain")}


def plan_inputs(ctx) -> Dict[str, Any]:
    """Everything step 2a's model call is built from, by name. Built from the
    plan half of the context and the catalogs only: it never calls
    ctx.regional(), ctx.regional_monthly() or ctx.regional_seasonal(), never
    reads 04_analysis, and takes no comparison with observations."""
    qt = question_type(ctx)
    question = (getattr(ctx, "plan", None) or {}).get("question")
    # THE RECIPE THE QUESTION CALLS FOR (analysis_recipes, 2026-11-09): guidance
    # read from the question's own words, holding no outcome; "" for none
    from agents.analysis import analysis_recipes as AR
    return {"question": question,
            "question_type": qt["type"], "question_type_from": qt["from"],
            "configuration": configuration(ctx),
            "catalog": catalog_text(ctx),
            "methods": compact_methods(),
            "recipe": "\n".join(AR.plan_lines(question))}


PLAN_LEAD = """\
You are writing the ANALYSIS PLAN of a whole-domain model study, before any
result is seen. Nothing above holds an outcome: the catalogs describe what can
be analysed and how, never what happened. Do not guess outcomes."""

PLAN_SCOPE = """\
The scripts that follow can look up only what the analyses (or the screen)
declare: anything else is refused until an amendment, appended with a time
stamp after outcomes are seen, declares it, and an amendment never counts as
planned. A method's full contract can be read at any time with
suite.method_contract(method_id)."""


def plan_task() -> str:
    """The task step 2a's model reads, rendered from PLAN_PARTS (what each key
    of the plan holds), PLAN_RULES (what the plan check refuses, in the
    check's own order) and each method's settings in domain_methods, so the
    task says exactly what plan_problems checks."""
    L = [PLAN_LEAD, "", "Decide, from the question as asked:"]
    for key, d in PLAN_PARTS.items():
        L += _wrapped(f"{key}: {d['means']}. "
                      + ("Each holds: " if d["shape"] == "list" else "It holds: ")
                      + "; ".join(f"{p.name} ({_part_means(p)})" for p in d["item"]) + ".",
                      "  - ", "    ")
    L += ["", "THE PLAN CHECK. Code checks the plan before its review, and a plan that breaks",
          "any of these rules fails the review whatever the review says:"]
    for r in PLAN_RULES:
        L += _wrapped(r.text() + ".", "  - ", "    ")
    L += [""] + settings_lines()
    L += [""] + output_lines()
    L += ["", PLAN_SCOPE, "", "Return ONLY JSON:"] + reply_json(PLAN_PARTS)
    return "\n".join(L) + "\n"


def output_lines() -> List[str]:
    """THE REQUIRED OUTPUTS as the plan task says them, from plan_contract:
    what proves an output, each kind with the keys it names, the statuses,
    what is required, and that completion is not support."""
    _elm()
    import domain_methods as M                                      # noqa: N812
    L = _wrapped("THE REQUIRED OUTPUTS. Apart from its scope (what a script may look up), "
                 "each analysis names in \"outputs\" what it must produce to count as "
                 "completed. " + PC.PROOF_WORDS, "", "")
    L += ["The kinds of output, each with the keys it names:"]
    for x in PC.kind_lines(M.TIMING_IDS):
        L += _wrapped(x + ".", "  - ", "    ")
    L += _wrapped(PC.SETTINGS_WORDS + ".", "", "")
    L += _wrapped("At the close each analysis is " + PC.STATUS_WORDS + ". " + PC.REQUIRED_WORDS
                  + ". " + PC.SUPPORT_WORDS + ".", "", "")
    return L


def __getattr__(name: str):
    # PLAN_TASK IS RENDERED, NOT WRITTEN (the user's decision on G part 2,
    # 2026-11-08): made from the definitions on first use, since its settings
    # table reads domain_methods, which this module imports only when needed.
    if name == "PLAN_TASK":
        return plan_task()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def plan_prompt(inputs: Dict[str, Any]) -> str:
    cfg = json.dumps(inputs["configuration"], indent=1, default=str)
    return ("THE USER ASKED:\n    " + str(inputs["question"]) + "\n\n"
            f"QUESTION TYPE (the Planner's, from the question): {inputs['question_type']}"
            f" ({inputs['question_type_from']})\n\n"
            "THE CONFIGURATION (the experiment record and the server's change list):\n"
            + cfg + "\n\nTHE DATA CATALOG:\n" + inputs["catalog"]
            + "\n\nTHE METHODS:\n" + inputs["methods"] + "\n\n"
            + (inputs["recipe"] + "\n\n" if inputs.get("recipe") else "") + plan_task())


def plan_problems(plan: Dict[str, Any], fields: Sequence[str], regions: Sequence[str],
                  planner_years: Sequence[int], question_type: Optional[str] = None
                  ) -> List[str]:
    """What code finds wrong with a plan, by PLAN_RULES (the same rules the
    task states): a field, region, method or setting that does not exist, the
    screen listed as an analysis, settings not keyed by a method, a timing
    setting timing.nc does not hold, a month, normal years other than the
    Planner's, or a screen missing or out of the order the question type sets
    (plan section 3.5)."""
    _elm()
    import domain_methods as M                                      # noqa: N812
    env = {"fields": set(fields), "regions": set(regions), "M": M,
           "planner_years": list(planner_years or []),
           "question_type": question_type or plan.get("question_type"),
           "outputs": PC.plan_index(plan)}
    out: List[str] = []
    for a in plan.get("analyses") or []:
        for r in PLAN_RULES:
            if r.scope == "analysis":
                out += r.check(a, env)
    for r in PLAN_RULES:
        if r.scope == "plan":
            out += r.check(plan, env)
    return out


def _catalog_names(ctx) -> Tuple[List[str], List[str]]:
    """The catalog's fields and the control's regions, by name."""
    _elm()
    import domain_methods as M                                      # noqa: N812
    dirs = suite_dirs(ctx)
    man = _read_json(Path(dirs.get("control", "")) / SUITE_MANIFEST)
    return M.catalog_fields(), [r.get("name") for r in man.get("regions") or []]


def _plain_parts(plan: Dict[str, Any]) -> Dict[str, Any]:
    """The plan with every "words" part (what, why, question, text) through
    strip_dashes, by PLAN_PARTS; names, ids and lists are left as written."""
    for key, d in PLAN_PARTS.items():
        words = [p.name for p in d["item"] if p.holds == "words"]
        items = plan.get(key) if d["shape"] == "list" else [plan.get(key)]
        for it in items or []:
            if isinstance(it, dict):
                for w in words:
                    if isinstance(it.get(w), str):
                        it[w] = strip_dashes(it[w])
    return plan


def _ask(client, label: str, model: Optional[str], prompt: str) -> str:
    if client is None:
        from agents.analysis.step2_investigate import analyzer_model
        from agents.llm_agent import SimpleLLMClient
        client = SimpleLLMClient(model=analyzer_model(model))
        client.label = label
    return client.ask([{"role": "user", "content": prompt}])


def _plan_record(ctx, inputs: Dict[str, Any], reply: str, attempt: int) -> Dict[str, Any]:
    """The plan file's record of one reply: its keys by PLAN_PARTS (a key of
    the wrong shape is replaced by an empty one, which the check then names),
    the model's prose without dashes, what was seen and withheld, and code's
    problems."""
    from agents.analysis.step2_investigate import _parse
    try:
        spec = _parse(reply)
    except (ValueError, json.JSONDecodeError) as e:
        spec = {"unreadable": str(e)}
    fields, regions = _catalog_names(ctx)
    plan = {"plan_version": PLAN_VERSION, "written_at": _now(), "step": "2a",
            "question": inputs["question"], "question_type": inputs["question_type"],
            "question_type_from": inputs["question_type_from"]}
    if attempt > 1:
        plan.update(attempt=attempt, rewrite_of=PLAN_V1_FILE)
    for k in PLAN_KEYS:
        default = _shape_default(k)
        v = spec.get(k)
        plan[k] = v if isinstance(v, type(default)) else default
    if not plan["baseline"].get("normal_years") and planner_normal_years(ctx):
        plan["baseline"]["normal_years"] = planner_normal_years(ctx)
    if plan["baseline"].get("normal_years"):
        plan["baseline"]["normal_years_from"] = normal_years_from(ctx)
    # NO DASHES IN THE MODEL'S WORDS, before the file is written and hashed
    _plain_parts(plan)
    plan["seen"] = list(SEEN) + ([SEEN_RECIPE] if inputs.get("recipe") else [])
    plan["withheld"] = list(WITHHELD)
    plan["problems"] = plan_problems(plan, fields, regions, planner_normal_years(ctx),
                                     plan["question_type"]) + \
        ([f"the reply was not readable JSON: {spec['unreadable']}"]
         if "unreadable" in spec else [])
    return plan


def write_plan(ctx, out_dir, client=None, model: Optional[str] = None) -> Dict[str, Any]:
    """Step 2a: the plan, from plan_inputs() only, written once. A plan
    already on disk is returned as it is (a change after outcomes is an
    amendment); the one exception is rewrite_plan, before any lookup."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / PLAN_FILE
    if p.is_file():
        return load_plan(out)
    inputs = plan_inputs(ctx)
    prompt = plan_prompt(inputs)
    reply = _ask(client, PLAN_LABEL, model, prompt)
    plan = _plan_record(ctx, inputs, reply, 1)
    p.write_text(json.dumps(plan, indent=1, default=str))
    from agents.analysis.script_runner import save_exchange
    save_exchange(out, "step2a", 1, prompt, reply)
    return plan


# ─────────────────────────────────────────────────────────────────────
# ONE REWRITE OF A FAILED PLAN, BEFORE ANY OUTCOME IS LOOKED UP
# (the user's decision on G part 2, 2026-11-08)
# ─────────────────────────────────────────────────────────────────────
# Questions 2 and 3 of G part 2 stopped at their plans: code found one fixable
# problem in each, and the study ended with no lookup. A plan that fails its
# review (code's check, or the model review says it does not follow, or the
# review's reply cannot be read) is now written once more by step 2a, from the
# same inputs as the first time (plan_inputs: nothing that carries an
# outcome) plus its first plan and what the review found. The rewrite becomes
# analysis_plan.json and gets exactly the same validation: plan_problems and
# the same review prompt, which is not told it is reviewing a rewrite. The
# first plan and its review are kept as analysis_plan.v1.json and
# analysis_plan_review.v1.json. At most one rewrite, and none once any lookup
# is logged or a screen record exists.
REWRITE_ASK = """\
YOUR FIRST PLAN DID NOT PASS ITS REVIEW. Write the plan once more, from the
same question, configuration and catalogs, fixing what the review found. No
result has been looked up. The new plan is checked by code and reviewed
exactly as the first was; if it fails too, the study stops with no analysis.
This is the only rewrite."""


def review_findings(rv: Dict[str, Any], plan: Optional[Dict[str, Any]] = None
                    ) -> Dict[str, Any]:
    """What a review found, split by who found it: the code check's problems
    and the model review's verdict, its why and its problems. A review
    record from before the split (2026-11-08) is read from its problems,
    where code's carry the prefix "code: "."""
    probs = [str(p) for p in rv.get("problems") or []]
    code = rv.get("code_problems")
    if code is None:
        code = [p[len("code: "):] for p in probs if p.startswith("code: ")] or \
            list((plan or {}).get("problems") or [])
    model = rv.get("model_problems")
    if model is None:
        model = [p for p in probs if not p.startswith("code: ")]
    # the model's words, wherever they are copied next, carry no dashes (a
    # record written before 2026-11-08 may still hold them)
    return {"code_problems": list(code), "follows": rv.get("follows"),
            "why": strip_dashes(rv.get("why") or ""),
            "model_problems": strip_dashes_in(list(model))}


def review_reason(rv: Dict[str, Any], plan: Optional[Dict[str, Any]] = None) -> str:
    """Why a review did not pass, in the order it is read: the code check's
    problems first when there are any, then the model review's verdict and
    problems, each labelled. Never the model's approving why alone: question 2
    of G part 2 gave as its reason the model's "follows" text, while code had
    failed the plan."""
    f = review_findings(rv, plan)
    parts = []
    if f["code_problems"]:
        n = len(f["code_problems"])
        parts.append(f"the code check found {n} problem{'s' if n > 1 else ''}: "
                     + "; ".join(f["code_problems"]))
    mp = f["model_problems"]
    listed = (f" and listed {len(mp)} problem{'s' if len(mp) > 1 else ''}: " + "; ".join(mp)
              if mp else " and listed no problem")
    if f["follows"] is True:
        parts.append("the model review said the plan follows from the question" + listed)
    elif f["follows"] is False:
        parts.append("the model review said the plan does not follow from the question"
                     + (f" ({f['why']})" if f["why"] else "") + listed)
    elif f["why"] or mp:
        # a record from before the split keeps no verdict: its why, labelled
        parts.append("the model review" + (f" said: {f['why']}" if f["why"] else "") + listed)
    return "; ".join(parts) or "the review gives no reason"


def can_rewrite(out_dir) -> Tuple[bool, str]:
    """(allowed, why not): one rewrite, only of a plan whose review failed,
    and only before any lookup is logged or a screen record exists."""
    out = Path(out_dir)
    if (out / PLAN_V1_FILE).exists():
        return False, ("the plan was rewritten once already: there is no third attempt "
                       f"({PLAN_V1_FILE} holds the first)")
    if not (out / PLAN_FILE).is_file():
        return False, "there is no plan to rewrite"
    rv = load_review(out)
    if rv.get("passed") is not False:
        return False, "a plan is rewritten only after its review failed"
    n = _lines(out / LOOKUPS_FILE)
    if n or (out / SCREEN_FILE).is_file():
        return False, (f"a plan is rewritten only before any outcome is looked up, and "
                       f"{LOOKUPS_FILE} holds {n} line(s)"
                       + (f" and {SCREEN_FILE} exists" if (out / SCREEN_FILE).is_file() else ""))
    return True, ""


def rewrite_plan(ctx, out_dir, client=None, model: Optional[str] = None) -> Dict[str, Any]:
    """Step 2a once more, after a failed review: the same inputs, the first
    plan and what its review found. Raises PlanNotApproved when a rewrite is
    not allowed (can_rewrite). The first plan and its review are moved to
    PLAN_V1_FILE and REVIEW_V1_FILE as they were, the rewrite is written to
    PLAN_FILE with the same check, and the exchange is saved as round 2."""
    out = Path(out_dir)
    ok, why = can_rewrite(out)
    if not ok:
        raise PlanNotApproved(f"no rewrite of the analysis plan: {why}")
    first, rv = load_plan(out), load_review(out)
    f = review_findings(rv, first)
    inputs = plan_inputs(ctx)
    shown = {k: first.get(k) for k in PLAN_KEYS}
    found = ["the code check: " + ("; ".join(f["code_problems"]) or "no problem"),
             "the model review: " + ("the plan follows from the question"
                                     if f["follows"] is True else
                                     "the plan does not follow from the question"
                                     if f["follows"] is False else "no verdict")
             + (f" ({f['why']})" if f["follows"] is False and f["why"] else "")
             + ("; its problems: " + "; ".join(f["model_problems"])
                if f["model_problems"] else "; it listed no problem")]
    prompt = (plan_prompt(inputs) + "\n" + REWRITE_ASK + "\n\nYOUR FIRST PLAN:\n"
              + json.dumps(shown, indent=1, default=str) + "\n\nWHAT ITS REVIEW FOUND:\n"
              + "\n".join(f"  - {x}" for x in found) + "\n")
    reply = _ask(client, REWRITE_LABEL, model, prompt)
    plan = _plan_record(ctx, inputs, reply, 2)
    # the first version, kept as it was: its review's hash still ties the two
    os.replace(out / PLAN_FILE, out / PLAN_V1_FILE)
    os.replace(out / REVIEW_FILE, out / REVIEW_V1_FILE)
    (out / PLAN_FILE).write_text(json.dumps(plan, indent=1, default=str))
    from agents.analysis.script_runner import save_exchange
    save_exchange(out, "step2a", 2, prompt, reply)
    return plan


def load_plan(out_dir) -> Dict[str, Any]:
    return _read_json(Path(out_dir) / PLAN_FILE)


def plan_sha256(out_dir) -> Optional[str]:
    p = Path(out_dir) / PLAN_FILE
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


# ─────────────────────────────────────────────────────────────────────
# THE REVIEW: STEP 3'S, ONCE
# ─────────────────────────────────────────────────────────────────────
def record_review(out_dir, passed: bool, why: str, problems: Sequence[str] = (),
                  reviewer: str = "step 3's model review", exchange=None,
                  follows: Optional[bool] = None,
                  model_problems: Optional[Sequence[str]] = None,
                  code_problems: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Step 3's review of the plan, written once, tied to the plan's hash: a
    plan changed after its review is no longer the plan that passed. Beside
    the combined `problems` (the model's, then code's prefixed "code: ") it
    keeps who found what: the model review's verdict (`follows`) and
    problems, and the code check's problems, so a failed review gives its
    real reason (review_reason). The model's words carry no dashes."""
    out = Path(out_dir)
    p = out / REVIEW_FILE
    if p.is_file():
        return load_review(out)
    plan = load_plan(out)
    rec = {"rule": RULE, "passed": bool(passed), "why": strip_dashes(why),
           "problems": strip_dashes_in(list(problems)), "reviewer": reviewer, "at": _now(),
           "plan_sha256": plan_sha256(out), "exchange": exchange or None,
           "approval": APPROVES, "attempt": int(plan.get("attempt") or 1)}
    if follows is not None or model_problems is not None or code_problems is not None:
        rec.update(follows=follows, model_problems=strip_dashes_in(list(model_problems or [])),
                   code_problems=list(code_problems or []))
    p.write_text(json.dumps(rec, indent=1, default=str))
    return rec


def load_review(out_dir) -> Dict[str, Any]:
    return _read_json(Path(out_dir) / REVIEW_FILE)


def plan_passed(out_dir) -> Tuple[bool, str]:
    """(passed, why): the plan exists, step 3's review passed it, and the plan
    file still hashes to what was reviewed. A failed review's why is its real
    reason (review_reason): the code check's problems first, then the model
    review's verdict and problems."""
    out = Path(out_dir)
    if not (out / PLAN_FILE).is_file():
        return False, "there is no analysis plan (step 2a has not written one)"
    rv = load_review(out)
    if not rv:
        return False, "the analysis plan has not been reviewed (step 3's review)"
    if not rv.get("passed"):
        return False, (f"the analysis plan did not pass step 3's review: "
                       f"{review_reason(rv, load_plan(out))}")
    if rv.get("plan_sha256") != plan_sha256(out):
        return False, "the analysis plan changed after its review"
    return True, "the analysis plan passed step 3's review"


def plan_versions(out_dir) -> List[Dict[str, Any]]:
    """Each version of the plan on record with its review, the first first:
    its files, whether its review passed, and what the code check and the
    model review found. One entry when the plan was never rewritten."""
    out = Path(out_dir)
    pairs = [(PLAN_V1_FILE, REVIEW_V1_FILE)] if (out / PLAN_V1_FILE).is_file() else []
    pairs.append((PLAN_FILE, REVIEW_FILE))
    rows = []
    for i, (pf, rf) in enumerate(pairs, 1):
        plan, rv = _read_json(out / pf), _read_json(out / rf)
        if not plan:
            continue
        f = review_findings(rv, plan) if rv else {}
        rows.append({"attempt": i, "plan_file": pf, "review_file": rf if rv else None,
                     "written_at": plan.get("written_at"), "reviewed": bool(rv),
                     "passed": bool(rv.get("passed")) if rv else False,
                     "code_problems": f.get("code_problems") or list(plan.get("problems") or []),
                     "follows": f.get("follows"), "model_why": f.get("why"),
                     "model_problems": f.get("model_problems") or [],
                     "reason": (review_reason(rv, plan) if rv and not rv.get("passed") else
                                None if rv else "not reviewed")})
    return rows


def check_gate(out_dir) -> None:
    ok, why = plan_passed(out_dir)
    if not ok:
        raise PlanNotApproved(f"no lookup before the plan has passed step 3's review: {why}")


# ─────────────────────────────────────────────────────────────────────
# AMENDMENTS: APPENDED, NEVER PLANNED
# ─────────────────────────────────────────────────────────────────────
def load_amendments(out_dir) -> List[Dict[str, Any]]:
    p = Path(out_dir) / AMENDMENTS_FILE
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _lines(p: Path) -> int:
    return sum(1 for x in p.read_text().splitlines() if x.strip()) if p.is_file() else 0


def append_amendment(out_dir, why: str, adds: Dict[str, Any]) -> Dict[str, Any]:
    """A change to the plan after outcomes were seen: appended with a time
    stamp and the number of lookups logged before it, never rewritten. Its
    analyses never count as planned: their findings are at most exploratory
    or an association. Only after the plan has passed its review. An
    amendment appended after the planned phase was closed is marked so, and
    in a hypothesis study one appended after the screen has run follows what
    the screen turned up: both carry the label "exploratory" (the user's
    decision on G part 2, 2026-11-08: new post-screen analyses are labelled
    exploratory; until then such an amendment was "found by screening, not
    tested"), and after_screen says it came after the screen. The why is the
    model's, kept without dashes."""
    out = Path(out_dir)
    check_gate(out)
    if not str(why or "").strip():
        raise ValueError("an amendment says why")
    # ITS SETTINGS ARE {method: {setting: value}} (2026-11-10): one written as
    # {"normal_years": [...]} was recorded and then stopped the whole analysis
    # where it was read. It is refused here, with what it should have been.
    st = adds.get("settings")
    if st and not (isinstance(st, dict) and all(isinstance(v, dict) for v in st.values())):
        bad = st if not isinstance(st, dict) else \
            sorted(k for k, v in st.items() if not isinstance(v, dict))
        raise ValueError(f"an amendment's settings are {{method: {{setting: value}}}}, and "
                         f"{bad} is not a method with its settings; the normal years are the "
                         f"plan's baseline and no setting of a method")
    keep = {k: adds.get(k) for k in SCOPE_KEYS + ("settings",) if adds.get(k)}
    rec = {"id": f"A{len(load_amendments(out)) + 1}", "at": _now(), "why": strip_dashes(why),
           "adds": keep, "planned": False,
           "after_lookups": _lines(out / LOOKUPS_FILE)}
    if (out / PLANNED_PHASE_FILE).is_file():
        rec.update(after_close=True, label=EXPLORATORY_LABEL)
    if load_plan(out).get("question_type") == "hypothesis" and (out / SCREEN_FILE).is_file():
        rec.update(after_screen=True, label=EXPLORATORY_LABEL)
    with open(out / AMENDMENTS_FILE, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


def lookups(out_dir) -> List[Dict[str, Any]]:
    p = Path(out_dir) / LOOKUPS_FILE
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def contract_reads(out_dir) -> List[Dict[str, Any]]:
    """The method contracts read through suite.method_contract: no lookup."""
    p = Path(out_dir) / CONTRACT_READS_FILE
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def lookups_of(out_dir, amendment_id: str) -> List[Dict[str, Any]]:
    """The lookups an amendment led to."""
    return [r for r in lookups(out_dir) if r.get("amendment") == amendment_id]


def named_settings(plan: Dict[str, Any], amendment: Optional[Dict[str, Any]] = None,
                   amendments: Sequence[Dict[str, Any]] = ()
                   ) -> Dict[str, Dict[str, List[Any]]]:
    """{method: {setting: [values]}} the plan names, and the amendments
    given: what settings_for accepts beside the contract's own. Which
    analysis or amendment a value belongs to is check_lookup's to enforce."""
    out: Dict[str, Dict[str, List[Any]]] = {}
    parts = [a.get("settings") or {} for a in plan.get("analyses") or [] if isinstance(a, dict)]
    for am in ([amendment] if amendment else []) + list(amendments or []):
        parts.append((am.get("adds") or {}).get("settings") or {})
    for st in parts:
        for m, kv in (st.items() if isinstance(st, dict) else []):
            # a recorded amendment from before 2026-11-10 may hold settings
            # that are no {setting: value}: they name no setting, and are skipped
            for k, v in (kv.items() if isinstance(kv, dict) else []):
                out.setdefault(m, {}).setdefault(k, [])
                for x in (v if isinstance(v, list) else [v]):
                    if x not in out[m][k]:
                        out[m][k].append(x)
    return out


# ─────────────────────────────────────────────────────────────────────
# THE GATE: WHAT THE REVIEWED PLAN DECLARES, AND IN WHAT ORDER
# (closing package E, 2026-11-05)
# ─────────────────────────────────────────────────────────────────────
def _year_of(p) -> Optional[int]:
    s = str(p)
    if s.startswith("WY") and s[2:].isdigit():
        return int(s[2:])
    return int(s[:4]) if s[:4].isdigit() else None


def _kind_of(p) -> Optional[str]:
    _elm()
    import domain_methods as M                                      # noqa: N812
    try:
        return M.parse_period(str(p))[0]
    except M.MethodError:
        return None


def scope_items(plan: Dict[str, Any], amendments: Sequence[Dict[str, Any]] = ()
                ) -> List[Dict[str, Any]]:
    """What may be looked up: one item per analysis of the reviewed plan,
    then one per amendment. An amendment declares exactly the values it
    names; a key it leaves out (fields, regions, periods, methods, steps) is
    that key over the whole plan, the screen's region and the periods of the
    screen's year included, and its settings are the plan's and its own."""
    _elm()
    import domain_methods as M                                      # noqa: N812
    union: Dict[str, List[Any]] = {k: [] for k in SCOPE_KEYS}

    def add(k, vals):
        for v in vals or []:
            if v not in union[k]:
                union[k].append(v)
    items = []
    for a in plan.get("analyses") or []:
        if not isinstance(a, dict):
            continue
        it = {k: list(a.get(k) or []) for k in SCOPE_KEYS}
        it.update(id=a.get("id"), amendment=None,
                  settings=named_settings({"analyses": [a]}))
        items.append(it)
        for k in SCOPE_KEYS:
            add(k, it[k])
    sc = plan.get("screen") or {}
    if sc.get("region"):
        add("regions", [sc["region"]])
    if str(sc.get("year")).isdigit():
        add("periods", M.screen_periods(int(sc["year"])))
    for am in amendments or []:
        adds = am.get("adds") or {}
        it = {k: list(adds.get(k) or union[k]) for k in SCOPE_KEYS}
        it.update(id=am.get("id"), amendment=am.get("id"), label=am.get("label"),
                  settings=named_settings(plan, amendment=am))
        items.append(it)
    return items


def _lookup(ask: Dict[str, Any]) -> Dict[str, Any]:
    """A call's scope: its method, field, region, period (a timing call's
    year, a series' step) and settings, as the call was made."""
    k = ask.get("keys") or {}
    return {"call": ask.get("call"), "method": ask.get("method"), "field": k.get("field"),
            "region": k.get("region"), "period": k.get("period"), "year": k.get("year"),
            "step": k.get("step"), "settings": dict(ask.get("settings") or {})}


def _not_declared(it: Dict[str, Any], lk: Dict[str, Any]) -> Optional[str]:
    """None when the item declares the lookup, else what it does not
    declare (the first such key)."""
    _elm()
    import domain_methods as M                                      # noqa: N812
    m, call = lk["method"], lk["call"]
    if m is not None and m not in it["methods"]:
        return f"method {m!r}"
    if lk["field"] is not None and lk["field"] not in it["fields"]:
        return f"field {lk['field']!r}"
    if lk["region"] is not None and lk["region"] not in it["regions"]:
        return f"region {lk['region']!r}"
    periods = [str(p) for p in it["periods"]]
    if call == "screen":
        if str(lk["year"]) not in periods:
            return f"year {lk['year']!r}"
    elif lk["period"] is not None:
        if str(lk["period"]) not in periods:
            return f"period {lk['period']!r}"
    elif lk["year"] is not None and m in M.TIMING_KIND:
        y = int(lk["year"])
        if not {str(y), M.period_label(M.TIMING_KIND[m], y)} & set(periods):
            return f"period {lk['year']!r}"
    elif lk["step"] is not None:
        if lk["step"] not in it["steps"] and lk["step"] not in {_kind_of(p) for p in periods}:
            return f"series step {lk['step']!r}"
    for k, v in lk["settings"].items():
        if m not in M.METHODS or k not in M.METHODS[m].defaults():
            continue                      # settings_for refuses it, with its own reason
        s = M.METHODS[m].setting(k)
        if v == s.default or v in s.alternatives:
            continue
        if v not in ((it.get("settings") or {}).get(m) or {}).get(k, []):
            return f"setting {m}.{k}={v!r}"
    return None


def _describe(lk: Dict[str, Any]) -> str:
    bits = [f"{k} {lk[k]!r}" for k in ("method", "field", "region", "period", "year", "step")
            if lk.get(k) is not None]
    bits += [f"setting {k}={v!r}" for k, v in lk["settings"].items()]
    return f"{lk['call']} ({', '.join(bits)})"


def _timing_note(lk: Dict[str, Any]) -> str:
    """For a timing setting that is neither the default nor a contract
    alternative: the settings timing.nc holds, since no amendment can add one."""
    _elm()
    import domain_methods as M                                      # noqa: N812
    m = lk["method"]
    if m not in M.TIMING_IDS:
        return ""
    out = []
    for k, v in lk["settings"].items():
        if k in M.METHODS[m].defaults():
            st = M.METHODS[m].setting(k)
            if v != st.default and v not in st.alternatives:
                out.append(f"{m} {k}={v!r} is not a setting timing.nc holds (it holds the "
                           f"default {st.default!r} and the contract's alternatives "
                           f"{list(st.alternatives)}); a timing measure is never computed on "
                           f"the fly")
    return ("; " + "; ".join(out)) if out else ""


def declared_lookups(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every combination the plan's analyses' scopes allow: one per method
    (a map when an analysis names none), field, region and period. overlap and
    elevation_split take maps already looked up (one per region); a timing
    measure reads its own field (one per region and period). What MAY be
    looked up, never what must be: since 2026-11-08 an analysis is judged by
    its required outputs (plan_contract), and these combinations decide
    neither the screen's order nor any status."""
    _elm()
    import domain_methods as M                                      # noqa: N812
    out = []
    for a in plan.get("analyses") or []:
        if not isinstance(a, dict):
            continue
        fs, rs, ps = a.get("fields") or [None], a.get("regions") or [None], \
            a.get("periods") or [None]
        for m in a.get("methods") or [None]:
            if m == "screen":
                continue
            if m in ARRAY_METHODS:
                combos = [(None, r, None) for r in rs]
            elif m in M.TIMING_IDS:
                combos = [(None, r, p) for r in rs for p in ps]
            elif m is None:
                combos = [(f, None, p) for f in fs for p in ps]
            else:
                combos = [(f, r, p) for f in fs for r in rs for p in ps]
            out += [{"item": a.get("id"), "method": m, "field": f, "region": r, "period": p}
                    for f, r, p in combos]
    return out


def _has_run(d: Dict[str, Any], rec: Dict[str, Any]) -> bool:
    """A declared lookup has run when a call the gate passed as planned under
    its analysis falls on it (answered, or refused by the data with a reason)."""
    if rec.get("planned") is not True or d["item"] not in (rec.get("plan_items") or []):
        return False
    k = rec.get("keys") or {}
    if d["method"] is None:
        if rec.get("call") == "get_map":
            return k.get("field") == d["field"] and \
                (d["period"] is None or str(k.get("period")) == str(d["period"]))
        if rec.get("call") == "get_series":
            return k.get("field") == d["field"] and \
                (d["period"] is None or k.get("step") == _kind_of(d["period"]))
        return False
    if rec.get("method") != d["method"]:
        return False
    if d["field"] is not None and k.get("field") != d["field"]:
        return False
    if d["region"] is not None and k.get("region") != d["region"]:
        return False
    if d["period"] is not None:
        if "period" in k:
            return str(k["period"]) == str(d["period"])
        if "year" in k:
            return _year_of(d["period"]) == int(k["year"])
    return True


def pending_lookups(out_dir, plan: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """The scope combinations (declared_lookups) no planned lookup has made
    yet: a description, never a requirement."""
    plan = plan if plan is not None else load_plan(out_dir)
    done = lookups(out_dir)
    return [d for d in declared_lookups(plan) if not any(_has_run(d, r) for r in done)]


def _pending_words(d: Dict[str, Any]) -> str:
    return " ".join(str(x) for x in (d["item"], d["method"] or "map", d["field"],
                                     d["region"] and repr(d["region"]), d["period"])
                    if x is not None)


# ─────────────────────────────────────────────────────────────────────
# THE CLOSE OF THE PLANNED-ANALYSIS PHASE
# (the user's decision on G part 2, 2026-11-08)
# ─────────────────────────────────────────────────────────────────────
# A hypothesis study runs its planned analyses first and its screen last.
# Until 2026-11-08 the screen waited for every lookup the plan declared, so a
# plan that declared more field, region and period combinations than its
# figures made never reached its screen (question 1 of G part 2: 70 planned
# lookups made, the screen refused for 69 not made). Now the framework closes
# the planned phase explicitly, once: every analysis of the reviewed plan is
# marked completed, failed or skipped, each with its reason, and only then
# may the screen run. Every lookup after the close is exploratory. An
# analysis one of the plan's expectations names is required: a required
# analysis skipped or failed means the hypothesis was not fully tested, and
# it never counts as acceptance success (acceptance()).
# BY ITS REQUIRED OUTPUTS, NOT ITS SCOPE (the user's decision after G part 2's
# fixes, 2026-11-08): until then an analysis was completed only when every
# combination of its fields, regions, periods and methods had run, so question
# 1's P1, P3, P5 and P8, carried out, read as skipped (7 of 15, 11 of 21, 13
# of 24 and 13 of 25 combinations). Now each analysis names its outputs and
# plan_contract.evaluate decides each one: produced, not estimable or unmet,
# by a lookup that matches it exactly; a figure that ran proves nothing.
PHASE_STATUSES = PC.STATUSES
PHASE_RULE = (f"{PC.PROOF_WORDS} An analysis is {PC.STATUS_WORDS}. {PC.REQUIRED_WORDS}. "
              f"{PC.SUPPORT_WORDS}")
NOT_CARRIED = PC.NOT_CARRIED


def load_planned_phase(out_dir) -> Dict[str, Any]:
    return _read_json(Path(out_dir) / PLANNED_PHASE_FILE)


def required_analyses(plan: Dict[str, Any]) -> Dict[str, List[str]]:
    """{analysis id: the expectations that name it}: the analyses a hypothesis
    study must carry out to test what it expected. None for an exploratory
    question, which states no expectation."""
    if plan.get("question_type") != "hypothesis":
        return {}
    ids = {a.get("id") for a in plan.get("analyses") or [] if isinstance(a, dict)}
    out: Dict[str, List[str]] = {}
    for e in plan.get("expectations") or []:
        if isinstance(e, dict) and e.get("analysis") in ids:
            out.setdefault(e["analysis"], []).append(e.get("id"))
    return out


def phase_statuses(out_dir, plan: Optional[Dict[str, Any]] = None,
                   figures: Sequence[Dict[str, Any]] = (),
                   skipped: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """Each analysis of the reviewed plan at the close, by its required
    outputs (plan_contract.evaluate): each output produced, not estimable or
    unmet, with the record that settled it and every attempt at it; the
    analysis's status and reason; whether it is required; and its attempts in
    order. `figures` are the rounds' figures ({id, plan_items, ok, error,
    round, lines}); `skipped` is step 2b's own {analysis: reason}. Its scope
    (what it may look up) is listed beside, and decides nothing."""
    _elm()
    import domain_methods as M                                      # noqa: N812
    plan = plan if plan is not None else load_plan(out_dir)
    log = lookups(out_dir)
    req = required_analyses(plan)
    said = {str(k): v for k, v in (skipped or {}).items() if str(v or "").strip()}
    index = PC.plan_index(plan)
    out = []
    for a in plan.get("analyses") or []:
        if not isinstance(a, dict):
            continue
        aid = a.get("id")
        row = PC.evaluate(a, log, figures, said.get(str(aid)), timing_ids=M.TIMING_IDS,
                          strip=strip_dashes, index=index, defaults=_method_defaults)
        row.update(question=a.get("question"), required=aid in req,
                   required_by=req.get(aid, []))
        out.append(row)
    return out


def close_planned_phase(out_dir, when: str, by: str,
                        figures: Sequence[Dict[str, Any]] = (),
                        skipped: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """The close of the planned-analysis phase, written ONCE by the framework
    (never by a figure script) to planned_phase.json: the time, the number of
    lookup lines before it, when and by what it was closed, the rule, every
    analysis of the reviewed plan with its outputs and status
    (phase_statuses), and the plan's expectations as they were stated. It
    holds no judgement of support: that is the traced tested claims'. A
    phase already closed is returned as it is. Only once the plan has passed
    its review."""
    out = Path(out_dir)
    p = out / PLANNED_PHASE_FILE
    if p.is_file():
        return load_planned_phase(out)
    if os.environ.get(IN_SCRIPT_ENV):
        raise PlanNotApproved("the planned-analysis phase is closed by the framework, never by "
                              "a figure script")
    check_gate(out)
    plan = load_plan(out)
    rows = phase_statuses(out, plan, figures, skipped)
    rec = {"file": PLANNED_PHASE_FILE, "at": _now(), "lookups_before": _lines(out / LOOKUPS_FILE),
           "closed": when, "by": by, "question_type": plan.get("question_type"),
           "plan_file": PLAN_FILE, "plan_sha256": plan_sha256(out), "rule": PHASE_RULE,
           "counts": {s: sum(r["status"] == s for r in rows) for s in PHASE_STATUSES},
           "required": [r["id"] for r in rows if r["required"]], "analyses": rows,
           "expectations": [{k: (strip_dashes(e.get(k)) if k == "text" else e.get(k))
                             for k in ("id", "analysis", "field", "sign", "text")}
                            for e in plan.get("expectations") or [] if isinstance(e, dict)],
           "support": PC.SUPPORT_WORDS}
    p.write_text(json.dumps(rec, indent=1, default=str))
    return rec


def untestable(phase: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Each expectation whose analysis has a required output found not
    estimable: which outputs, their records and the method's words, and
    whether every output of that analysis is so (then nothing in the run can
    test the expectation)."""
    rows = {r.get("id"): r for r in (phase or {}).get("analyses") or []}
    out = []
    for e in (phase or {}).get("expectations") or []:
        r = rows.get(e.get("analysis")) or {}
        ne = [o for o in r.get("outputs") or [] if o.get("outcome") == PC.NOT_ESTIMABLE]
        if ne:
            out.append({"expectation": e.get("id"), "analysis": r.get("id"),
                        "outputs": [{"output": o.get("id"), "record": o.get("record"),
                                     "why": o.get("why")} for o in ne],
                        "none_estimable": len(ne) == len(r.get("outputs") or [])})
    return out


def acceptance(phase: Optional[Dict[str, Any]], plan_ok: bool = True) -> Dict[str, Any]:
    """What keeps a study from acceptance success: each required analysis
    failed or skipped (an output unmet), with its reason. Beside it, kept
    apart: the required outputs found not estimable (their task done, their
    expectation without an estimate) and the required analyses completed
    after an earlier failure. Never a judgement of support. A study whose
    plan did not pass, or with no close on record, has no tested analysis to
    count."""
    if not plan_ok:
        return {"ok": False, "blockers": [], "not_estimable": [], "recovered": [],
                "untestable": [],
                "why": "the analysis plan did not pass its review, so no planned analysis "
                       "was carried out"}
    if not phase:
        return {"ok": False, "blockers": [], "not_estimable": [], "recovered": [],
                "untestable": [],
                "why": f"no close of the planned phase is on record ({PLANNED_PHASE_FILE})"}
    req = [r for r in phase.get("analyses") or [] if r.get("required")]
    blockers = [{"analysis": r["id"], "status": r["status"], "reason": r["reason"],
                 "unmet": r.get("unmet") or [], "required_by": r.get("required_by") or []}
                for r in req if r.get("status") in (PC.FAILED, PC.SKIPPED)]
    out = {"ok": not blockers, "blockers": blockers,
           "not_estimable": [{"analysis": r["id"], "output": o["id"], "record": o.get("record"),
                              "why": o.get("why"), "required_by": r.get("required_by") or []}
                             for r in req for o in r.get("outputs") or []
                             if o.get("outcome") == PC.NOT_ESTIMABLE],
           "recovered": [{"analysis": r["id"], "reason": r["reason"]} for r in req
                         if r.get("status") == PC.COMPLETED_AFTER_FAILURE],
           # NOT ESTIMABLE, MADE PLAIN (the user's decision of 2026-11-08 on the
           # open points): its task is complete and blocks nothing, and the
           # expectation it serves cannot be tested on that output from this run
           "untestable": untestable(phase),
           "rule": "a required analysis (one the plan's expectations name) failed or skipped, "
                   "an output of it unmet, means the hypothesis was not fully tested; it never "
                   "counts as acceptance success. An output found not estimable completes its "
                   "task and gives its expectation no estimate",
           "support": PC.SUPPORT_WORDS}
    if phase.get("question_type") != "hypothesis":
        out["why"] = "an exploratory question has no required analysis"
    return out


def check_lookup(out_dir, ask: Dict[str, Any], amendment: Optional[str] = None
                 ) -> Dict[str, Any]:
    """The gate of one call on `suite`: the marks its log record carries, or
    PlanNotApproved with the reason. In order: the plan has passed step 3's
    review; a call that reads no outcome passes; a lookup must fall inside an
    analysis of the reviewed plan (planned) or inside an amendment appended
    before it (the script's own amendment first; marked not planned, linked
    to that amendment); the screen must be the plan's (or an amendment's);
    then the order the question type sets (plan section 3.5): an exploratory
    question's other lookups wait for its screen, and a hypothesis
    question's screen waits for the close of the planned-analysis phase
    (planned_phase.json, written by the framework once every figure that does
    not run the screen has run). EVERY lookup after that close is marked not
    planned, "exploratory" and after_close, and after_screen once the screen
    has run; the hypothesis screen itself keeps "found by screening, not
    tested" (the user's decision on G part 2, 2026-11-08). Lines logged
    before the close keep their marks."""
    out = Path(out_dir)
    check_gate(out)
    call = ask.get("call")
    if call in SUPPORT_CALLS:
        return {"planned": None, "support": True}
    plan = load_plan(out)
    qt = plan.get("question_type") if plan.get("question_type") in QUESTION_TYPES \
        else "exploratory"
    lk = _lookup(ask)
    items = scope_items(plan, load_amendments(out))
    marks: Optional[Dict[str, Any]] = None
    sc = plan.get("screen") or {}
    if call == "screen" and sc and \
            (sc.get("region") is None or lk["region"] == sc.get("region")) and \
            (not str(sc.get("year")).isdigit() or int(lk["year"]) == int(sc["year"])):
        screen_item = {"methods": ["screen"], "fields": [], "regions": [lk["region"]],
                       "periods": [str(lk["year"])], "steps": [],
                       "settings": named_settings(plan)}
        if _not_declared(screen_item, lk) is None:
            marks = {"planned": qt == "exploratory", "plan_items": ["screen"],
                     "label": EXPLORATORY_LABEL if qt == "exploratory" else SCREENING_LABEL}
    if marks is None and call != "screen":
        hits = [it["id"] for it in items
                if it["amendment"] is None and _not_declared(it, lk) is None]
        if hits:
            marks = {"planned": True, "plan_items": hits}
    if marks is None:
        fit = [it for it in items if it["amendment"] and _not_declared(it, lk) is None]
        pick = ([it for it in fit if it["amendment"] == amendment] or fit or [None])[0]
        if pick is None:
            near = "; ".join(f"{it['id']} declares no {_not_declared(it, lk)}"
                             for it in items[:6])
            raise PlanNotApproved(
                f"the reviewed plan declares no {_describe(lk)}, and no amendment does"
                f" ({near or 'the plan declares no analysis'}); declare it first in an "
                f"amendment (appended with a time stamp; its lookups never count as planned)"
                + _timing_note(lk))
        marks = {"planned": False, "plan_items": [], "amendment": pick["amendment"]}
        if pick.get("label"):
            marks["label"] = pick["label"]
    if qt == "exploratory" and call != "screen" and not (out / SCREEN_FILE).is_file():
        raise PlanNotApproved(
            f"an exploratory question runs the screen first (plan section 3.5): no other "
            f"lookup until the plan's screen (year {sc.get('year')}, region "
            f"{sc.get('region')!r}) has run and its scope, scoring rule and display rule are "
            f"recorded in {SCREEN_FILE}")
    closed = (out / PLANNED_PHASE_FILE).is_file()
    # THE SCREEN WAITS FOR THE CLOSE, NOT FOR EVERY DECLARED LOOKUP (the
    # user's decision on G part 2, 2026-11-08). Question 1 of G part 2 made 70
    # planned lookups and its screen was still refused, because 69 of the
    # field, region and period combinations its plan declared had not run. The
    # framework now closes the planned phase explicitly, each analysis marked
    # completed, failed or skipped with its reason, and the screen runs after.
    if qt == "hypothesis" and call == "screen" and not closed:
        raise PlanNotApproved(
            f"a hypothesis question runs the screen last (plan section 3.5): the "
            f"planned-analysis phase is not closed yet (there is no {PLANNED_PHASE_FILE}); "
            f"the framework closes it once every figure that does not run the screen has "
            f"run, then runs the screen's figure")
    if closed:
        marks = dict(marks, planned=False, after_close=True)
        if not (qt == "hypothesis" and call == "screen"):
            marks["label"] = EXPLORATORY_LABEL
        if (out / SCREEN_FILE).is_file():
            marks["after_screen"] = True
    return marks


# ─────────────────────────────────────────────────────────────────────
# THE LOOKUPS LOG
# ─────────────────────────────────────────────────────────────────────
class LookupLog:
    """The log a SuiteAccess writes to: one JSON line per call, with the
    time, the script and the gate's marks (planned or not, the plan's items,
    the amendment, a label). A contract read goes to its own file, never
    among the lookups. The plan's screen leaves its scope, scoring rule and
    display rule in screen_record.json, once."""

    def __init__(self, out_dir, script: Optional[str] = None,
                 amendment: Optional[str] = None):
        self.path = Path(out_dir) / LOOKUPS_FILE
        self.out_dir = Path(out_dir)
        self.script, self.amendment = script, amendment

    def _append(self, path: Path, line: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(line, default=str) + "\n")

    def _screen_csv(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """The screen's whole table as SCREEN_CSV, named by the lookup this
        line becomes (L<n>, as claim_trace counts the log's lines)."""
        _elm()
        import domain_methods as M                                  # noqa: N812
        held = self.path.read_text().splitlines() if self.path.is_file() else []
        name = SCREEN_CSV.format(n=1 + sum(1 for x in held if x.strip()))
        p = M.write_screen_csv({"rows": rows}, self.out_dir / name)
        return {"file": name, "rows": len(rows), "columns": list(M.SCREEN_COLUMNS),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

    def __call__(self, rec: Dict[str, Any]) -> None:
        line = {"at": _now(), "script": self.script, **rec}
        if rec.get("contract_read"):
            self._append(self.out_dir / CONTRACT_READS_FILE, line)
            return
        if "planned" not in line and "refused" not in line:
            line["planned"] = False             # a call no gate marked is never planned
        tab = line.pop("screen_table", None)
        if tab is not None and "sha256" in line:
            line["screen_csv"] = self._screen_csv(tab)
        sr = line.pop("screen_record", None)
        if sr is not None:
            p = self.out_dir / SCREEN_FILE
            if line.get("plan_items") == ["screen"] and "sha256" in line and not p.is_file():
                p.write_text(json.dumps(
                    {"at": line["at"], "script": self.script,
                     "question_type": load_plan(self.out_dir).get("question_type"),
                     "label": line.get("label"), "keys": line.get("keys"),
                     "sha256": line.get("sha256"), **sr}, indent=1, default=str))
                line["screen_record"] = SCREEN_FILE
            else:
                line["screen_record"] = sr
        self._append(self.path, line)


# ─────────────────────────────────────────────────────────────────────
# THE RUNNER'S SIDE
# ─────────────────────────────────────────────────────────────────────
class WithheldTableError(RuntimeError):
    """A script of a study with a suite read a whole outcome table."""


def withheld_words(name: str) -> str:
    call = TABLE_CALLS.get(name, TABLE_CALLS["reg"])
    return (f"{name} is not available in a study with a suite: values are reached only "
            f"through the scoped, logged suite calls. Use {call} for a region's values "
            f"(\"wateryear\" and \"day\" are steps too), suite.get_map(run, field, period) "
            f"for a map, or a method call such as suite.anomaly(field, period, region); "
            f"each must be declared by the reviewed plan or an amendment")


class WithheldTable:
    """What a script of a study with a suite finds under reg, reg_monthly and
    reg_seasonal (package F1, 2026-11-05). THE BYPASS IS CLOSED: before F1 the
    runner bound the whole tables once the plan had passed its review, so a
    script could read any value without a scoped, logged lookup. The payload
    now holds none of them; this stand-in holds no value, and any read of it
    (an attribute, an item, its length, iteration, truth, arithmetic) raises
    WithheldTableError naming the suite call that answers instead. Testing the
    name (`reg is not None`) reads nothing, so it does not fail."""

    __slots__ = ("_name",)

    def __init__(self, name: str):
        object.__setattr__(self, "_name", name)

    def _refuse(self, *a, **k):
        raise WithheldTableError(withheld_words(object.__getattribute__(self, "_name")))

    def __getattr__(self, attr):
        self._refuse()

    def __repr__(self):
        return f"<{object.__getattribute__(self, '_name')}: withheld, use the suite calls>"

    __str__ = __repr__
    __getitem__ = __setitem__ = __setattr__ = __delattr__ = __call__ = _refuse
    __iter__ = __len__ = __bool__ = __contains__ = _refuse
    __add__ = __sub__ = __mul__ = __truediv__ = __radd__ = __rsub__ = _refuse
    __rmul__ = __rtruediv__ = __lt__ = __le__ = __gt__ = __ge__ = __array__ = _refuse


def analysis_dir(ctx) -> Path:
    return Path(str(getattr(ctx, "run_dir", "") or ".")).resolve() / "04_analysis"


def access_spec(ctx) -> Optional[Dict[str, Any]]:
    """What a script's `suite` is built from, or None for a study that is not
    suite-bearing. Plain data, so it pickles into the script's payload."""
    dirs = suite_dirs(ctx)
    if not dirs:
        return None
    out = analysis_dir(ctx)
    plan = load_plan(out)
    ny = (plan.get("baseline") or {}).get("normal_years") or planner_normal_years(ctx)
    return {"suites": dirs, "analysis_dir": str(out), "normal_years": [int(y) for y in ny],
            "study_dir": str(Path(str(getattr(ctx, "run_dir", "") or ".")).resolve())}


def make_access(spec: Dict[str, Any], script: Optional[str] = None,
                amendment: Optional[str] = None):
    """The `suite` of one script: gated by check_lookup, logged by LookupLog,
    with the settings the plan and its amendments name."""
    _elm()
    import domain_access as DA                                      # noqa: N812
    out = Path(spec["analysis_dir"])
    ams = load_amendments(out)
    if amendment and not any(a.get("id") == amendment for a in ams):
        raise PlanNotApproved(f"no amendment {amendment!r} in {AMENDMENTS_FILE}")
    return DA.SuiteAccess(spec["suites"],
                          gate=lambda ask: check_lookup(out, ask, amendment=amendment),
                          log=LookupLog(out, script=script, amendment=amendment),
                          normal_years=spec.get("normal_years") or (),
                          named=named_settings(load_plan(out), amendments=ams),
                          study_dir=spec.get("study_dir"))


def draw_map(suite, x, out_path=None, ax=None, units: str = "", cmap: str = "viridis",
             vmin=None, vmax=None):
    """A per-cell map on the grid's own lat and lon, in the figure style of
    domain_report: a colour bar with its units, no title stating a result."""
    import matplotlib.pyplot as plt
    import numpy as np
    from agents.analysis.domain_report import _figure_style
    fs = _figure_style()
    fs.manuscript(fs.WIDTH["single"], base_pt=10)
    ll = suite.lat_lon()
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(fs.WIDTH["single"], fs.WIDTH["single"]))
    m = ax.pcolormesh(ll["lon"], ll["lat"], np.asarray(x, dtype=float), shading="nearest",
                      cmap=cmap, vmin=vmin, vmax=vmax)
    cb = plt.colorbar(m, ax=ax)
    cb.set_label(units)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    if own and out_path:
        ax.figure.savefig(out_path, dpi=150, bbox_inches="tight")
    return ax


def brief_parts(ctx, out_dir) -> Dict[str, str]:
    """Step 2b's reading for a suite-bearing study: the data catalog, the
    compact methods view and the plan with its review and amendments."""
    out = Path(out_dir)
    return {"catalog": catalog_text(ctx), "methods": compact_methods(),
            "plan": json.dumps(load_plan(out), indent=1, default=str),
            "review": json.dumps(load_review(out), indent=1, default=str),
            "amendments": json.dumps(load_amendments(out), indent=1, default=str)}
