#!/usr/bin/env python3
"""
The planner
src/agents/planner.py

One model call turns Reception's brief into the study plan, strategy.json.
Nothing here corrects the plan. The model server checks a design where
compute starts being spent.

TWO PROMPTS (2026-11-13). A whole-domain brief gets prompts/planner_domain.txt,
a prompt of its own. Every other kind of study gets prompts/planner.txt, which
writes a sampling strategy for single columns and never coordinates. Both end
with prompts/house_style.txt, the rules for words a person reads.

A WHOLE-DOMAIN PLAN copies the control run, the new runs and the years that
Reception settled with the person. The Planner chooses where and in which
quantities the answer is read, judges whether the design can answer, gives
its reason in one plain sentence and names the limits of the answer for this
question. It is shown the model server's rulebook without measured numbers
(_rulebook_for_prompt).

THE SERVER'S OWN LIMITS ARE CARRIED BY CODE (carried_caveats). A live plan
copied a change's long limit and lost its "not", so the caveat said the
opposite. Code now writes those limits into the plan's caveats in the
server's words, and keeps what the Planner wrote itself apart.
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from agents.llm_agent import SimpleLLMClient
from agents.prompts import load_prompt

# The prompt each kind of brief gets (prompt_for). The calls below name them
# in full, as load_prompt("planner"), so that tools/make_architecture_doc.py,
# which reads only a literal name, lists both.
PROMPT = "planner"
DOMAIN_PROMPT = "planner_domain"
WHOLE_DOMAIN = "whole_domain"


def prompt_for(brief: Optional[Dict[str, Any]]) -> str:
    """The name of the prompt a brief is planned with. A whole-domain brief
    has a prompt of its own, and every other kind keeps planner.txt."""
    arche = str((brief or {}).get("design_archetype") or "").strip().lower()
    return DOMAIN_PROMPT if arche == WHOLE_DOMAIN else PROMPT


# ─────────────────────────────────────────────────────────────────────────────
# THE SERVER'S WORDS, CARRIED BY CODE (2026-11-13)
# ─────────────────────────────────────────────────────────────────────────────
# The prompt used to tell the Planner to copy each change's `not_meaning` into
# its caveats. In a live study it copied "a consistent change of climate" and
# lost the "not" in front of it, so the caveat said the opposite of the
# server. A copy that code makes cannot drop a word. So code writes these
# three kinds of limit, in the server's words, and the Planner is told not to.
NOT_MEANING = "The difference from the control run is not {}."
UNTESTED = ("No run with the change {} has been completed in this framework yet, "
            "so its result is untested.")
NOT_IN_IT = "The control run does not contain {}."
# `caveats_by` says the caveats were put together here (interface I4). A page
# then prints the server's limits from the server's records and, of the
# Planner's words, only `question_limits` and `planner_caveats`.
CAVEATS_BY = "code"


def _dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _clause(text: Any) -> str:
    """Server words set at the end of a sentence of ours. They are put on one
    line, and a closing period is dropped because the sentence adds its own."""
    return " ".join(str(text or "").split()).rstrip(" .")


def _same(text: Any) -> str:
    """The key under which two texts are one caveat. Spacing, case and a
    closing period do not make a second caveat."""
    return _clause(text).casefold()


class _Unfilled(dict):
    """A placeholder with no value stays as it is written, so a text is never
    lost for want of one parameter."""

    def __missing__(self, key):
        return "{" + key + "}"


def _filled(text: Any, change: Dict[str, Any], used: Dict[str, Any]) -> str:
    """A change's text with its {param} places filled from the run's own
    parameters, and from the list's defaults where the run gives none. The
    published list leaves a text such as "deposition for {year}" unfilled,
    since only a run names the year. A whole year reads as one, never 2000.0."""
    spec = _dict(change.get("params"))
    params = {k: p.get("default") for k, p in spec.items()
              if isinstance(p, dict) and p.get("default") is not None}
    params.update(_dict(used.get("params")))
    for k, v in list(params.items()):
        if _dict(spec.get(k)).get("kind") == "integer" and isinstance(v, float) \
                and v == int(v):
            params[k] = int(v)
    try:
        return str(text).format_map(_Unfilled(params))
    except (ValueError, IndexError, KeyError, AttributeError, TypeError):
        return str(text)


def _changes_used(design: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every change the design's runs make, in the order the runs list them."""
    out = []
    for run in _list(design.get("runs")):
        for c in _list(_dict(run).get("changes")):
            if isinstance(c, dict) and isinstance(c.get("id"), str) and c["id"].strip():
                out.append(c)
    return out


def _untested_words(rulebook: Dict[str, Any]) -> str:
    """The list's own words for a change no run has completed."""
    return str(_dict(rulebook.get("changes")).get("demonstrated_false") or "")


def _carried(rulebook: Any, design: Any) -> List[Tuple[str, str]]:
    """(the caveat, the server's text it was made from), each caveat once."""
    rb, de = _dict(rulebook), _dict(design)
    menu = {c["id"]: c for c in _list(_dict(rb.get("changes")).get("changes"))
            if isinstance(c, dict) and isinstance(c.get("id"), str)}
    used = _changes_used(de)
    out: List[Tuple[str, str]] = []

    def add(caveat: str, source: Any) -> None:
        if caveat not in [c for c, _ in out]:
            out.append((caveat, str(source)))

    # 1. what each change's difference is not. The same change with other
    # parameters may say another thing, and then both are said.
    for c in used:
        entry = _dict(menu.get(c["id"]))
        if str(entry.get("not_meaning") or "").strip():
            text = _filled(entry["not_meaning"], entry, c)
            if _clause(text):
                add(NOT_MEANING.format(_clause(text)), text)
    # 2. a change no run has completed, once for each change
    for cid in dict.fromkeys(c["id"] for c in used):
        if _dict(menu.get(cid)).get("demonstrated") is False:
            add(UNTESTED.format(cid), _untested_words(rb))
    # 3. what the control run does not contain
    name = _dict(de.get("control")).get("name")
    for row in _list(_dict(rb.get("controls")).get("controls")):
        if isinstance(row, dict) and name is not None and row.get("name") == name:
            for item in _list(row.get("not_in_it_words")):
                if _clause(item):
                    add(NOT_IN_IT.format(_clause(item)), item)
            break
    return out


def carried_caveats(rulebook: Optional[Dict[str, Any]],
                    design: Optional[Dict[str, Any]]) -> List[str]:
    """The model server's limits of a whole-domain design, in its own words.

    `rulebook` is the server's rulebook ({"controls", "changes"}) and `design`
    the plan's domain_experiment. In this order, each text once:

      1. for each change the runs use, in order of first use,
         "The difference from the control run is not <its not_meaning>."
      2. for each such change the list marks `demonstrated` False,
         "No run with the change <id> has been completed in this framework
         yet, so its result is untested."
      3. for each item of the control run's `not_in_it_words`,
         "The control run does not contain <the item>."

    Pure. A missing rulebook, an unknown change or a control run the
    rulebook does not hold gives fewer caveats, never an error."""
    return [c for c, _ in _carried(rulebook, design)]


def _texts(items: Any) -> List[str]:
    """The texts of a list the Planner wrote, as strings, empty ones left out.
    A single string is a list of one, and an item {"text": ...} is its text."""
    if isinstance(items, str):
        items = [items]
    out = []
    for x in _list(items):
        t = x.get("text") if isinstance(x, dict) else x
        if t is not None and str(t).strip():
            out.append(str(t).strip())
    return out


def _holds_design(rulebook: Any, design: Dict[str, Any]) -> bool:
    """Whether the rulebook holds the design's control run or a change its
    runs use, so that code has the server's words for it."""
    rb = _dict(rulebook)
    name = _dict(design.get("control")).get("name")
    if name is not None and any(isinstance(r, dict) and r.get("name") == name
                                for r in _list(_dict(rb.get("controls")).get("controls"))):
        return True
    menu = {c.get("id") for c in _list(_dict(rb.get("changes")).get("changes"))
            if isinstance(c, dict)}
    return any(c["id"] in menu for c in _changes_used(design))


def with_carried_caveats(design: Dict[str, Any],
                         rulebook: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """A copy of the plan's domain_experiment with its caveats put together by
    code (interface I4).

    `planner_caveats` keeps what the Planner wrote under `caveats`, less any
    text that is already a carried caveat, the server's text one was made
    from, or a question limit. `caveats` holds the carried caveats, then the
    question limits, then the Planner's own, each text once. `caveats_by`
    says that code put them together. Applied to its own result it gives the
    same result, so a design the Planner is shown again is not changed twice.

    NO RULEBOOK FOR THE DESIGN, NO `caveats_by` (2026-11-13). A missing
    rulebook, or one that holds neither the design's control run nor any
    change its runs use, gives code none of the server's words. Marked as
    put together by code, the design would then show no limit of the server
    at all, since a page prints those from the server's records. So it is
    returned as the Planner wrote it, and a page prints its caveats as it
    does for a record made before code carried them."""
    de = dict(design)
    if not _holds_design(rulebook, de):
        return dict(as_the_planner_wrote_it(de))
    pairs = _carried(rulebook, de)
    limits = _texts(de.get("question_limits"))
    known = {_same(t) for pair in pairs for t in pair if _same(t)}
    known |= {_same(q) for q in limits}
    mine: List[str] = []
    for c in _texts(de.get("caveats")):
        if _same(c) not in known and _same(c) not in {_same(m) for m in mine}:
            mine.append(c)
    caveats: List[str] = []
    for c in [c for c, _ in pairs] + limits + mine:
        if _same(c) not in {_same(x) for x in caveats}:
            caveats.append(c)
    de["planner_caveats"] = mine
    de["caveats"] = caveats
    de["caveats_by"] = CAVEATS_BY
    return de


def as_the_planner_wrote_it(design: Dict[str, Any]) -> Dict[str, Any]:
    """A plan's domain_experiment as the Planner wrote it, for the round after
    the model server refused it. Its `caveats` are its own again, so that it
    does not copy the carried ones back into its next design. A record made
    before code carried the caveats is returned as it is."""
    if not isinstance(design, dict) or design.get("caveats_by") != CAVEATS_BY:
        return design
    de = {k: v for k, v in design.items() if k not in ("planner_caveats", "caveats_by")}
    de["caveats"] = list(_texts(design.get("planner_caveats")))
    return de


def _summary_for_prompt(reception: Dict[str, Any]) -> Dict[str, Any]:
    """What the planner is shown: the brief, plus the observation SUMMARY.

    Never the raw series. The summary carries station ids and coordinates,
    which is all the planner needs to pin validation columns — the measurements
    themselves are the analyzer's business and would be ~170 kB here.

    NEVER THE SOIL OR THE RAIN EITHER. reception.soil holds a horizon profile
    for every candidate column location (~2.5 kB a point, 150 kB for a basin's
    grid) and reception.precipitation holds a daily series for every one of
    them (121 kB for a single year, ~2.4 MB for twenty). The planner
    stratifies on ELEVATION and reads neither, so every byte of them here
    would be paid for and unused. They are excluded twice on purpose: both are
    siblings of `brief`, so building this dict by naming keys already leaves
    them out, and the pops below stop them reaching the prompt if either is
    ever written into the brief by mistake. The second guard is the one that
    matters — the first is a property of today's shape, not a rule anyone
    stated.
    """
    brief = dict(reception.get("brief") or {})
    brief.pop("observations_summary", None)
    brief.pop("soil", None)
    brief.pop("precipitation", None)
    brief.pop("subsurface", None)
    return {"brief": brief,
            "observations_summary": (reception.get("brief") or {}).get(
                "observations_summary") or {}}


# OBSERVED NUMBERS STAY AWAY FROM THE PLANNER (2026-11-03). The rulebook the
# server returns carries how each control compares with observations (its
# evaluation, its observation list, the not-in-the-run items with their
# observed numbers, the observed burned area). The design page shows the
# scientist all of it, so strategy.json keeps the whole rulebook; but the
# planner's caveats go on to the Analyzer, which reads the model's output
# only, and a caveat copied from the text the planner was shown would carry
# an observed number there. So the planner's copy has none of it, and keeps
# `not_in_it_words`, the same items without their numbers.
#
# THE PAGE'S SUMMARY IS FOR THE PAGE (2026-11-13). The control's summary for
# the design page (`page_summary`) says how the control compares with
# measurements, with numbers, and it reached the planner whole since the day
# it was added. It is left out of the planner's copy as the evaluation is.
# What the planner does need of it, to judge which of the control's known
# errors matter for a question, it reads in `known_errors_words`: the same
# knowledge in words alone, which the server keeps free of numbers
# (domain_controls.summary).
_CONTROL_KEYS_NOT_FOR_PLANNER = ("evaluation", "observations", "not_in_it", "page_summary")

# NOR THE CONTROL'S OWN RESULTS (2026-11-12). The Planner's copy kept the
# control's numbers by year (plant growth, leaf area, rain, evaporation,
# runoff, snow, stored carbon, water closure) and the model's own burned area
# by year. The Planner chooses where and in which quantities an answer is
# read, judges whether the question can be answered and writes the caveats,
# and its goals and choices are handed to the step that plans the analysis:
# a design written by a reader of the results is not written before them.
# It needs none of them to design: the fields a control holds, its years,
# its settings, its fire and its regions all stay. A note in place of each
# says who reports it.
RESULTS_ARE_THE_ANALYSIS = (
    "on record, and not shown to you: these are the control's results, which the "
    "analysis reports. Design without them")
_FACTS_NOT_FOR_PLANNER = ("per_year", "burned_area_km2")


def _rulebook_for_prompt(rulebook: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The rulebook as the planner is shown it: every control without its
    comparison with observations and without its own results by year. A
    copy; the rulebook given is not changed."""
    if not rulebook:
        return rulebook
    out = dict(rulebook)
    ctl = rulebook.get("controls")
    if isinstance(ctl, dict) and isinstance(ctl.get("controls"), list):
        rows = []
        for c in ctl["controls"]:
            if not isinstance(c, dict):
                rows.append(c)
                continue
            c = {k: v for k, v in c.items() if k not in _CONTROL_KEYS_NOT_FOR_PLANNER}
            facts = c.get("facts")
            if isinstance(facts, dict):
                c["facts"] = {k: (RESULTS_ARE_THE_ANALYSIS
                                  if k in _FACTS_NOT_FOR_PLANNER and v else v)
                              for k, v in facts.items()}
            rows.append(c)
        out["controls"] = dict(ctl, controls=rows)
    return out


class Planner:
    """reception.json -> planner.json."""

    def __init__(self, model: str = "claude-opus-5-5-project",
                 pinning: Optional[Dict[str, Any]] = None,
                 constraints: Optional[Dict[str, Any]] = None):
        """`pinning` is the model server's own `pinning` block.

        WHICH STATIONS MAY BE PINNED IS THE MODEL'S ANSWER, and it used to be
        frozen into planner.txt as prose about 1-D columns and lateral
        transport. That was true of ELM and would have been false of the next
        backend, which is a wrong plan rather than a missing one — the study
        still runs, and spends columns on comparisons that cannot be made.

        Passed in rather than fetched here because the caller already holds the
        MCP clients and knows which backend it is about to drive; the planner
        does not, and giving it a client to guess with would put the same
        knowledge back one layer down.
        """
        self.llm = SimpleLLMClient(model=model)
        self.system = load_prompt("planner")
        # a whole-domain brief is planned with a prompt of its own (prompt_for)
        self.system_domain = load_prompt("planner_domain")
        self.pinning = pinning
        # THE OTHER HALF OF THE SAME QUESTION, set by the caller after
        # reception has named a model — `pinning` is what a column may be
        # COMPARED against, `constraints` is what a plan may CHOOSE. Both come
        # from the chosen model's own server; neither is stated in planner.txt
        # any more, because both were ELM's answers written as the framework's.
        self.constraints = constraints
        # A WHOLE-DOMAIN STUDY'S RULEBOOK, set by the caller for a whole_domain
        # brief: the model server's control runs (with facts measured from
        # their output) and its change list. Like the two above, the server's
        # answers, never stated in a prompt.
        self.domain_rulebook: Optional[Dict[str, Any]] = None

    # THE SERVER'S FINDINGS COME BACK HERE (2026-10-28). The model server
    # checks a design before the design review now, and a design it refuses
    # (a run named after an earlier study's case, a field with soil levels, a
    # region reaching outside the one a change applies in) is handed back with
    # its own words, once or twice, before anyone reads the page. Only what
    # each finding needs may change: the design the person agreed with
    # Reception stays.
    FINDINGS_NOTE = (
        "THE MODEL SERVER CHECKED THE DESIGN YOU EMITTED AND REFUSED IT. Its "
        "findings, in its own words, are below, after the design it refused. "
        "Emit the whole design again, changed only as each finding requires, "
        "and keep everything else as it was: the question, the region, the "
        "years, the runs' changes and the analysis the person agreed to. Never "
        "drop a run or a region to make a finding go away when a finding names "
        "the change that fixes it.")

    def messages(self, reception: Dict[str, Any],
                 findings: Optional[list] = None,
                 previous: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """(the system prompt, the user content) that plan() sends, built and
        not sent, so that what a brief is asked can be read without a call."""
        payload = _summary_for_prompt(reception)
        whole = prompt_for(payload.get("brief")) == DOMAIN_PROMPT
        if whole:
            # read from disk when this object was made without __init__
            system = getattr(self, "system_domain", None) or load_prompt("planner_domain")
        else:
            system = self.system
        # THE COLUMN BLOCKS ARE LEFT OUT OF A WHOLE-DOMAIN PROMPT (2026-11-13).
        # The design limits and the pinning rules describe single columns, the
        # forcing they read and the stations they may be pinned to. A
        # whole-domain study samples no column, and its own limits are in its
        # rulebook below, so the two blocks would only be read as limits of
        # the wrong study. The workflow still keeps `pinning` beside the plan.
        if self.constraints and not whole:
            system += ("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                       "━━━━━━\nDESIGN LIMITS — what this model's studies "
                       "cannot vary, from the\nserver that will run it. This "
                       "is the fence referred to above.\n━━━━━━━━━━━━━━━━━━━━"
                       "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                       + json.dumps(self.constraints, indent=1, default=str))
        if self.domain_rulebook:
            system += ("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                       "━━━━━━\nWHOLE-DOMAIN RULEBOOK: the control runs and the "
                       "change list,\nfrom the server that will run this "
                       "study\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                       "━━━━━━━━━━━\n"
                       + json.dumps(_rulebook_for_prompt(self.domain_rulebook),
                                    indent=1, default=str))
        if self.pinning and not whole:
            system += ("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                       "━━━━━━\nPINNING RULES — from the model server that "
                       "will run this study\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                       "━━━━━━━━━━━━━━━━━━━━━━━\n"
                       + json.dumps(self.pinning, indent=1, default=str))
        # THE WORDS A PERSON READS (2026-11-12). The Planner's reasons and
        # caveats are printed on the design page. The rules for such words are
        # given to it here, in full: it is a model call of its own and
        # inherits no style from anywhere.
        system += "\n\n" + load_prompt("house_style").rstrip()
        content = json.dumps(payload, indent=1, default=str)
        if findings:
            # the refused design without the blocks this code adds after the
            # planner (they come back from the server's own answers), and
            # with the caveats it wrote itself, not the ones code carried
            prior = {k: v for k, v in (previous or {}).items()
                     if k not in ("pinning", "domain_rulebook")}
            if isinstance(prior.get("domain_experiment"), dict):
                prior["domain_experiment"] = as_the_planner_wrote_it(
                    prior["domain_experiment"])
            content += ("\n\n" + self.FINDINGS_NOTE
                        + "\n\nTHE DESIGN IT REFUSED:\n"
                        + json.dumps(prior, indent=1, default=str)
                        + "\n\nTHE SERVER'S FINDINGS:\n"
                        + json.dumps(list(findings), indent=1, default=str))
        return system, content + "\n\nEmit ONLY the JSON."

    def plan(self, reception: Dict[str, Any],
             findings: Optional[list] = None,
             previous: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        system, content = self.messages(reception, findings=findings, previous=previous)
        raw = self.llm.ask([{"role": "user", "content": content}], system_message=system)
        plan = self._parse(raw)
        # A WHOLE-DOMAIN PLAN'S CAVEATS ARE PUT TOGETHER BY CODE (2026-11-13):
        # the server's own limits in its words, then the Planner's limits for
        # the question, then any caveat of its own (with_carried_caveats).
        if isinstance(plan.get("domain_experiment"), dict):
            plan["domain_experiment"] = with_carried_caveats(
                plan["domain_experiment"], getattr(self, "domain_rulebook", None))
        return plan

    @staticmethod
    def _parse(text: str) -> Dict[str, Any]:
        cleaned = re.sub(r"```json\s*", "", text or "")
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()
        a, b = cleaned.find("{"), cleaned.rfind("}")
        if a == -1 or b <= a:
            return {}
        try:
            return json.loads(cleaned[a:b + 1])
        except json.JSONDecodeError:
            return {}
