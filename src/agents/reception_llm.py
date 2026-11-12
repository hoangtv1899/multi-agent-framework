#!/usr/bin/env python3
"""
Reception: a person's question in, a brief for the Planner out.

A language model with a small tool loop (ToolLoopAgent). It reads what the
model server says about itself, its prepared runs and the changes a what-if
run may make, puts one message to the person, and ends with a JSON brief, or
with a plain statement that the question cannot be answered here. What to
read, what to ask and what the answer means are the model's; this class runs
the loop, shows the prepared run's card, and reads the final JSON.

WHOLE-DOMAIN STUDIES ONLY (2026-11-11). Reception used to set up four kinds of
study. Its prompt was rewritten for the one this version runs, and the code
here that served the other three was deleted with it: the gather of a basin's
terrain, stations, rain and subsurface, the carry-forward of a prior run for a
coupling, and the dropping of a place from a sweep. A whole-domain study runs
on its control's own grid and fetches nothing, so nothing took their place.
A design brief of another kind is refused here, whatever the developer's
switch says: nothing in the prompt can produce one, and nothing here could
set it up.

workflow.py reaches this class directly.
"""
import json
import re
from typing import Any, Dict, List, Optional

from agents.prompts import load_prompt
from core.model_servers import model_tools
from agents.tool_loop import ToolLoopAgent

# WHAT THE LLM MAY CALL (2026-11-11). This version runs whole-domain studies
# only, and Reception's prompt was rewritten for them alone. A whole-domain
# study reads its control and its changes from the model server and fetches
# nothing, so Reception is given no data tool, and of the framework's own
# questions only the four its prompt uses. Before, it was also handed the
# sweep menu and two terrain tools, which the prompt no longer mentions: a
# tool with no instruction behind it is an invitation to call it, and a live
# whole-domain conversation did call the sweep menu, for nothing.
DEFAULT_ALLOWLIST: set = set()
RECEPTION_TOOLS = ("list_servers", "describe_server", "describe_domain_controls",
                   "describe_domain_changes")


def reception_tools(clients) -> Dict[str, Any]:
    """The framework's own questions Reception may ask: model_servers'
    tools, less those its prompt does not use."""
    every = model_tools(clients)
    return {name: every[name] for name in RECEPTION_TOOLS if name in every}


# What a design brief of another kind is told, when the developer's switch is
# set and core.supported_routes therefore says nothing.
NOT_FROM_A_CONVERSATION = (
    "Reception sets up whole-domain studies only. This request was read as {what}, "
    "which is no longer set up from a conversation, so nothing was designed, fetched "
    "or run")


# WHAT OF A CONTROL'S RECORD THE LLM IS NOT SHOWN (2026-11-11): its comparison
# with measurements and its results by year. Reception settles what is run; it
# does not answer the science. Handed those numbers, it quoted them to the
# person and then judged them (which of the run's numbers to trust more, how
# the run's own seasons compare), in four of eight live conversations, and a
# rule against it in the prompt did not stop it. A conclusion cannot be
# drawn from a number that was never read, so the numbers are left out of the
# model's copy and a note in their place says who does report them: the
# design page prints the
# comparison with measurements in the record's own words, before the person
# approves anything, and the numbers by year are results, which the analysis
# reports. The Planner's copy is trimmed in the same way
# (planner._rulebook_for_prompt).
SHOWN_ON_THE_DESIGN_PAGE = (
    "on record, and not shown to you: the design page prints it for the person, in "
    "the record's own words, before they approve anything. Say nothing about it")
REPORTED_BY_THE_ANALYSIS = (
    "on record, and not shown to you: the run's numbers by year are results, and "
    "the analysis reports results. Say nothing about them")
CONTROL_KEYS_NOT_FOR_THE_LLM = {"evaluation": SHOWN_ON_THE_DESIGN_PAGE}
FACTS_NOT_FOR_THE_LLM = {"per_year": REPORTED_BY_THE_ANALYSIS}


def controls_for_the_llm(answer: Any) -> Any:
    """describe_domain_controls' answer as the model reads it: every control
    without its comparison with measurements and its results by year, each
    replaced by a note saying who reports it, when the record holds it.
    A copy; the answer given is not changed, and anything that is not that
    answer's shape is returned as it is."""
    menu = answer.get("controls") if isinstance(answer, dict) else None
    if not (isinstance(menu, dict) and isinstance(menu.get("controls"), list)):
        return answer
    rows = []
    for c in menu["controls"]:
        if isinstance(c, dict):
            c = dict(c)
            for key, note in CONTROL_KEYS_NOT_FOR_THE_LLM.items():
                if c.get(key):
                    c[key] = note
            if isinstance(c.get("facts"), dict):
                c["facts"] = {k: (FACTS_NOT_FOR_THE_LLM[k]
                                  if k in FACTS_NOT_FOR_THE_LLM and v else v)
                              for k, v in c["facts"].items()}
        rows.append(c)
    return dict(answer, controls=dict(menu, controls=rows))


CARD_WIDTH = 98


def control_card(rows: List[Dict[str, Any]]) -> str:
    """The prepared controls as the person sees them before Reception's first
    question: what each is, in the server's own words, and where the rest of
    its record is shown.

    IDENTITY ONLY (the user's decision of 2026-11-11). The card used to print
    the control's whole comparison with measurements, its baseline years and
    what it does not contain: a long paragraph of numbers about other years,
    ahead of any answer to the question asked. That record is printed in full
    on the design page, which the person reads before approving anything
    (design_review), so nothing is withheld: it is shown where the decision
    is made. The card says so, and names only what the record holds.

    ONLY FOR A RUN NOT YET INTRODUCED (2026-11-11). The coordinator's start
    banner (core/scope_banner) says what each prepared run is before the
    first question, so a card under it said the same thing twice, a few lines
    apart. The card is for a run the banner did not name, and for Reception
    used without the coordinator (LLMReceptionAgent.introduced)."""
    import textwrap
    lines = ["", "  ── The prepared run ──"]
    later = []
    for r in rows:
        lines += textwrap.wrap(
            " ".join(f"{r.get('name')}: {r.get('description') or 'no description'}".split()),
            width=CARD_WIDTH, initial_indent="  ", subsequent_indent="    ")
        ev = r.get("evaluation") or {}
        if isinstance(ev, dict) and ev.get("summary") and \
                "how it compares with measurements" not in later:
            later.append("how it compares with measurements")
        if r.get("not_in_it") and "what it does not contain" not in later:
            later.append("what it does not contain")
    if later:
        said = " and ".join(later)
        said = said[0].upper() + said[1:]
        lines += textwrap.wrap(
            f"{said} {'are' if len(later) > 1 else 'is'} on the design page, before you "
            f"are asked to approve anything.",
            width=CARD_WIDTH, initial_indent="  ", subsequent_indent="    ")
    return "\n".join(lines)


class LLMReceptionAgent:
    """Tool-using reception: request -> framed brief (+ tool trace)."""

    def __init__(self,
                 model: str,
                 mcp_clients: Dict[str, Any],
                 allowlist: set = None,
                 # RAISED FROM 10 (2026-08-17). Choosing the model now costs a
                 # round to list the servers and one per server described, on
                 # top of resolving the basin and the period. Running out of
                 # rounds does not fail loudly — the loop asks for a final
                 # answer without tools, and the brief that comes back is
                 # whatever could be assembled without the fetch it was mid-way
                 # through. Headroom is cheaper than that.
                 max_rounds: int = 14,
                 verbose: bool = True,
                 interactive: bool = False):
        # NOTHING IS SUBSTITUTED INTO THE PROMPT ANY MORE (2026-08-18). Two
        # things used to be: the forcing window (until 2026-08-17, from
        # framework code that opened ELM's DATM directory) and the sweep menu
        # (until today, rendered from a call hardcoded to "elm"). Both were one
        # model's answer pasted into EVERY request before the request was read
        # — a PFLOTRAN study of 2024 was refused for a gap in ELM's forcing,
        # and a PFLOTRAN sweep would have been offered ELM's factors under a
        # heading saying they came from "the model server just now".
        #
        # Everything about a model now comes from that model's own server, read
        # by the LLM after it has named it: the forcing window and the initial
        # state under `constraints`, the sweep menu from describe_sweep. The
        # prompt says how to read them, once, for any model.
        self.system = load_prompt("reception_agentic")
        # ONE PROMPT (2026-11-11). It is written for whole-domain studies and
        # nothing else. Until then it was the prompt of four kinds of study
        # with a block appended that told the model to disregard three of
        # them. The rule itself is in code (core/supported_routes), in
        # process() below and twice more downstream; the prompt only keeps
        # Reception from offering what the code will refuse.
        self._clients = mcp_clients or {}
        # THE MODEL IS CHOSEN BY READING, NOT BY LOOKUP (2026-08-17). There
        # used to be a third substituted block here, a MENU of models that this
        # file built by asking every server and then summarising the answers
        # into a dozen lines each. It was replaced by two tools the model calls
        # itself, because a summary written before the request is read decides
        # what matters before knowing what the question is — and one of its
        # lines, "Do not choose it", was a verdict rather than evidence.
        #
        # The cost is rounds: two more before it can name a model, and it is
        # paid on every reception. That is the trade the user asked for
        # explicitly — a choice reasoned from what the servers actually say is
        # worth more here than a fast one.
        self.loop = ToolLoopAgent(
            model=model,
            mcp_clients=mcp_clients,
            allowlist=allowlist if allowlist is not None else DEFAULT_ALLOWLIST,
            max_rounds=max_rounds,
            verbose=verbose,
            interactive=interactive,
            local_tools=reception_tools(self._clients),
        )
        self._card_shown = False
        # the prepared runs the person has already been told about, by name:
        # the coordinator sets it from its start banner, and no card is shown
        # for a run named here (control_card)
        self.introduced: set = set()
        # THE CARD IS SHOWN WHEN THE CONTROLS ARE READ (2026-10-27), not before
        # Reception's first question: on a live try it asked nothing and went
        # straight to the design page, and the card never appeared. The card
        # is made from the whole record; the model reads its own copy
        # (controls_for_the_llm).
        dc = self.loop.local.get("describe_domain_controls")
        if dc:
            read = dc["fn"]

            def described(**kw):
                out = read(**kw)
                card = self._control_card([{"tool": "describe_domain_controls",
                                            "result": out}])
                if card:
                    print(card)
                return controls_for_the_llm(out)
            dc["fn"] = described

    @property
    def exposed_tools(self):
        return [t["function"]["name"] for t in self.loop.tools]

    def process(self, user_request: str, context: Dict[str, Any] = None,
                run_dir=None) -> Dict[str, Any]:
        """The conversation, then the package the coordinator saves.

        run_dir MAY BE A FUNCTION OF THE BRIEF (2026-08-18): `run_dir(brief)
        -> path`. The coordinator names a study's folder for the model that
        will run it, and the model is not known until the conversation has
        ended. A callable is resolved here, and only on the `design` route, so
        a question that ends any other way makes no folder. The coordinator
        still owns the folder: it is its function that makes it.

        Returns {route, brief, provenance, trace, rounds, raw, run_dir}, and
        four empty records (observations, grid, precipitation, subsurface)
        that the saved reception keeps the places of: a whole-domain study
        fetches none of them, and `provenance` says so. `route` carries the
        framework's dispatch (design, clarify, unsupported, analyze_existing,
        resume) so the brief stays science.
        """
        # a revision continues a conversation in which the card was shown
        self._card_shown = bool((context or {}).get("revision_of"))
        msg = f"User request: {user_request}\n\n"
        if context:
            msg += ("CONVERSATION CONTEXT (a prior experiment this session):\n"
                    + json.dumps(context, indent=2) + "\n\n")
        msg += ("Settle what is run with the person, as your instructions say, "
                "then emit ONLY your final JSON object.")
        out = self.loop.run(self.system, msg)
        brief = self._parse(out["content"])
        first = out["content"] or ""
        if brief.get("intent") == "parse_error" and out.get("messages"):
            out = self._ask_for_the_json(out, brief)
            brief = self._parse(out["content"])

        pkg = {"route": self._route(brief), "brief": brief,
               "observations": {}, "grid": {}, "precipitation": {},
               "subsurface": {}, "provenance": [],
               "trace": out["trace"], "rounds": out["rounds"],
               "raw": out["content"]}
        # A BRIEF ASKED FOR TWICE SAYS SO (2026-11-03): the second reply
        # replaces the first in `raw`, so without this a reader of
        # reception.json could not tell the brief came from a second request,
        # nor see what the unreadable first reply said.
        if out.get("json_asked_again"):
            pkg["json_asked_again"] = True
            pkg["raw_first"] = first[:400]
            pkg["provenance"] = [{
                "tool": None, "args": {}, "fetched_at": None, "ok": True,
                "error": None,
                "skipped": ("the final reply was not readable JSON and was "
                            "asked for once more"),
                "first_reply": first[:400]}]
        if pkg["route"]["action"] != "design":
            return pkg
        # A KIND OF STUDY THIS VERSION DOES NOT RUN STOPS HERE (2026-11-10,
        # core/supported_routes): before a study folder is made. A live
        # question about a watershed with no prepared control was refused in
        # words and then, on "proceed anyway", became a single-column design
        # with a prompt to run it. The reply typed in the dialog cannot change
        # this: the rule reads the brief, not the conversation. Since
        # 2026-11-11 the developer's switch does not open it either: the code
        # that set those studies up from a conversation is gone.
        from core import supported_routes as _routes
        kind = _routes.kind(brief)
        if kind != "whole_domain":
            why = _routes.problem(brief) or NOT_FROM_A_CONVERSATION.format(
                what=_routes.WORDS.get(kind, kind))
            pkg["route"] = {"action": "unsupported", "why": why,
                            "questions": [], "prior_run_dir": None,
                            "llm_intent": pkg["route"].get("llm_intent")}
            return pkg
        if callable(run_dir):
            run_dir = run_dir(brief)         # named for the model, see above
        pkg["run_dir"] = str(run_dir) if run_dir else None

        # A PLACE GIVEN AS A NAME (2026-10-28). A live whole-domain question
        # came back with "domain": "naches". The brief's domain is a record
        # everywhere downstream (the design page's Where line read a bare
        # name as "no basin"), so a name becomes {"name": ...} here, once,
        # and anything else that is not a record becomes none.
        if isinstance(brief.get("domain"), str):
            name = brief["domain"].strip()
            brief["domain"] = {"name": name} if name else {}
        elif brief.get("domain") is not None and not isinstance(brief["domain"], dict):
            brief["domain"] = {}

        # A WHOLE-DOMAIN EXPERIMENT FETCHES NOTHING. Its grid is its control's,
        # every cell of it, and its answer is read from the model's output
        # over the regions registered for its control
        # (describe_domain_controls); no observation enters it (2026-11-03).
        # Recorded, so that the empty records above read as a decision.
        ctl = ((brief.get("domain_experiment") or {}).get("control") or {})
        pkg["provenance"] = list(pkg["provenance"]) + [{
            "tool": None,
            "args": {"design_archetype": "whole_domain", "control": ctl.get("name")},
            "fetched_at": None,
            "ok": True,
            "error": None,
            "skipped": ("no DEM grid, rain, water table or station observations "
                        "were fetched: a whole-domain experiment runs on its "
                        "control's own grid and its answer is read from the "
                        "model's output over the control's registered regions. "
                        "This is a decision, not a failure."),
        }]
        return pkg

    def _control_card(self, trace) -> Optional[str]:
        """Shown once, when a prepared control has been described
        (2026-10-27): asked to introduce the control itself, Reception left
        out a different fact on each of four live tries, so the facts are
        shown by code, in the server's own words. Not shown for a control the
        person was already told about (`introduced`)."""
        if getattr(self, "_card_shown", False):
            return None
        rows: List[Dict[str, Any]] = []
        for step in trace:
            if step.get("tool") == "describe_domain_controls":
                res = step.get("result") or {}
                rows = list(((res.get("controls") or {}).get("controls")) or [])
        told = getattr(self, "introduced", None) or set()
        rows = [r for r in rows if isinstance(r, dict) and r.get("usable")
                and r.get("name") not in told]
        if not rows:
            return None
        self._card_shown = True
        return control_card(rows)

    # A FINAL REPLY THAT IS NOT THE JSON (2026-10-27). A live reception that
    # had introduced the control and been answered ended without the brief,
    # and the study stopped on "I need some clarification:" with nothing under
    # it. The model is asked once more, as the Analyzer is for a missing
    # verdict; a reply still unreadable becomes a sentence the user can act on.
    JSON_AGAIN = ("Your last reply could not be read as the final JSON object the OUTPUT "
                  "section asks for ({error}). Reply with that JSON object only, nothing "
                  "before or after it, and call no tool.")
    UNREADABLE = ("my last reply could not be read as a design, so nothing was set up; "
                  "please ask the question again")
    NOTHING_TO_ASK = ("I could not settle a design from that exchange, so nothing was set "
                      "up; please ask again, saying what you would like changed")

    def _ask_for_the_json(self, out: Dict[str, Any], brief: Dict[str, Any]
                          ) -> Dict[str, Any]:
        messages = list(out["messages"]) + [{"role": "user", "content": self.JSON_AGAIN
                                             .format(error=brief.get("error") or
                                                     "no JSON found")}]
        try:
            msg = self.loop._create(messages).choices[0].message
        except Exception:                                   # noqa: BLE001
            return out
        return dict(out, content=msg.content or "", json_asked_again=True,
                    messages=messages + [{"role": "assistant",
                                          "content": msg.content or ""}])

    @classmethod
    def _route(cls, brief: Dict[str, Any]) -> Dict[str, Any]:
        """Framework dispatch, kept OUT of the science brief.

        `intent` used to live inside the brief, where it was the only key the
        planner prompt never mentions — it is routing, not science. A
        clarification always carries a question: an unreadable reply, or one
        that asks nothing, says so instead of printing an empty list.
        """
        raw = brief.get("intent", "parse_error")
        # A QUESTION THAT CANNOT BE ANSWERED ENDS AS A STATEMENT (2026-11-10).
        # Reception could only stop through a clarification, so a refusal the
        # person had already accepted ("no pumping change exists; you asked
        # to stop") was printed under "I need some clarification:".
        if raw == "cannot_answer":
            why = " ".join(str(brief.get("reason") or "the question cannot be answered "
                               "in this version").split()).rstrip(". ")
            return {"action": "unsupported", "why": why, "questions": [],
                    "prior_run_dir": None, "llm_intent": raw}
        action = {"design": "design",
                  "analyze_existing": "analyze_existing",
                  "resume": "resume"}.get(raw, "clarify")
        questions = list(brief.get("questions") or [])
        if action == "clarify" and not questions:
            questions = [cls.UNREADABLE if raw == "parse_error" else cls.NOTHING_TO_ASK]
        return {"action": action,
                "questions": questions,
                "prior_run_dir": brief.get("run_dir"),
                "llm_intent": raw}

    @staticmethod
    def _parse(text: str) -> Dict[str, Any]:
        cleaned = re.sub(r"```json\s*", "", text)
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError as e:
                return {"intent": "parse_error", "error": str(e),
                        "raw": text[:400]}
        return {"intent": "parse_error", "error": "no JSON found",
                "raw": text[:400]}
