#!/usr/bin/env python3
"""
Reception dialogue tests — a scripted human, so nobody has to sit there
tools/dialogue_tests.py

    python3 tools/dialogue_tests.py --list
    python3 tools/dialogue_tests.py --case conceptual_vague
    python3 tools/dialogue_tests.py --all                 # costs real money

HOW THE DIALOGUE IS HELD WITHOUT A HUMAN. Reception asks through the `ask_user`
tool, which lands in tool_loop._human_answer(question, tty, interactive) and
reads a line from a terminal. That function RECEIVES THE QUESTION, so a test can
substitute it and answer from a persona. Nothing in the pipeline changes —
production still reads a terminal, and no flag exists that a real run could
trip over.

Two kinds of stand-in, and the difference matters:

    scripted   rules matched against the question text. Deterministic, free,
               and repeatable — so it can assert. It answers only what it was
               told to answer and says "you decide" to everything else, which
               is itself a test: reception must cope with a user who will not
               settle something.

    played     a second model told who it is. Realistic, costs a call per
               question, and answers differently every time — so it is for
               READING, not for asserting. Use it when judging whether the
               conversation feels right; use `scripted` when checking it still
               works.

WHAT THESE TESTS CANNOT DO. They judge the conversation and the design that
falls out of it. They do not run ELM. A design that passes here can still be
refused by the server, which is why every case ends by handing the settled
design to check_conceptual_design — the one assertion that is free and catches
the most.
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from core.mcp_manager import MCPManager                        # noqa: E402
from agents import tool_loop                                   # noqa: E402
from agents.reception_llm import LLMReceptionAgent             # noqa: E402

MODEL = "claude-opus-5-project"


# ─────────────────────────────────────────────────────────────────────
# THE STAND-IN HUMAN
# ─────────────────────────────────────────────────────────────────────
class Scripted:
    """Answers by matching the question. Records everything it was asked.

    `rules` is ordered: the first pattern that matches wins, so put the
    specific ones first. Anything unmatched gets `fallback` — which defaults to
    a REFUSAL TO DECIDE rather than a plausible answer, because a stand-in that
    invents agreement would hide the case where reception asked something the
    user cannot answer.
    """

    def __init__(self, rules: List[tuple], fallback: str =
                 "I don't know — you decide, but tell me what you chose."):
        self.rules = [(re.compile(p, re.I), a) for p, a in rules]
        self.fallback = fallback
        self.asked: List[str] = []
        self.answered: List[str] = []

    def __call__(self, question: str, tty=None, interactive=True) -> Dict[str, Any]:
        self.asked.append(question)
        for pat, ans in self.rules:
            if pat.search(question):
                self.answered.append(ans)
                print(f"\n  ❓ {question}\n  🤖 {ans}")
                return {"answer": ans}
        self.answered.append(self.fallback)
        print(f"\n  ❓ {question}\n  🤖 [fallback] {self.fallback}")
        return {"answer": self.fallback}


class Played:
    """A second model told who it is. For reading, not for asserting.

    IT SEES ONLY THE QUESTION, never reception's prompt or the sweep menu.
    That is the whole reason it is worth running: a stand-in that had read the
    menu would answer in the menu's own vocabulary and could never discover
    that a question is unanswerable to someone who has not.

    IT ALSO REMEMBERS, since 2026-08-15. Every turn used to be an independent
    call carrying nothing but the current question, so the persona could
    cheerfully contradict what it had said two questions earlier — agreeing to
    seven clay levels and then asking for three. A real user is not a fresh
    mind each time they are asked something, and a reception failure and a
    persona amnesia are indistinguishable in the transcript afterwards.
    """

    def __init__(self, persona: str, model: str = MODEL):
        from agents.llm_agent import SimpleLLMClient
        self.llm = SimpleLLMClient(model=model)
        self.persona = persona
        self.asked: List[str] = []
        self.answered: List[str] = []
        # The running conversation, in the persona's own frame: what it was
        # asked is `user`, what it replied is `assistant`.
        self.history: List[Dict[str, str]] = []

    def __call__(self, question: str, tty=None, interactive=True) -> Dict[str, Any]:
        self.asked.append(question)
        sys_p = (f"You are a scientist using a simulation framework. {self.persona}\n"
                 f"Answer the assistant's question in ONE or TWO short "
                 f"sentences, as a person would. Do not explain yourself and "
                 f"do not be more decisive than your character would be.\n"
                 f"You have already answered the earlier questions in this "
                 f"conversation. BE CONSISTENT WITH THEM: do not revise a "
                 f"number you already gave unless you are explicitly told why "
                 f"it will not work.")
        msgs = self.history + [{"role": "user", "content": question}]
        try:
            out = self.llm.ask(msgs, system_message=sys_p)
            ans = (out or "").strip() or "you decide"
        except Exception as e:                                  # noqa: BLE001
            ans = f"(persona failed: {type(e).__name__}) you decide"
        self.answered.append(ans)
        self.history = msgs + [{"role": "assistant", "content": ans}]
        print(f"\n  ❓ {question}\n  🎭 {ans}")
        return {"answer": ans}


# ─────────────────────────────────────────────────────────────────────
# ASSERTIONS — each returns (ok, message)
# ─────────────────────────────────────────────────────────────────────
def archetype_is(want: str):
    def _f(pkg, brief, sweep, verdict, human):
        got = brief.get("design_archetype")
        return got == want, f"archetype {got!r} (want {want!r})"
    return _f


def has_sweep(want: bool = True):
    def _f(pkg, brief, sweep, verdict, human):
        got = bool(sweep)
        return got == want, f"sweep block {'present' if got else 'absent'}"
    return _f


def asked_at_least(n: int):
    def _f(pkg, brief, sweep, verdict, human):
        return len(human.asked) >= n, f"asked {len(human.asked)} question(s)"
    return _f


def asked_about(pattern: str, label: str = None):
    rx = re.compile(pattern, re.I)
    def _f(pkg, brief, sweep, verdict, human):
        hit = any(rx.search(q) for q in human.asked)
        return hit, f"{'asked' if hit else 'never asked'} about {label or pattern}"
    return _f


def factors_within_menu():
    """No factor named is off the server's menu. The core invention guard.

    VACUOUSLY TRUE WITH NO FACTORS, deliberately. The first version failed a
    design that named none — but refusing to design anything is the CORRECT
    answer to "warm the climate by 2 degrees", so that check was failing the
    behaviour it exists to encourage. Presence is a separate question; use
    has_factor() when a case actually requires one.
    """
    def _f(pkg, brief, sweep, verdict, human):
        named = {f.get("name") for f in (sweep.get("factors") or [])}
        if not _MENU_NAMES:
            return False, "could not read the server's menu — cannot judge"
        bad = named - set(_MENU_NAMES)
        if not named:
            return True, "no factors named (nothing invented)"
        return not bad, (f"factors {sorted(named)}"
                         + (f" — INVENTED {sorted(bad)}" if bad else " all on the menu"))
    return _f


def has_factor(name: str):
    """A named factor is present. Separate from whether it was invented."""
    def _f(pkg, brief, sweep, verdict, human):
        named = [f.get("name") for f in (sweep.get("factors") or [])]
        return name in named, f"factors {named}"
    return _f


def no_buildable_design():
    """Nothing runnable escaped. Either it was refused in conversation, or
    the server refused it — both are correct, and the first version of this
    check counted a conversational refusal as a failure because the server
    had never been asked.
    """
    def _f(pkg, brief, sweep, verdict, human):
        if not (sweep.get("factors") or []):
            return True, "reception produced no design — refused in conversation"
        if verdict is None:
            return False, "a design exists but the server was never asked"
        if verdict.get("buildable"):
            return False, ("a BUILDABLE design escaped: "
                           f"{[(f.get('name'), f.get('levels')) for f in sweep['factors']]}")
        why = "; ".join(w.get("why", "")[:70] for w in (verdict.get("wont_build") or []))
        return True, f"server refused it — {why}"
    return _f


def fit_and_availability_separate():
    """The two must both be present and must not be the same sentence."""
    def _f(pkg, brief, sweep, verdict, human):
        # Top-level since 2026-08-16 — every archetype names its model, not
        # only a sweep. `sweep` is still passed in and is still the right
        # place for the factors; the model never belonged inside it.
        why = (brief.get("model_rationale") or "").strip()
        avail = (brief.get("model_availability") or "").strip()
        if not why or not avail:
            return False, "one of rationale/availability is empty"
        return why != avail, ("both present and distinct" if why != avail
                              else "rationale and availability are the SAME text")
    return _f


def buildable(want: bool = True):
    def _f(pkg, brief, sweep, verdict, human):
        if verdict is None:
            return False, "the server was not asked"
        got = bool(verdict.get("buildable"))
        why = "; ".join(w.get("why", "")[:60]
                        for w in (verdict.get("wont_build") or []))
        return got == want, f"buildable={got} {why}"
    return _f


def server_flags(pattern: str):
    rx = re.compile(pattern, re.I)
    def _f(pkg, brief, sweep, verdict, human):
        pool = (verdict or {}).get("unusual", []) + (verdict or {}).get("wont_build", [])
        hit = any(rx.search(str(x.get("why", ""))) for x in pool)
        return hit, f"{'flagged' if hit else 'did not flag'} {pattern}"
    return _f


def mentions(pattern: str, label: str = None):
    """Somewhere in what reception said — an answer, a note, or a question."""
    rx = re.compile(pattern, re.I)
    def _f(pkg, brief, sweep, verdict, human):
        blob = json.dumps(brief, default=str) + " ".join(human.asked)
        hit = bool(rx.search(blob))
        return hit, f"{'mentioned' if hit else 'never mentioned'} {label or pattern}"
    return _f


def weather_is_written(want: bool = True):
    """The design writes the weather rather than borrowing a real cell's.

    Either shape counts: `held_fixed.weather` gives every column the same
    written weather, a `prescribed_weather` factor varies it between them.
    Both mean no real cell's climate bounds the result, which is the thing
    being asserted.
    """
    def _f(pkg, brief, sweep, verdict, human):
        held = (sweep.get("held_fixed") or {}).get("weather")
        as_factor = any(f.get("name") == "prescribed_weather"
                        for f in (sweep.get("factors") or []))
        got = bool(held) or as_factor
        how = ("held_fixed.weather" if held else
               "prescribed_weather factor" if as_factor else "borrowed")
        return got == want, f"weather is {how}"
    return _f


def offered_written_weather():
    """Reception RAISED writing the weather, whether or not it was taken up.

    Separate from weather_is_written on purpose: a user may hear the option and
    decline it, and that is a good outcome. What must not happen is the user
    choosing between borrowed climates without being told the third option
    exists — which is what the capability was invisible for before.
    """
    rx = re.compile(r"prescrib|written weather|write the weather|"
                    r"uniform|idealis|idealiz|synthetic weather|"
                    r"held_fixed\.weather", re.I)
    def _f(pkg, brief, sweep, verdict, human):
        blob = json.dumps(brief, default=str) + " ".join(human.asked)
        hit = bool(rx.search(blob))
        return hit, ("offered writing the weather" if hit
                     else "NEVER offered it — the user chose among borrowed "
                          "climates without knowing the option existed")
    return _f


def quoted_column_count():
    def _f(pkg, brief, sweep, verdict, human):
        if sweep.get("n_columns"):
            return True, f"n_columns={sweep['n_columns']}"
        asked = " ".join(human.asked)
        return bool(re.search(r"\b\d+\s+column", asked, re.I)), "no column count anywhere"
    return _f


# ─────────────────────────────────────────────────────────────────────
# THE CASES
# ─────────────────────────────────────────────────────────────────────
SITE_RULES = [(r"period|year", "1995"),
              (r"watershed|basin|which.*huc", "the Naches sub-watershed")]

CASES: Dict[str, Dict[str, Any]] = {

    # ── does it classify, and does it converse ───────────────────────
    "conceptual_vague": {
        "request": "I want to understand how soil affects runoff.",
        "persona": "You are curious but have no specific design in mind. "
                   "You accept sensible suggestions.",
        "rules": [(r"vary|factor|texture|sweep", "soil texture sounds right"),
                  (r"where|weather|location|site|climate|cell",
                   "somewhere wet in the Cascades — you pick the cell"),
                  (r"year|period", "1995 is fine"),
                  (r"how many|levels|column", "whatever you suggest")],
        "checks": [archetype_is("conceptual"), has_sweep(True),
                   asked_at_least(2), factors_within_menu(),
                   has_factor("soil_texture"),
                   asked_about(r"where|weather|location|climate|cell", "the weather cell"),
                   quoted_column_count(), buildable(True)],
    },

    "conceptual_specific": {
        "request": ("Sweep soil clay content from 5 to 55 percent at "
                    "47.11 N, -121.39 W for 1995, everything else held fixed."),
        "persona": "You know exactly what you want and answer tersely.",
        "rules": [(r".*", "yes, that's right — go ahead")],
        "checks": [archetype_is("conceptual"), has_sweep(True),
                   factors_within_menu(), buildable(True),
                   quoted_column_count()],
    },

    # ── the model choice ─────────────────────────────────────────────
    "model_choice": {
        "request": "How does soil texture control infiltration versus runoff?",
        "persona": "You have not decided which model to use and want advice.",
        "rules": [(r"model|elm|pflotran", "which would you recommend, and why?"),
                  (r"where|weather|location|climate|cell", "use a wet Cascades cell"),
                  (r"year|period", "1995")],
        "checks": [archetype_is("conceptual"), has_sweep(True),
                   fit_and_availability_separate(),
                   mentions(r"pflotran", "the other backend")],
    },

    # ── the objection you raised ─────────────────────────────────────
    # REWRITTEN 2026-08-15, because the right answer changed. It used to be
    # enough for reception to EXPLAIN that the coordinates are whose weather is
    # borrowed — that was the best available answer when the weather could only
    # come from a real cell. Now it can be written, so explaining is no longer
    # the best answer: the user's objection is correct and there is a setting
    # that acts on it. A run that only explains is now a run that withheld the
    # fix.
    "no_location_objection": {
        "request": "How does soil texture split rain between runoff and drainage?",
        "persona": "You believe a conceptual study should have no location at "
                   "all and push back when asked for one.",
        "rules": [(r"where|weather|location|site|climate|cell",
                   "there IS no location — this is conceptual, not a place"),
                  (r"year|period", "1995"),
                  (r"vary|texture|levels", "clay, your suggested levels")],
        "checks": [archetype_is("conceptual"),
                   mentions(r"weather|forcing|NLDAS|grid cell", "why a cell is still needed"),
                   offered_written_weather(),
                   has_sweep(True), buildable(True)],
    },

    # ── written weather, the three ways it comes up ──────────────────
    "uniform_precip_pushback": {
        "request": ("Run a soil texture sweep with constant rainfall all year, "
                    "so every column gets exactly the same water."),
        "persona": "You asked for constant rain and want to know if that is a "
                   "problem before you commit to it.",
        "rules": [(r"intensity|storm|constant|uniform|steady|instead|scale",
                   "if intensity matters for this question, tell me plainly, "
                   "then do it my way anyway"),
                  (r"how much|rate|mm|precip", "about 3 mm per day"),
                  (r"year|period", "1995"),
                  (r"where|location|cell|coordinates", "you pick — I don't care where"),
                  (r"vary|texture|levels|clay", "your suggested clay levels")],
        # THE POINT OF THIS CASE. Constant rain removes rainfall intensity,
        # which is the mechanism a partitioning question is ABOUT. Reception
        # must say so — and must then build what the user chose. Warning and
        # obeying are both required; either alone is the wrong behaviour.
        "checks": [archetype_is("conceptual"),
                   mentions(r"intensity|storm|episod", "that intensity is removed"),
                   weather_is_written(True),
                   has_factor("soil_texture"),
                   buildable(True)],
    },

    "weather_as_the_factor": {
        "request": ("Keep the soil the same and compare a dry year, a normal "
                    "year and a wet year — but I want the same storms in each, "
                    "just more or less rain."),
        "persona": "You want the storm pattern kept and only the amount changed.",
        "rules": [(r"how much|scale|factor|percent|multipl|dry|wet",
                   "half, normal, and double"),
                  (r"year|period", "1995"),
                  (r"where|location|cell|coordinates", "the Naches cell is fine"),
                  (r"soil|texture|clay", "keep the soil fixed")],
        # `scale` keeps the storm structure and changes the amount — which is
        # exactly what was asked for, and is NOT `uniform`. If reception
        # reaches for uniform here it has misread "the same storms".
        "checks": [archetype_is("conceptual"),
                   has_factor("prescribed_weather"),
                   factors_within_menu(),
                   buildable(True), quoted_column_count()],
    },

    # ── refusals, flags and misunderstandings ────────────────────────
    "impossible_soil": {
        "request": "Vary clay from 5 to 200 percent at 47.11 N, -121.39 W in 1995.",
        "persona": "You insist on the range you asked for.",
        "rules": [(r"clay|percent|range|200", "yes, 200 percent, as I said"),
                  (r".*", "as I said")],
        # Either reception refuses it, or the server does. Both are correct;
        # what must NOT happen is a buildable design containing clay 200.
        "checks": [no_buildable_design()],
    },

    # RENAMED FROM off_menu_warming, 2026-08-15, and the assertion is now the
    # opposite one. This was the refusal test: +2 C was declared but not
    # runnable, so the right behaviour was to say so and design without it. It
    # is on the menu now — `offset` on TBOT — so the same request that had to be
    # turned down has to be built. A test left asserting the refusal would have
    # gone on passing while reporting a capability the framework has.
    "warming_experiment": {
        "request": "Run a soil texture sweep but also warm the climate by 2 degrees.",
        "persona": "You want the warming and will ask why not if refused.",
        "rules": [(r"warm|temperature|climate|degree|offset", "yes, plus 2 degrees"),
                  (r"where|weather|location|cell", "a wet Cascades cell"),
                  (r"year|period", "1995"),
                  (r"vary|texture|levels|clay", "your suggested clay levels"),
                  (r".*", "go ahead")],
        # Two factors is 7 x 2 = 14 columns, so the count matters here.
        "checks": [archetype_is("conceptual"),
                   factors_within_menu(),
                   weather_is_written(True),
                   has_factor("soil_texture"),
                   buildable(True), quoted_column_count()],
    },

    "soil_depth_misunderstanding": {
        "request": "Compare a 2 m soil sitting on bedrock against a 5 m soil.",
        "persona": "You picture soil over bedrock and expect the column to end "
                   "where the soil ends.",
        "rules": [(r"depth|bedrock|grid|column|layer", "yes, 2 m then rock"),
                  (r"where|weather|location|cell", "a wet Cascades cell"),
                  (r"year|period", "1995"),
                  (r".*", "explain that to me")],
        "checks": [mentions(r"42|fixed|15 layer|substrate|not configurable",
                            "the fixed soil grid")],
    },

    "flat_sweep": {
        # PHRASING MATTERS, and the first version of this case got it wrong.
        # "Compare runoff at 47.11 N against 47.12 N" reads as a SITE study —
        # two real coordinates, a real place — and reception classified it that
        # way, correctly. The case then tested nothing, because a site study
        # never reaches the sweep checks. Said as a controlled experiment, the
        # same two points become a forcing_site sweep and the collision is on
        # the path it was written for.
        "request": ("Run a controlled conceptual experiment: the same soil "
                    "under two different climates, using grid cells at "
                    "47.11 N -121.380 W and 47.12 N -121.385 W, in 1995. "
                    "No real site, no observations."),
        "persona": "You chose those two points deliberately and will insist.",
        "rules": [(r"cell|point|coordinate|climate|apart|close|same",
                   "yes, exactly those two points — keep them"),
                  (r"year|period", "1995"),
                  (r".*", "yes, go ahead with those two")],
        # Either reception notices, or the server does. What must not happen is
        # a buildable two-level sweep whose levels share one NLDAS cell.
        "checks": [no_buildable_design()],
    },

    # ── the regressions that matter most ─────────────────────────────
    "site_regression": {
        "request": ("Simulate the Naches sub-watershed for 1995 and validate "
                    "against the stream gauges."),
        "persona": "A normal site study; answer plainly.",
        "rules": SITE_RULES,
        "checks": [archetype_is("site"), has_sweep(False)],
    },

    "site_vague_regression": {
        "request": "Study snow and runoff in a Cascades watershed.",
        "persona": "You mean a real place and will name one when asked.",
        "rules": SITE_RULES + [(r".*", "the Naches, 1995")],
        "checks": [archetype_is("site"), has_sweep(False)],
    },
}

_MENU_NAMES: List[str] = []


# ─────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────
def save_transcript(out_dir, name: str, spec: Dict[str, Any], pkg: Dict[str, Any],
                    human, verdict, results) -> str:
    """The whole exchange, verbatim, as a readable file.

    WRITTEN BECAUSE THE FIRST RUN WAS FILTERED THROUGH grep AND LOST. A dialogue
    test whose output is a PASS line tells you it worked and nothing about
    whether the conversation was any good — which is the only thing a human can
    judge and the only reason to run it against a real model at all.
    """
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    brief = pkg.get("brief") or {}
    sweep = brief.get("sweep") or {}
    L = [f"# {name}", "",
         "## the user's request", "", f"> {spec['request']}", "",
         "## the exchange", ""]
    for q, ans in zip(human.asked, human.answered):
        L += ["**reception asked:**", "", q, "",
              f"**the user answered:** {ans}", "", "---", ""]
    if not human.asked:
        L += ["*(reception asked nothing)*", ""]
    L += ["## what reception produced", "",
          f"- route: `{(pkg.get('route') or {}).get('action')}`",
          f"- archetype: `{brief.get('design_archetype')}`",
          f"- rounds: {pkg.get('rounds')}", ""]
    if sweep:
        L += ["```json", json.dumps(sweep, indent=2, default=str), "```", ""]
    else:
        L += ["*(no `sweep` block)*", ""]
    L += ["## what the elm server said about it", ""]
    L += (["```json", json.dumps(verdict, indent=2, default=str), "```"]
          if verdict else ["*(not asked — reception produced no design)*"])
    L += ["", "## checks", ""]
    L += [f"- {'PASS' if ok else 'FAIL'} — {msg}" for ok, msg in results]
    p = d / f"{name}.md"
    p.write_text("\n".join(L))
    return str(p)


def run_case(name: str, spec: Dict[str, Any], clients: Dict[str, Any],
             played: bool = False, save: Optional[str] = None) -> bool:
    human = (Played(spec["persona"]) if played
             else Scripted(spec.get("rules") or []))

    # THE SUBSTITUTION. tool_loop calls this by module attribute, so patching
    # it here reaches the live loop without touching a line of production code.
    original = tool_loop._human_answer
    tool_loop._human_answer = human
    try:
        agent = LLMReceptionAgent(model=MODEL, mcp_clients=clients,
                                  interactive=True)
        print("=" * 72)
        print(f"CASE {name}")
        print(f"  request: {spec['request']}")
        print("=" * 72)
        pkg = agent.process(spec["request"])
    finally:
        tool_loop._human_answer = original

    brief = pkg.get("brief") or {}
    sweep = brief.get("sweep") or {}

    verdict = None
    if clients.get("elm") and sweep.get("factors"):
        design = {"factors": [{"name": f.get("name"), "levels": f.get("levels")}
                              for f in sweep["factors"]],
                  "held_fixed": sweep.get("held_fixed") or {}}
        try:
            verdict = clients["elm"].call_tool_json(
                "check_conceptual_design", {"design": design})
        except Exception as e:                                  # noqa: BLE001
            verdict = {"buildable": False,
                       "wont_build": [{"why": f"check failed: {e}"}]}

    print(f"\n  archetype : {brief.get('design_archetype')}")
    if sweep:
        print(f"  factors   : {[(f.get('name'), f.get('levels')) for f in (sweep.get('factors') or [])]}")
        print(f"  held fixed: {sweep.get('held_fixed')}")
        print(f"  n_columns : {sweep.get('n_columns')}")

    print("\n  checks")
    ok_all = True
    results = []
    for chk in spec["checks"]:
        try:
            ok, msg = chk(pkg, brief, sweep, verdict, human)
        except Exception as e:                                  # noqa: BLE001
            ok, msg = False, f"check raised {type(e).__name__}: {e}"
        results.append((ok, msg))
        ok_all &= ok
        print(f"    {'PASS' if ok else 'FAIL'}  {msg}")
    if save:
        print(f"    → {save_transcript(save, name, spec, pkg, human, verdict, results)}")
    print(f"\n  {name}: {'PASS' if ok_all else 'FAIL'}\n")
    return ok_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append", help="run one (repeatable)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--played", action="store_true",
                    help="let a model play the user — realistic, "
                         "non-deterministic, for reading rather than asserting")
    ap.add_argument("--mcp-config", default="mcp_config.json")
    ap.add_argument("--save", metavar="DIR",
                    help="write each case's full exchange to DIR/<case>.md")
    a = ap.parse_args()

    if a.list or not (a.case or a.all):
        print("cases (each is one reception run — real model calls):\n")
        for k, v in CASES.items():
            print(f"  {k:30} {v['request'][:60]}")
        print(f"\n  --case <name>   one case")
        print(f"  --all           all {len(CASES)} — costs real money")
        print(f"  --played        a model plays the user instead of rules")
        return 0

    clients = {}
    try:
        clients = MCPManager(a.mcp_config).get_all_clients() or {}
    except Exception as e:                                      # noqa: BLE001
        print(f"⚠️  no MCP clients ({type(e).__name__}: {e})")

    global _MENU_NAMES
    if clients.get("elm"):
        try:
            d = clients["elm"].call_tool_json(
                "describe_conceptual_factors", {}) or {}
            _MENU_NAMES = [f["name"] for f in (d.get("factors") or [])]
            print(f"menu from the elm server: {_MENU_NAMES}\n")
        except Exception as e:                                  # noqa: BLE001
            print(f"⚠️  could not read the factor menu ({e})\n")

    names = list(CASES) if a.all else [c for c in (a.case or []) if c in CASES]
    unknown = [c for c in (a.case or []) if c not in CASES]
    for u in unknown:
        print(f"⚠️  no such case: {u}")

    results = {n: run_case(n, CASES[n], clients, played=a.played, save=a.save)
               for n in names}
    print("=" * 72)
    for n, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
    n_ok = sum(1 for v in results.values() if v)
    print(f"\n{n_ok}/{len(results)} passed")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
