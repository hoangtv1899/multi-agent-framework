#!/usr/bin/env python3
"""
The required outputs of an analysis, and what proves one
src/agents/analysis/plan_contract.py

ONE CONTRACT, THREE READERS (the user's decision after G part 2's fixes,
2026-11-08). An analysis of the plan has a SCOPE (its fields, regions,
periods, steps and methods: what a script MAY look up, which the gate
enforces) and, apart from it, its REQUIRED OUTPUTS: each the result of one
suite call, named exactly. The plan check (analysis_plan.plan_problems), the
tasks the model reads (analysis_plan.plan_task, step 2b's SUITE_TASK) and the
close of the planned phase (analysis_plan.phase_statuses) all read the
definitions here, so what the model is told an output is and what code counts
as one cannot drift apart.

Until then an analysis counted as completed only when every combination of
its fields, regions, periods and methods had been looked up: the scope read
as an obligation, so question 1's analyses, carried out, read as skipped.

WHAT PROVES AN OUTPUT. Only a lookup that matches it exactly (its call and
every key the output names) and that is planned under the analysis, made
before the close, answered, and made by a figure that ran to its end. A
figure that ran, or a field read by some other call, proves nothing. Each
output ends as exactly one of:
    produced       such a lookup gave a value
    not estimable  such a lookup was made and the method said its value
                   cannot be estimated (its own words are kept)
    unmet          no such lookup gave a value or a not-estimable answer:
                   never made, refused, failed, made by a figure that failed,
                   made after the close, or made under an amendment
An analysis is completed when no output is unmet; "completed after an
earlier failure" when it is completed and an earlier attempt failed (every
attempt is kept, in order); failed when an output is unmet and the analysis
was attempted; skipped when it was never attempted.

COMPLETION IS NOT SUPPORT. A completed analysis says its required outputs
exist (or were found not estimable), never that they support an
expectation: only the traced tested claims say that. A not-estimable output
completes its task and leaves its expectation without an estimate.
"""
import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

CONTROL = "control"
NORMAL = "normal"
PRODUCED, NOT_ESTIMABLE, UNMET = "produced", "not estimable", "unmet"
COMPLETED = "completed"
COMPLETED_AFTER_FAILURE = "completed after an earlier failure"
FAILED, SKIPPED = "failed", "skipped"
STATUSES = (COMPLETED, COMPLETED_AFTER_FAILURE, FAILED, SKIPPED)
DONE = (COMPLETED, COMPLETED_AFTER_FAILURE)
NOT_CARRIED = "no figure carried it out and the scripts gave no reason"
# what an attempt's line came to, beside produced and not estimable: each an
# earlier failure when the output is met later
FAILURES = ("refused by the gate", "refused by the data", "failed")

# THE OUTPUT KINDS. An output of a method kind is one call of that method;
# "map" is suite.get_map and "series" suite.get_series. Each timing measure
# (domain_methods.TIMING_IDS) names a region and a period (its year, as 2015
# or WY2015). The screen is never an output: it is the plan's own entry.
#   names  the keys an output of the kind must give (region null: a per-cell map)
#   may    the keys it may give, with the value meant when it gives none
KINDS: Dict[str, Dict[str, Any]] = {
    "map": {"call": "suite.get_map", "names": ("field", "period"), "may": {"run": CONTROL}},
    "series": {"call": "suite.get_series", "names": ("field", "region", "step"),
               "may": {"run": CONTROL}},
    "anomaly": {"call": "suite.anomaly", "names": ("field", "period", "region"),
                "may": {"run": CONTROL}},
    "difference": {"call": "suite.difference", "names": ("run", "field", "period", "region"),
                   "may": {}},
    "overlap": {"call": "suite.overlap", "names": ("region", "inputs"), "may": {"by_band": False}},
    "elevation_split": {"call": "suite.elevation_split", "names": ("region", "inputs"),
                        "may": {}},
}
TIMING = {"call": "suite.<measure>", "names": ("region", "period"),
          "may": {"run": CONTROL, "against": NORMAL}}
INPUTS = {"overlap": 2, "elevation_split": 1}
# the logged call of each kind, and the path of the map an overlap or an
# elevation split may take from it (a get_map is its map whole)
CALL = {"map": "get_map", "series": "get_series"}
MAP_PATH = {"map": None, "anomaly": "anomaly", "difference": "difference"}
NULL_REGION = ("anomaly", "difference")
# AN OUTPUT'S SETTINGS (the user's decision of 2026-11-08 on the open points):
# an output of these kinds is produced with the settings its analysis names
# for its method, else the method's defaults, unless it names its own
# ("settings", values the analysis allows); a map, a series and a timing
# measure name none (every stored setting of a timing measure comes with each
# of its results, and the range rule reads them)
SETTING_KINDS = ("anomaly", "difference", "overlap", "elevation_split")
SETTINGS_WORDS = ("An anomaly, a difference, an overlap or an elevation split is produced with the "
                  "settings its analysis names for that method, else the method's defaults, "
                  "unless the output names its own \"settings\" (values its analysis allows); a "
                  "map, a series and a timing measure name none, since every stored setting of a "
                  "timing measure comes with each of its results")

RULE_WORDS = ("outputs: each analysis names at least one required output, apart from its scope: "
              "each one suite call named exactly, by its kind and the keys that kind names "
              "(below), and inside the analysis's own scope, so the gate allows it; an "
              "overlap or an elevation split names as \"inputs\" the map outputs it takes, in "
              "the order the call takes them: \"O1\" for one of its own analysis, \"P2.O1\" "
              "for one of another")
PROOF_WORDS = ("The framework makes each required output's call itself, exactly as listed for it, "
               "before the figures run (after the screen in an exploratory study), and an "
               "output is produced only by a logged lookup that matches it exactly: its call, "
               "every key it names and, for an overlap or an elevation split, the maps of the "
               "outputs it names as inputs, in that order; planned under its analysis, made "
               "before the planned phase is closed, and answered. A figure that ran, or a field "
               "read by another call, proves nothing, and a figure that fails to plot takes no "
               "output away. A method's answer that a value cannot be estimated is kept as the "
               "output's outcome, \"not estimable\", apart from the outputs produced and those "
               "unmet.")
STATUS_WORDS = ("completed: no output unmet; completed after an earlier failure: the same, "
                "after an attempt that failed (every attempt is kept); failed: an output unmet "
                "after the analysis was attempted; skipped: never attempted, with step 2b's "
                "reason when it gave one")
REQUIRED_WORDS = ("Required: an analysis one of the plan's expectations names (a hypothesis "
                  "question only); a required analysis failed or skipped means the hypothesis "
                  "was not fully tested and never counts as acceptance success. Put an output "
                  "that tests no expectation in an analysis no expectation names")
SUPPORT_WORDS = ("Completion is not support: a completed analysis says its outputs exist, never "
                 "that they support an expectation, which only the traced tested claims say; a "
                 "not-estimable output completes its task and leaves its expectation without an "
                 "estimate")
# step 2b's task is a fixed string, so its words name no measure
STEP2_WORDS = ("Each analysis of the plan names its required outputs, and the brief lists the "
               "call that produces each one. " + PROOF_WORDS + " Your figures do not make "
               "those calls: each names the outputs it draws by label, as \"P1.O1\", and the "
               "framework draws them from the calls' own results. "
               + SETTINGS_WORDS + ". An analysis of the plan these figures do not draw goes "
               "in `skipped`, with your reason; its required outputs are made all the same. "
               + REQUIRED_WORDS + ". " + SUPPORT_WORDS
               + ".")
Defaults = Callable[[str], Dict[str, Any]]


def _no_defaults(kind: str) -> Dict[str, Any]:
    return {}


def named_settings(a: Dict[str, Any], kind: str) -> Dict[str, Any]:
    """The settings an analysis names for one method, {setting: value or
    [values]}, read safely (a malformed entry is the plan check's to name)."""
    st = (a.get("settings") or {}) if isinstance(a.get("settings"), dict) else {}
    kv = st.get(kind)
    return dict(kv) if isinstance(kv, dict) else {}


def expected_settings(o: Dict[str, Any], a: Dict[str, Any],
                      defaults: Defaults = _no_defaults) -> Optional[Dict[str, Any]]:
    """The settings an output is produced with: the method's defaults, then
    the one value its analysis names for each setting, then the output's
    own. None for a kind that takes no settings."""
    k = o.get("kind")
    if k not in SETTING_KINDS:
        return None
    eff = dict(defaults(k) or {})
    for key, v in named_settings(a, k).items():
        vals = v if isinstance(v, list) else [v]
        if len(vals) == 1:
            eff[key] = vals[0]
    if isinstance(o.get("settings"), dict):
        eff.update(o["settings"])
    return eff


def kind_of(o: Dict[str, Any], timing_ids: Sequence[str]) -> Optional[Dict[str, Any]]:
    k = o.get("kind")
    return KINDS.get(k) or (TIMING if k in tuple(timing_ids) else None)


def kind_lines(timing_ids: Sequence[str]) -> List[str]:
    """Each output kind as the plan task lists it: the call, the keys it
    names, and the keys it may give with what is meant when it gives none."""
    out = []
    for k, d in list(KINDS.items()) + [(", ".join(timing_ids) or "a timing measure", TIMING)]:
        may = "; may give " + ", ".join(f"{m} (else {v!r})" for m, v in d["may"].items()) \
            if d["may"] else ""
        nul = " (region null for a per-cell map)" if k in NULL_REGION else ""
        out.append(f"{k} ({d['call']}): names {', '.join(d['names'])}{nul}{may}")
    return out


def _year(p) -> Optional[int]:
    s = str(p)
    if s.startswith("WY") and s[2:].isdigit():
        return int(s[2:])
    return int(s[:4]) if s[:4].isdigit() else None


def as_lookup(o: Dict[str, Any], timing_ids: Sequence[str]) -> Dict[str, Any]:
    """An output as the gate would see its call (analysis_plan._lookup's
    shape), so the plan check can ask the gate's own rule whether the
    analysis's scope allows it."""
    k = o.get("kind")
    own = o.get("settings") if k in SETTING_KINDS and isinstance(o.get("settings"), dict) else {}
    lk = {"call": CALL.get(k, k), "method": None if k in CALL else k, "field": o.get("field"),
          "region": o.get("region"), "period": None, "year": None, "step": None,
          "settings": dict(own)}
    if k in tuple(timing_ids):
        lk["year"] = _year(o.get("period"))
    elif k == "series":
        lk["step"] = o.get("step")
    elif k not in INPUTS:
        lk["period"] = o.get("period")
    if k in INPUTS:
        lk["field"] = None
    return lk


# ─────────────────────────────────────────────────────────────────────
# THE PLAN CHECK'S RULE
# ─────────────────────────────────────────────────────────────────────
def plan_index(plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Every output of the plan by "P2.O1", the name an input of another
    analysis gives it."""
    out = {}
    for a in plan.get("analyses") or []:
        if isinstance(a, dict):
            for o in a.get("outputs") or []:
                if isinstance(o, dict) and o.get("id"):
                    out[f"{a.get('id')}.{o['id']}"] = o
    return out


def _input(x: Any, own: Dict[str, Dict[str, Any]],
           index: Optional[Dict[str, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """An input's output: "O1" of its own analysis, "P2.O1" of another."""
    x = str(x)
    return own.get(x) if "." not in x else (index or {}).get(x)


def output_problems(a: Dict[str, Any], timing_ids: Sequence[str],
                    not_declared: Callable[[Dict[str, Any]], Optional[str]],
                    index: Optional[Dict[str, Dict[str, Any]]] = None,
                    defaults: Defaults = _no_defaults) -> List[str]:
    """What is wrong with an analysis's required outputs: none named, an id
    missing or repeated, a kind that does not exist (the screen is never
    one), a key its kind names missing or one it does not know, an output
    the analysis's own scope does not allow (`not_declared`, the gate's
    rule), a method kind the analysis does not run, or inputs that are not
    map outputs of the plan (`index`, by "P2.O1")."""
    aid = a.get("id")
    outs = a.get("outputs")
    if not isinstance(outs, list) or not outs:
        return [f"{aid}: names no required output (\"outputs\"): each analysis names the outputs "
                f"it must produce, apart from its scope"]
    probs, seen = [], {}
    for o in outs:
        if not isinstance(o, dict):
            probs.append(f"{aid}: an output is not an object: {o!r}")
            continue
        oid = o.get("id")
        if not oid or not isinstance(oid, str):
            probs.append(f"{aid}: an output has no id: {o!r}")
            continue
        if oid in seen:
            probs.append(f"{aid}: output {oid!r} is named twice")
            continue
        seen[oid] = o
    for oid, o in seen.items():
        at = f"{aid}: output {oid!r}"
        k = o.get("kind")
        d = kind_of(o, timing_ids)
        if k == "screen":
            probs.append(f"{at}: the screen is never an output (it is the plan's own entry)")
            continue
        if d is None:
            probs.append(f"{at}: no output kind {k!r} (one of: "
                         f"{', '.join(list(KINDS) + list(timing_ids))})")
            continue
        miss = [n for n in d["names"] if n not in o]
        if miss:
            probs.append(f"{at}: a {k} output names {', '.join(d['names'])}; it lacks "
                         f"{', '.join(miss)}")
            continue
        if "settings" in o:
            st = o["settings"]
            if k not in SETTING_KINDS:
                probs.append(f"{at}: a {k} output names no settings ("
                             + ("every stored setting of a timing measure comes with each of its "
                                "results" if k in timing_ids else "it takes none") + ")")
                continue
            if not isinstance(st, dict):
                probs.append(f"{at}: its settings are not an object: {st!r}")
                continue
            unknown = sorted(set(st) - set(defaults(k) or {}))
            if unknown:
                probs.append(f"{at}: {k} has no setting {', '.join(map(repr, unknown))}")
                continue
        elif k in SETTING_KINDS:
            many = [key for key, v in named_settings(a, k).items()
                    if isinstance(v, list) and len(v) > 1]
            if many:
                probs.append(f"{at}: its analysis names several values of {k}'s "
                             f"{', '.join(many)}: the output names the one it is produced with "
                             f"(\"settings\")")
                continue
        extra = sorted(set(o) - {"id", "kind"} - set(d["names"]) - set(d["may"])
                       - ({"settings"} if k in SETTING_KINDS else set()))
        if extra:
            probs.append(f"{at}: a {k} output knows no {', '.join(extra)}")
            continue
        if o.get("region") is None and "region" in d["names"] and k not in NULL_REGION:
            probs.append(f"{at}: a {k} output names its region")
            continue
        if k == "difference" and (not o.get("run") or o.get("run") == CONTROL):
            probs.append(f"{at}: a difference names the run it compares with the control")
            continue
        if k in timing_ids and _year(o.get("period")) is None:
            probs.append(f"{at}: its period {o.get('period')!r} is not a year (2015, WY2015)")
            continue
        if k not in CALL and k not in (a.get("methods") or []):
            probs.append(f"{at}: its kind {k!r} is not among the analysis's methods")
            continue
        why = not_declared(as_lookup(o, timing_ids))
        if why:
            probs.append(f"{at}: outside the analysis's scope: it declares no {why}, so the gate "
                         f"would refuse the call")
            continue
        if k in INPUTS:
            ins = o.get("inputs")
            if not isinstance(ins, list) or len(ins) != INPUTS[k]:
                probs.append(f"{at}: a {k} names {INPUTS[k]} input"
                             f"{'s' if INPUTS[k] > 1 else ''} (map outputs: \"O1\" of its own "
                             f"analysis, \"P2.O1\" of another), not {ins!r}")
                continue
            for x in ins:
                src = _input(x, seen, index)
                if src is None:
                    probs.append(f"{at}: its input {x!r} is no output of the plan (\"O1\" of "
                                 f"{aid}, or \"P2.O1\" of another analysis)")
                elif src.get("kind") not in MAP_PATH or \
                        (src.get("kind") in NULL_REGION and src.get("region") is not None):
                    probs.append(f"{at}: its input {x!r} is not a map (a map output, or a "
                                 f"per-cell anomaly or difference: region null)")
    return probs


# ─────────────────────────────────────────────────────────────────────
# WHAT A LOOKUP PROVES
# ─────────────────────────────────────────────────────────────────────
def answer(line: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """What an answered line gives its output: produced, or not estimable
    with the method's own words, read from the value each kind returns."""
    call, res = line.get("call"), line.get("result") or {}
    if call in ("get_map", "get_series"):
        return PRODUCED, None
    if call in ("anomaly", "difference"):
        w = res.get("why")
        return (NOT_ESTIMABLE, str(w)) if isinstance(w, str) and w.strip() else (PRODUCED, None)
    if call == "overlap":
        if res.get("correlation") is None:
            return NOT_ESTIMABLE, str(res.get("correlation_why_not") or res.get("why")
                                      or "no correlation could be made")
        return PRODUCED, None
    if call == "elevation_split":
        if res.get("concentration") is None:
            return NOT_ESTIMABLE, str(res.get("why") or "no cell of the region holds a value "
                                      "of its map")
        return PRODUCED, None
    path = "variants.default.shift.median"            # a timing measure
    if path not in res:
        return UNMET, f"its result holds no {path}"
    if res[path] is None:
        return NOT_ESTIMABLE, str(res.get("across_settings.why") or "no cell has a date in both "
                                  "the year and its baseline")
    return PRODUCED, None


def _source(lines: Sequence[Dict[str, Any]], sha: Any, before: int) -> Optional[Tuple[int, Any]]:
    """The answered line before `before` that returned the map of this hash,
    and the path of the map in its result (None for a get_map's whole)."""
    for j in range(min(before, len(lines)) - 1, 0, -1):
        ln = lines[j - 1]
        if "sha256" not in ln or ln.get("refused") or ln.get("failed"):
            continue
        if ln.get("call") == "get_map" and ln.get("sha256") == sha:
            return j, None
        for p, h in (ln.get("arrays") or {}).items():
            if h == sha and ln.get("call") in ("anomaly", "difference"):
                return j, p
    return None


def _same(o: Dict[str, Any], ln: Dict[str, Any], timing_ids: Sequence[str]) -> bool:
    """The output's call and every key it names, against a line's (inputs
    apart)."""
    k = o.get("kind")
    if ln.get("call") != CALL.get(k, k):
        return False
    keys = ln.get("keys") or {}
    d = kind_of(o, timing_ids) or {"names": (), "may": {}}
    for n in d["names"]:
        if n == "inputs":
            continue
        if n == "period" and k in tuple(timing_ids):
            if keys.get("year") is None or _year(o.get("period")) != int(keys["year"]):
                return False
        elif n == "period":
            if str(o.get("period")) != str(keys.get("period")):
                return False
        elif o.get(n) != keys.get(n):
            return False
    for n, dflt in d["may"].items():
        if (o.get(n) if o.get(n) is not None else dflt) != \
                (keys.get(n) if keys.get(n) is not None else dflt):
            return False
    return True


def canon(v: Any) -> Any:
    """A setting's value as the lookups log keeps it (domain_access._canon):
    a float by its repr, so the plan's 0.7 and the logged "0.7" are one
    value. Found by the test that runs each listed call (2026-11-08)."""
    if isinstance(v, dict):
        return {str(k): canon(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [canon(x) for x in v]
    if isinstance(v, float):
        return repr(float(v))
    return v


def line_settings(o: Dict[str, Any], ln: Dict[str, Any],
                  defaults: Defaults = _no_defaults) -> Dict[str, Any]:
    """The settings a line's call ran with: the method's defaults and what
    the call named, both as the log keeps a value (canon)."""
    eff = canon(dict(defaults(o.get("kind")) or {}))
    eff.update(canon(ln.get("settings") or {}))
    return eff


def matches(o: Dict[str, Any], i: int, lines: Sequence[Dict[str, Any]],
            outs: Dict[str, Dict[str, Any]], timing_ids: Sequence[str],
            index: Optional[Dict[str, Dict[str, Any]]] = None,
            expect: Optional[Dict[str, Any]] = None,
            defaults: Defaults = _no_defaults) -> bool:
    """Whether line i (1-based) is a call of the output: the same call and
    keys, the settings it is produced with (`expect`, when its kind takes
    settings), and for an overlap or an elevation split, maps from lookups
    that are the outputs it names as inputs, in their order."""
    ln = lines[i - 1]
    if not _same(o, ln, timing_ids):
        return False
    if expect is not None and line_settings(o, ln, defaults) != canon(expect):
        return False
    if o.get("kind") not in INPUTS:
        return True
    keys = ln.get("keys") or {}
    shas = [keys.get("a"), keys.get("b")] if o["kind"] == "overlap" else [keys.get("x")]
    for x, sha in zip(o.get("inputs") or [], shas):
        src, io = _source(lines, sha, i), _input(x, outs, index)
        if src is None or io is None:
            return False
        j, path = src
        if path != MAP_PATH.get(io.get("kind"), "?") or not _same(io, lines[j - 1], timing_ids):
            return False
    return True


def _round(r) -> str:
    return f" (round {r})" if r is not None else ""


def _figure(i: int, ln: Dict[str, Any], figures: Sequence[Dict[str, Any]]):
    """The figure attempt that made line i: the one whose lines hold it, else
    (a record without line ranges) the last attempt of its script."""
    for f in figures:
        lo, hi = (list(f.get("lines") or []) + [None, None])[:2]
        if lo is not None and hi is not None and lo <= i <= hi:
            return f
    same = [f for f in figures if f.get("id") == ln.get("script") and not f.get("lines")]
    return same[-1] if same else None


def attempt(i: int, ln: Dict[str, Any], aid: str, figures: Sequence[Dict[str, Any]],
            closed_after: Optional[int]) -> Tuple[str, Optional[str]]:
    """What line i, a call of one of the analysis's outputs, came to. A call
    the gate refused carries no marks, so it is the output's by its call
    alone; any other line counts for the analysis only when the gate passed
    it under the analysis."""
    r = str(ln.get("refused") or "")
    if r.startswith("the gate:"):
        return "refused by the gate", r
    items = ln.get("plan_items") or []
    if ln.get("after_close") is True or (closed_after is not None and i > closed_after):
        return "made after the close", "a lookup after the close is exploratory, never planned"
    if ln.get("amendment") or ln.get("planned") is not True:
        return "not planned", ("made under amendment " + str(ln.get("amendment"))
                               if ln.get("amendment") else "the gate did not pass it as planned")
    if aid not in items:
        return "not under this analysis", "the gate passed it under " + \
            (", ".join(items) or "no analysis")
    if ln.get("failed"):
        return "failed", str(ln["failed"])
    if r:
        return (NOT_ESTIMABLE, r) if r.lower().startswith("not estimable") \
            else ("refused by the data", r)
    # PLOTTING INDEPENDENCE (the user's requirement, confirmed 2026-11-08): a
    # logged, answered, planned call is the output whatever its figure did
    # after; a figure that failed to plot stays in the attempts only
    if "sha256" not in ln:
        return "unanswered", "the line holds no answer"
    return answer(ln)


# ─────────────────────────────────────────────────────────────────────
# ONE ANALYSIS AT THE CLOSE
# ─────────────────────────────────────────────────────────────────────
def evaluate(a: Dict[str, Any], lines: Sequence[Dict[str, Any]],
             figures: Sequence[Dict[str, Any]] = (), stated: Optional[str] = None,
             timing_ids: Sequence[str] = (), closed_after: Optional[int] = None,
             strip: Callable[[str], str] = str,
             index: Optional[Dict[str, Dict[str, Any]]] = None,
             defaults: Defaults = _no_defaults) -> Dict[str, Any]:
    """An analysis at the close: each required output with its outcome
    (produced, not estimable or unmet), the record that settled it and every
    attempt at it; the analysis's status and reason; and its attempts in
    order (each figure that named it). The outputs produced, not estimable
    and unmet are listed apart."""
    aid = a.get("id")
    outs = {o["id"]: o for o in a.get("outputs") or [] if isinstance(o, dict) and o.get("id")}
    figs = [f for f in figures or [] if aid in (f.get("plan_items") or [])]
    history = [{"round": f.get("round"), "figure": f.get("id"), "ok": bool(f.get("ok")),
                "error": None if f.get("ok") else f.get("error"), "lines": f.get("lines"),
                **({"what": f["what"]} if f.get("what") else {})}
               for f in figs]
    # the failures of the output's own calls, each where it falls in the log;
    # a figure that failed to plot is no failure of an output (plotting
    # independence), and is kept in the attempts and named when one is unmet
    failures: List[Tuple[float, str]] = []
    plotting = [(f.get("what") or f"the figure '{f['figure']}'")
                + f"{_round(f['round'])} failed: {f['error']}"
                for f in history if not f["ok"]]
    rows = []
    for oid, o in outs.items():
        tried = []
        exp = expected_settings(o, a, defaults)
        for i in range(1, len(lines) + 1):
            if not matches(o, i, lines, outs, timing_ids, index, defaults=defaults):
                continue
            if exp is not None and line_settings(o, lines[i - 1], defaults) != canon(exp):
                # ITS CALL, WITH OTHER SETTINGS: not the output, kept as tried
                tried.append({"line": f"L{i}", "at": i, "result": "made with other settings",
                              "why": f"made with {line_settings(o, lines[i - 1], defaults)}, "
                                     f"not {exp}"})
                continue
            res, why = attempt(i, lines[i - 1], aid, figures, closed_after)
            tried.append({"line": f"L{i}", "at": i, "result": res, "why": why})
        got = next((t for t in tried if t["result"] == PRODUCED), None) or \
            next((t for t in tried if t["result"] == NOT_ESTIMABLE), None)
        failures += [(t["at"], f"{t['line']} ({oid}) {t['result']}: {t['why']}") for t in tried
                     if t["result"] in FAILURES]
        if got:
            row = {"id": oid, "spec": o, "outcome": got["result"], "record": got["line"],
                   "at": got["at"], "why": got["why"]}
            if exp is not None:
                row["settings"] = exp
        else:
            ran = [f["figure"] for f in history if f["ok"]]
            why = ("; ".join(f"{t['line']} {t['result']}: {t['why']}" for t in tried[:3])
                   if tried else "no planned lookup matched it"
                   + (f" (the figure{'s' if len(ran) > 1 else ''} "
                      f"{', '.join(repr(x) for x in ran)} naming this analysis ran without it)"
                      if ran else ""))
            row = {"id": oid, "spec": o, "outcome": UNMET, "record": None, "why": why}
            if exp is not None:
                row["settings"] = exp
        row["attempts"] = [{k: t[k] for k in ("line", "result", "why")} for t in tried]
        rows.append(row)
    unmet = [r for r in rows if r["outcome"] == UNMET]
    tried_any = bool(history) or any(r["attempts"] for r in rows)
    failures.sort(key=lambda x: x[0])
    if not rows:
        status, reason = SKIPPED, "it names no required output"
    elif not unmet:
        ne = sum(r["outcome"] == NOT_ESTIMABLE for r in rows)
        done = f"{len(rows) - ne} required output{'s' if len(rows) - ne != 1 else ''} produced" \
            + (f", {ne} not estimable" if ne else "")
        # AN EARLIER FAILURE is one before the line that met the last output;
        # a failed attempt after it stays in the attempts and changes nothing
        last = max(r["at"] for r in rows)
        earlier = [w for at, w in failures if at < last]
        if earlier:
            status, reason = COMPLETED_AFTER_FAILURE, (f"{done}, after an earlier failure: "
                                                       f"{earlier[0]}")
        else:
            status, reason = COMPLETED, done
    elif tried_any:
        status = FAILED
        reason = "; ".join(f"{r['id']} unmet: {r['why']}" for r in unmet[:3]) + \
            (f"; and {len(unmet) - 3} more unmet" if len(unmet) > 3 else "")
        if failures and failures[0][1] not in reason:
            reason += f"; {failures[0][1]}"
        elif plotting and plotting[0] not in reason:
            reason += f"; {plotting[0]}"
    else:
        status = SKIPPED
        reason = f"step 2b's reason: {strip(stated)}" if str(stated or "").strip() else NOT_CARRIED
    for r in rows:
        r.pop("at", None)
    return {"id": aid, "status": status, "reason": strip(reason),
            "scope": {k: list(a.get(k) or []) for k in ("fields", "regions", "periods",
                                                         "steps", "methods")},
            "outputs": rows,
            "produced": [r["id"] for r in rows if r["outcome"] == PRODUCED],
            "not_estimable": [r["id"] for r in rows if r["outcome"] == NOT_ESTIMABLE],
            "unmet": [r["id"] for r in unmet], "attempts": history,
            "failures": [strip(w) for _, w in failures]}


# ─────────────────────────────────────────────────────────────────────
# THE CALL THAT PRODUCES EACH OUTPUT, AS STEP 2b IS TOLD IT
# (the user's decision of 2026-11-08 on the open points)
# ─────────────────────────────────────────────────────────────────────
# An overlap's inputs keep their order (its quarter split is by its second
# map), so the brief writes every required output as the exact call that
# produces it, from this contract: what step 2b is told to call is what
# matches() accepts (tests/test_plan_contract.py runs each call and checks).
def _py(v: Any) -> str:
    return json.dumps(v) if isinstance(v, str) else repr(v)


def _owner(x: Any, aid: Any) -> Tuple[str, str]:
    x = str(x)
    return tuple(x.split(".", 1)) if "." in x else (str(aid), x)     # type: ignore


def call_spec(o: Dict[str, Any], a: Dict[str, Any], timing_ids: Sequence[str],
              defaults: Defaults = _no_defaults) -> Dict[str, Any]:
    """The call that produces an output, as data: the suite method, its
    positional and keyword arguments, and in place of an input map the
    plan name of the output that gives it ({"input": "P2.O1"}). The brief's
    list (call_text) and the framework's own calls (analysis_plan.
    produce_required) are both made from this, so what step 2b is told and
    what is called cannot differ."""
    k, run = o.get("kind"), o.get("run") or CONTROL
    st = expected_settings(o, a, defaults)
    d = (defaults(k) or {}) if st is not None else {}
    diff = {x: v for x, v in (st or {}).items() if d.get(x) != v}
    kw: Dict[str, Any] = {}
    if k == "map":
        return {"fn": "get_map", "args": [run, o.get("field"), str(o.get("period"))],
                "kwargs": {}}
    if k == "series":
        return {"fn": "get_series", "args": [run, o.get("field"), o.get("region"),
                                             o.get("step")], "kwargs": {}}
    if k == "anomaly":
        kw["region"] = o.get("region")
        if run != CONTROL:
            kw["run"] = run
        args = [o.get("field"), str(o.get("period"))]
        fn = "anomaly"
    elif k == "difference":
        kw["region"] = o.get("region")
        args, fn = [o.get("run"), o.get("field"), str(o.get("period"))], "difference"
    elif k in tuple(timing_ids):
        ag = o.get("against") or NORMAL
        if run != CONTROL:
            kw["run"] = run
        if ag != NORMAL:
            kw["against"] = ag
        return {"fn": k, "args": [_year(o.get("period")), o.get("region")], "kwargs": kw}
    else:
        own = a.get("id")
        ins = [{"input": ".".join(_owner(x, own))} for x in o.get("inputs") or []]
        args, fn = ins + [o.get("region")], k
        if k == "overlap" and o.get("by_band"):
            kw["by_band"] = True
    if diff:
        kw["settings"] = diff
    return {"fn": fn, "args": args, "kwargs": kw}


def map_code(io: Dict[str, Any], ia: Dict[str, Any], timing_ids: Sequence[str],
             defaults: Defaults = _no_defaults) -> str:
    """The expression giving an input output's map: a get_map whole, or the
    map of a per-cell anomaly or difference by its path."""
    c = call_text(io, ia, {}, timing_ids, defaults)
    path = MAP_PATH.get(io.get("kind"))
    return c if path is None else f"{c}[{_py(path)}]"


def call_text(o: Dict[str, Any], a: Dict[str, Any], index: Optional[Dict[str, Dict[str, Any]]],
              timing_ids: Sequence[str], defaults: Defaults = _no_defaults,
              maps: Optional[Dict[str, str]] = None,
              analyses: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    """The suite call that produces an output, written from call_spec. An
    input map is written as the call that gives it (or, with `maps`, as the
    name of a variable already holding it)."""
    cs = call_spec(o, a, timing_ids, defaults)

    def arg(v):
        if not (isinstance(v, dict) and "input" in v):
            return _py(v)
        label = v["input"]
        owner = label.split(".", 1)[0]
        if maps is not None:
            return maps.get(label, "map_" + label.replace(".", "_"))
        io = (index or {}).get(label)
        if io is None and owner == str(a.get("id")):
            io = {p.get("id"): p for p in a.get("outputs") or []
                  if isinstance(p, dict)}.get(label.split(".", 1)[1])
        ia = (analyses or {}).get(owner) or (a if owner == str(a.get("id")) else {})
        if io is None:
            return f"<the map of {label}>"
        return f"<the map of {label}: {map_code(io, ia, timing_ids, defaults)}>"
    parts = [arg(v) for v in cs["args"]] + [f"{k}={_py(v)}" for k, v in cs["kwargs"].items()]
    return f"suite.{cs['fn']}({', '.join(parts)})"


def required_calls(plan: Dict[str, Any], timing_ids: Sequence[str],
                   defaults: Defaults = _no_defaults) -> List[Dict[str, Any]]:
    """Every required output of the plan with its call, in the order the
    framework makes them: each map an overlap or an elevation split takes
    before the call that takes it (array kinds last)."""
    out = []
    for a in plan.get("analyses") or []:
        if not isinstance(a, dict):
            continue
        for o in a.get("outputs") or []:
            if isinstance(o, dict) and o.get("id"):
                out.append({"label": f"{a.get('id')}.{o['id']}", "analysis": a.get("id"),
                            "output": o, "call": call_spec(o, a, timing_ids, defaults)})
    return sorted(out, key=lambda c: c["output"].get("kind") in INPUTS)


def call_lines(plan: Dict[str, Any], timing_ids: Sequence[str],
               defaults: Defaults = _no_defaults,
               units: Optional[Dict[str, str]] = None) -> List[str]:
    """Step 2b's list of every required output with the call that produces
    it, analysis by analysis, from the reviewed plan; with `units`, the unit
    a bars or series panel draws each output in (figure_spec.output_unit)."""
    index = plan_index(plan)
    analyses = {str(a.get("id")): a for a in plan.get("analyses") or [] if isinstance(a, dict)}
    import textwrap
    L = textwrap.wrap("THE REQUIRED OUTPUTS, EACH WITH THE CALL THAT PRODUCES IT (from the plan "
                      "above: the framework makes each call exactly as written, an overlap or "
                      "an elevation split given the maps named, in this order, and a figure "
                      "names an output by its label to draw it"
                      + ("; beside each output a bars or series panel can draw, its unit"
                         if units else "") + "):", 79)
    for aid, a in analyses.items():
        for o in a.get("outputs") or []:
            if isinstance(o, dict) and o.get("id"):
                lab = f"{aid}.{o['id']}"
                L.append(f"    {lab}: "
                         + call_text(o, a, index, timing_ids, defaults, analyses=analyses)
                         + (f"  (unit: {units[lab]})" if units and units.get(lab) else ""))
    return L
