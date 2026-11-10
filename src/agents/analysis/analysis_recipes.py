"""Analysis recipes: how a kind of question is covered with the methods the
catalog already has (the user's decision of 2026-11-09).

DRAFT, FOR THE USER'S REVIEW. A recipe is guidance for the analysis plan. It
is kept apart from the numerical methods: domain_methods and
docs/ANALYSIS_METHODS.md define those, and a recipe only says which of their
existing calls a kind of question needs. It names no watershed, no value and
no result to expect. The plan check does not enforce it.

Which parts of a recipe apply is read from the question's own words, by code
(applies), so one question always gets the same guidance, whatever model
plans it.

    applies(question)             {part: the question's words that call for it}
    plan_lines(question)          the block step 2a reads before its task
    coverage(question, plan)      what a plan holds of each part that applies
    report_lines(question, plan)  the block step 3 reads, for the parts the
                                  plan carried out

THE RECIPE: SEASONAL EVOLUTION AND SPATIAL CONTEXT. The accepted reports of
2026-11-08 answered a question about a season with the season's and the
year's regional values only: nothing said in which months a departure began
or ended, nor where in the region each field's departure sat.
"""
import re
from typing import Any, Dict, List, Optional

NAME = "seasonal evolution and spatial context"
SEASONAL, SPATIAL = "seasonal_evolution", "spatial_context"
# THE WORDS THAT CALL FOR EACH PART, whole words in any case. Seasonal: a
# season or a month the question names (below), or a word that asks about
# timing (analysis_plan.timing_words, the timing rule's own). Spatial: a word
# about place within the region, or one that sets two things side by side.
SEASON_WORDS = ("spring", "summer", "autumn", "fall", "winter", "season", "seasonal",
                "month", "monthly", "january", "february", "march", "april", "may", "june",
                "july", "august", "september", "october", "november", "december")
PLACE_WORDS = ("where", "elevation", "elevations", "altitude", "altitudes", "band", "bands",
               "spatial", "spatially", "pattern", "patterns", "areas", "places", "located")
RELATION_WORDS = ("accompany", "accompanies", "accompanied", "coincide", "coincides",
                  "coincided", "together", "alongside", "co-occur", "co-occurs",
                  "co-occurred", "co-located", "linked", "related", "associated")
MONTH_KINDS = ("anomaly", "difference")      # the kinds that give a region's month
MAP_KINDS = ("map", "anomaly", "difference")
_WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
_MONTH = re.compile(r"^(\d{4})-(\d{2})$")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def _found(question: Any, words) -> List[str]:
    seen = []
    for w in _WORD.findall(str(question or "")):
        if w.lower() in words and w.lower() not in seen:
            seen.append(w.lower())
    return seen


def applies(question: Any) -> Dict[str, List[str]]:
    """{part: the question's words that call for it}, a part only when the
    question has such a word."""
    from agents.analysis import analysis_plan as AP
    timing = [w.lower() for w in AP.timing_words(question)]
    out = {SEASONAL: list(dict.fromkeys(timing + _found(question, SEASON_WORDS))),
           SPATIAL: _found(question, PLACE_WORDS + RELATION_WORDS)}
    return {k: v for k, v in out.items() if v}


def _said(words: List[str]) -> str:
    return ", ".join(f'"{w}"' for w in words)


def plan_lines(question: Any) -> List[str]:
    """The recipe as step 2a reads it, for the parts the question calls for;
    [] when it calls for none."""
    parts = applies(question)
    if not parts:
        return []
    y = _YEAR.search(str(question or ""))
    y = y.group(0) if y else "YYYY"
    L = [f"THE ANALYSIS RECIPE FOR THIS QUESTION: {NAME}.",
         "Guidance for the plan, apart from the methods: it uses the methods above only, the",
         "plan check does not enforce it, and it expects no result. Follow it unless the",
         "catalog lacks what a step needs, and then say so in that analysis's \"why\".",
         "KEY FIELDS: the fields your expectations name (for an exploratory question, the",
         "fields the question itself names; with none, plan nothing from this recipe).",
         "Put the recipe's outputs in analyses of their own that no expectation names: they",
         "describe, and they test nothing."]
    if SEASONAL in parts:
        L += ["",
              f"SEASONAL EVOLUTION (the question says {_said(parts[SEASONAL])}). A season's or a",
              "year's value does not say in which months a departure begins, is largest or ends.",
              "For each key field, over the question's region:",
              "  - its departure in each of the twelve months of the question's year: one",
              "    anomaly output a month (a difference, when a run is compared with the",
              f"    control), periods \"{y}-01\" to \"{y}-12\", each naming the region (a month",
              "    is a region's value only, never a per-cell map);",
              "  - its series at step \"day\" (\"month\" for a field saved monthly), which a figure",
              "    draws against the normal years' range;",
              "  - its value for the whole year beside the season's, when the question names a",
              "    season."]
    if SPATIAL in parts:
        L += ["",
              f"SPATIAL CONTEXT (the question says {_said(parts[SPATIAL])}). A region's mean does",
              "not say where in the region a departure sits, for one field or for two. For each",
              "key field, over the question's region and period:",
              "  - its per-cell departure: an anomaly (or a difference) with region null, which a",
              "    figure draws as a map;",
              "  - that map by elevation band: an elevation_split of it over the region, for each",
              "    key field on its own;",
              "  - for two key fields, the overlap of their maps over the region, with by_band",
              "    true."]
    return L


def _outputs(plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {f"{a.get('id')}.{o.get('id')}": o for a in (plan or {}).get("analyses") or []
            if isinstance(a, dict) for o in a.get("outputs") or [] if isinstance(o, dict)}


def coverage(question: Any, plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """What a plan holds of each part the question calls for. Seasonal:
    {"months": {field: its months planned as a region's anomaly or
    difference}, "series": {field: its steps}}. Spatial: {"maps": the fields
    with a per-cell map, "splits": the fields whose map an elevation split
    takes, "overlaps": how many, "by_band": how many of them by band}."""
    outs = _outputs(plan)
    out: Dict[str, Dict[str, Any]] = {}
    parts = applies(question)
    if SEASONAL in parts:
        months: Dict[str, List[str]] = {}
        series: Dict[str, List[str]] = {}
        for o in outs.values():
            m = _MONTH.match(str(o.get("period") or ""))
            if o.get("kind") in MONTH_KINDS and m and o.get("region") and o.get("field"):
                months.setdefault(str(o["field"]), []).append(m.group(2))
            if o.get("kind") == "series" and o.get("field") and o.get("step") in ("day", "month"):
                series.setdefault(str(o["field"]), []).append(str(o["step"]))
        out[SEASONAL] = {"months": {f: sorted(set(ms)) for f, ms in months.items()},
                         "series": {f: sorted(set(s)) for f, s in series.items()}}
    if SPATIAL in parts:
        def field_of(ref: Any, owner: str) -> Optional[str]:
            key = str(ref) if "." in str(ref) else f"{owner}.{ref}"
            return (outs.get(key) or {}).get("field")
        maps = sorted({str(o["field"]) for o in outs.values() if o.get("field") and (
            o.get("kind") == "map" or (o.get("kind") in MAP_KINDS and "region" in o
                                       and o.get("region") is None))})
        splits, overlaps, by_band = set(), 0, 0
        for key, o in outs.items():
            owner = key.split(".")[0]
            if o.get("kind") == "elevation_split":
                splits |= {f for f in (field_of(x, owner) for x in o.get("inputs") or []) if f}
            elif o.get("kind") == "overlap":
                overlaps += 1
                by_band += bool(o.get("by_band"))
        out[SPATIAL] = {"maps": maps, "splits": sorted(splits), "overlaps": overlaps,
                        "by_band": by_band}
    return out


def report_lines(question: Any, plan: Dict[str, Any]) -> List[str]:
    """What step 3 is asked to say of the parts the plan carried out: the
    course through the year, and where in the region, each from its own
    records; [] when the plan carried out none."""
    cov = coverage(question, plan)
    months = (cov.get(SEASONAL) or {}).get("months") or {}
    sp = cov.get(SPATIAL) or {}
    L: List[str] = []
    if months:
        L += [f"  - THE COURSE THROUGH THE YEAR (months planned for {', '.join(sorted(months))}):",
              "    for each of these fields, at least one claim on its month-by-month course,",
              "    from its months' own records: the months in which its departure is largest,",
              "    and the sign of its departure in the months outside the season the question",
              "    names. Where the question names a season, give the year's value beside it."]
    if sp.get("splits") or sp.get("overlaps"):
        L += ["  - WHERE IN THE REGION"
              + (f" (maps split by band for {', '.join(sp['splits'])})" if sp.get("splits")
                 else "") + ":",
              "    for each field whose map is split by band, at least one claim on where its",
              "    departure sits, from the split's own record: its band means, and whether its",
              "    sign is the same in every band. A claim on two fields' overlap says which two",
              "    maps it compares."]
    if not L:
        return []
    return ([f"THE PLAN CARRIED OUT THE RECIPE \"{NAME}\". Its outputs describe: a claim from",
             "them is an observation (an association, for an overlap), never a cause."] + L
            + ["In \"answer\", after what was found, say in one sentence each how the departure",
               "ran through the year and where in the region it sat, each restating a claim; the",
               "answer may then run to six sentences."])
