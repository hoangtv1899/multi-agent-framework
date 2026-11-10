#!/usr/bin/env python3
"""
The years a whole-domain study reports, as distinct from the years it simulates
src/core/analysis_years.py

TWO KINDS OF YEARS, KEPT APART (2026-11-10).

  simulated   domain_experiment.years: what a run executes. A run starts from
              the control's saved state, so these start at the control's start
              year whatever the question asks about.
  reported    the years the numbers are read for: the years the person asked
              about, and the normal years they are compared with.

One list did both jobs. A question about one year of a what-if run then got
the start year in its numbers and on its page, labelled as the person's own
choice, and a question about the control alone was refused until it started
at the start year although nothing was run.

The reported years are decided here, by code, from records that exist before
any job: the plan's own `analysis.years` when the Planner wrote them, else the
period the person asked about (Reception's `requested_period`) with the plan's
normal years. The design page, the regional table and the study record all
read this one function, so they cannot disagree.
"""
from typing import Any, Dict, List, Optional, Tuple


def _ints(xs) -> List[int]:
    out = []
    for x in xs or []:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            return []
    return out


def _span(ys: List[int]) -> str:
    """Years as a person reads them: 2031, 2031-2033, or 2031, 2033."""
    ys = sorted(set(ys))
    if not ys:
        return "no year"
    if ys == list(range(ys[0], ys[-1] + 1)):
        return str(ys[0]) if len(ys) == 1 else f"{ys[0]}–{ys[-1]}"
    return ", ".join(map(str, ys))


def requested(brief: Optional[Dict[str, Any]]) -> List[int]:
    """The years the person asked about, as Reception recorded them."""
    rp = ((brief or {}).get("run_settings") or {}).get("requested_period") or {}
    y0, y1 = rp.get("yr_start"), rp.get("yr_end")
    if not isinstance(y0, int) or isinstance(y0, bool):
        return []
    y1 = y1 if isinstance(y1, int) and not isinstance(y1, bool) and y1 >= y0 else y0
    return list(range(y0, y1 + 1))


def reported(brief: Optional[Dict[str, Any]], strategy: Optional[Dict[str, Any]]
             ) -> Tuple[List[int], Dict[str, Any]]:
    """(the years the numbers are reported for, how they were decided).

    The second value is a record for the page and the table:
      simulated  the experiment's years
      asked      the years the person asked about ([] when not recorded)
      normal     the plan's normal years, inside the simulated years
      left_out   simulated years that are not reported
      source     "plan" (the Planner's analysis.years), "request" (the years
                 asked about, with the normal years) or "all" (every simulated
                 year: nothing recorded narrows them)
    """
    de = dict((brief or {}).get("domain_experiment") or {})
    de.update((strategy or {}).get("domain_experiment") or {})
    sim = _ints(de.get("years"))
    an = de.get("analysis") or {}
    normal = [y for y in _ints(an.get("normal_years")) if y in sim]
    asked = requested(brief)
    planned = _ints(an.get("years"))
    if planned and set(planned) <= set(sim):
        years, source = sorted(set(planned)), "plan"
    elif asked and set(asked) <= set(sim):
        years, source = sorted(set(asked) | set(normal)), "request"
    else:
        years, source = list(sim), "all"
    return years, {"simulated": sim, "asked": asked, "normal": normal,
                   "left_out": [y for y in sim if y not in years], "source": source}


def when_line(brief: Optional[Dict[str, Any]], strategy: Optional[Dict[str, Any]],
              start_year: Optional[int] = None, executes: bool = False) -> Optional[str]:
    """The page's When line for a whole-domain study, each part labelled by
    whose choice it is: the person's, the design's, or a rule of the model.
    None when the study has no whole-domain experiment years."""
    years, how = reported(brief, strategy)
    sim, asked, normal = how["simulated"], how["asked"], how["normal"]
    if not sim:
        return None
    parts = []
    if asked:
        parts.append(f"{_span(asked)} (the years you asked about)")
    extra_normal = [y for y in normal if y not in asked]
    if extra_normal:
        parts.append(f"compared with {_span(extra_normal)} (the normal years, chosen by "
                     f"the design)")
    others = [y for y in years if y not in asked and y not in normal]
    if others:
        parts.append(f"{_span(others)} (" + ("chosen by the design" if asked else
                                              "the design's years; the request named none")
                     + ")")
    line = "; ".join(parts) if parts else _span(years)
    if how["left_out"]:
        rule = (f"a run starts from the control's saved state at {start_year}"
                if executes and start_year is not None and min(sim) == start_year
                else "the design's own span")
        line += (f". Also simulated and not reported: {_span(how['left_out'])} "
                 f"(required: {rule})")
    return line
