#!/usr/bin/env python3
"""
Agentic reception agent.

A pure-LLM reception that DRIVES the MCP tools itself (via ToolLoopAgent) to
turn a natural-language request into a framed brief for the planner — or a
clarification / analysis route. All "what to fetch / when to stop / what it
means" decisions live in the LLM; this class only runs the loop and parses the
final JSON.

This is the Reception box of the framework. workflow.py reaches it through
workflow.py directly; tools/run_pipeline.py uses it for the dry path.
"""
import json
import re
from typing import Any, Dict

from agents.prompts import load_prompt
from core.sweep_menu import render_sweep_menu
from core.model_servers import model_tools
from agents.tool_loop import ToolLoopAgent
from core import data_gather as gather

# What the LLM may call. Deliberately tiny.
#
# Everything else reception fetches — the DEM grid, the water table, the three
# observation sets, the subsurface and the rain — is fetched by CODE once
# and period are fixed
# (see reception_gather). Those are not decisions, so they should not cost an
# LLM round each: with twelve tools the loop spent 100-250 s of a 167-315 s
# reception choosing tools, and the model then summarised results truncated to
# 12,000 characters, which is how it came to report 25 stream gauges where the
# validator found 93.
#
# The model's job is the part that needs judgement: what is being asked, where,
# and when. The subsurface and the rain are not among the LLM's tools either —
# ELM gets both from its own inputs (the CONUS 1 km donor gridcell the warm
# start subsets, and NLDAS-2 off disk), and what a non-ELM model needs is
# fetched by code (gather_subsurface, gather_precipitation), not chosen.
#
# WHICH MODEL IS THE EXCEPTION, and deliberately so. It is a judgement, not a
# fetch, and it is the one judgement that used to be made for the LLM by code
# before the request was even read. `list_servers` and `describe_server` are
# LOCAL tools — the framework answers them, no server is asked — so they are
# not in this allowlist; see model_servers.model_tools.
DEFAULT_ALLOWLIST = {
    "terrain__resolve_watershed",
    "terrain__get_elevation",        # fallback: a point or town, no HUC given
}


def _is_conceptual(brief: Dict[str, Any]) -> bool:
    """Did reception classify this as a controlled sweep?

    One reading of the field, so the gather gate and the domain drop can never
    disagree about what this brief is.
    """
    return str((brief or {}).get("design_archetype") or "").strip().lower() \
        == "conceptual"


def _drop_domain(brief: Dict[str, Any]) -> Dict[str, Any]:
    """Remove a basin from a conceptual brief, returning what was removed.

    RETURNED RATHER THAN DISCARDED. Setting it to null and saying nothing would
    hide a real disagreement — the model both classified the request as a sweep
    and resolved a watershed for it, and someone reading the run later should
    be able to see that happened. It is also the fastest way to notice the
    classifier is mis-reading a whole class of request.
    """
    dom = brief.get("domain") or {}
    had = {k: dom.get(k) for k in ("name", "huc", "bbox") if dom.get(k)}
    if had:
        brief["domain"] = None
        brief["heterogeneity"] = None
    return had


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
        # THE FORCING WINDOW IS NO LONGER SUBSTITUTED HERE (2026-08-17). It was,
        # from core/forcing_availability.py — framework code that opened ELM's
        # DATM directory and parsed ELM's filenames. Two things were wrong with
        # that: it put ELM knowledge in front of the MCP boundary, and it pasted
        # ELM's answer into EVERY request. A PFLOTRAN study of 2024 was refused
        # because ELM's forcing stops in 2023, which is not a fact about that
        # study at all.
        #
        # The window now comes from the model's own report, under
        # `constraints.forcing.years`, read by the same describe_server call
        # reception already makes to choose a model. STEP 5 says so.
        self.system = load_prompt(
            "reception_agentic",
            sweep_menu=render_sweep_menu(mcp_clients))
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
            local_tools=model_tools(self._clients),
        )

    @property
    def exposed_tools(self):
        return [t["function"]["name"] for t in self.loop.tools]

    def process(self, user_request: str, context: Dict[str, Any] = None,
                run_dir=None) -> Dict[str, Any]:
        """LLM loop, then the deterministic gather. Returns the reception package.

        Two phases, and the split is the point:

          LLM     what is being asked, WHERE (domain) and WHEN (period).
                  Judgement, and the only part that can need a question.
          CODE    the DEM grid, the water table, and the observation sets for
                  that period. Not decisions, so not the model's.

        run_dir: where the modelled water table's GeoTIFF is written. It is the
        one thing reception produces that is not JSON, so it needs a directory
        rather than a return value. Optional — without it that single fetch is
        skipped and everything else still runs, which is what lets a caller with
        no run directory (a dry check, a test) use this unchanged.

        Returns {route, brief, observations, grid, provenance, trace, rounds,
        raw} — `route` carries the framework's dispatch (design / clarify /
        analyze_existing) so the brief stays science.
        """
        msg = f"User request: {user_request}\n\n"
        if context:
            msg += ("CONVERSATION CONTEXT (a prior experiment this session):\n"
                    + json.dumps(context, indent=2) + "\n\n")
        msg += ("Resolve the domain and the period, then emit ONLY your final "
                "JSON. Do NOT attempt to fetch observations — they are "
                "collected for you once the period is fixed.")
        out = self.loop.run(self.system, msg)
        brief = self._parse(out["content"])

        pkg = {"route": self._route(brief), "brief": brief,
               "observations": {}, "grid": {}, "precipitation": {},
               "subsurface": {}, "provenance": [],
               "trace": out["trace"], "rounds": out["rounds"],
               "raw": out["content"]}
        if pkg["route"]["action"] != "design":
            return pkg                       # nothing to gather for yet

        prov: list = []
        dom = (brief.get("domain") or {})
        bbox = dom.get("bbox") or {}
        period = ((brief.get("run_settings") or {}).get("resolved_period") or {})
        y0, y1 = period.get("yr_start"), period.get("yr_end")

        # ── A CONCEPTUAL REQUEST TOUCHES THE MODEL SERVER AND NOTHING ELSE ──
        # This used to be true only by luck. The fetch below was gated on
        # `if bbox:`, so a conceptual study avoided the data servers ONLY
        # because the model happened to leave `domain` empty — and the prompt
        # asking it to was the whole enforcement.
        #
        # "How does soil texture split rain in the Cascades?" breaks that. A
        # model can reasonably read it as conceptual AND resolve a domain,
        # because the user named a region. One bbox and the entire gather
        # fires: the DEM grid, the gauges, the wells, and two calls to a
        # university-run server that this project is asked to use sparingly.
        #
        # A sweep still needs coordinates — the domain file and the warm start
        # cannot do without a point — but those live in held_fixed.lat/lon.
        # A BASIN, with a bounding box and a boundary, is a different thing and
        # a conceptual brief has no use for one.
        if _is_conceptual(brief):
            skipped = _drop_domain(brief)
            # RECORDED, NOT MERELY ABSENT. A conceptual run ends with
            # observations == {}; so does a site run where every fetch failed.
            # Same bytes, opposite meanings. Without this line the Analyzer
            # cannot tell "there was nothing to compare against, by design"
            # from "the servers were down", and neither can a reader.
            prov.append({
                "tool": None,
                "args": {"design_archetype": "conceptual"},
                "fetched_at": None,
                "ok": True,
                "error": None,
                "skipped": ("no observations, no DEM grid and no water table "
                            "were fetched: a controlled sweep has no basin to "
                            "fetch them for. This is a decision, not a "
                            "failure."),
                "domain_dropped": skipped or None,
            })
            pkg["provenance"] = prov
            return pkg

        if bbox:
            pkg["grid"] = gather.gather_grid(
                self._clients, bbox, huc=str(dom.get("huc") or ""),
                boundary=dom.get("boundary"), provenance=prov)
            # heterogeneity is DERIVED from the grid, not written by the model:
            # the planner stratifies on it, so it must be the same numbers the
            # sampler will see.
            # setdefault IS NOT ENOUGH: it only fills a MISSING key, and the
            # schema tells the model to emit `heterogeneity: null`. A site
            # brief that resolved a bbox and still wrote null — which happens
            # when the request names bare coordinates rather than a basin —
            # crashed here with "'NoneType' does not support item assignment".
            if not isinstance(brief.get("heterogeneity"), dict):
                brief["heterogeneity"] = {}
            brief["heterogeneity"]["relief_m"] = pkg["grid"].get("relief_m")
            brief["heterogeneity"]["elevation_min_m"] = pkg["grid"].get("elevation_min_m")
            brief["heterogeneity"]["elevation_max_m"] = pkg["grid"].get("elevation_max_m")
            brief["heterogeneity"]["n_grid_points"] = pkg["grid"].get("n_in_basin")

            if isinstance(y0, int) and isinstance(y1, int):
                # RAIN AT THE SAME POINTS, over the resolved period. Soil says
                # what the ground is; this says what falls on it. Both are
                # keyed by the same "lat,lon" string, so a column reads them
                # with one lookup. Needs the period, which is why it sits here
                # and not beside soil. Same reasoning on placement as soil: a
                # sibling of `brief`, never inside it.
                pkg["precipitation"] = gather.gather_precipitation(
                    self._clients, pkg["grid"], y0, y1, provenance=prov)

                bs = (f'{bbox["min_lon"]},{bbox["min_lat"]},'
                      f'{bbox["max_lon"]},{bbox["max_lat"]}')
                # The polygon comes from the GRID, not from the brief: when the
                # brief carries no boundary, gather_grid is what fetched it from
                # the HUC. Passing it here is what tags each station in- or
                # out-of-basin; without it the observations are bbox-wide while
                # the grid is basin-clipped, and the two disagree about where
                # the study is.
                pkg["observations"] = gather.gather_observations(
                    self._clients, bs, y0, y1,
                    boundary=pkg["grid"].get("boundary"),
                    # WHERE THE MODELLED WATER TABLE IS WRITTEN. It used to be
                    # `grid_points`, sampled at the 58 basin-clipped points —
                    # but sample_columns snaps columns onto the CONUS grid and
                    # moves them, so those were never the points anyone would
                    # later ask about. It is a GeoTIFF beside reception.json
                    # now, readable at any location by static_wtd.sample().
                    run_dir=run_dir,
                    provenance=prov)
                brief["observations_summary"] = gather.summarise(pkg["observations"])

                # THE SUBSURFACE, as a FIELD beside the water table rather than
                # a value per column. Same reason: the columns do not exist yet,
                # and a field answers at any point chosen later. It needs the
                # run directory, which is why it sits here with the other file
                # products and not beside soil.
                #
                # It replaces what used to be invented. A survey profile stops
                # at about 1.5 m; everything below that was the deepest horizon
                # copied downward. This parameterises 392 m.
                pkg["subsurface"] = gather.gather_subsurface(
                    self._clients, bs, run_dir=run_dir, provenance=prov)
        pkg["provenance"] = prov
        return pkg

    @staticmethod
    def _route(brief: Dict[str, Any]) -> Dict[str, Any]:
        """Framework dispatch, kept OUT of the science brief.

        `intent` used to live inside the brief, where it was the only key the
        planner prompt never mentions — it is routing, not science.
        """
        raw = brief.get("intent", "parse_error")
        action = {"design": "design",
                  "analyze_existing": "analyze_existing",
                  "resume": "resume"}.get(raw, "clarify")
        return {"action": action,
                "questions": list(brief.get("questions") or []),
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
