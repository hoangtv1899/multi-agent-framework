#!/usr/bin/env python3
"""
A user-question battery for DESIGN_REVIEW.md.

Twenty-plus questions of the shapes real users type — every archetype,
periods stated/carried/defaulted, stations present and absent, long and
unicode and markdown-hostile wording, thin and broken records — each
paired with the reception/strategy records that question would leave
behind. For every one, the page must render without error, stay one
readable screen, carry the question verbatim, and say only what the
records say. The battery is the contract: a renderer change that breaks
a question shape fails here by name.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from agents import design_review as dr                     # noqa: E402


# ── builders: the records a question leaves behind ───────────────────
def site(request, *, period=(1979, 1979, "user"), stations=("642:WA:SNTL",),
         verdict="partial", why="no routed discharge in a 1-D column",
         domain=True, goals=None, conflicts=(), gaps=(), n=7,
         justification="relief 900 m -> 3 bands x 2 + 1 pin"):
    y0, y1, src = period
    brief = {"design_archetype": "site", "model": "elm",
             "model_rationale": "the land surface is the question",
             "run_settings": {"resolved_period":
                              {"yr_start": y0, "yr_end": y1, "source": src},
                              "conflicts": list(conflicts)},
             "known_data_gaps": list(gaps)}
    if domain:
        brief["domain"] = {"name": "Testville", "huc": "01234567",
                           "area_km2": 100.4, "source": "USGS WBD"}
    strategy = {"archetype": "site",
                "goals": goals if goals is not None
                else ["Quantify runoff against elevation."],
                "feasibility": {"verdict": verdict, "why": why},
                "sampling": {"approach": "stratified by elevation band",
                             "n_columns": n, "justification": justification},
                "validation": [{"variable": "swe",
                                "stations": list(stations),
                                "comparison": "daily"}]}
    return {"user_request": request, "brief": brief}, strategy


def sweep(request, factors, held=None, held_by=None, n=None):
    brief = {"design_archetype": "conceptual", "model": "pflotran",
             "sweep": {"factors": factors, "held_fixed": held or {},
                       "held_fixed_settled_by": held_by or {}}}
    strategy = {"archetype": "conceptual", "goals": [],
                "feasibility": {"verdict": "full", "why": "its core output"},
                "sampling": ({"n_columns": n} if n else {})}
    return {"user_request": request, "brief": brief}, strategy


def coupling(request, *, prior="pf_run_1", n_prior=3, variable="solved water "
             "table -> next initial state", strategy_block=True,
             period_src="carried"):
    brief = {"design_archetype": "coupling", "model": "elm",
             "run_settings": {"resolved_period":
                              {"yr_start": 2010, "yr_end": 2010,
                               "source": period_src}},
             "coupling": {"prior_experiment": prior,
                          "n_columns_prior": n_prior,
                          "from_model": "pflotran", "to_model": "elm",
                          "coupling_variable": variable}}
    strategy = {"archetype": "coupling", "goals": [],
                "feasibility": {"verdict": "full", "why": "carried forward"},
                "coupling": ({"prior_experiment": prior}
                             if strategy_block else None)}
    return {"user_request": request, "brief": brief}, strategy


def _noted(request, note, stations=()):
    """A site study whose swe validation carries a prose comparison note."""
    rec, s = site(request, stations=stations)
    s["validation"][0]["comparison"] = note
    return rec, s


def _with_drivers(request):
    """A site study whose planner recorded controlled-vs-carried drivers."""
    rec, s = site(request)
    s["drivers"] = {
        "controlled": ["elevation (5 bands wider than the forcing cell)"],
        "carried": ["soil — recorded per column, entangled with elevation",
                    "starting water table"]}
    return rec, s


def _long_rationale(request):
    """A model rationale past the cap, so the trim must end at a sentence."""
    rec, s = site(request)
    rec["brief"]["model_rationale"] = \
        "The model computes the flux where it stands. " * 12
    return rec, s


LONG = ("How deep does one year of rain and snowmelt actually reach in the "
        "unsaturated soils of a mountain watershed, whether that depth is "
        "different between the wet valley floor and the dry ridge tops, and "
        "does any of it arrive at the water table within the same year or is "
        "it all held in the root zone and returned to the air, considering "
        "both the coarse glacial outwash and the fine lake-bed silts that "
        "the soil survey maps show side by side across the basin?")

# ── the battery: (name, reception, strategy, must_contain, must_not) ─
CASES = [
    ("site_classic",
     *site("How does runoff vary with elevation in Testville for 1979?"),
     ["you stated it", "3 bands x 2 + 1 pin", "partial",
      "642:WA:SNTL"], []),
    ("site_no_period_stated",
     *site("How does runoff vary with elevation?",
           period=(1988, 1988, "default"),
           conflicts=["period unstated; 1988 chosen as default"]),
     ["a DEFAULT the model chose", "period unstated; 1988"], []),
    ("site_span_of_years",
     *site("Partition the water balance over 1995-2004.",
           period=(1995, 2004, "user")),
     ["1995–2004"], []),
    ("site_no_stations_anywhere",
     *site("Is there any observation to check against?", stations=()),
     ["no in-basin station"], ["station(s):"]),
    ("site_many_stations_capped",
     *site("Check snow against every pillow.",
           stations=[f"s{i}" for i in range(6)]),
     ["6 station(s): s0, s1, s2, s3"], ["s4"]),
    ("site_full_feasibility",
     *site("A fully answerable question.", verdict="full",
           why="every goal maps to a model output"),
     ["Feasibility: full", "every goal maps"], []),
    ("site_no_basin_resolved",
     *site("Runoff, somewhere unstated.", domain=False),
     ["no basin"], ["Testville"]),
    ("site_unicode_and_markdown",
     *site("Compare `QDRAI` **verbatim** > naïve — 100 µm/s, 3°C?"),
     ["`QDRAI` **verbatim** > naïve — 100 µm/s, 3°C?"], []),
    ("site_long_question_trimmed",
     *site(LONG), ["How deep does one year of rain", "…"], []),
    ("site_goals_capped_at_four",
     *site("Many goals.", goals=[f"goal number {i}" for i in range(7)]),
     ["goal number 3"], ["goal number 4"]),
    ("site_gaps_and_conflicts_capped",
     *site("Thin data.", conflicts=[f"conflict {i}" for i in range(9)],
           gaps=[f"gap {i}" for i in range(5)]),
     ["conflict 4", "data gap: gap 2"], ["conflict 5", "gap 3"]),
    # An empty station list has two meanings, and the page must show the
    # planner's own sentence saying which one — never the false blanket
    # "no in-basin station" when a gauge exists but cannot be pinned.
    ("site_unpinnable_quotes_the_note",
     *_noted("Check streamflow anyway.",
             "basin-aggregate — the gauge integrates its upstream area; "
             "gauge USGS-1 is compared without co-location"),
     ["basin-aggregate — the gauge integrates", "USGS-1"],
     ["no in-basin station"]),
    ("site_pinned_note_travels",
     *_noted("Check snow.",
             "co-located — four pillows; Corral dropped as in_basin:false",
             stations=("642:WA:SNTL",)),
     ["642:WA:SNTL", "dropped as in_basin:false"], []),
    ("site_one_word_cadence_stays_out",
     *site("Plain check."), ["642:WA:SNTL"], ["— daily"]),
    ("site_drivers_controlled_vs_carried",
     *_with_drivers("What drives what?"),
     ["separated on purpose", "elevation (5 bands",
      "varies underneath, recorded but not separated",
      "starting water table"], []),
    ("site_rationale_ends_at_a_sentence",
     *_long_rationale("Why this model, at length?"),
     ["Why this model: The model computes the flux where it stands."],
     ["…"]),
    ("sweep_classic",
     *sweep("In one loam column, does water-table depth change how far a "
            "year of rain gets?",
            [{"name": "water_table_m", "levels": [2, 5, 10, 20],
              "settled_by": "user"}],
            held={"soil": "loam"}, held_by={"soil": "user"}, n=4),
     ["water_table_m", "[2, 5, 10, 20]", "loam", "no basin",
      "4 column(s), one per level"], []),
    ("sweep_two_factors",
     *sweep("Cross soil texture with rain amount.",
            [{"name": "soil", "levels": ["clay", "sand"]},
             {"name": "rain_mm_yr", "levels": [250, 500]}], n=4),
     ["**soil**", "**rain_mm_yr**", "[250, 500]"], []),
    ("sweep_nothing_held_fixed",
     *sweep("Vary one thing, hold nothing.",
            [{"name": "recharge_mm_yr", "levels": [10, 100]}]),
     ["recharge_mm_yr"], ["hold **"]),
    ("coupling_classic",
     *coupling("Re-run ELM from the prior PFLOTRAN run pf_run_1."),
     ["pf_run_1", "3 of them", "pflotran → elm",
      "carried from the prior run", "solved water table"], []),
    ("coupling_without_prior_count",
     *coupling("Couple from the prior run.", n_prior=None),
     ["reused verbatim"], ["of them"]),
    ("coupling_planner_wrote_null",
     *coupling("Drive it from the prior run.", strategy_block=False),
     ["pf_run_1"], []),
    ("coupling_variable_prose_trimmed",
     *coupling("Couple, verbosely.", variable="x" * 400),
     ["…"], ["x" * 240]),
]

DEGRADED = [
    ("empty_strategy",
     {"user_request": "A question with no plan yet.",
      "brief": {"design_archetype": "site"}}, {},
     ["records no design lines", "Feasibility: unstated"]),
    ("flat_brief_no_wrapper",
     {"user_request": "A flat package.", "design_archetype": "site",
      "domain": {"name": "Flatland"}}, {"archetype": "site"},
     ["Flatland"]),
    ("request_only_in_brief",
     {"brief": {"user_request": "The request hid in the brief.",
                "design_archetype": "site"}}, {},
     ["The request hid in the brief."]),
    ("unknown_period_source",
     *site("Who chose these years?", period=(2001, 2001, ""))[:2],
     ["source unrecorded"]),
]


# ── the battery runs ─────────────────────────────────────────────────
@pytest.mark.parametrize("name,reception,strategy,must,must_not",
                         CASES, ids=[c[0] for c in CASES])
def test_the_question_shapes_render_true(name, reception, strategy,
                                         must, must_not):
    page = dr.render(reception, strategy, name)
    for needle in must:
        assert needle in page, f"{name}: missing {needle!r}"
    for needle in must_not:
        assert needle not in page, f"{name}: must not contain {needle!r}"


@pytest.mark.parametrize("name,reception,strategy,must",
                         DEGRADED, ids=[d[0] for d in DEGRADED])
def test_degraded_records_render_honestly(name, reception, strategy, must):
    page = dr.render(reception, strategy, name)
    for needle in must:
        assert needle in page, f"{name}: missing {needle!r}"


# ── invariants that hold for EVERY question ──────────────────────────
EVERY = [(c[0], c[1], c[2]) for c in CASES] + \
        [(d[0], d[1], d[2]) for d in DEGRADED]


@pytest.mark.parametrize("name,reception,strategy",
                         EVERY, ids=[e[0] for e in EVERY])
def test_every_page_holds_the_invariants(name, reception, strategy):
    page = dr.render(reception, strategy, name)
    lines = page.splitlines()
    assert len(lines) <= 60, f"{name}: {len(lines)} lines is not one screen"
    assert page.count("**Decision:**") == 1
    assert "pending" in page
    assert "# Design review" in lines[0]
    assert "the authority" in page          # the records stay authoritative
    # the verdict word on the page IS the record's, never reworded
    verdict = ((strategy.get("feasibility") or {}).get("verdict"))
    if verdict:
        assert f"Feasibility: {verdict}." in page
    # the years on the page are the record's
    rs = ((reception.get("brief") or reception)
          .get("run_settings") or {}).get("resolved_period") or {}
    if rs.get("yr_start"):
        assert str(rs["yr_start"]) in page


def test_rendering_is_deterministic_apart_from_the_clock():
    reception, strategy = site("Same records, same page.")
    strip = lambda p: [l for l in p.splitlines() if "Rendered from" not in l]
    assert strip(dr.render(reception, strategy, "x")) == \
           strip(dr.render(reception, strategy, "x"))


def test_recording_a_decision_twice_does_not_stack(tmp_path):
    reception, strategy = site("Decide once.")
    (tmp_path / "reception.json").write_text(json.dumps(reception))
    (tmp_path / "strategy.json").write_text(json.dumps(strategy))
    p = dr.write_review(tmp_path)
    dr.record_decision(tmp_path, "accepted at the terminal")
    once = p.read_text()
    dr.record_decision(tmp_path, "accepted again")
    assert p.read_text() == once           # no pending line left to replace
    assert once.count("**Decision:**") == 1


def test_every_run_on_this_disk_still_renders():
    """Safety net over the real records, wherever this repo has run."""
    outs = ROOT / "workflow_outputs"
    dirs = [d for d in outs.glob("*") if (d / "strategy.json").is_file()
            and (d / "reception.json").is_file()] if outs.is_dir() else []
    if not dirs:
        pytest.skip("no run directories on this machine")
    for d in dirs:
        page = dr.render(json.loads((d / "reception.json").read_text()),
                         json.loads((d / "strategy.json").read_text()),
                         d.name)
        assert len(page.splitlines()) <= 60, d.name
