#!/usr/bin/env python3
"""
Controlled sweeps — what ELM can vary, whether a design will work, and the columns
mcp/elm-mcp/src/conceptual.py

    in   a design: which factors, at which levels, with what held fixed
    out  a declaration, a verdict on the design, and the columns to run

A CONCEPTUAL STUDY IS A COLUMN LIST WHERE SOME FIELDS VARY AND THE REST DO NOT.
There is no basin, no observation, and no spatial sampling — there is a question
about a mechanism, and a set of columns identical in every respect but one.

WHY THIS IS IN THE SERVER AND NOT THE FRAMEWORK. Every fact below is a fact
about ELM: that soil arrives through FSURDAT, that the forcing year is a runtime
key, that the soil column is a fixed 15-layer grid whatever profile you hand it.
The framework holds the arithmetic of a sweep — how many columns a factorial
has, what it pins — because a PFLOTRAN sweep needs the identical arithmetic and
an entirely different list. It does not hold this.

THREE JOBS, ONE VALIDATION. `check()` runs before compute, while a person can
still be asked a question; `build_columns()` runs after the design is settled.
They share `_validate()`, because a design that passes the check and then fails
the build is the worst of both.

WHAT THIS DOES NOT CHECK, and says so rather than staying quiet:

    forcing years   core/forcing_availability.py reads the DATM directory and
                    Reception states the window BEFORE a design exists, so the
                    years in a design have already been checked against the
                    filesystem. Re-checking here would mean either a backwards
                    import or a second copy of the rule.
    the science     whether soil texture is the right question, or whether a
                    1-D column can answer it, is not a fault in the design.
                    That is a caveat, and caveats are assigned elsewhere.
"""
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from elm_wrapper import RUNTIME_KEYS
# The fills a weather level may name, and the variables they may act on. TAKEN
# FROM THE WRITER, never restated: a list here that drifted from forcing.py
# would accept a design the writer rejects at case-build time, which is the
# latest possible moment to find out. forcing.py imports netCDF4 lazily, so
# this costs nothing at import.
from forcing import FILLS as _FILLS, VARIABLES as _WEATHER_VARIABLES

# ─────────────────────────────────────────────────────────────────────
# WHAT THE DECLARATION RESTS ON
# ─────────────────────────────────────────────────────────────────────
# Each factor names the runtime keys it needs. If one is removed from
# elm_wrapper, the declaration stops rather than offering a factor the wrapper
# will refuse — which is the same trade the wrapper itself makes when it raises
# on an unknown key instead of running with the default in its place.
#
# The alternative is a hand-written list that agrees with RUNTIME_KEYS on the
# day it is typed and never again.
_NEEDS: Dict[str, Tuple[str, ...]] = {
    "soil_texture":  ("FSURDAT",),
    "forcing_year":  ("DATM_CLMNCEP_YR_START", "DATM_CLMNCEP_YR_END",
                      "RUN_STARTDATE", "STOP_N"),
    "forcing_site":  ("FSURDAT", "LND_DOMAIN_FILE", "ATM_DOMAIN_FILE"),
    # No "initial_state" entry. It was a factor — warm against cold — and was
    # removed 2026-08-15 when conceptual runs went cold unconditionally. FINIDAT
    # is still a runtime key the site path uses; what is gone is the ability to
    # SWEEP it, because half its levels needed the CONUS restart.
    # The stream file names the domain and the run's years; both come off the
    # runtime config, so losing either key would write a stream pointing at
    # nothing. There is no key for the forcing itself — CIME reads
    # user_datm.streams.txt.CLMMOSARTTEST from the case directory on its own.
    "prescribed_weather": ("LND_DOMAIN_FILE", "LND_DOMAIN_PATH",
                           "DATM_CLMNCEP_YR_START", "DATM_CLMNCEP_YR_END"),
}

_missing = {f: sorted(set(k) - RUNTIME_KEYS) for f, k in _NEEDS.items()}
_missing = {f: k for f, k in _missing.items() if k}
if _missing:                                                # pragma: no cover
    raise ImportError(
        f"conceptual.py declares factors whose runtime keys elm_wrapper no "
        f"longer has: {_missing}. Offering them would design studies the "
        f"wrapper refuses to build. Fix the declaration or restore the keys.")


# ─────────────────────────────────────────────────────────────────────
# THE FIXED SOIL GRID — the thing users are most often wrong about
# ─────────────────────────────────────────────────────────────────────
# ELM's soil column is 15 layers, 1.75 cm at the top to 13.85 m at the bottom,
# 42.10 m in total, with the first 10 layers making the 3.80 m that does the
# hydrology. A STUDY DOES NOT CONFIGURE THIS. A prescribed profile says what
# material fills the depths it covers; the substrate setting fills the rest.
#
# So "a 2 m soil" gives 2 m of the stated material on top of 40 m more of it
# (SUBSTRATE=extrapolate), not 2 m of soil on bedrock. Nobody is wrong to ask —
# the request is ambiguous, not invalid — but the answer is a conversation
# rather than a validation error, so this is here to be QUOTED, not enforced.
SOIL_GRID_LAYERS   = 15
SOIL_GRID_TOTAL_M  = 42.10
ACTIVE_LAYERS      = 10
ACTIVE_DEPTH_M     = 3.8021

# NLDAS-2, as inputs.py computes it. Two levels landing in one cell share their
# weather exactly, which makes a forcing sweep flat by construction.
NLDAS_DEG = 0.125
NLDAS_LAT0, NLDAS_LON0 = 25.0, -125.0

# Where the soil physics stops interpolating and starts extrapolating. ELM's
# hydraulic properties come from pedotransfer relations fitted to measured
# soils; outside that envelope the numbers are produced by extending the fit,
# not by measurement. A DEFENSIBLE EXPERIMENT, not an error — so this bound
# raises a flag and never a refusal.
FITTED_CLAY_MAX_PCT = 60.0
FITTED_SAND_MAX_PCT = 95.0

# What a sweep needs to be a sweep. One level is a single run wearing the word.
MIN_LEVELS = 2

# ── WHAT A WRITTEN WEATHER VALUE MAY BE ──────────────────────────────
# The same two-tier judgement the soil gets, for the same reason: IMPOSSIBLE is
# refused, MERELY EXTREME is built and flagged. Negative rain is not an
# experiment; ten times the wettest place on Earth is a defensible one as long
# as nobody mistakes it for a climate.
#
# ADDED AFTER A REAL RUN GOT THROUGH. Reception settled on a constant
# PRECTmms of 0.1 for a texture sweep. The units are mm per SECOND, so that is
# 8,640 mm/day — 3.15 MILLION mm/year, about 2,300 times Naches — and every
# check passed. The variable name was valid, the fill was valid, seven columns
# materialised. Nothing in the framework knew what the number MEANT.
#
# {variable: (hard_min, hard_max, plausible_min, plausible_max)}. Outside the
# hard pair the design is refused; outside the plausible pair it is flagged.
# None means unbounded on that side.
WEATHER_BOUNDS: Dict[str, Tuple] = {
    # mm/s. 3.76e-4 is Mawsynram, the wettest inhabited place (11,872 mm/yr);
    # 1e-3 is ~31,500 mm/yr, past anywhere real.
    "PRECTmms": (0.0,    None,   0.0,     1.0e-3),
    # K. Vostok read 184 K; 331 K is the highest reliable surface air reading.
    "TBOT":     (0.0,    None,   180.0,   340.0),
    "WIND":     (0.0,    None,   0.0,     60.0),     # m/s, 60 is a strong hurricane
    "QBOT":     (0.0,    None,   0.0,     0.05),     # kg/kg, saturation is ~0.04 at 310 K
    "FLDS":     (0.0,    None,   50.0,    600.0),    # W/m2 downwelling longwave
    "FSDS":     (0.0,    None,   0.0,     1400.0),   # W/m2, the solar constant is 1361
    "PSRF":     (0.0,    None,   50000.0, 110000.0), # Pa, 50 kPa is ~5,500 m
}

# A multiplier of 0 deletes a variable; a negative one inverts it. Neither is a
# scaling experiment. Ten-fold is the edge of arguable.
SCALE_MIN, SCALE_MAX = 0.0, 10.0


def nldas_cell(lat, lon) -> Optional[Tuple[int, int]]:
    """(row, col) of the NLDAS-2 cell holding this point, or None.

    Deliberately the same arithmetic as inputs._nldas_cell. The duplicate is
    four lines and the alternative is importing the input builder — with its
    CONUS readers — to answer a question about two floats.
    """
    try:
        return (math.floor((float(lat) - NLDAS_LAT0) / NLDAS_DEG),
                math.floor((float(lon) - NLDAS_LON0) / NLDAS_DEG))
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────
# THE PROFILE A TEXTURE LEVEL BECOMES
# ─────────────────────────────────────────────────────────────────────
def uniform_profile(clay_pct: float, sand_pct: Optional[float] = None,
                    organic_pct: float = 2.0, bulk_density: float = 1.4,
                    depth_cm: int = 200) -> Dict[str, Any]:
    """One synthetic layer of stated texture — the shape the builder consumes.

    UNIFORM ON PURPOSE, and this is the method rather than a shortcut. Organic
    matter and bulk density are pinned so that texture is the only thing moving
    between columns; a real soil varies all three with depth, and a sweep over
    real profiles would answer a question about profiles rather than texture.

    That is also exactly why the finished study needs a caveat saying these are
    not real soils. Isolating a variable and being representative are opposite
    goals, and a study can only have one of them.

    `sand_pct` defaults to the remainder after clay on a sand-rich baseline, so
    a single clay number produces a sensible sand/silt split without the caller
    inventing one.
    """
    clay = float(clay_pct)
    sand = float(sand_pct) if sand_pct is not None else max(0.0, 87.0 - clay)
    silt = round(max(0.0, 100.0 - clay - sand), 1)
    # NUMBERS, MATCHING THE DONOR PROFILE. make_finidat_subset.donor_soil_profile
    # emits round(depth, 1) and consumers do arithmetic on it —
    # sampling_design.py computes layer thickness as depth_bot - depth_top. The
    # first version of this copied make_soil_sweep.py, which writes those two as
    # STRINGS: harmless in a standalone script that only ever wrote JSON, and a
    # TypeError the moment a prescribed profile reached the design figure. A
    # synthetic profile has to be the same SHAPE as a real one, not just carry
    # the same field names.
    return {
        "source": "synthetic uniform (conceptual sweep)",
        "num_layers": 1,
        "layers": [{
            "component":          "synthetic uniform",
            "texture_class":      f"clay{clay:.0f}",
            "depth_top_cm":       0.0,
            "depth_bot_cm":       round(float(depth_cm), 1),
            "sand_pct":           round(sand, 1),
            "silt_pct":           silt,
            "clay_pct":           round(clay, 1),
            "organic_matter_pct": float(organic_pct),
            "bulk_density_gcc":   float(bulk_density),
        }],
    }


# ─────────────────────────────────────────────────────────────────────
# THE DECLARATION
# ─────────────────────────────────────────────────────────────────────
# `holds_fixed` describes what is pinned when this factor runs ALONE. A caller
# combining factors subtracts whatever another chosen factor deliberately
# varies; nothing here needs to know about combinations.
_DECLARATION: List[Dict[str, Any]] = [
    {
        "name":        "soil_texture",
        "applies_to":  "soil_profile",
        "label":       "Soil texture — a sand-to-clay gradient at one place",
        "levels_are":  "clay percent; sand co-varies, silt is the remainder",
        "reaches_elm": "a synthetic surface dataset, handed over as FSURDAT",
        # The gradient the existing soil sweep runs. Seven because five is too
        # coarse to show a curve and nine buys smoothness nobody reads.
        "suggested_levels": [5, 10, 18, 27, 35, 45, 55],
        "holds_fixed": ["lat", "lon", "years", "organic_matter_pct",
                        "bulk_density_gcc", "depth_cm", "pft"],
        "caveats":     ["conceptual_uniform_soil"],
        "note":        "Organic matter and bulk density are pinned so texture "
                       "is the only thing that moves. That is the method "
                       "working, and also why these are not real soils.",
    },
    {
        "name":        "forcing_year",
        "applies_to":  "years",
        "label":       "Which year — the same column under different weather",
        "levels_are":  "a calendar year, or a [start, end] pair",
        "reaches_elm": "DATM_CLMNCEP_YR_START/END, RUN_STARTDATE and STOP_N",
        # No default: which years are interesting depends entirely on the
        # question, and a suggested pair would be a guess wearing authority.
        "suggested_levels": [],
        "holds_fixed": ["lat", "lon", "soil_profile", "pft"],
        "caveats":     ["conceptual_one_place"],
        "note":        "Pick years that differ in precipitation, or the sweep "
                       "varies nothing that matters. Reception has already "
                       "checked which years have forcing on disk.",
    },
    {
        "name":        "forcing_site",
        "applies_to":  "latlon",
        "label":       "Which climate — the same soil in wetter and drier places",
        "levels_are":  "a [lat, lon] pair, one NLDAS cell each",
        "reaches_elm": "the column's coordinates, which select the forcing cell "
                       "and the CONUS grid cell whose saved state supplies "
                       "(donates) the warm start",
        "suggested_levels": [],
        "holds_fixed": ["soil_profile", "years", "pft"],
        "caveats":     ["conceptual_moved_column"],
        "note":        "Columns still carry coordinates — no sampling does not "
                       "mean no location. Levels must fall in different NLDAS "
                       "cells (0.125 deg) or they share weather exactly.",
    },
    # REMOVED 2026-08-15: "initial_state", warm against cold. Conceptual runs
    # now start cold unconditionally, so one of its two levels no longer
    # exists. Sweeping it would have meant reaching for the CONUS restart in a
    # study whose whole point is to depend on no real place. See cannot_vary.
    {
        "name":        "prescribed_weather",
        "applies_to":  "weather",
        "label":       "Prescribed weather — written rather than borrowed",
        "levels_are":  "a fill: 'copy', or {fill: uniform|scale|offset, "
                       "values: {VARIABLE: number}}",
        "reaches_elm": "single-gridcell DATM files and a user stream file in "
                       "the case directory, which CIME reads instead of the "
                       "generated one",
        # A wet/dry pair around the real year. Deliberately NOT a uniform
        # default: see the note. The real year is included as `copy` so the
        # sweep has an unmodified reference in it rather than only extremes.
        "suggested_levels": [
            {"fill": "scale", "values": {"PRECTmms": 0.5}},
            "copy",
            {"fill": "scale", "values": {"PRECTmms": 2.0}},
        ],
        "holds_fixed": ["lat", "lon", "soil_profile", "pft"],
        "caveats":     ["conceptual_prescribed_weather"],
        "note":        "As a FACTOR this varies the weather between columns. "
                       "To give every column the SAME written weather — which "
                       "is what removes the borrowed climate from a soil study "
                       "— put one spec in held_fixed.weather instead and sweep "
                       "something else. UNIFORM REMOVES RAINFALL INTENSITY, "
                       "which is what drives the runoff/drainage split: a "
                       "constant drizzle soaks into almost any soil, so a "
                       "texture sweep under it reports that texture barely "
                       "matters. scale and offset keep the storm structure and "
                       "are the better instrument for that question. All four "
                       "are offered; the choice is the user's.",
    },
]

_BY_NAME = {d["name"]: d for d in _DECLARATION}

# _LEVEL_KINDS = {"warm", "cold"} went with the initial_state factor.


def declare() -> Dict[str, Any]:
    """What ELM can be swept over. Reception offers this; the planner designs to it.

    Also carries the fixed soil grid, because the commonest misunderstanding in
    a conceptual soil study is that a study chooses the column depth. Stating
    it beside the menu is cheaper than discovering it in a figure.
    """
    return {
        "server": "elm",
        "factors": _DECLARATION,
        "soil_grid": {
            "layers": SOIL_GRID_LAYERS,
            "total_depth_m": SOIL_GRID_TOTAL_M,
            "active_layers": ACTIVE_LAYERS,
            "active_depth_m": ACTIVE_DEPTH_M,
            "note": (f"Fixed. {SOIL_GRID_LAYERS} layers to "
                     f"{SOIL_GRID_TOTAL_M} m, of which the first "
                     f"{ACTIVE_LAYERS} ({ACTIVE_DEPTH_M} m) do the hydrology. "
                     f"A prescribed profile says what fills the depths it "
                     f"covers; the substrate setting fills the rest. Asking "
                     f"for 'a 2 m soil' gives 2 m of that material on top of "
                     f"40 m more of it, not soil on bedrock."),
        },
        "cannot_vary": [
            {"what": "PFT parameters (rooting depth, Vcmax)",
             "needs": "a paramfile in user_nl_elm; RUNTIME_KEYS carries none"},
            # "a graded initial water table" LEFT THIS LIST 2026-08-19:
            # set_initial_water_table writes ZWT/WA into a finidat subset
            # (src/set_water_table.py). It serves the COUPLING archetype —
            # PFLOTRAN's solved water table back into ELM's start — and is
            # not offered as a sweep factor here until someone asks for one.
            {"what": "the starting state (warm against cold)",
             "needs": "nothing — it is a DECISION, not a missing capability. "
                      "Conceptual runs start cold so that no real gridcell's "
                      "water content enters a study designed to depend on no "
                      "real place. Say this plainly if a user asks for a warm "
                      "start: they can have one in a site study."},
        ],
        # STATED WHERE THE MENU IS READ, so it reaches the user in the
        # conversation rather than in a caveat after the design is settled.
        "always_true": [
            {"what": "every column starts cold",
             "detail": "from ELM's own defaults, not from a real gridcell. All "
                       "columns start IDENTICALLY, which is what makes the "
                       "comparison controlled. The cost is that soil moisture "
                       "takes years to equilibrate from cold, so early time "
                       "reflects the initialisation rather than the soil — "
                       "there is no spin-up step, and that is accepted rather "
                       "than hidden."},
        ],
        # SETTINGS THAT ARE NOT FACTORS. A factor varies BETWEEN columns; these
        # apply to every column at once. Listed separately because offering
        # `weather` only as something to sweep hides the use that matters most
        # — writing ONE weather for the whole study, which is what stops a soil
        # result from being bounded by a borrowed climate.
        "held_fixed_options": [
            {"name": "weather",
             "takes": "one fill spec, the same shape as a prescribed_weather "
                      "level",
             "does": "every column runs on written weather instead of the "
                     "NLDAS cell it sits in",
             "unlocks": "the location stops being a scientific choice. With "
                        "the weather written rather than borrowed, no cell's "
                        "climate bounds the result and conceptual_one_climate "
                        "no longer applies — the columns still carry "
                        "coordinates for the domain, the warm start and the "
                        "nearest-neighbour match, but they become bookkeeping "
                        "rather than part of the finding. This is the answer "
                        "to 'a conceptual study should have no place'.",
             "note": "Cannot be combined with a prescribed_weather factor — "
                     "one says every column gets the same weather and the "
                     "other says they differ."},
        ],
        "weather_fills": sorted(_FILLS),
        "weather_variables": dict(_WEATHER_VARIABLES),
        "not_yet_runnable": [],
        "min_levels": MIN_LEVELS,
    }


# ─────────────────────────────────────────────────────────────────────
# CHECKING A DESIGN
# ─────────────────────────────────────────────────────────────────────
def _as_rate(var: str, v: float) -> str:
    """A weather value in units someone can judge.

    0.1 mm/s is not obviously wrong to anybody; 8,640 mm/day is obviously wrong
    to everybody. The refusal is only useful if it says the second one.
    """
    if var == "PRECTmms":
        return f"{v:g} mm/s = {v * 86400:,.0f} mm/day = {v * 86400 * 365:,.0f} mm/year"
    return f"{v:g} {_WEATHER_VARIABLES.get(var, '')}".strip()


def _check_weather_values(where: str, resolved: Dict[str, Any],
                          wont: List[Dict], odd: List[Dict]) -> None:
    """Is each number physically possible, and is it anywhere near real?

    Only `uniform` values are judged against the bounds — they ARE the weather.
    A `scale` factor is judged as a multiplier, and `offset` is left alone: a
    +2 K shift is the point of the experiment and the shifted series is still
    real weather, so bounding it would refuse the study it exists for.
    """
    kind, values = resolved["kind"], resolved["values"]

    if kind == "uniform":
        for var, v in values.items():
            lo, hi, plo, phi = WEATHER_BOUNDS.get(var, (None,) * 4)
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                wont.append({"factor": where,
                             "why": f"{var} = {_as_rate(var, v)} is not a "
                                    f"physically possible value"})
                continue
            if (plo is not None and v < plo) or (phi is not None and v > phi):
                odd.append({"factor": where, "level": {var: v},
                            "why": f"{var} = {_as_rate(var, v)} is outside "
                                   f"anything measured on Earth (plausible "
                                   f"range {plo:g} to {phi:g}). It will run. "
                                   f"Whatever it produces is a statement about "
                                   f"the model under that forcing, not about a "
                                   f"climate.",
                            "caveat": "conceptual_outside_fitted_range"})

    if kind == "scale":
        for var, k in values.items():
            if k < SCALE_MIN:
                wont.append({"factor": where,
                             "why": f"scaling {var} by {k:g} inverts it — "
                                    f"negative weather is not an experiment"})
            elif k > SCALE_MAX:
                odd.append({"factor": where, "level": {var: k},
                            "why": f"scaling {var} by {k:g} is far past any "
                                   f"climate-change scenario (over "
                                   f"{SCALE_MAX:g}x). It will run.",
                            "caveat": "conceptual_outside_fitted_range"})


def _levels_of(design: Dict[str, Any], name: str) -> List[Any]:
    for f in (design.get("factors") or []):
        if f.get("name") == name:
            return list(f.get("levels") or ())
    return []


def _validate(design: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """The one implementation. `check()` reports it; `build_columns()` obeys it.

    Returns (wont_build, unusual, facts). Kept as three lists rather than one
    with a severity field, because the caller does three different things with
    them: refuse, caveat, and consider.
    """
    wont: List[Dict[str, Any]] = []
    odd:  List[Dict[str, Any]] = []
    fact: List[Dict[str, Any]] = []

    factors = design.get("factors") or []
    if not factors:
        wont.append({"factor": None,
                     "why": "the design varies nothing — a sweep needs at "
                            "least one factor with levels"})
        return wont, odd, fact

    for f in factors:
        name = f.get("name")
        levels = list(f.get("levels") or ())
        if name not in _BY_NAME:
            wont.append({"factor": name,
                         "why": f"this server does not offer {name!r}; it "
                                f"offers {sorted(_BY_NAME)}"})
            continue
        if len(levels) < MIN_LEVELS:
            wont.append({"factor": name,
                         "why": f"{len(levels)} level(s) — a sweep needs at "
                                f"least {MIN_LEVELS}, or it is one run with a "
                                f"comparison implied and never made"})
        seen, dupes = set(), []
        for lv in levels:
            key = repr(lv)
            if key in seen:
                dupes.append(lv)
            seen.add(key)
        if dupes:
            wont.append({"factor": name,
                         "why": f"repeated level(s) {dupes} — two identical "
                                f"columns measure the model's determinism, "
                                f"not the factor"})

    # ── soil texture ────────────────────────────────────────────────
    fixed = design.get("held_fixed") or {}
    for clay in _levels_of(design, "soil_texture"):
        try:
            c = float(clay)
        except (TypeError, ValueError):
            wont.append({"factor": "soil_texture",
                         "why": f"level {clay!r} is not a number of percent"})
            continue
        if not 0.0 <= c <= 100.0:
            wont.append({"factor": "soil_texture",
                         "why": f"clay {c}% is outside 0-100"})
            continue
        prof = uniform_profile(
            c, fixed.get("sand_pct"),
            organic_pct=fixed.get("organic_matter_pct", 2.0),
            bulk_density=fixed.get("bulk_density_gcc", 1.4),
            depth_cm=fixed.get("depth_cm", 200))
        lay = prof["layers"][0]
        total = lay["clay_pct"] + lay["sand_pct"]
        if total > 100.0:
            wont.append({"factor": "soil_texture",
                         "why": f"clay {lay['clay_pct']}% with sand "
                                f"{lay['sand_pct']}% is {total}% of a soil"})
        if c > FITTED_CLAY_MAX_PCT:
            odd.append({"factor": "soil_texture", "level": c,
                        "why": f"clay {c}% is above {FITTED_CLAY_MAX_PCT}%, "
                               f"where ELM's hydraulic properties come from "
                               f"extending the pedotransfer fit rather than "
                               f"from measured soils. It will run; the result "
                               f"is a statement about the extrapolation.",
                        "caveat": "conceptual_outside_fitted_range"})
        if lay["sand_pct"] > FITTED_SAND_MAX_PCT:
            odd.append({"factor": "soil_texture", "level": c,
                        "why": f"sand {lay['sand_pct']}% is above "
                               f"{FITTED_SAND_MAX_PCT}%, the same "
                               f"extrapolation at the other end",
                        "caveat": "conceptual_outside_fitted_range"})

    depth = (design.get("held_fixed") or {}).get("depth_cm")
    if depth is not None:
        try:
            d = float(depth)
        except (TypeError, ValueError):
            d = None
        if d is not None and d <= 0:
            wont.append({"factor": "soil_texture",
                         "why": f"a profile depth of {depth} cm has no soil "
                                f"in it"})
        elif d is not None and d < 100.0 * SOIL_GRID_TOTAL_M:
            fact.append({"what": "prescribed depth is shorter than the column",
                         "detail": f"the profile covers {d} cm; ELM's grid is "
                                   f"{SOIL_GRID_TOTAL_M} m deep, so the "
                                   f"substrate setting fills the rest. This is "
                                   f"normal — it is only surprising if you "
                                   f"expected soil over bedrock."})

    # ── whose weather, and where the column sits ────────────────────
    # A SWEEP HAS NO STUDY LOCATION AND STILL CANNOT RUN WITHOUT A POINT: the
    # domain file, DATM's nearest-neighbour match and (on a site run) the warm
    # start all need one. Held here rather than in the framework's strategy
    # gate (moved 2026-08-18): the reason is ELM's — a PFLOTRAN sweep has no
    # place at all and must not be refused for lacking one — so the server
    # that needs the point is the one that asks for it.
    #
    # TWO WORDINGS, because the two cases differ. With the weather WRITTEN
    # (held_fixed.weather or a prescribed_weather factor) the coordinates are
    # bookkeeping; saying "ELM reads its weather from a grid cell" to that
    # design would be a false explanation of a correct refusal.
    varies_site = bool(_levels_of(design, "forcing_site"))
    written = (fixed.get("weather") is not None
               or bool(_levels_of(design, "prescribed_weather")))
    if not varies_site and not (fixed.get("lat") is not None
                                and fixed.get("lon") is not None):
        wont.append({"factor": "held_fixed.lat/lon",
                     "why": "the design names no coordinates — a sweep has no "
                            "study location, "
                            + ("but the domain file and the nearest-neighbour "
                               "match still need a point, so held_fixed must "
                               "say which (the weather is written, so this is "
                               "bookkeeping rather than a scientific choice)"
                               if written else
                               "but ELM still reads its weather from a grid "
                               "cell, so held_fixed must say which — or sweep "
                               "forcing_site")})

    # ── the soil nobody chose ───────────────────────────────────────
    # WHAT THE COLUMNS ACTUALLY GET when the design says nothing about soil.
    # Reported because saying nothing is what let a real run tell the user
    # "held fixed, as you asked (one loam-ish profile)" on a design whose
    # held_fixed was {lat, lon, years} and contained no soil at all. The soil
    # was invented in the sentence and never existed in the study; nothing
    # downstream could contradict it, because nothing downstream was asked.
    #
    # A weather sweep legitimately does not specify soil — it is held fixed by
    # not varying. The point is to say WHICH soil that is, so a claim about it
    # has something to be checked against.
    soil_named = bool(_levels_of(design, "soil_texture")) or any(
        k in fixed for k in ("soil_profile", "soil_texture", "clay_pct"))
    if not soil_named:
        fact.append({"what": "no soil was specified",
                     "detail": "every column gets the same soil from the "
                               "default surface dataset — a real site's "
                               "profile, identical across columns because "
                               "nothing varies it. It is held fixed, but "
                               "nobody chose it, so do not describe it to the "
                               "user as a particular texture. If the study "
                               "needs a known soil, add soil_texture with one "
                               "level or put a profile in held_fixed."})

    # ── which climate ───────────────────────────────────────────────
    sites = _levels_of(design, "forcing_site")
    cells: Dict[Tuple[int, int], List[Any]] = {}
    for lv in sites:
        try:
            lat, lon = lv[0], lv[1]
        except (TypeError, IndexError, KeyError):
            wont.append({"factor": "forcing_site",
                         "why": f"level {lv!r} is not a [lat, lon] pair"})
            continue
        cell = nldas_cell(lat, lon)
        if cell is None:
            wont.append({"factor": "forcing_site",
                         "why": f"level {lv!r} has no NLDAS cell"})
            continue
        cells.setdefault(cell, []).append(lv)
    for cell, members in cells.items():
        if len(members) > 1:
            # THE ONE THAT WOULD NOT BE NOTICED. Every column runs, the count
            # is right, and the figures come out on top of each other.
            wont.append({"factor": "forcing_site",
                         "why": f"{len(members)} levels fall in NLDAS cell "
                                f"{cell} and would share their weather "
                                f"exactly: {members}. The sweep would be flat "
                                f"by construction and nothing downstream "
                                f"would report it."})
    if cells:
        fact.append({"what": "forcing cells",
                     "detail": {str(k): v for k, v in cells.items()}})

    # ── starting state ──────────────────────────────────────────────
    # EVERY CONCEPTUAL COLUMN STARTS COLD, so there is nothing here to check —
    # only something to say. The old caveat was conceptual_donor_state: the
    # CONUS restart carried a real gridcell's water content, belonging to a
    # column of different texture. That is gone with the restart.
    #
    # What replaced it is not smaller. A cold column starts at ELM's own
    # defaults and takes YEARS to forget them, so a short run reports the
    # initialisation as much as the soil. The trade was made deliberately —
    # borrowed water that is wrong for the texture, against known water that is
    # wrong for everything and fades — and it is stated on every sweep rather
    # than only when someone asks.
    n_years = 1
    yrs = fixed.get("years")
    if isinstance(yrs, (list, tuple)) and len(yrs) >= 2:
        try:
            n_years = max(1, int(yrs[-1]) - int(yrs[0]) + 1)
        except (TypeError, ValueError):
            n_years = 1
    odd.append({"factor": "initial_state",
                "why": f"every column starts cold, from ELM's defaults rather "
                       f"than a real gridcell's state. Soil moisture takes "
                       f"years to equilibrate from cold, and this design runs "
                       f"{n_years} year(s), so early time reflects the "
                       f"initialisation rather than the soil. The columns "
                       f"still differ from each other correctly — they start "
                       f"from the SAME state, which is what a controlled "
                       f"comparison needs — but an absolute number from year "
                       f"one is not the soil's answer.",
                "caveat": "conceptual_cold_start"})

    # ── written weather ─────────────────────────────────────────────
    # VALIDATED BY THE WRITER ITSELF. forcing.spec_to_fill is what will
    # construct the fill at case-build time, so asking it here means a design
    # that passes check() cannot fail later on the same spec. A second opinion
    # written out in this file would be a second definition of "valid".
    import forcing

    weather_levels = _levels_of(design, "prescribed_weather")
    fixed_weather = fixed.get("weather")
    if weather_levels and fixed_weather is not None:
        wont.append({"factor": "prescribed_weather",
                     "why": "the design both sweeps the weather and pins it in "
                            "held_fixed.weather. One says the columns differ in "
                            "weather and the other says they do not; whichever "
                            "won silently would produce a study that is not the "
                            "one designed."})

    to_check = [("prescribed_weather", lv) for lv in weather_levels]
    if fixed_weather is not None:
        to_check.append(("held_fixed.weather", fixed_weather))

    # WHAT THE LEVELS MEAN, not how they were spelt. The generic duplicate
    # check near the top compares repr(), so `"copy"` and
    # `{"fill": "scale", "values": {"PRECTmms": 1.0}}` read as two different
    # levels — and they are the same weather, so they are two identical columns
    # measuring the model's determinism. Only the resolved form can see that.
    seen_weather: Dict[str, Any] = {}

    for where, lv in to_check:
        try:
            resolved = forcing.spec_to_fill(lv)
        except ValueError as e:
            wont.append({"factor": where, "why": str(e)})
            continue
        if resolved.get("normalised_from") and where == "held_fixed.weather":
            # Harmless as a control INSIDE a factor list; as the study's one
            # weather setting it means the design claims to write the weather
            # and writes the real cell's — a year of netCDF per column to
            # reproduce what ELM would have read anyway.
            fact.append({"what": "the weather setting is an identity",
                         "detail": f"{resolved['normalised_from']} is the real "
                                   f"cell unchanged, recorded as 'copy'. Every "
                                   f"column still gets identical weather "
                                   f"because they share a location, and the "
                                   f"borrowed-climate caveat still applies — "
                                   f"nothing was written that ELM would not "
                                   f"have read."})

        sig = f"{resolved['kind']}:{sorted(resolved['values'].items())}"
        if where == "prescribed_weather":
            if sig in seen_weather:
                wont.append({"factor": where,
                             "why": f"levels {seen_weather[sig]!r} and {lv!r} "
                                    f"are the same weather written two ways "
                                    f"({resolved['kind']}) — two identical "
                                    f"columns measure the model's determinism, "
                                    f"not the factor"})
            seen_weather[sig] = lv

        _check_weather_values(where, resolved, wont, odd)

        if resolved["kind"] == "uniform" and "PRECTmms" in resolved["values"]:
            # A MEASUREMENT ABOUT THE DESIGN, not a verdict on the study.
            # This used to say "a soil study under it will understate how much
            # texture matters" — true for a runoff/drainage question and WRONG
            # for an equilibrium one, where steady input is the method rather
            # than a flaw. The server cannot tell those apart because it does
            # not know the question. It states what the forcing lacks;
            # Reception, which does know the question, says whether that
            # matters, and the framework assigns the severity.
            odd.append({"factor": where, "level": lv,
                        "why": "constant precipitation has no storm structure: "
                               "rainfall intensity, wet spells and dry spells "
                               "are all removed, and the same annual total "
                               "arrives as a steady drip. Whether that helps "
                               "or ruins the study depends on the question — "
                               "it is the method for an equilibrium question "
                               "and removes the mechanism for a "
                               "runoff/drainage one. 'scale' keeps the "
                               "structure and changes the amount.",
                        "caveat": "conceptual_uniform_precip"})
        if resolved["kind"] == "uniform" and "FSDS" in resolved["values"] \
                and not resolved["flat_solar"]:
            odd.append({"factor": where, "level": lv,
                        "why": "DATM redistributes shortwave by the cosine of "
                               "the solar zenith angle, so a constant FSDS in "
                               "the file STILL produces a diurnal cycle. That "
                               "is correct physics and probably not what "
                               "'uniform light' meant — set flat_solar: true "
                               "on this level to pin it.",
                        "caveat": "conceptual_prescribed_weather"})

    if weather_levels or fixed_weather is not None:
        fact.append({"what": "weather is written, not borrowed",
                     "detail": "single-gridcell DATM files are written at each "
                               "column's own coordinates and a user stream "
                               "file in the case directory points ELM at them. "
                               "The columns still carry coordinates — the "
                               "domain, the warm start and the "
                               "nearest-neighbour match all need them — but "
                               "the climate is no longer the cell's."})
        fact.append({"what": "no ELM case has yet run on written weather",
                     "detail": "the writer round-trips a real NLDAS cell with "
                               "zero differences across all seven variables, "
                               "so the files are right. What is unproven is "
                               "that ELM reads them, which would fail at model "
                               "init rather than quietly."})

    # ── years, and why they are not checked here ────────────────────
    if _levels_of(design, "forcing_year"):
        fact.append({"what": "forcing years not re-checked here",
                     "detail": "core/forcing_availability.py reads the DATM "
                               "directory and Reception states the window "
                               "before a design exists, so these years have "
                               "already been checked against the filesystem."})

    return wont, odd, fact


def check(design: Dict[str, Any]) -> Dict[str, Any]:
    """Will this design do what it looks like it does?

    THREE LISTS, NEVER A SEVERITY. This server reports what is true about a
    design; how much each thing matters is assigned where every other caveat's
    severity is assigned. Step 1 already learned this the hard way — it used to
    add eight caveats and had them removed, because a server that measures and
    also grades is two authorities on what a study may conclude.
    """
    wont, odd, fact = _validate(design)
    return {
        "buildable": not wont,
        "wont_build": wont,
        "unusual": odd,
        "facts": fact,
        "n_columns": _n_columns(design),
    }


def _n_columns(design: Dict[str, Any]) -> int:
    total = 1
    factors = design.get("factors") or []
    if not factors:
        return 0
    for f in factors:
        total *= max(len(f.get("levels") or ()), 0)
    return total


# ─────────────────────────────────────────────────────────────────────
# BUILDING THE COLUMNS
# ─────────────────────────────────────────────────────────────────────
def build_columns(design: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The design becomes the same column list a site run produces.

    THE MERGE POINT. Everything after columns.json is untouched by conceptual
    studies, because the Experiment Manager does not care WHY two columns
    differ. Emitting the site shape here is what buys that.

    NO `band` FIELD. The Analyzer compares planned bands against delivered
    bands; on a conceptual run both are absent and the claim passes. Setting
    band=1 — the natural thing when mirroring the site shape — makes it report
    a failure to deliver bands nobody asked for, straight into the report.

    Raises on an unbuildable design rather than building part of it. A sweep
    missing one level is not a smaller sweep; it is a different experiment with
    the same name.
    """
    wont, _, _ = _validate(design)
    if wont:
        lines = "\n".join(f"  - {w.get('factor')}: {w['why']}" for w in wont)
        raise ValueError(f"This design cannot be built:\n{lines}")

    fixed = dict(design.get("held_fixed") or {})
    factors = design.get("factors") or []

    # First factor slowest, so a clay-by-year sweep reads as all years for
    # clay 5, then all years for clay 10. Stable because a run re-materialised
    # tomorrow must produce the same col_01 as today.
    rows: List[Dict[str, Any]] = [{}]
    for f in factors:
        rows = [{**r, f["name"]: lv} for r in rows for lv in (f.get("levels") or ())]

    lat0 = fixed.get("lat")
    lon0 = fixed.get("lon")
    years0 = fixed.get("years")

    columns: List[Dict[str, Any]] = []
    for i, combo in enumerate(rows, 1):
        lat, lon = lat0, lon0
        if "forcing_site" in combo:
            lat, lon = combo["forcing_site"][0], combo["forcing_site"][1]

        col: Dict[str, Any] = {
            "id": f"col_{i:02d}",
            "lat": round(float(lat), 5),
            "lon": round(float(lon), 5),
            # A conceptual column has no DEM behind it. None rather than 0:
            # zero is a real elevation and would be believed.
            "elevation_m": fixed.get("elevation_m"),
            "pinned": False,
            # WHY THIS COLUMN DIFFERS, carried so the Analyzer can group by it
            # without re-deriving the design from seven soil profiles.
            "treatment": {k: v for k, v in combo.items()},
        }

        if "soil_texture" in combo:
            col["soil_profile"] = uniform_profile(
                combo["soil_texture"], fixed.get("sand_pct"),
                organic_pct=fixed.get("organic_matter_pct", 2.0),
                bulk_density=fixed.get("bulk_density_gcc", 1.4),
                depth_cm=fixed.get("depth_cm", 200))
            # THE FLAG attach_donor_soil READS. Without it the warm start
            # replaces this profile with the donor gridcell's, every column at
            # one lat/lon gets the SAME soil, and the sweep finishes cleanly
            # with no gradient in it.
            col["soil_source"] = "prescribed"

        years = combo.get("forcing_year", years0)
        if years is not None:
            yrs = years if isinstance(years, (list, tuple)) else [years, years]
            col["forcing_start"] = int(yrs[0])
            col["forcing_end"] = int(yrs[-1])

        # WRITTEN ON EVERY COLUMN, not inferred from its absence. Conceptual
        # runs are cold, and a reader finding no start type cannot tell a cold
        # column from one nobody decided about.
        col["warm_start"] = False

        # THE SPEC TRAVELS, NOT THE FILES. Writing weather here would put a
        # year of netCDF behind every column before anything has decided the
        # study is going ahead, and would have to guess the case directory the
        # stream must sit beside. The spec is small, JSON, and survives
        # columns.json → the run plan → case_inputs.json; elm_wrapper turns it
        # into files at the one moment the case directory exists.
        weather = combo.get("prescribed_weather", fixed.get("weather"))
        if weather is not None:
            col["weather"] = weather

        columns.append(col)

    return columns


def as_columns_file(design: Dict[str, Any]) -> Dict[str, Any]:
    """The whole columns.json payload, in the shape expand_sampling writes.

    `bands` and `sampling_design` are present and empty rather than absent: a
    reader that walks the site shape should find the keys it expects and see
    that this run had no bands, instead of raising on a missing key.
    """
    columns = build_columns(design)
    return {
        "approach": "factor_sweep",
        "bbox": None,
        "n_requested": _n_columns(design),
        "n_columns": len(columns),
        "bands": [],
        "sampling_design": {
            "approach": "factor_sweep",
            "n_columns_requested": _n_columns(design),
            "factors": [{"name": f.get("name"),
                         "levels": list(f.get("levels") or ())}
                        for f in (design.get("factors") or [])],
            "held_fixed": dict(design.get("held_fixed") or {}),
            "n_pinned": 0,
            "n_stratified": 0,
            "pinned_stations": [],
        },
        "columns": columns,
    }
