#!/usr/bin/env python3
"""
A model quantity, a unit, a region and a period in the words a person writes
src/agents/analysis/field_words.py

    words("GPP")                    "plant growth (GPP)"
    words("GPP", with_code=False)   "plant growth"
    units_words("gC/m^2/month")     "gC/m2"
    region_words("watershed, vegetated")    "the watershed's vegetated land"
    period_words("2033-JJA")        "summer 2033"

WHY A LIST WRITTEN BY HAND (the report rebuild, 2026-11-13). The report's
pages and figures name each quantity in words, with the model's code in
brackets the first time ("plant growth (GPP)"). The field notes hold each
field's meaning as a phrase ("gross primary production"), which is exact but
not what a reader says, and the claim trace cuts a name from that phrase
(claim_trace._plain_name) to find a field a claim mentions. That cut is for
matching, not for reading: it gives "gross primary production" for GPP. So
the short names are written here, one per field of the catalog
(mcp/elm-mcp/domain_field_notes.json), and a test holds that every field of
the catalog has one. The notes file is the server's and gains no key.

A field not in the list is named by its code, unchanged, so a new field is
never given a wrong name, only a plain one.
"""
import re
from typing import Iterable, Optional, Sequence

NAMES = {
    # weather in
    "FSDS": "incoming solar radiation",
    "RAIN": "rain",
    "SNOW": "snowfall",
    "TBOT": "air temperature",
    # snow
    "FSNO": "snow cover",
    "H2OSNO": "snowpack water",
    "QSNOMELT": "snowmelt",
    "SNOWDP": "snow depth",
    # soil water and groundwater
    "H2OSFC": "standing surface water",
    "H2OSOI": "soil moisture",
    "QCHARGE": "aquifer recharge",
    "SOILWATER_10CM": "topsoil water",
    "TWS": "total water storage",
    "ZWT": "water table depth",
    # runoff
    "QDRAI": "subsurface runoff",
    "QH2OSFC": "runoff from standing water",
    "QOVER": "surface runoff",
    "QRUNOFF": "total runoff",
    # water to the air
    "BTRAN": "water stress index",
    "EFLX_LH_TOT": "latent heat",
    "QSOIL": "soil evaporation",
    "QVEGE": "canopy evaporation",
    "QVEGT": "transpiration",
    # energy
    "FIRA": "net longwave radiation",
    "FSA": "absorbed solar radiation",
    "FSH": "sensible heat",
    "RH2M": "relative humidity",
    "TSA": "air temperature at 2 m",
    "TSOI_10CM": "topsoil temperature",
    # carbon in and out
    "AR": "plant respiration",
    "ER": "ecosystem respiration",
    "GPP": "plant growth",
    "HR": "microbial respiration",
    "NEE": "net carbon exchange",
    "NPP": "net plant growth",
    "PSNSHA": "shaded-leaf photosynthesis",
    "PSNSUN": "sunlit-leaf photosynthesis",
    # leaves
    "ELAI": "exposed leaf area index",
    "LEAFC": "leaf carbon",
    "TLAI": "leaf area index",
    # carbon stores
    "CWDC": "dead wood carbon",
    "FROOTC": "fine root carbon",
    "TOTECOSYSC": "ecosystem carbon",
    "TOTLITC": "litter carbon",
    "TOTSOMC": "soil organic carbon",
    "TOTVEGC": "plant carbon",
    # nitrogen
    "FPG": "nitrogen limitation index",
    "FPI": "nitrogen immobilization index",
    "PLANT_NDEMAND": "plant nitrogen demand",
    "SMINN": "soil mineral nitrogen",
    # fire
    "COL_FIRE_CLOSS": "carbon lost to fire",
    "FAREA_BURNED": "area burned",
}


def words(field, with_code: bool = True) -> str:
    """A quantity's short plain name, with its code in brackets ("plant
    growth (GPP)") or without. A field not in the list: its code, as it is."""
    f = str(field)
    name = NAMES.get(f)
    if name is None:
        return f
    return f"{name} ({f})" if with_code else name


def capital(text: str) -> str:
    """The first letter upper case, the rest as written: "Plant growth (GPP)"."""
    t = str(text)
    return t[:1].upper() + t[1:]


def join(parts: Iterable[str]) -> str:
    """"a", "a and b", "a, b and c"."""
    xs = [str(x) for x in parts if str(x)]
    if len(xs) < 2:
        return "".join(xs)
    return ", ".join(xs[:-1]) + " and " + xs[-1]


# A UNIT AS A PERSON WRITES IT. The results store gives a total's unit with the
# period it is summed over ("gC/m^2/month", "mm/season"); a panel or a table
# column already says the period, so the unit drops it. Powers lose their
# caret, a pure number ("1") has no unit, and a proportion is a fraction.
PER_PERIOD = re.compile(r"/(d|day|month|season|yr|year|water year)$")
PERIOD_OF_UNIT = {"d": "day", "day": "day", "month": "month", "season": "season",
                  "yr": "year", "year": "year", "water year": "water year"}


def units_words(units) -> str:
    """"gC/m^2/month" as "gC/m2", "W/m^2" as "W/m2", "1" as "", "proportion"
    as "fraction"; None or "" as ""."""
    u = str(units or "").strip()
    u = PER_PERIOD.sub("", u)
    u = u.replace("^", "")
    if u in ("1", "unitless", "none", "-"):
        return ""
    return "fraction" if u == "proportion" else u


def units_period(units) -> Optional[str]:
    """The period a total's unit is summed over ("month" of "gC/m^2/month"),
    or None for a unit that is not a total over a period."""
    m = PER_PERIOD.search(str(units or "").strip())
    return PERIOD_OF_UNIT.get(m.group(1)) if m else None


# A REGION IN WORDS. The results store names a region by the registry's name,
# ", vegetated" for its vegetated cells and the band after that
# ("watershed, vegetated, 1000-1500 m"). A name the list does not hold is
# written as it is, its underscores as spaces.
REGION_NAMES = {"watershed": "the watershed", "burned_any_year": "the land that burned in any year",
                "vegetated cells": "the vegetated land"}
BAND = re.compile(r"^(below \d+(?:\.\d+)? m|\d+(?:\.\d+)?-\d+(?:\.\d+)? m|"
                  r"\d+(?:\.\d+)? m and above)$")


def band_words(band: str) -> str:
    """An elevation band as a person writes it: "below 1000 m", "1000 to
    1500 m", "2000 m and above"."""
    b = str(band)
    m = re.fullmatch(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?) m", b)
    return f"{m.group(1)} to {m.group(2)} m" if m else b


def region_words(region) -> str:
    """"the watershed's vegetated land", "the vegetated land", "the
    watershed's vegetated land at 1000 to 1500 m"."""
    parts = [p.strip() for p in str(region or "").split(",") if p.strip()]
    if not parts:
        return "the region"
    band = parts.pop() if BAND.match(parts[-1]) else None
    if not parts or parts == ["vegetated cells"] or parts == ["vegetated"]:
        out = "the vegetated land"           # a band of every vegetated cell, too
    else:
        owner = REGION_NAMES.get(parts[0], parts[0].replace("_", " "))
        if not owner.startswith("the "):
            owner = "the " + owner
        if parts[1:] == ["vegetated"]:
            out = f"{owner}'s vegetated land" if " " not in owner[4:] else \
                f"the vegetated part of {owner}"
        elif len(parts) == 1:
            out = owner
        else:
            out = owner + ", " + ", ".join(parts[1:])
    if band:
        w = band_words(band)
        out += " " + (w if w.startswith("below") else "at " + w)
    return out


# A PERIOD IN WORDS, as the store names it: a year ("2033"), a water year
# ("WY2033", October of the year before to September), a season ("2033-JJA";
# December to February is the winter that ends in that year) or a month
# ("2033-06").
SEASON_NAMES = {"DJF": "winter", "MAM": "spring", "JJA": "summer", "SON": "autumn"}
SEASON_MONTHS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December")


def period_words(period) -> str:
    """"summer 2033", "winter 2032 to 2033", "2033", "water year 2033",
    "June 2033"; anything else as it is."""
    p = str(period)
    m = re.fullmatch(r"(\d{4})-([A-Z]{3})", p)
    if m and m.group(2) in SEASON_NAMES:
        y = int(m.group(1))
        if m.group(2) == "DJF":
            return f"winter {y - 1} to {y}"
        return f"{SEASON_NAMES[m.group(2)]} {y}"
    m = re.fullmatch(r"WY(\d{4})", p)
    if m:
        return f"water year {m.group(1)}"
    m = re.fullmatch(r"(\d{4})-(\d{2})", p)
    if m and 1 <= int(m.group(2)) <= 12:
        return f"{MONTHS[int(m.group(2)) - 1]} {m.group(1)}"
    return p


def months_words(months: Sequence[int]) -> str:
    """Calendar months as a span: "June to August" for 6, 7, 8 (December to
    February runs over the new year), "June" alone, or each named for months
    that are not one span."""
    ms = [int(m) for m in months]
    if not ms:
        return ""
    if len(ms) == 1:
        return MONTHS[ms[0] - 1]
    span = all((b - a) % 12 == 1 for a, b in zip(ms, ms[1:]))
    if span:
        return f"{MONTHS[ms[0] - 1]} to {MONTHS[ms[-1] - 1]}"
    return join(MONTHS[m - 1] for m in ms)


def years_words(years: Sequence[int]) -> str:
    """"2033", "2033 and 2034", "2033 to 2036"."""
    ys = sorted({int(y) for y in years})
    if len(ys) > 2 and ys == list(range(ys[0], ys[-1] + 1)):
        return f"{ys[0]} to {ys[-1]}"
    return join(str(y) for y in ys)
