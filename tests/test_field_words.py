"""Names in words: a quantity, a unit, a region and a period as a person
writes them (src/agents/analysis/field_words.py, interface I4 of the report
rebuild).

    pytest tests/test_field_words.py -v
"""
import pytest

from agents.analysis import analysis_plan as AP
from agents.analysis import field_words as FW


def _catalog_rows():
    AP._elm()
    import domain_catalog as DC
    return [f for f, e in DC.load_notes().items() if DC.is_row(e)]


def test_every_field_of_the_catalog_has_a_short_name():
    rows = _catalog_rows()
    assert rows, "the catalog holds no field"
    missing = [f for f in rows if f not in FW.NAMES]
    assert not missing, f"no name in words for {missing}"
    for f in rows:
        name = FW.NAMES[f]
        assert name == name.strip() and name[:1].islower(), (f, name)
        assert f not in name, f"{f}: the name repeats the code"


def test_a_name_with_its_code_and_without():
    assert FW.words("GPP") == "plant growth (GPP)"
    assert FW.words("GPP", with_code=False) == "plant growth"
    assert FW.words("SOILWATER_10CM", with_code=False) == "topsoil water"
    assert FW.words("COL_FIRE_CLOSS") == "carbon lost to fire (COL_FIRE_CLOSS)"


@pytest.mark.parametrize("code", ["XYZ_NEW", "gpp", ""])
def test_an_unknown_field_is_its_code_unchanged(code):
    assert FW.words(code) == code
    assert FW.words(code, with_code=False) == code


@pytest.mark.parametrize("units,words,period", [
    ("gC/m^2/month", "gC/m2", "month"),
    ("gC/m^2/season", "gC/m2", "season"),
    ("mm/yr", "mm", "year"),
    ("mm/water year", "mm", "water year"),
    ("mm/d", "mm", "day"),
    ("W/m^2", "W/m2", None),
    ("kg/m2", "kg/m2", None),
    ("1", "", None),
    ("proportion", "fraction", None),
    ("proportion/month", "fraction", "month"),
    ("K", "K", None),
    (None, "", None),
])
def test_a_unit_as_a_person_writes_it(units, words, period):
    assert FW.units_words(units) == words
    assert FW.units_period(units) == period


@pytest.mark.parametrize("region,words", [
    ("basin, vegetated", "the basin's vegetated land"),
    ("watershed, vegetated", "the watershed's vegetated land"),
    ("vegetated cells", "the vegetated land"),
    ("basin, vegetated, 1000-1500 m", "the basin's vegetated land at 1000 to 1500 m"),
    ("basin, vegetated, below 1000 m", "the basin's vegetated land below 1000 m"),
    ("basin, vegetated, 2000 m and above", "the basin's vegetated land at 2000 m and above"),
    ("below 1000 m", "the vegetated land below 1000 m"),
    ("burned_any_year, vegetated", "the vegetated part of the land that burned in any year"),
])
def test_a_region_in_words(region, words):
    assert FW.region_words(region) == words


@pytest.mark.parametrize("period,words", [
    ("2033-JJA", "summer 2033"), ("2033-DJF", "winter 2032 to 2033"),
    ("2033", "2033"), ("WY2033", "water year 2033"), ("2033-06", "June 2033"),
    ("normal", "normal"),
])
def test_a_period_in_words(period, words):
    assert FW.period_words(period) == words


def test_months_years_and_lists_in_words():
    assert FW.months_words([6, 7, 8]) == "June to August"
    assert FW.months_words([12, 1, 2]) == "December to February"
    assert FW.months_words([6]) == "June"
    assert FW.months_words([1, 6]) == "January and June"
    assert FW.years_words([2034, 2033]) == "2033 and 2034"
    assert FW.years_words([2031, 2032, 2033]) == "2031 to 2033"
    assert FW.years_words([2031, 2033, 2035]) == "2031, 2033 and 2035"
    assert FW.join(["a", "b", "c"]) == "a, b and c"
    assert FW.capital("plant growth (GPP)") == "Plant growth (GPP)"
    assert FW.band_words("1000-1500 m") == "1000 to 1500 m"
