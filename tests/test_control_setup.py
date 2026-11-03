"""The control's fire setup, grid and configuration, from its own files, on the
card, the design page and the top of every report.

    pytest tests/test_control_setup.py -v
    pytest tests/test_control_setup.py --runlocaldata -v   # + the Naches control

A synthetic case is written here (CaseDocs/lnd_in and datm_in with its stream
file, env XML, a domain file and a surface file), so every fact is known in
advance: the grid's land area is the domain's area x frac over its land cells
times ELM's earth radius squared, worked out below from the arrays written,
and every sentence is the one the files call for.
"""
import json
import math
import os
import time
from pathlib import Path

import netCDF4
import numpy as np
import pytest

import domain_controls as DCT

ROOT = Path(__file__).resolve().parents[1]
R_KM = 6371.22

LND_IN = """&elm_inparm
 co2_ppmv = 367.0
 co2_type = 'constant'
 fatmlndfrc = '{domain}'
 fsurdat = '{surface}'
 nu_com = 'RD'
 suplnitro = 'NONE'
 suplphos = 'ALL'
 use_century_decomp = .false.
 use_cn = .true.
 use_fates = .false.
 use_lch4 = .true.
 use_nofire = {nofire}
 use_vertsoilc = .true.
/
&ndepdyn_nml
 stream_fldfilename_ndep = '/in/fndep_simyr1849-2106.nc'
 stream_year_first_ndep = 2000
 stream_year_last_ndep = 2000
/
&pdepdyn_nml
 stream_fldfilename_pdep = '/in/fpdep_simyr2000.nc'
 stream_year_first_pdep = 2000
 stream_year_last_pdep = 2000
/
&popd_streams
 popdensmapalgo = 'bilinear'
 stream_fldfilename_popdens = '/in/hdm_simyr1850-2100.nc'
 stream_year_first_popdens = 1850
 stream_year_last_popdens = 1850
/
&light_streams
 lightngmapalgo = 'bilinear'
 stream_fldfilename_lightng = '/in/lnfm_climo.nc'
 stream_year_first_lightng = 0001
 stream_year_last_lightng = 0001
/
"""

DATM_IN = """&shr_strdata_nml
  datamode = "CLMNCEP"
  domainfile = "{domain}"
  streams = "datm.streams.txt.CLMMOSARTTEST 2013 2013 2023",
      "datm.streams.txt.presaero.clim_2000 1 1 1"
  taxmode = "cycle", "cycle"
/
"""

STREAM = """<?xml version="1.0"?>
<file id="stream" version="1.0">
<domainInfo>
  <variableNames>xc lon yc lat</variableNames>
  <filePath>/in/share/domains</filePath>
  <fileNames>domain.lnd.nldas2.nc</fileNames>
</domainInfo>
<fieldInfo>
  <variableNames>TBOT tbot PRECTmms precn</variableNames>
  <filePath>/in/atm/datm7/NLDAS</filePath>
  <fileNames>clmforc.nldas.2013-01.nc
clmforc.nldas.2023-12.nc</fileNames>
</fieldInfo>
</file>
"""

ENV = {"COMPSET": "2000_DATM%QIA_ELM%CNPRDCTCBC_SICE_SOCN_SROF_SGLC_SWAV_SIAC_SESP",
       "ELM_NML_USE_CASE": "2000_control",
       "ELM_BLDNML_OPTS": "-bgc bgc -nutrient cnp -nutrient_comp_pathway rd  -soil_decomp ctc "
                          "-methane ",
       "CCSM_CO2_PPMV": "367.0", "ELM_CO2_TYPE": "constant", "DATM_MODE": "CLMMOSARTTEST",
       "DATM_PRESAERO": "clim_2000", "DATM_TOPO": "observed", "DATM_CO2_TSERIES": "none",
       "DATM_CLMNCEP_YR_ALIGN": "2013", "DATM_CLMNCEP_YR_START": "2013",
       "DATM_CLMNCEP_YR_END": "2023", "RUN_STARTDATE": "2015-01-01"}

NJ, NI, D = 3, 4, 1.0 / 120.0


def _domain(path):
    """A 4 x 3 box at 1/120 degree near 46.8 N, one cell masked out and one
    half land; areas in square radians as a domain file holds them."""
    lat = 46.70417 + D * np.arange(NJ)
    lon = 238.40416 + D * np.arange(NI)
    xc, yc = np.meshgrid(lon, lat)
    area = np.radians(D) ** 2 * np.cos(np.radians(yc))
    mask = np.ones((NJ, NI), "i4")
    mask[2, 3] = 0
    frac = np.ones((NJ, NI))
    frac[0, 0] = 0.5
    with netCDF4.Dataset(path, "w") as f:
        f.createDimension("nj", NJ)
        f.createDimension("ni", NI)
        for n, a in (("xc", xc), ("yc", yc), ("area", area), ("frac", frac)):
            f.createVariable(n, "f8", ("nj", "ni"))[:] = a
        f.createVariable("mask", "i4", ("nj", "ni"))[:] = mask
    return {"path": Path(path), "xc": xc, "yc": yc, "area": area, "mask": mask, "frac": frac}


def _surface(path, shape=(NJ, NI)):
    with netCDF4.Dataset(path, "w") as f:
        f.createDimension("lsmlat", shape[0])
        f.createDimension("lsmlon", shape[1])
        two = ("lsmlat", "lsmlon")
        f.createVariable("gdp", "f8", two)[:] = np.full(shape, 31.73)
        f.createVariable("peatf", "f8", two)[:] = np.zeros(shape)
        abm = np.full(shape, 4.0)
        abm[0, 1] = 9.0
        f.createVariable("abm", "f8", two)[:] = abm
    return Path(path)


def _case(tmp_path, nofire=".false.", domain=True, env=None):
    cd = tmp_path / "case"
    (cd / "CaseDocs").mkdir(parents=True)
    dom = _domain(tmp_path / "domain.nc") if domain else {"path": tmp_path / "gone.nc"}
    surf = _surface(tmp_path / "surfdata.nc")
    (cd / "CaseDocs" / "lnd_in").write_text(LND_IN.format(domain=dom["path"], surface=surf,
                                                          nofire=nofire))
    (cd / "CaseDocs" / "datm_in").write_text(DATM_IN.format(domain=dom["path"]))
    (cd / "CaseDocs" / "datm.streams.txt.CLMMOSARTTEST").write_text(STREAM)
    (cd / "env_run.xml").write_text("".join(
        f'<entry id="{k}" value="{v}">\n' for k, v in (env or ENV).items()))
    return cd, dom


class TestTheFireSetup:

    def test_the_models_own_fire_with_its_inputs(self, tmp_path):
        cd, _ = _case(tmp_path)
        fire = DCT.setup_facts(cd)["fire"]
        assert fire["on"] is True and fire["use_nofire"] is False
        assert fire["population_density"] == {"file": "/in/hdm_simyr1850-2100.nc",
                                              "file_name": "hdm_simyr1850-2100.nc",
                                              "years": [1850, 1850], "map": "bilinear"}
        assert fire["lightning"]["years"] == [1, 1]
        assert set(fire["namelist"]) == {
            "use_nofire", "popdensmapalgo", "stream_fldfilename_popdens",
            "stream_year_first_popdens", "stream_year_last_popdens", "lightngmapalgo",
            "stream_fldfilename_lightng", "stream_year_first_lightng",
            "stream_year_last_lightng"}
        assert fire["prescribed_burned_area"] == []
        assert fire["surface"]["abm"] == {"what": "month of peak crop-residue burning",
                                          "present": True, "min": 4.0, "max": 9.0}
        assert fire["sentence"] == (
            "the fire is ELM's own fire model (use_nofire false), lit by lightning from "
            "lnfm_climo.nc (one year of data used the same way every year, stream year 1) "
            "with population density held at year 1850 "
            "(hdm_simyr1850-2100.nc); no observed fire is prescribed: the land namelist "
            "names no burned-area input")

    def test_fire_off(self, tmp_path):
        cd, _ = _case(tmp_path, nofire=".true.")
        fire = DCT.setup_facts(cd)["fire"]
        assert fire["on"] is False
        assert fire["sentence"].startswith("the fire model is off (use_nofire true), so "
                                           "nothing burns in the run; no observed fire")


    # WHOSE FIRE, AND "NOTHING BURNS" ONLY WHEN THE CASE SAYS SO (2026-11-03)
    @pytest.mark.parametrize("nl, on, sentence", [
        ({"use_cn": [False], "use_fates": [True], "fates_spitfire_mode": [1]}, True,
         "the fire is FATES's own fire model, not ELM's (fates_spitfire_mode 1)"),
        ({"use_fates": [True], "fates_spitfire_mode": [0]}, False,
         "FATES runs the vegetation with its fire model off (fates_spitfire_mode 0)"),
        ({"use_fates": [True]}, None,
         "FATES runs the vegetation and the land namelist does not say whether its fire "
         "model is on (no fates_spitfire_mode)"),
        ({"use_cn": [False], "use_fates": [False]}, False,
         "ELM's fire model does not run: it is part of the carbon-nutrient cycle, which "
         "is off (use_cn false, use_nofire not set)"),
        ({}, False, "the fire setup could not be read: the land namelist sets neither "
                    "use_nofire nor use_cn"),
        ({"use_cn": [True]}, True, "the fire is ELM's own fire model (use_nofire not set), "
                                   "with no lightning file named"),
    ])
    def test_whose_fire_without_use_nofire(self, nl, on, sentence):
        fire = DCT.fire_setup(nl)
        assert fire["on"] is on
        assert fire["sentence"].startswith(sentence)
        assert "None" not in fire["sentence"]
        assert "nothing burns" not in fire["sentence"]

    def test_fates_fire_settings_are_listed(self):
        fire = DCT.fire_setup({"use_fates": [True], "fates_spitfire_mode": [1]})
        assert fire["namelist"] == {"fates_spitfire_mode": 1}


class TestTheWords:
    """The idea before the code (2026-11-03)."""

    def test_a_year_with_no_calendar_date_is_not_called_a_year(self):
        assert DCT._held([1, 1]) == ("one year of data used the same way every year, "
                                     "stream year 1")
        assert DCT._held([1850, 1850]) == "held at year 1850"
        assert DCT._held([2013, 2023]) == "read for 2013-2023"

    def test_century_and_an_unknown_competition_code(self, tmp_path):
        cd, _ = _case(tmp_path, env=dict(ENV, ELM_BLDNML_OPTS="-bgc bgc -soil_decomp century"))
        lnd = cd / "CaseDocs" / "lnd_in"
        lnd.write_text(lnd.read_text().replace("nu_com = 'RD'", "nu_com = 'XYZ'"))
        words = DCT.setup_facts(cd)["configuration"]["sentences"]["carbon_nutrient"]
        assert "nutrient competition set by nu_com XYZ" in words
        assert "soil decomposition by the CENTURY scheme," in words


class TestTheGrid:

    def test_every_number_from_the_domain_file(self, tmp_path):
        cd, dom = _case(tmp_path)
        g = DCT.setup_facts(cd)["grid"]
        assert (g["rows"], g["columns"], g["cells"], g["land_cells"]) == (3, 4, 12, 11)
        land = float(np.sum(np.where(dom["mask"] > 0, dom["area"] * dom["frac"], 0.0)))
        assert g["land_area_km2"] == pytest.approx(round(land * R_KM ** 2, 1), abs=1e-9)
        mean_lat = float(dom["yc"].mean())
        assert g["mean_lat"] == pytest.approx(round(mean_lat, 3))
        km = math.pi * R_KM / 180.0 * D
        assert g["cell_size_km"]["north_south"] == pytest.approx(round(km, 3), abs=1e-3)
        assert g["cell_size_km"]["east_west"] == pytest.approx(
            round(km * math.cos(math.radians(mean_lat)), 3), abs=1e-3)
        assert g["lat_range"] == [round(float(dom["yc"].min()), 4),
                                  round(float(dom["yc"].max()), 4)]
        # longitudes as a reader writes them, west negative
        assert g["lon_range"][0] == pytest.approx(238.40416 - 360.0, abs=1e-4)
        assert g["surface_matches"] is True
        assert g["sentence"] == (
            f"4 x 3 cells (columns by rows), 12 in all, 11 of them land; cells of 1/120 "
            f"degree, about {g['cell_size_km']['east_west']:.2f} km east-west by "
            f"{g['cell_size_km']['north_south']:.2f} km north-south at the box's mean "
            f"latitude, {mean_lat:.1f} degrees; {land * R_KM ** 2:,.0f} km2 of land")

    def test_a_surface_of_another_shape_is_flagged(self, tmp_path):
        cd, dom = _case(tmp_path)
        _surface(tmp_path / "surfdata.nc", shape=(2, 4))
        assert DCT.setup_facts(cd)["grid"]["surface_matches"] is False

    def test_a_missing_domain_file_spoils_only_the_grid(self, tmp_path):
        cd, _ = _case(tmp_path, domain=False)
        f = DCT.setup_facts(cd)
        assert "no readable land domain file" in f["grid"]["error"]
        assert f["fire"]["sentence"] and f["configuration"]["sentences"]


class TestTheConfiguration:

    def test_every_sentence(self, tmp_path):
        cd, _ = _case(tmp_path)
        c = DCT.setup_facts(cd)["configuration"]
        assert c["compset"] == ENV["COMPSET"]
        assert c["carbon_nutrient"]["build_flags"] == {
            "bgc": "bgc", "nutrient": "cnp", "nutrient_comp_pathway": "rd",
            "soil_decomp": "ctc", "methane": True}
        assert c["deposition"]["nitrogen"]["years"] == [2000, 2000]
        assert c["weather"]["source"] == "NLDAS" and c["weather"]["time_axis"] == "cycle"
        assert c["sentences"] == {
            "compset": ENV["COMPSET"] + " (land namelist use case 2000_control)",
            "carbon_nutrient": "carbon, nitrogen and phosphorus cycles (use_cn true), "
                               "plants and microbes share nutrients by relative demand "
                               "(nu_com RD), soil decomposition by the converging trophic "
                               "cascade scheme (CTC), "
                               "soil carbon resolved by depth, methane on",
            "co2": "held constant at 367 ppmv",
            "deposition": "nitrogen deposition held at year 2000 from "
                          "fndep_simyr1849-2106.nc; phosphorus deposition held at year "
                          "2000 from fpdep_simyr2000.nc",
            "supplements": "nitrogen not supplemented (suplnitro NONE); phosphorus "
                           "supplemented to meet plant demand everywhere (suplphos ALL)",
            "weather": "weather read from NLDAS files by the data atmosphere, years "
                       "2013-2023, model year Y reading weather year Y, repeated in a "
                       "cycle past the last year (DATM mode CLMMOSARTTEST, data mode "
                       "CLMNCEP)"}

    def test_a_shifted_weather_year_is_said(self, tmp_path):
        cd, _ = _case(tmp_path, env=dict(ENV, DATM_CLMNCEP_YR_ALIGN="2015"))
        w = DCT.setup_facts(cd)["configuration"]["sentences"]["weather"]
        assert ", year 2013 read as model year 2015," in w

    def test_no_dash_as_punctuation_in_any_sentence(self, tmp_path):
        cd, _ = _case(tmp_path)
        f = DCT.setup_facts(cd)
        text = json.dumps([f["fire"]["sentence"], f["grid"]["sentence"],
                           f["configuration"]["sentences"]], ensure_ascii=False)
        assert "\u2014" not in text and "\u2013" not in text


# ── the cache ────────────────────────────────────────────────────────────────
@pytest.fixture
def control(tmp_path):
    from test_domain_controls import _control
    c, reg, _ = _control(tmp_path, with_obs=False)
    cd, _ = _case(tmp_path / "setup")
    for p in (cd / "CaseDocs").iterdir():
        (Path(c["case_dir"]) / "CaseDocs" / p.name).write_bytes(p.read_bytes())
    (Path(c["case_dir"]) / "env_run.xml").write_text((cd / "env_run.xml").read_text())
    return c, reg


class TestTheCache:

    def test_computed_facts_carry_the_version_and_the_setup(self, control):
        c, reg = control
        f = DCT.compute_facts("synth", str(reg))
        assert f["version"] == DCT.FACTS_VERSION
        assert f["fire"]["on"] is True and f["grid"]["cells"] == 12
        assert DCT.facts("synth", str(reg))["configuration"]["compset"] == ENV["COMPSET"]

    def test_a_cache_of_the_old_format_gains_the_setup_and_keeps_its_time(self, control):
        c, reg = control
        f = DCT.compute_facts("synth", str(reg))
        p = Path(reg).parent / DCT.FACTS_DIR / "synth.json"
        old = {k: v for k, v in f.items()
               if k not in ("version", "fire", "grid", "configuration")}
        p.write_text(json.dumps(old, default=str))
        t = time.time() - 5
        os.utime(p, (t, t))
        h = sorted(Path(c["run_dir"]).glob("*.elm.h*.nc"))
        for q in h:
            os.utime(q, (t - 60, t - 60))
        got = DCT.facts("synth", str(reg))
        assert got["version"] == DCT.FACTS_VERSION and got["grid"]["land_cells"] == 11
        assert got["per_year"] == json.loads(json.dumps(f["per_year"], default=str))
        assert p.stat().st_mtime == pytest.approx(t, abs=1e-3)
        assert json.loads(p.read_text())["fire"]["sentence"] == f["fire"]["sentence"]
        # still stale once a history file is newer than the year facts
        os.utime(h[0], (t + 60, t + 60))
        assert DCT.facts("synth", str(reg)) is None

    # A CACHE THE SERVER CANNOT WRITE, AND A PART THAT FAILED (2026-11-03)
    def _old_cache(self, c, reg):
        f = DCT.compute_facts("synth", str(reg))
        p = Path(reg).parent / DCT.FACTS_DIR / "synth.json"
        old = {k: v for k, v in f.items()
               if k not in ("version", "fire", "grid", "configuration")}
        p.write_text(json.dumps(old, default=str))
        t = time.time() + 5
        os.utime(p, (t, t))
        return p, old

    def test_a_cache_the_server_cannot_write_still_gives_the_facts(self, control):
        import stat
        c, reg = control
        p, old = self._old_cache(c, reg)
        os.chmod(p.parent, stat.S_IRUSR | stat.S_IXUSR)
        try:
            if os.access(str(p.parent), os.W_OK):
                pytest.skip("this user can write a read-only folder")
            d = DCT.describe(str(reg))
            got = DCT.facts("synth", str(reg))
        finally:
            os.chmod(p.parent, stat.S_IRWXU)
        assert d["controls"][0]["facts"]["grid"]["cells"] == 12
        assert got["version"] == DCT.FACTS_VERSION and got["fire"]["on"] is True
        assert json.loads(p.read_text()) == json.loads(json.dumps(old, default=str))
        assert sorted(q.name for q in p.parent.iterdir()) == ["synth.json"]

    def test_a_setup_part_that_failed_is_not_written_and_is_read_again(self, control):
        c, reg = control
        p, old = self._old_cache(c, reg)
        lnd = Path(c["case_dir"]) / "CaseDocs" / "lnd_in"
        lnd.rename(lnd.with_suffix(".away"))
        try:
            first = DCT.facts("synth", str(reg))
        finally:
            lnd.with_suffix(".away").rename(lnd)
        assert "error" in first["grid"]
        assert "version" not in json.loads(p.read_text())
        again = DCT.facts("synth", str(reg))
        assert "error" not in again["grid"] and again["grid"]["cells"] == 12
        assert json.loads(p.read_text())["grid"]["cells"] == 12

    def test_a_cache_of_this_format_with_a_failed_part_reads_it_again(self, control):
        c, reg = control
        f = DCT.compute_facts("synth", str(reg))
        p = Path(reg).parent / DCT.FACTS_DIR / "synth.json"
        d = json.loads(p.read_text())
        d["fire"] = {"error": "FileNotFoundError: lnd_in"}
        p.write_text(json.dumps(d))
        t = time.time() + 5
        os.utime(p, (t, t))
        got = DCT.facts("synth", str(reg))
        assert got["fire"]["sentence"] == f["fire"]["sentence"]
        assert json.loads(p.read_text())["fire"]["sentence"] == f["fire"]["sentence"]
        assert p.stat().st_mtime == pytest.approx(t, abs=1e-3)

    def test_a_format_this_code_does_not_know_is_not_used(self, control):
        c, reg = control
        DCT.compute_facts("synth", str(reg))
        p = Path(reg).parent / DCT.FACTS_DIR / "synth.json"
        d = json.loads(p.read_text())
        p.write_text(json.dumps(dict(d, version=DCT.FACTS_VERSION + 1)))
        assert DCT.facts("synth", str(reg)) is None

    def test_reception_sees_the_sentences_and_stays_small(self, control):
        c, reg = control
        DCT.compute_facts("synth", str(reg))
        d = DCT.describe(str(reg))
        row = d["controls"][0]["facts"]
        assert set(row["fire"]) == {"on", "sentence"}
        assert set(row["grid"]) == set(DCT.SETUP_SUMMARY["grid"])
        assert set(row["configuration"]) == {"sentences"}
        # the tool result as the loop measures it: the JSON text, dumped again
        assert len(json.dumps(json.dumps(d, indent=1, default=str))) < 12000

    def test_facts_without_the_setup_say_so(self):
        row = DCT.summary({"name": "x", "usable": True}, {"settings": {}})
        assert row["facts"]["fire"] == {
            "error": "not in these facts: refresh_domain_control_facts reads it"}


# ── the card and the page ────────────────────────────────────────────────────
FACTS = {"output_years": [2015, 2021], "start_year": 2015,
         "fire": {"on": True, "sentence": "the fire is ELM's own fire model"},
         "grid": {"cells": 12, "sentence": "4 x 3 cells (columns by rows)"}}


def test_the_card_quotes_the_fire_and_the_grid():
    from agents.reception_llm import control_card
    card = control_card([{"name": "synth", "usable": True, "description": "d",
                          "facts": FACTS}])
    assert card.splitlines()[2:5] == ["  synth: d",
                                      "  Fire: the fire is ELM's own fire model",
                                      "  Grid: 4 x 3 cells (columns by rows)"]
    old = control_card([{"name": "synth", "usable": True, "description": "d",
                         "facts": {"fire": {"error": "not in these facts"}}}])
    assert "  Fire: not in the control's facts (not in these facts)" in old
    assert "Fire" not in control_card([{"name": "s", "usable": True, "facts": None}])


def test_the_design_page_quotes_them_too():
    from agents import design_review as DR
    de = {"control": {"name": "synth", "reuse": True}, "years": [2015], "runs": []}
    page = DR._whole_domain_lines({}, {"domain_experiment": de, "domain_rulebook": {
        "controls": {"controls": [{"name": "synth", "description": "d", "facts": FACTS}]}}})
    k = page.index("- its output covers 2015–2021, from the saved state of 2015")
    assert page[k + 1:k + 3] == ["- its fire: the fire is ELM's own fire model",
                                 "- its grid: 4 x 3 cells (columns by rows)"]


def test_the_registry_example_says_whose_fire_the_burned_region_holds():
    ex = json.loads((ROOT / "mcp" / "elm-mcp" / "domain_controls.json.example").read_text())
    r = ex["controls"]["my_basin"]["regions"]["burned_any_year"]
    assert r["what"] == ("cells where MODIS saw fire in any year 2015-2021: where real "
                         "fires burned, not where the model's fire burns")
    assert (r["var"], r["op"], r["over"]) == ("burned_fraction_year", ">", "any")


# ── the model setup at the top of the reports ────────────────────────────────
def _setup_facts(tmp_path):
    cd, _ = _case(tmp_path / "ctl")
    s = DCT.setup_facts(cd)
    return {k: v for k, v in DCT._setup_summary(s).items()}


def _run_dir(tmp_path, fields=("GPP", "QRUNOFF"), variables=None, with_setup=True):
    import domain_analysis as A
    rd = tmp_path / "run"
    (rd / "01_inputs").mkdir(parents=True)
    (rd / "03_results").mkdir()
    facts = {"output_years": [2015, 2021], "start_year": 2015}
    if with_setup:
        facts.update(_setup_facts(tmp_path))
    (rd / "01_inputs" / "domain_control_facts.json").write_text(json.dumps({
        "control": {"name": "synth", "reuse": True},
        "facts": {"name": "synth", "description": "a synthetic control", "facts": facts}}))
    meta = {"stream": "h0", "how": "sum", "units": "mm/yr", "note": None}
    table = {"years": [2015, 2016], "regions": {"vegetated cells": {}},
             "runs": {"control": {"fields": {f: meta for f in fields}, "values": {}}}}
    (rd / "03_results" / "regional.json").write_text(json.dumps(table))
    (rd / "experiment.json").write_text(json.dumps({
        "kind": "whole_domain", "regional": table, "control_facts": facts,
        "limitations": json.loads(json.dumps(A.LIMITATIONS)),
        "experiment": {"control": {"name": "synth", "reuse": True}, "runs": [
            {"run": "control", "reused": True, "changes": []},
            {"run": "dry", "reused": False, "status": "completed", "changes": [
                {"id": "weather_scale", "params": {"precip_factor": 0.8, "months": [6, 7]},
                 "summary": "multiply the rain and snow"}]}]},
        "chain": {"years_planned": [2015, 2016], "years_passed": [2015, 2016]}}))
    (rd / "strategy.json").write_text(json.dumps({"domain_experiment": {
        "analysis": {"variables": list(variables or fields)}}}))
    return rd


class TestTheModelSetup:

    def test_the_block_quotes_the_records(self, tmp_path):
        import domain_analysis as A
        from agents.analysis import model_setup as MS
        rd = _run_dir(tmp_path)
        rec = MS.collect(rd)
        lines = MS.render(rec)
        s = _setup_facts(tmp_path / "again")
        assert lines[0] == "## Model setup"
        assert "- the control: **synth**, a synthetic control" in lines
        assert f"- grid: {s['grid']['sentence']}" in lines
        assert ("- years: reported 2015-2016; the control's output covers 2015-2021, from "
                "the saved state of 2015") in lines
        assert f"  - fire: {s['fire']['sentence']}" in lines
        for key, label in MS.CONFIG_ORDER:
            if key != "fire":
                assert f"  - {label}: {s['configuration']['sentences'][key]}" in lines
        assert "  - control: the control, reused: its existing output was read" in lines
        assert ("  - dry: the control with weather_scale (precip_factor 0.8, months 6, 7): "
                "multiply the rain and snow") in lines
        caveat = A.LIMITATIONS["structural"][0]["caveat"]
        assert f"- runoff (QRUNOFF): {caveat}" in lines

    def test_no_runoff_field_no_runoff_line(self, tmp_path):
        from agents.analysis import model_setup as MS
        rd = _run_dir(tmp_path, fields=("GPP", "QRUNOFF"), variables=["GPP"])
        assert MS.collect(rd)["runoff"] is None
        assert not any(x.startswith("- runoff") for x in MS.render(MS.collect(rd)))

    def test_facts_from_before_the_setup_say_so(self, tmp_path):
        from agents.analysis import model_setup as MS
        lines = MS.render(MS.collect(_run_dir(tmp_path, with_setup=False)))
        # nothing in the older record or the description: said so, line by line
        assert f"- grid: {MS.NOT_HELD}" in lines and f"  - fire: {MS.NOT_HELD}" in lines
        assert f"  - compset: {MS.NOT_HELD}" in lines
        assert any("refresh_domain_control_facts reads them" in x for x in lines)

    def test_older_facts_show_what_they_hold(self, tmp_path):
        # THE OLDER RECORD'S SETTINGS AND DESCRIPTION (2026-11-03), as the
        # Naches control's facts of 2026-10-24 hold them
        from agents.analysis import model_setup as MS
        rd = _run_dir(tmp_path, with_setup=False)
        p = rd / "01_inputs" / "domain_control_facts.json"
        d = json.loads(p.read_text())
        d["facts"]["description"] = (
            "the Naches ELM-BGC control: one case over the 144 x 96 box at 1/120 degree "
            "(cells about 0.6 km east-west by 0.9 km north-south; 8,126 km2 of land, about "
            "3.3 times the watershed region), 2015-2021 from a saved state at 2015-01-01, "
            "NLDAS weather, fire model on, CO2 constant")
        d["facts"]["facts"].update({
            "forcing_years": [2013, 2023],
            "settings": {"use_nofire": False, "co2_ppmv": 367.0, "co2_type": "constant",
                         "stream_year_first_ndep": 2000, "stream_year_last_ndep": 2000,
                         "ndep_stream_years": [1849, 2106], "suplnitro": "NONE",
                         "suplphos": "ALL"}})
        p.write_text(json.dumps(d))
        e = json.loads((rd / "experiment.json").read_text())
        e.pop("control_facts")
        (rd / "experiment.json").write_text(json.dumps(e))
        lines = MS.render(MS.collect(rd))
        for line in (
                "- grid: one case over the 144 x 96 box at 1/120 degree (cells about 0.6 km "
                "east-west by 0.9 km north-south; 8,126 km2 of land, about 3.3 times the "
                "watershed region) (the control's description)",
                f"  - compset: {MS.NOT_HELD}",
                f"  - carbon and nutrients: {MS.NOT_HELD}",
                "  - fire: the switch that turns the fire model off, use_nofire, is false "
                "(older facts); fire model on (the control's description)",
                "  - CO2: held constant at 367 ppmv (co2_type constant, co2_ppmv 367; older "
                "facts)",
                "  - deposition: nitrogen deposition held at year 2000 "
                "(stream_year_first_ndep 2000, stream_year_last_ndep 2000; its file is named "
                "for 1849-2106; older facts); phosphorus deposition: not in the control's "
                "facts",
                "  - nutrient supplements: nitrogen not supplemented (suplnitro NONE); "
                "phosphorus supplemented to meet plant demand everywhere (suplphos ALL) "
                "(older facts)",
                "  - weather: NLDAS weather (the control's description); weather for the "
                "model years 2013-2023 (older facts)"):
            assert line in lines, line
        assert not any(MS.MISSING in x for x in lines)

    def test_a_run_that_is_not_whole_domain_has_no_block(self):
        from agents.analysis import model_setup as MS
        assert MS.render(None) == []
        assert MS.insert_after_answer(["# q", "## The answer", "a", "## Next"], []) == [
            "# q", "## The answer", "a", "## Next"]

    def test_it_sits_under_the_answer_in_the_numbers_report(self, tmp_path):
        pytest.importorskip("matplotlib")
        from agents.analysis import domain_report as DR
        rd = _run_dir(tmp_path)
        DR.build(rd)
        md = (rd / "04_analysis" / "DOMAIN_REPORT.md").read_text()
        assert md.index("## The answer") < md.index("## Model setup") < \
            md.index("## What was run")
        rec = json.loads((rd / "04_analysis" / "domain_report.json").read_text())
        assert rec["model_setup"]["runoff"]["fields"] == ["QRUNOFF"]

    def test_and_in_the_analyzers_report(self, tmp_path):
        from agents.analysis import model_setup as MS
        from agents.analysis import step4_report as S4
        rd = _run_dir(tmp_path)
        rep = {"question": "q", "model": "elm", "status": "supported", "kind": "whole_domain",
               "answer": "a", "claims": [], "caveats": [], "withheld": [],
               "provenance": {}, "cost": {}, "model_setup": MS.collect(rd)}
        md = S4.render_markdown(rep)
        assert md.index("## The answer") < md.index("## Model setup") < \
            md.index("## What the run found")
        assert "## Model setup" not in S4.render_markdown(dict(rep, model_setup=None))

    def test_the_analyzers_report_reads_it_only_for_a_whole_domain_run(self, tmp_path):
        from types import SimpleNamespace
        from agents.analysis import step4_report as S4
        rd = _run_dir(tmp_path)
        ctx = SimpleNamespace(plan={}, caveats=[], data={"kind": "whole_domain"},
                              is_domain=True)
        rep = S4.build(ctx, {}, {}, {}, rd)
        assert rep["model_setup"]["control"]["name"] == "synth"
        rep = S4.build(SimpleNamespace(plan={}, caveats=[], data={}, is_domain=False),
                       {}, {}, {}, rd)
        assert rep["model_setup"] is None


# ── the Naches control, where it is on disk ──────────────────────────────────
@pytest.mark.localdata
def test_the_naches_controls_setup():
    cfg = ROOT / "tests" / "local_oracle_paths.json"
    if not cfg.is_file():
        pytest.skip("no local_oracle_paths.json")
    b = json.loads(cfg.read_text()).get("naches_bgc_gates") or {}
    cd = Path(b.get("fix_case_dir") or "/nonexistent")
    if not (cd / "CaseDocs" / "lnd_in").is_file():
        pytest.skip("the Naches control's case is not on this machine")
    f = DCT.setup_facts(cd)
    g = f["grid"]
    assert (g["rows"], g["columns"], g["cells"], g["land_cells"]) == (96, 144, 13824, 13824)
    assert g["land_area_km2"] == pytest.approx(8125.9, abs=0.1)
    assert g["sentence"].startswith("144 x 96 cells (columns by rows), 13,824 in all, every "
                                    "one of them land; cells of 1/120 degree")
    assert g["surface_matches"] is True
    fire = f["fire"]
    assert fire["on"] is True and fire["population_density"]["years"] == [1850, 1850]
    assert fire["lightning"]["years"] == [1, 1]
    c = f["configuration"]
    assert c["sentences"]["co2"] == "held constant at 367 ppmv"
    assert c["deposition"]["nitrogen"]["years"] == [2000, 2000]
    assert c["weather"]["source"] == "NLDAS" and c["weather"]["years"] == [2013, 2023]


# ONLY WHAT AN OLDER RECORD HOLDS (2026-11-03): a setting the record holds as
# None is marked as not held, never quoted as "None".
def test_older_lines_quote_only_the_settings_held():
    from agents.analysis.model_setup import older_lines
    co2 = older_lines({"older": {"settings": {"CCSM_CO2_PPMV": "367.0"}}})["co2"]
    assert co2 == "co2_ppmv 367 (older facts); co2_type: not in the control's facts"
    co2 = older_lines({"older": {"settings": {"co2_type": "constant",
                                              "co2_ppmv": None}}})["co2"]
    assert co2 == "co2_type constant (older facts); co2_ppmv: not in the control's facts"
    dep = older_lines({"older": {"settings": {"stream_year_first_ndep": 2000,
                                              "stream_year_last_ndep": None}}})["deposition"]
    assert dep == ("nitrogen deposition's first year is 2000 (stream_year_first_ndep 2000; "
                   "older facts); its last year (stream_year_last_ndep): not in the control's "
                   "facts; phosphorus deposition: not in the control's facts")
    dep = older_lines({"older": {"settings": {"stream_year_last_ndep": 2001,
                                              "ndep_stream_years": [1849, 2106]}}})
    assert dep["deposition"].startswith(
        "nitrogen deposition's last year is 2001 (stream_year_last_ndep 2001; its file is "
        "named for 1849-2106; older facts); its first year (stream_year_first_ndep): not in")
    for line in (co2, dep["deposition"]):
        assert "None" not in line
