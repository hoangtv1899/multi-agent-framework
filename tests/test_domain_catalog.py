"""The data catalog (domain_catalog) and the field notes, on small fake files.

    pytest tests/test_domain_catalog.py -v
    pytest tests/test_domain_catalog.py --review-gate -v   # + fails while a note is a draft

The fake run copies the control's layout (elm_gridded_fixtures): a daily
stream holding the notes' 44 daily fields, a monthly stream holding them, the
eight monthly-only fields, FIRE and a few others, and a 3-hourly stream. Its
values are random: nothing in a catalog may depend on them, and the test of
that writes the same run twice with other values and wants the same catalog.
"""
import json
import re
import shutil
from pathlib import Path

import netCDF4
import numpy as np
import pytest

import domain_analysis as A
import domain_catalog as C
import elm_case_fixtures as cf
import elm_gridded_fixtures as fx
import gridded as G

NOTES = C.load_notes()
ROWS = {n: e for n, e in NOTES.items() if C.is_row(e)}
DAILY = sorted(n for n, e in ROWS.items() if e["cadence"] == "daily")
MONTHLY_ONLY = sorted(n for n, e in ROWS.items() if e["cadence"] == "monthly")
YEARS = [2015, 2016]

# Units as the Naches control's files carry them (h1 and h0 headers, read
# 2026-11-04): every note's field, FIRE among them, and nothing else.
UNITS = {
    "AR": "gC/m^2/s", "BTRAN": "1", "COL_FIRE_CLOSS": "gC/m^2/s", "CWDC": "gC/m^2",
    "EFLX_LH_TOT": "W/m^2", "ELAI": "m^2/m^2", "ER": "gC/m^2/s", "FAREA_BURNED": "proportion",
    "FIRA": "W/m^2", "FPG": "proportion", "FPI": "proportion", "FROOTC": "gC/m^2",
    "FSA": "W/m^2", "FSH": "W/m^2", "FSNO": "1", "GPP": "gC/m^2/s", "H2OSNO": "mm",
    "H2OSOI": "mm3/mm3", "HR": "gC/m^2/s", "LEAFC": "gC/m^2", "NEE": "gC/m^2/s",
    "NPP": "gC/m^2/s", "QDRAI": "mm/s", "QOVER": "mm/s", "QRUNOFF": "mm/s",
    "QSNOMELT": "mm/s", "QSOIL": "mm/s", "QVEGE": "mm/s", "QVEGT": "mm/s", "RAIN": "mm/s",
    "SMINN": "gN/m^2", "SNOW": "mm/s", "SNOWDP": "m", "SOILWATER_10CM": "kg/m2", "TBOT": "K",
    "TLAI": "1", "TOTECOSYSC": "gC/m^2", "TOTLITC": "gC/m^2", "TOTSOMC": "gC/m^2",
    "TOTVEGC": "gC/m^2", "TSA": "K", "TSOI_10CM": "K", "TWS": "mm", "ZWT": "m",
    "FIRE": "W/m^2", "FSDS": "W/m^2", "H2OSFC": "mm", "PLANT_NDEMAND": "gN/m^2/s",
    "PSNSHA": "umolCO2/m^2/s", "PSNSUN": "umolCO2/m^2/s", "QCHARGE": "mm/s",
    "QH2OSFC": "mm/s", "RH2M": "%"}
OTHER_MONTHLY = ("OTHER_A", "OTHER_B")      # monthly fields with no note, like the control's

# The assignment the trial tested, in the fixed order.
GROUP_FIELDS = (
    ("weather in", ("TBOT", "RAIN", "SNOW", "FSDS")),
    ("snow", ("H2OSNO", "SNOWDP", "FSNO", "QSNOMELT")),
    ("soil water and groundwater", ("SOILWATER_10CM", "H2OSOI", "ZWT", "TWS", "QCHARGE",
                                    "H2OSFC")),
    ("runoff", ("QOVER", "QDRAI", "QRUNOFF", "QH2OSFC")),
    ("water to the air", ("QVEGT", "QVEGE", "QSOIL", "EFLX_LH_TOT", "BTRAN")),
    ("energy", ("FSA", "FIRA", "FSH", "TSA", "TSOI_10CM", "RH2M")),
    ("carbon in and out", ("GPP", "NPP", "AR", "HR", "ER", "NEE", "PSNSUN", "PSNSHA")),
    ("leaves", ("TLAI", "ELAI", "LEAFC")),
    ("carbon stores", ("TOTVEGC", "FROOTC", "CWDC", "TOTLITC", "TOTSOMC", "TOTECOSYSC")),
    ("nitrogen", ("SMINN", "FPG", "FPI", "PLANT_NDEMAND")),
    ("fire", ("FAREA_BURNED", "COL_FIRE_CLOSS")),
)
ORDER = [n for _, fs in GROUP_FIELDS for n in sorted(fs)]


def _fill(path, names, rng):
    """Add fields to a fixture file, with random values and the control's units."""
    with netCDF4.Dataset(path, "a") as d:
        n = len(d.dimensions["time"])
        shape = (len(d.dimensions["lat"]), len(d.dimensions["lon"]))
        for name in names:
            layered = name == "H2OSOI"
            dims = ("time", "levgrnd", "lat", "lon") if layered else ("time", "lat", "lon")
            v = d.createVariable(name, "f4", dims, fill_value=fx.FILL)
            v.units = UNITS.get(name, "1")
            v.long_name = f"the long name of {name}"
            v.cell_methods = "time: mean"
            size = (n, fx.NLEV) + shape if layered else (n,) + shape
            v[:] = (rng.uniform(0.1, 10.0, size) * 1.2345678).astype(np.float32)


def _revalue(run_dir, seed):
    """The same run with other values: every field's numbers redrawn."""
    rng = np.random.default_rng(seed)
    for p in sorted(Path(run_dir).glob("*.nc")):
        with netCDF4.Dataset(p, "a") as d:
            for name, v in d.variables.items():
                if name in UNITS or name in OTHER_MONTHLY or name == "XTRA":
                    v[:] = (rng.uniform(0.1, 10.0, v.shape) * 7.654321).astype(np.float32)


def _polygon(tmp, g, rows=2):
    lat, lon = g["lat"], g["lon"] - 360.0
    d = (lat[1] - lat[0]) / 2
    ring = [[float(lon[0] - d), float(lat[0] - d)], [float(lon[-1] + d), float(lat[0] - d)],
            [float(lon[-1] + d), float(lat[rows - 1] + d)],
            [float(lon[0] - d), float(lat[rows - 1] + d)], [float(lon[0] - d), float(lat[0] - d)]]
    p = tmp / "box.geojson"
    p.write_text(json.dumps({"type": "Polygon", "coordinates": [ring]}))
    return p


def _run(root, seed=1, extra_daily=(), last_file=True):
    """A control: its case (lnd_in naming the surface), surface and history."""
    root = Path(root)
    g = fx.grid()
    case_dir, run_dir = root / "case", root / "run"
    (case_dir / "CaseDocs").mkdir(parents=True)
    surf = cf.write_surface(root / "surfdata.nc", g)
    (case_dir / "CaseDocs" / "lnd_in").write_text(
        cf.LND_IN.format(restart=root / "finidat.nc", surface=surf))
    rng = np.random.default_rng(seed)
    for p in fx.write_daily(run_dir, YEARS, case="synth", variables=(), last_file=last_file):
        _fill(p, DAILY + list(extra_daily), rng)
    for p in fx.write_monthly(run_dir, YEARS, case="synth", variables=()):
        _fill(p, DAILY + MONTHLY_ONLY + ["FIRE"] + list(OTHER_MONTHLY), rng)
    fx.write_daily(run_dir, YEARS, case="synth", stream="h2", steps_per_day=8,
                   variables=("TSA",))
    box = _polygon(root, g)
    return {"name": "synth", "case_dir": str(case_dir), "run_dir": str(run_dir),
            "regions": {"box": {"polygon": str(box),
                                "what": "the first two rows of cells, about 4 km2"}}}


def _study(root, normal=(2016,)):
    d = Path(root) / "study_one"
    (d / "01_inputs").mkdir(parents=True)
    change = {"id": "weather_scale", "params": {"temperature_K": 2.0},
              "summary": "warm the air the land model receives",
              "meaning": "the response to air 2 K warmer, every other input the control's",
              "not_meaning": "a consistent change of climate"}
    (d / "experiment.json").write_text(json.dumps({
        "kind": "whole_domain",
        "experiment": {"runs": [{"run": "control", "reused": True, "changes": []},
                                {"run": "warm", "changes": [change]}]},
        "chain": {"years_planned": YEARS, "years_passed": YEARS}}))
    (d / "01_inputs" / "domain_experiment.json").write_text(json.dumps({
        "runs": [{"name": "warm", "changes": [{"id": "weather_scale"}]}],
        "analysis": {"regions": ["box"], "variables": ["GPP"],
                     "normal_years": list(normal)}}))
    return d


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    root = tmp_path_factory.mktemp("catalog")
    control = _run(root)
    study = _study(root)
    cat = C.build(control, study_dir=study)
    return {"root": root, "control": control, "study": study, "cat": cat,
            "md": C.render(cat)}


# ── the notes ────────────────────────────────────────────────────────────────

def test_the_notes_hold_every_field_once_and_their_content_is_whole():
    assert len(ROWS) == 52 and len(DAILY) == 44 and len(MONTHLY_ONLY) == 8
    assert sorted(MONTHLY_ONLY) == sorted(["H2OSFC", "QH2OSFC", "QCHARGE", "PSNSUN", "PSNSHA",
                                           "RH2M", "FSDS", "PLANT_NDEMAND"])
    # every note names a field of the control's output (its units above)
    assert set(NOTES) == set(UNITS)
    assert C.check_notes(NOTES) == []
    for name, e in ROWS.items():
        assert e["meaning"].strip() and e["type"] in C.TYPES and e["scale"] in C.SCALES, name
        # THE USER'S APPROVAL (2026-11-05): every note approved for integration, by
        # the user, at its current versions (only the user records a review)
        want = {"state": C.APPROVED_STATE, "by": "the user", "date": "2026-11-05",
                "text_revision": e["text_revision"],
                "behaviour_version": e["behaviour_version"]}
        if name == "FPG":       # re-recorded at text revision 4 (closing package E)
            want["note"] = "re-recorded after completing the FPG correction the user asked for"
        assert e["status"]["review"] == want, name
        # the notes' how is the units' own rule: no override is needed today
        assert e["how"] == C.unit_rule(name, UNITS[name]), name
    assert {n for n, e in ROWS.items() if e["type"] == "event"} == {"FAREA_BURNED",
                                                                     "COL_FIRE_CLOSS"}
    assert {n for n, e in ROWS.items() if e["type"] == "bounded"} == {"BTRAN", "FPG", "FPI",
                                                                       "FSNO"}
    assert {n for n, e in ROWS.items() if e["type"] == "store"} == \
        set(dict(GROUP_FIELDS)["carbon stores"]) | {"SMINN"}
    for n in ("TBOT", "TSA", "TSOI_10CM", "TWS"):
        assert ROWS[n]["scale"] == "interval", n
    # REVIEW ROUND 1 (2026-11-05): the scale is physical, so a signed flux with
    # a zero that means no net flow is ratio-scale; whether a percent is given
    # is the separate percent policy
    for n in ("NPP", "NEE", "FSH", "FIRA", "QVEGE", "QSOIL", "EFLX_LH_TOT", "QCHARGE"):
        assert ROWS[n]["scale"] == "ratio" and ROWS[n]["percent"]["allowed"] is False, n


def test_the_percent_policy_follows_the_written_defaults():
    no = {"TBOT", "TSA", "TSOI_10CM", "TWS", "ZWT", "NPP", "NEE", "FSH", "FIRA", "QVEGE",
          "QSOIL", "EFLX_LH_TOT", "QCHARGE", "RH2M"} | \
        {n for n, e in ROWS.items() if e["type"] in ("event", "bounded")}
    for name, e in ROWS.items():
        assert e["percent"]["allowed"] is (name not in no), name
        assert e["percent"]["why"].strip(), name
    assert "percentage points" in ROWS["RH2M"]["percent"]["why"]
    assert any("positive down" in c for c in ROWS["ZWT"]["cautions"])
    assert "type_why" in ROWS["LEAFC"] and ROWS["LEAFC"]["type"] == "state"
    assert any("same time step" in c for c in ROWS["PSNSUN"]["cautions"])
    assert any("same time step" in c for c in ROWS["PSNSHA"]["cautions"])
    fpg = " ".join(ROWS["FPG"]["cautions"])
    for words in ("verified configuration", "suplphos = 'ALL'", "suplnitro = 'NONE'",
                  "only nitrogen", "BTRAN",
                  # STEP 0 (2026-11-05), CORRECTED WHEN CLOSING PACKAGE E: nitrogen
                  # only; FPG_P exactly 1 in the run where it was checked, so the two
                  # cannot co-limit; both needed without the supplement; FPG shows
                  # the nitrogen limit at the time, not nitrogen's earlier effects
                  "nitrogen limitation only", "FPG_P", "was exactly 1 in every cell and "
                  "month of the one control run where this was checked", "160,728 finite "
                  "cell-months a year", "cannot co-limit", "without the phosphorus "
                  "supplement", "needed together", "rules out only a same-time nitrogen "
                  "limit"):
        assert words in fpg, words
    assert "stays at 1" not in fpg and "nothing else" not in fpg
    assert NOTES["FPG"]["text_revision"] == 4 and NOTES["FPG"]["behaviour_version"] == 1
    assert NOTES["FPG"]["status"]["review"]["note"] == \
        "re-recorded after completing the FPG correction the user asked for"
    # the statuses, four and kept apart, each tied to the entry's versions
    for name, e in NOTES.items():
        assert C.note_status_problems(name, e) == [], name
        # PACKAGE E (2026-11-05): every catalog row is reachable through the suite
        assert e["status"]["available_to_analyzer"] is C.is_row(e), name
    assert ROWS["RH2M"]["behaviour_version"] == 2 and ROWS["GPP"]["behaviour_version"] == 1


def test_the_notes_carry_the_two_field_notes_of_gridded_verbatim():
    assert G.FIELD_NOTES["FAREA_BURNED"] in NOTES["FAREA_BURNED"]["cautions"]
    fire = NOTES["FIRE"]
    assert fire["cautions"] == [G.FIELD_NOTES["FIRE"]]
    assert fire["row"] is False and fire["cadence"] == "monthly" and not C.is_row(fire)


def test_the_cautions_name_no_watershed_and_no_dash():
    # the notes' words, not their test records (which name test files)
    text = json.dumps({n: {k: v for k, v in e.items() if k != "status"} for n, e in NOTES.items()},
                      ensure_ascii=False)
    assert "\u2014" not in text and "\u2013" not in text
    for word in ("Naches", "naches", "Yakima", "Washington", "2015", "2021"):
        assert word not in text, word


def test_the_groups_are_the_trial_s_in_the_fixed_order():
    assert C.GROUPS == tuple(g for g, _ in GROUP_FIELDS)
    for g, fs in GROUP_FIELDS:
        assert sorted(n for n, e in ROWS.items() if e["group"] == g) == sorted(fs), g


@pytest.mark.review_gate
def test_no_field_note_is_still_a_draft():
    # passes on approved for integration, provisional or reviewed at the
    # note's current versions; a draft or a review of an older version fails
    drafts = sorted(n for n, e in NOTES.items() if not C.is_approved(
        e.get("status"), e.get("text_revision"), e.get("behaviour_version")))
    assert not drafts, f"{len(drafts)} field note(s) still a draft: {', '.join(drafts)}"


def test_the_note_gate_fails_on_a_draft_or_an_older_version():
    # THE USER'S APPROVAL (2026-11-05): what the gate above accepts and refuses
    e = json.loads(json.dumps(NOTES["GPP"]))
    tr, bv = e["text_revision"], e["behaviour_version"]
    for state in (C.APPROVED_STATE, C.PROVISIONAL_STATE, C.REVIEWED_STATE):
        e["status"]["review"]["state"] = state
        assert C.is_approved(e["status"], tr, bv), state
        assert not C.is_approved(e["status"], tr + 1, bv), state     # an older text
        assert not C.is_approved(e["status"], tr, bv + 1), state     # an older behaviour
    e["status"]["review"]["state"] = C.DRAFT_STATE
    assert not C.is_approved(e["status"], tr, bv)
    # an approval names who made it and when
    e["status"]["review"] = {"state": C.APPROVED_STATE, "text_revision": tr,
                             "behaviour_version": bv}
    assert any("names who made it" in p for p in C.status_problems(e["status"], tr, bv))


# ── the catalog ──────────────────────────────────────────────────────────────

def test_every_field_is_a_row_with_a_meaning_a_type_and_a_scale(fake):
    cat = fake["cat"]
    assert cat["problems"] == []
    assert [r["name"] for r in cat["fields"]] == ORDER
    for r in cat["fields"]:
        assert r["meaning"] and r["type"] in C.TYPES and r["scale"] in C.SCALES, r["name"]
        assert r["cadence"] == ("monthly" if r["name"] in MONTHLY_ONLY else "daily")
        assert r["units"] == UNITS[r["name"]]
        assert r["long_name"] == f"the long name of {r['name']}"
    h2osoi = next(r for r in cat["fields"] if r["name"] == "H2OSOI")
    assert h2osoi["levels"] == [{"dim": "levgrnd", "size": fx.NLEV}]
    gpp = next(r for r in cat["fields"] if r["name"] == "GPP")
    assert gpp["how"] == "total" and gpp["amount_units"] == "gC/m^2"
    burned = next(r for r in cat["fields"] if r["name"] == "FAREA_BURNED")
    assert burned["how"] == "total" and burned["amount_units"] == "proportion"


def test_the_markdown_keeps_the_group_order_and_alphabetical_rows(fake):
    md = fake["md"]
    heads = re.findall(r"^### (.+)$", md, flags=re.M)
    assert heads == list(C.GROUPS)
    names = re.findall(r"^\| ([A-Z][A-Z0-9_]+) \|", md, flags=re.M)
    assert names == ORDER
    assert md.index("## 1. Runs") < md.index("## 2. Regions") < md.index("## 3. Periods") \
        < md.index("## 4. Fields") < md.index("## 5. Methods") < md.index("## 6. How to load") \
        < md.index("## 7. Other output")


def test_the_catalog_is_within_its_length_and_refuses_to_pass_it(fake, monkeypatch):
    assert len(fake["md"]) <= C.MAX_CHARS == 25000
    monkeypatch.setattr(C, "MAX_CHARS", 1000)
    with pytest.raises(C.CatalogError, match="over the"):
        C.render(fake["cat"])


def test_the_catalog_regenerates_identically(fake):
    again = C.build(fake["control"], study_dir=fake["study"])
    assert json.dumps(again, sort_keys=True) == json.dumps(fake["cat"], sort_keys=True)
    assert C.render(again) == fake["md"]
    out1, out2 = fake["root"] / "out1", fake["root"] / "out2"
    p1, p2 = C.write(fake["cat"], out1), C.write(again, out2)
    assert p1["json"].read_bytes() == p2["json"].read_bytes()
    assert p1["md"].read_bytes() == p2["md"].read_bytes()


def test_no_number_in_the_catalog_comes_from_the_values(fake):
    # The same run with every value redrawn: a catalog that printed an
    # outcome (a mean, a total, a difference, an extreme) would change.
    other = fake["root"] / "other"
    shutil.copytree(fake["root"] / "run", other / "run")
    _revalue(other / "run", seed=99)
    control = dict(fake["control"], run_dir=str(other / "run"))
    cat = C.build(control, study_dir=fake["study"])
    assert C.render(cat) == fake["md"]
    a = dict(cat, control=None)
    b = dict(fake["cat"], control=None)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_every_number_in_the_catalog_is_a_count_a_year_an_area_or_written_text(fake):
    cat, md = fake["cat"], fake["md"]
    tokens = lambda s: set(re.findall(r"\d+(?:[.,]\d+)*", s))   # noqa: E731
    allowed_text = " ".join([
        json.dumps(NOTES), json.dumps(fake["control"]["regions"]),
        (fake["study"] / "experiment.json").read_text(),
        (fake["study"] / "01_inputs" / "domain_experiment.json").read_text(),
        json.dumps(UNITS), json.dumps(cat["streams"]), json.dumps(cat["runs"]),
        json.dumps(cat["other_output"]), json.dumps(cat["notes"]["entries"]),
        json.dumps(cat["methods"]),
        json.dumps([[r["cells"], r["rule"]] for r in cat["regions"]["regions"]]),
        " ".join(f"{r['area_km2']:,.1f}" for r in cat["regions"]["regions"]),
        "1 2 3 4 5 6 7 24", str(len(cat["fields"])), str(len(C.GROUPS)), "0"])
    stray = tokens(md) - tokens(allowed_text)
    assert not stray, f"numbers in CATALOG.md from nowhere allowed: {sorted(stray)}"


def test_the_runs_regions_and_periods_are_the_files_and_the_records(fake):
    cat, md = fake["cat"], fake["md"]
    assert [r["run"] for r in cat["runs"]] == ["control", "warm"]
    assert cat["runs"][0]["years"] == YEARS
    assert "the response to air 2 K warmer" in md and "Does not mean: a consistent" in md
    assert "Normal years, the plan's definition: 2016." in md
    # regions as domain_analysis.masks makes them
    grid = G.open_stream(fake["control"]["run_dir"], "h1").grid
    reg = fake["control"]["regions"]
    ms = A.masks(grid, A._surface(fake["control"]["case_dir"]),
                 [A.VEGETATED, "box", A.BANDS], reg)["masks"]
    want = [(n, m["n_cells"], m["area_km2"]) for n, m in ms.summary().items()]
    got = [(r["name"], r["cells"], r["area_km2"]) for r in cat["regions"]["regions"]]
    assert got == want and got[1][0] == "box, vegetated"
    assert "about 4 km2" in md
    daily = cat["streams"]["h1"]
    assert daily["cadence"] == "daily" and daily["complete_years"] == YEARS
    assert daily["partial_years"] == {} and daily["records"] == 365 * len(YEARS)
    assert "Daily stream h1: 2015-01-01 to 2016-12-31, 730 records in 3 files" in md


def test_parts_not_built_say_so_and_name_nothing(fake):
    cat, md = fake["cat"], fake["md"]
    assert all(r["products"] == C.SUITE_NOT_BUILT for r in cat["fields"])
    assert md.count(f"| {C.SUITE_NOT_BUILT} |") == len(cat["fields"]) + len(cat["runs"])
    body = lambda head: md.split(head)[1].split("\n## ")[0]    # noqa: E731
    # PACKAGE E (2026-11-05): section 6 is generated from the access code
    import domain_access as DAC
    calls = DAC.how_to_load()
    assert body("## 6. How to load").strip() == "\n".join(C._access_lines(calls)).strip()
    for name in ("get_map", "get_series", "get_timing", "region_mask", "cell_area",
                 "find_fields", "anomaly", "difference", "screen", "overlap",
                 "elevation_split", "MEASURE"):
        assert any(c["call"].startswith(f"suite.{name}(") for c in calls), name
        assert hasattr(DAC.SuiteAccess, name) or name == "MEASURE", name
    assert "load_series" not in md
    other = body("## 7. Other output")
    # the monthly stream's other fields as a count only: DOYCODE, FIRE, OTHER_A, OTHER_B
    assert "holds 4 other fields" in other
    assert "holds 2 fields every 3 hours" in other
    assert f"H2OSOI holds {fx.NLEV} single layers (levgrnd)" in other
    for name in ("DOYCODE", "OTHER_A", "STEPCODE"):
        assert not re.search(rf"\b{name}\b", md), name
    # FIRE is named once, in the count of the field notes, with its caution only
    assert len(re.findall(r"\bFIRE\b", md)) == 1
    assert "plus FIRE, kept only for its caution" in md


def test_the_methods_that_exist_are_listed_with_their_statuses(fake):
    # REVIEW ROUND 1 (2026-11-05): section 5 lists the methods that exist,
    # pointing to docs/ANALYSIS_METHODS.md, with the four statuses kept apart
    import domain_methods as M
    cat, md = fake["cat"], fake["md"]
    body = md.split("## 5. Methods")[1].split("\n## ")[0]
    assert "docs/ANALYSIS_METHODS.md" in body and "not built yet" not in body
    assert [m["id"] for m in cat["methods"]["methods"]] == list(M.METHODS)
    for mid, m in M.METHODS.items():
        assert f"| {mid} | {m.version} | approved for integration at text revision " \
               f"{m.text_revision}, version " \
               f"{m.version} | version {m.version} | version {m.version}, on " \
               f"{m.status['tested']['date']} | yes |" in body, mid
    notes = md.split("## 4. Fields")[1].split("\n### ")[0]
    for part in ("- review: 53 approved for integration (by the user on 2026-11-05), an "
                 "approval for integration, not a final scientific approval; 0 drafts or "
                 "reviews of an older version", "- implemented: 53 at their behaviour version",
                 "- tested: 53 at their behaviour version", "- available to the Analyzer: 52"):
        assert part in notes, part
    assert "PCT_NATVEG >= 100" in md.split("## 2. Regions")[1].split("\n## ")[0]
    assert "| field | meaning | how | type | scale | percent |" in md
    assert "RH2M: already a percentage" in md.split("No percent change, by the field notes' "
                                                   "policy: ")[1].split("\n")[0]
    # no facts cache for this control: its configuration is said to be unknown
    assert cat["configuration"] is None and "its facts cache records none" in md


def test_the_controls_verified_configuration_is_read_from_its_facts_cache(fake, tmp_path):
    reg = tmp_path / "registry" / "domain_controls.json"
    (reg.parent / C.FACTS_DIR).mkdir(parents=True)
    facts = {"configuration": {"compset": "X", "carbon_nutrient": {
                 "nu_com": "RD", "build_flags": {"nutrient": "cnp"}},
                 "supplements": {"nitrogen": "NONE", "phosphorus": "ALL"}},
             "fire": {"on": True, "use_nofire": False, "prescribed_burned_area": []}}
    (reg.parent / C.FACTS_DIR / "synth.json").write_text(json.dumps(facts))
    control = dict(fake["control"], registry=str(reg))
    cfg = C.control_configuration(control)
    assert (cfg["nutrient"], cfg["nitrogen_supplement"], cfg["phosphorus_supplement"]) == \
        ("cnp", "NONE", "ALL")
    text = "\n".join(C._configuration_lines(cfg))
    assert "only nitrogen can limit growth by nutrients" in text
    assert "FPG_P, in the monthly output, was exactly 1 in every cell and month of " \
        "2015-2023 in the Naches control run" in text and "cannot co-limit" in text
    assert "FPG measures only that nitrogen limitation" in text and "nothing else" not in text
    assert "the model's fire is the only fire in the control" in text
    facts["configuration"]["supplements"]["phosphorus"] = "NONE"
    (reg.parent / C.FACTS_DIR / "synth.json").write_text(json.dumps(facts))
    text = "\n".join(C._configuration_lines(C.control_configuration(control)))
    assert "only nitrogen" not in text and "on its own" in text


def test_a_suite_manifest_gives_products_and_its_version(fake, tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / C.SUITE_FILE).write_text(json.dumps({
        "SUITE_VERSION": 3,
        "fields": {"GPP": {"products": ["maps_year", "series_day"]}},
        "fields_left_out": {"H2OSOI": "layered: the suite holds its top-1 m mean only"},
        "periods": {"year": ["2015", "2016"]},
        "periods_absent": {"wateryear": {"WY2015": "it needs October-December 2014"}},
        "regions": [{"name": "vegetated cells", "cells": 7, "area_km2": 4.5, "rule": "r1"},
                    {"name": "box, vegetated", "cells": 3, "area_km2": 1.5, "rule": "r2"},
                    {"name": "box, vegetated, below 1000 m", "cells": 2, "area_km2": 1.0,
                     "rule": "r3"}]}))
    cat = C.build(fake["control"], study_dir=fake["study"], suite_dir=suite)
    rows = {r["name"]: r for r in cat["fields"]}
    assert rows["GPP"]["products"] == ["maps_year", "series_day"]
    assert rows["H2OSOI"]["products"].startswith("left out of the suite: layered")
    assert rows["TSA"]["products"] == "the suite's manifest lists no product for it"
    assert cat["runs"][0]["suite"] == "suite version 3"
    md = C.render(cat)
    assert "| maps_year, series_day |" in md and "Suite periods, year: 2 (2015 to 2016)." in md
    assert "Suite periods left out, wateryear WY2015: it needs October-December 2014." in md
    # THE SUITE'S REGIONS (2026-11-04): the manifest's, bands within a region among them
    got = [(r["name"], r["cells"], r["area_km2"], r["rule"]) for r in cat["regions"]["regions"]]
    assert got == [("vegetated cells", 7, 4.5, "r1"), ("box, vegetated", 3, 1.5, "r2"),
                   ("box, vegetated, below 1000 m", 2, 1.0, "r3")]
    assert cat["regions"]["regions"][1]["what"] == fake["control"]["regions"]["box"]["what"]
    assert "| box, vegetated, below 1000 m | 2 | 1.0 | r3 |" in md


def test_a_note_without_a_field_and_a_field_without_a_note_are_problems(fake, tmp_path):
    notes = json.loads(C.NOTES_PATH.read_text())
    notes["fields"]["NOT_A_FIELD"] = dict(notes["fields"]["GPP"])
    notes["fields"]["TSA"] = dict(notes["fields"]["TSA"], how="total")
    p = tmp_path / "notes.json"
    p.write_text(json.dumps(notes))
    root = tmp_path / "x"
    control = _run(root, extra_daily=("XTRA",), last_file=False)
    cat = C.build(control, notes_path=p)
    # without the daily stream's last file, 2016 lacks its 31 December
    assert cat["streams"]["h1"]["partial_years"] == {
        "2016": {"records": 364, "missing": 1, "first_missing": "2016-12-31"}}
    assert cat["runs"][0]["years"] == [2015]
    probs = "\n".join(cat["problems"])
    assert "NOT_A_FIELD: the field notes name it, and neither" in probs
    assert "XTRA: the daily stream h1 holds it and the field notes have no entry" in probs
    assert "TSA: the notes say total, its units (K) say average" in probs
    rows = {r["name"]: r for r in cat["fields"]}
    assert rows["XTRA"]["group"] == C.NO_NOTE_GROUP and rows["TSA"]["how"] == "average"
    md = C.render(cat)
    assert md.index("### fire") < md.index(f"### {C.NO_NOTE_GROUP}") < md.index("## Problems")
    assert "No study: this is the control's catalog." in md
