"""docs/ANALYSIS_METHODS.md must be what tools/make_methods_doc.py writes now,
and the method catalog must name only what exists.

    pytest tests/test_methods_doc.py -v
    pytest tests/test_methods_doc.py --review-gate -v   # + fails while an entry is a draft

THE METHOD CATALOG IS GENERATED, SO IT CAN BE CHECKED (2026-11-05). The
Analyzer reads the file before it plans; a method changed in the code and not
in the file would be a false capability claim, so a stale file fails here
with the command that refreshes it, as tests/test_architecture_doc.py does
for docs/ARCHITECTURE.md. Every field and method a contract or a mechanism
names must exist (the catalog's rows, the module's METHODS), and every test a
contract cites as its validation evidence must be defined where it says.
"""
import ast
import importlib.util
import json
from pathlib import Path

import pytest

import domain_catalog as C
import domain_methods as M

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "tools" / "make_methods_doc.py"
DOC = ROOT / "docs" / "ANALYSIS_METHODS.md"
REFRESH = "python tools/make_methods_doc.py"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("make_methods_doc", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def text(gen):
    return gen.build()


def test_committed_document_is_current(text):
    assert DOC.is_file(), f"{DOC.relative_to(ROOT)} is missing. Write it with:\n    {REFRESH}"
    assert DOC.read_text() == text, (
        f"{DOC.relative_to(ROOT)} is out of date with the method catalog.\n"
        f"Refresh it with:\n    {REFRESH}\nand commit the result.")


def test_check_flag_reports_a_stale_copy(gen, tmp_path, text):
    stale = tmp_path / "ANALYSIS_METHODS.md"
    stale.write_text(text.replace("## 1.", "## one.", 1))
    assert gen.main(["--check", "--out", str(stale)]) == 1
    stale.write_text(text)
    assert gen.main(["--check", "--out", str(stale)]) == 0


def test_no_dash_used_as_a_pause(text):
    assert "\u2014" not in text and "\u2013" not in text


def test_every_method_and_mechanism_is_rendered(text):
    for mid in M.METHODS:
        assert f"`{mid}` (version" in text, mid
    for e in M.load_mechanisms():
        assert f"### `{e['id']}`" in text, e["id"]
    assert "never rules" in text


def test_every_field_and_method_named_exists():
    bad = []
    for m in M.METHODS.values():
        bad += M.check_names(m.fields, (), f"method {m.id}")
    for e in M.load_mechanisms():
        names = M.mechanism_fields(e) + [f for c in e["competing"] for f in c["fields"]]
        bad += M.check_names(names, e["methods"], f"mechanism {e['id']}")
        assert e["links"] and e["expected_when"] and e["competing"] and e["limits"] and \
            e["methods"], e["id"]
    assert not bad, "\n".join(bad)


def test_every_mechanism_is_links_with_conditions():
    # REVIEW ROUND 1 (2026-11-05): explicit links that may branch, each with
    # its kind and the conditions it holds under; no single ordered list
    kinds = ("direct", "conditional carry-over")
    for e in M.load_mechanisms():
        assert "variables" not in e, e["id"]
        for k in e["links"]:
            assert set(k) == {"from", "to", "kind", "how", "conditions"}, (e["id"], k)
            assert k["kind"] in kinds and k["how"].strip(), (e["id"], k)
            assert k["conditions"] and all(c.strip() for c in k["conditions"]), (e["id"], k)
    snow = {e["id"]: e for e in M.load_mechanisms()}["snow_water_to_next_year_leaves"]
    assert sorted(k["to"] for k in snow["links"] if k["from"] == "BTRAN") == ["GPP", "QVEGT"]
    carry = [k for k in snow["links"] if (k["from"], k["to"]) == ("GPP", "LEAFC")]
    assert len(carry) == 1 and carry[0]["kind"] == "conditional carry-over"
    text = " ".join(carry[0]["conditions"])
    for word in ("allocat", "stor", "turn over", "vegetation type", "disturbance"):
        assert word in text, word
    mech = {e["id"]: json.dumps(e) for e in M.load_mechanisms()}
    assert "suplphos = 'ALL'" in mech["nitrogen_to_gpp"] and \
        "suplnitro = 'NONE'" in mech["nitrogen_to_gpp"]
    # STEP 0 (2026-11-05), CORRECTED WHEN CLOSING PACKAGE E: the supplement
    # switches phosphorus limitation off (FPG_P exactly 1 in the one run where
    # it was checked, said only of that run), without it FPG and FPG_P are
    # needed together, and FPG shows the nitrogen limit at the time only
    for words in ("FPG_P", "was exactly 1 in every cell and month of 2015-2023 in the Naches "
                  "control run", "160,728 finite cell-months a year", "cannot co-limit",
                  "needed together", "shows in neither FPG nor FPG_P",
                  "FPG measures only the nitrogen limitation of potential GPP",
                  "rules out only a same-time nitrogen limit"):
        assert words in mech["nitrogen_to_gpp"], words
    for gone in ("stays at 1", "nothing else", "is not a nitrogen effect"):
        assert gone not in mech["nitrogen_to_gpp"], gone
    assert "whatever the screen marks" in mech["burned_area_to_transpiration"]
    assert "background fire only" not in mech["burned_area_to_transpiration"]


def test_every_entry_has_four_statuses_tied_to_its_versions():
    # REVIEW ROUND 1 (2026-11-05): review, implemented, tested and
    # available_to_analyzer kept apart, each tied to the version it refers to;
    # the tests named as passed are tests that exist, and a method's are among
    # its validation evidence
    defined = {}

    def exists(tid):
        path, name = tid.split("::")
        if path not in defined:
            defined[path] = {n.name for n in ast.walk(ast.parse((ROOT / path).read_text()))
                             if isinstance(n, ast.FunctionDef)}
        return name in defined[path]
    bad = []
    for m in M.METHODS.values():
        bad += C.status_problems(m.status, m.text_revision, m.version, f"method {m.id}")
        cited = {t for t, _ in m.validation}
        bad += [f"method {m.id}: {t} is not among its validation evidence"
                for t in m.status["tested"]["passed"] if t not in cited]
        # PACKAGE E (2026-11-05): available where the access path reaches it,
        # each naming the tests that exercise it through that path
        assert m.status["available_to_analyzer"] is (m.id in M.ANALYZER_ACCESS), m.id
        assert all(exists(t) for t in M.ANALYZER_ACCESS[m.id]), m.id
        # THE USER'S APPROVAL (2026-11-05), recorded at the current versions;
        # the six timing measures' re-recorded on 2026-11-08 for the G part 1
        # fixes the user authorized (decisions 1 and 2), at versions 3
        if m.id in M.TIMING_IDS:
            assert (m.text_revision, m.version) == (3, 3), m.id
            assert m.status["review"] == {
                "state": M.APPROVED, "by": "the user", "date": "2026-11-08",
                "note": "re-recorded for the G part 1 fixes the user authorized on 2026-11-08 "
                        "(decisions 1 and 2)",
                "text_revision": m.text_revision, "behaviour_version": m.version}, m.id
            continue
        assert m.status["review"] == {"state": M.APPROVED, "by": "the user",
                                      "date": "2026-11-05", "text_revision": m.text_revision,
                                      "behaviour_version": m.version}, m.id
    for e in M.load_mechanisms():
        bad += C.status_problems(e.get("status"), e.get("text_revision"),
                                 e.get("behaviour_version"), f"mechanism {e['id']}")
        assert e["status"]["available_to_analyzer"] is True, e["id"]     # the compact view
    passed = [t for m in M.METHODS.values() for t in m.status["tested"]["passed"]] + \
        [t for e in M.load_mechanisms() for t in e["status"]["tested"]["passed"]]
    bad += [f"{t} is not defined" for t in passed if not exists(t)]
    assert not bad, "\n".join(bad)
    # a review of an older version does not count, and a stale test record is a problem
    st = json.loads(json.dumps(M.METHODS["anomaly"].status))
    st["review"]["state"] = "reviewed"
    assert C.is_reviewed(st, M.METHODS["anomaly"].text_revision, M.METHODS["anomaly"].version)
    assert not C.is_reviewed(st, M.METHODS["anomaly"].text_revision + 1,
                             M.METHODS["anomaly"].version)
    assert C.status_problems(st, 2, M.METHODS["anomaly"].version + 1, "x")


def test_the_check_fails_on_a_name_that_does_not_exist():
    assert M.check_names(["GPP", "NOT_A_FIELD"], ["anomaly", "no_such_method"], "x") == [
        "x: no field 'NOT_A_FIELD'", "x: no method 'no_such_method'"]


def test_every_cited_test_is_defined():
    defined = {}
    missing = []
    for m in M.METHODS.values():
        assert m.validation, m.id
        for t, data in m.validation:
            path, name = t.split("::")
            if path not in defined:
                p = ROOT / path
                assert p.is_file(), f"{m.id} cites {path}, which does not exist"
                defined[path] = {n.name for n in ast.walk(ast.parse(p.read_text()))
                                 if isinstance(n, ast.FunctionDef)}
            if name not in defined[path]:
                missing.append(f"{m.id}: {t}")
            assert data and not any(ch.isdigit() for ch in data.replace("2015-2023", "")), \
                f"{m.id}: the evidence names data, never a value: {data!r}"
    assert not missing, "\n".join(missing)


def test_every_contract_is_complete_and_checks_are_callables():
    for m in M.METHODS.values():
        assert m.estimates and m.inputs and m.applicability and m.outcomes and m.outputs and \
            m.limitations, m.id
        assert m.version >= 1 and m.text_revision >= 1
        assert [k for k, _ in m.numerics] == ["units", "area weights", "precision", "tolerances",
                                              "near zero"], m.id
        assert m.run is not None and m.run.method is m
        for c in m.applicability:
            assert callable(c.test), f"{m.id}.{c.id}"
        for s in m.settings:
            assert s.reason and s.status in C.STATUSES, f"{m.id}.{s.name}"
            # a setting carries its method's approval; the two-peak settings
            # of growth_peak only as provisional values
            two_peak = (m.id, s.name) in (("growth_peak", "near"), ("growth_peak", "dip"))
            assert s.status == (M.PROVISIONAL if two_peak else M.APPROVED), f"{m.id}.{s.name}"
            assert (s.review["by"], s.review["date"], s.review["text_revision"],
                    s.review["behaviour_version"]) == (
                        "the user", "2026-11-08" if m.id in M.TIMING_IDS else "2026-11-05",
                        m.text_revision, m.version)
            assert s.default not in s.alternatives, f"{m.id}.{s.name}"


@pytest.mark.review_gate
def test_no_method_contract_or_mechanism_is_still_a_draft(gen):
    left = gen.drafts()
    assert not left, f"{len(left)} entries still a draft: {', '.join(left)}"


def test_the_gate_fails_on_a_draft_or_an_older_version(gen):
    # THE USER'S APPROVAL (2026-11-05): the gate passes on approved for
    # integration and provisional at the current versions, and still fails on
    # a draft or on a review that names an older version
    import dataclasses
    mech = M.load_mechanisms()
    assert gen.drafts() == []
    m = M.METHODS["growth_peak"]
    old = dict(m.status, review={**m.status["review"], "text_revision": m.text_revision - 1})
    draft = dict(m.status, review={**m.status["review"], "state": M.DRAFT})
    for st in (old, draft):
        assert gen.drafts({"growth_peak": dataclasses.replace(m, status=st)}, mech) == \
            ["method growth_peak"]
    s = m.setting("near")
    assert s.status == M.PROVISIONAL
    for rv in ({**s.review, "behaviour_version": m.version - 1}, {"state": M.DRAFT}):
        alt = dataclasses.replace(m, settings=tuple(
            dataclasses.replace(x, review=rv) if x.name == "near" else x for x in m.settings))
        assert gen.drafts({"growth_peak": alt}, mech) == ["setting growth_peak.near"]
    e = json.loads(json.dumps(mech[0]))
    e["status"]["review"]["text_revision"] = e["text_revision"] - 1
    assert gen.drafts({}, [e]) == [f"mechanism {e['id']}"]


def test_the_document_says_the_approval_is_not_final(text):
    head = text.split("## Outcomes")[0]
    assert "not a final scientific approval" in head and "by the user on 2026-11-05" in head
    assert "`growth_peak.near` and `growth_peak.dip` approved only as provisional" in head
    assert "Drafts or reviews of an older version: 0." in head
    gp = text.split("`growth_peak` (version")[1].split("\n## ")[0]
    assert gp.count("| **provisional**: approved for integration as a provisional value") == 2
    assert text.count("not a final scientific approval") >= len(M.METHODS) + 1


def test_the_compact_view_is_current_within_its_limit_and_whole(gen):
    # PACKAGE E (2026-11-05): the Analyzer's view of the methods, made from the
    # module; each method's quantity, applicability, settings and limitations
    text = gen.build_compact()
    p = ROOT / gen.COMPACT_OUT
    assert p.is_file() and p.read_text() == text, f"stale: run {REFRESH}"
    assert len(text) <= gen.COMPACT_LIMIT == 25_000
    assert gen.main(["--check"]) == 0
    for m in M.METHODS.values():
        assert f"### `{m.id}` (version {m.version}" in text and m.estimates in text, m.id
        assert all(c.text in text for c in m.applicability), m.id
        assert all(x in text for x in m.limitations), m.id
        assert all(f"`{s.name}` = " in text for s in m.settings), m.id
    for e in M.load_mechanisms():
        assert f"### `{e['id']}`" in text and e["status"]["available_to_analyzer"] is True
    assert text.count("PROVISIONAL") == 2 and "not a final scientific approval" in text
    assert "\u2014" not in text and "\u2013" not in text
