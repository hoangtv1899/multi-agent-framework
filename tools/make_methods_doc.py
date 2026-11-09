#!/usr/bin/env python3
"""docs/ANALYSIS_METHODS.md, generated from the method catalog.

    python tools/make_methods_doc.py            # rewrite the document
    python tools/make_methods_doc.py --check    # exit 1 when it is stale

GENERATED, NOT WRITTEN (2026-11-05). The Analyzer reads this file before it
plans, so it must say exactly what the code does: every method's contract is
read from its Method record in mcp/elm-mcp/src/domain_methods.py (what it
estimates, inputs, applicability, settings with their reasons and the
alternatives computed beside them, outcomes, outputs, limitations and the
tests that hold it), and the candidate mechanisms from
mcp/elm-mcp/domain_mechanisms.json. Nothing is kept here by hand except a few
lines of orientation, so the file cannot name a method that no longer exists.
tests/test_methods_doc.py regenerates it and fails when the committed copy
differs. No value measured on any study appears: validation evidence names
tests and data, never their results.

REVIEW ROUND 1 (2026-11-05). Each method shows its numerical contract and,
like each mechanism, its four statuses kept apart (review, implemented,
tested, available to the Analyzer; domain_catalog.status_lines). A mechanism
is rendered as its explicit links, each with its kind and its conditions,
since the links may branch.

THE USER'S APPROVAL (2026-11-05). Every method, setting and mechanism is
approved for integration by the user, growth_peak's two-peak settings as
provisional values; the document says, wherever it shows a review, that this
is not a final scientific approval. drafts() is what the --review-gate fails
on: a draft, or a review that names an older version.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("docs") / "ANALYSIS_METHODS.md"
SRC = ROOT / "mcp" / "elm-mcp" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import domain_catalog as C  # noqa: E402
import domain_methods as M  # noqa: E402

DASHES = ("\u2014", "\u2013")

ORIENTATION = (
    "Each method below is a function in `mcp/elm-mcp/src/domain_methods.py` carrying its "
    "contract. Code decides, per cell and period, whether a method applies; where it does not, "
    "the outcome is a reason with a count of the cells and periods it covers, never a filled-in "
    "value. A setting other than the default runs only when the study's recorded plan names it "
    "(or an amendment, whose results count as added after seeing the data); the alternatives a "
    "contract names are computed with every result, beside the default. Every result carries "
    "the method's id, version and settings.",
    "Picking a method is a choice of what to look at; a method's existence is no reason to "
    "report a date or a rank. Sign, timing and overlap support a mechanism and do not "
    "establish it.",
    "Every method and mechanism carries four statuses, kept apart: review (a draft until the "
    "user reviews it; then approved for integration, provisional for a value approved only "
    "as provisional, or reviewed for a final review; each at the text revision and behaviour "
    "version named, and counting only at the current ones), implemented (the "
    "behaviour version the code runs), tested (the tests that passed, their date and the "
    "behaviour version they ran on) and available to the Analyzer (yes where the suite's "
    "access calls reach it and a test exercises that path). A text revision counts changes to an entry's words; a behaviour version "
    "counts changes to what its code computes.")


_value = M.format_value


def _method(i: int, m: "M.Method") -> list:
    """A method's section: domain_methods.contract_lines, the same text the
    Analyzer's suite.method_contract call returns (closing package E)."""
    return M.contract_lines(i, m)


def _mechanism(e: dict) -> list:
    L = [f"### `{e['id']}` (text revision {e.get('text_revision')})", "", f"{e['name']}.", "",
         "**Fields:** " + ", ".join(f"`{v}`" for v in M.mechanism_fields(e)) + ".", "",
         "**Links** (explicit, one step each; a field can act on more than one other, and each "
         "link holds only under its conditions):", ""]
    for k in e["links"]:
        kind = "" if k.get("kind") == "direct" else f" ({k.get('kind')})"
        L.append(f"* `{k['from']}` to `{k['to']}`{kind}: {k['how']}. Holds when: "
                 + "; ".join(k.get("conditions") or ["no condition written"]) + ".")
    L += ["", "**Expected when:**", ""] + [f"* {x}" for x in e["expected_when"]]
    L += ["", "**Competing mechanisms:**", ""]
    L += [f"* {c['text']} ({', '.join(f'`{f}`' for f in c['fields'])})" for c in e["competing"]]
    L += ["", "**Limits of inference:**", ""] + [f"* {x}" for x in e["limits"]]
    L += ["", "**Methods that can give evidence:** " +
          ", ".join(f"`{x}`" for x in e["methods"]) + ".", ""]
    L += ["**Status** (each kept apart):", ""] + \
        [f"* {x}" for x in C.status_lines(e.get("status") or {})] + [""]
    return L


def drafts(methods=None, mechanisms=None) -> list:
    """What the --review-gate fails on, among the methods, their settings and
    the mechanisms: a draft, or a review that names an older text revision or
    behaviour version. Approved for integration, provisional and reviewed pass
    at the current versions (domain_catalog.review_passes)."""
    ms = list((methods if methods is not None else M.METHODS).values())
    mech = mechanisms if mechanisms is not None else M.load_mechanisms()
    out = [f"method {m.id}" for m in ms
           if not C.is_approved(m.status, m.text_revision, m.version)]
    out += [f"setting {m.id}.{s.name}" for m in ms for s in m.settings
            if not C.review_passes(s.review, m.text_revision, m.version)]
    out += [f"mechanism {e['id']}" for e in mech
            if not C.is_approved(e.get("status"), e.get("text_revision"),
                                 e.get("behaviour_version"))]
    return out


def _review_summary(mech: list) -> str:
    """The review status line: counts by state, who and when, and that an
    approval for integration is not a final scientific approval."""
    n_draft = len(drafts(mechanisms=mech))
    recs = [m.status.get("review") or {} for m in M.METHODS.values()] + \
        [s.review for m in M.METHODS.values() for s in m.settings] + \
        [(e.get("status") or {}).get("review") or {} for e in mech]
    count = {k: sum(1 for r in recs if r.get("state") == k) for k in C.GATE_STATES}
    by = sorted({f"{r.get('by')} on {r.get('date')}" for r in recs
                 if r.get("state") in C.GATE_STATES and r.get("by")})
    s = "Review status: " + ", ".join(f"{n} {k}" for k, n in count.items() if n)
    s += f" (methods, settings and mechanisms; by {', '.join(by)})" if by else ""
    prov = [f"`{m.id}.{x.name}`" for m in M.METHODS.values() for x in m.settings
            if x.status == M.PROVISIONAL]
    if prov:
        s += ("; the settings " if len(prov) > 1 else "; the setting ") + \
            " and ".join(prov) + " approved only as provisional values"
    if count[C.APPROVED_STATE] or count[C.PROVISIONAL_STATE]:
        s += f". This is {C.NOT_FINAL}"
    s += f". Drafts or reviews of an older version: {n_draft}."
    return s


def build() -> str:
    mech = M.load_mechanisms()
    L = ["# Analysis methods", "",
         "Generated by `tools/make_methods_doc.py` from `mcp/elm-mcp/src/domain_methods.py` and "
         "`mcp/elm-mcp/domain_mechanisms.json`. Do not edit by hand: change the module or the "
         "file and run the script.", "",
         _review_summary(mech), ""]
    for p in ORIENTATION:
        L += [p, ""]
    L += ["Methods: " + ", ".join(f"`{m}`" for m in M.METHODS) + ".", "",
          "## Outcomes of the timing measures", "",
          "Each timing measure gives, per cell and period, a date or one of these reasons "
          "(the code is what `timing.nc` stores):", "",
          "| code | outcome |", "|---|---|"]
    L += [f"| {c} | {t} |" for k, (c, t) in M.REASONS.items()]
    L += ["", "Quality flags stand beside a date (bits of the flag value):", "",
          "| bit | flag | meaning |", "|---|---|---|"]
    L += [f"| {b} | `{k}` | {t} |" for k, (b, t) in M.FLAGS.items()]
    L += ["", "Dates are days of the year in the calendar the run's files declare, 1 = the year's "
          "first day. A water year is filed under the year it ends, and its dates before the new "
          "year count back (in a noleap calendar, 31 December = 0, 1 October = -91). The year "
          "starts in January and the water year in October unless the control's registry entry "
          f"names other months (`{M.YEAR_START_KEY}`, `{M.WATER_YEAR_START_KEY}`). A running "
          "mean is dated by its middle day and uses only windows inside the period; ties go to "
          "the earliest day. Region distributions are area-weighted (cell area times land "
          "fraction): the median and quartiles of the dates, the counts of dates, of each "
          "reason and of each flag (with the flag's area and the denominator of its share), "
          "the share of the area whose inputs are complete (missing time records and missing "
          "cell values counted apart), the shift in days against the normal years (a cell's "
          "normal is the mean of its dates when every normal year gives one) or the control, "
          "and, for each alternative setting, the sensitivity record: the setting, the "
          "region's median date and median shift beside the default's, and the median "
          "per-cell change of date. Every shift and change of date is on the cells with a "
          "date in both, and beside it stands the change of eligibility: the cells and area "
          "shares with a date in both, in one only, in neither, and the area the statistic "
          "covers.", ""]
    for i, m in enumerate(M.METHODS.values(), 1):
        L += _method(i, m)
    L += ["## Candidate mechanisms", "",
          "Candidates the analysis may look for, never rules: each says which fields the "
          "relation runs through, when it is expected, what else could give the same pattern, "
          "and what a run of the model can show about it.", ""]
    for e in mech:
        L += _mechanism(e)
    text = "\n".join(L).rstrip("\n") + "\n"
    for d in DASHES:
        if d in text:
            raise SystemExit("the document would contain a dash used as a pause; recast the "
                             "text in domain_methods.py or domain_mechanisms.json")
    return text


# THE COMPACT VIEW FOR THE ANALYZER (package E, 2026-11-05). The full
# document is about 80,000 characters, too long to send with every plan and
# every round, so the Analyzer reads a view made from the same module: each
# method's quantity, where it applies, its settings and its limitations, the
# reason codes, and each mechanism's links and competing explanations. It is
# held to COMPACT_LIMIT characters by a test; the full document stays for
# people.
COMPACT_OUT = Path("docs") / "ANALYSIS_METHODS_COMPACT.md"
COMPACT_LIMIT = 25_000


def _compact_method(m: "M.Method") -> list:
    L = [f"### `{m.id}` (version {m.version}; review: {m.status['review'].get('state')})",
         f"Estimates: {m.estimates}", "Applies where (checked by code per cell and period): "
         + " ".join(c.text for c in m.applicability)]
    if m.settings:
        L.append("Settings (the default; the alternatives are computed beside it):")
        for x in m.settings:
            alts = ", ".join(_value(a) for a in x.alternatives) or "none"
            L.append(f"- `{x.name}` = {_value(x.default)} ({x.units}); alternatives: {alts}"
                     + ("; PROVISIONAL" if x.status == M.PROVISIONAL else ""))
    L.append("Limitations: " + " ".join(m.limitations))
    return L + [""]


def build_compact() -> str:
    mech = M.load_mechanisms()
    L = ["# Analysis methods: the compact view", "",
         "Generated by `tools/make_methods_doc.py` for the Analyzer from the same module as "
         "`docs/ANALYSIS_METHODS.md`, which holds the full contracts (inputs, outcomes, "
         "outputs, numerical contract, tests). Do not edit by hand.", "",
         _review_summary(mech), "", ORIENTATION[0], "", ORIENTATION[1], "",
         "Timing outcomes, per cell and period: a date, or one of these reason codes: "
         + "; ".join(f"{c} {t}" for c, t in M.REASONS.values()) + ". Flags beside a date: "
         + "; ".join(f"`{k}` {t}" for k, (b, t) in M.FLAGS.items()) + ".", "",
         "## Methods", ""]
    for m in M.METHODS.values():
        L += _compact_method(m)
    L += ["## Candidate mechanisms (never rules)", ""]
    for e in mech:
        L.append(f"### `{e['id']}`: {e['name']}")
        L.append("Links: " + "; ".join(
            f"{k['from']} to {k['to']}" + ("" if k.get("kind") == "direct" else
                                           f" ({k.get('kind')})")
            + f" (holds when: {'; '.join(k.get('conditions') or [])})" for k in e["links"]))
        L.append("Competing: " + "; ".join(
            f"{c['text']} ({', '.join(c['fields'])})" for c in e["competing"]))
        L.append("Methods: " + ", ".join(e["methods"]) + ".")
        L.append("")
    text = "\n".join(L).rstrip("\n") + "\n"
    for d in DASHES:
        if d in text:
            raise SystemExit("the compact view would contain a dash used as a pause")
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="exit 1 when the committed document differs; write nothing")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    docs = [(Path(a.out), build())]
    if Path(a.out) == OUT:            # the compact view travels with the default output
        docs.append((COMPACT_OUT, build_compact()))
    if a.check:
        stale = 0
        for rel, text in docs:
            out = ROOT / rel
            if (out.read_text() if out.is_file() else "") != text:
                print(f"{rel} is out of date; run: python tools/make_methods_doc.py")
                stale = 1
            else:
                print(f"{rel} is current")
        return stale
    for rel, text in docs:
        out = ROOT / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"wrote {rel} ({len(text.splitlines())} lines, {len(text):,} characters)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
