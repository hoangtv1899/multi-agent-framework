"""The two views of an Analyzer report: the scientific report and its audit
appendix (the user's decision of 2026-11-09).

    REPORT.md   what a reader reads first: the answer, the limitations beside
                it, a short model and data summary, three to five key
                findings, their figures, the remaining uncertainty and one
                verification statement linking to the appendix
    AUDIT.md    every check, record and decision, with the whole page the
                report used to be, so nothing in the record is lost

Both are rendered from the same report record (analysis.json) and the same
verified evidence (claim_trace.evidence_records). The report's narrative uses
only what the audit kept: a summary sentence with a number no kept claim's
record gives (claim_trace.summary_support) is left out of it, and the
appendix says which and why. Nothing here changes a value, a check or a
decision; it chooses where each recorded thing is shown.

    write_views(report, out_dir, records, runs, plan, figure_dir)
"""
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from agents.analysis import claim_trace as CT
from agents.analysis.dashes import strip_dashes

MAIN_MD, AUDIT_MD = "REPORT.md", "AUDIT.md"
KEY_FINDINGS = (3, 5)                    # at least, at most
CAUSAL_CLASS = "in-model attribution"
NO_CAUSE = ("No run in this study isolates a cause: these results are what happened "
            "together within the model, and how closely, not what caused what.")
SEASON_WORDS = {"DJF": "winter (DJF)", "MAM": "spring (MAM)", "JJA": "summer (JJA)",
                "SON": "autumn (SON)"}
MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December")

# WHAT A PROPOSED RUN CAN AND CANNOT SETTLE, by what it changes: the report
# says it in a sentence; the run as the server would make it stays in the
# appendix.
SCOPE_TEMPERATURE = ("changing the air temperature also changes snow, vegetation, soil "
                     "processes and the air's humidity as the land model sees it")
SCOPE_PRECIP = "changing rain and snow also changes the snowpack, soil water, runoff and growth"


def _p(text) -> str:
    return strip_dashes(str(text if text is not None else ""))


def _refs(nums: Sequence[int], word: str = "claim") -> str:
    nums = sorted(set(int(n) for n in nums))
    if not nums:
        return ""
    if len(nums) == 1:
        return f"{word} {nums[0]}"
    return f"{word}s {', '.join(str(n) for n in nums[:-1])} and {nums[-1]}"


def _years_text(years) -> str:
    ys = sorted({int(y) for y in years})
    if len(ys) > 2 and ys == list(range(ys[0], ys[-1] + 1)):
        return f"{ys[0]} to {ys[-1]}"
    return ", ".join(str(y) for y in ys)


def _short(sentence: Optional[str], stop: Optional[str] = ",") -> str:
    """A setup sentence without its codes and file names: the parentheses
    go, and the sentence stops at its first `stop` (None: it runs on, its
    semicolons read as "and")."""
    s = str(sentence or "")
    for _ in range(2):
        s = re.sub(r"\s*\([^()]*\)", "", s)
    s = s.split(stop)[0] if stop else s.replace("; ", " and ")
    return s.strip().rstrip(".")


# ── the parts of the main report ────────────────────────────────────────────

def kept_summary(support: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in support or [] if not row.get("unsupported")]


def left_out(support: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in support or [] if row.get("unsupported")]


def key_findings(report: Dict[str, Any]) -> List[int]:
    """Three to five kept claims, by step 3's ranking (the headline's
    order), by number from 1."""
    n = len(report.get("claims") or [])
    order = [int(i) for i in (report.get("headline_selection") or {}).get("order") or []
             if isinstance(i, int) and 1 <= i <= n]
    order += [i for i in range(1, n + 1) if i not in order]
    return order[:max(KEY_FINDINGS[0], min(KEY_FINDINGS[1], n))][:n]


def _claim_records(c: Dict[str, Any], records: Dict[str, Dict[str, Any]]) -> List[Dict]:
    return [records[i] for i in c.get("evidence") or [] if i in records]


def _period_words(period: Optional[str]) -> str:
    p = str(period or "")
    m = re.match(r"^(\d{4})-([A-Z]{3})$", p)
    if m and m.group(2) in SEASON_WORDS:
        return f"{SEASON_WORDS[m.group(2)]} {m.group(1)}"
    return p


def overlap_notes(report: Dict[str, Any], nums: Sequence[int],
                  records: Dict[str, Dict[str, Any]], timing: Sequence[str] = ()) -> List[str]:
    """What each correlation among these claims compares: its two maps, by
    field and period, and that it uses no timing date."""
    claims = report.get("claims") or []
    overlaps: Dict[str, Dict[str, Any]] = {}
    for i in nums:
        for r in _claim_records(claims[i - 1], records):
            if r.get("call") == "overlap" and r.get("inputs"):
                overlaps.setdefault(r["id"], {"rec": r, "claims": []})["claims"].append(i)
    L = []
    for o in overlaps.values():
        r = o["rec"]
        ins = {x.get("role"): x for x in r["inputs"]}
        a, b = ins.get("a") or {}, ins.get("b") or {}
        what = (f"the {_period_words(a.get('period'))} {a.get('field')} anomaly with the "
                f"{_period_words(b.get('period'))} {b.get('field')} anomaly, cell by cell"
                + (f", over \"{r.get('region')}\"" if r.get("region") else ""))
        not_dates = f"; it does not use the {' or '.join(timing)} dates" if timing else ""
        L.append(f"- **What the correlation compares** ({_refs(o['claims'])}, record "
                 f"{r['id']}): {what}{not_dates}.")
    return L


def timing_named(report: Dict[str, Any], records: Dict[str, Dict[str, Any]]) -> List[str]:
    """The timing measures the kept claims rest on, in their order."""
    M = CT._elm_modules()[2]
    out = []
    for c in report.get("claims") or []:
        for r in _claim_records(c, records):
            if r.get("call") in getattr(M, "TIMING_IDS", ()) and r["call"] not in out:
                out.append(r["call"])
    return out


def limitation_lines(report: Dict[str, Any], support: Sequence[Dict[str, Any]],
                     records: Dict[str, Dict[str, Any]], also: Sequence[int] = ()
                     ) -> List[str]:
    """The limitations that go with the answer: the warnings its claims
    carry, in the registry's words; what each timing measure dates, in its
    contract's words; what each correlation compares; the plan's and the
    binding caveats that touch it; and whether a cause was isolated. Its
    claims: those its summary sentences are tied to, and `also` (the
    claims shown as the answer's own evidence)."""
    claims = report.get("claims") or []
    used = sorted({i for row in kept_summary(support) for i in row["claims"]} | set(also))
    L: List[str] = []
    seen = set()
    for i in used:
        for cav in CT.headline_caveats(claims[i - 1]):
            key = (cav["code"], cav["means"])
            if key in seen:
                continue
            seen.add(key)
            whose = [j for j in used if cav["code"] in
                     [x["code"] for x in CT.headline_caveats(claims[j - 1])]]
            L.append(f"- **{cav['label']}** ({_refs(whose)}): {_p(cav['means'])}")
    M = CT._elm_modules()[2]
    timing = []
    for i in used:
        for r in _claim_records(claims[i - 1], records):
            if r.get("call") in getattr(M, "TIMING_IDS", ()) and r["call"] not in timing:
                timing.append(r["call"])
                L.append(f"- **{r['call']}** ({_refs([i])}): {_p(M.METHODS[r['call']].estimates)}")
    L += overlap_notes(report, used, records, timing)
    # a binding caveat always; a caveat that qualifies, when it names a year
    # the answer names (the rest stand in the appendix)
    years = {y for row in kept_summary(support) for y in CT.years_named(row["sentence"])}
    for c in report.get("caveats") or []:
        st = str(c.get("statement") or "")
        if c.get("severity") == "blocking" or (
                c.get("severity") == "qualify" and set(CT.years_named(st)) & years):
            L.append(f"- **{c.get('id')}**: {_p(st)}")
    if not any(c.get("evidence_class") == CAUSAL_CLASS for c in claims):
        L.append(f"- **Cause**: {NO_CAUSE}")
    return L


def model_data_lines(report: Dict[str, Any], records: Dict[str, Dict[str, Any]],
                     support: Sequence[Dict[str, Any]], also: Sequence[int] = ()
                     ) -> List[str]:
    """A short model and data summary: model, region, years, baseline, the
    configuration that shapes the results, and that these are model output."""
    ms = report.get("model_setup") or {}
    sent = (ms.get("configuration") or {}).get("sentences") or {}
    ctl = ms.get("control") or {}
    years = (ms.get("years") or {}).get("reported") or []
    grid = ms.get("grid") or {}
    claims = report.get("claims") or []
    used = sorted({i for row in kept_summary(support) for i in row["claims"]} | set(also)) or \
        list(range(1, len(claims) + 1))
    recs = [r for i in used for r in _claim_records(claims[i - 1], records)]
    regions = sorted({r.get("region") for r in recs if r.get("region")})
    normal = sorted({int(y) for r in recs for y in (r.get("support") or {}).get("years") or []
                     if str(y).isdigit()})
    who = str(ctl.get("description") or "").split(":")[0].strip() or \
        f"the {ctl.get('name') or report.get('model') or 'model'} control"
    L = [f"- **Model:** {who}" + (f", {_years_text(years)}" if years else "")
         + (f"; {_short(sent.get('weather'))}" if sent.get("weather") else "") + "."]
    if grid.get("cells"):
        L.append(f"- **Grid:** {grid['cells']:,} cells"
                 + (f", {grid['land_area_km2']:,.0f} km2 of land" if grid.get("land_area_km2")
                    else "") + ".")
    if regions:
        def area(reg):
            a = next((v for r in recs if r.get("region") == reg
                      for q, v in (r.get("values") or {}).items()
                      if q.endswith("region_area_km2") and isinstance(v, (int, float))), None)
            return f"\"{reg}\"" + (f" ({a:,.0f} km2)" if a else "")
        L.append(f"- **Region:** {'; '.join(area(r) for r in regions)}.")
    if normal:
        L.append(f"- **Baseline:** the normal years {_years_text(normal)}.")
    conf = [x for x in (_short(sent.get("supplements"), None),
                        _short((ms.get("fire") or {}).get("sentence")),
                        ("CO2 " + _short(sent.get("co2"))) if sent.get("co2") else "") if x]
    if conf:
        L.append(f"- **Configuration that shapes these results:** {'; '.join(conf)}.")
    L.append("- **These are model outputs.** Observations do not enter the analysis.")
    return L


def finding_lines(report: Dict[str, Any], records: Dict[str, Dict[str, Any]],
                  fmap: Dict[str, List[int]]) -> List[str]:
    claims = report.get("claims") or []
    figure_no = {fid: k for k, fid in enumerate(fmap, 1)}
    L = []
    for k, i in enumerate(key_findings(report), 1):
        c = claims[i - 1]
        refs = [f"claim {i}", f"{c.get('evidence_class')}"]
        if c.get("evidence"):
            refs.append("record" + ("s " if len(c["evidence"]) > 1 else " ")
                        + ", ".join(c["evidence"]))
        figs = [figure_no[f] for f, cs in fmap.items() if i in cs]
        if figs:
            refs.append(_refs(figs, "Figure"))
        cav = [x["label"] for x in CT.headline_caveats(c)]
        if cav:
            refs.append("warnings: " + ", ".join(cav))
        L.append(f"{k}. {_p(c.get('claim'))} *({'; '.join(refs)})*")
    return L


def figure_map(report: Dict[str, Any], fig_records: Dict[str, Sequence[str]]
               ) -> Dict[str, List[int]]:
    """{figure: the key findings (claims, by number) it draws}: a figure the
    report lists draws a claim when it is the claim's own finding or draws
    one of the claim's evidence records; in the order the findings name them."""
    claims = report.get("claims") or []
    listed = list(((report.get("figures") or {}).get("investigation") or {}).keys())
    out: Dict[str, List[int]] = {}
    for i in key_findings(report):
        c = claims[i - 1]
        ev = set(c.get("evidence") or [])
        for fid in listed:
            if fid == c.get("finding_id") or ev & set(fig_records.get(fid) or []):
                out.setdefault(fid, [])
                if i not in out[fid]:
                    out[fid].append(i)
    return out


def figures_for(report: Dict[str, Any], fig_records: Optional[Dict[str, Sequence[str]]] = None
                ) -> List[str]:
    """The figures the key findings rest on, in the order they are first
    named."""
    return list(figure_map(report, fig_records or {}))


def figure_lines(report: Dict[str, Any], questions: Dict[str, str],
                 fig_records: Optional[Dict[str, Sequence[str]]] = None) -> List[str]:
    claims = report.get("claims") or []
    fmap = figure_map(report, fig_records or {})
    L = []
    for k, fid in enumerate(fmap, 1):
        whose = fmap[fid]
        cav = []
        for i in whose:
            cav += [x["label"] for x in CT.headline_caveats(claims[i - 1]) if x["label"] not in cav]
        name = Path(str(((report.get("figures") or {}).get("investigation") or {})[fid])).name
        L += [f"![Figure {k}]({name})", "",
              f"*Figure {k}. {_p(questions.get(fid) or fid)} Shown for {_refs(whose)}."
              + (f" Warnings: {', '.join(cav)}." if cav else "") + "*", ""]
    return L


def _months_text(months) -> str:
    """Months in words: a run of three or more as "November to March",
    across the year's end too."""
    ms = sorted({int(m) for m in months or []})
    if not ms or len(ms) == 12:
        return "every month"
    for start in ms:
        run = [(start - 1 + k) % 12 + 1 for k in range(len(ms))]
        if len(ms) > 2 and sorted(run) == ms:
            return f"{MONTH_NAMES[run[0] - 1]} to {MONTH_NAMES[run[-1] - 1]}"
    return ", ".join(MONTH_NAMES[m - 1] for m in ms)


def experiment_sentence(change: Dict[str, Any]) -> str:
    """A proposed run in words: what it changes, what it tests, and what it
    cannot settle."""
    cid, p = change.get("id"), change.get("params") or {}
    if cid == "weather_scale":
        t = float(p.get("temperature_K") or 0.0)
        f = p.get("precip_factor")
        f = None if f in (None, 1, 1.0) else float(f)
        what, tests, scope = [], [], []
        if t:
            what.append(f"the air {abs(t):g} K {'cooler' if t < 0 else 'warmer'}")
            tests.append(f"{'cooler' if t < 0 else 'warmer'} air")
            scope.append(SCOPE_TEMPERATURE)
        if f:
            what.append(f"rain and snow times {f:g}")
            tests.append(f"{'more' if f > 1 else 'less'} rain and snow")
            scope.append(SCOPE_PRECIP)
        where = f"over {p['region']}" if p.get("region") else "over the whole domain"
        if not what:
            return "A run with the weather unchanged (weather_scale), the control again."
        return (f"A run with {' and '.join(what)} in {_months_text(p.get('months'))}, {where}, "
                f"every other input the control's (weather_scale). It would show how the "
                f"results respond to {' and '.join(tests)} in those months. "
                f"{'; '.join(scope).capitalize()}, so the run cannot isolate any one of them"
                f"{', snow included' if t else ''}.")
    return (f"A run with the change {cid} {json.dumps(p, sort_keys=True)}. What it would and "
            f"would not show is the server's own account, in the audit appendix.")


def uncertainty_lines(report: Dict[str, Any], support: Sequence[Dict[str, Any]]) -> List[str]:
    """What remains uncertain, and the proposed sensitivity experiments: a
    cause in step 3's words only when the audit withheld nothing (a cause may
    lean on a withheld result), else in the evidence classes' terms."""
    claims = report.get("claims") or []
    L = []
    classes = {c.get("evidence_class") for c in claims}
    general = False                              # the general line, said once
    for oc in report.get("open_causes") or []:
        cause = _p(oc.get("cause"))
        if not report.get("withheld") and not re.search(r"\d", cause):
            L.append(f"- **Open:** {cause}")
        elif not general:
            general = True
            seen = [w for w, cls in (("that they happened in the same year",
                                      "tested expectation"),
                                     ("how closely they varied together across cells",
                                      "association")) if cls in classes]
            n = len(report.get("open_causes") or [])
            L.append("- **Open:** whether one of these results caused another. The results "
                     "show " + (" and ".join(seen) or "what happened")
                     + "; they do not show a cause. Step 3's own "
                     + ("statement of this open cause is" if n == 1 else
                        f"statements of its {n} open causes are")
                     + " in the audit appendix, because this study has a claim the audit "
                       "withheld.")
        for ch in ((oc.get("proposed_run") or {}).get("changes") or []):
            L.append(f"- **Proposed sensitivity experiment (not run):** "
                     f"{experiment_sentence(ch)} Its configuration, as the server would "
                     f"run it, is in the audit appendix.")
        if not (oc.get("proposed_run") or {}).get("changes"):
            L.append("- No sensitivity experiment was proposed for it.")
    if not report.get("open_causes"):
        L.append("- No open cause was recorded."
                 + ("" if any(c == CAUSAL_CLASS for c in classes) else " " + NO_CAUSE))
    return L


# ── the three-part page (the user's decision of 2026-11-09) ────────────────
# When the plan carried out the recipe "seasonal evolution and spatial
# context", the scientific page explains three things in place of a list of
# key findings: how the year evolved, how the spatial patterns differ, and
# which mechanisms the evidence is consistent with or cannot distinguish.
# Each part gives the Analyzer's own summary sentences on it, labelled as
# interpretation and never as a cause, and under them the verified claims as
# evidence. Nothing is written here that the saved replies do not hold: a
# part the summary says nothing on says so.
YEAR, SPACE, TESTED, MECHANISMS = "year", "space", "tested", "mechanisms"
PART_TITLES = {YEAR: "How the year evolved",
               SPACE: "How the spatial patterns differ",
               MECHANISMS: "Which mechanisms the evidence is consistent with, or cannot "
                           "distinguish"}
INTERPRETATION = "**Interpretation** (the Analyzer's own summary sentences; no cause is stated):"
HOW_TO_READ = ("*How to read the three parts below. Each opens with interpretation: the "
               "Analyzer's own summary sentences on it, a reading of the evidence and not a "
               "statement of cause. A sentence with no claim number beside it is tied to no "
               "single verified claim. Under it stand the verified claims it rests on.*")
NO_SENTENCE = "*The Analyzer's saved summary holds no sentence on this.*"
NO_RULING = ("The plan named these competing explanations before any result was seen. The "
             "saved reply does not rule on them one by one, so each is shown with the "
             "verified claims on its fields, and no verdict.")
_MONTH_PERIOD = re.compile(r"^\d{4}-\d{2}$")
_BAND = re.compile(r"\d+\s*m\b")


def _record_part(r: Dict[str, Any], tested: set) -> str:
    if _MONTH_PERIOD.match(str(r.get("period") or "")) or (
            r.get("call") == "get_series" and r.get("step") in ("day", "month")):
        return YEAR
    if r.get("call") in ("elevation_split", "overlap", "get_map") or (
            r.get("call") in ("anomaly", "difference") and r.get("region") is None) or (
            "," in str(r.get("region") or "") and _BAND.search(str(r.get("region")))):
        return SPACE
    return TESTED if set(r.get("plan_items") or []) & tested else MECHANISMS


def claim_parts(report: Dict[str, Any], records: Dict[str, Dict[str, Any]],
                plan: Dict[str, Any]) -> Dict[int, str]:
    """Each kept claim's part of the page, by its evidence records: the year
    (a record of a month, or a daily or monthly series), space (a map, a
    split by band, an overlap, or a band's region), tested (a record of an
    analysis one of the plan's expectations names), else mechanisms."""
    tested = {str(e.get("analysis")) for e in (plan or {}).get("expectations") or []
              if isinstance(e, dict)}
    out = {}
    for i, c in enumerate(report.get("claims") or [], 1):
        kinds = {_record_part(r, tested) for r in _claim_records(c, records)}
        out[i] = next((k for k in (YEAR, SPACE, TESTED) if k in kinds), MECHANISMS)
    return out


def sentence_parts(support: Sequence[Dict[str, Any]], parts: Dict[int, str]
                   ) -> List[Dict[str, Any]]:
    """Each kept summary sentence with its part: the year when it names a
    month, space when it has a word of place, else the part of the claims
    its numbers tie it to; with no number, the answer's lead when it opens
    the summary and mechanisms otherwise."""
    from agents.analysis import analysis_recipes as AR
    out = []
    for k, row in enumerate(kept_summary(support)):
        s = str(row["sentence"])
        words = {w.lower() for w in re.findall(r"[A-Za-z]+", s)}
        named = {kind for kind, _ in CT.named_periods(s)}
        mine = [parts[i] for i in row.get("claims") or [] if i in parts]
        if "month" in named or "month by month" in s.lower():
            part = YEAR
        elif words & set(AR.PLACE_WORDS) or _BAND.search(s):
            part = SPACE
        elif mine:
            part = next((x for x in (MECHANISMS, YEAR, SPACE) if x in mine), TESTED)
        else:
            part = TESTED if k == 0 else MECHANISMS
        out.append(dict(row, part=part))
    return out


def _said(rows: Sequence[Dict[str, Any]]) -> str:
    said = []
    for row in rows:
        s = _p(row["sentence"])
        said.append(f"{s.rstrip('.')} ({_refs(row['claims'])})." if row.get("claims") else s)
    return " ".join(said)


def _records_text(ids: Sequence[str]) -> str:
    ids = [str(i) for i in ids]
    nums = [int(i[1:]) for i in ids if re.fullmatch(r"L\d+", i)]
    if len(ids) > 3 and len(nums) == len(ids) and nums == list(range(nums[0], nums[0] + len(nums))):
        return f"records {ids[0]} to {ids[-1]}"
    return ("records " if len(ids) > 1 else "record ") + ", ".join(ids)


def _evidence(report: Dict[str, Any], nums: Sequence[int], figure_no: Dict[int, List[int]],
              amended: Optional[Dict[int, Dict[str, Any]]] = None) -> List[str]:
    """The verified claims of a part, each with its class, records, figure
    and warnings: the evidence under the interpretation. amended: a claim's
    correction made in review ({claim number: the amendment}); the claim
    stays as the audit kept it, marked, with the corrected wording and its
    reason under it."""
    claims = report.get("claims") or []
    L = []
    for i in nums:
        c = claims[i - 1]
        a = (amended or {}).get(i)
        refs = [f"claim {i}" + (", as the audit kept it; corrected below" if a else ""),
                str(c.get("evidence_class"))]
        if c.get("evidence"):
            refs.append(_records_text(c["evidence"]))
        if figure_no.get(i):
            refs.append(_refs(figure_no[i], "Figure"))
        cav = [x["label"] for x in CT.headline_caveats(c)]
        if cav:
            refs.append("warnings: " + ", ".join(cav))
        L.append(f"- {_p(c.get('claim'))} *({'; '.join(refs)})*")
        if a:
            ids = [str(x) for x in a.get("records") or []]
            L.append(f"  - **Claim {i}, corrected in review** ({_p(a.get('date') or 'undated')}; "
                     f"an editorial correction, not the Analyzer's wording): {_p(a.get('text'))} "
                     f"*({_records_text(ids)})*")
            L.append(f"  - **Why it was corrected:** "
                     + " ".join(_p(s.get("text")) for s in a.get("reason") or []))
    return L


def mechanism_lines(report: Dict[str, Any], records: Dict[str, Dict[str, Any]],
                    plan: Dict[str, Any]) -> List[str]:
    """The competing explanations the plan named before any result, each
    with the kept claims on its fields, by number; a field no kept claim
    reads says so, and whether the audit struck a claim on it."""
    claims = report.get("claims") or []
    by_field: Dict[str, List[int]] = {}
    for i, c in enumerate(claims, 1):
        for r in _claim_records(c, records):
            if r.get("field") and i not in by_field.setdefault(str(r["field"]), []):
                by_field[str(r["field"])].append(i)
    struck = " ".join(str(w.get("claim") or "") for w in report.get("withheld") or [])
    L = []
    for m in (plan or {}).get("competing") or []:
        if not isinstance(m, dict) or not m.get("text"):
            continue
        bits = []
        for f in m.get("fields") or []:
            if by_field.get(str(f)):
                bits.append(f"{f} ({_refs(by_field[str(f)])})")
            else:
                bits.append(f"{f} (no kept claim" + ("; the audit struck one, see the appendix)"
                                                     if re.search(rf"\b{re.escape(str(f))}\b",
                                                                  struck) else ")"))
        L.append(f"- \"{_p(m['text'])}\" Its fields: {', '.join(bits) or 'none named'}.")
    return L


def three_part_lines(report: Dict[str, Any], support: Sequence[Dict[str, Any]],
                     records: Dict[str, Dict[str, Any]], plan: Dict[str, Any],
                     questions: Dict[str, str], fig_records: Dict[str, Sequence[str]]
                     ) -> Dict[str, List[str]]:
    """{"lead" (the answer's own sentences), "tested" (their evidence),
    "parts" (the three parts' lines)}."""
    claims = report.get("claims") or []
    parts = claim_parts(report, records, plan)
    rows = sentence_parts(support, parts)
    listed = list(((report.get("figures") or {}).get("investigation") or {}).keys())
    draws = {fid: [i for i, c in enumerate(claims, 1)
                   if fid == c.get("finding_id")
                   or set(c.get("evidence") or []) & set(fig_records.get(fid) or [])]
             for fid in listed}
    placed: List[str] = []                       # figures in the order they are shown
    figure_no: Dict[int, List[int]] = {}

    def figures(nums):
        mine = [fid for fid in listed if fid not in placed and set(draws[fid]) & set(nums)]
        placed.extend(mine)
        for fid in listed:
            if fid in placed:
                for i in draws[fid]:
                    if placed.index(fid) + 1 not in figure_no.setdefault(i, []):
                        figure_no[i].append(placed.index(fid) + 1)
        L = []
        for fid in mine:
            k = placed.index(fid) + 1
            name = Path(str(((report.get("figures") or {}).get("investigation") or {})[fid])).name
            L += [f"![Figure {k}]({name})", "",
                  f"*Figure {k}. {_p(questions.get(fid) or fid)} It draws "
                  f"{_refs(draws[fid])}.*", ""]
        return L

    def nums_of(part):
        return [i for i in range(1, len(claims) + 1) if parts[i] == part]

    lead = [r for r in rows if r["part"] == TESTED]
    out = {"lead": _said(lead), "lead_rows": lead, "parts": [HOW_TO_READ, ""],
           "tested_nums": nums_of(TESTED)}
    tested_figs = figures(nums_of(TESTED))
    out["tested"] = (["**Verified evidence**", ""] + _evidence(report, nums_of(TESTED), figure_no)
                     + [""] + tested_figs) if nums_of(TESTED) else []
    timing = timing_named(report, records)
    for part in (YEAR, SPACE, MECHANISMS):
        nums = nums_of(part)
        said = _said([r for r in rows if r["part"] == part])
        figs = figures(nums)
        L = [f"## {PART_TITLES[part]}", "", INTERPRETATION, "", said or NO_SENTENCE, ""]
        if part == MECHANISMS:
            mech = mechanism_lines(report, records, plan)
            if mech:
                L += [f"**Competing explanations.** {NO_RULING}", ""] + mech + [""]
        if nums:
            L += ["**Verified evidence**", ""] + _evidence(report, nums, figure_no) \
                + overlap_notes(report, nums, records, timing) + [""]
        else:
            L += ["**Verified evidence:** none. No kept claim rests on a record of this kind.",
                  ""]
        out["parts"] += L + figs
    return out


# ── the editorial synthesis (the user's decision of 2026-11-09) ────────────
# A page's synthesis may be an EDITORIAL REVISION: sentences written after
# the run by an editor, not by the Analyzer, from the claims the audit kept.
# It is data beside the study (a JSON file), never part of this code, and the
# page says whose words it is. Every sentence is held to the kept claims it
# names (editorial_problems), so the synthesis is no way back in for a claim
# the audit withheld; a synthesis with any problem is not rendered.
#
#   {"by": who, "date": when,
#    "answer":     [S, ...],
#    "caveats":    [S with "covers": [warning codes, caveat ids], ...],
#    "sections":   [{"title", "paragraphs": [[S, ...], ...],
#                    "explanations": [{"status", "reading": [S, ...]}, ...],
#                    "evidence": [claim numbers], "figures": [figure ids],
#                    "notes": [S with "on": the claim it is a note on]}],
#    "unresolved": [S, ...]}
#   S = {"text": one sentence, "claims": [kept claims, by number from 1]}
#
# "explanations": one for each competing explanation the plan named, in the
# plan's order. A caveat, or a sentence on what remains uncertain, may name no
# claim: it then prints no number, names no field and uses no causal word.
#
# A LATER REVISION (2026-11-09, after an outside review of the first) may add:
#
#    "revision": 2,
#    "read_by_hand": [S, ...]   what a person read where code does not check
#                               (a ranking against its whole comparison),
#    "history":    [{"revision", "date", "where" (an earlier revision's
#                    folder), "changes": [what changed, in the editor's words]}],
#    "amendments": [{"claim": a kept claim, "date", "text": its corrected
#                    wording, "records": [the records the corrected wording
#                    rests on], "reason": [S, ...]}]
#
# AN AMENDMENT corrects the wording of a claim the audit kept and a later
# reading found wrong. The claim stays on the page as the audit kept it, marked
# as corrected; the corrected wording stands under it with its reason and is
# traced to its records as a claim is (amendment_problems). It may cite a
# record the claim did not: every record it cites must have been made again by
# a numbers check, the run's own or a review check (record_checks). The run's
# own record (analysis.json) is never changed.
EDITORIAL_LABEL = "*Editorial synthesis.*"
EVIDENCE_LABEL = "**Verified evidence** (the Analyzer's claims, as the audit kept them)"
KEEP_IN_MIND = "**What to keep in mind**"
STATUSES = {"consistent": "Consistent with the evidence",
            "not supported": "Not supported by the evidence",
            "not distinguished": "Not distinguished by the evidence",
            "not assessed": "Not assessed"}
STATUS_NOTE = ("*How to read the statuses. \"Not supported\" means the accepted evidence, at "
               "the scale it was read, gives the explanation no support; it does not rule the "
               "explanation out. \"Not distinguished\" means the evidence fits the explanation "
               "and at least one other. \"Not assessed\" means no accepted claim reads its "
               "fields.*")
QUESTION_TYPE_STATUS = {
    "hypothesis": "Question with expectations stated before the results were seen.",
    "exploratory": "Exploratory question: no expectation was stated before the results were "
                   "seen."}
STATUS_MEANING = {"supported": "the analysis ran and its claims passed the audit; it is not "
                               "a verdict on any explanation"}
QUESTION_TYPE_LINES = {
    "hypothesis": "The plan stated its expectations before any result was seen. A claim "
                  "marked \"tested expectation\" tests one of them; every other claim "
                  "describes what the run shows.",
    "exploratory": "This question is exploratory: no expectation was stated before the "
                   "results were seen, so the findings describe the model's year and test "
                   "nothing."}
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]{3,}")
_CLAIM_REF = re.compile(r"\bclaims? \d+(?:(?:, | and | to )\d+)*", re.I)
_ENDINGS = ("iness", "iest", "ness", "ied", "ity", "ier", "ily", "est", "ies", "ing", "er",
            "ed", "ly", "es", "s", "y")


def amendments(ed: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """{claim number: its correction} of an editorial revision."""
    return {a["claim"]: a for a in (ed or {}).get("amendments") or []
            if isinstance(a.get("claim"), int) and not isinstance(a.get("claim"), bool)}


def _extra_records(ed: Dict[str, Any]) -> Dict[int, List[str]]:
    """The records a correction adds to its claim: a sentence that names the
    claim is read on them too."""
    return {n: [str(i) for i in a.get("records") or []] for n, a in amendments(ed).items()}


def editorial_note(ed: Dict[str, Any]) -> str:
    rev = f"Editorial revision {ed['revision']}" if ed.get("revision") else "Editorial revision"
    am = sorted(amendments(ed))
    unchanged = ("The claims listed as verified evidence are the Analyzer's, unchanged."
                 if not am else
                 f"The claims listed as verified evidence are the Analyzer's, as the audit kept "
                 f"them; {_refs(am)} {'carries' if len(am) == 1 else 'carry'} a correction "
                 f"made in review, shown under the original wording with its reason.")
    return (f"{rev} ({_p(ed.get('by') or 'an editor')}, "
            f"{_p(ed.get('date') or 'undated')}). The synthesis on this page was written "
            f"after the run, from the claims the audit kept. It is not the Analyzer's output: "
            f"the Analyzer's own summary is in the audit appendix, sentence by sentence. Each "
            f"sentence of the synthesis names the claims it rests on, and code checked it "
            f"against those claims' records as a claim is checked. A claim reference covers "
            f"the sentences before it, back to the previous reference. {unchanged}"
            + (" What changed since the earlier revision is listed in the audit appendix."
               if ed.get("history") else ""))


def _stem(word: str) -> str:
    w, cut = str(word).lower(), True
    while cut:
        cut = False
        for e in _ENDINGS:
            if w.endswith(e) and len(w) - len(e) >= 2:
                w, cut = w[:-len(e)], True
                break
    return w


def withheld_words(report: Dict[str, Any]) -> Dict[str, str]:
    """{stem: word}: the words that carry what a withheld claim said, which
    no synthesis sentence may use. They are the words of the withheld claim
    that name a field it reads (the field's code and its meaning in the
    catalog: "relative", "humidity") or compare ("driest", "warmer"), and
    that no kept claim, nor the question, uses. Words are compared by stem,
    so "drier" and "dry" are "driest"."""
    notes = CT._elm_modules()[1].load_notes()
    kept = {_stem(w) for c in report.get("claims") or []
            for w in _WORD.findall(str(c.get("claim") or ""))}
    kept |= {_stem(w) for w in _WORD.findall(str(report.get("question") or ""))}
    out: Dict[str, str] = {}
    for w in report.get("withheld") or []:
        text = str(w.get("claim") or "")
        own = {_stem(x) for f in CT.field_mentions(text, notes)
               for x in _WORD.findall(f"{f} {_meaning(notes, f)}")}
        for word in _WORD.findall(text):
            if _stem(word) not in kept and (_stem(word) in own
                                            or re.search(r"(?:er|est)$", word.lower())):
                out.setdefault(_stem(word), word)
    return out


def _regions(records: Dict[str, Dict[str, Any]], plan: Dict[str, Any]) -> List[str]:
    regs = {r["region"] for r in records.values() if r.get("region")}
    for a in (plan or {}).get("analyses") or []:
        regs |= {str(x) for x in (a or {}).get("regions") or []}
    if ((plan or {}).get("screen") or {}).get("region"):
        regs.add(plan["screen"]["region"])
    return sorted(regs)


def _paragraphs(sec: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    return [p for p in sec.get("paragraphs") or [] if p]


def editorial_sentences(ed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every sentence of an editorial synthesis, with where it stands."""
    out = [dict(s, where="the answer") for s in ed.get("answer") or []]
    out += [dict(s, where="what to keep in mind", free=True) for s in ed.get("caveats") or []]
    for sec in ed.get("sections") or []:
        out += [dict(s, where=f"\"{sec.get('title')}\"") for p in _paragraphs(sec) for s in p]
        for k, ex in enumerate(sec.get("explanations") or [], 1):
            out += [dict(s, where=f"\"{sec.get('title')}\", explanation {k}")
                    for s in ex.get("reading") or []]
        out += [dict(s, where=f"\"{sec.get('title')}\", note on claim {s.get('on')}")
                for s in sec.get("notes") or []]
    # the reason of a correction rests on the claim it corrects
    for n, a in amendments(ed).items():
        out += [dict(s, claims=list(s.get("claims") or [n]),
                     where=f"the correction of claim {n}, its reason")
                for s in a.get("reason") or []]
    out += [dict(s, where="the remaining uncertainty", free=True)
            for s in ed.get("unresolved") or []]
    return out + [dict(s, where="the verification, read by hand")
                  for s in ed.get("read_by_hand") or []]


def sentence_problems(s: Dict[str, Any], report: Dict[str, Any],
                      records: Dict[str, Dict[str, Any]], plan: Dict[str, Any],
                      runs: Dict[str, Dict[str, Any]],
                      extra: Optional[Dict[int, Sequence[str]]] = None) -> List[str]:
    """One editorial sentence against the kept claims it names: it names at
    least one, all kept; read as a claim on their records, it passes the
    claim trace (its numbers, years, periods and regions are theirs, it has
    no causal word, it keeps the percent and range rules and their
    warnings); it names no field their records do not read; and it uses no
    word that only a withheld claim uses. extra: the records a correction
    adds to a claim ({claim number: [record ids]})."""
    claims = report.get("claims") or []
    text = str(s.get("text") or "")
    nums = list(s.get("claims") or [])
    probs = [f"it names claim {n}, which is no kept claim" for n in nums
             if not (isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= len(claims))]
    # a claim it refers to in its words ("claim 6") is one of those it names,
    # and is no number of its own
    for m in _CLAIM_REF.finditer(text):
        probs += [f"it refers to claim {n} in its words and does not name it"
                  for n in map(int, re.findall(r"\d+", m.group(0))) if n not in nums]
    text = _CLAIM_REF.sub(lambda m: re.sub(r"\d+", "", m.group(0)), text)
    nums = [n for n in nums if isinstance(n, int) and not isinstance(n, bool)
            and 1 <= n <= len(claims)]
    ids: List[str] = []
    if nums:
        ids = list(dict.fromkeys(str(i) for n in nums
                                 for i in list(claims[n - 1].get("evidence") or [])
                                 + list((extra or {}).get(n) or [])))
        carried = sorted({x["code"] for n in nums for x in CT.headline_caveats(claims[n - 1])}
                         | {str(w) for n in nums for w in claims[n - 1].get("warnings") or []})
        probs += CT.check_claim({"claim": text, "evidence": ids, "values": [],
                                 "evidence_class": "observation", "warnings": carried},
                                records, plan or {}, runs or {},
                                _regions(records, plan))["problems"]
    elif not s.get("free"):
        probs.append("it names no claim: every sentence of the synthesis rests on kept claims")
    else:
        if [w for w, _, _ in CT._number_words(text) if not CT._YEARLIKE.match(w.lstrip("-"))]:
            probs.append("it prints a number and names no claim")
        if CT._CAUSAL.search(text):
            probs.append(f"the causal word \"{CT._CAUSAL.search(text).group(0)}\" in a "
                         f"sentence that names no claim")
    recs = [records[i] for i in ids if i in records]
    fields = {r.get("field") for r in recs} | {x.get("field") for r in recs
                                               for x in r.get("inputs") or []}
    notes = CT._elm_modules()[1].load_notes()
    seen = CT.field_mentions(text, notes)

    def name(f):
        return CT._plain_name((notes.get(f) or {}).get("meaning"))
    calls = " ".join(str(r.get("call")).replace("_", " ") for r in recs)

    def read(f):
        """A field its records read; or, named by a plain word and not by
        its code, a word that is part of the name of a field or of a method
        its records hold ("runoff" in "total liquid runoff", "half runoff";
        two fields may share "air temperature")."""
        if f in fields:
            return True
        if re.search(rf"\b{re.escape(f)}\b", text) or not name(f):
            return False
        return any(g and name(f) in name(g) for g in fields) or name(f) in calls
    probs += [f"it names {f}, which none of its claims' records reads"
              for f in sorted(seen) if not read(f)]
    blocked = withheld_words(report)
    for w in dict.fromkeys(_WORD.findall(text)):
        if _stem(w) in blocked:
            probs.append(f"it uses \"{w}\": a word of a claim the audit withheld "
                         f"(\"{blocked[_stem(w)]}\") and of no kept claim")
    return probs


def claims_by_field(report: Dict[str, Any], records: Dict[str, Dict[str, Any]]
                    ) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for i, c in enumerate(report.get("claims") or [], 1):
        for r in _claim_records(c, records):
            if r.get("field") and i not in out.setdefault(str(r["field"]), []):
                out[str(r["field"])].append(i)
    return out


def record_checks(report: Dict[str, Any], review: Optional[Dict[str, Any]] = None
                  ) -> Dict[str, Dict[str, Any]]:
    """{record id: how a numbers check made it again}: "by" the run's own
    check (the report's evidence_numbers: every lookup a claim cites) or a
    later review check (the same check, on lookups no claim cites), the
    numbers "compared" and "mismatched", and "ok" (some compared, none
    mismatched). A record in neither was made again by no check."""
    out: Dict[str, Dict[str, Any]] = {}
    for by, rec in (("review", review or {}), ("run", report.get("evidence_numbers") or {})):
        bad: Dict[str, int] = {}
        for m in rec.get("mismatches") or []:
            bad[str(m.get("record"))] = bad.get(str(m.get("record")), 0) + 1
        for rid, r in (rec.get("records") or {}).items():
            n, k = int(r.get("compared") or 0), max(int(r.get("mismatched") or 0),
                                                    bad.get(str(rid), 0))
            out[str(rid)] = {"by": by, "call": r.get("call"), "compared": n, "mismatched": k,
                             "ok": n > 0 and k == 0 and not r.get("error"),
                             "why": r.get("error") or r.get("note")
                             or "; ".join(r.get("notes") or [])}
    return out


def amendment_problems(ed: Dict[str, Any], report: Dict[str, Any],
                       records: Dict[str, Dict[str, Any]], plan: Dict[str, Any],
                       runs: Optional[Dict[str, Dict[str, Any]]] = None,
                       review: Optional[Dict[str, Any]] = None) -> List[str]:
    """Why a correction of a kept claim may not be shown: it names no kept
    claim, or a claim twice; it changes no word, gives no reason or cites no
    record; its corrected wording, read on its records as a claim of the same
    class is read, fails the claim trace; it uses a word of a withheld claim;
    or a record it cites was made again by no numbers check."""
    claims = report.get("claims") or []
    checks = record_checks(report, review)
    blocked = withheld_words(report) if ed.get("amendments") else {}
    out, seen = [], set()
    for a in ed.get("amendments") or []:
        n = a.get("claim")
        if not (isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= len(claims)):
            out.append(f"a correction names claim {n}, which is no kept claim")
            continue
        if n in seen:
            out.append(f"claim {n} has two corrections: one each")
        seen.add(n)
        c, text = claims[n - 1], str(a.get("text") or "").strip()
        ids = [str(i) for i in a.get("records") or []]
        if not text or text == str(c.get("claim") or "").strip():
            out.append(f"the correction of claim {n} changes no word of it")
        if not a.get("reason"):
            out.append(f"the correction of claim {n} gives no reason")
        if not ids:
            out.append(f"the correction of claim {n} cites no record")
        out += [f"the correction of claim {n} cites {i}, which is no record of the study"
                for i in ids if i not in records]
        carried = sorted({x["code"] for x in CT.headline_caveats(c)}
                         | {str(w) for w in c.get("warnings") or []})
        out += [f"the correction of claim {n}: {p}" for p in CT.check_claim(
            {"claim": text, "evidence": ids, "values": [],
             "evidence_class": c.get("evidence_class") or "observation", "warnings": carried},
            records, plan or {}, runs or {}, _regions(records, plan))["problems"]]
        out += [f"the correction of claim {n} uses \"{w}\": a word of a claim the audit "
                f"withheld" for w in dict.fromkeys(_WORD.findall(text)) if _stem(w) in blocked]
        out += [f"the correction of claim {n} cites {i}, which no numbers check has made again"
                for i in ids if i in records and not (checks.get(i) or {}).get("ok")]
    return out


def editorial_problems(ed: Dict[str, Any], report: Dict[str, Any],
                       records: Dict[str, Dict[str, Any]], plan: Dict[str, Any],
                       runs: Optional[Dict[str, Dict[str, Any]]] = None,
                       review: Optional[Dict[str, Any]] = None) -> List[str]:
    """Why an editorial synthesis may not be rendered: a sentence's own
    problems (sentence_problems); a correction's (amendment_problems); a
    warning the answer's claims carry that no caveat covers; a competing
    explanation of the plan left without a reading, or read with no kept
    claim on its fields; a kept claim or a listed figure shown nowhere, or
    twice; a line of the revision history that uses a withheld claim's word."""
    claims = report.get("claims") or []
    out = amendment_problems(ed, report, records, plan, runs, review)
    extra = _extra_records(ed)
    for s in editorial_sentences(ed):
        out += [f"{s['where']}: \"{s.get('text')}\": {p}"
                for p in sentence_problems(s, report, records, plan, runs or {}, extra)]
    blocked = withheld_words(report) if ed.get("history") else {}
    for h in ed.get("history") or []:
        for line in h.get("changes") or []:
            out += [f"the revision history: \"{line}\": it uses \"{w}\", a word of a claim the "
                    f"audit withheld" for w in dict.fromkeys(_WORD.findall(str(line)))
                    if _stem(w) in blocked]
    # the caveats beside the answer cover every warning its claims carry
    known = {x["code"] for c in claims for x in CT.headline_caveats(c)} | \
        {str(c.get("id")) for c in report.get("caveats") or []}
    covered = {str(x) for s in ed.get("caveats") or [] for x in s.get("covers") or []}
    for x in sorted(covered - known):
        out.append(f"a caveat covers \"{x}\", which is no warning of a kept claim and no "
                   f"caveat of the report")
    for n in sorted({n for s in ed.get("answer") or [] for n in s.get("claims") or []
                     if isinstance(n, int) and 1 <= n <= len(claims)}):
        for cav in CT.headline_caveats(claims[n - 1]):
            if cav["code"] not in covered:
                out.append(f"the answer names claim {n}, which carries the warning "
                           f"\"{cav['label']}\" ({cav['code']}), and no caveat covers it")
    # each competing explanation of the plan has its reading
    by_field = claims_by_field(report, records)
    comp = [m for m in (plan or {}).get("competing") or [] if isinstance(m, dict) and m.get("text")]
    for sec in ed.get("sections") or []:
        exs = sec.get("explanations")
        if exs is None:
            continue
        if len(exs) != len(comp):
            out.append(f"\"{sec.get('title')}\" reads {len(exs)} explanations and the plan "
                       f"named {len(comp)}")
            continue
        for k, (ex, m) in enumerate(zip(exs, comp), 1):
            read = any(by_field.get(str(f)) for f in m.get("fields") or [])
            if ex.get("status") not in STATUSES:
                out.append(f"explanation {k}: its status is one of {', '.join(STATUSES)}")
            elif not read and (ex.get("status") != "not assessed" or ex.get("reading")):
                out.append(f"explanation {k}: no kept claim reads its fields, so it is "
                           f"\"not assessed\" and has no reading")
            elif ex.get("status") != "not assessed" and not ex.get("reading"):
                out.append(f"explanation {k}: a status other than \"not assessed\" needs a "
                           f"reading that names its claims")
    # every kept claim and every listed figure is shown once
    shown = [n for sec in ed.get("sections") or [] for n in sec.get("evidence") or []]
    for n in range(1, len(claims) + 1):
        if shown.count(n) != 1:
            out.append(f"claim {n} is listed as evidence {shown.count(n)} times: once each")
    figs = [f for sec in ed.get("sections") or [] for f in sec.get("figures") or []]
    for fid in ((report.get("figures") or {}).get("investigation") or {}):
        if figs.count(fid) != 1:
            out.append(f"the figure {fid} is shown {figs.count(fid)} times: once each")
    for fid in figs:
        if fid not in ((report.get("figures") or {}).get("investigation") or {}):
            out.append(f"the figure {fid} is none of the report's")
    return out


def withheld_line(report: Dict[str, Any]) -> Optional[str]:
    """What the audit withheld, by field, and that the page says nothing on
    it: code's words, since the synthesis may not speak of it."""
    held = report.get("withheld") or []
    if not held:
        return None
    notes = CT._elm_modules()[1].load_notes()
    names = []
    for w in held:
        for f in sorted(CT.field_mentions(str(w.get("claim") or ""), notes)):
            name = f"{CT._plain_name((notes.get(f) or {}).get('meaning'))} ({f})"
            if name not in names:
                names.append(name)
    n = len(held)
    return (f"The audit withheld {n} claim{'s' if n > 1 else ''}"
            + (f", on {', '.join(names)}" if names else "")
            + f". This report makes no statement on {'them' if n > 1 else 'it'}; the audit "
              f"appendix gives the claim{'s' if n > 1 else ''} and the rule that struck "
              f"{'them' if n > 1 else 'it'}.")


def _meaning(notes: Dict[str, Any], field: str) -> str:
    """A field's meaning in the catalog's words, up to its first comma or
    colon: "relative humidity at 2 m"."""
    return re.split(r"[,:]", str((notes.get(field) or {}).get("meaning") or field), 1)[0].strip()


def _id_runs(ids: Sequence[str]) -> str:
    """Record ids with each run of three or more in a row given as a range:
    "L12, L18 to L41, L46, L47"."""
    out, run = [], []

    def close():
        out.extend([f"{run[0]} to {run[-1]}"] if len(run) > 2 else run)
        run.clear()
    for i in ids:
        m, last = re.fullmatch(r"L(\d+)", str(i)), run[-1] if run else None
        if run and not (m and re.fullmatch(r"L\d+", last) and int(m.group(1)) == int(last[1:]) + 1):
            close()
        run.append(str(i))
    if run:
        close()
    return ("records " if len(ids) > 1 else "record ") + ", ".join(out)


def experiment_lines(report: Dict[str, Any]) -> List[str]:
    L = []
    for oc in report.get("open_causes") or []:
        for ch in ((oc.get("proposed_run") or {}).get("changes") or []):
            L.append(f"- **Proposed sensitivity experiment (not run):** "
                     f"{experiment_sentence(ch)} Its configuration, as the server would "
                     f"run it, is in the audit appendix.")
    return L or ["- No sensitivity experiment was proposed."]


def _paragraph(sents: Sequence[Dict[str, Any]]) -> str:
    """Sentences with the claims they rest on. A reference covers the
    sentences before it back to the previous reference: sentences in a row
    that rest on the same claims share one."""
    out = []
    for k, s in enumerate(sents):
        nxt = sents[k + 1] if k + 1 < len(sents) else None
        if s.get("claims") and not (nxt and sorted(nxt.get("claims") or []) == sorted(s["claims"])):
            out.append(f"{_p(s['text']).rstrip('.')} ({_refs(s['claims'])}).")
        else:
            out.append(_p(s["text"]))
    return " ".join(out)


def _answer_rows(ed: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"sentence": s.get("text"), "claims": list(s.get("claims") or [])}
            for s in ed.get("answer") or []]


# ── what a figure draws beyond the claims, and which check made it again ───
# The run's numbers check makes again every lookup a CLAIM cites. A figure may
# draw lookups no kept claim cites (a band no claim reads, the record of a
# withheld claim, a series, a map): those were made again by no check of the
# run. A page that says its numbers were verified says so of the claims, and
# says of each figure what else it draws and whether a review check (the same
# check, run later on those lookups) made it again.
_NO_NUMBER = {"get_series": "a series", "get_map": "a map", "get_timing": "a set of dates"}


def _cited_ids(report: Dict[str, Any]) -> set:
    return {str(i) for c in report.get("claims") or [] for i in c.get("evidence") or []}


def _withheld_fields(report: Dict[str, Any]) -> Dict[str, str]:
    """{field: the rule that struck the claim on it} of the withheld claims."""
    if not report.get("withheld"):
        return {}
    notes = CT._elm_modules()[1].load_notes()
    return {f: str(w.get("struck_by")) for w in report.get("withheld") or []
            for f in CT.field_mentions(str(w.get("claim") or ""), notes)}


def _cap(s: str) -> str:
    """The first letter upper case, the rest as it is ("Records L42 and L43")."""
    return s[:1].upper() + s[1:]


def _ids_words(ids: Sequence[str]) -> str:
    """"record L9" / "records L1, L2 and L9" / "records L18 to L41 and L44"."""
    body = _id_runs(ids)
    word, _, rest = body.partition(" ")
    bits = rest.split(", ")
    return f"{word} " + (bits[0] if len(bits) == 1 else f"{', '.join(bits[:-1])} and {bits[-1]}")


def figure_extra(report: Dict[str, Any], records: Dict[str, Dict[str, Any]],
                 drawn: Sequence[str], checks: Dict[str, Dict[str, Any]]
                 ) -> Dict[str, List[str]]:
    """The records a figure draws that no kept claim cites, by what a numbers
    check did with them: "ok" (made again, no mismatch), "none" (the log
    keeps no number of them), "bad" (a mismatch, or the check failed on it),
    "unchecked" (in no check)."""
    cited = _cited_ids(report)
    out: Dict[str, List[str]] = {"ok": [], "none": [], "bad": [], "unchecked": []}
    for i in dict.fromkeys(str(x) for x in drawn):
        if i in cited:
            continue
        st = checks.get(i)
        key = "unchecked" if not st else "ok" if st["ok"] else \
            "none" if not st["compared"] and not st["mismatched"] else "bad"
        out[key].append(i)
    return out


def figure_check_line(report: Dict[str, Any], records: Dict[str, Dict[str, Any]],
                      drawn: Sequence[str], checks: Dict[str, Dict[str, Any]]
                      ) -> Optional[str]:
    """One line under a figure on what it draws beyond the kept claims'
    records, or None when it draws nothing more."""
    ex = figure_extra(report, records, drawn, checks)
    every = ex["ok"] + ex["none"] + ex["bad"] + ex["unchecked"]
    if not every:
        return None
    every.sort(key=lambda r: int(r[1:]) if r[1:].isdigit() else 0)
    s = [f"Beyond the claims' records, this figure draws {_ids_words(every)}, which no "
         f"accepted claim cites."]
    one_kind = sum(1 for v in ex.values() if v) == 1

    def subject(ids):
        """The records by name, or "it"/"they" when they are all the figure adds."""
        return ("It" if len(ids) == 1 else "They") if one_kind else _cap(_ids_words(ids))
    if ex["ok"]:
        n = sum(checks[i]["compared"] for i in ex["ok"])
        by = {checks[i]["by"] for i in ex["ok"]}
        who = "The review check" if by == {"review"} else "A numbers check"
        of = ("its" if len(ex["ok"]) == 1 else "their") if one_kind else "the"
        s.append(f"{who} made {of} {n:,} numbers"
                 + ("" if one_kind else f" of {_ids_words(ex['ok'])}")
                 + " again from the model output, with no mismatch.")
    if ex["none"]:
        kinds = sorted({_NO_NUMBER.get(str(checks[i].get("call")), "values")
                        for i in ex["none"]})
        s.append(f"{subject(ex['none'])} {'holds' if len(ex['none']) == 1 else 'hold'} "
                 f"{' or '.join(kinds)} of which the log keeps no number, so no check made "
                 f"{'it' if len(ex['none']) == 1 else 'them'} again.")
    if ex["bad"]:
        s.append(f"{subject(ex['bad'])} did NOT pass the review check: the audit appendix "
                 f"gives the mismatch.")
    if ex["unchecked"]:
        s.append(f"{subject(ex['unchecked'])} "
                 f"{'was' if len(ex['unchecked']) == 1 else 'were'} made again by no check.")
    struck = _withheld_fields(report)
    for i in every:
        f = str((records.get(i) or {}).get("field") or "")
        if f in struck:
            notes = CT._elm_modules()[1].load_notes()
            s.append(f"Record {i} is {_meaning(notes, f)} ({f}), the record of the claim the "
                     f"audit withheld"
                     + (": the percent rule struck that claim for how it words a percent, not "
                        "for this record's values" if struck[f] == "percent" else
                        f" under the rule `{struck[f]}`")
                     + ". No sentence on this page reads it.")
    return " ".join(s)


# WHAT WAS VERIFIED, AND WHAT WAS NOT (the user's decision of 2026-11-09).
# Until the numbers check covers what the figures draw (a plotted series among
# it) and the claim trace holds a ranking to its whole comparison, a page
# says of each kind of statement whether code checked it. The review of an
# interpretation stays a person's: no code here judges whether a sentence
# follows from its claims.
_RANKING = re.compile(r"\b(?:most|least|largest|smallest|highest|lowest|earliest|latest|"
                      r"greatest|warmest|coldest|every|all)\b", re.I)


def ranking_claims(report: Dict[str, Any]) -> List[int]:
    """The kept claims that use a ranking word or speak of every member of a
    set: the claim trace holds their numbers to the records they cite, not
    the ranking to records they leave out."""
    return [i for i, c in enumerate(report.get("claims") or [], 1)
            if _RANKING.search(str(c.get("claim") or ""))]


def coverage_lines(report: Dict[str, Any], ed: Dict[str, Any],
                   records: Dict[str, Dict[str, Any]],
                   fig_drawn: Dict[str, Sequence[str]],
                   review: Optional[Dict[str, Any]] = None) -> List[str]:
    """The verification section of an editorial page of a later revision:
    what code checked (the claims' numbers by the run's own check, each claim
    against its records, what else the figures draw by a review check, the
    synthesis sentence by sentence) and what it did not (a plotted series the
    log keeps no number of, a ranking against records a claim leaves out,
    whether an interpretation follows). ed["read_by_hand"]: the editor's
    sentences on what a person read where code does not."""
    nc, en = report.get("numbers_check") or {}, report.get("evidence_numbers") or {}
    checks = record_checks(report, review)
    parts = []
    if nc.get("n_compared") is not None:
        parts.append(f"the {nc['n_compared']:,} numbers of the regional table")
    if en.get("n_compared") is not None:
        k = len(en.get("records") or {})
        parts.append(f"the {en['n_compared']:,} numbers of the "
                     + (f"{k} " if k else "") + "lookups the claims cite")
    ok = (nc.get("pass") is not False) and (en.get("pass") is not False)
    done, not_done = [], []
    if parts:
        done.append("**The numbers in the accepted claims.** Before this report was written, "
                    "an independent reading made them again from the model output "
                    f"({' and '.join(parts)}), "
                    + ("with no mismatch." if ok else "and the appendix lists the mismatches."))
    else:
        not_done.append("**The numbers in the accepted claims.** This record holds no numbers "
                        "check.")
    held = len(report.get("withheld") or [])
    line = (f"**Each claim against its records.** Every claim was traced to its evidence "
            f"records; the audit kept {len(report.get('claims') or [])} and withheld {held}.")
    am = amendments(ed)
    if am:
        added = sorted({str(i) for a in am.values() for i in a.get("records") or []
                        if str(i) not in _cited_ids(report)},
                       key=lambda r: int(r[1:]) if r[1:].isdigit() else 0)
        line += (f" {_cap(_refs(sorted(am)))} {'carries' if len(am) == 1 else 'carry'} a "
                 f"correction made in review. The corrected wording was traced to its records "
                 f"as a claim is"
                 + (f", and {_ids_words(added)}, which it adds, was made again by the review "
                    f"check" if added else "")
                 + ". The run's own record is unchanged.")
    done.append(line)
    ex: Dict[str, List[str]] = {"ok": [], "none": [], "bad": [], "unchecked": []}
    where: Dict[str, List[int]] = {"none": [], "bad": [], "unchecked": []}
    for k, drawn in enumerate(fig_drawn.values(), 1):
        for key, ids in figure_extra(report, records, drawn, checks).items():
            ex[key] += [i for i in ids if i not in ex[key]]
            if ids and key in where:
                where[key].append(k)
    total = sum(len(v) for v in ex.values())
    if total:
        when = str((review or {}).get("checked_at") or "")[:10]
        line = (f"**The other records the figures draw.** The figures draw {total} "
                f"lookup{'s' if total > 1 else ''} that no accepted claim cites, which the "
                f"reading above did not cover.")
        if ex["ok"]:
            n = sum(checks[i]["compared"] for i in ex["ok"])
            some = (f"all {total}" if len(ex["ok"]) == total and total > 1 else
                    "it" if total == 1 else f"{len(ex['ok'])} of them")
            line += (f" In the review{' of ' + when if when else ''}, the same reading made "
                     f"{some} again ({n:,} numbers, no mismatch). A note under each figure "
                     f"says which.")
            done.append(line)
        kinds = " or ".join(sorted({_NO_NUMBER.get(str(checks[i].get("call")), "values")
                                    for i in ex["none"]}))
        for key, text in (("none", "{verb} " + kinds + " of which the log keeps no number, "
                                   "so no check made {it} again"),
                          ("unchecked", "{was} made again by no check"),
                          ("bad", "did NOT pass the review check; the appendix gives the "
                                  "mismatch")):
            if ex[key]:
                k = len(ex[key])
                not_done.append(
                    f"**{'Plotted series' if key == 'none' else 'Other records the figures draw'}"
                    f".** {k} of the lookups the figures draw "
                    + text.format(verb="holds" if k == 1 else "hold",
                                  it="it" if k == 1 else "them", was="was" if k == 1 else "were")
                    + f" ({_refs(where[key], 'Figure')}).")
    sents = editorial_sentences(ed)
    free = sum(1 for x in sents if not x.get("claims"))
    done.append(f"**The editorial synthesis.** {len(sents) - free} of its {len(sents)} "
                f"sentences name kept claims, and every number, year, season, month and region "
                f"in them is given by the records of those claims"
                + (" or of their corrections" if am else "")
                + f". The other {free} state a limitation, name no claim and print no number.")
    ranked = ranking_claims(report)
    by_hand = " ".join(_p(x.get("text")) for x in ed.get("read_by_hand") or [])
    if ranked:
        not_done.append(
            f"**Ranking words.** The claim trace holds a claim's numbers to the records it "
            f"cites. It does not hold a ranking, or a statement about every member of a set "
            f"(\"most\", \"least\", \"largest\", \"every band\"), to records the claim leaves "
            f"out. {_cap(_refs(ranked))} {'uses' if len(ranked) == 1 else 'use'} such words."
            + (f" *Read by hand in review:* {by_hand}" if by_hand else ""))
    plain = sum(1 for x in sents if not re.search(r"\d", _CLAIM_REF.sub("", str(x.get("text")))))
    not_done.append(
        f"**Interpretation.** Code checks a sentence's numbers, periods, regions, fields and "
        f"wording. It cannot tell whether the sentence follows from the claims it names; "
        f"{plain} of the synthesis's {len(sents)} sentences print no number at all. That "
        f"reading is a person's, and no code stands in for it.")
    return (["**Checked by code**", ""] + [f"- {x}" for x in done]
            + ["", "**Not checked by code**", ""] + [f"- {x}" for x in not_done]
            + ["", f"Every check, record and decision is in the [audit appendix]({AUDIT_MD})."])


def render_editorial(report: Dict[str, Any], records: Dict[str, Dict[str, Any]],
                     plan: Dict[str, Any], runs: Optional[Dict[str, Dict[str, Any]]],
                     ed: Dict[str, Any], questions: Optional[Dict[str, str]] = None,
                     fig_records: Optional[Dict[str, Sequence[str]]] = None,
                     note: Optional[str] = None,
                     review: Optional[Dict[str, Any]] = None) -> str:
    """REPORT.md with an editorial synthesis in place of the Analyzer's
    summary. Pure. Raises ValueError when the synthesis has a problem
    (editorial_problems): no page is made from it. A later revision
    (ed["revision"]) names the question's type beside the status, says of
    each figure what it draws beyond the claims' records and which check
    made that again (review: a review check's record), and words its
    verification section to match (coverage_lines)."""
    probs = editorial_problems(ed, report, records, plan, runs, review)
    if probs:
        raise ValueError("the editorial synthesis is not rendered:\n  " + "\n  ".join(probs))
    claims = report.get("claims") or []
    questions, fig_records = questions or {}, fig_records or {}
    later = bool(ed.get("revision"))
    amended = amendments(ed)
    checks = record_checks(report, review)
    listed = (report.get("figures") or {}).get("investigation") or {}
    placed = [f for sec in ed.get("sections") or [] for f in sec.get("figures") or []]
    draws = {fid: [i for i, c in enumerate(claims, 1) if fid == c.get("finding_id")
                   or set(c.get("evidence") or []) & set(fig_records.get(fid) or [])]
             for fid in placed}
    figure_no = {i: [k for k, fid in enumerate(placed, 1) if i in draws[fid]]
                 for i in range(1, len(claims) + 1)}
    status = f"Status **{report.get('status')}**"
    if later:
        qts = QUESTION_TYPE_STATUS.get(str((plan or {}).get("question_type")))
        mean = STATUS_MEANING.get(str(report.get("status")))
        status = (f"{qts} " if qts else "") + status + (f" ({mean})" if mean else "")
    L = [f"# {_p(report.get('question') or '(no question recorded)')}", "",
         f"*{status}. Generated {report.get('generated_at')}. "
         f"The audit appendix: [{AUDIT_MD}]({AUDIT_MD}).*", ""]
    if note:
        L += [f"*{_p(note)}*", ""]
    L += [f"*{editorial_note(ed)}*", "", "## The answer", "", EDITORIAL_LABEL, "",
          _paragraph(ed.get("answer") or []), "", KEEP_IN_MIND, ""]
    L += [f"- {_paragraph([s])}" for s in ed.get("caveats") or []]
    qt = QUESTION_TYPE_LINES.get(str((plan or {}).get("question_type")))
    corrected = None
    if amended:
        one = len(amended) == 1
        corrected = (f"{_refs(sorted(amended)).capitalize()} {'is' if one else 'are'} shown "
                     f"with a correction made in review: the wording the audit kept, the "
                     f"corrected wording and the reason stand together in the evidence below "
                     f"and in the audit appendix.")
    code_lines = [qt, corrected, withheld_line(report),
                  None if any(c.get("evidence_class") == CAUSAL_CLASS for c in claims)
                  else NO_CAUSE]
    L += [f"- {x}" for x in code_lines if x]
    L += ["- The registry's exact wording of each limitation, and the standing limitations "
          "of the model setup, are in the audit appendix.", ""]
    by_field = claims_by_field(report, records)
    comp = [m for m in (plan or {}).get("competing") or [] if isinstance(m, dict) and m.get("text")]
    notes = CT._elm_modules()[1].load_notes()
    struck = " ".join(str(w.get("claim") or "") for w in report.get("withheld") or [])
    timing = timing_named(report, records)
    for sec in ed.get("sections") or []:
        L += [f"## {_p(sec.get('title'))}", "", EDITORIAL_LABEL, ""]
        for para in _paragraphs(sec):
            L += [_paragraph(para), ""]
        if sec.get("explanations") is not None:
            L += ["**The competing explanations the plan named before any result was seen, "
                  "each with its reading**", ""]
            if later:
                L += [STATUS_NOTE, ""]
            for ex, m in zip(sec["explanations"], comp):
                L.append(f"- The plan's words: \"{_p(m['text'])}\"  ")
                L.append(f"  **{STATUSES[ex['status']]}.** {_paragraph(ex.get('reading') or [])}"
                         .rstrip() + "  ")
                unread = [str(f) for f in m.get("fields") or [] if not by_field.get(str(f))]
                if unread:
                    named = [f"{f} ({_meaning(notes, f)})"
                             + ("; the audit withheld the claim on it"
                                if re.search(rf"\b{re.escape(f)}\b", struck) else "")
                             for f in unread]
                    L.append(f"  No accepted claim reads {', '.join(named)}.")
            L.append("")
        nums = list(sec.get("evidence") or [])
        if nums:
            L += [EVIDENCE_LABEL, ""] + _evidence(report, nums, figure_no, amended)
            L += [f"- **Editor's note on claim {s.get('on')}:** {_paragraph([s])}"
                  for s in sec.get("notes") or []]
            L += overlap_notes(report, nums, records, timing) + [""]
        for fid in sec.get("figures") or []:
            k = placed.index(fid) + 1
            L += [f"![Figure {k}]({Path(str(listed[fid])).name})", "",
                  f"*Figure {k}. {_p(questions.get(fid) or fid)}"
                  + (f" It draws {_refs(draws[fid])}." if draws[fid] else "") + "*", ""]
            beyond = figure_check_line(report, records, fig_records.get(fid) or [],
                                       checks) if later else None
            if beyond:
                L += [f"*{beyond}*", ""]
    L += ["## Remaining uncertainty and proposed sensitivity experiments", "",
          EDITORIAL_LABEL, "", _paragraph(ed.get("unresolved") or []), ""] \
        + experiment_lines(report) + [""]
    used = sorted({n for s in ed.get("answer") or [] for n in s.get("claims") or []})
    L += ["## Model and data", ""] + model_data_lines(report, records, [], used) + [""]
    sents = editorial_sentences(ed)
    free = sum(1 for s in sents if not s.get("claims"))
    if later:
        L += ["## Verification", ""] + coverage_lines(
            report, ed, records, {f: fig_records.get(f) or [] for f in placed}, review) + [""]
        return "\n".join(L)
    L += ["## Verification", "", verification_line(report)
          + f" The editorial synthesis was checked by code. {len(sents) - free} of its "
            f"{len(sents)} sentences name kept claims, and every number, year, season, month "
            f"and region in them is given by those claims' records. The other {free} state a "
            f"limitation, name no claim and print no number.", ""]
    return "\n".join(L)


def editorial_audit_lines(report: Dict[str, Any], records: Dict[str, Dict[str, Any]],
                          ed: Dict[str, Any], review: Optional[Dict[str, Any]] = None
                          ) -> List[str]:
    """For the appendix: what changed between the revisions of the synthesis,
    each correction of a kept claim with its original wording, the review
    check, the synthesis sentence by sentence with the claims and records
    each was checked against, and the answer's limitations in the registry's
    words (what the report page gives in plain words)."""
    claims = report.get("claims") or []
    extra, amended = _extra_records(ed), amendments(ed)
    checks = record_checks(report, review)
    L: List[str] = []
    if ed.get("history"):
        L += ["## The editorial synthesis: its revisions", "",
              "*The editor's own account of what changed and why. Each earlier revision is "
              "kept where it was written.*", ""]
        for h in ed["history"]:
            L.append(f"- **Revision {h.get('revision')}** ({_p(h.get('date') or 'undated')})"
                     + (f", kept at `{_p(h['where'])}`" if h.get("where") else "")
                     + (f": {_p(h['summary'])}" if h.get("summary") else ""))
            L += [f"  - {_p(x)}" for x in h.get("changes") or []]
        L.append("")
    if amended:
        L += ["## Corrections to claims the audit kept", "",
              "*A correction is editorial: it was made in review, after the run. The run's own "
              "record (analysis.json) is unchanged and still holds the wording the audit kept, "
              "which \"The whole record\" below shows.*", ""]
        for n in sorted(amended):
            a, c = amended[n], claims[n - 1]
            ids = [str(i) for i in a.get("records") or []]
            made = []
            for by, who in (("review", "the review check"),
                            ("run", "the numbers check of the run")):
                got = [i for i in ids if (checks.get(i) or {}).get("ok")
                       and checks[i]["by"] == by]
                if got:
                    made.append(f"{_cap(_ids_words(got))} {'was' if len(got) == 1 else 'were'} "
                                f"made again by {who} "
                                f"({sum(checks[i]['compared'] for i in got):,} numbers, no "
                                f"mismatch).")
            L += [f"- **Claim {n}, as the audit kept it:** \"{_p(c.get('claim'))}\" "
                  f"({_records_text([str(i) for i in c.get('evidence') or []])})",
                  f"  **Corrected wording** ({_p(a.get('date') or 'undated')}): "
                  f"\"{_p(a.get('text'))}\" ({_records_text(ids)})",
                  "  **Why:** " + " ".join(_p(s.get("text")) for s in a.get("reason") or []),
                  "  **Checks:** the corrected wording was traced to its records as a claim of "
                  "the same class is, with no problem. " + " ".join(made)]
        L.append("")
    if review:
        asked = review.get("asked") or {}
        L += ["## The review check", "",
              f"*{_p(review.get('note') or '')} Made {_p(review.get('checked_at') or '')}; "
              f"{int(review.get('n_compared') or 0):,} numbers compared, "
              f"{int(review.get('n_mismatched') or 0)} mismatched. Its record is "
              f"`review_numbers_check.json`, beside this page.*", ""]
        for i in sorted(asked, key=lambda r: int(r[1:]) if r[1:].isdigit() else 0):
            st, r = checks.get(i) or {}, records.get(i) or {}
            what = ", ".join(str(x) for x in (r.get("call") or st.get("call"), r.get("field"),
                                              r.get("period"), r.get("region")) if x)
            done = (f"{st['compared']:,} numbers made again, no mismatch" if st.get("ok") else
                    f"not made again: {_p(st.get('why') or 'the log keeps no number of it')}"
                    if st and not st.get("mismatched") else
                    "DID NOT PASS" if st else "in no check")
            L.append(f"- {i} ({_p(what)}); {'; '.join(_p(x) for x in asked[i])}: {done}.")
        for m in review.get("mismatches") or []:
            L.append(f"- MISMATCH {m.get('record')} {_p(m.get('path'))}: logged {m.get('logged')}, "
                     f"made again {m.get('recomputed')} ({_p(m.get('why'))})")
        L.append("")
    L += ["## The editorial synthesis, sentence by sentence", "",
          f"*{editorial_note(ed)}*", ""]
    for s in editorial_sentences(ed):
        ids = list(dict.fromkeys(str(i) for n in s.get("claims") or []
                                 for i in list(claims[n - 1].get("evidence") or [])
                                 + list(extra.get(n) or [])))
        L.append(f"- {s['where']}: \"{_p(s.get('text'))}\" "
                 + (f"Rests on {_refs(s['claims'])}; checked against {_id_runs(ids)}."
                    if s.get("claims") else "Names no claim and prints no number."))
    used = sorted({n for s in ed.get("answer") or [] for n in s.get("claims") or []})
    lim = limitation_lines(report, _answer_rows(ed), records, used)
    return L + ["", "## The answer's limitations, in the registry's words", ""] + lim + [""]


def verification_line(report: Dict[str, Any]) -> str:
    nc = report.get("numbers_check") or {}
    en = report.get("evidence_numbers") or {}
    parts = []
    if nc.get("n_compared") is not None:
        parts.append(f"the {nc['n_compared']:,} numbers of the regional table")
    if en.get("n_compared") is not None:
        parts.append(f"the {en['n_compared']:,} numbers of the lookups the claims cite")
    ok = (nc.get("pass") is not False) and (en.get("pass") is not False)
    held = len(report.get("withheld") or [])
    s = ""
    if parts:
        # ITS COVERAGE, SAID EXACTLY (2026-11-09): the run's check makes again
        # the lookups a claim cites, not what else a figure draws
        s = ("Every number in the accepted claims was made again from the model output by an "
             f"independent reading before this report was written ({' and '.join(parts)}), "
             + ("with no mismatch. " if ok else "and the appendix lists the mismatches. ")
             + "That reading covers the lookups the claims cite: a figure may also draw other "
               "records, a series among them, which it does not cover, and a ranking word is "
               "not held to records a claim leaves out. ")
    s += (f"Each claim was traced to its evidence records; the audit kept "
          f"{len(report.get('claims') or [])} and withheld {held}. Every check, record and "
          f"decision is in the [audit appendix]({AUDIT_MD}).")
    return s


# ── the two views ───────────────────────────────────────────────────────────

def render_main(report: Dict[str, Any], support: Sequence[Dict[str, Any]],
                records: Dict[str, Dict[str, Any]],
                questions: Optional[Dict[str, str]] = None,
                fig_records: Optional[Dict[str, Sequence[str]]] = None,
                plan: Optional[Dict[str, Any]] = None, note: Optional[str] = None) -> str:
    """REPORT.md, the scientific report. Pure. With a plan that carried out
    the recipe "seasonal evolution and spatial context", the page is in three
    parts (three_part_lines); else it lists the key findings."""
    from agents.analysis import analysis_recipes as AR
    claims = report.get("claims") or []
    q = _p(report.get("question") or "(no question recorded)")
    three = bool(plan) and bool(AR.report_lines(report.get("question"), plan))
    L = [f"# {q}", "",
         f"*Status **{report.get('status')}**. Generated {report.get('generated_at')}. "
         f"The audit appendix: [{AUDIT_MD}]({AUDIT_MD}).*", ""]
    if note:
        L += [f"*{_p(note)}*", ""]
    L += ["## The answer", ""]
    kept = kept_summary(support)
    tp = three_part_lines(report, support, records, plan, questions or {},
                          fig_records or {}) if three else None
    if tp and tp["lead"]:
        L += [tp["lead"], ""]
    elif kept and not three:
        L += [_said(kept), ""]
    elif report.get("headline"):
        L += [f"**{_p(report['headline'])}**", ""]
    # the limitations beside the answer are those of its own sentences and of
    # the claims shown as its evidence; each part below carries its own
    also = tp["tested_nums"] if three else ()
    lim = limitation_lines(report, tp["lead_rows"] if three else support, records, also)
    if lim:
        L += ["**Read the answer with these limitations:**", ""] + lim + [""]
    if three:
        L += tp["tested"] + tp["parts"]
        L += ["## Remaining uncertainty and proposed sensitivity experiments", ""] \
            + uncertainty_lines(report, support) + [""]
        L += ["## Model and data", ""] + model_data_lines(report, records, tp["lead_rows"],
                                                          also) + [""]
    else:
        fmap = figure_map(report, fig_records or {})
        L += ["## Model and data", ""] + model_data_lines(report, records, support) + [""]
        L += [f"## Key findings ({len(key_findings(report))} of {len(claims)} kept claims)", ""] \
            + finding_lines(report, records, fmap) + [""]
        if fmap:
            L += ["## Figures", ""] + figure_lines(report, questions or {}, fig_records)
        L += ["## Remaining uncertainty and proposed sensitivity experiments", ""] \
            + uncertainty_lines(report, support) + [""]
    L += ["## Verification", "", verification_line(report), ""]
    return "\n".join(L)


_N_EQ = re.compile(r"(?<=; )n = (\d+)(?=;)")
_AREA_NONE = re.compile(r";\s*region area km2 None cells")


def render_audit(report: Dict[str, Any], support: Sequence[Dict[str, Any]],
                 records: Dict[str, Dict[str, Any]],
                 withheld_notes: Optional[Dict[str, str]] = None,
                 editorial: Optional[Dict[str, Any]] = None,
                 review: Optional[Dict[str, Any]] = None) -> str:
    """AUDIT.md: the summary sentence by sentence, why each withheld claim
    was withheld, and the whole page the report used to be. The page is the
    record's own rendering (step4_report.render_markdown), with its figure
    paths made relative, "n = X" said in words and a malformed eligibility
    phrase of older records tidied; nothing else is changed."""
    q = _p(report.get("question") or "(no question recorded)")
    L = [f"# Audit appendix: {q}", "",
         f"*Every check, record and decision behind [the report]({MAIN_MD}), rendered from "
         f"the same analysis.json. The report's narrative uses only what the audit kept; "
         f"everything else is here.*", ""]
    if editorial:
        L += editorial_audit_lines(report, records, editorial, review)
    L += ["## The summary, sentence by sentence", ""]
    if editorial:
        L += ["*The Analyzer's own summary, which the editorial synthesis replaces on the "
              "report page.*", ""]
    for row in support or []:
        if row.get("unsupported"):
            L.append(f"- left out of the report: \"{_p(row['sentence'])}\" It prints "
                     f"{', '.join(row['unsupported'])}, which no kept claim's record gives.")
        else:
            L.append(f"- kept: \"{_p(row['sentence'])}\""
                     + (f" ({_refs(row['claims'])})" if row["claims"] else " (no number)"))
    L.append("")
    if report.get("withheld"):
        L += ["## Claims the audit withheld, and why", ""]
        for w in report["withheld"]:
            L += [f"- \"{_p(w.get('claim'))}\"",
                  f"  struck by `{w.get('struck_by')}`: {_p(w.get('struck_because'))}"]
            note = (withheld_notes or {}).get(str(w.get("claim"))) or {}
            if note.get("reason"):
                L.append(f"  The rule applied again to its records: {_p(note['reason'])}")
            if w.get("struck_by") == "percent" and note.get("percent_of"):
                # A CURRENT LIMITATION, NOT A WRONG PERCENTAGE (the user's
                # decision of 2026-11-09)
                L.append(f"  This is a current limitation of the percent rule, not an error "
                         f"in the percentage: the rule judges a claim as a whole, so a claim "
                         f"that names {', '.join(note['no_percent'])}, whose notes give no "
                         f"percent, is withheld even when its percent is of "
                         f"{', '.join(note['percent_of'])}, whose notes allow one, and matches "
                         f"that record. Until the rule attributes each percent to its own "
                         f"quantity, such results are given as separate claims, one per "
                         f"quantity.")
            elif w.get("struck_by") == "percent":
                L.append("  The percent rule judges a claim whole: a claim naming a field "
                         "whose notes give no percent prints no percent.")
        L.append("")
    L += ["## The whole record", "", "*The page the report used to be, as the record renders "
          "it.*", ""] + record_page(report).splitlines()
    return "\n".join(L) + "\n"


def record_page(report: Dict[str, Any]) -> str:
    """The page the report used to be (step4_report.render_markdown), as the
    appendix holds it: its headings one level down, its figure paths made
    relative, "n = X" said in words and the malformed eligibility phrase of
    older records taken out. Nothing else is changed."""
    from agents.analysis import step4_report as S4
    page = S4.render_markdown(report)
    for path in ((report.get("figures") or {}).get("investigation") or {}).values():
        page = page.replace(str(path), Path(str(path)).name)
    page = _N_EQ.sub(lambda m: f"values drawn in its figure: {m.group(1)}", page)
    page = _AREA_NONE.sub("", page)
    return "\n".join(("#" + line) if line.startswith("#") else line
                     for line in page.splitlines())


def withheld_notes(analysis_dir, records: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """{a claim struck by the percent rule: {"reason" (the rule's reason,
    given again from its own records), "no_percent" (its fields whose notes
    give none), "percent_of" (the other fields whose percent value it
    prints)}}: a record from before 2026-11-09 kept the reason "a percent of
    TSA" for a percent that was leaf area's."""
    try:
        it = json.loads((Path(analysis_dir) / "interpretation.json").read_text())
    except (OSError, ValueError):
        return {}
    out = {}
    for c in it.get("struck") or []:
        text = str(c.get("claim") or "")
        recs = [records[i] for i in c.get("evidence") or [] if i in records]
        why = CT.percent_problems(text, recs) if recs else []
        if why:
            none = sorted({str(r.get("field")) for r in recs if r.get("percent_allowed") is False})
            of = sorted({f for fs in CT.percent_attribution(text, recs).values() for f in fs})
            out[text] = {"reason": why[0], "no_percent": none,
                         "percent_of": [f for f in of if f not in none]}
    return out


def figure_records(analysis_dir) -> Dict[str, List[str]]:
    """Each drawn figure's evidence records, from the investigation's own
    findings (a figure is a finding with a figure)."""
    p = Path(analysis_dir) / "investigation.json"
    try:
        inv = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return {str(f.get("id")): [str(x.get("id")) for x in f.get("lookups") or []
                               if isinstance(x, dict) and x.get("id")]
            for f in inv.get("findings") or [] if f.get("figure")}


def figure_questions(spec_dir) -> Dict[str, str]:
    """Each drawn figure's own question, from its recorded specification."""
    out = {}
    for p in Path(spec_dir).glob("*.spec.json"):
        try:
            out[p.name[:-len(".spec.json")]] = str(json.loads(p.read_text()).get("question")
                                                   or "")
        except (OSError, ValueError):
            continue
    return out


def write_views(report: Dict[str, Any], out_dir, records: Dict[str, Dict[str, Any]],
                runs: Dict[str, Dict[str, Any]], plan: Dict[str, Any],
                figure_dir=None, analysis_dir=None, note: Optional[str] = None,
                editorial: Optional[Dict[str, Any]] = None,
                review: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """REPORT.md and AUDIT.md in out_dir, from one report record and its
    evidence records, with every figure the record names copied beside them
    from figure_dir (or from where the record says it is), so their links
    are relative. analysis_dir: the study's 04_analysis (its specifications,
    investigation and interpretation), by default where the figures were.
    review: a review check's record (the numbers check run again on lookups
    no claim cites), for an editorial page of a later revision."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    support = CT.summary_support(report.get("answer") or "", report.get("claims") or [],
                                 records, runs, plan)
    for fid, path in ((report.get("figures") or {}).get("investigation") or {}).items():
        src = Path(figure_dir) / Path(str(path)).name if figure_dir else Path(str(path))
        if src.is_file():
            shutil.copyfile(src, out / src.name)
    spec_dir = Path(analysis_dir) if analysis_dir else Path(str(next(iter(
        ((report.get("figures") or {}).get("investigation") or {"": ""}).values())))).parent
    questions = figure_questions(spec_dir) if spec_dir.is_dir() else {}
    if editorial:                                # raises before anything is written
        main = render_editorial(report, records, plan, runs, editorial, questions,
                                figure_records(spec_dir), note=note, review=review)
    else:
        main = render_main(report, support, records, questions, figure_records(spec_dir),
                           plan=plan, note=note)
    (out / MAIN_MD).write_text(main)
    (out / AUDIT_MD).write_text(render_audit(report, support, records,
                                             withheld_notes(spec_dir, records),
                                             editorial=editorial, review=review))
    return {"main": str(out / MAIN_MD), "audit": str(out / AUDIT_MD)}
