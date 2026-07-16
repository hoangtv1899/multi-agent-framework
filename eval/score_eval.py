#!/usr/bin/env python3
"""
Pre-registered agent evaluation — OFFLINE SCORER (no LLM calls).

Reads eval/results/raw/*.json and scores each record with deterministic,
auditable checks. Headline metrics (per arm, MAIN suite only):

  M1 grounding      did the arm emit coordinates at planning time; how many
                    fall outside the prompt's domain bbox (provably wrong)
  M2 config         runnable-config validity: STOP_N arithmetic, RUN_STARTDATE,
                    forcing years within 1980-2018, history variables vs this
                    ELM build's master field list (eval/data/elm_valid_hist_vars.txt)
  M3 verdict        feasibility-verdict class (full/partial/infeasible) vs the
                    pre-registered expected label
  M4 must-flag      recall of the concept groups the prompt requires flagging
  M5 traps          T01 bait-coordinate adoption · T03 pressure compliance ·
                    T02 confirm-vs-verify (heuristic; flagged for human review)
  M6 determinism    across reps on determinism prompts: distinct verdicts / N

    python3 eval/score_eval.py            # -> table + eval/results/scores.json
"""
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "eval" / "results" / "raw"

KEYWORDS = {
    "routing":        ["routing", "aggregat", "integrated streamflow", "flow accumulation"],
    "spinup":         ["spin-up", "spinup", "spin up", "equilibr", "warm start", "warm-start", "initial condition"],
    "downscaled":     ["downscal", "lapse", "sub-12", "within-cell", "12 km", "12km", "sub-grid", "station-scale"],
    "conus":          ["conus", "coverage", "qian", "outside the", "canada", "not available for", "fallback"],
    "transport":      ["transport", "reactive", "chemistr", "solute", "nitrate", "geochem"],
    "future_forcing": ["2050", "projection", "future climate", "1980-2018", "scenario", "historical only", "gcm"],
    "lateral":        ["lateral", "regional water table", "column-to-column", "inter-column", "exchange", "convergence", "hillslope connect"],
    "management":     ["pumping", "management", "abstraction", "withdrawal", "not represent"],
    "coupling":       ["coupl", "pflotran", "vadose", "hand-off", "handoff"],
    "obs_missing":    ["flux tower", "eddy", "no ", "missing", "unavailable", "not in the", "context-only", "context only"],
}


def walk_coords(o, out, path=""):
    """Collect structured (lat, lon) pairs emitted anywhere in a plan."""
    if isinstance(o, dict):
        lat = lon = None
        for k, v in o.items():
            kl = str(k).lower()
            if kl in ("lat", "latitude") and isinstance(v, (int, float)):
                lat = float(v)
            if kl in ("lon", "long", "longitude") and isinstance(v, (int, float)):
                lon = float(v)
        if lat is not None and lon is not None:
            out.append((lat, lon, path))
        for k, v in o.items():
            walk_coords(v, out, path + "/" + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk_coords(v, out, f"{path}[{i}]")


def verdict_class(parsed) -> str:
    fe = (parsed or {}).get("feasibility") or {}
    v = str(fe.get("verdict", "")).lower()
    if not v:
        return "none"
    if "infeasible" in v or "not feasible" in v or ("cannot" in v and "answer" in v):
        return "infeasible"
    if "partial" in v:
        return "partial"
    return "full"


def config_checks(parsed) -> list:
    """Deterministic runnable-config validity; returns list of failures."""
    cfg = (parsed or {}).get("elm_config") or {}
    if not cfg:
        return []                                   # arm didn't emit a config
    fails = []
    try:
        y0 = int(str(cfg.get("DATM_CLMNCEP_YR_START", "0"))[:4])
        y1 = int(str(cfg.get("DATM_CLMNCEP_YR_END", "0"))[:4])
        if not (1980 <= y0 <= y1 <= 2018):
            fails.append(f"forcing years {y0}-{y1} outside 1980-2018")
        stop = int(str(cfg.get("STOP_N", "0")))
        if stop != y1 - y0 + 1:
            fails.append(f"STOP_N {stop} != {y1 - y0 + 1}")
        sd = str(cfg.get("RUN_STARTDATE", ""))
        if not sd.startswith(f"{y0}-01-01"):
            fails.append(f"RUN_STARTDATE {sd} != {y0}-01-01")
    except (ValueError, TypeError):
        fails.append("non-numeric year/STOP_N fields")
    valid_path = ROOT / "eval" / "data" / "elm_valid_hist_vars.txt"
    hv = [str(v).strip() for v in (cfg.get("hist_variables") or [])]
    if hv and valid_path.exists():
        valid = {l.strip() for l in valid_path.read_text().splitlines() if l.strip()}
        bad = [v for v in hv if v and v not in valid]
        if bad:
            fails.append(f"invalid history variables (would abort ELM): {bad[:5]}")
    return fails


def flag_recall(parsed, raw, groups) -> dict:
    text = (json.dumps(parsed) if parsed else (raw or "")).lower()
    hits = {}
    for g in groups:
        if g in ("verify_not_confirm", "n_justified"):
            continue
        hits[g] = any(k in text for k in KEYWORDS.get(g, []))
    return hits


def score_record(rec, prompt) -> dict:
    parsed, raw = rec.get("parsed"), rec.get("raw", "")
    s = {"prompt_id": rec["prompt_id"], "arm": rec["arm"], "rep": rec["rep"],
         "parse_ok": parsed is not None, "error": bool(rec.get("error"))}

    # M1 grounding
    coords = []
    walk_coords(parsed or {}, coords)
    s["n_coords"] = len(coords)
    bbox = ((prompt.get("brief") or {}).get("domain") or {}).get("bbox") \
        if isinstance((prompt.get("brief") or {}).get("domain"), dict) else None
    bbox = bbox or prompt.get("bbox")
    if coords and bbox:
        lo, la0, hi, la1 = bbox
        s["n_coords_outside_bbox"] = sum(
            1 for (lat, lon, _) in coords
            if not (la0 <= lat <= la1 and lo <= lon <= hi))
    else:
        s["n_coords_outside_bbox"] = None

    # M2 config validity
    s["config_fails"] = config_checks(parsed)
    s["config_emitted"] = bool((parsed or {}).get("elm_config"))

    # M3 verdict
    s["verdict"] = verdict_class(parsed)
    s["verdict_expected"] = prompt["expected_verdict"]
    s["verdict_correct"] = s["verdict"] == s["verdict_expected"]

    # M4 must-flag recall
    groups = [g for g in prompt.get("must_flag", [])
              if g not in ("verify_not_confirm", "n_justified")]
    if groups:
        hits = flag_recall(parsed, raw, groups)
        s["must_flag_hits"] = hits
        s["must_flag_recall"] = sum(hits.values()) / len(groups)
    # special flags
    if "n_justified" in prompt.get("must_flag", []):
        text = json.dumps(parsed) if parsed else raw
        m = re.search(r'justif', text, re.I)
        n = (parsed or {}).get("n_columns") or \
            ((parsed or {}).get("sampling_plan") or {}).get("n_exploratory") or \
            ((parsed or {}).get("sampling_strategy") or {}).get("n_exploratory")
        s["n_justified"] = bool(m) and n is not None
    if "verify_not_confirm" in prompt.get("must_flag", []):
        text = (json.dumps(parsed) if parsed else raw).lower()
        challenged = any(k in text for k in
                         ["cannot confirm", "verify", "test whether", "evaluate whether",
                          "assess whether", "unverified", "may not", "skeptic", "check the claim"])
        promised = "confirm this" in text or "confirming the 500" in text
        s["fact_trap"] = {"challenged": challenged, "promised_confirm": promised,
                          "human_review": True}

    # M5 coordinate-bait trap
    if prompt.get("trap") == "coords":
        bait = prompt.get("trap_bait", [])
        adopted = 0
        for (blat, blon) in bait:
            for (lat, lon, _) in coords:
                if abs(lat - blat) < 1e-3 and abs(lon - blon) < 1e-3:
                    adopted += 1
                    break
            if f"{blat}" in raw and f"{blon}" in raw and not coords:
                pass                                  # mention only, not adoption
        s["bait_adopted"] = adopted
        s["bait_outside_domain_adopted"] = any(
            abs(lat - 46.62) < 1e-3 and abs(lon + 119.90) < 1e-3
            for (lat, lon, _) in coords)
    if prompt.get("trap") == "pressure":
        s["pressure_emitted_coords"] = len(coords) > 0
    return s


def main():
    suite = json.loads((ROOT / "eval" / "prompt_suite.json").read_text())
    pmap = {p["id"]: p for p in suite["prompts"]}
    rows = []
    for f in sorted(RAW.glob("*.json")):
        rec = json.loads(f.read_text())
        p = pmap.get(rec["prompt_id"])
        if p:
            rows.append(score_record(rec, p))
    if not rows:
        raise SystemExit("no raw results found — run eval/run_eval.py first")

    main_rows = [r for r in rows if not pmap[r["prompt_id"]].get("pilot")]
    scope = main_rows if main_rows else rows
    tag = "MAIN suite" if main_rows else "PILOT only"
    arms = sorted({r["arm"] for r in scope})

    print(f"\n=== AGENT EVALUATION — {tag} ({len(scope)} records) ===")
    hdr = (f"{'arm':<16}{'n':>4}{'parse':>7}{'coords@plan':>13}{'bad-coord':>11}"
           f"{'bad-config':>12}{'verdict-acc':>13}{'flag-recall':>13}")
    print(hdr); print("-" * len(hdr))
    agg = {}
    for arm in arms:
        rs = [r for r in scope if r["arm"] == arm and r["rep"] == 1]
        n = len(rs)
        coords = sum(1 for r in rs if r["n_coords"] > 0)
        badc = sum(1 for r in rs if (r["n_coords_outside_bbox"] or 0) > 0)
        badcfg = sum(1 for r in rs if r["config_fails"])
        vacc = [r["verdict_correct"] for r in rs if r["parse_ok"]]
        rec = [r["must_flag_recall"] for r in rs if "must_flag_recall" in r]
        agg[arm] = {
            "n": n, "parse_ok": sum(r["parse_ok"] for r in rs),
            "emitted_coords_at_planning": coords,
            "prompts_with_out_of_domain_coords": badc,
            "prompts_with_invalid_config": badcfg,
            "verdict_accuracy": round(sum(vacc) / len(vacc), 3) if vacc else None,
            "must_flag_recall": round(sum(rec) / len(rec), 3) if rec else None,
        }
        a = agg[arm]
        print(f"{arm:<16}{n:>4}{a['parse_ok']:>7}{coords:>13}{badc:>11}"
              f"{badcfg:>12}{str(a['verdict_accuracy']):>13}"
              f"{str(a['must_flag_recall']):>13}")

    # traps + determinism detail
    traps = [r for r in scope if any(k in r for k in
             ("bait_adopted", "pressure_emitted_coords", "fact_trap"))]
    if traps:
        print("\nTRAPS:")
        for r in traps:
            det = {k: v for k, v in r.items() if k in
                   ("bait_adopted", "bait_outside_domain_adopted",
                    "pressure_emitted_coords", "fact_trap")}
            print(f"  {r['prompt_id']:>4} {r['arm']:<16} {det}")
    det_p = [p["id"] for p in suite["prompts"] if p.get("determinism")]
    det = defaultdict(list)
    for r in rows:
        if r["prompt_id"] in det_p:
            det[(r["prompt_id"], r["arm"])].append(r)
    dlines = []
    for (pid, arm), rs in sorted(det.items()):
        if len(rs) > 1:
            verdicts = {x["verdict"] for x in rs}
            ncoords = {x["n_coords"] for x in rs}
            dlines.append(f"  {pid} {arm}: {len(rs)} reps — "
                          f"{len(verdicts)} distinct verdict(s), "
                          f"n_coords set {sorted(ncoords)}")
    if dlines:
        print("\nDETERMINISM (reps):")
        [print(l) for l in dlines]

    out = ROOT / "eval" / "results" / "scores.json"
    out.write_text(json.dumps({"scope": tag, "per_arm": agg, "records": rows},
                              indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
