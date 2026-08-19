#!/usr/bin/env python3
"""
Has an ELM<->PFLOTRAN iteration converged? Read one PFLOTRAN leg and say.
tools/coupling_delta.py

    python tools/coupling_delta.py <pflotran_run_dir> [--tol 0.10]

CONVERGENCE IS MOVEMENT BETWEEN ITERATIONS, NOT AN OFFSET INSIDE ONE. The
question a loop asks is "did anything change since last time?", so the test
compares THIS leg's solved water table against the PREVIOUS leg's solved
water table, column by column. The previous leg is found by walking the
record: this run's columns name the ELM run they were coupled from
(`coupled_from`), and that run's reception names the PFLOTRAN run IT was
coupled from (`coupling.prior_run_dir`).

WHY NOT GIVEN-MINUS-SOLVED, which this file used to test. A PFLOTRAN leg
also records the anchor each column was GIVEN (columns.json
`water_table_m` — ELM's answer from the leg before), and solved-minus-given
looks like a convergence measure. It is not: it is a WITHIN-RUN offset
between the anchor and where the solver puts the crossing, and on Brandywine
it sat at exactly +0.050 m in every iteration while the state itself moved
46.8 m -> 28.3 m -> 27.8 m. Judged on that number the loop declared
CONVERGED at iteration 0 with the answer still 18 m away (tol 0.10), and
NOT CONVERGED forever at tol 0.02 — wrong in both directions, because the
quantity does not shrink with iteration. It is kept and printed as a
diagnostic, and never as the verdict.

The FIRST leg of a chain has no previous leg. That is reported as
undecidable rather than converged: an iteration that has not happened yet
has not converged.

Exit code: 0 converged, 3 not converged, 1 undecidable/error — so the
coupling loop's shell can branch on it. Prints one line per column and one
verdict line.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _solved_fn():
    import importlib.util
    pkg = ROOT / "mcp" / "pflotran-mcp" / "compare"
    spec = importlib.util.spec_from_file_location(
        "compare_pflotran", pkg / "__init__.py",
        submodule_search_locations=[str(pkg)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("compare_pflotran", mod)
    spec.loader.exec_module(mod)
    from compare_pflotran import water_table as wt
    return wt.solved_water_table_m


def _solved_by_column(run_dir):
    """{column id: PFLOTRAN's solved water table} for one leg, plus the anchors.

    Both come off the leg's own record: the anchor each column was GIVEN is
    on columns.json, the solved depth is read out of the extract by the one
    public function that owns that definition.
    """
    rd = Path(run_dir)
    cols = json.loads((rd / "columns.json").read_text())
    cols = cols.get("columns", cols) if isinstance(cols, dict) else cols
    data = json.loads((rd / "03_results" / "extracted.json").read_text())
    data = data.get("columns") or {}
    solved_of = _solved_fn()
    given, solved = {}, {}
    for c in cols:
        cid = c.get("id")
        given[cid] = c.get("water_table_m")
        solved[cid] = solved_of(data.get(cid) or {})
    return given, solved, [c.get("id") for c in cols]


def previous_leg(run_dir):
    """The PFLOTRAN leg before this one, walked out of the record.

    this pflotran run -> its columns' `coupled_from` (the ELM leg) -> that
    run's reception `coupling.prior_run_dir` (the pflotran leg before it).
    Returns None for the first leg of a chain, which has no predecessor.
    """
    rd = Path(run_dir).resolve()
    try:
        cols = json.loads((rd / "columns.json").read_text())
        cols = cols.get("columns", cols) if isinstance(cols, dict) else cols
        prior_elm = (cols[0] or {}).get("coupled_from")
    except Exception:                                           # noqa: BLE001
        return None
    if not prior_elm:
        return None
    elm_dir = Path(prior_elm)
    if not elm_dir.is_absolute():
        elm_dir = rd.parent / prior_elm
    try:
        rec = json.loads((elm_dir / "reception.json").read_text())
        prev = (((rec.get("brief") or {}).get("coupling") or {})
                .get("prior_run_dir"))
    except Exception:                                           # noqa: BLE001
        return None
    if not prev:
        return None
    p = Path(prev)
    if not p.is_absolute():
        p = rd.parent / prev
    return p if (p / "columns.json").is_file() else None


def deltas(run_dir, prior_run_dir=None):
    """Per column: how far the solved water table MOVED since the leg before.

    `delta_m` is this leg's solved depth minus the previous leg's solved
    depth — the quantity that goes to zero as the loop settles. The anchor
    offset (solved minus given, within this run) rides along as
    `offset_m`, printed but never judged: see the module docstring.
    """
    given, solved, ids = _solved_by_column(run_dir)
    prior = prior_run_dir if prior_run_dir is not None else previous_leg(run_dir)
    prev_solved = _solved_by_column(prior)[1] if prior else {}
    out = []
    for cid in ids:
        s, g = solved.get(cid), given.get(cid)
        ps = prev_solved.get(cid)
        out.append({
            "id": cid, "given_m": g, "solved_m": s,
            "prev_solved_m": ps,
            "offset_m": (None if s is None or g is None
                         else round(s - float(g), 3)),
            "delta_m": (None if s is None or ps is None
                        else round(float(s) - float(ps), 3))})
    return out, (str(prior) if prior else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--tol", type=float, default=0.10,
                    help="metres; converged when no column MOVED more than "
                         "this since the previous leg")
    ap.add_argument("--prior", default=None,
                    help="the previous PFLOTRAN leg (default: walk the record)")
    a = ap.parse_args()
    rows, prior = deltas(a.run_dir, a.prior)

    if prior is None:
        for r in rows:
            print(f"  {r['id']}: solved {r['solved_m']} m "
                  f"(anchor offset {r['offset_m']})")
        print("CONVERGENCE UNDECIDABLE: this is the first leg of the chain — "
              "there is no previous iteration to have moved from")
        sys.exit(1)

    print(f"  (comparing against {Path(prior).name})")
    worst = 0.0
    for r in rows:
        d = r["delta_m"]
        print(f"  {r['id']}: was {r['prev_solved_m']} m -> now {r['solved_m']} m"
              f"  (moved {d if d is not None else 'n/a — left the domain'}"
              f"; anchor offset {r['offset_m']})")
        if d is None:
            print(f"CONVERGENCE UNDECIDABLE: {r['id']} has no water table in "
                  f"the domain in one of the two legs")
            sys.exit(1)
        worst = max(worst, abs(d))
    ok = worst <= a.tol
    print(f"{'CONVERGED' if ok else 'NOT CONVERGED'}: max movement "
          f"{worst:.3f} m vs tol {a.tol:g} m")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
