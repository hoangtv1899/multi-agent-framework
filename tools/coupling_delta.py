#!/usr/bin/env python3
"""
Has an ELM<->PFLOTRAN iteration converged? Read one PFLOTRAN leg and say.
tools/coupling_delta.py

    python tools/coupling_delta.py <pflotran_run_dir> [--tol 0.10]

A PFLOTRAN leg's own record holds both halves of the question: the anchor
each column was GIVEN (columns.json `water_table_m` — ELM's answer from the
leg before) and where PFLOTRAN's water table actually SAT (the pressure
crossing at the final output time, read by the one public function that owns
that definition, beside the PFLOTRAN server). The delta per column is
solved minus given; the iteration is done when the largest |delta| is under
the tolerance.

Exit code: 0 converged, 3 not converged, 1 error — so the coupling loop's
shell can branch on it. Prints one line per column and one verdict line.
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


def deltas(run_dir):
    rd = Path(run_dir)
    cols = json.loads((rd / "columns.json").read_text())
    cols = cols.get("columns", cols) if isinstance(cols, dict) else cols
    data = json.loads((rd / "03_results" / "extracted.json").read_text())
    data = data.get("columns") or {}
    solved_of = _solved_fn()
    out = []
    for c in cols:
        cid = c.get("id")
        given = c.get("water_table_m")
        solved = solved_of(data.get(cid) or {})
        out.append({"id": cid, "given_m": given, "solved_m": solved,
                    "delta_m": (None if solved is None or given is None
                                else round(solved - float(given), 3))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--tol", type=float, default=0.10,
                    help="metres; converged when every |delta| is under this")
    a = ap.parse_args()
    rows = deltas(a.run_dir)
    worst = 0.0
    for r in rows:
        d = r["delta_m"]
        print(f"  {r['id']}: given {r['given_m']} m -> solved {r['solved_m']} m"
              f"  (delta {d if d is not None else 'n/a — left the domain'})")
        if d is None:
            print(f"CONVERGENCE UNDECIDABLE: {r['id']} has no water table in "
                  f"the domain")
            sys.exit(1)
        worst = max(worst, abs(d))
    ok = worst <= a.tol
    print(f"{'CONVERGED' if ok else 'NOT CONVERGED'}: max |delta| "
          f"{worst:.3f} m vs tol {a.tol:g} m")
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
