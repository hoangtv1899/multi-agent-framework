#!/usr/bin/env python3
"""Phase 1d — does build_elm_inputs_from_location write what the old path wrote?

docs/ELM_MCP_PLAN.md §9 phase 1d. Runs the tool against a completed run's
columns.json and compares the case_inputs.json it produces, field for field,
against the one that run actually built and ran from.

    python3 scripts/parity_case_inputs.py <completed-run-dir> [<scratch-dir>]

Paths under the run directory are normalised before comparison: FINIDAT is
subset into <run_dir>/warmstart/, so a different run directory legitimately
yields a different absolute path and an identical file. Everything else must
match exactly.

Also asserts the two failures that shipped before: an ELMAgentAdapter repr in
the file (the object crossing instead of its config), and a case with no
runtime_config at all — a file that exists, parses, and names nothing the build
needs.
"""
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parents[3] / "src"))


def norm(v, rd):
    if isinstance(v, str) and v.startswith(str(rd)):
        return "<RUNDIR>" + v[len(str(rd)):]
    return v


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    old_rd = Path(argv[1]).resolve()
    new_rd = Path(argv[2]).resolve() if len(argv) > 2 else Path("/tmp/elm_parity")

    old_f = old_rd / "01_inputs" / "case_inputs.json"
    if not old_f.is_file():
        sys.exit(f"no case_inputs.json under {old_rd}")

    cols = old_rd / "01_inputs" / "columns.json"
    if not cols.is_file():
        cols = old_rd / "columns.json"
    (new_rd / "01_inputs").mkdir(parents=True, exist_ok=True)
    (new_rd / "01_inputs" / "columns.json").write_text(cols.read_text())

    import importlib.util
    spec = importlib.util.spec_from_file_location("m", HERE.parent / "main.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    rc0 = json.loads(old_f.read_text())[0]["runtime_config"]
    out = json.loads(m.build_elm_inputs_from_location(
        run_dir=str(new_rd),
        yr_start=int(rc0["DATM_CLMNCEP_YR_START"]),
        yr_end=int(rc0["DATM_CLMNCEP_YR_END"])))
    if not out.get("ok"):
        sys.exit(f"the tool failed: {out.get('error')}")

    old = {r["case_name"]: r for r in json.loads(old_f.read_text())}
    new = {r["case_name"]: r
           for r in json.loads((new_rd / "01_inputs" / "case_inputs.json").read_text())}

    bad = []
    if set(old) != set(new):
        bad.append(f"case names differ: {set(old) ^ set(new)}")

    n_field = n_rc = 0
    for name in sorted(set(old) & set(new)):
        o, n = old[name], new[name]
        for k in sorted(set(o) | set(n)):
            if k in ("case_dir", "runtime_config"):
                continue                       # case_dir is written by the BUILD
            n_field += 1
            if o.get(k) != n.get(k):
                bad.append(f"{name}.{k}: {o.get(k)!r} vs {n.get(k)!r}")
        ro, rn = o.get("runtime_config") or {}, n.get("runtime_config") or {}
        for k in sorted(set(ro) | set(rn)):
            n_rc += 1
            a, b = norm(ro.get(k), old_rd), norm(rn.get(k), new_rd)
            if a != b:
                bad.append(f"{name}.runtime_config.{k}: {a!r} vs {b!r}")

    raw = (new_rd / "01_inputs" / "case_inputs.json").read_text()
    reprs = re.findall(r"ELMAgentAdapter|<[a-zA-Z_.]+ object at 0x", raw)
    if reprs:
        bad.append(f"{len(reprs)} object repr(s) in the file")
    empty = [n for n, r in new.items() if not (r.get("runtime_config") or {})]
    if empty:
        bad.append(f"cases with no runtime_config: {empty}")
    gone = [n for n, r in new.items()
            if not Path(r["runtime_config"].get("FINIDAT", "")).is_file()]
    if gone:
        bad.append(f"FINIDAT named but absent: {gone}")

    print(f"cases            {len(old)} old, {len(new)} new")
    print(f"fields compared  {n_field} + {n_rc} runtime_config values")
    print(f"object reprs     {len(reprs)}")
    if bad:
        print(f"\nFAIL — {len(bad)} difference(s):")
        for b in bad[:20]:
            print("   " + b)
        return 1
    print("\nPARITY — identical, modulo run-directory paths")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
