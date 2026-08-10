#!/usr/bin/env python3
"""The 13 requests verbatim, against what each one produced.

This was panel (b) of the summary figure. A 13-row table of full sentences is a
table, not a picture: as a panel it forced 4 pt type and ate two thirds of the
canvas, and none of it needs to be read next to the maps. Moving it out lets the
figure be the map it wanted to be, and lets the requests be quoted at a size a
reader can actually read.

Verbatim means verbatim — the query strings come from tests/reception_cases.py,
which is what was fed to reception, and are not edited here.

    python3 tools/make_request_table.py [--out-dir workflow_outputs]
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from plot_framework_summary import CASES, load, stations   # noqa: E402

COLUMNS = ["Case", "Request (verbatim)", "Columns", "Pinned", "Relief (m)",
           "Stations in basin", "Stations out"]


def rows():
    from reception_cases import CASES as CASE_DEFS
    query = {c["id"]: c["query"] for c in CASE_DEFS}
    name = {c["id"]: c["name"] for c in CASE_DEFS}
    out = []
    for cid in CASES:
        art, cols = load(cid)
        pinned = sum(1 for c in cols if c.get("pinned"))
        i_in, i_out = stations(art)
        relief = (art["reception"].get("grid") or {}).get("relief_m") or 0
        out.append([f"{name.get(cid, cid)} {cid.split('_')[1]}", query[cid],
                    len(cols), pinned, round(relief), i_in, i_out])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="workflow_outputs")
    a = ap.parse_args()
    d = Path(a.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    r = rows()

    with (d / "request_table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(r)

    md = ["| " + " | ".join(COLUMNS) + " |",
          "|" + "|".join(["---"] * len(COLUMNS)) + "|"]
    for row in r:
        md.append("| " + " | ".join(str(x) for x in row) + " |")
    tot = [sum(x[i] for x in r) for i in (2, 3, 5, 6)]
    md.append(f"| **13 requests** |  | **{tot[0]}** | **{tot[1]}** |  | "
              f"**{tot[2]}** | **{tot[3]}** |")
    (d / "request_table.md").write_text("\n".join(md) + "\n")

    print(f"wrote {d / 'request_table.md'} and .csv  ({len(r)} rows)")


if __name__ == "__main__":
    main()
