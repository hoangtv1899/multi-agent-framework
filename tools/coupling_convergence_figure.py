#!/usr/bin/env python3
"""
The two-way loop's convergence, drawn from the record.
tools/coupling_convergence_figure.py

    python tools/coupling_convergence_figure.py [pflotran_run_dir]

One figure, two panels, nothing interpreted:
  (a) the water-table depth each PFLOTRAN round solved, per column, with
      round 0 being the state the chain started from (the anchor the first
      leg was given — ELM's own answer);
  (b) how far each column MOVED between successive rounds — the number
      tools/coupling_delta.py judges — against the loop's tolerance.

The chain is walked out of the record exactly the way the convergence
checker walks it (coupled_from -> the ELM leg -> coupling.prior_run_dir),
so the figure and the verdict can never disagree about which runs are in
the iteration. Re-run it after another loop lands and the new rounds
appear; no run names live in this file.

Defaults to the newest workflow_outputs/pflotran_run_* that is a coupled
leg. Writes docs/paper/fig_coupling_convergence.png and .pdf.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import coupling_delta as cd                                  # noqa: E402

TOL_M = 0.10          # the loop's default; --tol on the checker, drawn here


def chain_of(run_dir):
    """Every PFLOTRAN leg in this iteration, oldest first."""
    legs = [Path(run_dir).resolve()]
    while True:
        prev = cd.previous_leg(legs[0])
        if not prev:
            break
        legs.insert(0, Path(prev).resolve())
    return legs


def main(run_dir):
    legs = chain_of(run_dir)
    given0, solved, ids = cd._solved_by_column(legs[0])
    series = {cid: [given0[cid], solved[cid]] for cid in ids}
    for leg in legs[1:]:
        _, s, _ = cd._solved_by_column(leg)
        for cid in ids:
            series[cid].append(s.get(cid))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figstyle import manuscript, WIDTH

    k = manuscript(WIDTH["double"], base_pt=8)
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(WIDTH["double"], WIDTH["double"] * 0.42))
    fig.subplots_adjust(wspace=0.34)

    rounds = list(range(len(legs) + 1))
    labels = ["start\n(ELM's answer)"] + [f"round {i}" for i in rounds[1:]]
    styles = ["-", "-", "--", "-.", ":"]
    for n, cid in enumerate(ids):
        ax.plot(rounds, series[cid], marker="o", markersize=4, label=cid,
                linestyle=styles[n % len(styles)])
    ax.set_xticks(rounds, labels)
    ax.invert_yaxis()
    ax.set_ylabel("water-table depth (m)")
    ax.legend()
    ax.set_title("(a) the depth each round solved")

    per_round = {}
    for n, cid in enumerate(ids):
        s = series[cid]
        moves = [abs(s[i] - s[i - 1]) for i in range(1, len(s))
                 if s[i] is not None and s[i - 1] is not None]
        xs = list(range(1, len(moves) + 1))
        bx.plot(xs, moves, marker="o", markersize=4, label=cid,
                linestyle=styles[n % len(styles)])
        for x, m in zip(xs, moves):
            per_round[x] = max(per_round.get(x, 0.0), m)
    # one number per round — the max, which is what the verdict reads
    for x, m in per_round.items():
        bx.annotate(f"max {m:.3g}", (x, m), textcoords="offset points",
                    xytext=(6, 5))
    bx.axhline(TOL_M, linestyle="--", color="0.4")
    bx.annotate(f"tolerance {TOL_M} m", (0.98, TOL_M), xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(0, 4), ha="right", color="0.35")
    bx.set_yscale("log")
    bx.set_xticks(list(range(1, len(legs) + 1)),
                  [f"round {i}" for i in range(1, len(legs) + 1)])
    bx.set_ylabel("movement since the previous round (m)")
    bx.set_title("(b) what the convergence check judges")

    fig.suptitle("Brandywine ELM ↔ PFLOTRAN two-way loop — "
                 + " → ".join(p.name.replace("pflotran_run_", "PF ")
                                   for p in legs),
                 y=1.02)
    out = ROOT / "docs" / "paper" / "fig_coupling_convergence"
    fig.savefig(f"{out}.png", bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print("legs, oldest first:")
    for p in legs:
        print("  ", p.name)
    for cid in ids:
        print(f"  {cid}: " + " -> ".join(
            "?" if v is None else f"{v:.3f}" for v in series[cid]))
    print(f"wrote {out}.png and .pdf")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        rd = sys.argv[1]
    else:
        cands = sorted((ROOT / "workflow_outputs").glob("pflotran_run_*"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        rd = next((c for c in cands
                   if (c / "columns.json").exists() and "coupled_from" in
                   (c / "columns.json").read_text()), None)
        if rd is None:
            sys.exit("no coupled pflotran_run_* under workflow_outputs")
    main(rd)
