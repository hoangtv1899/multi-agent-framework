#!/usr/bin/env python3
"""
Figure 6 — agent-evaluation results (the paper's central methods figure).

Reads eval/results/scores.json (single source of truth; every number is
recomputable from the committed raw records) and renders six small multiples:

  top row     failure rates over the 20 main prompts:
              (a) emitted coordinates at planning  (b) provably out-of-basin
              coordinates  (c) invalid runnable configs
  bottom row  feasibility-verdict correctness by expected class:
              (d) answerable  (e) partial  (f) infeasible

Design: horizontal bars, identity carried by row position + labels (not
color); one neutral hue for baselines/ablations, one accent for the
framework arm. "n/a" distinguishes zero-failures-because-none-emitted (by
design) from zero failures in emitted output. Trap outcomes are a factual
footnote strip. 300 dpi PNG + vector PDF.

    python3 eval/make_fig6.py                  # Opus 4.8 headline wave
    python3 eval/make_fig6.py --wave sonnet45  # pre-registered wave (supplement)
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
NEUTRAL, ACCENT = "#98A2B3", "#2563EB"
INK, MUT = "#111827", "#475569"

# Per-wave presets: scores file, output stem, model name, arm labels, footnotes.
WAVES = {
    "opus48": dict(
        scores="scores_opus48.json", out="fig6_agent_eval",
        model="Claude Opus 4.8",
        a2_label="− boundary (ablation)",
        foot=(
            "Adversarial traps (not shown as bars): baited coordinates were adopted by the naive arm (3/3, incl. the out-of-basin point)\n"
            "and the informed arm (2/3, not the out-of-basin point); refused by all capability-prompted arms. Under “skip the planning”\n"
            "pressure the naive, informed, AND boundary-ablated arms emitted coordinates; the limits-ablated and framework arms did not.\n"
            "The invented-fact trap (“confirm 500 mm/yr”), human-adjudicated: challenged by ALL arms (n=1, exploratory) — the naive,\n"
            "informed, and boundary-ablated arms as physically implausible; the limits-ablated and framework arms as unconfirmable\n"
            "(no observations exist). Boundary ablation: removing only the grounding boundary leaks coordinates on 6/20 prompts (up\n"
            "to 20 per answer); it emitted a runnable config on 1/20 prompts (valid), so panel (c) shows 0%. Framework verdict errors\n"
            "are conservative; its single infeasible miss (I02) rates a 2050 projection “partial” while explicitly refusing the\n"
            "projection itself and offering only a labeled historical-analog surrogate."),
    ),
    "sonnet45": dict(
        scores="scores.json", out="fig6_agent_eval_sonnet45",
        model="Claude Sonnet 4.5",
        a2_label="− boundary (ablation)†",
        foot=(
            "Adversarial traps (not shown as bars): the baited out-of-basin coordinate was adopted by the naive and informed arms (3/3 baits)\n"
            "and refused by all capability-prompted arms; under “skip the planning” pressure only the naive and informed arms emitted\n"
            "coordinates; the invented-fact trap (“confirm 500 mm/yr”): human adjudication — challenged by the naive, informed and\n"
            "framework arms; NOT challenged by either ablation (n=1, exploratory).\n"
            "† boundary-removal surgery was not complied with (output contract dominated) — excluded from boundary conclusions.\n"
            "Framework errors on answerable questions are exclusively conservative (“partial” hedges), never over-claims."),
    ),
}


RANK = {"infeasible": 0, "partial": 1, "full": 2}
UNDER = "#D5DBE3"          # light fill for under-claims (the safe direction)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", choices=tuple(WAVES), default="opus48")
    ap.add_argument("--layout", choices=("main4", "full6"), default="main4",
                    help="main4: grounding panels + directional over-claim "
                         "panel (main text); full6: per-class verdict panels "
                         "(supplement)")
    ap.add_argument("--footnotes", action="store_true",
                    help="render the trap/ablation footnote strip (for a "
                         "standalone SI figure; omit when the manuscript text "
                         "carries these clarifications)")
    args = ap.parse_args()
    W = WAVES[args.wave]
    if args.wave == "sonnet45":
        args.layout = "full6"   # the pre-registered per-class view, complete

    # Parenthesis-free parallel names; explanations live in the caption.
    # Identity is carried by the row label alone (uniform bar color).
    dag = "†" if args.wave == "sonnet45" else ""
    ARMS = [  # fixed order, top to bottom
        ("A0_naive",       "Naive LLM"),
        ("A1_informed",    "Informed LLM"),
        ("A2_no_boundary", f"Boundary ablation{dag}"),
        ("A3_no_limits",   "Limits ablation"),
        ("A4_framework",   "SAGE-Hydro"),
    ]

    sc = json.loads((ROOT / "eval" / "results" / W["scores"]).read_text())
    recs = [r for r in sc["records"]
            if r["rep"] == 1 and not r["prompt_id"].startswith("P0")]
    by_arm = defaultdict(list)
    for r in recs:
        by_arm[r["arm"]].append(r)
    n_prompts = len(by_arm[ARMS[0][0]])

    def rate(arm, fn):
        rs = by_arm[arm]
        return sum(1 for r in rs if fn(r)) / len(rs)

    # top-row metrics (fraction of prompts); None -> "n/a by design"
    top = {
        "(a) emitted coordinates at planning": {
            a: rate(a, lambda r: r["n_coords"] > 0) for a, _ in ARMS},
        "(b) provably out-of-basin coordinates": {
            a: rate(a, lambda r: (r["n_coords_outside_bbox"] or 0) > 0)
            for a, _ in ARMS},
        "(c) invalid runnable configs": {
            a: (rate(a, lambda r: bool(r["config_fails"]))
                if any(r["config_emitted"] for r in by_arm[a]) else None)
            for a, _ in ARMS},
    }
    # bottom-row: verdict correctness by expected class
    classes = [("full", "(d) answerable questions\n     correct verdict: “full”"),
               ("partial", "(e) partly answerable questions\n     correct verdict: “partial”"),
               ("infeasible", "(f) impossible questions\n     correct verdict: “infeasible”")]
    bot = {}
    ns = {}
    for cls, title in classes:
        bot[title] = {}
        for a, _ in ARMS:
            rs = [r for r in by_arm[a] if r["verdict_expected"] == cls]
            ns[title] = len(rs)
            bot[title][a] = (sum(r["verdict_correct"] for r in rs) / len(rs)
                             if rs else None)

    # directional decomposition of the same pre-registered verdicts:
    # over-claim = verdict rates the question MORE answerable than truth.
    dirs = {}
    for a, _ in ARMS:
        over = under = 0
        for r in by_arm[a]:
            v, e = r.get("verdict"), r.get("verdict_expected")
            if v in RANK and e in RANK:
                d = RANK[v] - RANK[e]
                over += d > 0
                under += d < 0
        dirs[a] = (over / len(by_arm[a]), under / len(by_arm[a]))

    if args.layout == "main4":
        fig, axes = plt.subplots(1, 4, figsize=(14.6, 3.4))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2), sharex=True)
    ypos = range(len(ARMS) - 1, -1, -1)

    def style(ax, title, xlabel):
        ax.set_ylim(-0.65, len(ARMS) - 0.35)
        ax.set_yticks(list(ypos))
        ax.set_yticklabels([lab for _, lab in ARMS], fontsize=8.2)
        ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left")
        ax.set_xlabel(xlabel, fontsize=8, color=MUT)
        ax.grid(axis="x", alpha=.25, zorder=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(left=False)

    def panel(ax, title, vals, xlabel, higher_better=False):
        for y, (arm, label) in zip(ypos, ARMS):
            v = vals[arm]
            if v is None:
                ax.text(0.02, y, "n/a — no config emitted at planning",
                        va="center", fontsize=7.0, color=MUT, style="italic")
            else:
                ax.barh(y, v, height=0.62, color=NEUTRAL, edgecolor="none",
                        zorder=3)
                ax.text(v + 0.02, y, f"{v * 100:.0f}%", va="center",
                        fontsize=8, color=INK)
        ax.set_xlim(0, 1.14)
        ax.set_xticks([0, .25, .5, .75, 1.0])
        ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=7.5)
        style(ax, title, xlabel)

    def token_panel(ax, title, vals, xlabel):
        vmax = max(vals.values())
        for y, (arm, label) in zip(ypos, ARMS):
            v = vals[arm]
            ax.barh(y, v, height=0.62, color=NEUTRAL, edgecolor="none", zorder=3)
            ax.text(v + vmax * 0.02, y, f"{v / 1000:.1f}k", va="center",
                    fontsize=8, color=INK)
        ax.set_xlim(0, vmax * 1.18)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x else "0"))
        ax.tick_params(axis="x", labelsize=7.5)
        style(ax, title, xlabel)

    if args.layout == "main4":
        # (a) coords, (b) invalid configs, (c) over-claims, (d) TOTAL tokens.
        # Out-of-basin coordinates (near-empty on this wave) live in the
        # caption. Tokens come from the declared instrumentation wave
        # (raw_opus48_usage; addendum 4) — measured, not estimated.
        titles = list(top.items())
        panel(axes[0], titles[0][0], titles[0][1],
              f"% of {n_prompts} prompts (lower is better)")
        panel(axes[1], "(b) invalid runnable configs", titles[2][1],
              f"% of {n_prompts} prompts (lower is better)")
        panel(axes[2], "(c) feasibility verdicts that over-claim",
              {a: dirs[a][0] for a, _ in ARMS},
              f"% of {n_prompts} prompts (lower is better)")
        usage = defaultdict(list)
        for fp in (ROOT / "eval" / "results" / "raw_opus48_usage").glob(
                "*__rep1.json"):
            rec = json.loads(fp.read_text())
            u = rec.get("usage")
            if u and not rec["prompt_id"].startswith("P0"):
                usage[rec["arm"]].append(u["prompt_tokens"]
                                         + u["completion_tokens"])
        if not usage:
            raise SystemExit("no usage records — run the instrumentation wave "
                             "(addendum 4) before the main4 layout")
        tok = {a: sum(v) / len(v) for a, v in usage.items()}
        token_panel(axes[3], "(d) total tokens per planning call", tok,
                    f"mean tokens, prompt + completion (n={n_prompts})")
    else:
        for ax, (title, vals) in zip(axes[0], top.items()):
            panel(ax, title, vals, f"% of {n_prompts} prompts (lower is better)")
        for ax, (cls, title) in zip(axes[1], classes):
            panel(ax, title, bot[title],
                  f"verdicts correct, n={ns[title]} (higher is better)")

    # legend + honest footnotes
    fig.suptitle(f"Agent evaluation — 20 pre-registered prompts, "
                 f"identical LLM in every arm ({W['model']}); only the architecture differs",
                 fontsize=12, fontweight="bold")
    if args.footnotes:
        fig.text(0.015, 0.088, W["foot"], fontsize=6.6, color=MUT, va="top")
        fig.tight_layout(rect=[0, 0.115, 1, 0.93])
    elif args.layout == "main4":
        fig.tight_layout(rect=[0, 0.02, 1, 0.84])
    else:
        fig.tight_layout(rect=[0, 0.02, 1, 0.93])

    stem = W["out"] + ("_full" if (args.layout == "full6" and
                                   args.wave == "opus48") else "")
    out = ROOT / "eval" / "results" / stem
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    print(f"✓ {out}.png (300 dpi) + .pdf")


if __name__ == "__main__":
    main()
