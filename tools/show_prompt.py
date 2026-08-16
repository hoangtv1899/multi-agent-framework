#!/usr/bin/env python3
"""Print the exact prompt a step would send, without sending it.

    python3 tools/show_prompt.py --run-dir <RUN> [--step 2|3|both]

WHY THIS CAN EXIST AT ALL. Both briefs are built by deterministic code from
files already on disk — step 2's by `context_brief(ctx, step1)`, step 3's by
`review_brief(ctx, comparison, investigation)`. Neither needs a model, a
network call or a scheduler, so the text is free to look at and identical to
what the pipeline would send.

WHY IT IS WORTH LOOKING AT. The worst bug this pipeline has had was not in the
data and not in the rules: the evidence shown to step 3 was silently cut at 700
characters, so the model was handed a fifth of a result and told to quote from
it exactly. The audit then struck nine of fourteen true claims. Every number was
right, every rule was right, and the prompt was wrong — and nobody could see the
prompt.

A LIVE RUN NOW SAVES ITS OWN. Since 2026-08-14 each round writes
04_analysis/step<N>_round<R>_prompt.txt and ..._reply.txt. This tool is for the
runs that came before, for a run you have not executed yet, and for trying an
edited prompt by hand before spending anything on it.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build(run_dir: str, step: str):
    """(label, prompt_text) for each requested step."""
    from agents.analysis import (step0_context, step1_compare,
                                 step2_investigate, step3_interpret)

    ctx = step0_context.load(run_dir)
    analysis_dir = Path(run_dir) / "04_analysis"

    # READ, NEVER RECOMPUTE. Re-running step 1 here would call the comparison
    # package and redraw five figures into a finished run directory — a tool
    # for looking at a prompt must not modify the study it is looking at.
    comparison = step1_compare.load(analysis_dir)
    investigation = step2_investigate.load(analysis_dir)

    out = []
    if step in ("2", "both"):
        body = step2_investigate.context_brief(ctx, comparison or None)
        out.append(("step 2 — investigate",
                    body + "\n" + step2_investigate.TASK))
    if step in ("3", "both"):
        if not investigation:
            out.append(("step 3 — interpret",
                        "(no investigation.json in this run — step 3's prompt "
                        "quotes step 2's findings, so there is nothing to "
                        "build. Run step 2 first, or pick a finished run.)"))
        else:
            body = step3_interpret.review_brief(ctx, comparison or {},
                                                investigation)
            out.append(("step 3 — interpret",
                        body + "\n" + step3_interpret.TASK))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Print a step's prompt without calling a model.")
    ap.add_argument("--run-dir", required=True,
                    help="a finished run directory, e.g. "
                         "workflow_outputs/elm_run_20260813_233645")
    ap.add_argument("--step", default="both", choices=["2", "3", "both"])
    ap.add_argument("--out", default=None,
                    help="write to this file instead of stdout; with "
                         "--step both, writes <out>.step2.txt and .step3.txt")
    a = ap.parse_args()

    try:
        built = build(a.run_dir, a.step)
    except FileNotFoundError as e:
        sys.exit(f"cannot read that run: {e}")

    for label, text in built:
        n = a.step if a.step != "both" else label.split()[1]
        if a.out:
            p = Path(a.out if len(built) == 1 else f"{a.out}.step{n}.txt")
            p.write_text(text)
            print(f"{label}: {len(text):,} characters -> {p}")
        else:
            bar = "═" * 72
            print(f"\n{bar}\n{label}   ({len(text):,} characters)\n{bar}\n")
            print(text)


if __name__ == "__main__":
    main()
