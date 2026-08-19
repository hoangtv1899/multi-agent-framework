# Reference pages

Five HTML pages. Open them in a browser; all are self-contained (no CDN, no
external assets) and follow the viewer's light/dark theme.

| Page | What it answers | How to refresh |
|---|---|---|
| [`arch_map.html`](arch_map.html) | Every stage of the pipeline, which box owns it, and which function implements it. File lists and line counts are read from disk on each build, and every step's reference is verified to exist. | `python3 tools/make_arch_map.py --out docs/reference/arch_map.html` |
| [`planner_walkthrough.html`](planner_walkthrough.html) | The planner on one page: the flow diagram — reception's summary from the left, two blocks read once from the chosen model's server from the right, one LLM call, `strategy.json` with the pinning block attached, and the Experiment Manager's gate below it — and a few sentences on each box. | Hand-written. Edit in place. |
| [`exp_manager_walkthrough.html`](exp_manager_walkthrough.html) | The Experiment Manager on one page: the flow diagram — two files in, the check gate, materialize with the sampler or the server-built sweep and the one build call per model, then case inputs, the ELM-only case build with its two SLURM jobs, run, extract, package, the Analyzer hand-off, and the three servers it calls on the right — a few sentences on each box, and two tables: which tool per stage per model, and what lands in the run directory when. | Hand-written. Edit in place. |
| [`reception_walkthrough.html`](reception_walkthrough.html) | Reception on one page: the flow diagram — the LLM loop and its tools above the brief, two gates, the code gather below it, the outputs, and every MCP server it reaches on the right — and a few sentences on each box in it. | Hand-written. Edit in place. |
| [`analyzer_walkthrough.html`](analyzer_walkthrough.html) | The Analyzer on one page: the flow diagram — three files in, step 0's context and the preflight gate, step 1's comparison through the model's package beside its server, the two language-model calls (figures as code; interpretation) with the code audit and the one loop between them, the report — a few sentences on each box, and a table of what each step reads, writes and spends. Rewritten short on 2026-08-18; the long reference edition (every contract, failure path, design history) is in git history before that. | Hand-written. Edit in place. |

**The arch map is generated; never edit the HTML.** Edit `tools/make_arch_map.py`
and rebuild. It fails loudly when a step names a function that no longer exists
(`N stale`), which is the check that makes it worth consulting — a rotted
architecture diagram is worse than none, because it gets believed. A reference
may name a `def`, a `class`, or a module-level assignment: the key lists are the
clearest thing to point a reader at for "what does this stage decide to carry",
and every one of them is an assignment.

The walkthrough is prose and can drift. It states the date it was last revised
in its header; if that is far behind `git log`, treat it as history rather than
as the current design.
