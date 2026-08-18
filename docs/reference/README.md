# Reference pages

Three HTML pages. Open them in a browser; all are self-contained (no CDN, no
external assets) and follow the viewer's light/dark theme.

| Page | What it answers | How to refresh |
|---|---|---|
| [`arch_map.html`](arch_map.html) | Every stage of the pipeline, which box owns it, and which function implements it. File lists and line counts are read from disk on each build, and every step's reference is verified to exist. | `python3 tools/make_arch_map.py --out docs/reference/arch_map.html` |
| [`reception_walkthrough.html`](reception_walkthrough.html) | A reference for Reception: what it owns and does not, the LLM loop and its five tools, the code gather and its order, three requests traced (site ELM, conceptual ELM, site PFLOTRAN), what every key of `reception.json` is and who reads it, the schema and report contracts, every failure path, and which properties are enforced versus intended. Design history and open items are in appendices. | Hand-written. Edit in place. |
| [`analyzer_walkthrough.html`](analyzer_walkthrough.html) | A reference for the Analyzer: what it owns, the execution flow from the upstream artifacts to `analysis.json`, a uniform spec per step, artifact lineage, the submodule map, the key contracts, every failure path, and which properties are actually enforced versus merely intended. Design history and the key audit are in appendices. | Hand-written. Edit in place. |

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
