# Reference pages

Two generated HTML pages. Open them in a browser; both are self-contained (no
CDN, no external assets) and follow the viewer's light/dark theme.

| Page | What it answers | How to refresh |
|---|---|---|
| [`arch_map.html`](arch_map.html) | Every stage of the pipeline, which box owns it, and which function implements it. File lists and line counts are read from disk on each build, and every step's function reference is verified to exist. | `python3 tools/make_arch_map.py`, then copy `workflow_outputs/ideas_arch_map_v2.html` here |
| [`analyzer_walkthrough.html`](analyzer_walkthrough.html) | How the Analyzer reaches a conclusion — what each of its five steps may and may not do, the three audit checks, the caveats step 1 derives, and two real runs end to end. | Hand-written. Edit in place. |

**The arch map is generated; never edit the HTML.** Edit `tools/make_arch_map.py`
and rebuild. It fails loudly when a step names a function that no longer exists
(`N stale`), which is the check that makes it worth consulting — a rotted
architecture diagram is worse than none, because it gets believed.

The walkthrough is prose and can drift. It states the date it was last revised
in its header; if that is far behind `git log`, treat it as history rather than
as the current design.
