"""Both source trees on the path, for every test.

ELM's modules moved to mcp/elm-mcp/src/ (docs/ELM_MCP_PLAN.md §9 phase 1b), so
`import elm_exp_manager` now resolves there rather than under `core.`. Tests
used to arrange this individually with `sys.path.insert(0, "src")`, which was
relative to the working directory and therefore only worked when pytest was run
from the repository root.

NOTE ON WHAT THIS SUITE IS FOR. It is kept running as an IMPORT CHECK while the
ELM code relocates, and for nothing else — see the plan's "How phases are
proven". Its 894 passing tests did not catch a single one of the failures of
2026-08-06. A replacement is written from scratch in phase 7; nothing here is
being maintained in the meantime.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for p in (ROOT / "src", ROOT / "mcp" / "elm-mcp" / "src",
          ROOT / "mcp" / "pflotran-mcp"):     # the PFLOTRAN manager, 2026-08-18
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
