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
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for p in (ROOT / "src", ROOT / "mcp" / "elm-mcp" / "src",
          ROOT / "mcp" / "pflotran-mcp"):     # the PFLOTRAN manager, 2026-08-18
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# THE STREAM INDEX CACHE IS OFF IN EVERY TEST (2026-11-04). gridded keeps each
# history file's index on disk, by default under ~/.cache/elm-mcp/stream_index/
# (gridded, THE INDEX ON DISK). The tests write thousands of small history
# files and rewrite some in place; none of that may leave an entry in the
# user's home, and a test must read the files it wrote, not an index of them.
# Set here, before any fixture opens a stream; a test of the cache points the
# variable at a temporary directory of its own.
os.environ["ELM_STREAM_INDEX_CACHE"] = "off"

# THE ROUTES SET ASIDE ARE REACHABLE IN THE TEST SESSION (2026-11-10). This
# version runs whole-domain studies only (src/core/supported_routes.py); the
# single-column and coupling routes keep their code and their tests, and most
# of those tests predate the rule. The switch a developer sets by hand is set
# here for them. A test of the rule itself removes it (monkeypatch.delenv), so
# it sees the front door as a user gets it.
os.environ["IDEAS_DEVELOPER_ROUTES"] = "1"
