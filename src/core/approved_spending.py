#!/usr/bin/env python3
"""
The computing a yes allows, recorded with the yes and checked before any job
src/core/approved_spending.py

A person approves a design from a page that gives the model server's estimate
of its cost, in node-hours. That estimate is no part of what identifies the
design: the same runs cost more or less as the machine's rates change, and a
design must not stop being the design approved because a rate moved. So the
figure is kept apart from the design's fingerprint, as the limit of what the
yes allows (the user's decision of 2026-11-12), and it is checked on its own.

  recorded   with the approval (agents/design_review.approve), under
             "authorized": the node-hours the page gave, and where the figure
             came from.
  checked    by the Experiment Manager, after the model server has checked the
             design again and before anything is built or submitted. A design
             the server now estimates above the figure is refused, with both
             figures, and nothing is submitted. A person then approves it
             again, on a page that gives the new estimate.

A study with no approval on record, an approval made before this rule, or a
design the server gives no estimate for is not held to a limit: there is no
figure a person agreed to.

This module names no model. The estimate is whatever the model's own server
returns under check.cost.node_hours.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional

APPROVAL_FILE = "01_inputs/design_approval.json"
SOURCE = "the model server's estimate on the design page"


def _number(x) -> Optional[float]:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x) if x == x and x >= 0 else None


def estimate(preview: Optional[Dict[str, Any]]) -> Optional[float]:
    """The node-hours the model server's check estimates, from the record of
    its check (a design preview, or the check itself); None when it gives
    none."""
    chk = (preview or {}).get("check") if isinstance((preview or {}).get("check"), dict) \
        else (preview or {})
    return _number(((chk or {}).get("cost") or {}).get("node_hours"))


def authorized(preview: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """What a yes given on this design's page allows, for the approval's
    record; None when the page gave no estimate."""
    nh = estimate(preview)
    return None if nh is None else {"node_hours": nh, "from": SOURCE}


def problem(run_dir, node_hours) -> Optional[str]:
    """Why a design may not be built and submitted for its cost, or None: the
    server's estimate now, against the figure the yes on record allows."""
    try:
        rec = json.loads((Path(run_dir) / APPROVAL_FILE).read_text())
    except (OSError, ValueError):
        return None
    allowed = _number(((rec if isinstance(rec, dict) else {}).get("authorized") or {})
                      .get("node_hours"))
    now = _number(node_hours)
    if allowed is None or now is None or now <= allowed + 1e-9:
        return None
    return (f"the model server now estimates about {now:g} node-hours for this design, more "
            f"than the {allowed:g} node-hours the yes on record allows ({APPROVAL_FILE}); "
            f"nothing was submitted. Ask again to approve the design at its present cost")
