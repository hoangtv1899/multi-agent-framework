"""Where to tell the user their study finished — remembered between runs.

~/.ideas/notify.json, outside the repo on purpose: an address is personal, it
should survive a git operation, and it must not arrive in a collaborator's
checkout because someone committed their own.

Deliberately tiny. This is not a settings framework; it is one address and a
place for the next such preference to go without inventing a new mechanism.
"""
import json
import os
from pathlib import Path
from typing import Optional

PREFS_DIR = Path(os.environ.get("IDEAS_PREFS_DIR",
                                str(Path.home() / ".ideas")))
PREFS_FILE = PREFS_DIR / "notify.json"


def _read() -> dict:
    """Never raises. A corrupt or unreadable prefs file must degrade to "no
    preference" — losing a study over a malformed dotfile would be absurd."""
    try:
        return json.loads(PREFS_FILE.read_text()) or {}
    except Exception:                                           # noqa: BLE001
        return {}


def remembered_email() -> Optional[str]:
    v = str(_read().get("email") or "").strip()
    return v or None


def remember_email(addr: Optional[str]) -> Optional[str]:
    """Store (or clear, with None/empty) the notification address.

    Returns what was stored, or None. Failure to write is reported by the
    caller rather than raised: the run is what matters, not the bookkeeping.
    """
    prefs = _read()
    if addr:
        prefs["email"] = addr.strip()
    else:
        prefs.pop("email", None)
    PREFS_DIR.mkdir(parents=True, exist_ok=True)
    PREFS_FILE.write_text(json.dumps(prefs, indent=2) + "\n")
    try:
        PREFS_FILE.chmod(0o600)
    except Exception:                                           # noqa: BLE001
        pass
    return prefs.get("email")
