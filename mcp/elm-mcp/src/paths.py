#!/usr/bin/env python3
"""Where the bulk data lives — resolved, overridable, and reported.

Phase 1a proved this server can reach the CONUS restart and surfdata under the
five environment variables an MCP client actually forwards. It passes for one
reason: every data root is a HARDCODED ABSOLUTE PATH, so there is nothing for a
stripped environment to lose.

That is also the problem. The paths name one user's directories on one machine,
and one of them is not even ours (see WARNINGS below). So they have to become
overridable — without giving up the property that made the probe pass.

PRECEDENCE, and why there are four layers rather than two:

  1. an explicit argument         per call; the caller states it and it is
                                  recorded in the run's provenance
  2. IDEAS_CONUS_* in the env     works when the framework's MCPManager
                                  launches us, because mcp_client.py:78 does
                                  os.environ.copy()
  3. paths.json beside main.py    works under ANY client, because it involves
                                  no environment at all
  4. the hardcoded default        today's behaviour, unchanged

Layer 3 is the one that matters. A standard MCP client forwards only HOME,
LOGNAME, PATH, SHELL and USER — so layer 2 alone would work through our
framework and fail silently under Claude Code, which is exactly the asymmetry
that made every lambda tool in the reaction MCP report "No module named
'preprocessing'" under one launcher and succeed under another. A file the
server reads itself cannot have that failure.

WHAT THE RESOLVED PATH IS FOR, beyond opening: which CONUS restart a column
warm-started from is a FACT ABOUT THE SCIENCE, not a configuration detail. It
belongs in case_inputs.json and downstream in experiment.json, so a claim can
be traced to the data it rests on. Today it survives only as a NetCDF attribute
(make_finidat_subset.py sets finidat_subset_source), which nothing the Analyzer
reads ever sees.
"""
from __future__ import annotations

import json
import os
import pwd
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# paths.json sits beside main.py, not in src/ — it is configuration a user
# edits, not code.
PATHS_FILE = Path(__file__).resolve().parents[1] / "paths.json"

# name → (hardcoded default, environment variable, what it is)
DATA_PATHS: Dict[str, Tuple[str, str, str]] = {
    "conus_restart_manifest": (
        "/qfs/people/tran289/conus_restart_transfer/MANIFEST_conus_restarts.txt",
        "IDEAS_CONUS_RESTART",
        "CONUS restart manifest — one row per latitude band; the warm start "
        "subsets a column's FINIDAT out of the band that covers its latitude",
    ),
    "conus_surfdata": (
        "/qfs/people/tran289/IDEAS/1d_elm/conus_surfdata",
        "IDEAS_CONUS_SURFDATA",
        "CONUS surface data — supplies each column the soil of the grid cell "
        "it falls in, the only soil a warm-started run actually uses",
    ),
    "elm_input_files": (
        "/qfs/people/tran289/IDEAS/1d_elm/input_files",
        "IDEAS_ELM_INPUT_FILES",
        "ELM input-file templates",
    ),
}

# A manifest row: band, lat range, ..., absolute path. Mirrors
# make_warmstart.py's _MANIFEST_ROW — if that changes, this must too.
_MANIFEST_ROW = re.compile(r"^(lat\d+)\s+(\d+)-(\d+)N\s+\S+\s+\S+\s+(/\S+)")


def _from_file(name: str) -> Optional[str]:
    """paths.json, if a user wrote one. Malformed is reported, not ignored."""
    if not PATHS_FILE.is_file():
        return None
    try:
        return (json.loads(PATHS_FILE.read_text()) or {}).get(name)
    except Exception as e:                                      # noqa: BLE001
        # Deliberately not silent: a typo here would otherwise present as
        # "the default was used" and nobody would look at this file again.
        print(f"paths.json is unreadable ({e}) — using defaults", flush=True)
        return None


def resolve(name: str, override: Optional[str] = None) -> Tuple[Path, str]:
    """The path to use for `name`, and WHICH LAYER supplied it.

    The source is returned because "which restart did this run use" is
    unanswerable afterwards if only the value is kept.
    """
    if name not in DATA_PATHS:
        raise KeyError(f"unknown data path {name!r}; "
                       f"known: {', '.join(sorted(DATA_PATHS))}")
    default, env_var, _ = DATA_PATHS[name]

    if override:
        return Path(override), "argument"
    if os.environ.get(env_var):
        return Path(os.environ[env_var]), f"env:{env_var}"
    from_file = _from_file(name)
    if from_file:
        return Path(from_file), f"file:{PATHS_FILE.name}"
    return Path(default), "default"


def _owner(p: Path) -> Optional[str]:
    try:
        return pwd.getpwuid(p.stat().st_uid).pw_name
    except Exception:                                           # noqa: BLE001
        return None


def _warnings(p: Path, owner: Optional[str]) -> list:
    """Things that are true, readable today, and still worth saying."""
    out = []
    me = os.environ.get("USER") or os.environ.get("LOGNAME")
    parts = p.resolve().parts
    # /compyfs/<someone>/... is a personal scratch tree. /compyfs/inputdata is
    # the shared, curated one and is not the same kind of thing at all.
    personal_scratch = (len(parts) > 2 and parts[1] == "compyfs"
                        and parts[2] not in ("inputdata",))
    if owner and me and owner != me:
        out.append(f"owned by {owner}, not by you — you cannot protect it")
    if personal_scratch:
        out.append("on a personal scratch tree, which is what gets purged "
                   "when the filesystem fills")
    return out


def describe(name: str, override: Optional[str] = None) -> Dict[str, Any]:
    """Everything worth knowing about one data path, checked rather than assumed.

    Reports the SOURCE alongside the value, so a run that used an override is
    distinguishable from one that used the default — and reports owner and
    mtime, so a path that resolves but belongs to someone else is visible
    before a study rather than after one fails.
    """
    path, source = resolve(name, override)
    default, env_var, what = DATA_PATHS[name]
    exists = path.exists()

    info: Dict[str, Any] = {
        "path": str(path),
        "source": source,
        "is_default": source == "default",
        "present": exists,
        "what": what,
        "override_with": {"argument": name, "env": env_var,
                          "file": f"{PATHS_FILE.name} key {name!r}"},
    }
    if not exists:
        info["error"] = "does not exist"
        return info

    owner = _owner(path)
    st = path.stat()
    info["owner"] = owner
    info["modified"] = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d")
    info["readable"] = os.access(path, os.R_OK)
    warn = _warnings(path, owner)
    if warn:
        info["warnings"] = warn

    # The manifest existing is not the same as the restarts existing. A study
    # dies eight minutes into a CIME build on that distinction.
    if name == "conus_restart_manifest" and path.is_file():
        try:
            rows = [m.groups() for m in
                    (_MANIFEST_ROW.match(l.strip())
                     for l in path.read_text().splitlines()) if m]
            alive = [r for r in rows if Path(r[3]).is_file()]
            info["bands_total"] = len(rows)
            info["bands_resolving"] = len(alive)
            if not alive:
                info["present"] = False
                info["error"] = ("the manifest lists no restart file that "
                                 "exists — every warm start would fail")
            elif len(alive) < len(rows):
                info.setdefault("warnings", []).append(
                    f"only {len(alive)} of {len(rows)} bands resolve; a column "
                    f"in a missing band cannot be warm-started")
            if alive:
                d = Path(alive[0][3])
                o = _owner(d)
                info["restart_owner"] = o
                for w in _warnings(d, o):
                    info.setdefault("warnings", []).append(f"restarts: {w}")
        except Exception as e:                                  # noqa: BLE001
            info["error"] = f"unreadable manifest: {e}"
            info["present"] = False
    return info


def describe_all(overrides: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    overrides = overrides or {}
    return {n: describe(n, overrides.get(n)) for n in DATA_PATHS}


def provenance(overrides: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """The compact record that belongs in case_inputs.json.

    Value AND source, because a claim traced to "the default" is only
    reproducible for as long as the default is what it is today.
    """
    overrides = overrides or {}
    out = {}
    for n in DATA_PATHS:
        p, s = resolve(n, overrides.get(n))
        out[n] = str(p)
        out[f"{n}_source"] = s
    return out


if __name__ == "__main__":                                      # pragma: no cover
    print(json.dumps(describe_all(), indent=2))
