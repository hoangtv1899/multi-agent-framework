#!/usr/bin/env python3
"""Refresh src/_vendor/ from the framework, or check that it has not drifted.

    python3 scripts/sync_vendor.py --check     # exit 1 if a copy has drifted
    python3 scripts/sync_vendor.py --write     # copy the framework's files over

src/_vendor/ holds verbatim copies of the few framework modules the SERVER
needs, so that a standalone checkout of this repository can run its tools with
no framework beside it.  The copies are a fallback: framework_path puts the
real framework ahead of them on sys.path, so inside a multi-agent-framework
checkout the framework's own files are what actually execute.

That ordering is what makes vendoring safe here.  The usual objection to a
vendored copy is that it forks and then drifts, and nobody notices.  These
copies never execute while the framework is present, and
tests/test_elm_vendor_sync.py in the framework runs --check, so a drift is a
test failure rather than a silent divergence.

Only self-contained modules belong here.  core.exp_manager_base deliberately
does NOT: it is the base class ELM and PFLOTRAN share, it reaches the Analyzer
and the sampler, and a second copy of it would fork a contract two models
depend on.  Code that needs it asks for the framework and fails without it.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE.parent / "src" / "_vendor"

# framework-relative source -> path inside src/_vendor/
FILES = {
    "src/core/keyset.py": "core/keyset.py",
    "src/core/model_agent_base.py": "core/model_agent_base.py",
    "src/core/basemap.py": "core/basemap.py",
    "src/core/static_wtd.py": "core/static_wtd.py",
    "src/agents/analysis/compare_common.py": "agents/analysis/compare_common.py",
    "tools/figstyle.py": "figstyle.py",
}


def _framework(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    sys.path.insert(0, str(HERE.parent / "src"))
    from framework_path import framework_dir
    return framework_dir()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true",
                   help="report drift and exit 1 if any copy differs")
    g.add_argument("--write", action="store_true",
                   help="overwrite the vendored copies from the framework")
    ap.add_argument("--framework", metavar="DIR",
                    help="the multi-agent-framework checkout "
                         "(default: whatever framework_path resolves)")
    args = ap.parse_args()

    fw = _framework(args.framework)
    if not (fw / "src" / "core").is_dir():
        print(f"not a framework checkout: {fw}", file=sys.stderr)
        return 2

    drifted, missing, written = [], [], []
    for rel, dest_rel in sorted(FILES.items()):
        src, dest = fw / rel, VENDOR / dest_rel
        if not src.is_file():
            missing.append(rel)
            continue
        if args.write:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            written.append(dest_rel)
        elif not dest.is_file():
            drifted.append(f"{dest_rel}: not vendored at all")
        elif not filecmp.cmp(src, dest, shallow=False):
            drifted.append(f"{dest_rel}: differs from {rel}")

    if missing:
        print("MISSING from the framework (did a module move?):", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 2

    if args.write:
        print(f"vendored {len(written)} files from {fw}")
        for w in written:
            print(f"  {w}")
        return 0

    if drifted:
        print("src/_vendor/ has DRIFTED from the framework:", file=sys.stderr)
        for d in drifted:
            print(f"  {d}", file=sys.stderr)
        print("\nRefresh with: python3 scripts/sync_vendor.py --write", file=sys.stderr)
        return 1

    print(f"src/_vendor/ matches {fw} ({len(FILES)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
