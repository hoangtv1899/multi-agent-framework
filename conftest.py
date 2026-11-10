# conftest.py  ← project ROOT (same level as src/, tests/)
"""Pytest bootstrap + test tiers.

Paths: put the repo root AND src/ on sys.path so both `import src.x` and the
in-tree `from agents.. / from core..` style resolve.

Tiers (opt-in): heavy or external tests are SKIPPED by default and run only when
their flag is given, so plain `pytest` stays fast, offline and deterministic:
    pytest                 # offline unit tests only  (default)
    pytest --runlive       # + live MCP data-source tests        (network)
    pytest --runllm        # + real reception/planner round-trip (needs PNNL_API_KEY)
    pytest --runcompute    # + ELM build/run tests               (needs an salloc node)
    pytest --runlocaldata  # + checks against a finished run's output on disk
                           #   (paths in tests/local_oracle_paths.json, not tracked)
    pytest --review-gate   # + fails while any field note, method contract or
                           #   mechanism is still a draft
    pytest --runparked     # + the tests set aside in tests/parked_tests.json
                           #   (paths the whole-domain workflow does not use)
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
for _p in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(_p))

# marker -> (cli flag, one-line reason shown in --help)
_TIERS = {
    "live":    ("--runlive",    "live network APIs (MCP data sources)"),
    "llm":     ("--runllm",     "the live LLM endpoint (needs PNNL_API_KEY)"),
    "compute": ("--runcompute", "ELM build/run on SLURM (needs an salloc node)"),
    "localdata": ("--runlocaldata",
                  "a finished run's output and observations on local disk"),
    # THE REVIEW GATE (2026-11-04): fails while any field note is still a
    # draft (tests/test_domain_catalog.py), and since 2026-11-05 any method
    # contract or mechanism (tests/test_methods_doc.py), so it runs only when asked
    "review_gate": ("--review-gate",
                    "the user's review of the field notes, method contracts and mechanisms "
                    "(fails while one is a draft)"),
}


# TESTS SET ASIDE (2026-11-10). tests/parked_tests.json lists, with a class and
# a reason each, the tests of paths the whole-domain workflow does not use (the
# single-column route, PFLOTRAN) and of things that no longer exist. They are
# skipped by a plain run, so that a plain run with a failure means a defect on
# the path in use, and they still run with --runparked. A test of code the
# whole-domain workflow uses is repaired, never listed there
# (tests/test_parked_tests.py holds the list to its rules).
PARKED_LIST = ROOT / "tests" / "parked_tests.json"


def parked_tests():
    """{node id or file: (class, reason)} from tests/parked_tests.json."""
    d = json.loads(PARKED_LIST.read_text()) if PARKED_LIST.is_file() else {}
    out = {f: (v["class"], v["reason"]) for f, v in (d.get("files") or {}).items()}
    for g in d.get("groups") or []:
        for t in g["tests"]:
            out[t] = (g["class"], g["reason"])
    return out


def pytest_addoption(parser):
    for flag, reason in _TIERS.values():
        parser.addoption(flag, action="store_true", default=False,
                          help=f"run tests that hit {reason}")
    parser.addoption("--runparked", action="store_true", default=False,
                     help="run the tests set aside in tests/parked_tests.json")


def pytest_ignore_collect(collection_path, config):
    # a file set aside whole cannot even be imported: leave it uncollected
    if config.getoption("--runparked"):
        return None
    try:
        rel = Path(str(collection_path)).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None
    return True if "::" not in rel and rel in parked_tests() else None


def pytest_collection_modifyitems(config, items):
    # skip each opt-in tier unless its flag was passed
    for marker, (flag, _reason) in _TIERS.items():
        if config.getoption(flag):
            continue
        skip = pytest.mark.skip(reason=f"needs {flag}")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
    if not config.getoption("--runparked"):
        parked = parked_tests()
        for item in items:
            hit = parked.get(item.nodeid)
            if hit:
                item.add_marker(pytest.mark.skip(
                    reason=f"set aside ({hit[0]}): {hit[1]}; --runparked runs it"))
