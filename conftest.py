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
"""
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
}


def pytest_addoption(parser):
    for flag, reason in _TIERS.values():
        parser.addoption(flag, action="store_true", default=False,
                          help=f"run tests that hit {reason}")


def pytest_collection_modifyitems(config, items):
    # skip each opt-in tier unless its flag was passed
    for marker, (flag, _reason) in _TIERS.items():
        if config.getoption(flag):
            continue
        skip = pytest.mark.skip(reason=f"needs {flag}")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
