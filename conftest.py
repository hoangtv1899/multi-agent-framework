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
import os
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


# NO TEST REACHES THE MODEL GATEWAY UNLESS ASKED (2026-11-12). A step that is
# given no scripted client makes a real one, and with PNNL_API_KEY in the
# environment its call is real and paid. One test did exactly that for a day:
# its scripted plan failed its check, the rewrite then asked for had no
# scripted client, and every run of the suite spent a planning call and a
# review call, with the test passing. In a plain run the gateway's client is
# now a stand-in: it can be built, as the agents build theirs, and any request
# through it fails the test, even where the code under test catches the error.
# `--runllm` leaves the real client in place.
class GatewayReached(AssertionError):
    pass


@pytest.fixture(autouse=True)
def no_model_gateway(request, monkeypatch):
    if request.config.getoption("--runllm"):
        yield None
        return
    try:
        import agents.llm_agent as _la
    except Exception:                                   # no openai here: nothing to guard
        yield None
        return
    reached = []

    class NoGateway:
        """In the place of openai.OpenAI for one test."""

        def __init__(self, *a, **k):
            self.chat = self.completions = self

        def create(self, *a, **k):
            reached.append(str(k.get("model")))
            raise GatewayReached(
                "a test asked the real model gateway for a reply (model "
                f"{k.get('model')}): give the step a scripted client, or run with --runllm")
    monkeypatch.setattr(_la.openai, "OpenAI", NoGateway)
    yield reached                       # the guard's own test reads it, and empties it
    if reached:
        pytest.fail(f"the test asked the real model gateway {len(reached)} time(s) "
                    f"(model {', '.join(sorted(set(reached)))}): give the step a scripted "
                    f"client, or run with --runllm")


def pytest_configure(config):
    """Every test runs as Compy unless the test itself says otherwise.

    The tests are written for Compy's batch settings, the compy row of
    mcp/elm-mcp/src/machine_settings.py. On Perlmutter NERSC_HOST is set and
    env_nersc.sh names an account and a queue, which would reach every test,
    and the ELM server some tests load once per module, as Perlmutter's
    settings. Set here, before any import or fixture, the whole session runs
    as Compy on any machine. A test of another machine sets IDEAS_MACHINE or
    NERSC_HOST itself, as tests/test_machine_settings.py does."""
    for k in ("NERSC_HOST", "IDEAS_SLURM_ACCOUNT", "IDEAS_SLURM_QUEUE"):
        os.environ.pop(k, None)
    os.environ["IDEAS_MACHINE"] = "compy"


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
