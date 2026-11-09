"""The Analyzer's access path on the Naches control's suite: every method and
every field through `suite`.

    pytest tests/test_domain_access_naches_oracle.py --runlocaldata -v

OPT-IN, and only where the files are on disk: the naches_suite block of
tests/local_oracle_paths.json names the built suite ("suite_dir").

THE ACCESS PATH REACHES EVERYTHING IT CLAIMS (package E, 2026-11-05). Each
field the catalog lists is read through get_series, each method of
domain_methods runs as a call of domain_access.SuiteAccess (the six timing
measures from timing.nc), and each call is logged with a hash. The gate here
is an open one: the plan's gate is held by tests/test_domain_access.py.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import domain_access as DAC
import domain_methods as M

pytestmark = pytest.mark.localdata

CONFIG = Path(__file__).with_name("local_oracle_paths.json")
REGION = "watershed, vegetated"
NORMAL = [2016, 2017, 2018, 2019, 2020]


@pytest.fixture(scope="module")
def suite_dir():
    if not CONFIG.is_file():
        pytest.skip(f"no {CONFIG.name}")
    c = json.loads(CONFIG.read_text()).get("naches_suite") or {}
    if not c.get("suite_dir") or not (Path(c["suite_dir"]) / "suite.json").is_file():
        pytest.skip("no naches_suite block with a built suite_dir")
    return c["suite_dir"]


def test_every_method_and_field_through_the_access_path(suite_dir):
    log = []
    acc = DAC.SuiteAccess({"control": suite_dir}, gate=lambda ask: None, log=log.append,
                          normal_years=NORMAL)
    fields = M.catalog_fields()
    for f in fields:
        r = acc.get_series("control", f, REGION, "year")
        assert len(r["labels"]) == 9 and np.isfinite(r["values"]).any(), f
    x = acc.get_map("control", "GPP", "2021")
    n = acc.get_map("control", "GPP", "normal")
    snow = acc.get_map("control", "H2OSNO", "WY2021")
    got = {"anomaly": acc.anomaly("GPP", "2021", region=REGION),
           "difference": acc.difference("control", "GPP", "2021-JJA", region=REGION),
           "screen": acc.screen(2021, REGION),
           "overlap": acc.overlap(x - n, snow, REGION),
           "elevation_split": acc.elevation_split(x - n, REGION)}
    for mid in M.TIMING_IDS:
        got[mid] = getattr(acc, mid)(2021, REGION)
    assert set(got) == set(M.METHODS) == set(M.ANALYZER_ACCESS)
    for mid, r in got.items():
        assert r["method"] == mid and r["version"] == M.METHODS[mid].version, mid
    assert len(log) == len(fields) + 3 + len(got)
    assert all("sha256" in r and "refused" not in r for r in log)
