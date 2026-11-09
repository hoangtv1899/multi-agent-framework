"""The recorded studies of G part 2's run again (2026-11-08), fed back
through the fixes that followed it, with no model call. Real data: run with
--runlocaldata; the studies are read from tests/local_oracle_paths.json,
key "g3_replay" (each copied to tmp_path, never written in place).

    pytest tests/test_analyzer_g3_replay_naches_oracle.py --runlocaldata -v

Question 1 failed with no analysis: step 2b's reply was cut off at the output
limit, and was not saved. Its recorded plan, through the framework's own calls
of the required outputs, now gives every output with no step 2b reply at all.
Question 2 was withheld on two defects of the trace: its fourteen recorded
claims now all trace, and its report, its headline labelled exploratory by
the report's own code, is published. Question 3 was withheld on three
defects of the trace and on failures that are the checks working: only the
latter remain (a region its records do not name, a hypothesis headline from
an observation, a run not on the server's list).
"""
import json
import shutil
from pathlib import Path

import pytest

from g2_replay_support import forbid_model_clients

from agents.analysis import analysis_plan as AP          # noqa: E402
from agents.analysis import claim_trace as CT            # noqa: E402
from agents.analysis import plan_contract as PC          # noqa: E402
from agents.analysis import step0_context as S0          # noqa: E402
from agents.analysis import step2_investigate as S2      # noqa: E402
from agents.analysis import step4_report as S4           # noqa: E402

pytestmark = pytest.mark.localdata
CONFIG = Path(__file__).resolve().parent / "local_oracle_paths.json"


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    forbid_model_clients(monkeypatch)


@pytest.fixture(scope="module")
def replay():
    if not CONFIG.is_file():
        pytest.skip(f"no {CONFIG.name}")
    d = (json.loads(CONFIG.read_text()).get("g3_replay") or {}).get("dir")
    if not d or not (Path(d) / "q2" / "04_analysis" / "lookups.jsonl").is_file():
        pytest.skip("no g3_replay block naming the recorded studies")
    return Path(d)


def _copy(replay, q, tmp_path):
    rd = tmp_path / q
    shutil.copytree(replay / q, rd)
    return rd


def _j(p):
    return json.loads(Path(p).read_text())


def _trace(rd):
    out = rd / "04_analysis"
    ctx = S0.load(str(rd))
    interp = _j(out / "interpretation.json")
    claims = [dict(c) for c in interp["claims"]]
    head = interp["headline"]
    if AP.load_plan(out).get("question_type") == "exploratory":
        head = S4.exploratory_headline(head)
    return claims, CT.trace_evidence(claims, head, out, study_dir=rd,
                                     regions=S4.suite_regions(ctx),
                                     open_causes=interp.get("open_causes") or [])


def test_q1_the_framework_makes_every_required_output_of_its_recorded_plan(replay, tmp_path):
    rd = _copy(replay, "q1", tmp_path)
    out = rd / "04_analysis"
    calls = _j(rd / "calls.json")
    # the run: step 2b cut off at the limit, no lookup made, the reply not saved
    asks = {a["label"]: a.get("finish_reason") for a in calls["asks"]}
    assert asks["step2_investigate"] == "length" and not (out / AP.LOOKUPS_FILE).exists()
    assert not (out / "step2_round1_reply.txt").exists()
    ctx = S0.load(str(rd))
    assert AP.plan_passed(out)[0]
    plan = AP.load_plan(out)
    n_out = sum(len(a.get("outputs") or []) for a in plan["analyses"])
    got = AP.produce_required(ctx, out)
    rec = got["record"]
    assert n_out == 28 and len(rec["calls"]) == 28 and all(r["ok"] for r in rec["calls"])
    assert rec["lines"] == [1, 28] and all(
        x["script"] == AP.REQUIRED_SCRIPT for x in AP.lookups(out))
    ph = AP.close_planned_phase(out, when="before the screen", by="the replay",
                                figures=[AP.required_attempt(rec, plan)])
    assert ph["counts"] == {"completed": 5, "completed after an earlier failure": 0,
                            "failed": 0, "skipped": 0}
    acc = AP.acceptance(ph)
    assert acc["ok"] is True and acc["blockers"] == []
    # the calls the framework made are the ones step 2b's brief lists
    brief = " ".join(S2.suite_brief(ctx).split())
    for r in rec["calls"]:
        assert f"{r['label']}: {r['call']}" in brief, r["label"]
    (tmp_path / "q1_required.json").write_text(json.dumps(
        {"counts": ph["counts"], "lines": rec["lines"]}, indent=1))


def test_q2_every_recorded_claim_traces_and_its_labelled_report_publishes(replay, tmp_path):
    rd = _copy(replay, "q2", tmp_path)
    out = rd / "04_analysis"
    recorded = _j(out / "analysis.json")
    assert recorded["evidence_trace"]["claims_failed"] == [13, 14]          # as it ran
    claims, t = _trace(rd)
    assert len(claims) == 14 and t["claims_failed"] == [] and t["headline"]["problems"] == []
    assert not t.get("withheld_because")
    # the two claims the trace's defects failed: the year of an overlap's maps
    # (B2), an elevation read as a year (B1)
    assert "2022" in t["records"]["L41"]["names"]["years"]
    assert [x["record"] for x in t["records"]["L41"]["inputs"]] and all(
        x["record"] for x in t["records"]["L41"]["inputs"])
    # the report, built again from the recorded interpretation: published,
    # its headline labelled by the report's own code, the model's words kept
    ctx = S0.load(str(rd))
    rep = S4.build(ctx, {}, _j(out / "investigation.json"), _j(out / "interpretation.json"),
                   rd)
    assert rep["headline"].startswith(S4.EXPLORATORY_PREFIX)
    assert rep["headline_as_written"] == _j(out / "interpretation.json")["headline"]
    assert rep["evidence_trace"]["claims_failed"] == [] and not rep.get("withheld_because")
    assert (rep.get("evidence_numbers") or {}).get("pass") is True


def test_q3_only_the_failures_that_are_the_checks_working_remain(replay, tmp_path):
    rd = _copy(replay, "q3", tmp_path)
    out = rd / "04_analysis"
    recorded = _j(out / "analysis.json")
    assert recorded["evidence_trace"]["claims_failed"] == [1, 2, 3, 4, 5, 6, 7]   # as it ran
    claims, t = _trace(rd)
    # claims 1 to 5 and 7, failed by an elevation read as a year (B1), a
    # derived record's period (B2) and a share's percent (B3), now trace
    assert t["claims_failed"] == [6]
    # claim 6 names a region its records do not name: strict matching kept
    assert t["problems"][6] == ["it names the region 'vegetated cells', which none of its "
                                "records names"]
    # the headline still rests on an observation, and the open cause is still
    # no run of the server's list: step 3 now sees the expectations and the
    # list (C1, C2), and only a new run can show whether it uses them
    assert t["headline"]["problems"] == [
        "claim 3 is observation, not tested: a hypothesis question's headline comes from "
        "tested claims only"]
    assert len(t["open_cause_problems"]) == 1 and \
        "'snow_water_initial_increase' is not a run the server can make" in \
        t["open_cause_problems"][0]
    # the band shares are traced (claim 5), through the split's input and its shares
    l25 = t["records"]["L25"]
    assert l25["field"] == "GPP" and "JJA" in {p[1] for p in l25["names"]["periods"]}
