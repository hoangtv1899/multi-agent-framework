"""The study report job B sends: its verdict and its exit status.

    pytest tests/test_notify_study.py -v
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _notify():
    spec = importlib.util.spec_from_file_location("notify_study", ROOT / "tools" / "notify_study.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(tmp_path, analyze, experiment=True, docs=0):
    rd = tmp_path / "run"
    (rd / "04_analysis").mkdir(parents=True)
    stages = {s: {"status": "done"} for s in ("materialize", "build_case_inputs",
                                              "build_cases", "run", "extract", "package")}
    stages["analyze"] = analyze
    (rd / "run_state.json").write_text(json.dumps({"model": "elm", "stages": stages}))
    if experiment:
        (rd / "experiment.json").write_text(experiment if isinstance(experiment, str) else "{}")
    for k in range(docs):
        (rd / "04_analysis" / f"a{k}.json").write_text("{}")
    return rd


def _domain(planned, passed, first):
    return json.dumps({"kind": "whole_domain", "chain": {
        "years_planned": planned, "years_passed": passed, "first_unpassed": first}})


SKIPPED = {"status": "skipped", "reason": "whole-domain rows come next"}


class TestTheVerdict:

    def test_an_analysis_skipped_for_a_reason_is_complete(self, tmp_path):
        body, rc = _notify().build(str(_run(tmp_path, SKIPPED)))
        assert rc == 0
        assert body.startswith("ANALYSIS SKIPPED: model output and experiment.json "
                               "complete; whole-domain rows come next")

    def test_a_skip_because_nothing_succeeded_is_a_failure(self, tmp_path):
        rd = _run(tmp_path, {"status": "skipped", "reason": "no successful columns"})
        body, rc = _notify().build(str(rd))
        first = body.splitlines()[0]
        assert rc == 1 and first.startswith("INCOMPLETE") and "04_analysis is empty" in first

    def test_a_skip_without_its_product_is_a_failure(self, tmp_path):
        assert _notify().build(str(_run(tmp_path, SKIPPED, experiment=False)))[1] == 1

    def test_a_skip_without_a_reason_is_a_failure(self, tmp_path):
        assert _notify().build(str(_run(tmp_path, {"status": "skipped"})))[1] == 1

    def test_an_analysis_that_wrote_nothing_is_a_failure(self, tmp_path):
        body, rc = _notify().build(str(_run(tmp_path, {"status": "done"})))
        assert rc == 1 and "04_analysis is empty" in body

    def test_a_written_analysis_is_ok(self, tmp_path):
        body, rc = _notify().build(str(_run(tmp_path, {"status": "done"}, docs=2)))
        assert rc == 0 and body.startswith("OK") and "the analysis is written" in \
            body.splitlines()[0]


class TestAWholeDomainChain:
    """A failed year cancels the years after it: the study is complete only
    when every year it planned passed, whatever the analysis did."""

    def test_every_year_passed(self, tmp_path):
        rd = _run(tmp_path, SKIPPED, experiment=_domain([2015, 2016], [2015, 2016], None))
        body, rc = _notify().build(str(rd))
        assert rc == 0 and body.startswith("ANALYSIS SKIPPED: ")
        assert "chain : years passed [2015, 2016] of [2015, 2016]" in body

    def test_a_chain_that_stopped(self, tmp_path):
        rd = _run(tmp_path, SKIPPED, experiment=_domain([2015, 2016], [2015], "y2016"))
        body, rc = _notify().build(str(rd))
        first = body.splitlines()[0]
        assert rc == 1 and first.startswith("INCOMPLETE")
        assert "the chain stopped at y2016 (years passed [2015] of [2015, 2016])" in first
        assert "first stage without a passing check: y2016" in body

    def test_a_chain_that_never_reached_a_year(self, tmp_path):
        rd = _run(tmp_path, SKIPPED, experiment=_domain([2015], [], "case"))
        assert _notify().build(str(rd))[1] == 1

    def test_a_chain_with_no_planned_year(self, tmp_path):
        rd = _run(tmp_path, SKIPPED, experiment=_domain([], [], None))
        body, rc = _notify().build(str(rd))
        assert rc == 1 and "the chain stopped at ?" in body.splitlines()[0]
