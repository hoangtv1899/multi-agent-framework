"""The computing a yes allows, recorded with the yes and checked before any job.

    pytest tests/test_approved_spending.py -v

The design page gives the model server's estimate of a design's cost, in
node-hours. That figure is no part of what identifies the design, so it is
kept apart from the design's fingerprint: recorded with the approval as the
limit of what the yes allows, and checked on its own by the Experiment
Manager, after the server has checked the design again and before anything is
built or submitted (src/core/approved_spending.py).

    the rule        what a page's estimate allows, and when the estimate now
                    is over it; no figure on record is no limit.
    the record      the approval holds the figure of the page it was given on,
                    and a change of the figure changes neither the design's
                    fingerprint nor the plan's hash.
    the manager     with the made-up model server and scheduler of
                    tests/test_domain_experiment.py: a design at the cost it
                    was approved at is submitted as before, and one the server
                    now estimates above it is refused with both figures, with
                    nothing built and nothing submitted.

Every number is made up.
"""
import json

import pytest

from test_design_preview import _preview
from test_domain_experiment import NOFIRE, study  # noqa: F401
from test_domain_manager import mgr, srv  # noqa: F401

from agents import design_review as dr
from core import approved_spending as SP
from core.exp_manager_base import RefusedBeforeAnyJob


def _approve(rd, node_hours=None, **more):
    rec = dict({"decision": "accepted at the terminal", "when": "2031-01-01T00:00:00",
                "design_sha256": "a" * 64}, **more)
    if node_hours is not None:
        rec["authorized"] = {"node_hours": node_hours, "from": SP.SOURCE}
    p = rd / SP.APPROVAL_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec))
    return rec


# ── the rule ─────────────────────────────────────────────────────────────────
class TestTheRule:

    def test_the_two_modules_name_one_file(self):
        assert SP.APPROVAL_FILE == dr.APPROVAL_FILE

    @pytest.mark.parametrize("preview, node_hours", [
        ({"check": {"cost": {"node_hours": 4.5}}}, 4.5),
        ({"cost": {"node_hours": 3}}, 3.0),                 # the check itself
        ({"check": {"cost": {"node_hours": 0.0}}}, 0.0),    # a control read alone
        ({"check": {"cost": {}}}, None), ({"check": {}}, None), ({}, None), (None, None),
        ({"check": {"cost": {"node_hours": "4.5"}}}, None),
        ({"check": {"cost": {"node_hours": -1.0}}}, None),
        ({"check": {"cost": {"node_hours": float("nan")}}}, None),
        ({"check": {"cost": {"node_hours": True}}}, None)])
    def test_what_a_page_estimates(self, preview, node_hours):
        assert SP.estimate(preview) == node_hours
        want = None if node_hours is None else {"node_hours": node_hours, "from": SP.SOURCE}
        assert SP.authorized(preview) == want

    @pytest.mark.parametrize("allowed, now, over", [
        (4.5, 4.5, False), (4.5, 4.4, False), (4.5, 0.0, False),
        (4.5, 4.6, True), (0.0, 0.1, True), (4.5, 45.0, True),
        (4.5, None, False),                                  # no estimate now: nothing to hold
        (4.5, "9", False)])
    def test_when_the_estimate_now_is_over_what_the_yes_allows(self, tmp_path, allowed, now,
                                                               over):
        _approve(tmp_path, allowed)
        why = SP.problem(tmp_path, now)
        assert (why is not None) is over
        if over:
            assert f"about {now:g} node-hours" in why and f"the {allowed:g} node-hours" in why
            assert "nothing was submitted" in why and SP.APPROVAL_FILE in why

    def test_no_figure_on_record_is_no_limit(self, tmp_path):
        assert SP.problem(tmp_path, 99.0) is None            # no approval at all
        _approve(tmp_path)                                   # one made before this rule
        assert SP.problem(tmp_path, 99.0) is None
        (tmp_path / SP.APPROVAL_FILE).write_text("{ not json")
        assert SP.problem(tmp_path, 99.0) is None
        (tmp_path / SP.APPROVAL_FILE).write_text(json.dumps(["a list"]))
        assert SP.problem(tmp_path, 99.0) is None


# ── the record ───────────────────────────────────────────────────────────────
STRATEGY = {"domain_experiment": {"control": {"name": "synth", "reuse": True},
                                  "runs": [{"name": "warm", "changes": []}],
                                  "years": [2031, 2032]}}


class TestTheRecord:

    def _page(self, rd, node_hours):
        p = rd / dr.PREVIEW_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"refused": [], "error": None,
                                 "check": {"cost": {"node_hours": node_hours}}}))

    def test_the_approval_holds_the_figure_of_its_page(self, tmp_path):
        self._page(tmp_path, 4.5)
        dr.approve(tmp_path, STRATEGY, "accepted at the terminal")
        rec = json.loads((tmp_path / dr.APPROVAL_FILE).read_text())
        assert rec["authorized"] == {"node_hours": 4.5, "from": SP.SOURCE}
        assert rec["design_sha256"] == dr.design_fingerprint(STRATEGY)
        assert SP.problem(tmp_path, 4.5) is None and SP.problem(tmp_path, 5.0)

    def test_a_page_with_no_estimate_records_no_limit(self, tmp_path):
        dr.approve(tmp_path, STRATEGY, "accepted at the terminal")
        assert "authorized" not in json.loads((tmp_path / dr.APPROVAL_FILE).read_text())

    def test_the_cost_is_no_part_of_what_identifies_the_design(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        self._page(a, 4.5)
        self._page(b, 9.0)
        dr.approve(a, STRATEGY, "accepted at the terminal")
        dr.approve(b, STRATEGY, "accepted at the terminal")
        ra = json.loads((a / dr.APPROVAL_FILE).read_text())
        rb = json.loads((b / dr.APPROVAL_FILE).read_text())
        assert ra["design_sha256"] == rb["design_sha256"]
        assert ra["authorized"]["node_hours"] != rb["authorized"]["node_hours"]
        # and a design at another cost is still the design approved
        assert dr.approval_problem(a, STRATEGY) is None


# ── the page ─────────────────────────────────────────────────────────────────
class TestThePageSaysWhatAYesAllows:

    RUN = {"name": "warm", "changes": []}

    def _decision(self, runs, node_hours):
        from agents import design_page as DP
        strategy = {"feasibility": {"verdict": "full"},
                    "domain_experiment": {"control": {"name": "synth", "reuse": True},
                                          "runs": runs, "years": [2031, 2032]}}
        preview = {"refused": [], "error": None, "runs": runs,
                   "check": {"cost": {"node_hours": node_hours} if node_hours is not None
                             else {}}}
        page = DP.render({"user_request": "a question", "brief": {}}, strategy,
                         preview=preview)
        return " ".join(page.split("## Your decision", 1)[1].split())

    def test_a_design_with_a_run_gives_the_figure_a_yes_allows(self):
        said = self._decision([self.RUN], 4.5)
        assert "It allows the 4.5 node-hours estimated above, and no more." in said

    def test_a_design_with_no_estimate_or_no_run_names_no_figure(self):
        assert "It allows" not in self._decision([self.RUN], None)
        assert "It allows" not in self._decision([], 0.0)


# ── the manager, before anything is built ────────────────────────────────────
class TestTheManagerHoldsToIt:

    def _yes(self, study, node_hours):
        """The page's preview, then a yes that allows `node_hours`."""
        m, cfg, st, p = _preview(study)
        assert p["refused"] == [] and p["error"] is None, p
        now = SP.estimate(p)
        assert now is not None and now > 0
        _approve(m.run_dir, node_hours if node_hours is not None else now)
        return m, cfg, st, now

    def _again(self, m):
        from elm_exp_manager import ELMExpManager
        return ELMExpManager(base_output_dir=str(m.run_dir.parent), run_dir=str(m.run_dir))

    def test_a_design_at_the_cost_it_was_approved_at_is_submitted(self, study):
        m, cfg, st, now = self._yes(study, None)
        out = self._again(m).execute_plan(dict(st), dict(cfg, strategy=st))
        assert out.get("status") == "pending", out
        assert [j["stage"] for j in out["chain"]] == ["case", "smoke", "y2015", "y2016",
                                                      "report"]

    def test_a_design_the_server_now_estimates_above_it_is_refused(self, study):
        m, cfg, st, now = self._yes(study, 0.1)
        calls = cfg["mcp_clients"]["elm"].calls
        before = len(calls)
        with pytest.raises(RefusedBeforeAnyJob) as e:
            self._again(m).execute_plan(dict(st), dict(cfg, strategy=st))
        said = str(e.value)
        assert f"about {now:g} node-hours" in said and "the 0.1 node-hours" in said
        assert e.value.findings == [{"where": "the cost", "what": said}]
        # the server checked the design again, and nothing after that was asked of it
        assert calls[before:] == ["check_domain_experiment"]
        assert not m.slurm.sbatch()

    def test_before_the_yes_the_page_is_drawn_whatever_the_cost(self, study):
        """The preview makes the same check, with no approval on record."""
        m, cfg, st, p = _preview(study)
        assert not (m.run_dir / SP.APPROVAL_FILE).exists()
        assert p["error"] is None and SP.estimate(p) > 0
