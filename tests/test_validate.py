"""Offline unit tests for the deterministic brief/plan validator (no LLM)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.validate import check_brief, check_plan   # noqa: E402


class TestCheckBrief:
    def test_clean_site_brief(self):
        b = {"intent": "design", "design_archetype": "site",
             "domain": {"bbox": {"min_lon": -1, "min_lat": 1, "max_lon": 2, "max_lat": 3}},
             "observations_available": [], "observations_missing": []}
        assert check_brief(b) == []

    def test_site_missing_bbox(self):
        b = {"intent": "design", "design_archetype": "site", "domain": {}}
        assert any("bbox" in i for i in check_brief(b))

    def test_parse_error(self):
        assert check_brief({"intent": "parse_error", "error": "x"})

    def test_coupling_missing_block(self):
        assert any("coupling" in i for i in
                   check_brief({"intent": "design", "design_archetype": "coupling"}))

    def test_bad_archetype(self):
        assert any("archetype" in i for i in
                   check_brief({"intent": "design", "design_archetype": "nonsense"}))

    def test_obs_not_a_list(self):
        b = {"intent": "design", "design_archetype": "site",
             "domain": {"bbox": {"min_lon": -1, "min_lat": 1, "max_lon": 2, "max_lat": 3}},
             "observations_available": "oops"}
        assert any("observations_available" in i for i in check_brief(b))

    def test_not_a_dict(self):
        assert check_brief("nope")


class TestCheckPlan:
    def test_clean_site_plan(self):
        p = {"model_choice": {"design_archetype": "site"},
             "sampling_strategy": {"n_exploratory": 12},
             "sampling_plan": [{"group": "a", "n": 3, "reason": "x"}],
             "requires_capabilities": []}
        assert check_plan(p) == []

    def test_missing_sampling(self):
        iss = check_plan({"model_choice": {"design_archetype": "site"},
                          "sampling_strategy": {}})
        assert any("n_exploratory" in i for i in iss)
        assert any("sampling_plan" in i for i in iss)

    def test_coupling_needs_coupling_design(self):
        assert any("coupling_design" in i for i in
                   check_plan({"model_choice": {"design_archetype": "coupling"}}))

    def test_clean_coupling_plan(self):
        p = {"model_choice": {"design_archetype": "coupling"},
             "coupling_design": {"from_model": "ELM", "to_model": "PFLOTRAN"}}
        assert check_plan(p) == []

    def test_archetype_mismatch(self):
        p = {"model_choice": {"design_archetype": "site"},
             "sampling_strategy": {"n_exploratory": 3},
             "sampling_plan": [{"group": "a", "n": 3}]}
        assert any("archetype" in i for i in check_plan(p, {"design_archetype": "conceptual"}))

    def test_unparseable(self):
        assert check_plan(None)
