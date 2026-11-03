"""A design page a person could make the runs from (2026-10-28).

    pytest tests/test_design_preview.py -v

The test for the design review: the runs can be made from the page alone. The model server now checks a whole-domain design and builds every run
(recipe, checks, the scripts its jobs run) before the page, submitting
nothing; a design it refuses goes back to the planner with the server's own
findings; and the page carries the verdict, then a second part drawn from
what the server wrote. The manager and the server run together in process
(the manager tests' client), with the fake SLURM that fails a test on any
submission; no test reaches an LLM.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import domain_case as DC                                    # noqa: E402
from test_domain_manager import srv, mgr                    # noqa: E402,F401
from test_domain_experiment import study, NOFIRE            # noqa: E402,F401
from agents import design_review as dr                      # noqa: E402


def _reception():
    return {"user_request": "What does the model's own fire do to carbon in 2015-2016?",
            "brief": {"design_archetype": "whole_domain", "model": "elm",
                      "domain": {"name": "synthetic"}}}


def _preview(study, runs=None):
    m, cfg = study["mgr"], study["cfg"]
    st = json.loads(json.dumps(cfg["strategy"]))
    if runs is not None:
        st["domain_experiment"]["runs"] = runs
    return m, cfg, st, m.preview_design(st, dict(cfg, strategy=st))


# ── the server's check and build, before the page ───────────────────────────
class TestTheServerBuildsBeforeThePage:

    def test_it_writes_every_runs_files_and_submits_nothing(self, study):
        m, cfg, st, p = _preview(study)
        assert p["refused"] == [] and p["error"] is None, p
        assert not m.slurm.sbatch()
        run = p["runs"][0]
        assert (run["name"], run["case"], run["changes"]) == (
            "nofire", "src_case_nofire", ["fire_off"])
        assert (m.run_dir / run["case_script"]).is_file()
        assert (m.run_dir / run["stage_script"]).is_file()
        assert [j["stage"] for j in run["jobs"]] == ["case", "smoke", "y2015", "y2016",
                                                    "report"]
        # nothing armed: no chain record, so the yes still decides
        assert not (m.run_dir / "domain_case" / "nofire" / DC.CHAIN_FILE).exists()
        assert cfg["mcp_clients"]["elm"].calls == [
            "check_domain_experiment", "describe_domain_controls", "build_domain_runs",
            "check_domain_inputs", "read_domain_recipe", "write_domain_case"]
        assert (m.input_dir / "control_recipe.json").is_file()
        assert p["control_recipe"] == "01_inputs/control_recipe.json"
        assert p["checks_in_words"]["case"].startswith("before the model is built")
        on_disk = json.loads((m.input_dir / "design_preview.json").read_text())
        assert on_disk["runs"][0]["case_script"] == run["case_script"]

    def test_a_refused_design_is_recorded_not_raised(self, study):
        m, cfg, st, p = _preview(study, runs=[
            {"name": "moved", "changes": [{"id": "burn_footprint"}]}])
        assert p["runs"] == [] and p["error"] is None
        assert p["refused"] and p["refused"][0]["where"].startswith("run moved")
        assert "burn_footprint is not available here" in p["refused"][0]["what"]
        assert not m.slurm.sbatch()
        assert json.loads((m.input_dir / "design_preview.json").read_text())["refused"]

    def test_after_the_preview_a_yes_submits_as_before(self, study):
        """The coordinator previews with one manager and runs with another on
        the same run directory: the second builds again over the preview's
        files and submits the chain it always did."""
        from elm_exp_manager import ELMExpManager
        m, cfg, st, p = _preview(study)
        script = (m.run_dir / p["runs"][0]["case_script"]).read_text()
        again = ELMExpManager(base_output_dir=str(m.run_dir.parent), run_dir=str(m.run_dir))
        out = again.execute_plan(dict(st), dict(cfg, strategy=st))
        assert out.get("status") == "pending", out
        assert [(j["run"], j["stage"], j["job_id"]) for j in out["chain"]] == [
            ("nofire", "case", "2001"), ("nofire", "smoke", "2002"),
            ("nofire", "y2015", "2003"), ("nofire", "y2016", "2004"),
            ("nofire", "report", "2005")]
        rec = m.run_dir / "domain_case" / "nofire"
        assert (rec / DC.CHAIN_FILE).is_file()
        assert (rec / "create_case.sh").read_text() == script

    def test_the_control_alone_gets_its_recipe_and_no_run(self, study):
        m, cfg, st, p = _preview(study, runs=[])
        assert p["refused"] == [] and p["runs"] == []
        assert p["control_recipe"] == "01_inputs/control_recipe.json"
        assert "write_domain_case" not in cfg["mcp_clients"]["elm"].calls


# ── the page ─────────────────────────────────────────────────────────────────
class TestThePage:

    def test_the_verdict_first_and_the_recipe_after_the_decision(self, study):
        m, cfg, st, p = _preview(study)
        page = dr.render(_reception(), st, "x", preview=p, run_dir=m.run_dir)
        head, more = dr.decision_part(page)
        assert more and head.rstrip().endswith(dr._PENDING)
        assert "## The model server's check" in head and "**Passed.**" in head
        assert "create_newcase" not in head
        tail = page[len(head):]
        assert dr.APPENDIX_MARK in tail
        assert "+ use_nofire = .true." in tail             # the change, line by line
        assert "create_newcase --case" in tail             # the case script, verbatim
        assert "| report | " in tail and "every job above" in tail
        assert "no fire can burn with the fire model off (all_zero on FAREA_BURNED" in tail
        assert "the case job: before the model is built" in tail
        assert "`lnd_in` use_nofire = true" in tail
        assert "On a yes, the Experiment Manager has the server check the design" in head

    def test_a_refused_design_has_no_second_part_and_no_yes(self, study):
        m, cfg, st, p = _preview(study, runs=[
            {"name": "moved", "changes": [{"id": "burn_footprint"}]}])
        page = dr.render(_reception(), st, "x", preview=p, run_dir=m.run_dir)
        assert "**Refused.**" in page and "run moved" in page
        assert dr.APPENDIX_MARK not in page
        assert "Nothing can run until the design changes" in page

    def test_the_rounds_say_what_changed_and_why(self):
        p = {"refused": [], "error": None, "runs": [], "check": {}, "rounds": [
            {"findings": [{"where": "run warm", "what": "a case named c_warm exists"}],
             "design": "strategy.refused_1.json"}]}
        page = dr.render(_reception(), {}, "x", preview=p)
        assert "the server refused the planner's first design" in page
        assert "run warm: a case named c_warm exists" in page
        assert "`strategy.refused_1.json`" in page

    def test_write_review_reads_the_preview_from_disk(self, study):
        m, cfg, st, p = _preview(study)
        path = dr.write_review(m.run_dir, reception=_reception(), strategy=st)
        assert "**Passed.**" in path.read_text()

    def test_no_preview_leaves_the_page_as_it_was(self):
        page = dr.render(_reception(), {}, "x")
        assert "## The model server's check" not in page and dr.APPENDIX_MARK not in page


# ── a refusal goes back to the planner ───────────────────────────────────────
REFUSED = {"refused": [{"where": "run warm", "what": "a case named c_warm exists "
                        "already: give the run another name (for example warm_2)"}],
           "error": None, "runs": []}
PASSED = {"refused": [], "error": None, "runs": [], "check": {}}


def _coordinator(tmp_path, previews, interactive=False):
    import workflow as wf
    co = object.__new__(wf.WorkflowCoordinator)
    co.conversation_context = {}
    co.interactive_reception = interactive
    seq = iter(previews)
    co._preview = lambda plan, run_dir, result, output_dir: next(seq)
    co.calls = []

    def plan(result, run_dir, findings=None, previous=None):
        co.calls.append({"findings": findings, "previous": previous})
        return {"archetype": "whole_domain", "domain_experiment": {
            "control": {"name": "c", "reuse": True}, "years": [2015],
            "runs": [{"name": f"warm_{len(co.calls) + 1}"}]}}
    co._plan = plan
    return co


FIRST = {"archetype": "whole_domain", "domain_experiment": {
    "control": {"name": "c", "reuse": True}, "years": [2015], "runs": [{"name": "warm"}]}}


class TestARefusalGoesBackToThePlanner:

    def test_the_findings_and_the_refused_design_go_back(self, tmp_path):
        co = _coordinator(tmp_path, [dict(REFUSED), dict(PASSED)])
        plan, preview = co._review_ready(_reception(), FIRST, tmp_path, str(tmp_path))
        assert co.calls == [{"findings": REFUSED["refused"], "previous": FIRST}]
        assert json.loads((tmp_path / "strategy.refused_1.json").read_text()) == FIRST
        assert plan["domain_experiment"]["runs"][0]["name"] == "warm_2"
        assert preview["rounds"] == [{"findings": REFUSED["refused"],
                                      "design": "strategy.refused_1.json"}]
        rec = json.loads((tmp_path / "01_inputs" / "design_preview.json").read_text())
        assert rec["rounds"][0]["design"] == "strategy.refused_1.json"

    def test_it_gives_up_after_the_set_rounds(self, tmp_path):
        co = _coordinator(tmp_path, [dict(REFUSED) for _ in range(5)])
        plan, preview = co._review_ready(_reception(), FIRST, tmp_path, str(tmp_path))
        assert len(co.calls) == co.PREVIEW_ROUNDS == 2
        assert preview["refused"] and len(preview["rounds"]) == 2

    def test_unattended_a_refusal_stops_before_the_manager(self, tmp_path):
        co = _coordinator(tmp_path, [dict(REFUSED) for _ in range(5)])
        co._plan = lambda result, run_dir, findings=None, previous=None: FIRST
        co._execute = lambda **kw: pytest.fail("a refused design must not reach the manager")
        out = co._workflow_design_and_run(_reception(), str(tmp_path), tmp_path)
        assert out.startswith("🛑 The model server's check refused this design")
        assert "a case named c_warm exists already" in out
        assert "not run: the model server refused the design" in \
            (tmp_path / "DESIGN_REVIEW.md").read_text()

    def test_with_a_person_a_refused_design_has_no_yes(self, tmp_path, monkeypatch):
        co = _coordinator(tmp_path, [dict(REFUSED) for _ in range(5)], interactive=True)
        co._plan = lambda result, run_dir, findings=None, previous=None: FIRST
        co._execute = lambda **kw: pytest.fail("Enter at a refused design must not run")
        asked = []
        monkeypatch.setattr("builtins.input", lambda p="": asked.append(p) or "")
        out = co._workflow_design_and_run(_reception(), str(tmp_path), tmp_path)
        assert asked and asked[0].startswith("This design cannot run as it is")
        assert out.startswith("🛑 The model server's check refused this design, so "
                              "nothing was submitted.")
        assert "not run: the model server refused the design" in \
            (tmp_path / "DESIGN_REVIEW.md").read_text()

    def test_the_terminal_shows_the_decision_part(self, tmp_path, monkeypatch, capsys):
        co = _coordinator(tmp_path, [dict(PASSED)], interactive=True)
        co._plan = lambda result, run_dir, findings=None, previous=None: FIRST
        co._execute = lambda **kw: pytest.fail("declined")
        monkeypatch.setattr("builtins.input", lambda p="": "n")
        co._workflow_design_and_run(_reception(), str(tmp_path), tmp_path)
        out = capsys.readouterr().out
        assert "## The model server's check" in out and "**Passed.**" in out
        # the second part stays in the file, and the terminal says where (2026-11-03)
        page = (tmp_path / "DESIGN_REVIEW.md").read_text()
        assert dr.APPENDIX_MARK in page and "## The code" in page
        assert dr.APPENDIX_MARK not in out and "## The code" not in out
        assert ("The page's second part, how to make these runs by hand, is in "
                f"{tmp_path / 'DESIGN_REVIEW.md'}") in out


# ── a preview that failed is not a refusal (2026-11-03) ───────────────────────
ERROR = {"refused": [], "runs": [], "error": "TimeoutError: the server did not answer"}


class TestAFailedPreview:

    def test_unattended_an_error_says_the_design_could_not_be_built(self, tmp_path):
        co = _coordinator(tmp_path, [dict(ERROR)])
        co._plan = lambda result, run_dir, findings=None, previous=None: FIRST
        co._execute = lambda **kw: pytest.fail("an unbuilt design must not reach the manager")
        out = co._workflow_design_and_run(_reception(), str(tmp_path), tmp_path)
        assert out.startswith("🛑 The design could not be built before the review, so "
                              "nothing was submitted.")
        assert "the server did not answer" in out
        assert "refused" not in out and "Ask again with that change" not in out
        page = (tmp_path / "DESIGN_REVIEW.md").read_text()
        assert "not run: the design could not be built" in page
        assert "refused the design" not in page and "**Not built.**" in page

    def test_with_a_person_an_error_is_recorded_as_not_built(self, tmp_path, monkeypatch):
        co = _coordinator(tmp_path, [dict(ERROR)], interactive=True)
        co._plan = lambda result, run_dir, findings=None, previous=None: FIRST
        co._execute = lambda **kw: pytest.fail("Enter at an unbuilt design must not run")
        asked = []
        monkeypatch.setattr("builtins.input", lambda p="": asked.append(p) or "y")
        out = co._workflow_design_and_run(_reception(), str(tmp_path), tmp_path)
        assert asked and asked[0].startswith("This design cannot run as it is")
        assert out.startswith("🛑 The design could not be built before the review, so "
                              "nothing was submitted.")
        page = (tmp_path / "DESIGN_REVIEW.md").read_text()
        assert "not run: the design could not be built" in page
        assert "refused the design" not in page


# ── a refused design changed into one the server cannot check (2026-11-03) ───
class TestARefusalThenNoPreview:

    def test_the_preview_is_an_error_that_keeps_the_rounds(self, tmp_path):
        co = _coordinator(tmp_path, [dict(REFUSED), None])
        plan, preview = co._review_ready(_reception(), FIRST, tmp_path, str(tmp_path))
        assert preview["error"].startswith("the planner's changed design is not a "
                                           "whole-domain experiment any more")
        assert preview["rounds"] == [{"findings": REFUSED["refused"],
                                      "design": "strategy.refused_1.json"}]
        rec = json.loads((tmp_path / "01_inputs" / "design_preview.json").read_text())
        assert rec["error"] == preview["error"] and not rec["refused"]

    def test_with_a_person_it_has_no_yes(self, tmp_path, monkeypatch):
        co = _coordinator(tmp_path, [dict(REFUSED), None], interactive=True)
        co._execute = lambda **kw: pytest.fail("a design nobody checked must not run")
        # the refused round's record, as preview_design leaves it on disk
        (tmp_path / "01_inputs").mkdir()
        (tmp_path / "01_inputs" / "design_preview.json").write_text(json.dumps(REFUSED))
        asked = []
        monkeypatch.setattr("builtins.input", lambda p="": asked.append(p) or "y")
        co._workflow_design_and_run(_reception(), str(tmp_path), tmp_path)
        assert asked and asked[0].startswith("This design cannot run as it is")
        page = (tmp_path / "DESIGN_REVIEW.md").read_text()
        assert "**Not built.**" in page and "**Refused.**" not in page
        assert "not run: the design could not be built" in page

    def test_a_preview_given_as_none_is_not_read_from_disk(self, tmp_path):
        (tmp_path / "01_inputs").mkdir()
        (tmp_path / "01_inputs" / "design_preview.json").write_text(json.dumps(REFUSED))
        page = dr.write_review(tmp_path, reception=_reception(), strategy={},
                               preview=None).read_text()
        assert "## The model server's check" not in page
        # left out, the record on disk is still read
        page = dr.write_review(tmp_path, reception=_reception(), strategy={}).read_text()
        assert "**Refused.**" in page


# ── the patch line on the page (2026-11-03) ──────────────────────────────────
class TestThePatchLine:

    @staticmethod
    def _page(source):
        p = {"refused": [], "error": None, "runs": [], "check": {}, "source": source}
        return dr.render(_reception(), {}, "x", preview=p)

    def test_it_says_where_and_how_the_patch_applies(self):
        page = self._page({"srcroot": "/e3sm", "commit": "abc", "fingerprint": "f",
                           "changed": [" M cime/x.F90"],
                           "patch": "01_inputs/source_changes.patch",
                           "patch_left_out": []})
        assert ("saved as `01_inputs/source_changes.patch`, to apply with `git apply` at "
                "the checkout root once that commit and its submodules are checked out "
                "(untracked files are in it as new files)") in page
        assert "left out of the patch" not in page

    def test_files_left_out_of_the_patch_are_named(self):
        page = self._page({"srcroot": "/e3sm", "commit": "abc", "fingerprint": "f",
                           "changed": [], "untracked": ["big.nc", "New.F90"],
                           "patch": "01_inputs/source_changes.patch",
                           "patch_left_out": ["big.nc"]})
        assert "- left out of the patch, too large to carry, so copy them by hand: " \
               "`big.nc`" in page
        # untracked files alone still make a patch, and the page says so
        assert "no local changes to tracked files; the local changes are saved as" in page


# ── the planner's copy of the rulebook has no observed numbers (2026-11-03) ──
RULEBOOK = {
    "controls": {"registry": "r.json", "controls": [{
        "name": "naches_control", "usable": True,
        "evaluation": {"summary": "streamflow against the gauge: KGE 0.61"},
        "observations": {"fire": "MODIS MCD64A1 burned area"},
        "regions": {"watershed": "the watershed"},
        "facts": {"start_year": 2015,
                  "burned_area_km2": {"model": {"2020": 116.0}, "observed": {"2020": 286.0},
                                      "observed_source": "MODIS MCD64A1"}},
        "not_in_it": ["the observed fires: the run's fire is the model's own, lit and "
                      "spread in its own places and years, so a fire that happened is not "
                      "in it; burned area over the grid, the model against MODIS MCD64A1: "
                      "2020 116 against 286 km2"],
        "not_in_it_words": ["the observed fires: the run's fire is the model's own, lit "
                            "and spread in its own places and years, so a fire that "
                            "happened is not in it"]}]},
    "changes": {"changes": [{"id": "fire_off"}]}}


class TestThePlannersRulebook:

    def test_the_prompt_holds_no_observed_numbers(self):
        p = TestThePlannerSeesTheFindings._planner()
        p.domain_rulebook = json.loads(json.dumps(RULEBOOK))
        sent = {}

        def ask(messages, system_message=None):
            sent["system"] = system_message
            return "{}"
        p.llm.ask = ask
        p.plan({"brief": {"model": "elm"}})
        sys_msg = sent["system"]
        assert "WHOLE-DOMAIN RULEBOOK" in sys_msg
        for gone in ("KGE 0.61", '"evaluation"', '"observations"', '"not_in_it"',
                     "against", "286", '"observed"', '"observed_source"', "MCD64A1"):
            assert gone not in sys_msg, gone
        assert '"not_in_it_words"' in sys_msg and "a fire that happened is not in it" \
            in sys_msg
        assert '"model"' in sys_msg and "116" in sys_msg     # the model's own number stays
        assert '"fire_off"' in sys_msg
        # the rulebook itself stays whole: strategy.json carries it for the page
        assert p.domain_rulebook == RULEBOOK


class TestThePlannerSeesTheFindings:

    @staticmethod
    def _planner():
        from agents.planner import Planner
        p = object.__new__(Planner)
        p.system, p.constraints, p.pinning, p.domain_rulebook = "SYS", None, None, None

        class LLM:
            def ask(self, messages, system_message=None):
                self.content = messages[0]["content"]
                return "{}"
        p.llm = LLM()
        return p

    def test_the_refused_design_and_the_findings_are_in_the_message(self):
        p = self._planner()
        prev = dict(FIRST, pinning={"x": 1}, domain_rulebook={"y": 2})
        p.plan({"brief": {"model": "elm"}}, findings=REFUSED["refused"], previous=prev)
        c = p.llm.content
        assert p.FINDINGS_NOTE in c and "a case named c_warm exists" in c
        assert '"name": "warm"' in c
        assert '"pinning"' not in c and '"domain_rulebook"' not in c

    def test_without_findings_the_message_is_as_before(self):
        p = self._planner()
        p.plan({"brief": {"model": "elm"}})
        assert p.FINDINGS_NOTE not in p.llm.content


# ── what the server reads back for the page ──────────────────────────────────
class TestTheServerReadsItsOwnFiles:

    def test_an_anomaly_file_is_counted_from_the_file(self, tmp_path):
        import domain_weather as W
        ny, nx = 3, 4
        lon, lat = np.meshgrid(np.linspace(238.0, 238.3, nx), np.linspace(46.0, 46.2, ny))
        dom = {"xc": lon, "yc": lat, "area": np.full((ny, nx), 1e-6),
               "mask": np.ones((ny, nx), dtype=int), "frac": np.ones((ny, nx))}
        mask = np.zeros((ny, nx), dtype=bool)
        mask[1, 1:3] = True
        W.write_anomaly(tmp_path / "a.nc", dom, {"temperature_K": 2.0, "precip_factor": 0.8,
                                                 "months": [6, 7, 8]}, [2015], mask)
        s = W.summarize_anomaly(tmp_path / "a.nc")
        assert s["records"]["first"] == "2014-01" and s["records"]["n"] == 36
        t, p = s["fields"]["tas"], s["fields"]["pr"]
        assert t["months"] == [f"{y}-0{m}" for y in (2014, 2015, 2016) for m in (6, 7, 8)]
        assert (t["cells"], t["values"], t["identity"]) == (2, [2.0], 0.0)
        assert (p["cells"], p["values"], p["identity"]) == (2, [0.8], 1.0)
        assert s["params"]["months"] == [6, 7, 8]

    def test_the_controls_weather_streams(self, tmp_path):
        import domain_weather as W
        cd = tmp_path / "case"
        (cd / "CaseDocs").mkdir(parents=True)
        (cd / "CaseDocs" / "datm_in").write_text(
            '&shr_strdata_nml\n  streams = "datm.streams.txt.X 2013 2013 2014"\n/\n')
        files = "\n".join(f"clmforc.nldas.{y}-{m:02d}.nc" for y in (2013, 2014)
                          for m in range(1, 13))
        (cd / "CaseDocs" / "datm.streams.txt.X").write_text(
            '<?xml version="1.0"?>\n<file id="stream" version="1.0">\n'
            "<domainInfo><variableNames>time time</variableNames>"
            "<filePath>/d</filePath><fileNames>domain.nc</fileNames></domainInfo>\n"
            "<fieldInfo><variableNames>TBOT tbot\n PRECTmms precn</variableNames>"
            f"<filePath>/data/NLDAS</filePath><fileNames>{files}</fileNames></fieldInfo>\n"
            "</file>\n")
        (s,) = W.forcing_summary(cd)
        assert s["stream"] == "datm.streams.txt.X" and s["years"] == [2013, 2013, 2014]
        assert (s["n_files"], s["first"], s["last"]) == (24, "clmforc.nldas.2013-01.nc",
                                                         "clmforc.nldas.2014-12.nc")
        assert s["domain"] == "/d/domain.nc" and s["data_dir"] == "/data/NLDAS"
        assert s["fields"] == ["TBOT", "PRECTmms"]

    def test_the_checkouts_local_changes_as_a_patch(self, tmp_path):
        repo = tmp_path / "e3sm"
        repo.mkdir()

        def git(*a):
            return subprocess.run(["git", *a], cwd=str(repo), capture_output=True,
                                  text=True, check=True).stdout
        git("init", "-q")
        (repo / "Mod.F90").write_text("x = 1\n")
        git("add", "Mod.F90")
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c")
        (repo / "Mod.F90").write_text("x = 2\n")
        head = git("rev-parse", "HEAD").strip()
        patch = DC.source_patch(repo)
        assert f"at commit {head}" in patch
        assert "-x = 1" in patch and "+x = 2" in patch
        assert DC.source_state(repo)["changed"]
