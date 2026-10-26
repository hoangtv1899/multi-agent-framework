"""The Analyzer on a whole-domain experiment: steps 0 to 4 on regional numbers.

    pytest tests/test_domain_analyzer.py -v

docs/ELM_DOMAIN_EXPERIMENTS.md, step 5b. A run directory written here holds
the three boundary files with the keys the ELM manager's package writes for an
experiment (a reused control and a fire-off run, three years, two regions and
a burned-area observation), so every number a step should report is known.
No LLM: steps 2 and 3 are given a scripted client.
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
from agents.analysis import step0_context as S0          # noqa: E402
from agents.analysis import step1_compare as S1          # noqa: E402
from agents.analysis import step2_investigate as S2      # noqa: E402
from agents.analysis import step3_interpret as S3        # noqa: E402
from agents.analysis import step4_report as S4           # noqa: E402
from agents.analysis import script_runner as SR          # noqa: E402

YEARS = [2015, 2016, 2017]
REGIONS = {"vegetated cells": {"cells": 100, "area_km2": 60.0, "rule": "PCT_NATVEG >= 100"},
           "watershed, vegetated": {"cells": 40, "area_km2": 24.0,
                                    "rule": "cell centre inside the polygon, vegetated"}}
FIELDS = {"GPP": {"stream": "h0", "how": "sum", "units": "gC/m^2/yr", "note": None},
          "FAREA_BURNED": {"stream": "h0", "how": "sum", "units": "proportion/yr",
                           "note": "a fraction of the cell burned PER SECOND"}}
CONTROL_GPP = {"vegetated cells": {"2015": 487.0, "2016": 536.5, "2017": 406.0},
               "watershed, vegetated": {"2015": 643.7, "2016": 692.0, "2017": 591.0}}
NOFIRE_GPP = {"vegetated cells": {"2015": 480.0, "2016": 530.0, "2017": 400.0},
              "watershed, vegetated": {"2015": 640.0, "2016": 690.0, "2017": 588.0}}
FIRE = {"control": {"2015": 0.01, "2016": 0.005, "2017": 0.003},
        "nofire": {"2015": 0.0, "2016": 0.0, "2017": 0.0}}
FIRE_OFF = {"id": "fire_off", "params": {},
            "summary": "switch ELM's fire model off",
            "meaning": "the difference from the control is what the model's own fire does",
            "not_meaning": "the effect of an observed fire"}


def _run(name, gpp):
    return {"run_dir": f"/somewhere/{name}", "fields": FIELDS, "missing": {}, "problems": [],
            "values": {reg: {"GPP": dict(gpp[reg]), "FAREA_BURNED": dict(FIRE[name])}
                       for reg in REGIONS}}


def regional(event=True):
    runs = {"control": _run("control", CONTROL_GPP), "nofire": _run("nofire", NOFIRE_GPP)}
    t = {"regions": REGIONS, "years": YEARS, "runs": runs, "problems": [],
         "differences": {"nofire": {reg: {v: {y: runs["nofire"]["values"][reg][v][y]
                                              - runs["control"]["values"][reg][v][y]
                                              for y in map(str, YEARS)} for v in FIELDS}
                                    for reg in REGIONS}},
         "observed": {"burned_area_km2": {
             "what": "burned area", "units": "km2", "source": "MODIS MCD64A1",
             "values": {"vegetated cells": {"2015": 0.2, "2016": 0.1, "2017": 3.0},
                        "watershed, vegetated": {"2015": 0.1, "2016": 0.0, "2017": 2.5}},
             "model_field": "FAREA_BURNED",
             "model": {r: {reg: {y: FIRE[r][y] * REGIONS[reg]["area_km2"]
                                 for y in map(str, YEARS)} for reg in REGIONS}
                       for r in ("control", "nofire")}}}}
    if event:
        t["event"] = {"normal_years": [2016, 2017], "target_years": [2015], "problems": [],
                      "anomalies": {reg: {"GPP": {
                          "normal": (CONTROL_GPP[reg]["2016"] + CONTROL_GPP[reg]["2017"]) / 2,
                          "years": {"2015": {"value": CONTROL_GPP[reg]["2015"]}}}}
                          for reg in REGIONS}}
    return t


EXPERIMENT_RUNS = [
    {"run": "control", "case": "naches_bgc_fix", "case_dir": "/c/naches_bgc_fix",
     "reused": True, "status": "reused", "changes": [], "source": None, "built_from": None,
     "after": None},
    {"run": "nofire", "case": "naches_bgc_fix_nofire", "case_dir": "/c/nofire",
     "reused": False, "status": "completed", "changes": [FIRE_OFF], "source": None,
     "built_from": None, "after": None}]


def experiment_json(table=None, chain=None):
    """experiment.json as elm_exp_manager._package_experiment writes it."""
    rows = [{"case_name": "naches_bgc_fix", "kind": "whole_domain", "run": "control",
             "year": y, "status": "reused", "metrics": {"gpp_gC_m2_yr": CONTROL_GPP[
                 "vegetated cells"][str(y)]}} for y in YEARS] + \
        [{"case_name": "naches_bgc_fix_nofire", "kind": "whole_domain", "run": "nofire",
          "year": y, "status": "completed", "metrics": {"gpp_gC_m2_yr": NOFIRE_GPP[
              "vegetated cells"][str(y)]}} for y in YEARS]
    return {
        "model": "elm", "kind": "whole_domain", "archetype": "whole_domain",
        "run_dir": "/somewhere/run", "created": "2026-10-25T12:00:00",
        "domain": {"name": "Naches", "huc": None, "bbox": None},
        "experiment": {"control": {"name": "naches", "reuse": True},
                       "runs": EXPERIMENT_RUNS,
                       "check": {"facts": [{"where": "control",
                                            "what": "the control's own fire burns 60-115 "
                                                    "km2 a year"}],
                                 "unusual": [{"where": "control",
                                              "what": "reused without a record"}],
                                 "cost": {"node_hours": 2.4}}},
        "control_facts": {"name": "naches", "per_year": {}},
        "chain": chain or {"years_planned": YEARS, "years_passed": YEARS,
                           "first_unpassed": None, "runs": {},
                           "compute": {"runs": {"nofire": {"jobs": [], "seconds": 8580}},
                                       "seconds": 8580, "note": None,
                                       "control": "reused: its output was read, nothing "
                                                  "was spent on it"}},
        "columns_total": len(rows), "columns_succeeded": len(rows),
        "variable_units": {"gpp_gC_m2_yr": "gC/m^2/yr"},
        "strategy_check": {"stop": [], "warn": [], "corrections": []},
        "goals": ["say what the model's own fire does to GPP"],
        "columns": rows,
        "regional": regional() if table is None else table,
        "artifacts": {"regional": "03_results/regional.json"},
        "limitations": {
            "structural": [{"applies_to": "any comparison of runoff (QRUNOFF) with a "
                                          "stream gauge",
                            "caveat": "regional runoff is what the cells generate, not "
                                      "streamflow routed to a gauge.",
                            "kind": "structural", "severity": "blocking"}],
            "configuration": [{"applies_to": "differences between runs",
                               "caveat": "every run reads the control's weather.",
                               "kind": "configuration", "severity": "context"}]},
    }


STRATEGY = {
    "archetype": "whole_domain", "sampling": {"n_columns": 1},
    "goals": ["say what the model's own fire does to GPP"],
    "feasibility": {"verdict": "full", "why": "fire_off is available"},
    "domain_experiment": {
        "control": {"name": "naches", "reuse": True},
        "runs": [{"name": "nofire", "changes": [{"id": "fire_off"}]}],
        "years": YEARS,
        "analysis": {"regions": [{"name": "vegetated cells"}, {"name": "watershed"}],
                     "variables": ["GPP", "FAREA_BURNED"], "normal_years": [2016, 2017]},
        "caveats": ["the no-fire difference is the model's own fire \u2014 not an observed "
                    "fire"]}}


def write_run(tmp_path, experiment=None, strategy=None):
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "reception.json").write_text(json.dumps({
        "user_request": "What does the model's own fire do to GPP in the Naches?",
        "brief": {"design_archetype": "whole_domain", "domain": {"name": "Naches"}},
        "observations": {}}))
    (rd / "strategy.json").write_text(json.dumps(strategy or STRATEGY))
    (rd / "experiment.json").write_text(json.dumps(experiment or experiment_json()))
    return rd


@pytest.fixture
def ctx(tmp_path):
    return S0.load(write_run(tmp_path))


# ── step 0 ──────────────────────────────────────────────────────────────────
class TestStep0:

    def test_every_key_of_the_package_is_classified(self, ctx):
        assert ctx.is_domain
        assert ctx.data["kind"] == "whole_domain" and ctx.data["regional"]["years"] == YEARS
        assert "experiment" not in ctx.data and "limitations" not in ctx.data
        de = ctx.plan["experiment"]
        assert de["control"] == {"name": "naches", "reuse": True}
        assert [r["run"] for r in de["runs"]] == ["control", "nofire"]
        assert de["planned_runs"] == STRATEGY["domain_experiment"]["runs"]
        assert de["analysis"]["normal_years"] == [2016, 2017]

    def test_the_regional_frame(self, ctx):
        f = ctx.regional()
        assert list(f.columns) == ["run", "region", "variable", "year", "value", "units",
                                   "how", "source"]
        g = f[(f.run == "control") & (f.region == "vegetated cells") & (f.variable == "GPP")]
        assert dict(zip(g.year, g.value)) == {2015: 487.0, 2016: 536.5, 2017: 406.0}
        assert set(g.units) == {"gC/m^2/yr"} and set(g.how) == {"sum"}
        obs = f[f.source == "observation"]
        assert set(obs.run) == {"observed"} and set(obs.variable) == {"burned_area_km2"}
        mod = f[(f.variable == "burned_area_km2") & (f.source == "model")]
        assert set(mod.run) == {"control", "nofire"}
        assert mod[(mod.run == "control") & (mod.region == "vegetated cells")
                   & (mod.year == 2015)].value.item() == pytest.approx(0.6)
        # 2 runs x 2 regions x 2 fields x 3 years, then the observation and
        # each run's own burned area: 2 regions x 3 years each
        assert len(f) == 24 + 6 + 12

    def test_preflight_reads_the_frame(self, ctx):
        pre = ctx.preflight()
        assert pre["blocked"] is None
        assert pre["frames"]["regional"]["rows"] == 42
        assert pre["frames"]["series"] is None
        assert {"GPP", "FAREA_BURNED", "burned_area_km2"} <= set(pre["variables"])

    @pytest.mark.parametrize("table,why", [({"error": "ValueError: boom"}, "(ValueError: boom)"),
                                           ({}, "(its plan names no analysis")])
    def test_no_regional_numbers_blocks(self, tmp_path, table, why):
        e = experiment_json()
        if table:
            e["regional"] = table
        else:
            del e["regional"]
        c = S0.load(write_run(tmp_path, experiment=e))
        b = c.preflight()["blocked"]
        assert b.startswith("this whole-domain run has no regional numbers") and why in b

    def test_planned_against_actual(self, ctx):
        claims = {c["claim"]: c for c in ctx.planned_vs_actual()}
        assert set(claims) == {"years", "runs", "region:vegetated cells", "region:watershed",
                               "variable:GPP", "variable:FAREA_BURNED", "normal_years"}
        assert all(c["ok"] for c in claims.values()), claims
        assert claims["region:watershed"]["actual"] == ["watershed, vegetated"]
        assert claims["runs"]["actual"] == ["control", "nofire"]

    def test_what_the_table_lacks_is_unmet(self, tmp_path):
        s = json.loads(json.dumps(STRATEGY))
        s["domain_experiment"]["analysis"]["regions"].append({"name": "burned"})
        s["domain_experiment"]["analysis"]["variables"].append("NPP")
        # a run the plan named that was never built: planned runs are the
        # strategy's, not the package's
        s["domain_experiment"]["runs"].append({"name": "co2", "changes": [{"id": "co2_ppmv"}]})
        e = experiment_json(chain={"years_planned": YEARS, "years_passed": [2015, 2016],
                                   "first_unpassed": "nofire: y2017", "runs": {},
                                   "compute": {}})
        c = S0.load(write_run(tmp_path, experiment=e, strategy=s))
        unmet = {x["claim"] for x in c.planned_vs_actual() if not x["ok"]}
        assert unmet == {"years", "runs", "region:burned", "variable:NPP"}
        assert {x["claim"] for x in c.preflight()["unmet_plan_targets"]} == unmet

    def test_the_caveats_come_from_the_records(self, ctx):
        by = {c["id"]: c for c in ctx.caveats}
        nm = by["not_meaning_nofire_fire_off"]
        assert nm["severity"] == "blocking" and nm["statement"] == (
            "run nofire (fire_off): its difference from the control is NOT the effect of an "
            "observed fire")
        assert by["experiment_unusual_1"]["severity"] == "qualify"
        assert by["experiment_unusual_1"]["statement"] == "control: reused without a record"
        assert by["experiment_fact_1"]["severity"] == "context"
        assert by["plan_caveat_1"]["severity"] == "qualify"
        assert by["field_note_FAREA_BURNED"]["applies_to"] == "FAREA_BURNED"
        assert by["limitation_structural_1"]["severity"] == "blocking"
        assert by["limitation_configuration_2"]["severity"] == "context"
        assert "chain_incomplete" not in by
        assert not any("1 m" in c["statement"] or "column" in c["statement"].lower()
                       for c in ctx.caveats)

    def test_an_incomplete_chain_binds(self, tmp_path):
        e = experiment_json(chain={"years_planned": YEARS, "years_passed": [2015, 2016],
                                   "first_unpassed": "nofire: y2017", "runs": {}})
        c = S0.load(write_run(tmp_path, experiment=e))
        ch = {x["id"]: x for x in c.caveats}["chain_incomplete"]
        assert ch["severity"] == "blocking" and "nofire: y2017" in ch["statement"]

    def test_a_site_run_is_untouched(self, tmp_path):
        """No kind, no experiment: the plan carries no experiment and step 0
        builds no regional frame."""
        rd = tmp_path / "site"
        rd.mkdir()
        (rd / "reception.json").write_text(json.dumps({"user_request": "q"}))
        (rd / "strategy.json").write_text(json.dumps({"archetype": "site"}))
        (rd / "experiment.json").write_text(json.dumps({"model": "elm", "columns": []}))
        c = S0.load(rd)
        assert not c.is_domain and c.plan["experiment"] is None and c.regional() is None
        assert c.preflight()["blocked"].startswith("this run packaged no columns")


# ── the figure scripts' namespace ───────────────────────────────────────────
class TestTheScriptNamespace:

    def test_reg_regions_and_runs_are_bound(self, ctx):
        p = SR._payload(ctx)
        assert p["df"] is None and p["soil"] is None and p["prof"] is None
        assert len(p["reg"]) == 42 and p["regions"] == REGIONS
        assert [r["run"] for r in p["runs"]] == ["control", "nofire"]
        assert p["runs"][1]["changes"][0]["not_meaning"] == "the effect of an observed fire"

    def test_a_script_runs_against_reg_and_reruns_from_the_run_dir(self, ctx, tmp_path):
        code = ("g = reg[(reg.variable == 'GPP') & (reg.run == 'control') & "
                "(reg.region == 'vegetated cells')].sort_values('year')\n"
                "fig, ax = plt.subplots()\nax.plot(g.year, g.value)\n"
                "fig.savefig(out_path)\n"
                "result = {'n': int(len(g)), 'gpp': dict(zip(g.year.astype(str), g.value))}\n")
        out = SR.run(code, ctx, tmp_path / "f.png", script_path=tmp_path / "f.py")
        assert out["ok"], out.get("error")
        assert out["result"] == {"n": 3, "gpp": {"2015": 487.0, "2016": 536.5, "2017": 406.0}}
        import subprocess
        import sys
        (tmp_path / "f.png").unlink()
        r = subprocess.run([sys.executable, str(tmp_path / "f.py")], capture_output=True,
                           text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-2000:]
        assert (tmp_path / "f.png").is_file()


# ── step 1 ──────────────────────────────────────────────────────────────────
class TestStep1:

    def test_the_observation_beside_the_models_own(self, ctx, tmp_path):
        rec = S1.compare_all(ctx, tmp_path / "a")
        assert rec["note"] == S1.DOMAIN_NOTE and "skipped" not in rec
        (f,) = rec["findings"]
        assert f["id"] == "compare_burned_area_km2" and f["scale"] == "regions"
        assert f["n"] == 6                       # 2 regions x 3 years, both present
        b = f["result"]
        assert b["units"] == "km2" and b["regions"] == list(REGIONS)
        assert b["model"]["control"]["vegetated cells"]["2015"] == pytest.approx(0.6)
        assert b["observed"]["vegetated cells"]["2017"] == 3.0
        assert json.loads((tmp_path / "a" / S1.FILENAME).read_text())["findings"][0]["n"] == 6
        text = S1.format_comparison(rec["summary"])
        assert "burned_area_km2 [km2]: 2 region(s), 3 year(s), 6 region-years with both" in text
        assert "      vegetated cells, observed: 2015=0.2, 2016=0.1, 2017=3.0" in text
        assert "      watershed, vegetated, run control: 2015=0.24, 2016=0.12" in text

    def test_nothing_registered_beside_the_numbers(self, tmp_path):
        t = regional()
        del t["observed"]
        c = S0.load(write_run(tmp_path, experiment=experiment_json(table=t)))
        rec = S1.compare_all(c, tmp_path / "a")
        assert rec["findings"] == [] and rec["skipped"].startswith("no station comparison")


# ── step 2 ──────────────────────────────────────────────────────────────────
FIGURE = {"id": "gpp_by_year", "question": "How did GPP change through the years?",
          "why": "the domain's own trajectory", "scale": "overall", "variables": ["GPP"],
          "code": ("g = reg[(reg.variable == 'GPP') & (reg.run == 'control') & "
                   "(reg.region == 'vegetated cells')].sort_values('year')\n"
                   "fig, ax = plt.subplots()\nax.plot(g.year, g.value)\n"
                   "ax.set_xlabel('year', fontsize=16)\nfig.savefig(out_path)\n"
                   "result = {'n': int(len(g)), "
                   "'gpp': {str(int(y)): float(v) for y, v in zip(g.year, g.value)}}\n")}
CLAIM = {"claim": "Gross primary production (GPP) over the vegetated cells was 487 "
                  "gC/m^2/yr in 2015.",
         "finding_id": "gpp_by_year", "values": [487.0], "caveats": []}
REVIEW = {"claims": [CLAIM], "headline": CLAIM["claim"],
          "answer": "The control's GPP was highest in 2016 and lowest in 2017. These are "
                    "the model's numbers, not measurements.",
          "verdict": "sufficient", "feedback": ""}


class Scripted:
    """An LLM client that answers step 2 with figures and step 3 with a review.
    Which step is asking is read from the prompt's task: a client handed to a
    step directly is never labelled (only the clients a step builds are)."""
    calls = []

    def __init__(self, model=None, figures=(FIGURE,), review=REVIEW):
        self.label, self.figures, self.review = None, list(figures), review

    def ask(self, messages, system_message=None):
        content = messages[-1]["content"]
        text = content if isinstance(content, str) else content[0]["text"]
        step2 = "write the code that draws them" in text
        Scripted.calls.append((self.label, text))
        if step2:
            return json.dumps({"reasoning": "GPP is the carbon the question is about.",
                               "notes": "", "figures": self.figures})
        return json.dumps(self.review)


class TestStep2:

    def test_the_brief_is_the_experiments(self, ctx):
        b = S2.context_brief(ctx, S1.compare_all(ctx, ctx.run_dir / "04_analysis"))
        assert b.startswith("THE USER ASKED:\n    What does the model's own fire do to GPP")
        for line in (
                '    One ELM case over the whole prepared domain of the control "naches", '
                'model years 2015-2017.',
                "    The control was REUSED: its existing output was read, and nothing was "
                "run for it.",
                "      run nofire: the control with fire_off (status completed)",
                "          fire_off MEANS: the difference from the control is what the "
                "model's own fire does",
                "          it does NOT mean: the effect of an observed fire",
                "    watershed, vegetated: 40 cells, 24.0 km2 (cell centre inside the "
                "polygon, vegetated)",
                "NORMAL YEARS (the planner's, for a question about an event): 2016-2017; the "
                "years compared with them: 2015.",
                "    [not_meaning_nofire_fire_off] applies to: what run nofire's difference "
                "from the control is called",
                "THE PLANNER JUDGED THIS STUDY FULL:"):
            assert line in b, line
        assert "    GPP              gC/m^2/yr        total over the year" in b
        assert "burned_area_km2 [km2]: 2 region(s)" in b
        assert "1-D" not in b and "do not exchange" not in b and "basin total" not in b
        assert S2.task_for(ctx) is S2.DOMAIN_TASK and "reg       tidy frame" in S2.DOMAIN_TASK

    def test_a_proposal_runs_against_the_frame(self, ctx, tmp_path):
        absent = dict(FIGURE, id="npp", variables=["NPP"])
        inv = S2.investigate(ctx, tmp_path, client=Scripted(figures=(FIGURE, absent)))
        assert inv["n_succeeded"] == 1 and inv["findings"][0]["n"] == 3
        assert inv["findings"][0]["result"]["gpp"] == {"2015": 487.0, "2016": 536.5,
                                                       "2017": 406.0}
        (fail,) = inv["caveats"]
        assert fail["id"] == "figure_failed:npp" and "was not run" in fail["statement"]
        assert "DOMAIN_TASK" not in inv and Path(inv["plan_md"]).is_file()


# ── step 3 ──────────────────────────────────────────────────────────────────
class TestStep3:

    def test_the_facts_of_record(self, ctx):
        f = S3.run_facts(ctx)
        assert {"100", "40", "60", "24", "2015", "2016", "2017", "2", "60.0"} <= f
        assert str(len(ctx.columns)) not in f          # 6 run-year rows: not a fact

    def test_a_claim_quoting_a_region_passes_the_audit(self, ctx):
        inv = {"findings": [{"id": "gpp_by_year", "result": {"gpp": {"2015": 487.0}},
                             "n": 3, "blocked_by": []}]}
        claims = [dict(CLAIM, claim="Over the 40 vegetated cells of the watershed, 24 km2.",
                       values=[487.0, 40, 24.0])]
        a = S3.audit(claims, inv, ctx.caveats, facts=S3.run_facts(ctx))
        assert a["n_struck"] == 0, a["struck"]

    def test_interpret_with_a_scripted_reviewer(self, ctx, tmp_path):
        inv = S2.investigate(ctx, tmp_path, client=Scripted())
        out = S3.interpret(ctx, {}, inv, tmp_path, client=Scripted(), with_images=False)
        assert out["verdict"] == "sufficient" and out["audit"] == {"n_claims": 1,
                                                                   "n_struck": 0}
        assert out["readability"]["problems_after"] == []


# ── step 4 ──────────────────────────────────────────────────────────────────
class TestStep4:

    def test_the_report_speaks_of_years_and_runs(self, ctx, tmp_path):
        (ctx.run_dir / "04_analysis").mkdir()
        (ctx.run_dir / "04_analysis" / "DOMAIN_REPORT.md").write_text("numbers")
        inv = S2.investigate(ctx, tmp_path, client=Scripted())
        it = S3.interpret(ctx, {}, inv, tmp_path, client=Scripted(), with_images=False)
        rep = S4.build(ctx, {}, inv, it, ctx.run_dir,
                       steps={"context": True, "compare": True, "investigate": True,
                              "interpret": True})
        assert rep["status"] == S4.SUPPORTED and rep["kind"] == "whole_domain"
        assert rep["domain_report"] == "DOMAIN_REPORT.md"
        comp = rep["cost"]["compute"]
        assert comp["model_seconds"] == 8580 and comp["runs"] == {"nofire": 8580}
        assert comp["control"].startswith("reused") and comp["years_passed"] == YEARS
        line = ("years passed [2015, 2016, 2017] of [2015, 2016, 2017]; 1 run(s) executed, "
                "8580 s of model time; the control reused")
        assert S4.scope_line(rep) == line
        md = S4.render_markdown(rep)
        assert f"- {line}" in md and "- columns:" not in md
        assert ("- the numbers by region and year, written from the records before this "
                "analysis: DOMAIN_REPORT.md") in md
        assert f"compute  : {line}" in S4.summary(rep)

    def test_a_site_report_keeps_its_columns_line(self):
        rep = {"kind": None, "cost": {"compute": {"columns_succeeded": 3,
                                                  "columns_total": 4}}}
        assert "- columns: 3 of 4 succeeded" in S4.render_markdown(rep)
        assert S4.scope_line(rep) == "3 of 4 columns succeeded"


# ── the whole box ───────────────────────────────────────────────────────────
class TestTheAnalyzer:

    def test_steps_0_to_4_on_a_whole_domain_run(self, tmp_path, monkeypatch):
        import agents.llm_agent as L
        from agents.analyzer import Analyzer
        monkeypatch.setattr(L, "SimpleLLMClient", Scripted)
        Scripted.calls = []
        rd = write_run(tmp_path)
        st = Analyzer(str(rd), verbose=False).run(config={"analysis_with_images": False})
        assert st["steps"] == {"context": True, "compare": True, "investigate": True,
                               "interpret": True, "report": True}, st
        a = json.loads((rd / "04_analysis" / "analysis.json").read_text())
        assert a["status"] == "supported" and a["kind"] == "whole_domain"
        assert [c["finding_id"] for c in a["claims"]] == ["gpp_by_year"]
        assert [lab for lab, _ in Scripted.calls] == ["step2_investigate", "step3_interpret"]
        step3_brief = Scripted.calls[1][1]
        assert "id: compare_burned_area_km2" in step3_brief and "id: gpp_by_year" in step3_brief
        assert (rd / "04_analysis" / "REPORT.md").is_file()


# ── reading the numbers (2026-10-25, after the twin's "removes 23.2") ──────
class TestStocksAndTotals:

    def test_the_brief_says_which_fields_are_stocks(self, tmp_path):
        e = experiment_json()
        stock = {"stream": "h0", "how": "mean", "units": "gC/m^2", "note": None}
        for r in e["regional"]["runs"].values():
            r["fields"] = dict(r["fields"], TOTVEGC=stock)
            for reg in r["values"]:
                r["values"][reg]["TOTVEGC"] = {"2015": 1945.6, "2016": 1954.0, "2017": 1960.0}
        c = S0.load(write_run(tmp_path, experiment=e))
        lines = S3.domain_reading(c)
        assert "    stocks, averaged over the year: TOTVEGC" in lines
        assert "    totals over the year: FAREA_BURNED, GPP, burned_area_km2" in lines
        assert any("Never call it an amount" in x for x in lines)
        assert "HOW TO READ A NUMBER OF THIS EXPERIMENT:" in S3.review_brief(c, {}, {})

    def test_a_site_brief_has_no_such_block(self):
        class Site:
            is_domain, data, plan, caveats = False, {}, {}, []
        assert S3.domain_reading(Site()) == []

    def test_the_figure_tasks_keep_sentences_off_the_plot(self):
        for task in (S2.TASK, S2.DOMAIN_TASK):
            assert "no caveat, note or other sentence written on the plot" in task
        assert "never as an amount removed or added that year" in S2.DOMAIN_TASK
