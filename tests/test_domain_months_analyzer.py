"""The Analyzer on a whole-domain experiment's months and seasons.

    pytest tests/test_domain_months_analyzer.py -v

The run directory of test_domain_analyzer, its regional table given months
and seasons in the shape domain_analysis makes them (2026-11-03), so every
row a frame should hold is known. No LLM: steps 2 and 3 are given the
scripted client of test_domain_analyzer.
"""
import json

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
from agents.analysis import step0_context as S0          # noqa: E402
from agents.analysis import step2_investigate as S2      # noqa: E402
from agents.analysis import step3_interpret as S3        # noqa: E402
from agents.analysis import script_runner as SR          # noqa: E402
import test_domain_analyzer as T                         # noqa: E402

UNITS = {"GPP": {"month": "gC/m^2/month", "season": "gC/m^2/season"},
         "FAREA_BURNED": {"month": "proportion/month", "season": "proportion/season"}}


def _month(run, y, m):
    """A known value per run and month: the control's 10 x month, the
    no-fire run 1 less from June to August."""
    return 10.0 * m - (1.0 if run == "nofire" and m in (6, 7, 8) else 0.0)


def with_months(t):
    """The table of test_domain_analyzer with GPP's months and seasons, as
    the server makes them: DJF 2015 left out (no December 2014)."""
    for run, r in t["runs"].items():
        r["period_units"] = UNITS
        r["monthly"], r["seasonal"] = {}, {}
        for reg in T.REGIONS:
            r["monthly"][reg] = {"GPP": {f"{y}-{m:02d}": _month(run, y, m)
                                         for y in T.YEARS for m in range(1, 13)}}
            sea = {}
            for y in T.YEARS:
                if y > T.YEARS[0]:
                    sea[f"{y}-DJF"] = _month(run, y - 1, 12) + _month(run, y, 1) + \
                        _month(run, y, 2)
                for name, ms in (("MAM", (3, 4, 5)), ("JJA", (6, 7, 8)), ("SON", (9, 10, 11))):
                    sea[f"{y}-{name}"] = sum(_month(run, y, m) for m in ms)
            r["seasonal"][reg] = {"GPP": sea}
        r["seasons_left_out"] = {"2015-DJF": "DJF 2015 needs December 2014, which the "
                                             "run's h0 stream does not hold"}
    for kind in ("monthly", "seasonal"):
        c = t["runs"]["control"][kind]
        t[f"{kind}_differences"] = {"nofire": {
            reg: {"GPP": {p: x - c[reg]["GPP"][p] for p, x in
                          t["runs"]["nofire"][kind][reg]["GPP"].items()}}
            for reg in T.REGIONS}}
    return t


@pytest.fixture
def ctx(tmp_path):
    e = T.experiment_json(table=with_months(T.regional()))
    return S0.load(T.write_run(tmp_path, experiment=e))


class TestTheFrames:

    def test_the_monthly_frame(self, ctx):
        f = ctx.regional_monthly()
        assert list(f.columns) == ["run", "region", "variable", "month", "value", "units",
                                   "how", "minus_control"]
        assert len(f) == 2 * 2 * 36                    # runs x regions x months
        g = f[(f.run == "nofire") & (f.region == "vegetated cells") & (f.month == "2016-07")]
        (row,) = g.to_dict("records")
        assert row["value"] == 69.0 and row["minus_control"] == -1.0
        assert row["units"] == "gC/m^2/month" and row["how"] == "sum"
        assert f[f.run == "control"].minus_control.isna().all()

    def test_the_seasonal_frame(self, ctx):
        f = ctx.regional_seasonal()
        assert list(f.columns) == ["run", "region", "variable", "year", "season", "value",
                                   "units", "how", "minus_control"]
        c = f[(f.run == "control") & (f.region == "vegetated cells")]
        assert sorted(zip(c.year, c.season))[:4] == [(2015, "JJA"), (2015, "MAM"),
                                                     (2015, "SON"), (2016, "DJF")]
        assert not ((c.year == 2015) & (c.season == "DJF")).any()
        j = f[(f.run == "nofire") & (f.season == "JJA") & (f.year == 2017)]
        assert set(j.minus_control) == {-3.0}

    def test_a_table_without_months_has_no_such_frames(self, tmp_path):
        c = S0.load(T.write_run(tmp_path))
        assert c.regional_monthly() is None and c.regional_seasonal() is None
        assert c.regional() is not None
        assert "MONTHS AND SEASONS" not in S2.context_brief(c, {})


class TestTheFigureScripts:

    def test_reg_monthly_and_reg_seasonal_are_bound(self, ctx):
        p = SR._payload(ctx)
        assert len(p["reg_monthly"]) == 144 and len(p["reg_seasonal"]) == 2 * 2 * 11
        assert len(p["reg"]) == 42                     # the yearly frame is as it was

    def test_a_script_draws_the_months(self, ctx, tmp_path):
        code = ("g = reg_monthly[(reg_monthly.run == 'nofire') & "
                "(reg_monthly.region == 'vegetated cells') & "
                "(reg_monthly.month.str.startswith('2015'))].sort_values('month')\n"
                "s = reg_seasonal[(reg_seasonal.season == 'JJA') & (reg_seasonal.run == "
                "'nofire') & (reg_seasonal.region == 'vegetated cells')]\n"
                "fig, ax = plt.subplots()\nax.plot(range(12), g.minus_control)\n"
                "fig.savefig(out_path)\n"
                "result = {'n': int(len(g)), 'jja': [float(x) for x in s.minus_control]}\n")
        out = SR.run(code, ctx, tmp_path / "m.png", script_path=tmp_path / "m.py")
        assert out["ok"], out.get("error")
        assert out["result"] == {"n": 12, "jja": [-3.0, -3.0, -3.0]}


class TestTheBriefs:

    def test_step_2_says_what_the_frames_hold(self, ctx):
        b = S2.context_brief(ctx, {})
        assert ("MONTHS AND SEASONS IN `reg_monthly` AND `reg_seasonal` (name, units over a "
                "month, units over a season):") in b
        assert "    GPP              gC/m^2/month     gC/m^2/season" in b
        assert "    Not made: 2015-DJF (the run's output does not hold the December " \
               "before)." in b
        for text in ("reg_monthly   tidy frame", "reg_seasonal  tidy frame",
                     "answered from `reg_monthly` or `reg_seasonal`, never from `reg`"):
            assert text in S2.DOMAIN_TASK, text

    def test_step_3_reads_a_month_as_it_reads_a_year(self, ctx):
        lines = S3.domain_reading(ctx)
        assert "    over a month or a season, the same rule: a stock's value is its mean" \
            in lines
        assert "        from the year's value." in lines


MONTH_FIGURE = {
    "id": "gpp_summer_months", "question": "How did the summer months' GPP differ?",
    "why": "the months the change applies in", "scale": "overall", "variables": ["GPP"],
    "code": ("g = reg_monthly[(reg_monthly.run == 'nofire') & (reg_monthly.region == "
             "'vegetated cells')].sort_values('month')\n"
             "fig, ax = plt.subplots()\nax.plot(range(len(g)), g.minus_control)\n"
             "ax.set_xlabel('month', fontsize=16)\nfig.savefig(out_path)\n"
             "result = {'n': int(len(g)), 'july_2015': float(g[g.month == '2015-07']"
             ".minus_control.iloc[0])}\n")}
MONTH_CLAIM = {"claim": "In July 2015 the run's GPP over the vegetated cells was 1 gC/m^2 "
                        "below the control's.",
               "finding_id": "gpp_summer_months", "values": [-1.0], "caveats": []}


class TestTheAnalyzer:

    def test_a_monthly_figure_is_proposed_run_and_audited(self, ctx, tmp_path):
        inv = S2.investigate(ctx, tmp_path, client=T.Scripted(figures=(MONTH_FIGURE,)))
        assert inv["n_succeeded"] == 1
        (f,) = inv["findings"]
        assert f["n"] == 36 and f["result"]["july_2015"] == -1.0
        a = S3.audit([MONTH_CLAIM], inv, ctx.caveats, facts=S3.run_facts(ctx))
        assert a["n_struck"] == 0, a["struck"]

    def test_steps_0_to_4(self, tmp_path, monkeypatch):
        import agents.llm_agent as L
        from agents.analyzer import Analyzer

        class Months(T.Scripted):
            def __init__(self, model=None):
                super().__init__(model, figures=(MONTH_FIGURE,),
                                 review=dict(T.REVIEW, claims=[MONTH_CLAIM],
                                             headline=MONTH_CLAIM["claim"]))
        monkeypatch.setattr(L, "SimpleLLMClient", Months)
        T.Scripted.calls = []
        rd = T.write_run(tmp_path, experiment=T.experiment_json(
            table=with_months(T.regional())))
        st = Analyzer(str(rd), verbose=False).run(config={"analysis_with_images": False})
        assert st["steps"] == {"context": True, "compare": True, "investigate": True,
                               "interpret": True, "report": True}, st
        a = json.loads((rd / "04_analysis" / "analysis.json").read_text())
        assert [c["finding_id"] for c in a["claims"]] == ["gpp_summer_months"]
        step2 = T.Scripted.calls[0][1]
        assert "MONTHS AND SEASONS IN `reg_monthly`" in step2


class TestThePrompts:
    """A prompt that promises less than the code does is as wrong as one that
    promises more (2026-11-03): both front prompts say months and seasons are
    reported, and none says the answers are yearly only."""

    def test_reception_and_the_planner_say_months_and_seasons(self):
        from pathlib import Path
        root = Path(S2.__file__).resolve().parents[1] / "prompts"
        for name in ("planner.txt", "reception_agentic.txt"):
            text = " ".join((root / name).read_text().split())
            assert "reported by year, by month and by season" in text, name
            assert "DJF belongs to the year of its January" in text or \
                "belongs to the year of its January" in text, name
            assert "A question about a season or a month is answered from" in text, name
            for wrong in ("calendar years only", "yearly only", "annual values only"):
                assert wrong not in text.lower(), (name, wrong)
