#!/usr/bin/env python3
"""Analyzer step 2 — the deterministic half.

The LLM layer needs an API call, so what is pinned here is everything around
it: the brief it reads, and the runner that decides whether what it wrote
counts. Both are pure functions of the context, which is the point — most of
step 2 is testable offline against an archived run.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.analysis import script_runner as runner        # noqa: E402
from agents.analysis import step2_investigate as step2     # noqa: E402


class _Ctx:
    def __init__(self, cols=None, units=None, caveats=(), plan=None, sem=None,
                 prof=None):
        # `prof` mirrors AnalysisContext.profiles(): None for a backend with no
        # depth axis (ELM), a frame for one that has it (PFLOTRAN). Defaulting
        # to None keeps every existing case an ELM-shaped run.
        self._prof = prof
        self._c = cols if cols is not None else [
            {"case_name": "col_01", "elevation_m": 2400.0, "band": 1,
             "soil": None, "soil_profile": {"layers": []}},
            {"case_name": "col_02", "elevation_m": 3400.0, "band": 2,
             "soil": None, "soil_profile": {"layers": []}}]
        self.plan = plan or {"question": "How does recharge behave?",
                             "goals": ["quantify recharge"],
                             "period": {"yr_start": 2019, "yr_end": 2019}}
        self.caveats = list(caveats)
        self.data = {"variable_units": units if units is not None
                     else {"QOVER": "mm/s", "H2OSNO": "mm"},
                     "field_semantics": sem or {}}

    @property
    def columns(self):
        return self._c

    def series(self):
        """Mirrors the real frame, INCLUDING the mismatch that caused the bug.

        `variable_units` above says QOVER is mm/s — the raw ELM variable. The
        frame says mm/day, because _daily() converted and labelled it. Both are
        right about their own subject, and a fixture that made them agree would
        test nothing.
        """
        pd = pytest.importorskip("pandas")
        return pd.DataFrame({
            "date": ["2019-01-01", "2019-01-02"] * 2,
            "entity": ["col_01"] * 2 + ["col_02"] * 2,
            "variable": ["QOVER"] * 2 + ["H2OSNO"] * 2,
            "value": [0.864, 1.728, 100.0, 110.0],
            "units": ["mm/day"] * 2 + ["mm"] * 2,
            "source": ["model"] * 4})

    def profiles(self):
        return self._prof


class TestTheBriefQuotesTheFrameNotTheRawMap:
    """The data was never wrong. The BRIEF was.

    experiment.json carries two units records, and both are correct about their
    own subject: data["variable_units"] describes the RAW ELM variable (QOVER
    in mm/s), while variables[v]["daily"]["units"] describes the series that
    _daily() actually produced (mm/day). step0.series() already carries the
    second into the frame.

    Step 2's brief printed the first under a heading about `df`. The model
    believed the brief over the frame, multiplied a year of already-per-day
    values by 86400, and plotted 1.1e6 mm/yr of runoff. I then read that as a
    missing conversion and added one in the runner — which doubled the error,
    and is what finally located the real fault.
    """

    def test_the_runner_never_rescales_the_frame(self):
        """An earlier version multiplied fluxes by 86400 on the strength of
        data["variable_units"] — turning correct per-day values into 1e6 mm/yr.
        The frame is authoritative about itself; the runner passes it through.
        """
        pytest.importorskip("pandas")
        payload = runner._payload(_Ctx())
        q = payload["df"][payload["df"]["variable"] == "QOVER"]
        assert list(q["value"]) == [0.864, 1.728], "values must pass through"
        assert set(q["units"]) == {"mm/day"}

    def test_non_flux_variables_are_untouched(self):
        """H2OSNO is a storage in mm, not a rate — scaling it would be wrong."""
        pytest.importorskip("pandas")
        payload = runner._payload(_Ctx())
        s = payload["df"][payload["df"]["variable"] == "H2OSNO"]
        assert list(s["value"]) == [100.0, 110.0]
        assert set(s["units"]) == {"mm"}

    def test_the_brief_reports_the_frame_units_not_the_raw_map(self):
        """The whole bug in one assertion.

        ctx.data["variable_units"] says mm/s (the RAW ELM variable); the frame
        says mm/day (what _daily() actually produced and labelled). The brief
        must print the frame's, or the model rescales correct values.
        """
        ctx = _Ctx()
        assert ctx.data["variable_units"]["QOVER"] == "mm/s"
        assert step2._frame_units(ctx)["QOVER"] == "mm/day"
        brief = step2.context_brief(ctx)
        assert "QOVER      mm/day" in brief
        assert "86400" not in brief, "the brief must not ask for a rescale"


class TestTheRunnerRefusesRatherThanShrugs:
    """soil_attribution returned {} for the study's entire history and the
    figure simply did not appear. A generated script reintroduces that failure
    fresh every run, so an empty result has to be a stated refusal."""

    def _run(self, code, tmp_path):
        return runner.run(code, _Ctx(), tmp_path / "f.png")

    def test_a_result_below_the_floor_is_refused(self, tmp_path):
        pytest.importorskip("pandas")
        r = self._run('fig, ax = plt.subplots(); fig.savefig(out_path)\n'
                      'result = {"n": 0}', tmp_path)
        assert r["ok"] is False
        assert "below the floor" in r["error"]

    def test_a_result_without_n_is_refused(self, tmp_path):
        """n is how many data points the claim rests on. Without it there is
        nothing to check the claim's weight against."""
        pytest.importorskip("pandas")
        r = self._run('fig, ax = plt.subplots(); fig.savefig(out_path)\n'
                      'result = {"slope": 1.0}', tmp_path)
        assert r["ok"] is False and "no numeric `n`" in r["error"]

    def test_a_script_that_never_assigns_result_is_refused(self, tmp_path):
        pytest.importorskip("pandas")
        r = self._run('fig, ax = plt.subplots(); fig.savefig(out_path)', tmp_path)
        assert r["ok"] is False

    def test_a_raising_script_returns_a_reason_not_an_exception(self, tmp_path):
        """One bad script must not lose the four good ones alongside it."""
        pytest.importorskip("pandas")
        r = self._run("result = 1/0", tmp_path)
        assert r["ok"] is False and "ZeroDivisionError" in r["error"]

    def test_a_result_with_no_figure_is_refused(self, tmp_path):
        pytest.importorskip("pandas")
        r = self._run('result = {"n": 19}', tmp_path)
        assert r["ok"] is False and "no figure" in r["error"]

    def test_a_good_script_passes_with_its_numbers(self, tmp_path):
        pytest.importorskip("pandas")
        r = self._run('fig, ax = plt.subplots(figsize=(6,4))\n'
                      'ax.plot([1,2,3],[1,2,3]); fig.savefig(out_path, dpi=120)\n'
                      'result = {"n": 19, "slope": 1.0}', tmp_path)
        assert r["ok"] is True
        assert r["result"]["slope"] == 1.0
        assert Path(r["figure"]).exists()

    def test_the_script_is_saved_for_provenance(self, tmp_path):
        """The saved script IS the provenance for every number it produced —
        a reviewer can open it and re-run it."""
        pytest.importorskip("pandas")
        sp = tmp_path / "scripts" / "f.py"
        r = runner.run('fig, ax = plt.subplots(); ax.plot([1,2])\n'
                       'fig.savefig(out_path, dpi=120)\nresult = {"n": 5}',
                       _Ctx(), tmp_path / "f.png", script_path=sp)
        assert r["ok"] and sp.exists()
        assert "result = {\"n\": 5}" in sp.read_text()


class TestTheBriefIsGenericAndBounded:
    """Nothing here may name a basin, a variable set, or a station network —
    the framework is meant for any US watershed."""

    def test_it_carries_the_users_own_question(self):
        assert "How does recharge behave?" in step2.context_brief(_Ctx())

    def test_it_forbids_a_basin_total(self):
        """Nineteen unrouted 1-D columns do not sum to a watershed, and that is
        the sentence a model writes unless the brief forbids it."""
        b = step2.context_brief(_Ctx())
        assert "OUT OF SCOPE" in b and "routing" in b

    def test_blocking_caveats_are_quoted_with_their_scope(self):
        b = step2.context_brief(_Ctx(caveats=[
            {"id": "no_routing", "severity": "blocking",
             "statement": "columns are unrouted",
             "applies_to": "any discharge claim"}]))
        assert "BLOCKING CAVEATS" in b
        assert "no_routing" in b and "any discharge claim" in b

    def test_a_field_null_on_every_column_is_not_advertised(self):
        """`soil` is None on all 19 rows and sits beside `soil_profile` — it is
        the exact field the old soil_attribution read before returning {}."""
        b = step2.context_brief(_Ctx())
        keys = b.split("PER-COLUMN METADATA")[1].split("\n")[1]
        assert "soil_profile" in keys
        assert "soil," not in keys and not keys.strip().endswith("soil")

    def test_derived_metrics_are_shown_with_what_they_come_from(self):
        """runoff_fraction is QOVER/(QCHARGE+QOVER), not a fraction of P — I got
        that wrong myself by reading the name instead of the derivation."""
        b = step2.context_brief(_Ctx(sem={
            "runoff_fraction": {"units": "1", "from": ["QOVER", "QCHARGE"]}}))
        assert "runoff_fraction" in b and "from QOVER, QCHARGE" in b


class TestTheCeilingIsNotAQuota:

    def test_more_than_five_figures_are_cut(self):
        class _Client:
            def ask(self, messages, system_message=None):
                figs = [{"id": f"f{i}", "question": "q", "code": "pass"}
                        for i in range(9)]
                import json
                return json.dumps({"notes": "n", "figures": figs})
        spec = step2.propose(_Ctx(), client=_Client())
        assert len(spec["figures"]) == step2.MAX_PLOTS

    def test_duplicate_ids_are_dropped_not_overwritten(self):
        """Two figures with one id would write to the same PNG path, and the
        second would silently replace the first."""
        class _Client:
            def ask(self, messages, system_message=None):
                import json
                return json.dumps({"notes": "n", "figures": [
                    {"id": "same", "question": "a", "code": "pass"},
                    {"id": "same", "question": "b", "code": "pass"}]})
        spec = step2.propose(_Ctx(), client=_Client())
        assert len(spec["figures"]) == 1

    def test_a_fenced_json_reply_still_parses(self):
        class _Client:
            def ask(self, messages, system_message=None):
                return '```json\n{"notes": "n", "figures": []}\n```'
        assert step2.propose(_Ctx(), client=_Client())["figures"] == []


class TestPathsSurviveTheSubprocessCwd:
    """The script runs with cwd set to a temp dir, so a relative out_path or
    script_path stops resolving the moment it starts. Every figure in the first
    real Analyzer run failed with "can't open file" — and every test before it
    passed only because it handed in absolute scratchpad paths. The entry point
    that matters passes a relative run_dir."""

    def test_a_relative_out_path_still_produces_a_figure(self, tmp_path, monkeypatch):
        pytest.importorskip("pandas")
        monkeypatch.chdir(tmp_path)
        (tmp_path / "out").mkdir()
        r = runner.run('fig, ax = plt.subplots(figsize=(5,4))\n'
                       'ax.plot([1,2,3]); fig.savefig(out_path, dpi=120)\n'
                       'result = {"n": 9}',
                       _Ctx(), "out/f.png", script_path="out/f.py")
        assert r["ok"] is True, r["error"]
        assert (tmp_path / "out" / "f.png").exists()
        assert (tmp_path / "out" / "f.py").exists()


# ═════════════════════════════════════════════════════════════════════
# PREFLIGHT — what the run holds, decided before anything is spent
# ═════════════════════════════════════════════════════════════════════
class _PreCtx:
    """A context with one usable variable and one that was withheld."""
    plan = {"question": "q", "feasibility": {"verdict": "partial",
                                             "why": "no routing"}}
    caveats: list = []
    data = {"variable_units": {"QOVER": "mm/s"}, "field_semantics": {}}
    columns = [{"case_name": "col_01"}]
    series_withheld = {"SOILLIQ": {"why": "5280 values against 352 dates"}}

    def __init__(self, blocked=False):
        self._blocked = blocked

    # The brief describes whichever frames exist; this stub has none, which is
    # a legitimate state and the one the "df is None" branch is written for.
    def series(self):
        return None

    def soil(self):
        return None

    def profiles(self):
        return None

    def planned_vs_actual(self):
        return []

    def preflight(self):
        if self._blocked:
            return {"frames": {}, "variables": [], "withheld": {},
                    "feasibility": None, "unmet_plan_targets": [],
                    "n_columns": 0, "blocked": "no frames"}
        return {"frames": {"series": {"rows": 10, "variables": ["QOVER"]}},
                "variables": ["QOVER"], "withheld": self.series_withheld,
                "feasibility": self.plan["feasibility"],
                "unmet_plan_targets": [], "n_columns": 1, "blocked": None}


class TestAFigureIsRefusedBeforeItCostsASubprocess:
    """A figure naming only variables no frame holds cannot draw anything.
    Running it anyway spends a subprocess to return a KeyError from somewhere
    inside generated pandas — which is then what round 2 is told."""

    def test_a_figure_whose_variables_are_all_absent_is_not_run(self, tmp_path,
                                                               monkeypatch):
        ran = []
        monkeypatch.setattr(step2._runner, "run",
                            lambda *a, **k: ran.append(1) or {"ok": False,
                                                              "error": "x"})
        monkeypatch.setattr(step2, "propose", lambda *a, **k: {
            "notes": None, "reasoning": None,
            "figures": [{"id": "f1", "question": "?", "variables": ["SOILLIQ"],
                         "code": "pass"}]})
        out = step2.investigate(_PreCtx(), tmp_path)
        assert ran == [], "the script must not have been executed"
        assert out["n_succeeded"] == 0
        assert len(out["caveats"]) == 1

    def test_the_reason_names_the_variable_and_why_it_is_absent(self, tmp_path,
                                                               monkeypatch):
        monkeypatch.setattr(step2, "propose", lambda *a, **k: {
            "notes": None, "reasoning": None,
            "figures": [{"id": "f1", "question": "?", "variables": ["SOILLIQ"],
                         "code": "pass"}]})
        c = step2.investigate(_PreCtx(), tmp_path)["caveats"][0]
        assert "SOILLIQ" in c["statement"]
        assert "352 dates" in c["statement"], "say WHY, not just that it is gone"
        assert "QOVER" in c["statement"], "say what IS available"

    def test_a_figure_missing_only_some_variables_still_runs(self, tmp_path,
                                                            monkeypatch):
        """Five variables with one absent may still be worth drawing; the
        filter only refuses a figure that can draw nothing at all."""
        ran = []
        monkeypatch.setattr(step2._runner, "run",
                            lambda *a, **k: (ran.append(1),
                                             {"ok": False, "error": "x"})[1])
        monkeypatch.setattr(step2, "propose", lambda *a, **k: {
            "notes": None, "reasoning": None,
            "figures": [{"id": "f1", "question": "?",
                         "variables": ["QOVER", "SOILLIQ"], "code": "pass"}]})
        step2.investigate(_PreCtx(), tmp_path)
        assert ran == [1], "it names one variable that exists, so it runs"

    def test_a_figure_naming_no_variables_still_runs(self, tmp_path, monkeypatch):
        ran = []
        monkeypatch.setattr(step2._runner, "run",
                            lambda *a, **k: (ran.append(1),
                                             {"ok": False, "error": "x"})[1])
        monkeypatch.setattr(step2, "propose", lambda *a, **k: {
            "notes": None, "reasoning": None,
            "figures": [{"id": "f1", "question": "?", "code": "pass"}]})
        step2.investigate(_PreCtx(), tmp_path)
        assert ran == [1]


class TestTheBriefCarriesWhatWasAlreadyKnown:
    """Two signals were computed and read by nobody: the Planner's feasibility
    verdict, carried in ctx.plan since step 0 was written, and
    planned_vs_actual(), which had tests and no production caller."""

    def test_the_planners_feasibility_verdict_reaches_the_model(self):
        b = step2.context_brief(_PreCtx())
        assert "PARTIAL" in b and "no routing" in b

    def test_it_is_marked_as_design_not_result(self):
        b = step2.context_brief(_PreCtx())
        assert "DESIGN" in b, "a feasibility verdict must not read as a finding"


class TestPreflightFailsOpen:
    """It is a diagnostic and a cost saving, not a safety check. A
    context-shaped object without it must behave exactly as before."""

    def test_a_context_without_preflight_returns_empty(self):
        class Bare:
            pass
        assert step2.preflight_of(Bare()) == {}

    def test_a_preflight_that_raises_is_not_fatal(self):
        class Boom:
            def preflight(self):
                raise RuntimeError("nope")
        assert step2.preflight_of(Boom()) == {}


# ═════════════════════════════════════════════════════════════════════
# THE RUNNER IS CONSTRAINED, NOT SANDBOXED
# ═════════════════════════════════════════════════════════════════════
# It was called "sandboxed" until 2026-08-14 and was not one: the script ran as
# the invoking user with the whole environment inherited, which on this
# deployment includes a USGS API key, AmeriFlux credentials and a HydroFrame
# PIN. What is enforced now is written down in the module docstring and pinned
# below. The threat model is CARELESSNESS, not malice.

class _RunCtx:
    """Enough context to execute a real script — and, deliberately, NO
    preflight() method. That is the state a context-shaped stub, a standalone
    CLI or a future backend is in, and the filter must fail open for it."""
    from pathlib import Path as _P
    run_dir = _P(".").resolve()
    caveats: list = []
    plan = {"question": "how does water partition?"}
    data = {"variable_units": {}, "field_semantics": {}}
    columns = [{"case_name": "c1", "variables": {}}]
    series_withheld: dict = {}

    def planned_vs_actual(self):
        return []

    def series(self):
        pd = pytest.importorskip("pandas")
        return pd.DataFrame({"date": pd.to_datetime(["2010-01-01"] * 4),
                             "entity": ["c1"] * 4, "variable": ["QOVER"] * 4,
                             "value": [1.0, 2.0, 3.0, 4.0],
                             "units": ["mm/day"] * 4, "source": ["model"] * 4})

    def soil(self):
        return None

    def profiles(self):
        return None


_GOOD = """
fig, ax = plt.subplots()
ax.plot(df["value"].values)
fig.savefig(out_path, dpi=110)
result = {"n": len(df), "mean": float(df["value"].mean())}
"""


class TestTheScopeGuardrail:
    """AST-parsed, not regex-matched: `import subprocess` inside a string
    literal is not an import and a regex cannot tell."""

    @pytest.mark.parametrize("code,token", [
        ("import subprocess\nresult={'n':3}", "subprocess"),
        ("from urllib.request import urlopen\nresult={'n':3}", "urllib"),
        ("import socket\nresult={'n':3}", "socket"),
        ("import os\nos.system('id')\nresult={'n':3}", "os.system"),
        ("eval('1+1')\nresult={'n':3}", "eval"),
        ("import requests\nresult={'n':3}", "requests"),
    ])
    def test_out_of_scope_code_is_refused(self, code, token):
        objections = step2._runner.inspect_code(code)
        assert objections and any(token in o for o in objections)

    def test_ordinary_plotting_code_passes(self):
        assert step2._runner.inspect_code(_GOOD) == []

    def test_os_path_is_not_forbidden(self):
        """`os` is allowed; only some of its members are. os.path.join in
        plotting code is ordinary and refusing it would be noise."""
        assert step2._runner.inspect_code(
            "import os\np = os.path.join('a', 'b')\nresult={'n':3}") == []

    def test_a_mention_in_a_string_is_not_an_import(self):
        assert step2._runner.inspect_code(
            "s = 'import subprocess'\nresult={'n':3}") == []

    def test_unparseable_code_is_one_clear_objection(self):
        objections = step2._runner.inspect_code("def (:\n")
        assert len(objections) == 1 and "does not parse" in objections[0]

    def test_a_refused_script_is_never_executed(self, tmp_path):
        r = step2._runner.run("import subprocess\nresult={'n':5}",
                              _RunCtx(), tmp_path / "x.png")
        assert r["ok"] is False
        assert "refused before running" in r["error"]
        assert not (tmp_path / "x.png").exists()


class TestCredentialsDoNotReachTheChild:
    """env_compy.sh exports a USGS API key, AmeriFlux credentials and a
    HydroFrame PIN before anything runs. A figure script reads its data from a
    pickle and has no business seeing any of them."""

    def test_the_environment_is_an_allowlist(self, monkeypatch):
        monkeypatch.setenv("USGS_API_KEY", "secret")
        monkeypatch.setenv("HYDRODATA_PIN", "1234")
        monkeypatch.setenv("SOME_FUTURE_TOKEN", "also-secret")
        env = step2._runner._child_env()
        assert "USGS_API_KEY" not in env
        assert "HYDRODATA_PIN" not in env
        assert "SOME_FUTURE_TOKEN" not in env, \
            "deny by default — a denylist only blocks the names someone recalled"

    def test_what_the_stack_needs_survives(self, monkeypatch):
        monkeypatch.setenv("CONDA_PREFIX", "/opt/conda")
        env = step2._runner._child_env()
        assert env.get("PATH") and env.get("MPLBACKEND") == "Agg"
        assert env.get("CONDA_PREFIX") == "/opt/conda"


class TestTheResourceCeilings:
    """Each turns a runaway into a clean non-zero exit the caller reports as a
    caveat, rather than a node the Analyzer is sharing."""

    def test_the_limits_are_applied_in_the_child(self, tmp_path):
        r = step2._runner.run(
            "import resource\n"
            "a = resource.getrlimit(resource.RLIMIT_AS)[0]\n"
            "c = resource.getrlimit(resource.RLIMIT_CPU)[0]\n"
            "f = resource.getrlimit(resource.RLIMIT_FSIZE)[0]\n"
            "fig, ax = plt.subplots(); ax.plot([1,2,3]); fig.savefig(out_path)\n"
            "result = {'n': 3, 'a': a, 'c': c, 'f': f}",
            _RunCtx(), tmp_path / "r.png")
        assert r["ok"], r["error"]
        assert r["result"]["a"] == step2._runner.MAX_MEMORY_BYTES
        assert r["result"]["c"] == step2._runner.MAX_CPU_SECONDS
        assert r["result"]["f"] == step2._runner.MAX_FILE_BYTES

    def test_a_memory_runaway_is_a_reason_not_a_dead_node(self, tmp_path):
        r = step2._runner.run("x = bytearray(20 * 1024**3)\nresult={'n':3}",
                              _RunCtx(), tmp_path / "m.png")
        assert r["ok"] is False and "MemoryError" in r["error"]


class TestASavedScriptCanBeReRun:
    """The point of keeping the script is that a reviewer can execute it. Until
    2026-08-14 the preamble loaded a pickle from a temporary directory this
    module deletes in its `finally`, so every saved script died on a missing
    file the moment its round ended."""

    def _run_dir(self, tmp_path):
        """A minimal packaged run the preamble's fallback can actually load."""
        rd = tmp_path / "run"
        rd.mkdir()
        (rd / "experiment.json").write_text(json.dumps({
            "model": "elm", "columns_total": 1, "columns_succeeded": 1,
            "columns": [{"case_name": "c1", "status": "ok", "metrics": {},
                         "variables": {"QOVER": {"daily": {
                             "units": "mm/day",
                             "dates": ["2010-01-01", "2010-01-02",
                                       "2010-01-03", "2010-01-04"],
                             "values": [1.0, 2.0, 3.0, 4.0]}}}}]}))
        return rd

    def test_the_script_runs_again_after_its_payload_is_gone(self, tmp_path):
        import subprocess as _sp

        class C(_RunCtx):
            pass
        C.run_dir = self._run_dir(tmp_path)
        r = step2._runner.run(_GOOD, C(), tmp_path / "g.png",
                              script_path=tmp_path / "g.py")
        assert r["ok"], r["error"]
        assert not Path(r["script_path"]).parent.joinpath("ctx.pkl").exists()
        p = _sp.run([sys.executable, str(tmp_path / "g.py")],
                    capture_output=True, text=True, cwd=str(tmp_path))
        assert p.returncode == 0, p.stderr[-400:]

    def test_the_script_is_identified_by_hash(self, tmp_path):
        """Two runs of the same study can be compared on whether the analysis
        was the same analysis, which "same figure name" does not answer."""
        a = step2._runner.run(_GOOD, _RunCtx(), tmp_path / "a.png")
        b = step2._runner.run(_GOOD + "\n", _RunCtx(), tmp_path / "b.png")
        c = step2._runner.run(_GOOD.replace("110", "120"), _RunCtx(),
                              tmp_path / "c.png")
        assert a["script_sha256"] == b["script_sha256"]
        assert a["script_sha256"] != c["script_sha256"]


class TestStep2KeepsWhatTheModelSaid:
    """investigation.json records what the reply BECAME. These record what it
    WAS — the half that lets a mangled proposal leave a trace, and the half a
    replayed end-to-end test needs."""

    def _investigate(self, ctx, out_dir, reply):
        class FakeClient:
            label = None
            def ask(self, messages):
                return reply
        return step2.investigate(ctx, out_dir, step1={},
                                 client=FakeClient(), round_no=1)

    _REPLY = json.dumps({
        "reasoning": "mapped partitioning onto QOVER",
        "notes": "one figure only",
        "figures": [{"id": "f1", "question": "how?", "why": "elevation",
                     "scale": "overall", "variables": ["QOVER"],
                     "code": "fig, ax = plt.subplots()\n"
                             "ax.plot(df['value'].values)\n"
                             "fig.savefig(out_path)\n"
                             "result = {'n': 4}"}]})

    def test_the_reply_lands_where_the_name_says(self, tmp_path):
        out = self._investigate(_RunCtx(), tmp_path, self._REPLY)
        p = Path(out["exchange"]["reply"])
        assert p.name == "step2_round1_reply.txt"
        assert p.read_text() == self._REPLY

    def test_a_fenced_reply_is_stored_with_its_fence(self, tmp_path):
        """Stored before parsing, so a repair that goes wrong still has an
        original to be compared against."""
        fenced = "```json\n" + self._REPLY + "\n```"
        out = self._investigate(_RunCtx(), tmp_path, fenced)
        assert Path(out["exchange"]["reply"]).read_text() == fenced
        assert out["n_succeeded"] == 1, "and it still parsed"

    def test_the_prompt_is_saved_beside_it(self, tmp_path):
        out = self._investigate(_RunCtx(), tmp_path, self._REPLY)
        body = Path(out["exchange"]["prompt"]).read_text()
        assert "variables in `df`" in body

    def test_the_paths_are_recorded_in_the_json_record(self, tmp_path):
        """A file nothing points at is a file nobody finds."""
        out = self._investigate(_RunCtx(), tmp_path, self._REPLY)
        assert set(out["exchange"]) == {"prompt", "reply"}
        for p in out["exchange"].values():
            assert Path(p).is_file()

    def test_round_two_does_not_overwrite_round_one(self, tmp_path):
        """investigation.json is overwritten per round; these are not, because
        the question they answer is what CHANGED between the two."""
        class FakeClient:
            label = None
            def ask(self, messages):
                return TestStep2KeepsWhatTheModelSaid._REPLY
        step2.investigate(_RunCtx(), tmp_path, step1={},
                          client=FakeClient(), round_no=1)
        step2.investigate(_RunCtx(), tmp_path, step1={},
                          client=FakeClient(), round_no=2, feedback="try again")
        assert (tmp_path / "step2_round1_reply.txt").is_file()
        assert (tmp_path / "step2_round2_reply.txt").is_file()
        assert "try again" in (tmp_path / "step2_round2_prompt.txt").read_text()
        assert "try again" not in (tmp_path / "step2_round1_prompt.txt").read_text()

    def test_the_filter_fails_open_without_a_variable_inventory(self, tmp_path):
        """`set(x or [])` collapses "no inventory" and "empty inventory" into
        the same empty set, and then every figure naming any variable is
        refused — the opposite of the intent. A context with no preflight() must
        behave exactly as it did before the filter existed."""
        out = self._investigate(_RunCtx(), tmp_path, self._REPLY)
        assert out["n_succeeded"] == 1, \
            "_RunCtx has no preflight(); the filter must not judge"
        assert out["caveats"] == []
