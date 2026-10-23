"""A whole-domain case from its recipe, and the chain of gated jobs that runs it.

    pytest tests/test_domain_case.py -v

The generated scripts are RUN here, in a sandbox: CIME's commands are stubs
that record what they were asked, and the gate is the real run_gate.py on a
synthetic case, so a stage's exit code is the gate's verdict as it will be in
a job. SLURM is a fake that records every call and answers as Compy does; an
autouse guard fails any test that would reach the real scheduler.
"""
import datetime
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import domain_case as DC
import elm_case_fixtures as cf
import elm_gridded_fixtures as fx

STUB = "#!/bin/bash\n{body}\n"
REAL_NODE_LOCAL = DC.node_local            # before the guard below replaces it


@pytest.fixture(autouse=True)
def _no_real_scheduler(monkeypatch):
    """No test reaches sbatch, scancel, squeue or sacct unless it hands the
    code a fake, and a record under tmp_path (node-local /tmp) is allowed."""
    def refuse(argv, timeout=120):
        raise AssertionError(f"a test reached the real scheduler: {argv}")
    monkeypatch.setattr(DC, "_run", refuse)
    monkeypatch.setattr(DC, "node_local", lambda path: None)


def _stub(path, body):
    path.write_text(STUB.format(body=body))
    path.chmod(0o755)
    return path


def _recipe(case_root, case="synth", srcroot="/src/E3SM", **kw):
    base = dict(case=case, case_root=str(case_root), srcroot=str(srcroot),
                compset="2000_DATM%QIA_ELM%CNPRDCTCBC_SICE_SOCN_SROF_SGLC_SWAV",
                xml=[("LND_DOMAIN_PATH", "/in"), ("LND_DOMAIN_FILE", "domain.nc"),
                     ("NTASKS", "40"), ("RUN_TYPE", "startup"), ("RUN_STARTDATE", "2015-01-01")],
                user_nl={"elm": ["finidat = '/in/finidat.nc'", "fsurdat = '/in/surf.nc'",
                                 "! a comment $HOME stays literal",
                                 "hist_fincl1 = 'GPP','TSA'", "hist_nhtfrq = 0, -24"]},
                start="2015-01-01")
    base.update(kw)
    return DC.Recipe(**base)


class FakeSlurm:
    """sbatch, scancel, squeue and sacct as Compy answers them. states[id]:
    an active state (squeue lists it), a finished one (squeue calls the id
    invalid, sacct has it), GONE (cancelled while it waited: squeue calls the
    id invalid and sacct has no row) or, by default, NOANSWER (neither
    answers)."""

    def __init__(self, fail_at=None, states=None, scancel_rc=0, start=1000):
        self.calls, self.next, self.fail_at = [], start, fail_at
        self.states, self.scancel_rc = states or {}, scancel_rc

    def __call__(self, argv, timeout=120):
        self.calls.append(list(argv))
        if argv[0] == "sbatch":
            if self.fail_at and self.fail_at in argv:
                return 1, "", "sbatch: error: Batch job submission failed"
            self.next += 1
            return 0, f"{self.next}\n", ""
        if argv[0] == "scancel":
            return (0, "", "") if self.scancel_rc == 0 else \
                (self.scancel_rc, "", "scancel: error: Socket timed out")
        st = self.states.get(argv[argv.index("-j") + 1], "NOANSWER")
        if st == "NOANSWER":
            return 1, "", "error: Socket timed out on send/recv operation"
        if argv[0] == "squeue":
            return (0, st + "\n", "") if st in DC.ACTIVE else \
                (1, "", "slurm_load_jobs error: Invalid job id specified")
        return (0, "", "") if st == "GONE" or st in DC.ACTIVE else (0, f" {st} \n", "")


# ── the recipe ───────────────────────────────────────────────────────────────
class TestRecipe:

    def test_round_trip_and_what_it_names(self, tmp_path):
        r = _recipe(tmp_path, xml=[("JOB_WALLCLOCK_TIME", "00:20:00", {"subgroup": "case.run"}),
                                   ("RUN_STARTDATE", "2015-01-01")])
        assert DC.Recipe.from_json(r.to_json()) == r
        r = _recipe(tmp_path)
        assert r.case_dir == tmp_path / "synth" and r.first_year == 2015
        assert r.inputs() == {"surface": "/in/surf.nc", "domain": "/in/domain.nc",
                              "finidat": "/in/finidat.nc"}
        assert r.namelist()["hist_nhtfrq"] == [0, -24]
        assert r.check() == []

    @pytest.mark.parametrize("change,problem", [
        (dict(case="my case"), "is not a plain file name"),
        (dict(case=""), "no case"), (dict(srcroot=""), "no srcroot"),
        (dict(compset=""), "no compset"), (dict(mach=""), "no mach"),
        (dict(srcroot="E3SM"), "is not an absolute path"),
        (dict(start="2015-1-1"), "is not YYYY-MM-DD"),
        (dict(start="2015-07-01", xml=[]), "is not 1 January"),
        (dict(xml=[("STOP_N", "1")]), "STOP_N is run control"),
        (dict(xml=[("EXEROOT", "/old/case/bld")]), "EXEROOT names a case's own folder"),
        (dict(xml=[("RUNDIR", "/old/run")]), "RUNDIR names a case's own folder"),
        (dict(xml=[("ntasks", "4")]), "XML setting name"),
        (dict(xml=[("NTASKS", "4", {"file": "env_run.xml"})]), "xmlchange options ['file']"),
        (dict(xml=[("RUN_STARTDATE", "2016-01-01")]), "differs from start"),
        (dict(sourcemods={"src.elm/X.F90": {"from": "/x"}}), "has no sha256"),
        (dict(sourcemods={"../x.F90": {"from": "/x", "sha256": "0" * 64}}), "SourceMods path"),
        (dict(user_files={"user_nl_elm": {"from": "/x", "sha256": "0" * 64}}), "user file name"),
        (dict(user_files={"user_datm.streams.txt.X": {"from": "/x"}}), "has no sha256"),
        (dict(user_nl={"elm": ["NL_END_OF_RECIPE"]}), "heredoc marker"),
        (dict(user_nl={"elm; touch /tmp/x": ["a = 1"]}), "user_nl component"),
        (dict(create_options={"user-mods-dir": "/x"}), "not one a recipe reproduces"),
    ])
    def test_what_is_refused(self, tmp_path, change, problem):
        r = _recipe(tmp_path, **change)
        assert any(problem in p for p in r.check()), r.check()
        with pytest.raises(DC.DomainCaseError, match="cannot be used"):
            DC.write_case_script(r, tmp_path / "rec")

    def test_a_recipe_never_builds_in_the_case_it_was_copied_from(self, tmp_path):
        r = _recipe(tmp_path, copied_from=str(tmp_path / "synth"))
        assert any("the case it was copied from" in p for p in r.check())
        assert _recipe(tmp_path, copied_from=str(tmp_path / "other")).check() == []

    def test_an_unknown_recipe_key_is_refused(self, tmp_path):
        d = json.loads(_recipe(tmp_path).to_json())
        d["rundir"] = "/x"
        with pytest.raises(DC.DomainCaseError, match=r"\['rundir'\] are not recipe fields"):
            DC.Recipe.from_json(json.dumps(d))

    def test_an_indexed_user_nl_entry_is_left_out_not_a_crash(self, tmp_path):
        r = _recipe(tmp_path, user_nl={"elm": ["finidat = '/in/f.nc'", "hist_nhtfrq(2) = -24"]})
        assert r.inputs()["finidat"] == "/in/f.nc" and "hist_nhtfrq" not in r.namelist()


def _case_records(tmp_path, readme, status, user=None, env=None):
    """A case's records as CIME writes them."""
    cd = tmp_path / "cases" / "old"
    (cd / "SourceMods" / "src.elm").mkdir(parents=True)
    (cd / "SourceMods" / "src.elm" / "README").write_text("put mods here")
    (cd / "README.case").write_text(f"2026-10-14 10:12:19: {readme}\n ----\n")
    (cd / "CaseStatus").write_text("".join(
        f"2026-10-14 10:12:{20 + k}: {l}\n ---\n" for k, l in enumerate(status)))
    for name, text in (user or {"user_nl_elm": "finidat = 'f.nc'\n"}).items():
        (cd / name).write_text(text)
    if env is not None:
        (cd / "env_case.xml").write_text("".join(f'<entry id="{k}" value="{v}">\n'
                                                 for k, v in env.items()))
    return cd


CREATE = "/src/E3SM/cime/scripts/create_newcase --case {cd} --compset C%X --res ELM_USRDAT " \
         "--mach compy --compiler intel --project e3sm"
X = "xmlchange success <command> {}  </command>"
START = X.format("./xmlchange RUN_STARTDATE=2015-01-01")


class TestRecipeFromCase:

    def test_from_a_cases_own_records(self, tmp_path):
        cd = _case_records(tmp_path, CREATE.format(cd="old"), [
            X.format("./xmlchange A=1,B=two"),
            X.format("./xmlchange RUN_TYPE=startup,RUN_STARTDATE=2015-01-01"),
            "case.setup starting ", "case.setup success ",
            X.format("./xmlchange CONTINUE_RUN=FALSE,RUN_STARTDATE=2015-01-01,STOP_N=1"),
            X.format("./xmlchange JOB_QUEUE=short")],
            user={"user_nl_elm": "finidat = 'f.nc'\n! note\n",
                  "user_nl_datm": "!---- comments only\n"})
        mod = cd / "SourceMods" / "src.elm" / "PhenologyMod.F90"
        mod.write_text("! fixed\n")
        r = DC.recipe_from_case(cd, case="copy", case_root=str(tmp_path / "new"))
        assert (r.case, r.srcroot, r.compset, r.res, r.mach, r.compiler, r.project) == \
            ("copy", "/src/E3SM", "C%X", "ELM_USRDAT", "compy", "intel", "e3sm")
        assert r.xml == [("A", "1"), ("B", "two"), ("RUN_TYPE", "startup"),
                         ("RUN_STARTDATE", "2015-01-01"), ("JOB_QUEUE", "short")]
        assert r.notes[1] == "set after case.setup: [('JOB_QUEUE', 'short')]"
        assert r.notes[2].endswith("['CONTINUE_RUN', 'RUN_STARTDATE', 'STOP_N']")
        assert r.start == "2015-01-01" and r.user_nl == {"elm": ["finidat = 'f.nc'", "! note"]}
        assert r.sourcemods == {"src.elm/PhenologyMod.F90": {"from": str(mod),
                                                             "sha256": DC.sha256(mod)}}
        assert r.copied_from == str(cd) and r.check() == []
        with pytest.raises(DC.DomainCaseError, match="not a CIME case"):
            DC.recipe_from_case(tmp_path)

    def test_without_a_new_name_the_recipe_is_refused(self, tmp_path):
        cd = _case_records(tmp_path, CREATE.format(cd="old"), [START])
        assert any("the case it was copied from" in p for p in DC.recipe_from_case(cd).check())

    def test_the_source_tree_comes_from_env_case_whatever_was_typed(self, tmp_path):
        """README.case is the command as typed: ./create_newcase in the CIME
        guide's form, or a path relative to wherever the author stood."""
        for k, typed in enumerate(("./create_newcase", "../../cime/scripts/create_newcase")):
            cd = _case_records(tmp_path / str(k), CREATE.format(cd="old").replace(
                "/src/E3SM/cime/scripts/create_newcase", typed), [START],
                env={"SRCROOT": "/abs/E3SM"})
            assert DC.recipe_from_case(cd, case="c2").srcroot == "/abs/E3SM"
        cd = _case_records(tmp_path / "rel", CREATE.format(cd="old").replace(
            "/src/E3SM/cime/scripts/create_newcase", "./create_newcase"), [START])
        assert any("no srcroot" in p for p in DC.recipe_from_case(cd, case="c2").check())
        cd = _case_records(tmp_path / "both", CREATE.format(cd="old") + " --srcroot /other",
                           [START], env={"SRCROOT": "/abs/E3SM"})
        with pytest.raises(DC.DomainCaseError, match="is not the case's SRCROOT"):
            DC.recipe_from_case(cd, case="c2")

    def test_create_newcase_spellings(self, tmp_path):
        cd = _case_records(tmp_path, "/src/E3SM/cime/scripts/create_newcase -case old "
                           "-compset C -res ELM_USRDAT -mach compy -compiler intel "
                           "--project=e3sm --mpilib openmpi --pecount S --run-unsupported",
                           [START])
        r = DC.recipe_from_case(cd, case="c2")
        assert (r.compset, r.res, r.mach, r.compiler, r.project) == \
            ("C", "ELM_USRDAT", "compy", "intel", "e3sm")
        assert r.create_options == {"mpilib": "openmpi", "pecount": "S",
                                    "run-unsupported": True}
        line = next(l for l in DC.write_case_script(r, tmp_path / "rec").read_text()
                    .splitlines() if "create_newcase" in l and "--case" in l)
        assert "--machine compy" in line
        assert line.endswith("--project e3sm --mpilib openmpi --pecount S --run-unsupported")
        for opt in ("--user-mods-dir /x", "--output-root /x", "--ninst 2"):
            cd2 = _case_records(tmp_path / opt.split()[0][2:], CREATE.format(cd="o") + " " + opt,
                                [START])
            with pytest.raises(DC.DomainCaseError, match="not reproduced by a recipe"):
                DC.recipe_from_case(cd2, case="c2")

    def test_xmlchange_spellings(self, tmp_path):
        cd = _case_records(tmp_path, CREATE.format(cd="old"), [
            X.format("/x/c/xmlchange NTASKS=40"),
            X.format("./xmlchange JOB_WALLCLOCK_TIME=00:20:00 --subgroup case.run"),
            X.format("./xmlchange --id DATM_MODE --val CLMMOSARTTEST --force"),
            X.format("./xmlchange --append CAM_CONFIG_OPTS=-phys"),
            X.format(r"./xmlchange HIST_OPTION=nmonths\,x,HIST_N=2"),
            X.format("./xmlchange --id MOD_OPTS --val a,b"),
            START])
        r = DC.recipe_from_case(cd, case="c2")
        assert r.xml == [("NTASKS", "40"),
                         ("JOB_WALLCLOCK_TIME", "00:20:00", {"subgroup": "case.run"}),
                         ("DATM_MODE", "CLMMOSARTTEST"),
                         ("CAM_CONFIG_OPTS", "-phys", {"append": True}),
                         ("HIST_OPTION", r"nmonths\,x"), ("HIST_N", "2"),
                         ("MOD_OPTS", "a,b"),
                         ("RUN_STARTDATE", "2015-01-01")]
        text = DC.write_case_script(r, tmp_path / "rec").read_text()
        assert "./xmlchange --subgroup case.run JOB_WALLCLOCK_TIME=00:20:00" in text
        assert "./xmlchange --append CAM_CONFIG_OPTS=-phys" in text
        # xmlchange never splits --val: each value goes to CIME as it went before
        assert "./xmlchange --id HIST_OPTION --val 'nmonths\\,x'" in text
        assert "./xmlchange --id MOD_OPTS --val a,b" in text
        cd2 = _case_records(tmp_path / "f", CREATE.format(cd="o"),
                            [X.format("./xmlchange --file env_run.xml STOP_N=5")])
        with pytest.raises(DC.DomainCaseError, match="option --file"):
            DC.recipe_from_case(cd2, case="c2")

    def test_run_control_and_the_cases_own_folders_are_left_out_wherever_set(self, tmp_path):
        """Hand scripts set STOP_* before case.setup, and E3SM's run script
        sets EXEROOT and RUNDIR: a copy runs its own stages in its own folders."""
        cd = _case_records(tmp_path, CREATE.format(cd="old"), [
            X.format("./xmlchange STOP_OPTION=nyears,STOP_N=5"),
            X.format("./xmlchange EXEROOT=/old/case/build"),
            X.format("./xmlchange RUNDIR=/old/scratch/run"),
            START, "case.setup success "])
        r = DC.recipe_from_case(cd, case="copy", case_root=str(tmp_path / "new"))
        assert r.xml == [("RUN_STARTDATE", "2015-01-01")] and r.check() == []
        assert r.notes[-1].endswith("['EXEROOT', 'RUNDIR', 'STOP_N', 'STOP_OPTION']")
        text = DC.write_case_script(r, tmp_path / "rec").read_text()
        assert "EXEROOT" not in text and "RUNDIR" not in text

    def test_a_clone_is_refused_with_its_parent_named(self, tmp_path):
        cd = _case_records(tmp_path, "/src/E3SM/cime/scripts/create_clone --case /c/ctl "
                           "--clone /c/rep2015 --keepexe", [])
        with pytest.raises(DC.DomainCaseError, match="is a clone of /c/rep2015: the settings "
                                                     "its parent made"):
            DC.recipe_from_case(cd, case="c2")

    def test_user_files_travel_with_the_recipe(self, tmp_path):
        """A user_datm.streams.txt.* file changes the forcing (CIME copies
        user_* over the default stream file): it is part of the rebuild."""
        src = _fake_e3sm(tmp_path)
        cd = _case_records(tmp_path, CREATE.format(cd="old").replace("/src/E3SM", str(src)),
                           [START], user={"user_nl_elm": "finidat = 'f.nc'\n",
                                          "user_datm.streams.txt.CLMMOSARTTEST":
                                          "<file>adjusted</file>\n"})
        r = DC.recipe_from_case(cd, case="copy", case_root=str(tmp_path / "new"))
        assert set(r.user_files) == {"user_datm.streams.txt.CLMMOSARTTEST"}
        script = DC.write_case_script(r, tmp_path / "rec")
        (cd / "user_datm.streams.txt.CLMMOSARTTEST").unlink()        # the source case goes
        p = subprocess.run(["bash", str(script)], capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        assert (r.case_dir / "user_datm.streams.txt.CLMMOSARTTEST").read_text() == \
            "<file>adjusted</file>\n"

    def test_a_user_file_changed_after_it_was_read_is_refused(self, tmp_path):
        cd = _case_records(tmp_path, CREATE.format(cd="old"), [START],
                           user={"user_nl_elm": "a = 1\n", "user_datm.streams.txt.X": "a\n"})
        r = DC.recipe_from_case(cd, case="copy", case_root=str(tmp_path / "new"))
        (cd / "user_datm.streams.txt.X").write_text("b\n")
        with pytest.raises(DC.DomainCaseError, match=r"user file user_datm.streams.txt.X: .*"
                                                     r"\(sha256 differs\)"):
            DC.write_case_script(r, tmp_path / "rec")

    def test_a_start_date_never_changed_comes_from_env_run(self, tmp_path):
        cd = _case_records(tmp_path, CREATE.format(cd="old"), [])
        (cd / "env_run.xml").write_text('<entry id="RUN_STARTDATE" value="0001-01-01">\n')
        r = DC.recipe_from_case(cd, case="c2")
        assert r.start == "0001-01-01" and r.check() == []


class TestSourceState:

    def _repo(self, tmp_path):
        repo = tmp_path / "E3SM"
        repo.mkdir()
        git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
        (repo / "a.F90").write_text("x = 1\n")
        for argv in (["init", "-q"], ["add", "a.F90"], ["commit", "-q", "-m", "x"]):
            subprocess.run(git + argv, cwd=repo, check=True)
        return repo

    def test_commit_changes_and_their_fingerprint(self, tmp_path):
        repo = self._repo(tmp_path)
        clean = DC.source_state(repo)
        assert len(clean["commit"]) == 40 and clean["changed"] == [] and clean["untracked"] == []
        (repo / "a.F90").write_text("x = 2\n")
        dirty = DC.source_state(repo)
        assert dirty["commit"] == clean["commit"] and dirty["changed"] == ["M a.F90"]
        assert dirty["fingerprint"] != clean["fingerprint"]
        assert "error" in DC.source_state(tmp_path / "nowhere")

    def test_an_untracked_source_file_changes_the_fingerprint(self, tmp_path):
        """E3SM's CMake globs its source folders: an untracked module is built."""
        repo = self._repo(tmp_path)
        before = DC.source_state(repo)
        (repo / "NewFireMod.F90").write_text("module x\n")
        after = DC.source_state(repo)
        assert after["untracked"] == ["NewFireMod.F90"]
        assert after["fingerprint"] != before["fingerprint"]
        (repo / "NewFireMod.F90").write_text("module y\n")
        assert DC.source_state(repo)["fingerprint"] != after["fingerprint"]

    def test_a_change_inside_a_submodule_changes_the_fingerprint(self, tmp_path):
        """cime is a submodule of E3SM, and its local changes are built: a
        second edit there leaves the parent's own diff ('-dirty') as it was."""
        sub = tmp_path / "cime_src"
        sub.mkdir()
        g = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
        (sub / "x.py").write_text("a = 1\n")
        for argv in (["init", "-q"], ["add", "x.py"], ["commit", "-q", "-m", "x"]):
            subprocess.run(g + argv, cwd=sub, check=True)
        repo = self._repo(tmp_path)
        subprocess.run(g + ["-c", "protocol.file.allow=always", "submodule", "add", str(sub),
                            "cime"], cwd=repo, check=True, capture_output=True)
        subprocess.run(g + ["commit", "-q", "-m", "sub"], cwd=repo, check=True)
        clean = DC.source_state(repo)
        (repo / "cime" / "x.py").write_text("a = 2\n")
        one = DC.source_state(repo)
        (repo / "cime" / "x.py").write_text("a = 3\n")
        two = DC.source_state(repo)
        assert len({clean["fingerprint"], one["fingerprint"], two["fingerprint"]}) == 3
        assert one["changed"] == ["M cime"]

    def test_the_record_holds_the_source_state(self, tmp_path):
        repo = self._repo(tmp_path)
        DC.write_case_script(_recipe(tmp_path / "cases", srcroot=repo), tmp_path / "rec")
        rec = json.loads((tmp_path / "rec" / DC.RECIPE_FILE).read_text())
        assert rec["source"] == DC.source_state(repo)


# ── the case script, run against stub CIME commands ─────────────────────────
def _fake_e3sm(tmp_path):
    src = tmp_path / "E3SM"
    (src / "cime" / "scripts").mkdir(parents=True, exist_ok=True)
    _stub(src / "cime" / "scripts" / "create_newcase", """
case=""; while [ $# -gt 0 ]; do [ "$1" = "--case" ] && case=$2; shift; done
mkdir -p "$case"; echo "$0 $*" > "$case/created"; echo "$LANG $LC_ALL" > "$case/locale"
for t in xmlchange case.setup preview_namelists case.build; do
  printf '#!/bin/bash\\necho "%s $*" >> calls\\n' "$t" > "$case/$t"; chmod +x "$case/$t"; done""")
    return src


class TestCaseScript:

    def test_it_creates_the_case_as_recorded(self, tmp_path):
        src = _fake_e3sm(tmp_path)
        mod = tmp_path / "PhenologyMod.F90"
        mod.write_text("! the fix\n")
        r = _recipe(tmp_path / "cases", srcroot=src,
                    sourcemods={"src.elm/PhenologyMod.F90": {"from": str(mod),
                                                             "sha256": DC.sha256(mod)}})
        script = DC.write_case_script(r, tmp_path / "rec")
        assert json.loads((tmp_path / "rec" / DC.RECIPE_FILE).read_text())["case"] == "synth"
        p = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                           env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
        assert p.returncode == 0, p.stderr
        cd = r.case_dir
        assert (cd / "locale").read_text().split() == ["en_US.UTF-8"] * 2      # job 788213
        assert (cd / "user_nl_elm").read_text().splitlines() == r.user_nl["elm"]
        assert (cd / "SourceMods" / "src.elm" / "PhenologyMod.F90").read_text() == "! the fix\n"
        assert (cd / "calls").read_text().splitlines() == [
            "xmlchange LND_DOMAIN_PATH=/in", "xmlchange LND_DOMAIN_FILE=domain.nc",
            "xmlchange NTASKS=40", "xmlchange RUN_TYPE=startup",
            "xmlchange RUN_STARTDATE=2015-01-01", "case.setup ", "preview_namelists "]
        again = subprocess.run(["bash", str(script)], capture_output=True, text=True)
        assert again.returncode == 1 and "case exists" in again.stderr

    def test_the_sourcemods_travel_with_the_record(self, tmp_path):
        """Copied into the record when the script is written, so the case
        they came from can go; the copy is checked again at build time."""
        src = _fake_e3sm(tmp_path)
        mod = tmp_path / "PhenologyMod.F90"
        mod.write_text("! the fix\n")
        r = _recipe(tmp_path / "cases", srcroot=src,
                    sourcemods={"src.elm/PhenologyMod.F90": {"from": str(mod),
                                                             "sha256": DC.sha256(mod)}})
        script = DC.write_case_script(r, tmp_path / "rec")
        mod.unlink()
        p = subprocess.run(["bash", str(script)], capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        assert (r.case_dir / "SourceMods" / "src.elm" / "PhenologyMod.F90").read_text() == \
            "! the fix\n"

    def test_a_sourcemod_that_changed_is_refused(self, tmp_path):
        src = _fake_e3sm(tmp_path)
        mod = tmp_path / "PhenologyMod.F90"
        mod.write_text("! the fix\n")
        sm = {"src.elm/PhenologyMod.F90": {"from": str(mod), "sha256": DC.sha256(mod)}}
        r = _recipe(tmp_path / "cases", srcroot=src, sourcemods=sm)
        mod.write_text("! edited since\n")
        with pytest.raises(DC.DomainCaseError, match="sha256 differs"):
            DC.write_case_script(r, tmp_path / "rec")
        mod.write_text("! the fix\n")
        script = DC.write_case_script(r, tmp_path / "rec2")
        (tmp_path / "rec2" / "SourceMods" / "src.elm" / "PhenologyMod.F90").write_text("! no\n")
        p = subprocess.run(["bash", str(script)], capture_output=True, text=True)
        assert p.returncode != 0 and "FAILED" in p.stdout + p.stderr
        assert "case.setup" not in (r.case_dir / "calls").read_text()


# ── the stage script, run with the real gate on a synthetic case ────────────
def _stage_case(tmp_path, submit_rc=0):
    """A synthetic case (job 111, years 2015-2016) whose CIME commands are stubs."""
    c = cf.write_case(tmp_path / "x")
    cd = c["case_dir"]
    _stub(cd / "xmlquery", f'echo {c["run_dir"]}')
    _stub(cd / "xmlchange", 'echo "$*" >> xmlchange.log')
    _stub(cd / "case.submit", f'echo submitted >> submit.log; exit {submit_rc}')
    _stub(cd / "case.build", 'echo built >> build.log')
    r = _recipe(cd.parent, case=cd.name, srcroot=tmp_path / "E3SM")
    return c, r


def _run_stage(script, stage, *args, job="111", path_extra=None, env=None):
    e = dict(os.environ, SLURM_JOB_ID=job, **(env or {}))
    if path_extra:
        e["PATH"] = f"{path_extra}:{e['PATH']}"
    return subprocess.run(["bash", str(script), stage, *args], capture_output=True, text=True,
                          env=e, timeout=600)


def _git_tree(src):
    g = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    for argv in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "x"]):
        subprocess.run(g + argv, cwd=src, check=True)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=src, capture_output=True,
                          text=True).stdout.strip()


class TestStageScript:

    def test_the_first_year_starts_from_finidat_and_the_gate_decides(self, tmp_path):
        c, r = _stage_case(tmp_path)
        rec = tmp_path / "rec"
        script = DC.write_stage_script(r, rec)
        p = _run_stage(script, "y2015")
        assert p.returncode == 1, p.stderr      # the fixture's water does not close
        assert (c["case_dir"] / "xmlchange.log").read_text().strip() == \
            "CONTINUE_RUN=FALSE,STOP_OPTION=nyears,STOP_N=1,REST_OPTION=nyears,REST_N=1," \
            "RUN_STARTDATE=2015-01-01"
        chk = json.loads((rec / "checks" / "y2015_111.json").read_text())
        assert chk["pass"] is False and chk["years"]["2015"]["year"] == 2015
        status = (rec / DC.STATUS_FILE).read_text()
        assert "job=111 stage=y2015 started on" in status
        assert "job=111 stage=y2015 CHECK FAIL" in status and "finished rc=1" in status

    def test_a_later_year_continues(self, tmp_path):
        c, r = _stage_case(tmp_path)
        _run_stage(DC.write_stage_script(r, tmp_path / "rec"), "y2016")
        assert (c["case_dir"] / "xmlchange.log").read_text().strip() == \
            "CONTINUE_RUN=TRUE,STOP_OPTION=nyears,STOP_N=1,REST_OPTION=nyears,REST_N=1"

    def test_the_job_runs_in_a_utf8_locale(self, tmp_path):
        c, r = _stage_case(tmp_path)
        _stub(c["case_dir"] / "case.submit", 'echo "$LANG $LC_ALL" > locale.log')
        _run_stage(DC.write_stage_script(r, tmp_path / "rec"), "y2016",
                   env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
        assert (c["case_dir"] / "locale.log").read_text().split() == ["en_US.UTF-8"] * 2

    def test_a_model_step_that_fails_still_leaves_a_check_record(self, tmp_path):
        c, r = _stage_case(tmp_path, submit_rc=3)
        rec = tmp_path / "rec"
        p = _run_stage(DC.write_stage_script(r, rec), "y2016")
        assert p.returncode == 3                     # the first failure is the exit code
        assert (rec / "checks" / "y2016_111.json").is_file()

    def test_a_failed_xmlchange_does_not_run_the_model(self, tmp_path):
        c, r = _stage_case(tmp_path)
        _stub(c["case_dir"] / "xmlchange", "exit 4")
        p = _run_stage(DC.write_stage_script(r, tmp_path / "rec"), "y2016")
        assert p.returncode == 4 and not (c["case_dir"] / "submit.log").exists()

    def test_the_job_header(self, tmp_path):
        head = DC.write_stage_script(_recipe(tmp_path, project="myproj"),
                                     tmp_path / "rec").read_text().splitlines()[:6]
        assert "#SBATCH --account=myproj" in head
        assert "#SBATCH --exclusive" in head and "#SBATCH --nodes=1" in head
        r = _recipe(tmp_path, xml=[("NTASKS", "120"), ("RUN_STARTDATE", "2015-01-01")])
        assert "#SBATCH --nodes=3" in DC.write_stage_script(r, tmp_path / "r2").read_text()
        r = _recipe(tmp_path, xml=[("NTASKS_LND", "-2"), ("RUN_STARTDATE", "2015-01-01")])
        assert "#SBATCH --nodes=2" in DC.write_stage_script(r, tmp_path / "r3").read_text()
        with pytest.raises(DC.DomainCaseError, match="whitespace"):
            DC.write_stage_script(r, tmp_path / "a record")

    def test_a_smoke_run_that_passes(self, tmp_path):
        c = cf.write_case(tmp_path / "x")
        run = tmp_path / "smoke" / "run"
        run.mkdir(parents=True)
        g = fx.grid()
        fx._write(run / "s.elm.h1.2015-01-01-00000.nc", g, 2015,
                  [(-fx.DT, 0.0)] + [(k, k + 1.0) for k in range(5)],
                  [None] + [(2015, k + 1, 0) for k in range(5)], list(cf.DAILY_FIELDS))
        fx._write(run / "s.elm.h2.2015-01-01-00000.nc", g, 2015,
                  [(-fx.DT, 0.0)] + [(k / 8, (k + 1) / 8) for k in range(40)],
                  [None] + [(2015, k // 8 + 1, k % 8) for k in range(40)],
                  ["STEPCODE", "CELLID", "TSA"], steps_per_day=8)
        cf.write_logs(run, c["case_dir"], "s", "333", end=20150106, days=5.0)
        cd = c["case_dir"]
        _stub(cd / "xmlquery", f"echo {run}")
        _stub(cd / "xmlchange", 'echo "$*" >> xmlchange.log')
        _stub(cd / "case.submit", "exit 0")
        r = _recipe(cd.parent, case=cd.name)
        rec = tmp_path / "rec"
        p = _run_stage(DC.write_stage_script(r, rec, smoke_days=5), "smoke", job="333")
        assert p.returncode == 0, (rec / DC.STATUS_FILE).read_text()
        assert (cd / "xmlchange.log").read_text().strip() == \
            "CONTINUE_RUN=FALSE,RUN_STARTDATE=2015-01-01,STOP_OPTION=ndays,STOP_N=5," \
            "REST_OPTION=ndays,REST_N=5"
        assert json.loads((rec / "checks" / "smoke_333.json").read_text())["pass"] is True

    def test_the_case_stage_records_its_verdict_and_source(self, tmp_path):
        """The source line names the commit from the job too (Compy's
        login-node git is 1.8.3, which has no -C)."""
        src = _fake_e3sm(tmp_path)
        head = _git_tree(src)
        r = _recipe(tmp_path / "cases", srcroot=src)
        rec = tmp_path / "rec"
        DC.write_case_script(r, rec)
        p = _run_stage(DC.write_stage_script(r, rec), "case", job="10")
        assert p.returncode == 0, p.stderr
        assert (r.case_dir / "calls").read_text().splitlines()[-1] == "case.build "
        record = json.loads((rec / "checks" / "case_10.json").read_text())
        assert record["pass"] is True and record["source"]["commit"] == head
        assert f"stage=case source {head} fingerprint {record['source']['fingerprint']}" in \
            (rec / DC.STATUS_FILE).read_text()

    def test_a_case_that_exists_is_neither_built_nor_passed(self, tmp_path):
        src = _fake_e3sm(tmp_path)
        r = _recipe(tmp_path / "cases", srcroot=src)
        rec = tmp_path / "rec"
        DC.write_case_script(r, rec)
        r.case_dir.mkdir(parents=True)                    # someone's case is already there
        _stub(r.case_dir / "case.build", "echo BUILT >> build.log")
        p = _run_stage(DC.write_stage_script(r, rec), "case", job="10")
        assert p.returncode != 0 and not (r.case_dir / "build.log").exists()
        assert json.loads((rec / "checks" / "case_10.json").read_text())["pass"] is False

    def test_a_failed_build_fails_the_stage(self, tmp_path):
        src = _fake_e3sm(tmp_path)
        nc = src / "cime" / "scripts" / "create_newcase"
        nc.write_text(nc.read_text() + '\nprintf "#!/bin/bash\\nexit 3\\n" > "$case/case.build"\n')
        r = _recipe(tmp_path / "cases", srcroot=src)
        rec = tmp_path / "rec"
        DC.write_case_script(r, rec)
        p = _run_stage(DC.write_stage_script(r, rec), "case", job="10")
        assert p.returncode == 3 and "finished rc=3" in (rec / DC.STATUS_FILE).read_text()
        assert json.loads((rec / "checks" / "case_10.json").read_text())["reasons"] == \
            ["the case stage exited 3"]

    def test_a_case_script_changed_after_arming_is_refused(self, tmp_path):
        """SLURM keeps the stage script as it was at submission; the case script
        it runs is read from disk, so its sha256 is written into the stage script."""
        src = _fake_e3sm(tmp_path)
        r = _recipe(tmp_path / "cases", srcroot=src)
        rec = tmp_path / "rec"
        DC.write_case_script(r, rec)
        armed = DC.write_stage_script(r, rec)
        (rec / DC.CASE_SCRIPT).write_text((rec / DC.CASE_SCRIPT).read_text()
                                          .replace("NTASKS=40", "NTASKS=999"))
        p = _run_stage(armed, "case", job="10")
        assert p.returncode == 3 and not r.case_dir.exists()
        assert "is not the script this chain was armed with" in (rec / DC.STATUS_FILE).read_text()

    def test_the_report_names_every_job_even_one_slurm_never_ran(self, tmp_path):
        rec = tmp_path / "rec"
        script = DC.write_stage_script(_recipe(tmp_path), rec)
        stubs = tmp_path / "bin"
        stubs.mkdir()
        # as Compy's sacct: without -S it finds nothing that ended before midnight
        _stub(stubs / "sacct", f'echo "$*" >> {tmp_path}/sacct.calls\n'
                               'case " $* " in *" -S "*) ;; *) exit 0 ;; esac\n'
                               '[ "$2" = "11" ] && exit 0\necho "$2|run_x|COMPLETED|00:49:46|0:0"')
        p = _run_stage(script, "report", "10", "11", job="12", path_extra=stubs)
        status = (rec / DC.STATUS_FILE).read_text()
        assert "job=12 stage=report REPORT run_x job=10 state=COMPLETED elapsed=00:49:46" in status
        assert "REPORT job=11 state=UNKNOWN (no accounting record" in status
        assert "job=10 state=UNKNOWN" not in status and p.returncode == 0
        since = (datetime.date.today() - datetime.timedelta(days=DC.SACCT_LOOKBACK_DAYS))
        calls = (tmp_path / "sacct.calls").read_text().splitlines()
        assert len(calls) == 4 and all(f" -S {since.isoformat()} " in c for c in calls)

    def test_four_digit_stages_for_early_years(self, tmp_path):
        c, r = _stage_case(tmp_path)
        r = _recipe(r.case_root, case=r.case, start="0001-01-01",
                    xml=[("RUN_STARTDATE", "0001-01-01")])
        jobs = DC.plan_chain(r, [1, 2], with_case=False, smoke=False)
        assert [j.stage for j in jobs] == ["y0001", "y0002", "report"]
        _run_stage(DC.write_stage_script(r, tmp_path / "rec"), "y0001")
        assert "RUN_STARTDATE=0001-01-01" in (c["case_dir"] / "xmlchange.log").read_text()

    def test_an_unknown_stage(self, tmp_path):
        c, r = _stage_case(tmp_path)
        rec = tmp_path / "rec"
        p = _run_stage(DC.write_stage_script(r, rec), "y20x6")
        assert p.returncode == 2 and "FAILED unknown stage" in (rec / DC.STATUS_FILE).read_text()


# ── the year gate against a reference (the piece-3 options of run_gate) ─────
class TestReferenceGate:

    def _ref(self, c, tmp_path, scale=1.0, case="other"):
        import netCDF4
        ref = tmp_path / f"ref_{case}_{scale}"
        ref.mkdir(exist_ok=True)
        for p in c["run_dir"].glob("synth.elm.h1.*.nc"):
            q = ref / p.name.replace("synth", case)
            shutil.copy(p, q)
            if scale != 1.0:
                with netCDF4.Dataset(q, "a") as d:
                    d["GPP"][:] = d["GPP"][:] * scale
        return ref

    def _year(self, rec, y):
        return json.loads((rec / "checks" / f"y{y}_111.json").read_text())["years"][str(y)]

    def test_the_reference_is_gated_only_in_its_years(self, tmp_path):
        c, r = _stage_case(tmp_path)
        rec = tmp_path / "rec"
        script = DC.write_stage_script(r, rec, reference=DC.Reference(
            str(self._ref(c, tmp_path)), gate_years=[2016]))
        _run_stage(script, "y2016")
        y = self._year(rec, 2016)
        assert y["reference"]["gated"] is True and y["reference"]["pass"] is True
        assert y["reference"]["fields"]["GPP"]["bias_pct"] == 0.0
        _run_stage(script, "y2015")
        assert self._year(rec, 2015)["reference"]["gated"] is False
        assert "refGPP=+0.00%" in (rec / DC.STATUS_FILE).read_text()

    def test_a_gated_reference_that_differs_fails_the_year(self, tmp_path):
        c, r = _stage_case(tmp_path)
        rec = tmp_path / "rec"
        ref = self._ref(c, tmp_path, scale=1.2)
        script = DC.write_stage_script(r, rec, reference=DC.Reference(str(ref), gate_years=[2016]))
        _run_stage(script, "y2016")
        y = self._year(rec, 2016)
        assert y["reference"]["pass"] is False and y["pass"] is False
        assert any(x.startswith("reference: GPP: bias") for x in y["reasons"]), y["reasons"]
        _run_stage(script, "y2015")                       # report only: recorded, not gated
        y = self._year(rec, 2015)
        assert y["reference"]["pass"] is False
        assert not any(x.startswith("reference: ") for x in y["reasons"])

    def test_a_reference_that_cannot_be_read_fails_a_gated_year(self, tmp_path):
        c, r = _stage_case(tmp_path)
        rec = tmp_path / "rec"
        (tmp_path / "empty").mkdir()
        script = DC.write_stage_script(r, rec, reference=DC.Reference(str(tmp_path / "empty"),
                                                                      gate_years=[2016]))
        _run_stage(script, "y2016")
        assert any(x.startswith("reference: cannot compare") for x in
                   self._year(rec, 2016)["reasons"])

    def test_the_reference_pattern_picks_the_files(self, tmp_path):
        c, r = _stage_case(tmp_path)
        rec = tmp_path / "rec"
        keep = self._ref(c, tmp_path, case="keep")
        for p in self._ref(c, tmp_path, scale=2.0, case="decoy").iterdir():
            shutil.copy(p, keep / p.name)                 # another case's files beside them
        script = DC.write_stage_script(r, rec, reference=DC.Reference(
            str(keep), pattern="keep.elm.h1.*.nc", gate_years=[2016]))
        _run_stage(script, "y2016")
        y = self._year(rec, 2016)
        assert y["reference"]["pass"] is True, y["reference"].get("reasons")
        assert y["reference"]["fields"]["GPP"]["bias_pct"] == 0.0


# ── the chain ────────────────────────────────────────────────────────────────
def _armed(tmp_path, years=(2015, 2016), slurm=None, **kw):
    r = _recipe(tmp_path, **kw)
    rec = tmp_path / "rec"
    DC.write_stage_script(r, rec)
    out = DC.submit_chain(DC.plan_chain(r, list(years), tag="nb"), rec, run=slurm or FakeSlurm())
    return r, rec, out


def _pass(rec, *stage_jobs, ok=True):
    for s, j in stage_jobs:
        (rec / "checks" / f"{s}_{j}.json").write_text(json.dumps({"pass": ok}))


ALL_DONE = {k: "COMPLETED" for k in ("1001", "1002", "1003", "1004", "1005")}


class TestPlan:

    def test_the_plan(self, tmp_path):
        r = _recipe(tmp_path)
        jobs = DC.plan_chain(r, [2015, 2016, 2017], tag="nb", partition="slurm7")
        assert [(j.stage, j.after, j.walltime) for j in jobs] == [
            ("case", None, "01:00:00"), ("smoke", "case", "00:45:00"),
            ("y2015", "smoke", "01:55:00"), ("y2016", "y2015", "01:55:00"),
            ("y2017", "y2016", "01:55:00"), ("report", None, "00:10:00")]
        assert {j.partition for j in jobs} == {"slurm7"}
        resumed = DC.plan_chain(r, [2017, 2018], with_case=False, smoke=False)
        assert [j.stage for j in resumed] == ["y2017", "y2018", "report"]
        assert resumed[0].after is None
        for years, msg in (([2015, 2017], "not consecutive"), ([2014], "before the recipe"),
                           ([2016], "starts at 2015"), ([], "at least one model year")):
            with pytest.raises(DC.DomainCaseError, match=msg):
                DC.plan_chain(r, years)
        early = _recipe(tmp_path, start="0001-01-01", xml=[])
        with pytest.raises(DC.DomainCaseError, match="outside 1-9999"):
            DC.plan_chain(early, [0, 1], with_case=False, smoke=False)

    def test_years_past_the_forcing_are_refused(self, tmp_path):
        """datm cycles a CLMNCEP stream: model year 2024 on 2013-2023 forcing
        reads the 2013 weather, and the year gate cannot tell."""
        r = _recipe(tmp_path, xml=[("DATM_CLMNCEP_YR_ALIGN", "2013"),
                                   ("DATM_CLMNCEP_YR_START", "2013"),
                                   ("DATM_CLMNCEP_YR_END", "2023"),
                                   ("RUN_STARTDATE", "2015-01-01")])
        assert DC.plan_chain(r, list(range(2015, 2024)))[-2].stage == "y2023"
        with pytest.raises(DC.DomainCaseError, match="model year 2024 would read the 2013 "
                                                     "weather"):
            DC.plan_chain(r, list(range(2015, 2025)))
        assert DC.plan_chain(r, list(range(2015, 2025)), allow_forcing_cycle=True)[-2].stage == \
            "y2024"

    def test_the_chain_spec_is_checked(self):
        assert DC.chain_spec_problems({"years": [2015], "walls": {"year_wall": "01:00:00"},
                                       "reference": {"run_dir": "/r", "gate_years": [2015]}}) == []
        probs = DC.chain_spec_problems({"yr_start": 2015, "walls": {"year_walltime": "3:00:00"},
                                        "reference": {"rundir": "/r"}, "years": "2015"})
        assert any("'yr_start'" in p for p in probs) and any("'year_walltime'" in p for p in probs)
        assert any("reference needs a run_dir" in p for p in probs)
        assert any("years must be a list" in p for p in probs)
        assert any("'gate_year'" in p for p in DC.chain_spec_problems(
            {"reference": {"run_dir": "/r", "gate_year": [2015]}}))


class TestSubmit:

    def test_it_is_submitted_with_its_dependencies(self, tmp_path):
        slurm = FakeSlurm()
        r, rec, out = _armed(tmp_path, slurm=slurm)
        assert out["ok"]
        script = str(rec / DC.STAGE_SCRIPT)
        assert slurm.calls == [
            ["sbatch", "--parsable", "-p", "short", "-t", "01:00:00", "-J", "nb_case",
             script, "case"],
            ["sbatch", "--parsable", "-p", "short", "-t", "00:45:00", "-J", "nb_smoke",
             "--dependency=afterok:1001", "--kill-on-invalid-dep=yes", script, "smoke"],
            ["sbatch", "--parsable", "-p", "short", "-t", "01:55:00", "-J", "nb_y2015",
             "--dependency=afterok:1002", "--kill-on-invalid-dep=yes", script, "y2015"],
            ["sbatch", "--parsable", "-p", "short", "-t", "01:55:00", "-J", "nb_y2016",
             "--dependency=afterok:1003", "--kill-on-invalid-dep=yes", script, "y2016"],
            ["sbatch", "--parsable", "-p", "short", "-t", "00:10:00", "-J", "nb_report",
             "--dependency=afterany:1001:1002:1003:1004", script, "report",
             "1001", "1002", "1003", "1004"]]
        chain = json.loads((rec / DC.CHAIN_FILE).read_text())
        assert [(j["stage"], j["job_id"]) for j in chain["segments"][0]["jobs"]] == [
            ("case", "1001"), ("smoke", "1002"), ("y2015", "1003"), ("y2016", "1004"),
            ("report", "1005")]
        assert (rec / DC.STATUS_FILE).read_text().strip().endswith(
            "ARMED case=1001 smoke=1002 y2015=1003 y2016=1004 report=1005")

    def test_an_id_with_its_cluster(self, tmp_path):
        r = _recipe(tmp_path)
        rec = tmp_path / "rec"
        DC.write_stage_script(r, rec)
        out = DC.submit_chain(DC.plan_chain(r, [2015]), rec,
                              run=lambda argv, timeout=120: (0, "4242;compy\n", ""))
        assert out["ok"] and {j["job_id"] for j in out["jobs"]} == {"4242"}

    def test_a_failed_submission_cancels_what_was_queued(self, tmp_path):
        slurm = FakeSlurm(fail_at="nb_y2015")
        r, rec, out = _armed(tmp_path, slurm=slurm)
        assert not out["ok"] and out["error"].startswith("sbatch for y2015 failed")
        assert out["cancelled"] == ["1001", "1002"] and out["not_cancelled"] == []
        assert [c for c in slurm.calls if c[0] == "scancel"] == [["scancel", "1001"],
                                                                 ["scancel", "1002"]]
        assert "ARMING FAILED: sbatch for y2015 failed" in (rec / DC.STATUS_FILE).read_text()

    def test_a_cancellation_slurm_refused_is_not_claimed(self, tmp_path):
        r, rec, out = _armed(tmp_path, slurm=FakeSlurm(fail_at="nb_report", scancel_rc=1))
        assert out["cancelled"] == [] and out["not_cancelled"] == ["1001", "1002", "1003", "1004"]
        assert "NOT cancelled, check by hand: ['1001', '1002', '1003', '1004']" in out["error"]
        assert "; cancelled" not in out["error"]

    def test_a_record_that_cannot_be_written_cancels_what_was_queued(self, tmp_path,
                                                                    monkeypatch):
        """A full quota after the second job was queued: both are cancelled,
        and the failure to record that is itself in the answer."""
        r = _recipe(tmp_path)
        rec = tmp_path / "rec"
        DC.write_stage_script(r, rec)
        slurm = FakeSlurm()
        n = {"writes": 0}
        real = DC._write_json

        def full_quota(path, obj):
            n["writes"] += 1
            if n["writes"] >= 2:
                raise OSError(122, "Disk quota exceeded")
            real(path, obj)
        monkeypatch.setattr(DC, "_write_json", full_quota)
        out = DC.submit_chain(DC.plan_chain(r, [2015, 2016], tag="nb"), rec, run=slurm)
        assert not out["ok"] and "Disk quota exceeded" in out["error"]
        assert [c[1] for c in slurm.calls if c[0] == "scancel"] == ["1001", "1002"]
        assert "this could not be recorded" in out["error"]

    def test_a_record_on_a_node_local_disk_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(DC, "node_local", lambda path: f"{path} is on /dev/sda1")
        slurm = FakeSlurm()
        with pytest.raises(DC.DomainCaseError, match="is on /dev/sda1"):
            _armed(tmp_path, slurm=slurm)
        assert not slurm.calls

    def test_the_real_node_local_check(self, tmp_path):
        """Jobs whose record sits on a login node's own disk die in seconds
        with empty logs (770938/770939)."""
        def device(p):
            return subprocess.check_output(["df", "-P", str(p)], text=True).splitlines()[-1] \
                .split()[0]
        if not device(tmp_path).startswith("/dev/"):
            pytest.skip("this machine's tmp is not a local disk")
        assert "node-local" in REAL_NODE_LOCAL(tmp_path)
        here = Path(__file__).resolve().parent
        if not device(here).startswith("/dev/"):
            assert REAL_NODE_LOCAL(here) is None

    def test_re_arming_keeps_the_years_that_passed(self, tmp_path):
        r, rec, _ = _armed(tmp_path, years=(2015, 2016, 2017))
        _pass(rec, ("smoke", "1002"), ("y2015", "1003"))
        _pass(rec, ("y2016", "1004"), ok=False)
        assert DC.chain_records(rec)["first_unpassed"] == "y2016"
        out = DC.submit_chain(DC.plan_chain(r, [2016, 2017], with_case=False, smoke=False,
                                            tag="nb"), rec, run=FakeSlurm(start=2000))
        assert out["ok"] and [j["job_id"] for j in out["jobs"]] == ["2001", "2002", "2003"]
        chain = json.loads((rec / DC.CHAIN_FILE).read_text())
        assert len(chain["segments"]) == 2 and chain["segments"][0]["jobs"][0]["job_id"] == "1001"
        recs = DC.chain_records(rec)
        assert recs["years_passed"] == [2015] and recs["first_unpassed"] == "y2016"
        assert [(s["stage"], s["job_id"]) for s in recs["stages"]] == [
            ("case", "1001"), ("smoke", "1002"), ("y2015", "1003"), ("y2016", "2001"),
            ("y2017", "2002"), ("report", "2003")]
        _pass(rec, ("y2016", "2001"), ("y2017", "2002"))
        recs = DC.chain_records(rec)
        assert recs["years_passed"] == [2015, 2016, 2017] and recs["first_unpassed"] is None
        st = DC.chain_status(rec, run=FakeSlurm(states={**ALL_DONE, "2001": "COMPLETED",
                                                         "2002": "COMPLETED",
                                                         "2003": "COMPLETED"}))
        assert st["ok"] and st["years_passed"] == [2015, 2016, 2017]

    def test_a_chain_whose_arming_failed_is_never_ok(self, tmp_path):
        """The report's sbatch failed and SLURM refused the cancellations: the
        jobs ran and passed, but the chain was never armed whole."""
        r, rec, out = _armed(tmp_path, slurm=FakeSlurm(fail_at="nb_report", scancel_rc=1))
        _pass(rec, ("case", "1001"), ("smoke", "1002"), ("y2015", "1003"), ("y2016", "1004"))
        st = DC.chain_status(rec, run=FakeSlurm(states=ALL_DONE))
        assert st["done"] and st["failed_stage"] is None and not st["ok"]

    def test_an_unreadable_chain_record_is_never_armed_over(self, tmp_path):
        r, rec, _ = _armed(tmp_path)
        (rec / DC.CHAIN_FILE).write_text('{"segments": [{"jobs": [{"stage": "case", "job')
        slurm = FakeSlurm()
        with pytest.raises(DC.DomainCaseError, match="cannot be read"):
            DC.submit_chain(DC.plan_chain(r, [2015], tag="nb"), rec, run=slurm)
        assert not slurm.calls
        with pytest.raises(DC.DomainCaseError, match="cannot be read"):
            DC.chain_status(rec, run=slurm)


class TestStatus:

    def test_where_the_chain_stands(self, tmp_path):
        r, rec, _ = _armed(tmp_path)
        _pass(rec, ("smoke", "1002"), ("y2015", "1003"))
        mid = DC.chain_status(rec, run=FakeSlurm(states={
            "1001": "COMPLETED", "1002": "COMPLETED", "1003": "COMPLETED", "1004": "RUNNING",
            "1005": "PENDING"}))
        assert mid["active"] and not mid["done"] and mid["years_passed"] == [2015]
        _pass(rec, ("y2016", "1004"), ok=False)
        end = DC.chain_status(rec, run=FakeSlurm(states={**ALL_DONE, "1004": "FAILED"}))
        assert end["done"] and not end["ok"] and end["failed_stage"] == "y2016"
        _pass(rec, ("y2016", "1004"))
        ok = DC.chain_status(rec, run=FakeSlurm(states=ALL_DONE))
        assert ok["ok"] and ok["years_passed"] == [2015, 2016]
        with pytest.raises(DC.DomainCaseError, match="never armed"):
            DC.chain_status(tmp_path)

    def test_a_year_that_completed_without_its_record_has_not_passed(self, tmp_path):
        r, rec, _ = _armed(tmp_path)
        _pass(rec, ("smoke", "1002"), ("y2015", "1003"))
        st = DC.chain_status(rec, run=FakeSlurm(states=ALL_DONE))
        assert st["failed_stage"] == "y2016" and not st["ok"]
        assert DC.chain_records(rec)["first_unpassed"] == "y2016"

    def test_a_chain_whose_later_stages_were_cancelled_while_waiting_has_ended(self, tmp_path):
        """Compy keeps no record of a job cancelled before it started: squeue
        calls the id invalid and sacct has no row. Such a chain read as
        active for ever (the hand chain 788213-788216, whose build failed)."""
        r, rec, _ = _armed(tmp_path)
        st = DC.chain_status(rec, run=FakeSlurm(states={
            "1001": "FAILED", "1002": "GONE", "1003": "GONE", "1004": "GONE",
            "1005": "COMPLETED"}))
        assert not st["active"] and st["done"] and st["failed_stage"] == "case"
        assert [j["inferred"] is not None for j in st["jobs"]] == [False, True, True, True, False]

    def test_a_chain_cancelled_whole_while_it_waited_has_ended(self, tmp_path):
        """scancel on every job before any started: nothing failed first and
        the report never ran, so only GONE says the chain has ended."""
        r, rec, _ = _armed(tmp_path)
        st = DC.chain_status(rec, run=FakeSlurm(states={k: "GONE" for k in ALL_DONE}))
        assert not st["active"] and st["done"] and st["failed_stage"] == "case"
        assert all(j["state"] == DC.GONE and j["inferred"].startswith("ended")
                   for j in st["jobs"])

    def test_no_record_anywhere_is_not_no_answer(self):
        def slurm(squeue, sacct):
            return lambda argv, timeout=60: squeue if argv[0] == "squeue" else sacct
        invalid = (1, "", "slurm_load_jobs error: Invalid job id specified")
        timeout = (1, "", "slurm_load_jobs error: Socket timed out on send/recv operation")
        assert DC.slurm_state("9", run=slurm(invalid, (0, "", ""))) == DC.GONE
        assert DC.slurm_state("9", run=slurm((0, "", ""), (0, "", ""))) == DC.GONE
        assert DC.slurm_state("9", run=slurm(timeout, (0, "", ""))) is None
        assert DC.slurm_state("9", run=slurm(invalid, (1, "", "sacct: error"))) is None
        assert DC.slurm_state("9", run=slurm(invalid, (0, " CANCELLED by 297133 \n", ""))) == \
            "CANCELLED"
        assert DC.slurm_state("9", run=slurm((0, "RUNNING\n", ""), (0, "", ""))) == "RUNNING"

    def test_a_scheduler_that_does_not_answer_is_not_an_ended_chain(self, tmp_path):
        """None is NO ANSWER, not 'ended': a squeue that times out must not let
        the manager collect a running chain."""
        r, rec, _ = _armed(tmp_path)
        st = DC.chain_status(rec, run=FakeSlurm(states={"1001": "COMPLETED"}))
        assert st["active"] and not st["done"] and st["failed_stage"] is None

    def test_stages_an_earlier_failure_stopped_are_inferred_cancelled(self, tmp_path):
        r, rec, _ = _armed(tmp_path)
        _pass(rec, ("smoke", "1002"), ok=False)
        st = DC.chain_status(rec, run=FakeSlurm(states={"1001": "COMPLETED", "1002": "FAILED"}))
        assert st["failed_stage"] == "smoke" and not st["active"]
        assert st["jobs"][2]["inferred"].startswith("CANCELLED (inferred")
        assert DC.chain_records(rec)["first_unpassed"] == "smoke"

    def test_a_failed_report_is_not_the_failed_stage(self, tmp_path):
        r, rec, _ = _armed(tmp_path)
        _pass(rec, ("smoke", "1002"), ("y2015", "1003"), ("y2016", "1004"))
        st = DC.chain_status(rec, run=FakeSlurm(states={**ALL_DONE, "1005": "FAILED"}))
        assert st["failed_stage"] is None and st["years_passed"] == [2015, 2016] and st["ok"]

    def test_a_failed_case_stage_with_its_record(self, tmp_path):
        r, rec, _ = _armed(tmp_path)
        _pass(rec, ("case", "1001"), ok=False)
        assert DC.chain_records(rec)["first_unpassed"] == "case"
        st = DC.chain_status(rec, run=FakeSlurm(states={**ALL_DONE, "1002": "GONE",
                                                         "1003": "GONE", "1004": "GONE"}))
        assert st["failed_stage"] == "case" and st["done"]

    def test_sacct_is_asked_from_the_day_before_the_chain_was_armed(self, tmp_path):
        """sacct without -S looks back only to midnight, and any fixed window
        ends one day; the chain's own arming date never does."""
        r, rec, _ = _armed(tmp_path)
        armed = json.loads((rec / DC.CHAIN_FILE).read_text())["segments"][0]["armed"]
        slurm = FakeSlurm(states=ALL_DONE)
        DC.chain_status(rec, run=slurm)
        sacct = [c for c in slurm.calls if c[0] == "sacct"][0]
        since = datetime.date.fromisoformat(sacct[sacct.index("-S") + 1])
        assert since == datetime.date.fromisoformat(armed[:10]) - datetime.timedelta(days=1)
        lone = FakeSlurm(states={"5": "COMPLETED"})
        assert DC.slurm_state("5", run=lone) == "COMPLETED"
        s = lone.calls[-1][lone.calls[-1].index("-S") + 1]
        assert (datetime.date.today() - datetime.date.fromisoformat(s)).days == \
            DC.SACCT_LOOKBACK_DAYS

    def test_the_first_format_of_the_record_still_reads(self, tmp_path):
        """The proof chain of 2026-10-22 wrote one arming without segments."""
        rec = tmp_path / "rec"
        (rec / "checks").mkdir(parents=True)
        (rec / DC.CHAIN_FILE).write_text(json.dumps({"armed": "2026-10-22T23:09:12", "ok": True,
                                                     "jobs": [{"stage": "smoke", "job_id": "2"},
                                                              {"stage": "y2015", "job_id": "3"}]}))
        _pass(rec, ("smoke", "2"), ("y2015", "3"))
        recs = DC.chain_records(rec)
        assert recs["years_passed"] == [2015] and recs["segments"] == 1
