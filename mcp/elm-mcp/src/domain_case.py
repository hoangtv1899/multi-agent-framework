#!/usr/bin/env python3
"""One whole-domain ELM case, built from a recorded recipe and run as a chain
of jobs, each gated before the next may start.

A whole-domain experiment is one CIME case, not an ensemble of columns: the
case comes from a recipe that records every setting, and the run is a chain
of SLURM jobs, one model year each, so every job fits the 2-hour short queue
and every year is checked before the next one starts.

  Recipe              every setting of the case: the E3SM checkout, compset,
                      grid, machine, create_newcase options, CIME XML settings
                      in order, user_nl_* lines verbatim, other user_* files
                      and SourceMods with their fingerprints, the input set,
                      the run's start date (1 January: one stage, one year).
  recipe_from_case    the recipe of an existing case, from its own records
                      (env_case.xml, README.case, CaseStatus, user_* files,
                      SourceMods). A clone is refused: its parent's settings
                      are not in its records.
  source_state        the checkout's commit and a fingerprint of everything a
                      build compiles that git does not hold (local changes,
                      submodule changes, untracked files).
  write_case_script   create_newcase ... case.setup, preview_namelists, as a
                      script that refuses to touch an existing case.
  write_stage_script  the one job script every stage of the chain runs.
  plan_chain          the jobs: case (create and build), smoke, one per year,
                      report. Refuses years the forcing does not cover.
  submit_chain        sbatch them in order, afterok with --kill-on-invalid-dep
                      along the chain (a failed stage cancels everything after
                      it), the report afterany on every job, so it runs
                      whatever happens. Every job id is recorded in chain.json
                      (one segment per arming, so a re-armed chain keeps the
                      years that passed before) and STATUS.log.
  chain_status        each stage's SLURM state and check verdict.
  chain_records       the same from the files alone, for a finalize that runs
                      after every job has ended.

Ported from the hand scripts of the Naches ELM-BGC experiment (create_case.sh,
run_stage.sh, arm_fix_short.sh, 2026-10-12/14) and corrected by a review of
the first proof chain (2026-10-23). What they learned is built in:

  - Every job sets LANG/LC_ALL=en_US.UTF-8: an inherited C.UTF-8 locale kills
    the Intel compiler on CentOS 7 (job 788213 died of it).
  - case.submit --no-batch exits 0 when the model aborts, so every model stage
    ends with run_gate.py on its own job id, and the gate's verdict is the
    stage's exit code. The gate runs even when the model step failed, so a
    record names the failure. The case stage writes a record too.
  - The first model year starts from finidat (CONTINUE_RUN=FALSE at the
    recipe's start date); every later year continues from the restart the
    year before wrote, which only afterok can guarantee exists.
  - SourceMods and user files are copied into the record and checked against
    their sha256 when copied and again when the case is built.
  - A job cancelled while it waited is known to neither squeue nor sacct on
    Compy, and sacct without -S looks back only to midnight; the state of
    such a job is inferred from the chain, never taken to mean "running".

A record directory holds what the chain writes: the recipe, the two scripts,
chain.json, STATUS.log, checks/<stage>_<job id>.json and the job logs.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

LOCALE = {"LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8"}
RECIPE_FILE, CHAIN_FILE, STATUS_FILE = "domain_recipe.json", "chain.json", "STATUS.log"
CASE_SCRIPT, STAGE_SCRIPT = "create_case.sh", "run_stage.sh"
# A run of an experiment: what its jobs check against its control
# (domain_changes.expectations), frozen into the record at arming.
EXPECT_FILE = "expectations.json"
# Settings changed per stage by the chain itself, never part of a recipe.
RUN_CONTROL = {"CONTINUE_RUN", "STOP_OPTION", "STOP_N", "STOP_DATE", "REST_OPTION", "REST_N",
               "RESUBMIT"}
# Settings that name a case's own folders: a copy must get its own, or it would
# build into, or run in, the folders of the case it was copied from.
LOCATIONS = {"EXEROOT", "RUNDIR", "CIME_OUTPUT_ROOT", "DOUT_S_ROOT", "CASEROOT", "LOGDIR",
             "CASETOOLS", "CASEBUILD", "SCRIPTSROOT", "LIBROOT", "INCROOT", "SHAREDLIBROOT"}
# create_newcase options a recipe reproduces (canonical names), and how CIME
# spells them (cime/CIME/scripts/create_newcase.py).
_CREATE_OPTS = {"--case": "case", "-case": "case", "--compset": "compset", "-compset": "compset",
                "--res": "res", "-res": "res", "--machine": "machine", "-mach": "machine",
                "--compiler": "compiler", "-compiler": "compiler", "--mpilib": "mpilib",
                "-mpilib": "mpilib", "--project": "project", "-project": "project",
                "--pecount": "pecount", "-pecount": "pecount", "--walltime": "walltime",
                "--queue": "queue", "-q": "queue", "--srcroot": "srcroot",
                "--handle-preexisting-dirs": "handle-preexisting-dirs",
                "--run-unsupported": "run-unsupported"}
_CREATE_FLAGS = {"run-unsupported"}
_CREATE_PASS = ("mpilib", "pecount", "walltime", "queue", "run-unsupported")
# Every long option create_newcase has, so a typed abbreviation (--mach) is
# read as argparse read it, and an option the recipe cannot reproduce is named.
_CREATE_LONG = ("--case", "--compset", "--res", "--machine", "--compiler", "--multi-driver",
                "--ninst", "--mpilib", "--project", "--pecount", "--user-mods-dirs",
                "--user-mods-dir", "--pesfile", "--gridfile", "--workflow", "--srcroot",
                "--output-root", "--script-root", "--run-unsupported", "--test", "--walltime",
                "--queue", "--handle-preexisting-dirs", "--input-dir", "--driver",
                "--non-local", "--extra-machines-dir", "--case-group", "--ngpus-per-node")
_XML_OPTS = {"subgroup", "append"}
# The gate this server ships, run by every model stage.
RUN_GATE = Path(__file__).resolve().parent / "run_gate.py"
# MPI tasks per node, for the job's node count (Compy: 40 cores per node).
TASKS_PER_NODE = 40


class DomainCaseError(Exception):
    """A recipe, a case or a chain cannot be used, and the message says why."""


# ─────────────────────────────────────────────────────────────────────────────
# THE RECIPE
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Recipe:
    """Every setting of one whole-domain case. JSON-ready (asdict).

    xml            (name, value) or (name, value, {"subgroup": s, "append": True})
                   applied with xmlchange, in order, before case.setup
    user_nl        {"elm": [lines], "datm": [...]}, written verbatim
    user_files     {"user_datm.streams.txt.X": {"from": path, "sha256": hex}}
    sourcemods     {"src.elm/PhenologyMod.F90": {"from": path, "sha256": hex}}
    create_options create_newcase options beyond the four named fields
                   (mpilib, pecount, walltime, queue, run-unsupported)
    start          RUN_STARTDATE of the first model year: YYYY-01-01
    copied_from    the case the recipe was read from; never the case it builds
    """
    case: str
    case_root: str
    srcroot: str
    compset: str
    res: str = "ELM_USRDAT"
    mach: str = "compy"
    compiler: str = "intel"
    project: str = "e3sm"
    xml: List[Tuple] = field(default_factory=list)
    user_nl: Dict[str, List[str]] = field(default_factory=dict)
    sourcemods: Dict[str, Dict[str, str]] = field(default_factory=dict)
    start: str = ""
    notes: List[str] = field(default_factory=list)
    source: Dict = field(default_factory=dict)       # source_state() when written
    user_files: Dict[str, Dict[str, str]] = field(default_factory=dict)
    create_options: Dict[str, Any] = field(default_factory=dict)
    copied_from: str = ""

    @property
    def case_dir(self) -> Path:
        return Path(self.case_root) / self.case

    @property
    def first_year(self) -> int:
        return int(self.start[:4])

    def setting(self, name: str) -> Optional[str]:
        """The last value the recipe gives an XML setting, or None."""
        vals = [e[1] for e in self.xml if e[0] == name]
        return vals[-1] if vals else None

    def namelist(self, component: str = "elm") -> Dict[str, list]:
        """The user_nl_<component> keys and values, parsed. Indexed entries
        (hist_nhtfrq(2) = -24, valid in user_nl) are left out: no key read
        from here is ever set that way."""
        from run_gate import GateError, read_namelist
        import tempfile
        body = [l for l in self.user_nl.get(component, [])
                if not re.match(r"\s*[A-Za-z_]\w*\s*\(", l)]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "nl"
            p.write_text("&nl\n" + "\n".join(body) + "\n/\n")
            try:
                return read_namelist(p)
            except GateError as e:
                raise DomainCaseError(f"user_nl_{component}: {e}") from None

    def inputs(self) -> Dict[str, Optional[str]]:
        """The input set the recipe names: surface, domain, finidat."""
        nl = self.namelist("elm")
        dom = None
        if self.setting("LND_DOMAIN_FILE"):
            dom = str(Path(self.setting("LND_DOMAIN_PATH") or "") / self.setting("LND_DOMAIN_FILE"))
        return {"surface": (nl.get("fsurdat") or [None])[0], "domain": dom,
                "finidat": (nl.get("finidat") or [None])[0]}

    def check(self) -> List[str]:
        """What is wrong with this recipe, before anything is written."""
        bad = []
        for k in ("case", "case_root", "srcroot", "compset", "res", "mach", "compiler",
                  "project"):
            if not getattr(self, k):
                bad.append(f"no {k}")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.case or ""):
            bad.append(f"case name {self.case!r} is not a plain file name")
        if self.srcroot and not Path(self.srcroot).is_absolute():
            bad.append(f"srcroot {self.srcroot!r} is not an absolute path")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.start or ""):
            bad.append(f"start {self.start!r} is not YYYY-MM-DD")
        elif not self.start.endswith("-01-01"):
            bad.append(f"start {self.start} is not 1 January: each stage of the chain runs "
                       f"one calendar year")
        if self.copied_from and self.case_root and self.case and \
                Path(self.copied_from).resolve() == self.case_dir.resolve():
            bad.append(f"the recipe would build and run in {self.copied_from}, the case it was "
                       f"copied from: give the copy a new case name")
        for e in self.xml:
            k, v = e[0], e[1]
            if k in RUN_CONTROL:
                bad.append(f"{k} is run control, set by each stage, not by the recipe")
            if k in LOCATIONS:
                bad.append(f"{k} names a case's own folder: a new case gets its own")
            if not re.fullmatch(r"[A-Z0-9_]+", k):
                bad.append(f"XML setting name {k!r}")
            if len(e) > 2 and set(e[2]) - _XML_OPTS:
                bad.append(f"xmlchange options {sorted(set(e[2]) - _XML_OPTS)} for {k}")
        if self.setting("RUN_STARTDATE") not in (None, self.start):
            bad.append(f"RUN_STARTDATE {self.setting('RUN_STARTDATE')} differs from start "
                       f"{self.start}")
        for rel, m in self.sourcemods.items():
            if not re.fullmatch(r"src\.\w+/[\w.-]+", rel):
                bad.append(f"SourceMods path {rel!r}")
            if not m.get("sha256"):
                bad.append(f"SourceMods {rel} has no sha256")
        for name, m in self.user_files.items():
            if not re.fullmatch(r"user_[\w.+-]+", name) or name.startswith("user_nl_"):
                bad.append(f"user file name {name!r}")
            if not m.get("sha256"):
                bad.append(f"user file {name} has no sha256")
        for comp, lines in self.user_nl.items():
            if not re.fullmatch(r"\w+", comp):
                bad.append(f"user_nl component {comp!r}")
            if any(l.strip() == "NL_END_OF_RECIPE" for l in lines):
                bad.append(f"user_nl_{comp} holds the heredoc marker")
        for k in set(self.create_options) - set(_CREATE_PASS):
            bad.append(f"create_newcase option {k!r} is not one a recipe reproduces")
        return bad

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, text: str) -> "Recipe":
        d = json.loads(text)
        d["xml"] = [tuple(p) for p in d.get("xml", [])]
        known = set(cls.__dataclass_fields__)
        extra = sorted(set(d) - known)
        if extra:
            raise DomainCaseError(f"recipe keys {extra} are not recipe fields")
        return cls(**d)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# Untracked files larger than this are fingerprinted by name and size only.
_UNTRACKED_HASH_MAX = 64 << 20


def source_state(srcroot) -> Dict:
    """The E3SM checkout as it is: commit, files changed, untracked files, and
    one fingerprint of everything a build compiles that git does not hold
    (local changes, changes inside submodules such as cime, and untracked
    files, which E3SM's CMake globs and compiles). Two trees that can build
    differently never share the fingerprint.

    git runs FROM the checkout rather than with `git -C`: Compy's login-node
    git is 1.8.3, which has no -C (it reported every tree as no checkout)."""
    root = Path(srcroot)

    def git(*args):
        try:
            p = subprocess.run(["git", *args], capture_output=True, text=True, timeout=300,
                               cwd=str(root))
            return p.returncode, p.stdout, p.stderr
        except Exception as e:                                      # noqa: BLE001
            return 127, "", str(e)
    rc, head, _ = git("rev-parse", "HEAD")
    rc2, short, _ = git("status", "--short", "--untracked-files=no")
    rc3, diff, _ = git("diff", "HEAD")
    rc4, sub, _ = git("submodule", "foreach", "--quiet", "git diff HEAD")
    rc5, others, _ = git("ls-files", "--others", "--exclude-standard")
    if rc or rc2 or rc3 or rc5:
        return {"srcroot": str(root), "error": "not a git checkout, or git failed"}
    untracked = sorted(l for l in others.splitlines() if l.strip())
    h = hashlib.sha256()
    for part in (head.strip(), diff, sub if rc4 == 0 else "submodules unreadable"):
        h.update(part.encode())
        h.update(b"\0")
    for name in untracked:
        p = root / name
        try:
            size = p.stat().st_size
            tag = sha256(p) if size <= _UNTRACKED_HASH_MAX else f"size {size}"
        except OSError:
            tag = "unreadable"
        h.update(f"{name}\0{tag}\0".encode())
    return {"srcroot": str(root), "commit": head.strip(),
            "changed": [l.strip() for l in short.splitlines() if l.strip()],
            "untracked": untracked,
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
            "fingerprint": h.hexdigest()}


_XMLCHANGE = re.compile(r"xmlchange success <command> (\S*xmlchange)\s+(.*?)\s*</command>")
_ENTRY = r'<entry id="{}" value="([^"]*)"'


def _env_value(case_dir: Path, name: str) -> Optional[str]:
    """A setting as the case's env_*.xml files hold it, or None."""
    for p in sorted(case_dir.glob("env_*.xml")):
        m = re.search(_ENTRY.format(re.escape(name)), p.read_text(errors="replace"))
        if m:
            return m.group(1)
    return None


def _create_args(line: str) -> Dict[str, Any]:
    """create_newcase's arguments as README.case records them, with CIME's
    spellings and argparse's abbreviations of long ones. The record is
    " ".join(sys.argv), so it is split on whitespace: any quotes the author
    typed were spent by their shell, and a quote left in it is literal."""
    argv = line.split()
    out: Dict[str, Any] = {}
    k = 1
    while k < len(argv):
        tok = argv[k]
        if not tok.startswith("-"):
            raise DomainCaseError(f"README.case: create_newcase argument {tok!r} is not an option")
        name, val = (tok.split("=", 1) + [None])[:2] if tok.startswith("--") else (tok, None)
        if name not in _CREATE_OPTS and name.startswith("--"):
            hits = [o for o in _CREATE_LONG if o.startswith(name)]
            name = hits[0] if len(hits) == 1 else name
        if name not in _CREATE_OPTS:
            raise DomainCaseError(f"README.case: create_newcase {name} is not reproduced by a "
                                  f"recipe (it reproduces case, compset, res, machine, "
                                  f"compiler, project, {', '.join(_CREATE_PASS)})")
        key = _CREATE_OPTS[name]
        if key in _CREATE_FLAGS:
            out[key] = True
            k += 1
            continue
        if val is None:
            if k + 1 >= len(argv):
                raise DomainCaseError(f"README.case: create_newcase {name} has no value")
            val = argv[k + 1]
            k += 2
        else:
            k += 1
        out[key] = val
    return out


def _xmlchange_entries(args: str) -> List[Tuple]:
    """One xmlchange command's settings, as CIME's xmlchange reads them:
    NAME=VALUE[,NAME=VALUE] split at every comma not escaped with a backslash
    (the escape stays in the value, as CIME keeps it), or --id NAME --val
    VALUE, with --subgroup and --append kept, --force dropped and anything
    else refused. Split on whitespace, as _create_args is."""
    argv = args.split()
    opts: Dict[str, Any] = {}
    pairs: List[Tuple[str, str]] = []
    ident = None
    k = 0
    while k < len(argv):
        tok = argv[k]
        if tok in ("--force", "-force", "-f", "--noecho", "-noecho"):
            k += 1
        elif tok in ("--append", "-append"):
            opts["append"] = True
            k += 1
        elif tok in ("--subgroup", "-subgroup") and k + 1 < len(argv):
            opts["subgroup"] = argv[k + 1]
            k += 2
        elif tok in ("--id", "-id") and k + 1 < len(argv):
            ident = argv[k + 1]
            k += 2
        elif tok in ("--val", "-val") and k + 1 < len(argv):
            if not ident:
                raise DomainCaseError(f"xmlchange --val without --id in {args!r}")
            pairs.append((ident, argv[k + 1]))
            ident = None
            k += 2
        elif tok.startswith("-"):
            raise DomainCaseError(f"xmlchange option {tok} in {args!r} is not reproduced by a "
                                  f"recipe")
        else:
            for part in re.split(r"(?<!\\),", tok):
                if "=" not in part:
                    raise DomainCaseError(f"cannot read the xmlchange argument {args!r}")
                name, v = part.split("=", 1)
                pairs.append((name, v))
            k += 1
    return [(n, v, dict(opts)) if opts else (n, v) for n, v in pairs]


def recipe_from_case(case_dir, case: Optional[str] = None,
                     case_root: Optional[str] = None) -> Recipe:
    """The recipe of an existing case, from the case's own records.

    env_case.xml gives the E3SM checkout (absolute, whatever was typed);
    README.case the create_newcase command; CaseStatus every xmlchange in
    order; the user_* files and SourceMods the case's modifications. XML
    changes before case.setup are the recipe; later ones are run control
    (set by each stage, left out) or settings (kept, with a note). Run
    control and the case's own folders are left out wherever they appear,
    each with a note. A clone is refused. case / case_root name the copy; a
    recipe that names its own source case is refused by Recipe.check.
    """
    cd = Path(case_dir)
    readme = cd / "README.case"
    if not readme.is_file():
        raise DomainCaseError(f"{cd} has no README.case: not a CIME case")
    lines = readme.read_text(errors="replace").splitlines()
    first = next((l for l in lines if re.search(r"create_(newcase|clone)\b", l)), None)
    if not first:
        raise DomainCaseError(f"{readme} does not record create_newcase")
    if re.search(r"create_clone\b", first):
        parent = re.search(r"--clone(?:=|\s+)(\S+)", first)
        raise DomainCaseError(
            f"{cd.name} is a clone{' of ' + parent.group(1) if parent else ''}: the settings "
            f"its parent made are not in its own records. Read the parent's recipe and add the "
            f"clone's own changes.")
    cmd = first.split(": ", 1)[1] if re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d: ", first) \
        else first
    args = _create_args(cmd)
    typed = Path(cmd.split()[0])
    srcroot = _env_value(cd, "SRCROOT") or args.get("srcroot") or (
        str(typed.parents[2]) if typed.is_absolute() and typed.parts[-3:] ==
        ("cime", "scripts", "create_newcase") else "")
    if args.get("srcroot") and Path(args["srcroot"]).resolve() != Path(srcroot).resolve():
        raise DomainCaseError(f"README.case's --srcroot {args['srcroot']} is not the case's "
                              f"SRCROOT {srcroot}")
    xml: List[Tuple] = []
    notes: List[str] = [f"from the case records of {cd.name}"]
    left_out: List[str] = []
    setup_seen = False
    status = cd / "CaseStatus"
    for line in (status.read_text(errors="replace").splitlines() if status.is_file() else []):
        if "case.setup success" in line:
            setup_seen = True
        m = _XMLCHANGE.search(line)
        if not m:
            continue
        kept = []
        for e in _xmlchange_entries(m.group(2)):
            if e[0] in RUN_CONTROL or e[0] in LOCATIONS:
                left_out.append(e[0])
            elif setup_seen and e[0] == "RUN_STARTDATE":
                left_out.append(e[0])
            else:
                kept.append(e)
        xml += kept
        if setup_seen and kept:
            notes.append(f"set after case.setup: {[tuple(e[:2]) for e in kept]}")
    if left_out:
        notes.append(f"left out (run control, set by each stage, or the case's own folders): "
                     f"{sorted(set(left_out))}")
    user_nl, user_files = {}, {}
    for p in sorted(cd.glob("user_*")):
        if not p.is_file():
            continue
        if p.name.startswith("user_nl_"):
            body = p.read_text(errors="replace").splitlines()
            if any(l.strip() and not l.strip().startswith("!") for l in body):
                user_nl[p.name[len("user_nl_"):]] = body
        else:
            user_files[p.name] = {"from": str(p), "sha256": sha256(p)}
    mods = {}
    for p in sorted(cd.glob("SourceMods/src.*/*")):
        if p.is_file() and p.name != "README":
            mods[f"{p.parent.name}/{p.name}"] = {"from": str(p), "sha256": sha256(p)}
    start = dict((e[0], e[1]) for e in xml).get("RUN_STARTDATE") or \
        _env_value(cd, "RUN_STARTDATE") or ""
    if not args.get("project") and _env_value(cd, "PROJECT"):
        args["project"] = _env_value(cd, "PROJECT")
    return Recipe(case=case or cd.name, case_root=case_root or str(cd.parent), srcroot=srcroot,
                  compset=args.get("compset") or _env_value(cd, "COMPSET") or "",
                  res=args.get("res", ""),
                  mach=args.get("machine") or _env_value(cd, "MACH") or "",
                  compiler=args.get("compiler") or _env_value(cd, "COMPILER") or "",
                  project=args.get("project", ""), xml=xml, user_nl=user_nl,
                  sourcemods=mods, start=start, notes=notes, user_files=user_files,
                  create_options={k: args[k] for k in _CREATE_PASS if k in args},
                  copied_from=str(cd))


# ─────────────────────────────────────────────────────────────────────────────
# THE SCRIPTS
# ─────────────────────────────────────────────────────────────────────────────
def _q(x) -> str:
    return shlex.quote(str(x))


def _keep_copy(kind: str, name: str, m: Dict[str, str], dest: Path) -> None:
    """Copy a case modification into the record, refusing a changed file."""
    src = Path(m["from"])
    if src.resolve() == dest.resolve():
        return
    if not src.is_file():
        raise DomainCaseError(f"{kind} {name}: {src} is missing")
    if sha256(src) != m["sha256"]:
        raise DomainCaseError(f"{kind} {name}: {src} is not the file the recipe recorded "
                              f"(sha256 differs)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())


def write_case_script(recipe: Recipe, record_dir) -> Path:
    """create_newcase, the XML settings, user_nl_*, the other user files,
    SourceMods, case.setup and preview_namelists. Not run here: the chain's
    first job runs it, so the login node never builds.

    The SourceMods and user files are copied INTO the record now, each checked
    against its sha256, and the case is built from those copies (checked
    again): the record stands without the case they came from."""
    bad = recipe.check()
    if bad:
        raise DomainCaseError("the recipe cannot be used: " + "; ".join(bad))
    rd = Path(record_dir)
    rd.mkdir(parents=True, exist_ok=True)
    for rel, m in recipe.sourcemods.items():
        _keep_copy("SourceMods", rel, m, rd / "SourceMods" / rel)
    for name, m in recipe.user_files.items():
        _keep_copy("user file", name, m, rd / "user_files" / name)
    cd = recipe.case_dir
    create = [_q(Path(recipe.srcroot) / "cime" / "scripts" / "create_newcase"),
              "--case", '"$CASEDIR"', "--compset", _q(recipe.compset), "--res", _q(recipe.res),
              "--machine", _q(recipe.mach), "--compiler", _q(recipe.compiler),
              "--project", _q(recipe.project)]
    for k in _CREATE_PASS:
        if k in recipe.create_options:
            create += [f"--{k}"] if k in _CREATE_FLAGS else [f"--{k}",
                                                             _q(recipe.create_options[k])]
    lines = ["#!/bin/bash",
             f"# Create case {recipe.case} from {rd / RECIPE_FILE}.",
             "# Written by domain_case.write_case_script; refuses an existing case.",
             "set -euo pipefail",
             "export " + " ".join(f"{k}={v}" for k, v in LOCALE.items()),
             f"CASEDIR={_q(cd)}",
             '[ -e "$CASEDIR" ] && { echo "case exists: $CASEDIR" >&2; exit 1; }',
             " ".join(create),
             'cd "$CASEDIR"']
    for e in recipe.xml:
        opts = e[2] if len(e) > 2 else {}
        flags = (["--subgroup", _q(opts["subgroup"])] if opts.get("subgroup") else []) + \
            (["--append"] if opts.get("append") else [])
        # xmlchange splits NAME=VALUE at every comma not escaped; it never
        # splits --val, so a value holding a comma goes that way, unchanged.
        setting = ["--id", _q(e[0]), "--val", _q(e[1])] if "," in str(e[1]) else \
            [_q(f"{e[0]}={e[1]}")]
        lines.append(" ".join(["./xmlchange", *flags, *setting]))
    for comp, body in recipe.user_nl.items():
        lines.append(f"cat > user_nl_{comp} <<'NL_END_OF_RECIPE'")
        lines += body
        lines.append("NL_END_OF_RECIPE")
    for name, m in recipe.user_files.items():
        lines.append(f"cp {_q(rd / 'user_files' / name)} {_q(name)}")
        lines.append(f"echo {_q(m['sha256'] + '  ' + name)} | sha256sum -c -")
    for rel, m in recipe.sourcemods.items():
        lines.append(f"mkdir -p SourceMods/{_q(Path(rel).parent)}")
        lines.append(f"cp {_q(rd / 'SourceMods' / rel)} SourceMods/{_q(rel)}")
        lines.append(f"echo {_q(m['sha256'] + '  SourceMods/' + rel)} | sha256sum -c -")
    lines += ["./case.setup", "./preview_namelists", f'echo "case ready: $CASEDIR"', ""]
    out = rd / CASE_SCRIPT
    out.write_text("\n".join(lines))
    out.chmod(0o755)
    if not recipe.source:
        recipe.source = source_state(recipe.srcroot)
    (rd / RECIPE_FILE).write_text(recipe.to_json())
    return out


@dataclass
class Reference:
    """A reference run for the year gate: its run directory (or file glob),
    the stream to compare, and the years whose comparison is gated (the rest
    are reported only, for experiments expected to differ)."""
    run_dir: str
    stream: str = "h1"
    pattern: Optional[str] = None
    gate_years: List[int] = field(default_factory=list)


def _nodes(recipe: Recipe, per_node: int = TASKS_PER_NODE) -> int:
    """Nodes for the job, from the recipe's task counts (CIME: a negative
    count is a number of nodes)."""
    n = 1
    for e in recipe.xml:
        if re.fullmatch(r"NTASKS(_[A-Z]+)?", e[0]):
            try:
                t = int(e[1])
            except ValueError:
                continue
            n = max(n, -t if t < 0 else math.ceil(t / per_node))
    return n


def case_stage_record(record_dir, job: str, rc: int, srcroot) -> Path:
    """checks/case_<job>.json: the case stage's verdict and the source state
    the case was built from (called by the stage script after the build)."""
    rd = Path(record_dir) / "checks"
    rd.mkdir(parents=True, exist_ok=True)
    src = source_state(srcroot)
    rec = {"stage": "case", "job": str(job), "rc": int(rc), "pass": int(rc) == 0,
           "reasons": [] if int(rc) == 0 else [f"the case stage exited {rc}"],
           "source": src, "time": datetime.datetime.now().isoformat(timespec="seconds")}
    out = rd / f"case_{job}.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(out)
    print(f"source {src.get('commit')} fingerprint {src.get('fingerprint')}"
          + (f" ({src['error']})" if src.get("error") else ""))
    return out


def write_stage_script(recipe: Recipe, record_dir, *, smoke_days: int = 5,
                       reference: Optional[Reference] = None,
                       python: str = sys.executable,
                       expect: Optional[Dict] = None) -> Path:
    """The job script every stage of the chain runs: run_stage.sh <stage> [args].

    Stages: case (create and build, with its own check record), smoke
    (smoke_days from the start, gated on the logs, the requested fields and
    the carry-over from finidat), y<YEAR> (one model year, gated on the logs,
    the year's model facts and, for a gated year, the reference run), report
    (sacct lines of the given job ids into STATUS.log). Every stage appends to
    STATUS.log, and a non-zero exit cancels the stages queued after it.

    The case script's sha256 is written into this script, which SLURM keeps
    as it was at submission: a case script changed after the chain was armed
    is refused by the case stage.

    A RUN OF AN EXPERIMENT passes `expect` (domain_changes.expectations): it is
    written to expectations.json beside the scripts, its sha256 frozen here
    too, and two checks hold the run to its changes. The case stage compares
    the run's resolved namelists with its control's before building the model
    (run_gate.py settings); every year stage checks the changes' output rules
    (run_gate.py year --expect).
    """
    rd = Path(record_dir)
    if re.search(r"\s", str(rd)):
        raise DomainCaseError(f"the record directory {rd} holds whitespace, which a #SBATCH "
                              f"line cannot carry")
    (rd / "logs").mkdir(parents=True, exist_ok=True)
    (rd / "checks").mkdir(parents=True, exist_ok=True)
    case_script = rd / CASE_SCRIPT
    case_sha = sha256(case_script) if case_script.is_file() else "none"
    expect_sha, expect_args, settings = "none", "", ":"
    if expect is not None:
        if not expect.get("control_case_dir"):
            raise DomainCaseError("a run's expectations name no control_case_dir to "
                                  "compare its settings with")
        _write_json(rd / EXPECT_FILE, expect)
        expect_sha = sha256(rd / EXPECT_FILE)
        expect_args = f' --expect "$REC/{EXPECT_FILE}"'
        settings = (f'expect_ok && {{ "$PY" "$GATE" settings --case-dir "$CASEDIR" '
                    f'--control-case-dir {_q(expect["control_case_dir"])} '
                    f'--expect "$REC/{EXPECT_FILE}" --job "${{SLURM_JOB_ID:-none}}" '
                    f'--out "$REC/checks" --status "$REC/{STATUS_FILE}" --stage settings '
                    f'|| note $?; }}')
    s0 = recipe.start
    first = f"{recipe.first_year:04d}"
    ref = ""
    if reference:
        items = ["--reference", reference.run_dir, "--reference-stream", reference.stream]
        if reference.pattern:
            items += ["--reference-pattern", reference.pattern]
        gated = " ".join(f"{int(y):04d}" for y in reference.gate_years)
        ref = (f"REF=({' '.join(_q(x) for x in items)})\n"
               f"          case \" {gated} \" in *\" $Y \"*) ;; *) REF+=(--reference-report-only) ;; esac")
    year_set = "CONTINUE_RUN={c},STOP_OPTION=nyears,STOP_N=1,REST_OPTION=nyears,REST_N=1"
    # A change that sets each year's settings (weather_repeat_year: which
    # recorded year the forcing reads) gives them in the expectations; the
    # year's job adds them to its xmlchange, the smoke run takes the first
    # year's. CIME writes the namelists anew when each run starts.
    by_year = {int(y): "".join(f",{k}={v}" for k, v in kv)
               for y, kv in ((expect or {}).get("year_settings") or {}).items()}
    for y, extra in by_year.items():
        if not re.fullmatch(r"(,[A-Z][A-Z0-9_]*=[\w.+-]+)*", extra):
            raise DomainCaseError(f"year {y}'s settings {extra!r} are not NAME=value pairs")
        if any(k in RUN_CONTROL for k in re.findall(r",([A-Z][A-Z0-9_]*)=", extra)):
            raise DomainCaseError(f"year {y}'s settings {extra!r} set run control, which "
                                  f"each stage sets itself")
    year_case = 'YS=""'
    if by_year:
        # a year the settings do not name stops before its model runs: it
        # would keep the previous year's settings, which persist in env_run.xml
        year_case += ("\n          case \"$Y\" in\n" + "\n".join(
            f"              {y:04d}) YS={_q(e)} ;;" for y, e in sorted(by_year.items()))
            + '\n              *) log "FAILED no year settings for $Y in the expectations"; '
              'log "finished rc=2"; exit 2 ;;'
            + "\n          esac")
    smoke_extra = by_year.get(recipe.first_year, "")
    src_mod = Path(__file__).resolve().parent
    text = f"""#!/bin/bash
#SBATCH --account={recipe.project}
#SBATCH --nodes={_nodes(recipe)}
#SBATCH --exclusive
#SBATCH --output={rd}/logs/%x.%j.out
# One stage of case {recipe.case}'s chain, as its own SLURM job. Written by
# domain_case.write_stage_script. Every stage appends to STATUS.log; a non-zero
# exit cancels the stages queued after it (afterok + --kill-on-invalid-dep=yes).
#   usage: sbatch -J <name> -p <partition> -t <wall> run_stage.sh <stage> [job ids]
set -uo pipefail
export {" ".join(f"{k}={v}" for k, v in LOCALE.items())}
STAGE=$1; shift
REC={_q(rd)}; CASEDIR={_q(recipe.case_dir)}; PY={_q(python)}; GATE={_q(RUN_GATE)}
CASE_SCRIPT_SHA256={case_sha}; EXPECT_SHA256={expect_sha}
log() {{ echo "$(date '+%F %T') job=${{SLURM_JOB_ID:-none}} stage=$STAGE $*" >> "$REC/{STATUS_FILE}"; }}
log "started on $(hostname) case={recipe.case}"
rc=0
note() {{ [ "$rc" -eq 0 ] && rc=$1; return 0; }}      # keep the first failure
model() {{ ./xmlchange "$1" && ./case.submit --no-batch; }}
gate() {{ "$PY" "$GATE" "$@" --case-dir "$CASEDIR" --run-dir "$RUNDIR" --job "${{SLURM_JOB_ID:-none}}" \\
           --out "$REC/checks" --status "$REC/{STATUS_FILE}" --stage "$STAGE" || note $?; }}
expect_ok() {{ [ "$EXPECT_SHA256" = none ] && return 0
              echo "$EXPECT_SHA256  $REC/{EXPECT_FILE}" | sha256sum -c - >/dev/null 2>&1 && return 0
              log "FAILED $REC/{EXPECT_FILE} is not the file this chain was armed with"; note 3; return 1; }}
into_case() {{ cd "$CASEDIR" || {{ log "FAILED no case dir $CASEDIR"; exit 2; }}
              RUNDIR=$(./xmlquery RUNDIR --value) || {{ log "FAILED xmlquery RUNDIR"; exit 2; }}; }}
case $STAGE in
  case)   if ! echo "$CASE_SCRIPT_SHA256  $REC/{CASE_SCRIPT}" | sha256sum -c - >/dev/null 2>&1; then
              log "FAILED $REC/{CASE_SCRIPT} is not the script this chain was armed with"; note 3
          else
              bash "$REC/{CASE_SCRIPT}" || note $?
              if [ "$rc" -eq 0 ]; then {settings}; fi
              if [ "$rc" -eq 0 ]; then (cd "$CASEDIR" && ./case.build) || note $?; fi
          fi
          log "$("$PY" -c 'import sys; sys.path.insert(0, sys.argv[1]); import domain_case as D; D.case_stage_record(*sys.argv[2:6])' {_q(src_mod)} "$REC" "${{SLURM_JOB_ID:-none}}" "$rc" {_q(recipe.srcroot)} 2>&1 | tail -1)" ;;
  smoke)  into_case
          model "CONTINUE_RUN=FALSE,RUN_STARTDATE={s0},STOP_OPTION=ndays,STOP_N={smoke_days},REST_OPTION=ndays,REST_N={smoke_days}{smoke_extra}" || note $?
          gate smoke --days {smoke_days} ;;
  y[0-9][0-9][0-9][0-9])
          into_case
          Y=${{STAGE#y}}
          {year_case}
          if [ "$Y" = "{first}" ]; then
              model "{year_set.format(c="FALSE")},RUN_STARTDATE={s0}$YS" || note $?
          else
              model "{year_set.format(c="TRUE")}$YS" || note $?
          fi
          REF=()
          {ref}
          expect_ok
          gate year --year "$Y" ${{REF[@]+"${{REF[@]}}"}}{expect_args} ;;
  report) for j in "$@"; do
              sacct -j "$j" -X -n -P -S "$(date -d '{SACCT_LOOKBACK_DAYS} days ago' +%F)" \\
                    -o JobID,JobName,State,Elapsed,ExitCode | \\
              while IFS='|' read -r id name st el ex; do
                  log "REPORT $name job=$id state=$st elapsed=$el exit=$ex"; done
              [ -n "$(sacct -j "$j" -X -n -P -S "$(date -d '{SACCT_LOOKBACK_DAYS} days ago' +%F)" -o JobID)" ] || \\
                  log "REPORT job=$j state=UNKNOWN (no accounting record: cancelled before it started)"
          done ;;
  *)      log "FAILED unknown stage"; rc=2 ;;
esac
log "finished rc=$rc"
exit $rc
"""
    out = rd / STAGE_SCRIPT
    out.write_text(text)
    out.chmod(0o755)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# THE CHAIN
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Job:
    stage: str
    name: str
    partition: str
    walltime: str
    job_id: Optional[str] = None
    after: Optional[str] = None          # the stage it waits on (afterok)


def _stage(year: int) -> str:
    return f"y{int(year):04d}"


def forcing_problems(recipe: Recipe, years: Sequence[int]) -> List[str]:
    """Model years the recipe's forcing does not cover. With the CLMNCEP
    family of DATM modes the model year y reads data year START + (y - ALIGN)
    modulo the span, so a year outside [ALIGN, ALIGN + span) silently
    recycles earlier weather (datm's default taxmode cycles)."""
    s, e = recipe.setting("DATM_CLMNCEP_YR_START"), recipe.setting("DATM_CLMNCEP_YR_END")
    if not (s and e):
        return []
    s, e = int(s), int(e)
    align = int(recipe.setting("DATM_CLMNCEP_YR_ALIGN") or s)
    span = e - s + 1
    out = []
    for y in years:
        if not 0 <= int(y) - align < span:
            data = s + (int(y) - align) % span
            out.append(f"model year {y} would read the {data} weather: the forcing covers "
                       f"model years {align}-{align + span - 1} (data {s}-{e}) and cycles "
                       f"beyond them")
    return out


def plan_chain(recipe: Recipe, years: Sequence[int], *, tag: Optional[str] = None,
               with_case: bool = True, smoke: bool = True, partition: str = "short",
               case_wall: str = "01:00:00", smoke_wall: str = "00:45:00",
               year_wall: str = "01:55:00", report_wall: str = "00:10:00",
               allow_forcing_cycle: bool = False) -> List[Job]:
    """case > smoke > y<first> > ... > report, each waiting on the one before.

    years must be consecutive, inside 1-9999 and inside the forcing's years
    (unless allow_forcing_cycle), and, unless the case is new, may start after
    the recipe's first year (a chain re-armed after a failed year).
    """
    ys = [int(y) for y in years]
    if not ys:
        raise DomainCaseError("a chain needs at least one model year")
    if ys != list(range(ys[0], ys[0] + len(ys))):
        raise DomainCaseError(f"years {ys} are not consecutive: each year continues "
                              f"from the restart of the one before")
    if ys[0] < 1 or ys[-1] > 9999:
        raise DomainCaseError(f"years {ys[0]}-{ys[-1]} are outside 1-9999")
    if ys[0] < recipe.first_year:
        raise DomainCaseError(f"{ys[0]} is before the recipe's start {recipe.start}")
    if (with_case or smoke) and ys[0] != recipe.first_year:
        raise DomainCaseError(f"a chain that creates the case or runs the smoke starts at "
                              f"{recipe.first_year}, not {ys[0]}")
    cyc = forcing_problems(recipe, ys)
    if cyc and not allow_forcing_cycle:
        raise DomainCaseError("; ".join(cyc))
    tag = tag or recipe.case
    jobs: List[Job] = []
    if with_case:
        jobs.append(Job("case", f"{tag}_case", partition, case_wall))
    if smoke:
        jobs.append(Job("smoke", f"{tag}_smoke", partition, smoke_wall))
    for y in ys:
        jobs.append(Job(_stage(y), f"{tag}_{_stage(y)}", partition, year_wall))
    for k in range(1, len(jobs)):
        jobs[k].after = jobs[k - 1].stage
    jobs.append(Job("report", f"{tag}_report", partition, report_wall))
    return jobs


# The chain spec a framework writes to 01_inputs/domain_chain.json, and its
# parts: anything else is refused, so a misspelt key cannot vanish.
CHAIN_SPEC_KEYS = {"years", "smoke_days", "smoke", "with_case", "partition", "tag",
                   "reference", "walls", "allow_forcing_cycle"}
WALL_KEYS = {"case_wall", "smoke_wall", "year_wall", "report_wall"}
REFERENCE_KEYS = {"run_dir", "stream", "pattern", "gate_years"}


def chain_spec_problems(spec: Dict) -> List[str]:
    """What is wrong with a chain spec, before it is used."""
    out = []
    if not isinstance(spec, dict):
        return ["the chain spec is not an object"]
    for k in sorted(set(spec) - CHAIN_SPEC_KEYS):
        out.append(f"chain spec key {k!r} is not one of {sorted(CHAIN_SPEC_KEYS)}")
    for k in sorted(set(spec.get("walls") or {}) - WALL_KEYS):
        out.append(f"walls key {k!r} is not one of {sorted(WALL_KEYS)}")
    ref = spec.get("reference")
    if ref is not None:
        if not isinstance(ref, dict) or not ref.get("run_dir"):
            out.append("reference needs a run_dir")
        else:
            for k in sorted(set(ref) - REFERENCE_KEYS):
                out.append(f"reference key {k!r} is not one of {sorted(REFERENCE_KEYS)}")
    ys = spec.get("years")
    if ys is not None and (not isinstance(ys, list) or not all(isinstance(y, int) for y in ys)):
        out.append("years must be a list of whole years")
    return out


def _run(argv: Sequence[str], timeout: float = 120) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except Exception as e:                                          # noqa: BLE001
        return 127, "", f"{type(e).__name__}: {e}"


def node_local(path) -> Optional[str]:
    """Why a path is on a node-local filesystem a compute node cannot see, or
    None. Jobs whose script or record sits there die in seconds with empty
    logs (770938/770939)."""
    try:
        dev = subprocess.check_output(["df", "-P", str(path)], text=True,
                                      timeout=20).splitlines()[-1].split()[0]
    except Exception:                                               # noqa: BLE001
        return None
    return (f"{path} is on {dev}, a node-local filesystem a compute node cannot see"
            if dev.startswith("/dev/") else None)


def _status_line(record_dir, text: str) -> None:
    with open(Path(record_dir) / STATUS_FILE, "a") as f:
        f.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {text}\n")


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def read_chain(record_dir) -> Dict:
    """chain.json as {"segments": [...]}; a single-arming record (the first
    format) reads as one segment. Raises when the file cannot be read."""
    p = Path(record_dir) / CHAIN_FILE
    if not p.is_file():
        raise DomainCaseError(f"no {CHAIN_FILE} in {record_dir}: the chain was never armed")
    try:
        d = json.loads(p.read_text())
    except ValueError as e:
        raise DomainCaseError(f"{p} cannot be read ({e}): whether a chain runs here is "
                              f"unknown") from None
    if "segments" not in d:
        d = {"segments": [d]}
    return d


def submit_chain(jobs: List[Job], record_dir, run: Callable = None,
                 after_job: Optional[str] = None) -> Dict:
    """sbatch the chain in order. A record of this arming (a segment of
    chain.json, kept beside any earlier ones).

    afterok + --kill-on-invalid-dep=yes along the chain; the report waits on
    every job with afterany. If any submission fails, the jobs already queued
    are cancelled and the record says which cancellations SLURM accepted and
    which it did not. chain.json is replaced atomically after every job, so a
    queued job is never left unrecorded by a half-written file. A record on a
    node-local filesystem is refused before anything is submitted.

    after_job: a job outside this chain the first job must also wait for
    (afterok), such as the case job of the control a run of an experiment
    compares its settings with.
    """
    run = run or _run
    rd = Path(record_dir)
    script = rd / STAGE_SCRIPT
    if not script.is_file():
        raise DomainCaseError(f"{script} is missing: write_stage_script first")
    local = node_local(rd)
    if local:
        raise DomainCaseError(local)
    if after_job is not None and not str(after_job).isdigit():
        raise DomainCaseError(f"after_job {after_job!r} is not a job id")
    chain = read_chain(rd) if (rd / CHAIN_FILE).is_file() else {"segments": []}
    seg: Dict = {"record_dir": str(rd), "script": str(script),
                 "armed": datetime.datetime.now().isoformat(timespec="seconds"),
                 "jobs": [], "ok": False}
    if after_job:
        seg["after_job"] = str(after_job)
    chain["segments"].append(seg)
    by_stage: Dict[str, str] = {}

    def fail(why: str) -> Dict:
        cancelled, kept = [], []
        for jid in by_stage.values():
            rc, _, err = run(["scancel", jid])
            (cancelled if rc == 0 else kept).append(jid)
        seg["error"] = why + (f"; cancelled {cancelled}" if cancelled else "") + \
            (f"; NOT cancelled, check by hand: {kept}" if kept else "")
        seg["cancelled"], seg["not_cancelled"] = cancelled, kept
        msg = seg["error"]
        for write in (lambda: _write_json(rd / CHAIN_FILE, chain),
                      lambda: _status_line(rd, f"ARMING FAILED: {msg}")):
            try:
                write()
            except Exception as e:                                  # noqa: BLE001
                seg["error"] += f"; this could not be recorded ({type(e).__name__}: {e})"
        return seg
    try:
        for j in jobs:
            argv = ["sbatch", "--parsable", "-p", j.partition, "-t", j.walltime, "-J", j.name]
            extra: List[str] = []
            if j.stage == "report":
                ids = list(by_stage.values())
                argv += [f"--dependency=afterany:{':'.join(ids)}"] if ids else []
                extra = ids
            elif j.after or (after_job and not by_stage):
                deps = ([str(after_job)] if after_job and not by_stage else []) + \
                    ([by_stage[j.after]] if j.after else [])
                argv += [f"--dependency=afterok:{':'.join(deps)}", "--kill-on-invalid-dep=yes"]
            rc, out, err = run(argv + [str(script), j.stage] + extra)
            jid = out.strip().split(";")[0] if rc == 0 else ""
            if not jid.isdigit():
                return fail(f"sbatch for {j.stage} failed (rc {rc}: "
                            f"{(err or out).strip()[:300]})")
            j.job_id = jid
            by_stage[j.stage] = jid
            seg["jobs"].append(asdict(j))
            _write_json(rd / CHAIN_FILE, chain)
        seg["ok"] = True
        _write_json(rd / CHAIN_FILE, chain)
    except Exception as e:                                          # noqa: BLE001
        return fail(f"arming stopped: {type(e).__name__}: {e}")
    _status_line(rd, "ARMED " + " ".join(f"{j.stage}={j.job_id}" for j in jobs))
    return seg


ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED",
          "RESIZING", "REQUEUE_HOLD", "REQUEUE_FED", "SIGNALING", "STAGE_OUT", "STOPPED"}
# A job neither squeue nor sacct knows: cancelled before it started (Compy keeps
# no accounting record of those) or purged. Not running, whatever it was.
GONE = "GONE"
# sacct without -S looks back only to midnight; the report stage asks for this.
SACCT_LOOKBACK_DAYS = 366


def _since(armed: Optional[str] = None) -> str:
    if armed:
        try:
            return (datetime.date.fromisoformat(armed[:10]) - datetime.timedelta(days=1)).isoformat()
        except ValueError:
            pass
    return (datetime.date.today() - datetime.timedelta(days=SACCT_LOOKBACK_DAYS)).isoformat()


def slurm_state(job_id: str, run: Callable = None, since: Optional[str] = None) -> Optional[str]:
    """SLURM's word for a job: squeue, then sacct from `since`. GONE when
    squeue says the id is invalid and sacct has no row (cancelled before it
    started, or purged); None when the scheduler did not answer."""
    run = run or _run
    rc, out, err = run(["squeue", "-h", "-j", str(job_id), "-o", "%T"], 60)
    if rc == 0 and out.strip():
        return out.strip().splitlines()[0].strip().upper()
    unknown_to_squeue = (rc == 0 and not out.strip()) or "invalid job id" in (err or "").lower()
    rc2, out2, _ = run(["sacct", "-n", "-X", "-S", since or _since(), "-j", str(job_id),
                        "-o", "State"], 60)
    if rc2 == 0 and out2.strip():
        return out2.strip().splitlines()[0].split()[0].upper()
    if rc2 == 0 and unknown_to_squeue:
        return GONE
    return None


def _verdict(rd: Path, stage: str, job_id: str):
    chk = rd / "checks" / f"{stage}_{job_id}.json"
    if not chk.is_file():
        return None, None
    try:
        return bool(json.loads(chk.read_text()).get("pass")), chk
    except ValueError:
        return False, chk


def _merged(rd: Path, chain: Dict) -> List[Dict]:
    """Every stage once, in plan order, with its latest job: a re-armed chain's
    later segments take over the stages they re-ran."""
    order: List[str] = []
    latest: Dict[str, Dict] = {}
    for k, seg in enumerate(chain["segments"]):
        for j in seg.get("jobs", []):
            if j["stage"] not in latest:
                order.append(j["stage"])
            latest[j["stage"]] = dict(j, segment=k)
    rank = {"case": 0, "smoke": 1, "report": 99999}
    order.sort(key=lambda s: rank.get(s, int(s[1:]) + 10 if re.fullmatch(r"y\d{4}", s) else 50000))
    rows = []
    for s in order:
        j = latest[s]
        verdict, chk = _verdict(rd, s, j["job_id"])
        rows.append({"stage": s, "job_id": j["job_id"], "segment": j["segment"],
                     "check_pass": verdict, "check": str(chk) if chk else None})
    return rows


def _gated(stage: str) -> bool:
    return stage in ("case", "smoke") or bool(re.fullmatch(r"y\d{4}", stage))


def chain_records(record_dir) -> Dict:
    """The chain as its own files record it, without asking SLURM.

    For the run's finalize, which reads results from disk once every job has
    ended. Per stage (the latest job of a re-armed chain) its job id and
    check verdict (None where no check record was written), the model years
    planned and those whose check passed, and the first gated stage without a
    passing record. A case stage from before case records existed counts as
    passed when a later stage left a record (it could only run after it).
    """
    rd = Path(record_dir)
    chain = read_chain(rd)
    rows = _merged(rd, chain)
    later_ran = {r["stage"] for r in rows if r["check_pass"] is not None}
    years, planned, first = [], [], None
    for r in rows:
        s = r["stage"]
        if re.fullmatch(r"y\d{4}", s):
            planned.append(int(s[1:]))
            if r["check_pass"]:
                years.append(int(s[1:]))
        ok = r["check_pass"] is True or (s == "case" and r["check_pass"] is None
                                         and later_ran - {"case"})
        if _gated(s) and not ok and first is None:
            first = s
    return {"record_dir": str(rd), "armed": chain["segments"][0].get("armed")
            if chain["segments"] else None, "segments": len(chain["segments"]),
            "stages": rows, "years_planned": planned, "years_passed": years,
            "first_unpassed": first}


_REPORT_LINE = re.compile(r"\bREPORT (\S+) job=(\S+) state=(\S+) elapsed=(\S+) exit=(\S+)")


def report_lines(record_dir) -> List[Dict]:
    """The report stage's account of every job (the sacct lines it wrote into
    STATUS.log): name, job id, state, elapsed and its seconds, exit code. []
    before the report stage has run. A job reported more than once (a re-armed
    chain) keeps its last line."""
    p = Path(record_dir) / STATUS_FILE
    if not p.is_file():
        return []
    out: Dict[str, Dict] = {}
    for line in p.read_text().splitlines():
        m = _REPORT_LINE.search(line)
        if not m:
            continue
        name, job, state, el, ex = m.groups()
        secs = None
        try:
            d, _, hms = el.rpartition("-")
            h, mi, se = (int(x) for x in hms.split(":"))
            secs = int(d or 0) * 86400 + h * 3600 + mi * 60 + se
        except ValueError:
            pass
        out[job] = {"name": name, "job_id": job, "state": state, "elapsed": el,
                    "seconds": secs, "exit": ex}
    return list(out.values())


def chain_status(record_dir, run: Callable = None) -> Dict:
    """Where the chain stands. A record.

    Per stage: SLURM state and check verdict. A stage known to neither squeue
    nor sacct (GONE) has ended; one SLURM will not answer for is taken as
    ended only when the chain shows it cannot run any more (an earlier stage
    ended without passing, or the report, which waits on every job, has
    ended), and as active otherwise. A gated stage passed only with a passing
    check record. active: some stage may still run. failed_stage: the first
    stage that ended without passing. years_passed: the model years whose
    check passed.
    """
    rd = Path(record_dir)
    chain = read_chain(rd)
    since = _since(chain["segments"][0].get("armed") if chain["segments"] else None)
    rows = _merged(rd, chain)
    recs = chain_records(rd)
    states = {r["stage"]: slurm_state(r["job_id"], run, since) for r in rows}
    report = states.get("report")
    report_ended = report is not None and report not in ACTIVE
    out: Dict = {"record_dir": str(rd), "jobs": [], "active": False, "failed_stage": None,
                 "years_passed": recs["years_passed"], "years_planned": recs["years_planned"]}
    blocked = False
    for r in rows:
        s, st = r["stage"], states[r["stage"]]
        inferred = None
        if st is None and (blocked or report_ended):
            inferred = "CANCELLED (inferred: the chain cannot run it any more)"
        elif st == GONE:
            inferred = "ended (no record at SLURM: cancelled before it started, or purged)"
        out["jobs"].append({"stage": s, "job_id": r["job_id"], "state": st,
                            "inferred": inferred, "check_pass": r["check_pass"]})
        if (st is None and not inferred) or st in ACTIVE:
            out["active"] = True
            continue
        if s == "report":
            continue
        passed = r["check_pass"] is True or (s == "case" and r["check_pass"] is None
                                             and st == "COMPLETED")
        if not passed:
            blocked = True
            if out["failed_stage"] is None:
                out["failed_stage"] = s
    out["done"] = not out["active"]
    out["ok"] = (out["done"] and out["failed_stage"] is None
                 and bool(chain["segments"] and chain["segments"][-1].get("ok")))
    return out
