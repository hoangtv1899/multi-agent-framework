#!/usr/bin/env python3
"""
ELM Wrapper
mcp/elm-mcp/src/elm_wrapper.py

Simplified wrapper for creating, configuring, building, and
running a single-column ELM case on Compy.

Phase 1 design:
    - srun-only execution (no sbatch)
    - No caching (every prepare_case = fresh build)
    - Single-experiment scope (one case per instance)

Build sharing across experiments and proper caching are
deferred to the Phase 3 refactor.

Two config dicts — never mixed:
    FIXED_CONFIG   → machine/compset/paths, never changed
    runtime_config → planner sets these per experiment,
                     always applied LAST in xmlchange
                     so they always win over defaults
"""
import os
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# CIME subprocess environment
# ─────────────────────────────────────────────────────────────────────
# Intel's compiler toolchain (icc/mpiicc, used for COMPILER='intel' below)
# hard-fails ("Catastrophic error: could not set locale") under LANG=C.UTF-8
# with LC_ALL unset -- the ambient state of a typical Compy shell/MCP-spawned
# process. en_US.utf8 is a real installed locale (confirmed via `locale -a`)
# that Intel's toolchain accepts; C.UTF-8 is glibc's synthetic locale, which
# it does not. Applied to every subprocess call that invokes CIME scripts so
# case creation/build never depends on the caller's ambient locale.
def _cime_env() -> dict:
    return {**os.environ, 'LC_ALL': 'en_US.utf8', 'LANG': 'en_US.utf8'}


# ─────────────────────────────────────────────────────────────────────
# FIXED CONFIG — never changed by planner
# ─────────────────────────────────────────────────────────────────────
# ELM 1d input files (originals on NERSC; copy them here on Compy)
_ELM_INPUT_FILES_DIR = os.getenv(
    "ELM_INPUT_FILES_DIR",
    "/qfs/people/tran289/IDEAS/1d_elm/input_files",
)

FIXED_CONFIG = {
    # ELM-only, no river routing (2026-07-23). RES and COMPSET are COUPLED --
    # change both together or create_newcase hard-fails:
    #   * No ELM compset ALIAS has a stub river: all 149 aliases in
    #     components/elm/cime_config/config_compsets.xml hardcode MOSART
    #     (IELM = 2000_DATM%QIA_ELM%SP_SICE_SOCN_MOSART_SGLC_SWAV), so a
    #     compset longname is the only route. CIME accepts longnames and
    #     auto-appends the trailing stubs (_SIAC_SESP).
    #   * ELMMOS_USRDAT is gated on compset regex "(DATM|SATM).+ELM.+MOSART"
    #     (cime_config/config_grids.xml:71) and no longer matches; ELM_USRDAT
    #     ("(DATM|SATM).+ELM", :61) is the drop-in, identical except rof->null.
    # KEEP "DATM%QIA" VERBATIM: it sets DATM_MODE=CLM_QIAN, and
    # tools/set_nldas_forcing.py hardcodes datm.streams.txt.CLM_QIAN.* run-dir
    # filenames. Any other DATM modifier would make it write orphan files with
    # NO error -- cases would silently run Qian forcing while the run record
    # claims NLDAS-2. To restore routing: RES='ELMMOS_USRDAT', COMPSET='IELM'.
    'RES':      'ELM_USRDAT',
    'COMPSET':  '2000_DATM%QIA_ELM%SP_SICE_SOCN_SROF_SGLC_SWAV',
    'MACH':     'compy',
    'COMPILER': 'intel',
    'PROJECT':  'e3sm',
    'SRC_DIR':  '/qfs/people/tran289/E3SM',
    'INPUT_FILES_DIR': _ELM_INPUT_FILES_DIR,
    'FSURDAT':
        _ELM_INPUT_FILES_DIR + '/Surfacedata_Station_2006_.nc',
    'DOMAIN_FILE': 'Domainfile_station_2006_.nc',
}

# Fixed xmlchange settings (applied first; runtime_config can override)
FIXED_XML = {
    'LND_DOMAIN_FILE': FIXED_CONFIG['DOMAIN_FILE'],
    'ATM_DOMAIN_FILE': FIXED_CONFIG['DOMAIN_FILE'],
    'LND_DOMAIN_PATH': FIXED_CONFIG['INPUT_FILES_DIR'],
    'ATM_DOMAIN_PATH': FIXED_CONFIG['INPUT_FILES_DIR'],
    'NTASKS':          '1',
    # NLDAS-2 forcing at 1/8 deg (~12 km), set explicitly 2026-07-25.
    # The compset's %QIA modifier otherwise defaults DATM_MODE to CLM_QIAN =
    # Qian T62 (~1.9x2.5 deg). At T62 an entire small watershed falls inside
    # ONE forcing cell, so every column gets identical precip/temperature and
    # any elevation-gradient study is meaningless (observed: 11 of 14 Naches
    # columns produced near-identical QOVER/QDRAI across 398-1868 m relief).
    # CLMMOSARTTEST reads $DIN_LOC_ROOT/atm/datm7/NLDAS/clmforc.nldas.%ym.nc,
    # which on Compy covers 1979-2023 complete. Its name mentions MOSART but
    # it is purely a DATM stream preset -- orthogonal to the ROF component and
    # fine with SROF. NOTE: tools/set_nldas_forcing.py is NOT usable on Compy;
    # it points at atm_forcing.datm7.NLDAS2.0.125d.v1, whose Precip and TPQWL
    # directories are EMPTY here (Solar has only 1980-81).
    # DATM_MODE lives in env_run.xml, so changing it needs only case.setup
    # --reset, not a rebuild.
    'DATM_MODE':       'CLMMOSARTTEST',
}

# Fixed namelists for components other than ELM
# (ELM namelist is constructed in _write_namelists so it can use FSURDAT)
#
# No 'mosart' entry: the compset above uses a stub river (SROF), so case.setup
# generates only user_nl_{cpl,datm,elm} -- a user_nl_mosart would be a dead
# file CIME never reads. This also drops the external dependency on
# /compyfs/inputdata/rof/mosart/MOSART_NLDAS_8th_20160426.nc, which existed
# purely to satisfy build-namelist validation and was never read at runtime
# (it was also the source of the invalid-frivinp_mesh build failure).
# mapalgo is ONE entry because DATM_MODE=CLMMOSARTTEST is a SINGLE stream
# (TBOT,WIND,QBOT,PSRF,FLDS,PRECTmms,FSDS all in one file). It was 3-wide for
# CLM_QIAN, which has three streams (Solar/Precip/TPQW) -- if you ever switch
# DATM_MODE back, this must change with it.
FIXED_NAMELISTS = {
    'datm': 'mapalgo = "nn"\n',
}

# Default runtime config
DEFAULT_RUNTIME = {
    'STOP_N':                '1',
    'STOP_OPTION':           'nyears',
    'DATM_CLMNCEP_YR_START': '1981',
    'DATM_CLMNCEP_YR_END':   '1981',
    'RUN_STARTDATE':         '1981-01-01',
    'REST_N':                '1',
    'REST_OPTION':           'nyears',
}

# Keys planner can override
RUNTIME_KEYS = {
    'STOP_N', 'STOP_OPTION',
    'DATM_CLMNCEP_YR_START', 'DATM_CLMNCEP_YR_END',
    'RUN_STARTDATE', 'REST_N', 'REST_OPTION',
    'LND_DOMAIN_FILE', 'LND_DOMAIN_PATH',
    'ATM_DOMAIN_FILE', 'ATM_DOMAIN_PATH',
    'FSURDAT', 'FINIDAT',
}

# Subset that goes to xmlchange (vs namelist — FSURDAT/FINIDAT are namelist keys)
XML_RUNTIME_KEYS = RUNTIME_KEYS - {'FSURDAT', 'FINIDAT'}


# ─────────────────────────────────────────────────────────────────────
# SLICES — continue a built case in place, the walk-through-the-year seam.
# One case per column for the whole year; each slice advances it a window
# and ends on a restart, which a between-slice stamping may edit before
# the next slice reads it.
# ─────────────────────────────────────────────────────────────────────
def latest_restart(case_dir) -> Path:
    """The land restart file the next continuation will start from.

    Read from run/rpointer.lnd — the pointer ELM itself follows on
    CONTINUE_RUN — never guessed from a directory listing: a run dir
    accumulates restarts, and the newest mtime is not the pointer's choice.
    This is the file a between-slice stamping edits (set_water_table.apply /
    apply_profile take it like any finidat — a restart IS that format).
    """
    rd = Path(case_dir) / "run"
    rp = rd / "rpointer.lnd"
    if not rp.is_file():
        raise ValueError(
            f"no rpointer.lnd in {rd} — the case has not written a land "
            f"restart yet, so there is nothing to continue from or to stamp")
    name = rp.read_text().strip().splitlines()[0].strip()
    f = rd / name
    if not f.is_file():
        raise ValueError(f"rpointer.lnd names {name!r} but {f} does not exist")
    return f


def configure_continuation(case_dir, stop_n, stop_option="ndays"):
    """Point a built case at its NEXT slice: continue, this much further.

    xmlchange only — no build, no namelists, no run. With CONTINUE_RUN set,
    each later invocation of the executable advances the case by stop_n more
    stop_option from wherever it stopped, reading its state through the
    rpointer files — which is exactly what lets a stamping of
    latest_restart() between slices reach the model. REST is pinned to the
    same window so every slice ends on a restart.
    """
    cd = Path(case_dir)
    for key, val in (("CONTINUE_RUN", "TRUE"),
                     ("STOP_N", str(int(stop_n))),
                     ("STOP_OPTION", str(stop_option)),
                     ("REST_N", str(int(stop_n))),
                     ("REST_OPTION", str(stop_option))):
        subprocess.run(["./xmlchange", f"{key}={val}"], cwd=cd,
                       env=_cime_env(), check=True, capture_output=True,
                       text=True)
    # xmlchange edits env_run.xml ONLY; the executable reads the RUN-DIR
    # namelists, and CONTINUE_RUN becomes drv_in's start_type='continue'
    # only when those are regenerated. Measured live (job 774959): without
    # this line the second slice silently re-ran the FIRST week from the
    # original start — srun returned 0, the rpointer moved back to day 8,
    # and nothing said so.
    subprocess.run(["./preview_namelists"], cwd=cd, env=_cime_env(),
                   check=True, capture_output=True, text=True)


def run_built_case(case_dir) -> bool:
    """srun the case's executable from its run dir (blocking; needs a node).

    Module-level so a slice loop can rerun an existing case without
    reconstructing the agent that built it; GeneratedELMAgent.run_simulation
    delegates here.
    """
    case_dir = Path(case_dir)
    run_dir = case_dir / "run"
    # Resolve the exe via EXEROOT so --keepexe clones work (their exe lives
    # in the reference case's build dir, not <case>/build).
    try:
        exeroot = subprocess.check_output(
            ["./xmlquery", "EXEROOT", "--value"],
            cwd=case_dir, env=_cime_env(), text=True,
        ).strip()
        exe_path = Path(exeroot) / "e3sm.exe"
    except Exception:
        exe_path = case_dir / "build" / "e3sm.exe"

    if not exe_path.exists():
        raise RuntimeError(f"Executable not found: {exe_path}")

    # Ensure timing dir exists (ELM requires this)
    (run_dir / "timing" / "checkpoints").mkdir(parents=True, exist_ok=True)

    logger.info(f"Running via srun: {case_dir.name}")
    start = time.time()
    # THE CASE'S OWN ENVIRONMENT FIRST — .env_mach_specific.sh loads the
    # modules the executable was linked against (job 774958 died in one
    # second on libnetcdf.so.15 without it). Same pattern as
    # tools/run_cases.sh, which is how every ensemble run does it; sourcing
    # needs a shell, so the srun goes through bash -c.
    #
    # --mpi=pmi2 is REQUIRED on Compy: cases are built against Intel MPI
    # (mpilib=impi) and CIME's compy config specifies --mpi=pmi2 for it.
    # Without it Intel MPI falls back to its hydra bootstrap and dies
    # setting up proxies.
    result = subprocess.run(
        ["bash", "-c",
         "source ./.env_mach_specific.sh 2>/dev/null; "
         "cd run && srun --mpi=pmi2 --label -n 1 -N 1 -c 2 "
         f"--cpu_bind=cores '{exe_path}'"],
        cwd=case_dir,
        capture_output=True,
        text=True,
    )
    elapsed = (time.time() - start) / 60

    if result.returncode == 0:
        logger.info(f"srun completed in {elapsed:.1f} min")
        return True
    logger.error(f"srun failed after {elapsed:.1f} min: "
                 f"{result.stderr[-300:]}")
    return False


# ─────────────────────────────────────────────────────────────────────
# ELM AGENT
# ─────────────────────────────────────────────────────────────────────
class GeneratedELMAgent:
    """
    Creates, configures, builds, and runs a single 1D ELM case.

    Single responsibility: drive
        create_newcase → xmlchange → namelists →
        case.setup → case.build → srun

    Usage:
        agent = GeneratedELMAgent(
            case_suffix    = 'elm_baseline',
            runtime_config = {
                'STOP_N':                '5',
                'DATM_CLMNCEP_YR_START': '1981',
                'DATM_CLMNCEP_YR_END':   '1985',
                'RUN_STARTDATE':         '1981-01-01',
            }
        )
        case_dir = agent.prepare_case()
        success  = agent.run_simulation()
        summary  = agent.get_summary()
    """

    def __init__(self,
                 case_suffix:    Optional[str] = None,
                 runtime_config: Optional[dict] = None,
                 prescribed_weather: Optional[Any] = None):
        # Apply defaults, then validate and apply user overrides
        self.runtime_config = DEFAULT_RUNTIME.copy()
        if runtime_config:
            # RAISES, SINCE 2026-08-14. This used to log a warning and drop the
            # key, and the case then ran with the DEFAULT in its place — a
            # misspelt STOP_OPTION gives a case that builds, runs, and
            # simulates the wrong length, with the only evidence a warning
            # inside an MCP server's log during an unattended job. A key this
            # class cannot apply is a case it must not build.
            unknown = sorted(set(runtime_config) - RUNTIME_KEYS)
            if unknown:
                raise ValueError(
                    f"Unknown runtime key(s) {unknown} — this class cannot "
                    f"apply them, and running anyway means running with the "
                    f"DEFAULT in their place.\n"
                    f"  Allowed: {sorted(RUNTIME_KEYS)}\n"
                    f"  Add the key to RUNTIME_KEYS in elm_wrapper.py if ELM "
                    f"accepts it, or fix the caller."
                )
            for key, value in runtime_config.items():
                self.runtime_config[key] = str(value)

        # NOT VALIDATED HERE, and not deferred either — spec_to_fill runs now
        # so a malformed spec raises where the case is CONFIGURED rather than
        # eight minutes later where it is built. The resolved fill is thrown
        # away; only the spec is kept, because the case directory it needs does
        # not exist yet.
        self.prescribed_weather = prescribed_weather
        if prescribed_weather is not None:
            import forcing
            forcing.spec_to_fill(prescribed_weather)

        # State
        self.case_suffix  = case_suffix
        self.case_name    = None
        self.case_dir     = None
        self.is_built     = False
        self.is_completed = False
        self.forcing_written = None

        logger.info(
            f"ELMAgent init: suffix={case_suffix} | "
            f"STOP_N={self.runtime_config['STOP_N']} | "
            f"years={self.runtime_config['DATM_CLMNCEP_YR_START']}"
            f"–{self.runtime_config['DATM_CLMNCEP_YR_END']}"
        )

    # ─────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────
    def prepare_case(self, ref_case_dir: Optional[str] = None) -> Path:
        """
        Build ELM case. If ref_case_dir is given, clone from it with
        --keepexe (no compile, ~30s). Otherwise fresh build (~8 min).
        """
        if ref_case_dir:
            logger.info("Cloning ELM case from reference (--keepexe)")
            self._clone_case(ref_case_dir)
            self._configure_case(runtime_only=True)
            self._write_namelists()
            self._setup_case()   # case.setup: run-dir namelists + env, NO compile
            # No _build_case — exe inherited from ref
        else:
            logger.info("Building new ELM case from scratch...")
            self._create_case()
            self._configure_case()
            self._write_namelists()
            self._build_case()
        self.is_built = True
        return self.case_dir

    def run_simulation(self) -> bool:
        """Run ELM via srun (blocking, requires interactive node)."""
        if not self.is_built:
            raise RuntimeError(
                "Case not built. Call prepare_case() first."
            )
        ok = run_built_case(self.case_dir)
        if ok:
            self.is_completed = True
        return ok

    def get_summary(self) -> dict:
        """Return current case state."""
        return {
            'case_name':      self.case_name,
            'case_dir':       str(self.case_dir) if self.case_dir else None,
            'is_built':       self.is_built,
            'is_completed':   self.is_completed,
            'runtime_config': self.runtime_config,
            'history_files':  self._get_history_files(),
        }

    # Keep get_case_info as an alias for backward compatibility
    # with the existing adapter — remove once adapter is updated.
    def get_case_info(self) -> dict:
        return self.get_summary()

    # ─────────────────────────────────────────────────────────
    # PRIVATE — BUILD STEPS
    # ─────────────────────────────────────────────────────────
    def _create_case(self):
        """Run create_newcase to set up the case directory."""
        src_dir = FIXED_CONFIG['SRC_DIR']
        try:
            git_hash = subprocess.check_output(
                ['git', 'log', '-n', '1', '--format=%h'],
                cwd = src_dir,
            ).decode().strip()
        except Exception:
            git_hash = 'unknown'

        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        suffix    = f".{self.case_suffix}" if self.case_suffix else ""
        self.case_name = f"1D_ELM.{git_hash}.{timestamp}{suffix}"

        pscratch = os.environ.get('PSCRATCH', '/compyfs/tran289')
        self.case_dir = Path(pscratch) / "E3SMv3" / self.case_name

        logger.info(f"Creating: {self.case_name}")

        scripts_dir = Path(src_dir) / "cime" / "scripts"
        try:
            subprocess.run(
                [
                    str(scripts_dir / "create_newcase"),
                    "-case",     str(self.case_dir),
                    "-res",      FIXED_CONFIG['RES'],
                    "-mach",     FIXED_CONFIG['MACH'],
                    "-compiler", FIXED_CONFIG['COMPILER'],
                    "-compset",  FIXED_CONFIG['COMPSET'],
                    "--project", FIXED_CONFIG['PROJECT'],
                ],
                cwd            = scripts_dir,
                env            = _cime_env(),
                check          = True,
                capture_output = True,
                text           = True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"create_newcase failed:\n{e.stderr}"
            ) from e
    
    def _clone_case(self, ref_case_dir: str):
        """Run create_clone --keepexe to share executable with a reference case.

        Faster than _create_case + _build_case combined: skips compile entirely.
        Reference case must already be built (case.build complete).
        """
        src_dir = FIXED_CONFIG['SRC_DIR']
        try:
            git_hash = subprocess.check_output(
                ['git', 'log', '-n', '1', '--format=%h'],
                cwd = src_dir,
            ).decode().strip()
        except Exception:
            git_hash = 'unknown'
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        suffix    = f".{self.case_suffix}" if self.case_suffix else ""
        self.case_name = f"1D_ELM.{git_hash}.{timestamp}{suffix}"
        pscratch = os.environ.get('PSCRATCH', '/compyfs/tran289')
        self.case_dir = Path(pscratch) / "E3SMv3" / self.case_name
        ref_name = Path(ref_case_dir).name
        logger.info(f"Cloning from {ref_name} → {self.case_name}")
        scripts_dir = Path(src_dir) / "cime" / "scripts"
        try:
            subprocess.run(
                [
                    str(scripts_dir / "create_clone"),
                    "--case",  str(self.case_dir),
                    "--clone", str(ref_case_dir),
                    "--keepexe",
                ],
                cwd            = scripts_dir,
                env            = _cime_env(),
                check          = True,
                capture_output = True,
                text           = True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"create_clone failed:\n{e.stderr}"
            ) from e

    def _configure_case(self, runtime_only: bool = False):
        """
        Apply xmlchange settings.
        Order:
            1. Fixed settings (skip for clones — inherited from reference)
            2. EXEROOT (skip for clones — must point at reference's build)
            3. RUNDIR (always — symbolic, so a clone inherits its OWN)
            4. Runtime config LAST → always wins

        Args:
            runtime_only: if True (for clones), skip build-time settings
                          that env_build.xml inherits from the reference.
        """
        if not runtime_only:
            # Step 1 — fixed (skip if overridden by runtime_config)
            for key, value in FIXED_XML.items():
                if key not in self.runtime_config:
                    self._xmlchange(key, value)

            # Step 2 — EXEROOT (fresh build only)
            # Deliberately an ABSOLUTE literal, and the asymmetry with RUNDIR
            # below is the whole point: --keepexe works precisely BECAUSE the
            # clone inherits this path unresolved-per-case and keeps pointing
            # at the reference's build. Make this symbolic and every clone
            # looks for an e3sm.exe it never compiled.
            self._xmlchange('EXEROOT', str(self.case_dir / "build"))

        # Step 3 — RUNDIR (always). SYMBOLIC, not an absolute literal.
        #
        # create_clone copies env_run.xml verbatim and generates the clone's
        # namelists ITSELF, before prepare_case() ever reaches this line. With
        # an absolute literal here the clone spends that window pointing at the
        # REFERENCE's run directory, and writes its drv_in/lnd_in/datm_in on
        # top of the reference's — stamped with the clone's case_name. The
        # reference then runs under a sibling's identity: history files named
        # for another column and a `case` attribute to match, sitting in the
        # right directory with the right data inside.
        #
        # It survived unnoticed from 2026-07-25 to 2026-08-03 because the
        # clobbered lnd_in is generated from the clone's user_nl_elm while that
        # is still a byte copy of the reference's — so the physics came out
        # correct by INHERITANCE, not by design. Customize a clone any earlier
        # and the reference silently runs a sibling's forcing with no
        # mismatched name left to notice it by.
        #
        # $CASEROOT is per-case and create_clone rewrites it, so the inherited
        # value already points at the clone. Resolves to the same path this
        # always used ($case_dir/run) — nothing on disk moves.
        self._xmlchange('RUNDIR', '$CASEROOT/run')

        # Step 4 — runtime LAST (wins over fixed)
        for key in XML_RUNTIME_KEYS:
            if key in self.runtime_config:
                self._xmlchange(key, self.runtime_config[key])

        logger.info("XML configuration applied")
    
    def _xmlchange(self, key: str, value: str):
        """Run a single xmlchange command in the case directory."""
        subprocess.run(
            ['./xmlchange', f'{key}={value}'],
            cwd            = self.case_dir,
            env            = _cime_env(),
            check          = True,
            capture_output = True,
            text           = True,
        )

    def _write_namelists(self):
        """Write user_nl_* files."""
        # FSURDAT can be overridden by planner
        fsurdat = self.runtime_config.get(
            'FSURDAT', FIXED_CONFIG['FSURDAT']
        )

        # NOTE: hist_nhtfrq = -3, hist_mfilt = 365 matches the reference
        # bash script (3-hourly output, 365 records per file).
        # Change if analyzer expects a different output frequency.
        finidat = self.runtime_config.get('FINIDAT')
        finidat_line = f"finidat = '{finidat}'\n" if finidat else ""
        elm_namelist = (
            finidat_line +
            f"fsurdat = '{fsurdat}'\n"
            "hist_empty_htapes = .true.\n"
            "mksrf_lsmlon = 1\n"
            "mksrf_lsmlat = 1\n"
            # .false. so the sub-grid is [1 natveg + 15 urban] = 16 columns,
            # which is EXACTLY the layout of the CONUS 1-km restarts (verified:
            # zero type-2 columns across all 19.9M of them). That match is what
            # lets a CONUS gridcell be subset straight into a finidat, with no
            # carrier file and no prior run. elm_surface_generator folds
            # PCT_CROP into PCT_NATVEG so the surfdata agrees with this.
            #
            # THE REASON IS THE WARM START, so a COLD case does not have it.
            # With no finidat there is no restart layout to match, and the
            # constraint costs rather than buys: a cold conceptual run builds
            # its surface from the default template, which is the new CFT-based
            # format, and ELM refuses that outright —
            #     ERROR: New format surface datasets require
            #            create_crop_landunit TRUE   (surfrdMod.F90:1061)
            # Every one of the first conceptual runs died there. Keyed on
            # finidat rather than on an archetype flag because the restart is
            # the thing the setting is about.
            + (f"create_crop_landunit = "
               f"{'.false.' if finidat else '.true.'}\n")
            + "hist_fincl1 = "
            "'RAIN','SNOW','QOVER','QDRAI','QCHARGE',"
            "'TWS','H2OSOI','SOILLIQ','ZWT','WA',"
            "'H2OSNO','QSNOMELT','QINFL','QSOIL','QVEGE','QVEGT'\n"
            "hist_nhtfrq = -3\n"
            "hist_mfilt  = 365\n"
        )

        # Written weather, when the design asked for it. Must happen BEFORE
        # user_nl_datm is written, because a flat-solar level adds a line to it
        # — composing the file from both sources in one write is what stops the
        # two from racing each other's content.
        datm_namelist = FIXED_NAMELISTS['datm'] + self._write_prescribed_forcing()

        namelists = {
            'elm':    elm_namelist,
            'datm':   datm_namelist,
        }

        for name, content in namelists.items():
            nl_file = self.case_dir / f"user_nl_{name}"
            nl_file.write_text(content)
            logger.info(f"Wrote user_nl_{name}")

    def _write_prescribed_forcing(self) -> str:
        """Write the DATM files and the user stream file. Returns namelist lines.

        A NO-OP WITHOUT A SPEC, which is every site run and every conceptual run
        that kept the real weather — so this method existing changes nothing
        about the path that already works.

        RAISES ON FAILURE rather than logging and continuing. A missing stream
        file does not stop the run: CIME generates its own, DATM reads gridded
        NLDAS, and the case completes on the weather of whatever real cell the
        column sits in. The run record would say the weather was prescribed and
        the output would be the borrowed climate — the exact confusion this
        capability exists to remove.
        """
        import forcing

        if self.prescribed_weather is None:
            # A CLONE INHERITS THE REFERENCE CASE'S FILES. create_clone copies
            # the case directory, so a case with no written weather that was
            # cloned from one that had it would start life holding the
            # reference's user stream file — and run on the reference column's
            # weather while every record said it used NLDAS. Deleting it is the
            # difference between "this case has no prescribed weather" and
            # "nobody wrote one for this case".
            stale = self.case_dir / forcing.USER_STREAM_FILE
            if stale.exists():
                stale.unlink()
                logger.info(f"Removed inherited {forcing.USER_STREAM_FILE} — "
                            f"this case has no prescribed weather")
            return ""

        # The stream's domainInfo must name the SAME domain file the case uses,
        # or DATM interpolates the written cell onto a grid it was not written
        # for. Both come off the runtime config so they cannot disagree.
        dom_path = self.runtime_config.get('LND_DOMAIN_PATH',
                                           FIXED_XML['LND_DOMAIN_PATH'])
        dom_file = self.runtime_config.get('LND_DOMAIN_FILE',
                                           FIXED_XML['LND_DOMAIN_FILE'])
        y0 = int(self.runtime_config['DATM_CLMNCEP_YR_START'])
        y1 = int(self.runtime_config['DATM_CLMNCEP_YR_END'])

        # The column's own coordinates, read from the domain file rather than
        # passed in. mapalgo="nn" makes DATM take the nearest forcing cell
        # without checking containment, so a file written at any other point
        # would be used anyway and the run would finish on weather from a place
        # nobody chose.
        lat, lon = self._domain_latlon(Path(dom_path) / dom_file)

        written = forcing.install(
            case_dir    = self.case_dir,
            data_dir    = self.case_dir / "prescribed_forcing",
            spec        = self.prescribed_weather,
            lat         = lat,
            lon         = lon,
            years       = range(y0, y1 + 1),
            domain_file = str(Path(dom_path) / dom_file),
            src_dir     = forcing.real_nldas_dir(),
        )
        self.forcing_written = written
        logger.info(f"Wrote prescribed forcing: {written['fill']} "
                    f"{written['values'] or ''} — {written['n_files']} file(s) "
                    f"at ({lat:.4f}, {lon:.4f}), stream "
                    f"{Path(written['stream_file']).name}")
        return written["namelist"]

    @staticmethod
    def _domain_latlon(domain_path: Path) -> tuple:
        """The single gridcell's centre, from the domain file the case uses."""
        import netCDF4 as nc
        with nc.Dataset(str(domain_path)) as d:
            return (float(d.variables['yc'][:].ravel()[0]),
                    float(d.variables['xc'][:].ravel()[0]))

    def _setup_case(self):
        """case.setup only (regenerate run-dir namelists + .env_mach_specific,
        no compile) — needed for --keepexe clones, which skip _build_case."""
        subprocess.run(
            ['./case.setup', '--reset'],
            cwd            = self.case_dir,
            env            = _cime_env(),
            check          = True,
            capture_output = True,
            text           = True,
        )

    def _build_case(self):
        """Run case.setup followed by case.build."""
        try:
            logger.info("Running case.setup...")
            subprocess.run(
                ['./case.setup'],
                cwd            = self.case_dir,
                env            = _cime_env(),
                check          = True,
                capture_output = True,
                text           = True,
            )

            logger.info("Running case.build...")
            start = time.time()
            subprocess.run(
                ['./case.build'],
                cwd            = self.case_dir,
                env            = _cime_env(),
                check          = True,
                capture_output = True,
                text           = True,
            )
            elapsed = (time.time() - start) / 60
            logger.info(f"Build complete in {elapsed:.1f} min")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Build failed:\n{e.stderr}") from e

    # ─────────────────────────────────────────────────────────
    # PRIVATE — HELPERS
    # ─────────────────────────────────────────────────────────
    def _get_history_files(self) -> list:
        """Find ELM history files in the run directory."""
        if not self.case_dir:
            return []
        run_dir = self.case_dir / "run"
        if not run_dir.exists():
            return []
        return sorted(
            str(f) for f in run_dir.glob("*.elm.h0.*.nc")
        )