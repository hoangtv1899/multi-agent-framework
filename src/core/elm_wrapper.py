#!/usr/bin/env python3
"""
ELM Wrapper
src/core/elm_wrapper.py

Simplified wrapper for creating, configuring, building, and
running a single-column ELM case on Perlmutter.

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
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# FIXED CONFIG — never changed by planner
# ─────────────────────────────────────────────────────────────────────
FIXED_CONFIG = {
    'RES':      'ELMMOS_USRDAT',
    'COMPSET':  'IELM',
    'MACH':     'pm-cpu',
    'COMPILER': 'gnu',
    'PROJECT':  'm3780',
    'SRC_DIR':  '/global/u2/h/hvtran/E3SM',
    'INPUT_FILES_DIR':
        '/global/homes/h/hvtran/RCSFA/1d_elm/input_files',
    'FSURDAT':
        '/global/homes/h/hvtran/RCSFA/1d_elm/input_files/'
        'Surfacedata_Station_2006_.nc',
    'DOMAIN_FILE': 'Domainfile_station_2006_.nc',
}

# Fixed xmlchange settings (applied first; runtime_config can override)
FIXED_XML = {
    'LND_DOMAIN_FILE': FIXED_CONFIG['DOMAIN_FILE'],
    'ATM_DOMAIN_FILE': FIXED_CONFIG['DOMAIN_FILE'],
    'LND_DOMAIN_PATH': FIXED_CONFIG['INPUT_FILES_DIR'],
    'ATM_DOMAIN_PATH': FIXED_CONFIG['INPUT_FILES_DIR'],
    'NTASKS':          '1',
}

# Fixed namelists for components other than ELM
# (ELM namelist is constructed in _write_namelists so it can use FSURDAT)
FIXED_NAMELISTS = {
    'mosart': (
        "do_rtm = .false.\n"
        "frivinp_rtm = '/global/cfs/cdirs/e3sm/inputdata/"
        "rof/mosart/MOSART_NLDAS_8th_20160426.nc'\n"
        "frivinp_mesh = '/global/cfs/cdirs/e3sm/inputdata/"
        "rof/mosart/MOSART_NLDAS_8th_20160426.nc'\n"
        "wrmflag = .false.\n"
        "inundflag = .false.\n"
    ),
    'datm': 'mapalgo = "nn", "nn", "nn"\n',
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
    'FSURDAT',
}

# Subset that goes to xmlchange (vs namelist)
XML_RUNTIME_KEYS = RUNTIME_KEYS - {'FSURDAT'}


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
                 runtime_config: Optional[dict] = None):
        # Apply defaults, then validate and apply user overrides
        self.runtime_config = DEFAULT_RUNTIME.copy()
        if runtime_config:
            for key, value in runtime_config.items():
                if key in RUNTIME_KEYS:
                    self.runtime_config[key] = str(value)
                else:
                    logger.warning(
                        f"Unknown runtime key '{key}' ignored. "
                        f"Allowed: {sorted(RUNTIME_KEYS)}"
                    )

        # State
        self.case_suffix  = case_suffix
        self.case_name    = None
        self.case_dir     = None
        self.is_built     = False
        self.is_completed = False

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

        run_dir  = self.case_dir / "run"
        exe_path = self.case_dir / "build" / "e3sm.exe"

        if not exe_path.exists():
            raise RuntimeError(f"Executable not found: {exe_path}")

        # Ensure timing dir exists (ELM requires this)
        timing_dir = run_dir / "timing" / "checkpoints"
        timing_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running via srun: {self.case_name}")
        start  = time.time()
        result = subprocess.run(
            [
                "srun", "--label",
                "-n", "1", "-N", "1", "-c", "2",
                "--cpu_bind=cores",
                str(exe_path),
            ],
            cwd            = run_dir,
            capture_output = True,
            text           = True,
        )
        elapsed = (time.time() - start) / 60

        if result.returncode == 0:
            self.is_completed = True
            logger.info(f"srun completed in {elapsed:.1f} min")
            return True
        else:
            logger.error(
                f"srun failed after {elapsed:.1f} min: "
                f"{result.stderr[-300:]}"
            )
            return False

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

        pscratch = os.environ.get('PSCRATCH', '/tmp')
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
        pscratch = os.environ.get('PSCRATCH', '/tmp')
        self.case_dir = Path(pscratch) / "E3SMv3" / self.case_name
        ref_name = Path(ref_case_dir).name
        logger.info(f"Cloning from {ref_name} → {self.case_name}")
        scripts_dir = Path(src_dir) / "cime" / "scripts"
        try:
            subprocess.run(
                [
                    str(scripts_dir / "create_clone"),
                    "--case",       str(self.case_dir),
                    "--case2clone", str(ref_case_dir),
                    "--keepexe",
                ],
                cwd            = scripts_dir,
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
            3. RUNDIR (always — local to each case)
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
            self._xmlchange('EXEROOT', str(self.case_dir / "build"))

        # Step 3 — RUNDIR (always, local to this case)
        self._xmlchange('RUNDIR', str(self.case_dir / "run"))

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
        elm_namelist = (
            f"fsurdat = '{fsurdat}'\n"
            "hist_empty_htapes = .true.\n"
            "mksrf_lsmlon = 1\n"
            "mksrf_lsmlat = 1\n"
            "create_crop_landunit = .true.\n"
            "hist_fincl1 = "
            "'RAIN','QOVER','QDRAI','QCHARGE',"
            "'TWS','H2OSOI','SOILLIQ','ZWT','WA'\n"
            "hist_nhtfrq = -3\n"
            "hist_mfilt  = 365\n"
        )

        namelists = {
            'elm':    elm_namelist,
            'mosart': FIXED_NAMELISTS['mosart'],
            'datm':   FIXED_NAMELISTS['datm'],
        }

        for name, content in namelists.items():
            nl_file = self.case_dir / f"user_nl_{name}"
            nl_file.write_text(content)
            logger.info(f"Wrote user_nl_{name}")

    def _build_case(self):
        """Run case.setup followed by case.build."""
        try:
            logger.info("Running case.setup...")
            subprocess.run(
                ['./case.setup'],
                cwd            = self.case_dir,
                check          = True,
                capture_output = True,
                text           = True,
            )

            logger.info("Running case.build...")
            start = time.time()
            subprocess.run(
                ['./case.build'],
                cwd            = self.case_dir,
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