#!/usr/bin/env python3
"""
ELM Experiment Builder
src/core/elm_experiment_builder.py

Translates plan JSON from the Planner into a list of
ELMAgentAdapter instances, one per experiment.
"""
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any

from core.elm_input_agent import ELMAgentAdapter, ELM_AVAILABLE

# Optional generators for per-experiment domain/surface files
try:
    from core.elm_domain_generator import ELMDomainGenerator
    from core.elm_surface_generator import (
        ELMSurfaceGenerator,
        SURFACE_TEMPLATE,
    )
    GENERATORS_AVAILABLE = True
except ImportError:
    GENERATORS_AVAILABLE = False
    SURFACE_TEMPLATE     = None


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# ELM EXPERIMENT BUILDER
# ─────────────────────────────────────────────────────────────────────
class ELMExperimentBuilder:
    """
    Translates Planner output → ELMAgentAdapter list.

    Expected plan structure:
        {
            "CONDITIONS_COUPLERS": [
                {
                    "EXPERIMENT":            "elm_baseline",
                    "FORCING_PERIOD":        "baseline",
                    "STOP_N":                "5",
                    "DATM_CLMNCEP_YR_START": "1981",
                    "DATM_CLMNCEP_YR_END":   "1985",
                    "RUN_STARTDATE":         "1981-01-01",
                    "SOIL_CONFIG":           "native",
                    "SUBSTRATE":             "extrapolate",   # optional
                    "DESCRIPTION":           "..."
                }
            ],
            "ELM_CONFIG": {
                "base_stop_option": "nyears",
                "base_rest_n":      "1",
                "base_rest_option": "nyears",
                "lat":              45.0,    # optional, for location files
                "lon":              -120.0
            }
        }

    SOIL_CONFIG values: 'native' | 'sandy' | 'loamy' | 'clayey'
    SUBSTRATE values (only used when SOIL_CONFIG = 'native'):
        'template' | 'extrapolate' | 'sandy' | 'clayey'
    """

    def __init__(self, experiment_plan: Dict[str, Any]):
        if not ELM_AVAILABLE:
            raise RuntimeError("GeneratedELMAgent not available.")
        self.plan = experiment_plan
        self.experiments: List[Dict[str, Any]] = []

    # ─────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────
    def build_experiments(self) -> List[Dict[str, Any]]:
        """
        Build ELMAgentAdapter for each CONDITIONS_COUPLERS entry.

        Returns:
            List of experiment dicts, each with 'elm_agent' key.
        """
        couplers = self.plan.get('CONDITIONS_COUPLERS', [])
        if not couplers:
            raise ValueError("No CONDITIONS_COUPLERS in plan.")

        elm_cfg = self.plan.get('ELM_CONFIG', {})

        logger.info("=" * 60)
        logger.info("BUILDING ELM EXPERIMENTS")
        logger.info("=" * 60)
        logger.info(f"Experiments: {len(couplers)}")

        for idx, coupler in enumerate(couplers, start=1):
            exp = self._build_one(coupler, idx, len(couplers), elm_cfg)
            self.experiments.append(exp)

        logger.info(f"{len(self.experiments)} experiments built")
        return self.experiments

    def prepare_cases(self, output_dir: str) -> List[str]:
        """
        Call prepare_case() on each adapter.
        First case builds the executable from scratch (~8 min).
        Remaining cases clone in parallel with --keepexe (~30s each).

        Returns:
            List of case_dir paths (str), in original experiment order.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not self.experiments:
            raise RuntimeError("Call build_experiments() first.")

        logger.info("=" * 60)
        logger.info("PREPARING ELM CASES")
        logger.info("=" * 60)

        # ── First case: full build (the reference) ──
        ref_exp = self.experiments[0]
        logger.info(
            f"[1/{len(self.experiments)}] Building reference case: "
            f"{ref_exp['case_name']}"
        )
        ref_case_dir = ref_exp['elm_agent'].prepare_case(output_dir=output_dir)
        ref_exp['case_dir'] = ref_case_dir

        # ── Remaining cases: clone in parallel ──
        remaining = self.experiments[1:]
        if not remaining:
            logger.info(f"1 case prepared")
            return [ref_case_dir]

        logger.info(
            f"Cloning {len(remaining)} case(s) in parallel "
            f"with --keepexe from reference..."
        )

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(
                    exp['elm_agent'].prepare_case,
                    output_dir   = output_dir,
                    ref_case_dir = ref_case_dir,
                ): exp
                for exp in remaining
            }
            for future in as_completed(futures):
                exp = futures[future]
                try:
                    case_dir = future.result()
                    exp['case_dir'] = case_dir
                    logger.info(
                        f"   ✓ Cloned: {Path(case_dir).name}"
                    )
                except Exception as e:
                    # Expected and recovered: concurrent create_clone races on
                    # a parallel filesystem and the serial pass below retries
                    # it. Printing CIME's whole traceback here made a handled
                    # transient look like a failed run, so keep the cause to
                    # one line and save the detail in case the retry fails too.
                    exp['_clone_error'] = str(e)
                    first = next((ln.strip() for ln in reversed(str(e).splitlines())
                                  if ln.strip()), str(e))
                    logger.warning(
                        f"   ✗ Clone raced for {exp['case_name']} "
                        f"({first[:110]}) — will retry serially"
                    )
                    exp['case_dir'] = None

        # ── Serial retry pass ──
        # create_clone races on a parallel filesystem: concurrent clones hit
        # FileNotFoundError inside CIME's safe_copy -> os.utime(dst), in
        # whichever component's buildnml happens to lose (datm/drv/elm all
        # observed). The source files are fine; retrying the same clone
        # serially succeeds. Observed 2026-07-25: 3/13 parallel clones failed,
        # all 3 succeeded on serial retry (~17 s each).
        failed = [e for e in remaining if not e.get('case_dir')]
        if failed:
            logger.info(
                f"Retrying {len(failed)} failed clone(s) serially "
                f"(parallel-filesystem race)..."
            )
            for exp in failed:
                # Clear any partial case dir left by the failed attempt.
                stale = getattr(exp['elm_agent'], '_case_dir', None) or \
                        getattr(getattr(exp['elm_agent'], '_elm', None),
                                'case_dir', None)
                if stale and Path(stale).is_dir():
                    shutil.rmtree(stale, ignore_errors=True)
                try:
                    case_dir = exp['elm_agent'].prepare_case(
                        output_dir   = output_dir,
                        ref_case_dir = ref_case_dir,
                    )
                    exp['case_dir'] = case_dir
                    logger.info(f"   ✓ Cloned (retry): {Path(case_dir).name}")
                except Exception as e:
                    # Both attempts gone: now the detail earns its space,
                    # including the first failure, because this column will
                    # cold-start or be dropped and someone must diagnose it.
                    logger.error(
                        f"   ✗ Clone failed again for {exp['case_name']}: {e}"
                    )
                    if exp.get('_clone_error'):
                        logger.error(
                            f"     first attempt was: {exp['_clone_error']}"
                        )
                    exp['case_dir'] = None

        # Return in original experiment order
        case_dirs = [exp.get('case_dir') for exp in self.experiments]
        n_ok = sum(1 for c in case_dirs if c)
        logger.info(f"{n_ok}/{len(self.experiments)} cases prepared")
        return case_dirs

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Return summary dict — mirrors ExperimentBuilder interface."""
        return {
            'model_type':        'elm',
            'total_experiments': len(self.experiments),
            'experiments': [
                {
                    'scenario_index': e['scenario_index'],
                    'scenario_name':  e['scenario_name'],
                    'case_name':      e['case_name'],
                    'forcing_period': e['forcing_period'],
                    'forcing_start':  e['forcing_start'],
                    'forcing_end':    e['forcing_end'],
                    'stop_n':         e['stop_n'],
                    'soil_config':    e['soil_config'],
                    'substrate':      e['substrate'],
                    'description':    e['description'],
                    'case_dir':       e.get('case_dir', 'not_prepared'),
                }
                for e in self.experiments
            ]
        }

    # ─────────────────────────────────────────────────────────
    # PRIVATE
    # ─────────────────────────────────────────────────────────
    def _build_one(self,
                   coupler: Dict[str, Any],
                   idx:     int,
                   total:   int,
                   elm_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Build one experiment from one coupler entry."""
        name      = coupler.get('EXPERIMENT', f'elm_exp_{idx:02d}')
        case_name = name.lower().replace(' ', '_')

        yr_start = str(coupler.get('DATM_CLMNCEP_YR_START', '1981'))
        yr_end   = str(coupler.get('DATM_CLMNCEP_YR_END',   '1981'))
        stop_n   = str(coupler.get(
            'STOP_N',
            str(int(yr_end) - int(yr_start) + 1)
        ))
        start_date  = coupler.get('RUN_STARTDATE', f'{yr_start}-01-01')
        soil_config = coupler.get('SOIL_CONFIG', 'native')
        substrate   = coupler.get('SUBSTRATE',   'template')

        logger.info(f"[{idx}/{total}] {name}")
        logger.info(f"   years    : {yr_start} → {yr_end}")
        logger.info(f"   STOP_N   : {stop_n}")
        logger.info(f"   soil     : {soil_config}")
        if soil_config == 'native':
            logger.info(f"   substrate: {substrate}")

        # Runtime config passed to the adapter
        runtime_config = {
            'STOP_N':                stop_n,
            'STOP_OPTION':           elm_cfg.get('base_stop_option', 'nyears'),
            'DATM_CLMNCEP_YR_START': yr_start,
            'DATM_CLMNCEP_YR_END':   yr_end,
            'RUN_STARTDATE':         start_date,
            'REST_N':                elm_cfg.get('base_rest_n',     '1'),
            'REST_OPTION':           elm_cfg.get('base_rest_option', 'nyears'),
        }
        if coupler.get('FINIDAT'):
            runtime_config['FINIDAT'] = coupler['FINIDAT']   # warm-start initial state

        # Generate domain + surface files — per-coupler lat/lon/soil (spatial
        # columns), falling back to ELM_CONFIG (legacy single-site).
        lat = coupler.get('lat', elm_cfg.get('lat'))
        lon = coupler.get('lon', elm_cfg.get('lon'))
        soil_profile = coupler.get('soil_profile', elm_cfg.get('soil_profile', {}))

        if lat and lon and GENERATORS_AVAILABLE:
            runtime_config.update(
                self._generate_location_files(
                    lat         = float(lat),
                    lon         = float(lon),
                    soil_config = soil_config,
                    substrate   = substrate,
                    mcp_data    = soil_profile,
                    # Warm start subsets the CONUS gridcell's own surfdata and
                    # hands it over as the template, so fsurdat and finidat
                    # describe the same gridcell (ELM's check_weights gate).
                    surface_template = coupler.get('SURFACE_TEMPLATE'),
                )
            )

        adapter = ELMAgentAdapter(
            case_name      = case_name,
            runtime_config = runtime_config,
        )
        logger.info("   ELMAgentAdapter ready")

        return {
            'scenario_index': idx - 1,
            'scenario_name':  name,
            'case_name':      case_name,
            'forcing_period': coupler.get('FORCING_PERIOD', 'baseline'),
            'soil_config':    soil_config,
            'substrate':      substrate,
            'forcing_start':  int(yr_start),
            'forcing_end':    int(yr_end),
            'stop_n':         int(stop_n),
            'start_date':     start_date,
            'description':    coupler.get('DESCRIPTION', ''),
            # Place, so ELMResultsAnalyzer can relate results to terrain and
            # observations. tools/analyze_run.py reads these from the plan; the
            # manager path had no equivalent and emitted None.
            'lat':            float(lat) if lat is not None else None,
            'lon':            float(lon) if lon is not None else None,
            'elevation_m':    coupler.get('elevation_m',
                                          elm_cfg.get('elevation_m')),
            'band':           coupler.get('band'),
            'elm_agent':      adapter,
        }

    def _generate_location_files(self,
                                 lat:         float,
                                 lon:         float,
                                 soil_config: str,
                                 substrate:   str,
                                 mcp_data:    Dict,
                                 surface_template: str = None) -> Dict[str, str]:
        """
        Generate domain + surface files for a given location.

        substrate is only used when soil_config = 'native'; it controls
        what fills ELM levels deeper than MCP data covers.

        Returns dict of runtime_config additions (paths for ELM).
        """
        updates: Dict[str, str] = {}

        # Domain file
        try:
            domain_gen  = ELMDomainGenerator()
            domain_path = domain_gen.generate(lat, lon)
            domain_dir  = str(Path(domain_path).parent)
            domain_name = Path(domain_path).name

            updates['LND_DOMAIN_FILE'] = domain_name
            updates['ATM_DOMAIN_FILE'] = domain_name
            updates['LND_DOMAIN_PATH'] = domain_dir
            updates['ATM_DOMAIN_PATH'] = domain_dir
            logger.info(f"   Domain: {domain_name}")
        except Exception as e:
            logger.warning(f"Domain generation failed: {e}")

        # Surface file
        try:
            # With a CONUS-subset template the vegetation is already this
            # gridcell's own, at 1 km. Re-extracting it from the 0.5 degree
            # global file would overwrite it with a coarser mixture and break
            # the finidat/fsurdat weight agreement ELM checks.
            # Warm start is required, so a CONUS-subset template is always
            # present. Its moisture is equilibrated against that gridcell's own
            # soil and vegetation; overwriting either makes the inherited state
            # inconsistent with its own hydraulics and spends year one
            # relaxing. On a 14-column Naches run that produced five columns
            # draining MORE than their annual precipitation, one at 2.98x.
            veg_source, soil_source = 'template', 'conus'
            surface_gen = ELMSurfaceGenerator(template_path=surface_template)

            # native ALWAYS goes through the generator, even with no MCP soil
            # (it then writes template soils but CORRECTED lat/lon). Falling
            # through to the raw template gives a surface whose coordinates
            # can never match the per-column domain -> ELM aborts at init
            # (surfdata/fatmgrid lon/lat mismatch).
            if soil_config == 'native':
                # veg_source='conus': pull real per-location vegetation
                # (PCT_NAT_PFT/LAI/SAI/HEIGHT) from CONUS_SURFDATA_NC instead
                # of freezing whatever site the surface template was built
                # from. Falls back to template vegetation (logged warning)
                # if that global file isn't available in this environment.
                surface_path = surface_gen.generate_from_mcp(
                    lat        = lat,
                    lon        = lon,
                    mcp_data   = mcp_data or {},
                    substrate   = substrate,
                    veg_source  = veg_source,
                    soil_source = soil_source,
                )
            elif soil_config in ('sandy', 'loamy', 'clayey'):
                surface_path = surface_gen.generate_synthetic(
                    lat     = lat,
                    lon     = lon,
                    texture = soil_config,
                )
            else:
                surface_path = SURFACE_TEMPLATE
                logger.info("   Using default surface file")

            updates['FSURDAT'] = surface_path
            logger.info(f"   Surface: {Path(surface_path).name}")
        except Exception as e:
            logger.warning(f"Surface generation failed: {e}")

        return updates