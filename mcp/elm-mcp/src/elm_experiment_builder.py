#!/usr/bin/env python3
"""
ELM Experiment Builder
mcp/elm-mcp/src/elm_experiment_builder.py

Translates plan JSON from the Planner into a list of
ELMAgentAdapter instances, one per experiment.
"""
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any

from elm_input_agent import ELMAgentAdapter, ELM_AVAILABLE

# Optional generators for per-experiment domain/surface files
try:
    from elm_domain_generator import ELMDomainGenerator
    from elm_surface_generator import (
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

    def build_cases(self, output_dir: str) -> List[str]:
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
        logger.info("BUILDING ELM CASES")
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
            logger.info(f"1 case built")
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
        logger.info(f"{n_ok}/{len(self.experiments)} cases built")
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

        # Runtime config passed to the adapter. A coupler entry may carry its
        # own STOP_OPTION/REST_* — a case that will run in SLICES (the
        # walk-through-the-year coupling) is built with STOP_OPTION='ndays'
        # and REST pinned to the same window, so every slice ends on a
        # restart the next slice (possibly stamped) continues from.
        runtime_config = {
            'STOP_N':                stop_n,
            'STOP_OPTION':           str(coupler.get(
                'STOP_OPTION', elm_cfg.get('base_stop_option', 'nyears'))),
            'DATM_CLMNCEP_YR_START': yr_start,
            'DATM_CLMNCEP_YR_END':   yr_end,
            'RUN_STARTDATE':         start_date,
            'REST_N':                str(coupler.get(
                'REST_N', elm_cfg.get('base_rest_n', '1'))),
            'REST_OPTION':           str(coupler.get(
                'REST_OPTION', elm_cfg.get('base_rest_option', 'nyears'))),
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
                    # The column's soil IS the experiment, so mcp_data must be
                    # mapped onto ELM's levels rather than discarded.
                    prescribed_soil  = coupler.get('SOIL_SOURCE') == 'prescribed',
                )
            )

        # A PRESCRIBED SOIL THAT DID NOT REACH ELM IS NOT A WARNING. Surface
        # generation is wrapped in try/except below and only logs, so a failure
        # leaves FSURDAT unset and the wrapper falls back to the one fixed
        # station surfdata. On a site run that is a degraded column; on a
        # texture sweep it is EVERY column getting the same soil, four runs
        # that differ in nothing, and a clay gradient reported from labels
        # alone. Caught here because the sweep is the only caller for whom the
        # generated surface IS the experiment.
        if coupler.get('SOIL_SOURCE') == 'prescribed' and not runtime_config.get('FSURDAT'):
            raise RuntimeError(
                f"{name}: the prescribed soil profile never became a surface "
                f"dataset — surface generation failed and FSURDAT is unset, so "
                f"this column would run on the default station soil and the "
                f"sweep would have no gradient in it. See the "
                f"'Surface generation failed' warning above for the cause.")

        # NOT A RUNTIME KEY, deliberately. RUNTIME_KEYS is the set of CIME
        # variables the wrapper can xmlchange or write into a namelist; a fill
        # spec is neither. Passing it alongside keeps that validation exact —
        # adding it to RUNTIME_KEYS would mean excluding it from XML_RUNTIME_KEYS
        # by hand, and the next key added would inherit the exception.
        adapter = ELMAgentAdapter(
            case_name          = case_name,
            runtime_config     = runtime_config,
            prescribed_weather = coupler.get('PRESCRIBED_WEATHER'),
        )
        if coupler.get('PRESCRIBED_WEATHER'):
            logger.info(f"   weather : written "
                        f"({coupler['PRESCRIBED_WEATHER']})")
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
            # Carried as data, not only inside the adapter. case_inputs.json is
            # what a rebuild reads, and a case rebuilt without this would run on
            # the real NLDAS cell while every record said the weather was
            # written — the same class of silent divergence FINIDAT had.
            'prescribed_weather': coupler.get('PRESCRIBED_WEATHER'),
            'elm_agent':      adapter,
        }

    def _generate_location_files(self,
                                 lat:         float,
                                 lon:         float,
                                 soil_config: str,
                                 substrate:   str,
                                 mcp_data:    Dict,
                                 surface_template: str = None,
                                 prescribed_soil: bool = False) -> Dict[str, str]:
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
            # On a SITE run a CONUS-subset template is always present, because
            # the warm start makes one. Its moisture is equilibrated against
            # that gridcell's own soil and vegetation; overwriting either makes
            # the inherited state inconsistent with its own hydraulics and
            # spends year one relaxing. On a 14-column Naches run that produced
            # five columns draining MORE than their annual precipitation, one
            # at 2.98x.
            #
            # ON A CONCEPTUAL SWEEP THERE IS NO TEMPLATE, since 2026-08-15:
            # those runs are cold, so nothing subsets a donor surfdata and
            # surface_template arrives None. Every column then falls back to
            # the one fixed SURFACE_TEMPLATE — which is the right outcome for a
            # sweep rather than a gap, because it holds vegetation IDENTICAL
            # across columns and leaves texture the only thing moving.
            # 'conus' MEANS "IGNORE mcp_data AND KEEP THE TEMPLATE'S SOIL", and
            # that is right for a site run for the reason above: the warm
            # start's moisture is equilibrated against the donor gridcell's own
            # soil, so overwriting it leaves the water inconsistent with its
            # own hydraulics.
            #
            # IT IS EXACTLY WRONG FOR A PRESCRIBED SWEEP, and was applied there
            # anyway until 2026-08-15. A texture sweep's whole content is the
            # profile in mcp_data; discarding it gave every column the template
            # soil — 24% clay for a design whose levels were 5% and 55%. The
            # column record still said `clay=5.0%` because attach_donor_soil's
            # guard faithfully preserved a profile that then reached nothing,
            # so every print, figure and JSON agreed on a gradient that existed
            # in no surface file ELM would open.
            #
            # The justification for 'conus' is entirely about inherited water.
            # A conceptual run is cold, so there is none, and the reason goes
            # with it.
            veg_source = 'template'
            soil_source = 'profile' if prescribed_soil else 'conus'
            surface_gen = ELMSurfaceGenerator(template_path=surface_template)

            # native ALWAYS goes through the generator, even with no MCP soil
            # (it then writes template soils but CORRECTED lat/lon). Falling
            # through to the raw template gives a surface whose coordinates
            # can never match the per-column domain -> ELM aborts at init
            # (surfdata/fatmgrid lon/lat mismatch).
            if soil_config == 'native':
                # NOTE the value above is 'template', not 'conus'. This comment
                # described a veg_source the code stopped passing; left as-is
                # it read as a claim that vegetation comes per-location from
                # CONUS_SURFDATA_NC, which it does not.
                # The original rationale for 'conus', kept because it is why
                # the option exists: pull real per-location vegetation
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