#!/usr/bin/env python3
"""
ELM Input Agent (Adapter)
src/core/elm_input_agent.py

Adapter that wraps GeneratedELMAgent to satisfy the
ModelAgentBase contract used by the framework.

Single responsibility: translate between
    framework conventions  ←→  ELM wrapper conventions

The wrapper (core.elm_wrapper.GeneratedELMAgent) does the actual
work — driving create_newcase, xmlchange, case.build, srun.
This adapter makes ELM polymorphically interchangeable with
PFLOTRANInputAgent (and any future model agent) via the
ModelAgentBase contract.
"""
import logging
from typing import Optional, Dict, Any

from core.model_agent_base import ModelAgentBase
from core.elm_wrapper import GeneratedELMAgent


# Backward-compat shim for tests/legacy callers that import this flag.
# The wrapper import above is unconditional — a real failure raises
# ImportError at module load rather than silently setting this to False.
# This constant can be removed once tests are updated (step 1.1).
ELM_AVAILABLE = True

logger = logging.getLogger(__name__)


class ELMAgentAdapter(ModelAgentBase):
    """
    Adapter wrapping GeneratedELMAgent to satisfy ModelAgentBase.

    Usage:
        adapter = ELMAgentAdapter(
            case_name      = 'elm_baseline',
            runtime_config = {
                'STOP_N':                '5',
                'DATM_CLMNCEP_YR_START': '1981',
                'DATM_CLMNCEP_YR_END':   '1985',
                'RUN_STARTDATE':         '1981-01-01',
            }
        )
        adapter.prepare_case(output_dir)
        adapter.run_simulation()
        summary = adapter.get_run_summary()
    """

    def __init__(self,
                 case_name:      str,
                 runtime_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            case_name:      experiment name from planner
            runtime_config: ELM runtime parameters
                            (see RUNTIME_KEYS in elm_wrapper
                            for allowed keys)
        """
        self.case_name      = case_name
        self.runtime_config = runtime_config or {}
        self._status        = 'unknown'

        self._elm = GeneratedELMAgent(
            case_suffix    = case_name,
            runtime_config = runtime_config,
        )

    # ── Abstract property ─────────────────────────────────────
    @property
    def model_type(self) -> str:
        return 'elm'

    # ── Abstract method 1: prepare_case ───────────────────────
    def prepare_case(self,
                     output_dir:   Optional[str] = None,
                     ref_case_dir: Optional[str] = None) -> str:
        """
        Prepare ELM case. If ref_case_dir is provided, clone from it
        with --keepexe (fast, no compile). Otherwise fresh build.
        """
        try:
            case_dir = self._elm.prepare_case(ref_case_dir=ref_case_dir)
            return str(case_dir)
        except Exception as e:
            raise RuntimeError(
                f"ELM prepare_case failed for "
                f"{getattr(self._elm, 'case_name', '?')}: {e}"
            ) from e

    # ── Abstract method 2: run_simulation ─────────────────────
    def run_simulation(self,
                       exe_path: Optional[str] = None) -> bool:
        """
        Run the ELM simulation via srun (blocking; requires an
        interactive node allocation).

        exe_path is unused — the wrapper uses the executable
        produced by case.build. The parameter is kept to satisfy
        the ModelAgentBase contract.
        """
        if not self.is_ready():
            logger.warning(
                f"'{self.case_name}' is not built. "
                f"Call prepare_case() first."
            )
            return False

        logger.info(f"Running ELM: {self.case_name}")
        try:
            success = self._elm.run_simulation()
        except Exception as e:
            logger.error(f"run_simulation raised: {e}")
            self._status = 'failed'
            return False

        self._status = 'completed' if success else 'failed'
        logger.info(f"{self._status}: {self.case_name}")
        return success

    # ── Abstract method 3: get_run_summary ────────────────────
    def get_run_summary(self) -> dict:
        """Return standardized run summary for the framework."""
        wrapper_info = self._elm.get_summary()

        summary = {
            # Required by ModelAgentBase
            'case_name':  self.case_name,
            'case_dir':   wrapper_info.get('case_dir') or 'not_prepared',
            'status':     self._status,
            'model_type': self.model_type,
            # ELM-specific
            'runtime_config': wrapper_info.get('runtime_config', {}),
            'history_files':  wrapper_info.get('history_files', []),
            'elm_case_info':  wrapper_info,
        }
        self.validate_run_summary(summary)
        return summary

    # ── Override is_ready ─────────────────────────────────────
    def is_ready(self) -> bool:
        """True iff the underlying ELM case has been built."""
        return self._elm.is_built