#!/usr/bin/env python3
"""
Model Agent Base Class
src/core/model_agent_base.py

Abstract base class that defines the shared interface for all model agents.

Current implementations:
    - ELMAgentAdapter    (src/core/elm_input_agent.py)

Future implementations:
    - PFLOTRANInputAgent (src/core/pflotran_input_agent.py)  ← optional retrofit

Why this exists:
    ELMExpManager and workflow.py never need to know which model
    is running — they just call the same 3 methods on any agent.

Execution contract:
    prepare_case()    → sets up directories + input files (BLOCKING)
    run_simulation()  → executes model via subprocess   (BLOCKING)
    get_run_summary() → returns standardized result dict
"""
from abc  import ABC, abstractmethod
from typing import Optional

# ─────────────────────────────────────────────────────────────────────
# STANDARD RUN SUMMARY KEYS
# ─────────────────────────────────────────────────────────────────────
# Every get_run_summary() must return a dict containing AT MINIMUM
# these keys. Subclasses may add model-specific keys on top.
#
REQUIRED_SUMMARY_KEYS = {
    'case_name',    # str  — human-readable name
    'case_dir',     # str  — absolute path to case directory
    'status',       # str  — 'completed' | 'failed' | 'unknown'
    'model_type',   # str  — 'pflotran'  | 'elm'
}

# ─────────────────────────────────────────────────────────────────────
# ABSTRACT BASE CLASS
# ─────────────────────────────────────────────────────────────────────
class ModelAgentBase(ABC):
    """
    Contract that all model agents must satisfy.

    Every model agent (PFLOTRAN, ELM, future models) must implement:
        1. prepare_case()    → sets up input files + directories
        2. run_simulation()  → executes the model (BLOCKING)
        3. get_run_summary() → returns standardized result dict

    Shared utilities (free for all subclasses):
        - validate_run_summary() → checks required keys
        - is_ready()             → basic readiness check
        - model_type             → property subclass must set
    """

    # ─────────────────────────────────────────────────────────────────
    # ABSTRACT METHODS
    # Must be implemented by every subclass — no exceptions.
    # ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def prepare_case(self, output_dir: str) -> str:
        """
        Prepare simulation case.
        Creates directories and generates all input files.

        BLOCKING — returns only when preparation is complete.

        Args:
            output_dir: Base directory where case will be created.
                        For ELM      : advisory only (ELM uses PSCRATCH)
                        For PFLOTRAN : case created inside output_dir

        Returns:
            case_dir (str): Absolute path to prepared case directory.

        Raises:
            RuntimeError: If preparation fails.
        """
        pass

    @abstractmethod
    def run_simulation(self, exe_path: Optional[str] = None) -> bool:
        """
        Execute the simulation.

        BLOCKING — returns only when simulation finishes or fails.

        Args:
            exe_path: Path to model executable.
                      For PFLOTRAN : required (path to pflotran binary)
                      For ELM      : None (uses srun internally on
                                     pre-allocated interactive node)

        Returns:
            True  — simulation completed successfully
            False — simulation failed (check logs in case_dir)
        """
        pass

    @abstractmethod
    def get_run_summary(self) -> dict:
        """
        Return standardized run summary after simulation.

        Returns:
            dict with REQUIRED keys (see REQUIRED_SUMMARY_KEYS):
            {
                'case_name':  str,   # e.g. 'elm_baseline_run'
                'case_dir':   str,   # absolute path
                'status':     str,   # 'completed'|'failed'|'unknown'
                'model_type': str,   # 'elm' | 'pflotran'
                # subclasses add model-specific keys below:
                # ELM      → 'coupling_variables', 'history_files'
                # PFLOTRAN → 'newton_iterations',  'timestep_cuts'
            }

        Note:
            Call self.validate_run_summary(summary) before returning
            to ensure contract compliance.
        """
        pass

    # ─────────────────────────────────────────────────────────────────
    # ABSTRACT PROPERTY
    # Subclass must declare its model type as a class-level string.
    # ─────────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def model_type(self) -> str:
        """
        Identifies the model this agent runs.

        Returns:
            'elm'      — for ELMAgentAdapter
            'pflotran' — for PFLOTRANInputAgent (future retrofit)

        Example:
            @property
            def model_type(self) -> str:
                return 'elm'
        """
        pass

    # ─────────────────────────────────────────────────────────────────
    # SHARED UTILITIES
    # Free for all subclasses — do not override unless necessary.
    # ─────────────────────────────────────────────────────────────────

    def validate_run_summary(self, summary: dict) -> bool:
        """
        Verify run summary contains all required keys.

        Call this inside get_run_summary() before returning:
            summary = { ... }
            self.validate_run_summary(summary)   # raises if invalid
            return summary

        Args:
            summary: Dict to validate.

        Returns:
            True if valid.

        Raises:
            ValueError: If any required key is missing.
        """
        missing = REQUIRED_SUMMARY_KEYS - summary.keys()
        if missing:
            raise ValueError(
                f"{self.__class__.__name__}.get_run_summary() "
                f"is missing required keys: {missing}\n"
                f"Required: {REQUIRED_SUMMARY_KEYS}\n"
                f"Got:      {set(summary.keys())}"
            )
        # Validate status value
        valid_statuses = {'completed', 'failed', 'unknown'}
        if summary['status'] not in valid_statuses:
            raise ValueError(
                f"{self.__class__.__name__}.get_run_summary(): "
                f"'status' must be one of {valid_statuses}, "
                f"got '{summary['status']}'"
            )
        return True

    def is_ready(self) -> bool:
        """
        Check if agent is ready to call run_simulation().
        
        Base implementation always returns False.
        Subclasses should override with real state checks.

        Returns:
            True  — prepare_case() has completed successfully
            False — not yet prepared

        Example override in ELMAgentAdapter:
            def is_ready(self) -> bool:
                return self._elm.is_built
        """
        return False

    # ─────────────────────────────────────────────────────────────────
    # DUNDER
    # ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        ready = "ready" if self.is_ready() else "not ready"
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_type}, "
            f"status={ready})"
        )