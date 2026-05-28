"""Smoke test for the cloning wrapper. Doesn't actually run case.build —
just verifies the wrapper accepts ref_case_dir parameter and routes
to _clone_case() correctly."""
import sys
from unittest.mock import patch, MagicMock
sys.path.insert(0, "src")

from core.elm_wrapper import GeneratedELMAgent

def test_prepare_case_no_ref_calls_create():
    """Without ref_case_dir, prepare_case takes the fresh-build path."""
    w = GeneratedELMAgent.__new__(GeneratedELMAgent)  # bypass __init__
    w.runtime_config = {}
    w.case_suffix = "test"
    w.case_dir = None
    w.case_name = None

    with patch.object(w, '_create_case') as create, \
         patch.object(w, '_configure_case') as cfg, \
         patch.object(w, '_write_namelists'), \
         patch.object(w, '_build_case') as build:
        w.case_dir = "/tmp/fake_case"
        w.prepare_case()
        create.assert_called_once()
        build.assert_called_once()
        cfg.assert_called_once_with()  # no runtime_only kwarg

def test_prepare_case_with_ref_calls_clone():
    """With ref_case_dir, prepare_case takes the clone path."""
    w = GeneratedELMAgent.__new__(GeneratedELMAgent)
    w.runtime_config = {}
    w.case_suffix = "test"
    w.case_dir = None
    w.case_name = None

    with patch.object(w, '_clone_case') as clone, \
         patch.object(w, '_configure_case') as cfg, \
         patch.object(w, '_write_namelists'), \
         patch.object(w, '_build_case') as build:
        w.case_dir = "/tmp/fake_case"
        w.prepare_case(ref_case_dir="/tmp/ref_case")
        clone.assert_called_once_with("/tmp/ref_case")
        cfg.assert_called_once_with(runtime_only=True)
        build.assert_not_called()