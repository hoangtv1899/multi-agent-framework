#!/usr/bin/env python3
"""The PFLOTRAN manager's controlled-sweep path — offline, with a fake server.

The server's own tests (reaction_sandbox_mcp-upstream/tests/test_conceptual.py)
cover what a level becomes. What is pinned here is the FRAMEWORK side: that
the manager checks before it builds and stops with every reason named, that a
sweep's deck build passes no site_dir and carries the design's knobs, that
the sweep's keys are named in the column metadata (the merge raises on an
unnamed key), and that the design figure has a branch for a run with no place.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "mcp" / "pflotran-mcp"))

from pflotran_exp_manager import PFLOTRANExpManager   # noqa: E402

DESIGN = {"factors": [{"name": "water_table_m", "levels": [2, 10]}],
          "held_fixed": {"soil": "loam", "recharge_mm_yr": 300, "years": 1995,
                         "cap_m": 42}}


def _col(i, wt):
    return {"id": f"col_{i:02d}", "lat": None, "lon": None, "elevation_m": None,
            "pinned": False, "treatment": {"water_table_m": wt}, "soil": "loam",
            "substrate": None, "soil_depth_m": None, "soil_source": "textbook",
            "subsurface_profile": {"layers": [{"depth_top_m": 0, "depth_bot_m": 392,
                                               "material": "loam", "porosity": 0.43,
                                               "permeability_z": 0.0104, "vg_alpha": 3.6,
                                               "vg_n": 1.56, "sres": 0.18}]},
            "water_table_m": float(wt), "recharge_mm_yr": 300.0,
            "forcing_start": 1995, "forcing_end": 1995}


class _Server:
    """Answers the three sweep tools and the deck tool; records every call."""
    timeout = 300.0

    def __init__(self, buildable=True, columns=None):
        self.calls = []
        self.buildable = buildable
        self.columns = columns if columns is not None else [_col(1, 2), _col(2, 10)]

    def call_tool_json(self, name, args):
        self.calls.append((name, args))
        if name == "check_conceptual_design":
            return {"buildable": self.buildable, "n_columns": 2,
                    "wont_build": [] if self.buildable else
                    [{"factor": "soil", "why": "no soil anywhere"},
                     {"factor": "recharge_mm_yr / rain", "why": "say what water goes in"}],
                    "unusual": [{"factor": "held_fixed.years", "why": "no calendar year"}],
                    "facts": [{"what": "water table 2 m", "detail": "column runs to 7 m"}]}
        if name == "build_conceptual_columns":
            return {"approach": "factor_sweep", "columns": self.columns,
                    "deck_knobs": {"cap_m": 42.0}}
        if name == "create_decks_from_columns":
            rows = [dict(c, deck_status="built", depth_m=7.0, unsaturated_m=c["water_table_m"],
                         wt_in_domain=True, transient=False, n_forcing_steps=0,
                         warning="built STEADY at 300 mm/yr", case_dir=f"/x/{c['id']}",
                         input_file=f"/x/{c['id']}/{c['id']}.in")
                    for c in args["columns"]]
            return {"n_columns": len(rows), "n_built": len(rows),
                    "decks": [{"id": r["id"], "status": "built", "unsaturated_m": r["unsaturated_m"]}
                              for r in rows],
                    "run_plan": {"CONDITIONS_COUPLERS": rows, "PFLOTRAN_CONFIG": {},
                                 "assumptions_ledger": [], "incomplete": []}}
        raise AssertionError(f"unexpected tool {name}")


def _manager(tmp_path):
    m = PFLOTRANExpManager.__new__(PFLOTRANExpManager)
    m.run_dir = tmp_path
    m.input_dir = tmp_path / "01_inputs"
    m.input_dir.mkdir(exist_ok=True)
    return m


def _config(server, sweep=True):
    strategy = {"archetype": "conceptual", "sampling": {"approach": "factor_sweep"}} if sweep \
        else {"archetype": "site", "sampling": {"approach": "stratified"}}
    return {"mcp_clients": {"pflotran": server}, "strategy": strategy, "brief": {}}


class TestTheSweepIsCheckedThenBuilt:
    def test_check_precedes_build_and_knobs_are_kept(self, tmp_path):
        srv = _Server()
        m = _manager(tmp_path)
        out = m._build_sweep_columns(DESIGN, _config(srv))
        assert [c[0] for c in srv.calls] == ["check_conceptual_design", "build_conceptual_columns"]
        assert srv.calls[0][1] == {"design": DESIGN}
        assert len(out["columns"]) == 2
        assert m._deck_knobs == {"cap_m": 42.0}

    def test_a_refused_design_stops_with_every_reason(self, tmp_path):
        srv = _Server(buildable=False)
        m = _manager(tmp_path)
        with pytest.raises(RuntimeError) as e:
            m._build_sweep_columns(DESIGN, _config(srv))
        msg = str(e.value)
        assert "refused this sweep design" in msg
        assert "no soil anywhere" in msg and "say what water goes in" in msg
        assert [c[0] for c in srv.calls] == ["check_conceptual_design"], "no build after a refusal"

    def test_a_buildable_design_that_builds_nothing_raises(self, tmp_path):
        srv = _Server(columns=[])
        with pytest.raises(RuntimeError, match="returned no columns"):
            _manager(tmp_path)._build_sweep_columns(DESIGN, _config(srv))

    def test_no_client_is_a_clear_refusal(self, tmp_path):
        with pytest.raises(RuntimeError, match="MCP is required"):
            _manager(tmp_path)._build_sweep_columns(DESIGN, {"mcp_clients": {}, "strategy": {}})


class TestTheDeckBuildKnowsItIsASweep:
    def test_no_site_dir_and_the_designs_knobs(self, tmp_path):
        srv = _Server()
        m = _manager(tmp_path)
        m._build_sweep_columns(DESIGN, _config(srv))
        cols = [dict(c) for c in srv.columns]
        m._refine_columns(cols, _config(srv))
        name, args = srv.calls[-1]
        assert name == "create_decks_from_columns"
        assert "site_dir" not in args, "a sweep has no site to look up"
        assert args["cap_m"] == 42.0
        # the columns are replaced in place with the rows as built
        assert cols[0]["deck_status"] == "built" and cols[0]["treatment"] == {"water_table_m": 2}

    def test_a_site_run_still_passes_the_run_dir(self, tmp_path):
        srv = _Server()
        m = _manager(tmp_path)
        m._refine_columns([_col(1, 2)], _config(srv, sweep=False))
        name, args = srv.calls[-1]
        assert args["site_dir"] == str(tmp_path)


class TestTheSweepsKeysAreNamed:
    def test_a_sweep_row_passes_the_key_check(self, tmp_path):
        """The merge raises on a key nobody named. Every key a sweep row
        carries — the sweep's own facts and its arrays — is named keep or drop."""
        m = _manager(tmp_path)
        keys = m._column_keys()
        row = dict(_col(1, 2), weather={"fill": "seasonal", "mm_yr": 500, "peak_doy": 15},
                   precipitation_mm_day=[1.0] * 365, deck_status="built", depth_m=7.0,
                   unsaturated_m=2.0, wt_in_domain=True, transient=True, n_forcing_steps=365,
                   warning=None, case_dir="/x", input_file="/x/c.in", n_cells=14,
                   domain_why="rule", forcing_caveat="c", cell_dz_m={}, max_cell_m=0.5,
                   n_material_zones=1, source="s")
        keys.check(row)                                   # raises on an unnamed key
        for k in ("soil", "substrate", "soil_depth_m", "soil_source", "treatment", "weather"):
            assert k in keys.keep, k
        for k in ("subsurface_profile", "precipitation_mm_day"):
            assert k in keys.drop, k

    def test_field_semantics_explain_the_sweeps_inputs(self):
        fs = PFLOTRANExpManager.FIELD_SEMANTICS
        for k in ("soil", "substrate", "soil_depth_m", "weather"):
            assert k in fs, k
        assert "controlled sweep" in " ".join(fs["water_table_m"]["from"])


class TestTheDesignFigureHasASweepBranch:
    def test_it_draws_a_run_with_no_place(self, tmp_path):
        # through the manager's per-model loader — a bare `import sampling_design`
        # would collide with ELM's same-named module when both suites run
        sampling_design = PFLOTRANExpManager.__new__(PFLOTRANExpManager)._sibling_module("sampling_design")
        cols = []
        for i, wt in enumerate((2, 10), 1):
            c = _col(i, wt)
            c.update(deck_status="built", depth_m=7.0 if wt < 7 else 17.0, transient=True,
                     n_forcing_steps=365, weather={"fill": "storms", "mm_yr": 500, "n_storms": 12},
                     precipitation_mm_day=[0.0] * 364 + [500.0])
            c.pop("recharge_mm_yr")
            cols.append(c)
        cj = {"approach": "factor_sweep", "n_columns": 2,
              "sampling_design": {"approach": "factor_sweep",
                                  "factors": [{"name": "water_table_m", "levels": [2, 10]}],
                                  "held_fixed": {"soil": "loam", "rain": {"fill": "storms", "mm_yr": 500}}},
              "columns": cols}
        (tmp_path / "columns.json").write_text(json.dumps(cj))
        png = sampling_design.render_run(tmp_path, out=tmp_path / "d.png")
        assert Path(png).exists() and Path(png).stat().st_size > 10_000
