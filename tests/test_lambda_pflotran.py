#!/usr/bin/env python3
"""
LAMBDA-PFLOTRAN — tests/test_lambda_pflotran.py

The reactive backend is a SUBCLASS of the flow one, which makes most of it
inherited and correct by construction. These pin the parts that are not:
the shorter clock, the generated species list, and the chemistry the parent
knows nothing about.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from core.lambda_pflotran_exp_manager import LambdaPFLOTRANExpManager as L  # noqa: E402
from core.pflotran_exp_manager import PFLOTRANExpManager as P              # noqa: E402


class TestItIsTheFlowBackendPlusChemistry:

    def test_it_subclasses_the_flow_manager(self):
        """The reactive deck IS the flow deck with a graft, so materialize,
        run, and the flow half of extract must not be reimplemented."""
        assert issubclass(L, P)
        assert L.NEEDS_PREPARE is False and L.NEEDS_SCHEDULER is False

    def test_it_cannot_claim_the_flow_backends_plan(self):
        """Sharing a plan key would make a PFLOTRAN plan look already
        materialized to this manager, which then builds zero experiments
        without raising."""
        assert L.PLAN_KEY != P.PLAN_KEY
        m = L.__new__(L)
        assert not m._already_executable({P.PLAN_KEY: [1]})
        assert m._already_executable({L.PLAN_KEY: [1]})

    def test_it_keeps_the_flow_semantics_and_adds_chemistry(self):
        for k in P.FIELD_SEMANTICS:
            assert k in L.FIELD_SEMANTICS, "flow metrics are still reported"
        assert "ch2o_depleted_frac" in L.FIELD_SEMANTICS
        assert "reaction_network_source" in L.FIELD_SEMANTICS


class TestTheClockIsShorterAndSaysSo:
    """Measured on framework columns, the timestep collapses to machine
    epsilon at t = 1.02, 1.35 and 1.79 y — at two domain depths, with and
    without a water table, and with the build's own regression network. Every
    collapse is above 1.0 y and at 1.0 y every water-bearing column completes,
    so the cap is 1 y. Inheriting the parent's 20 y would hang every column."""

    def _plan(self, years):
        m = L.__new__(L)
        cols = [{"id": "col_01", "lat": 38.5, "lon": -107.0,
                 "elevation_m": 2800.0, "fan_wtd_m": 3.0}]
        return m._to_run_plan({}, cols, {"years": years}, {})

    def test_twenty_years_is_clamped_to_the_ceiling(self):
        plan = self._plan(20.0)
        assert plan["pflotran_settings"]["years"] == L.YEARS_CAP
        assert all(c["years"] == L.YEARS_CAP for c in plan[L.PLAN_KEY])

    def test_the_clamp_is_recorded_not_silent(self):
        """A run that quietly simulated a quarter of what was asked would
        produce plausible output and no way to notice."""
        led = {a["key"]: a for a in self._plan(20.0)["assumptions_ledger"]}
        assert "years_capped" in led
        assert "20.0" in led["years_capped"]["value"]
        assert "1.02" in led["years_capped"]["why"]

    def test_a_request_within_the_ceiling_is_left_alone(self):
        plan = self._plan(0.5)
        assert plan["pflotran_settings"]["years"] == 0.5
        led = {a["key"] for a in plan["assumptions_ledger"]}
        assert "years_capped" not in led, "nothing was capped, so say nothing"

    def test_it_states_that_it_is_a_demo_not_a_calibration(self):
        led = {a["key"] for a in self._plan(5.0)["assumptions_ledger"]}
        assert "reaction_parameters" in led
        assert "comparability_to_flow_only" in led


class TestTheSpeciesListFollowsTheNetwork:
    """A generated network names its donors after each bin's mean carbon
    number — C28/C21/C17 for one binning of SPS_0001, against the regression
    fixture's C47/C31/C22. A deck that declares one set and constrains another
    is rejected by PFLOTRAN."""

    NET = ("# header\n"
           "<=> -0.06 C28-DONOR + 0.82 HCO3- + 1.0 BIOMASS \n"
           "<=> -0.09 C21-DONOR + 0.96 HCO3- + 1.0 BIOMASS \n"
           "<=> -0.12 C17-DONOR + 1.12 HCO3- + 1.0 BIOMASS \n")

    def test_donors_are_read_back_out_of_the_network(self, tmp_path):
        p = tmp_path / "net.txt"
        p.write_text(self.NET)
        assert L._donors_in(p) == ["C28-DONOR", "C21-DONOR", "C17-DONOR"]

    def test_donors_are_deduplicated_in_first_appearance_order(self, tmp_path):
        p = tmp_path / "net.txt"
        p.write_text(self.NET + "<=> -0.06 C28-DONOR + 1.0 BIOMASS \n")
        assert L._donors_in(p) == ["C28-DONOR", "C21-DONOR", "C17-DONOR"]

    def test_the_deck_declares_and_constrains_the_same_species(self):
        """All three places — PRIMARY_SPECIES and BOTH constraints."""
        brd = _brd()
        block = brd.chemistry_block("/abs/net.txt",
                                    donors=["C28-DONOR", "C21-DONOR"])
        for d in ("C28-DONOR", "C21-DONOR"):
            assert block.count(d) == 3, f"{d} must appear in all three blocks"
        for d in ("C47-DONOR", "C31-DONOR", "C22-DONOR"):
            assert d not in block, "the fixture's donors must not leak in"

    def test_the_default_donors_reproduce_the_regression_deck(self):
        """The refactor must not have changed the demo's deck."""
        brd = _brd()
        block = brd.chemistry_block("/abs/net.txt")
        for d in brd.DEFAULT_DONORS:
            assert block.count(d) == 3

    def test_the_deck_traps_from_the_port_survive(self):
        """PASSIVE_GAS_SPECIES not GAS_SPECIES; the sandbox nested INSIDE
        CHEMISTRY; absolute database path. See docs/reaction_mcp_port_notes."""
        block = _brd().chemistry_block("/abs/net.txt")
        assert "PASSIVE_GAS_SPECIES" in block
        assert "\nGAS_SPECIES" not in block
        assert block.index("REACTION_SANDBOX") > block.index("CHEMISTRY")
        assert block.index("REACTION_SANDBOX") < block.index("END")


class TestANetworkThatCannotMakeACorrectDeckIsRejected:
    """Donor names are NOT guaranteed unique.

    Each bin is named for its mean carbon number, ROUNDED, so two bins whose
    means round to the same integer get the same species name. Observed from
    the reaction MCP's `cumulative` binning of SPS_0001 at n_bins=3:
    C21-DONOR, C24-DONOR, C21-DONOR — with molar masses 438.9 and 464.9 for
    the two C21 rows.

    Unguarded this is silently wrong rather than loudly broken: the deck
    declares the DEDUPLICATED list, so PFLOTRAN gets two species for three
    reactions and the database two rows under one key. Two physically distinct
    organic-matter pools merge into one and the run still produces plausible
    chemistry.
    """

    GOOD = ("# header\n"
            "<=> -0.06 C28-DONOR + 0.82 HCO3- + 1.0 BIOMASS \n"
            "<=> -0.09 C21-DONOR + 0.96 HCO3- + 1.0 BIOMASS \n"
            "<=> -0.12 C17-DONOR + 1.12 HCO3- + 1.0 BIOMASS \n")
    COLLIDED = ("# header\n"
                "<=> -0.09 C21-DONOR + 0.96 HCO3- + 1.0 BIOMASS \n"
                "<=> -0.08 C24-DONOR + 0.90 HCO3- + 1.0 BIOMASS \n"
                "<=> -0.09 C21-DONOR + 0.96 HCO3- + 1.0 BIOMASS \n")

    def test_a_clean_network_passes(self, tmp_path):
        p = tmp_path / "net.txt"
        p.write_text(self.GOOD)
        assert L._validate_network(p, L._donors_in(p)) is None

    def test_a_duplicate_donor_is_caught(self, tmp_path):
        p = tmp_path / "net.txt"
        p.write_text(self.COLLIDED)
        why = L._validate_network(p, L._donors_in(p))
        assert why and "duplicate donor" in why
        assert "C21-DONOR" in why

    def test_more_reactions_than_species_is_caught(self, tmp_path):
        """The count check is the backstop for a collision the name scan
        cannot see — a pool named something other than C<n>-DONOR."""
        p = tmp_path / "net.txt"
        p.write_text(self.GOOD + "<=> -0.04 SOMETHING-ELSE + 1.0 BIOMASS \n")
        why = L._validate_network(p, L._donors_in(p))
        assert why and "reaction" in why

    def test_comments_are_not_counted_as_reactions(self, tmp_path):
        """The generated network carries a five-line provenance header."""
        p = tmp_path / "net.txt"
        p.write_text("# a\n# b\n# c\n" + self.GOOD)
        assert L._validate_network(p, L._donors_in(p)) is None


class TestTheDatabaseGetsTheGeneratedDonors:
    """PFLOTRAN needs every primary species in the thermodynamic database. The
    generated database carries ONLY the donor rows; everything else, including
    the `null` section separators the parser depends on, has to come from the
    build's own file."""

    BASE = ("'temperature points' 8 0. 25.\n"
            "'H2O' 3.0 0.0 18.0153\n"
            "'BIOMASS' 3.0 0.0 24.6\n"
            "'C47-DONOR' 3.0 0.0 670\n"
            "'C31-DONOR' 3.0 0.0 552\n"
            "'null' 0 0 0\n"
            "'CO2(aq)' 3 -1.0 'H2O'\n")

    def test_the_fixture_donors_are_replaced_not_appended(self, tmp_path):
        base = tmp_path / "lambda.dat"
        base.write_text(self.BASE)
        out = _brd().merge_donor_database(
            ["'C28-DONOR' 3.0 0.0 534.4"], tmp_path / "merged.dat",
            base_db=base)
        text = out.read_text()
        assert "C28-DONOR" in text
        assert "C47-DONOR" not in text and "C31-DONOR" not in text

    def test_the_section_separators_survive(self, tmp_path):
        """Dropping a 'null' line silently changes which section PFLOTRAN
        thinks it is reading."""
        base = tmp_path / "lambda.dat"
        base.write_text(self.BASE)
        out = _brd().merge_donor_database(
            ["'C28-DONOR' 3.0 0.0 534.4"], tmp_path / "merged.dat",
            base_db=base)
        lines = out.read_text().splitlines()
        assert lines[0].startswith("'temperature points'")
        assert "'null' 0 0 0" in lines
        assert lines.index("'C28-DONOR' 3.0 0.0 534.4") < \
            lines.index("'null' 0 0 0"), "donors belong in the primary section"
        assert "'CO2(aq)' 3 -1.0 'H2O'" in lines


class TestChemistryIsReadByNameNotPosition:

    TEC = ('TITLE = "  5.00000E+00 [y]"\n'
           'VARIABLES="X [m]","Y [m]","Z [m]","Liquid Pressure [Pa]",'
           '"Liquid Saturation","pH","Total CH2O(s) [M]","Total BIOMASS [M]",'
           '"Total O2(aq) [M]","Material ID"\n'
           'ZONE T="5", I=1, J=1, K=2, DATAPACKING=POINT\n'
           ' 0.5 0.5 1.0 2.0E+05 1.0 6.5 1.0E+02 1.0E-06 4.0E-04 1 \n'
           ' 0.5 0.5 5.0 1.5E+05 0.6 7.5 8.0E+01 3.0E-06 2.0E-04 2 \n')

    def _tec(self, tmp_path, name="c-004.tec", text=None):
        p = tmp_path / name
        p.write_text(text or self.TEC)
        return p

    def test_units_are_stripped_from_the_names(self, tmp_path):
        f = L._read_tec_named(self._tec(tmp_path))
        assert "Total CH2O(s)" in f, "callers ask without the [M] suffix"
        assert f["Total CH2O(s)"] == [100.0, 80.0]
        assert f["pH"] == [6.5, 7.5]

    def test_the_flow_columns_are_still_where_the_parent_expects(self, tmp_path):
        """The reactive .tec inserts chemistry AFTER saturation and BEFORE
        Material ID, so X/Y/Z/pressure/saturation do not move — which is why
        the inherited flow metrics are correct without modification."""
        tm, z, sat, pres = P._read_tec(self._tec(tmp_path))
        assert tm == 5.0
        assert z == [1.0, 5.0]
        assert sat == [1.0, 0.6]
        assert pres == [2.0e5, 1.5e5]

    def test_depletion_uses_the_first_output_time(self, tmp_path):
        """Not the constraint value: the deck equilibrates its initial
        condition before t=0 is written, so the constraint is not what the
        column actually started from."""
        t0 = self._tec(tmp_path, "c-000.tec",
                       self.TEC.replace("1.0E+02", "2.0E+02")
                               .replace("8.0E+01", "2.0E+02"))
        t1 = self._tec(tmp_path, "c-004.tec")
        chem = L._chemistry([t0, t1])
        m = chem["metrics"]
        assert m["ch2o_initial_M"] == pytest.approx(200.0)
        assert m["ch2o_final_M"] == pytest.approx(90.0)
        assert m["ch2o_depleted_frac"] == pytest.approx(0.55)

    def test_no_tec_files_is_empty_not_a_crash(self):
        assert L._chemistry([]) == {}

    def test_profiles_carry_one_grid_per_output_time(self, tmp_path):
        t0 = self._tec(tmp_path, "c-000.tec")
        t1 = self._tec(tmp_path, "c-004.tec")
        prof = L._chemistry([t0, t1])["profiles"]
        assert set(prof) >= {"ch2o", "biomass", "o2", "ph"}
        assert len(prof["ch2o"]) == 2 and len(prof["ch2o"][0]) == 2


def _brd():
    """build_reactive_demo, loaded the way the manager loads it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "brd_test", str(ROOT / "tools" / "build_reactive_demo.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["brd_test"] = m
    spec.loader.exec_module(m)
    return m
