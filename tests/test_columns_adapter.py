"""Offline unit tests for the Tier-2->Tier-3 columns->ELM-plan adapter."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.columns_to_plan import columns_to_elm_plan   # noqa: E402

COLS = [
    {"id": "col_01", "lat": 46.7, "lon": -120.7, "elevation_m": 447, "band": 1,
     "soil_profile": {"num_layers": 3, "layers": [{"texture_class": "loam"}]}},
    {"id": "col_02", "lat": 46.9, "lon": -121.2, "elevation_m": 1200, "band": 3},  # no soil
]


class TestColumnsToElmPlan:
    def test_one_coupler_per_column(self):
        p = columns_to_elm_plan(COLS)
        assert len(p["CONDITIONS_COUPLERS"]) == 2
        assert "ELM_CONFIG" in p

    def test_per_column_latlon_and_soil(self):
        c0 = columns_to_elm_plan(COLS)["CONDITIONS_COUPLERS"][0]
        assert c0["lat"] == 46.7 and c0["lon"] == -120.7
        assert c0["EXPERIMENT"] == "col_01"
        assert c0["soil_profile"]["num_layers"] == 3

    def test_stop_n_from_years(self):
        c0 = columns_to_elm_plan(COLS, yr_start=1995, yr_end=1999)["CONDITIONS_COUPLERS"][0]
        assert c0["STOP_N"] == "5"
        assert c0["DATM_CLMNCEP_YR_START"] == "1995"
        assert c0["RUN_STARTDATE"] == "1995-01-01"

    def test_native_adds_substrate(self):
        c0 = columns_to_elm_plan(COLS, soil_config="native")["CONDITIONS_COUPLERS"][0]
        assert c0["SOIL_CONFIG"] == "native" and "SUBSTRATE" in c0

    def test_column_without_soil_omits_it(self):
        # col_02 has no soil_profile -> coupler omits it (builder falls back)
        c1 = columns_to_elm_plan(COLS)["CONDITIONS_COUPLERS"][1]
        assert "soil_profile" not in c1 and c1["lat"] == 46.9

    def test_synthetic_soil_no_substrate_no_profile(self):
        c0 = columns_to_elm_plan(COLS, soil_config="loamy")["CONDITIONS_COUPLERS"][0]
        assert c0["SOIL_CONFIG"] == "loamy"
        assert "SUBSTRATE" not in c0 and "soil_profile" not in c0
