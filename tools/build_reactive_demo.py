#!/usr/bin/env python3
"""
Reactive-transport demonstration deck builder (proof of concept).

Takes an existing flow-only 1-D PFLOTRAN column built by
build_pflotran_cases.py (real coordinates, SSURGO layering, Fan water table)
and adds the LAMBDA organic-matter reaction sandbox, producing a coupled
flow-and-reactive-transport deck for THIS PFLOTRAN build.

Encodes the three deck traps found during the port (see
docs/reaction_mcp_port_notes.md):
  1. PASSIVE_GAS_SPECIES, not the deprecated GAS_SPECIES;
  2. REACTION_SANDBOX nested INSIDE the CHEMISTRY block;
  3. absolute paths for the thermodynamic database and reaction network,
     because the shipped relative paths assume the regression directory.

    python3 tools/build_reactive_demo.py --column col_01 --recharge 100 10 --run
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PFL = Path(os.getenv("PFLOTRAN_DIR", "/qfs/people/tran289/pflotran"))
EXE = PFL / "src/pflotran/pflotran"
RT = PFL / "regression_tests/default/reaction_sandbox"
DB = PFL / "database/lambda.dat"
SRC_RUN = ROOT / "workflow_outputs/gunnison_pflotran/r100"
OUT = ROOT / "workflow_outputs/gunnison_reactive"

# LAMBDA sandbox parameters, taken verbatim from the build's own verified
# regression case. This is a demonstration configuration, not a calibration.
SANDBOX = """  REACTION_SANDBOX
    LAMBDA
      REACTION_NETWORK {network}
      MU_MAX 11 1/hr
      VH 1 m^3
      CC 1 M
      k_deg 1e-2 1/hr
      NH4_inhibit 1e-15 M
      SCALING_MINERAL A(s)
      INHIBITION_TYPE THRESHOLD
      CARBON_CONSUMPTION_SPECIES Carbon_Consumption
      ACTIVATION_ENERGY 1.d-2
      REFERENCE_TEMPERATURE 24.
    /
    EQUILIBRATE
      SPECIES_NAME O2(aq)
      EQUILIBRIUM_CONCENTRATION 4.06e-4
      HALF_LIFE 0.01 h
    /
  /
"""

CHEMISTRY = """#=========================== chemistry ========================================
CHEMISTRY
  PRIMARY_SPECIES
    CH2O(s)
    HCO3-
    NH4+
    HPO4--
    HS-
    H+
    O2(aq)
    BIOMASS
    A(aq)
{donor_species}  /
  SECONDARY_SPECIES
    OH-
    CO3--
    CO2(aq)
    NH3(aq)
  /
  IMMOBILE_SPECIES
    Carbon_Consumption
  /
  DECOUPLED_EQUILIBRIUM_REACTIONS
    HS-
  /
  PASSIVE_GAS_SPECIES
    CO2(g)
  /
  MINERALS
    A(s)
  /
  MINERAL_KINETICS
    A(s)
      RATE_CONSTANT 0.d0
    /
  /
  DATABASE {db}
  LOG_FORMULATION
  ACTIVITY_COEFFICIENTS OFF
  OUTPUT
    TOTAL
    PRIMARY_SPECIES
    PH
  /
{sandbox}END

#=========================== solute constraints ===============================
CONSTRAINT column
  CONCENTRATIONS
    CH2O(s)    1.1d2   T
    HCO3-      9.37d-5 T
    NH4+       1.01d-4 T
    HPO4--     1.d-10  T
    HS-        1.d-10  T
    H+         2.1379d-7 T
    O2(aq)     4.06d-4 T
    BIOMASS    1.d-6   T
    A(aq)      1.d-10  T
{donor_column}  /
  IMMOBILE
    Carbon_Consumption 1.d-10
  /
  MINERALS
    A(s) 1.d-5 1.d0
  /
END

CONSTRAINT inlet
  CONCENTRATIONS
    CH2O(s)    1.d-10  T
    HCO3-      9.37d-5 T
    NH4+       1.d-6   T
    HPO4--     1.d-10  T
    HS-        1.d-10  T
    H+         2.1379d-7 T
    O2(aq)     4.06d-4 T
    BIOMASS    1.d-10  T
    A(aq)      1.d-10  T
{donor_inlet}  /
  IMMOBILE
    Carbon_Consumption 1.d-10
  /
  MINERALS
    A(s) 1.d-5 1.d0
  /
END

TRANSPORT_CONDITION column_ic
  TYPE ZERO_GRADIENT
  CONSTRAINT_LIST
    0.d0 column
  /
END

TRANSPORT_CONDITION inlet_bc
  TYPE DIRICHLET_ZERO_GRADIENT
  CONSTRAINT_LIST
    0.d0 inlet
  /
END

TRANSPORT_CONDITION outlet_bc
  TYPE ZERO_GRADIENT
  CONSTRAINT_LIST
    0.d0 column
  /
END

"""


# The donors of the build's own verified regression network. The DEFAULT, so
# every existing caller keeps producing the identical deck.
DEFAULT_DONORS = ["C47-DONOR", "C31-DONOR", "C22-DONOR"]


def chemistry_block(network, donors=None, db=None):
    """The CHEMISTRY block, with the donor species written in three places.

    A generated reaction network names its donors after the mean carbon number
    of each bin — C28/C21/C17 for one binning of SPS_0001, against the
    regression network's C47/C31/C22 — so the donor list cannot be a constant.
    It appears in PRIMARY_SPECIES and in BOTH constraints, and a deck that
    declares one set and constrains another is rejected by PFLOTRAN.

    Kept here rather than in the Experiment Manager so the three deck traps
    from the port (PASSIVE_GAS_SPECIES, the sandbox nested inside CHEMISTRY,
    absolute paths) stay in one place.
    """
    donors = list(donors or DEFAULT_DONORS)
    return CHEMISTRY.format(
        db=(db or DB),
        sandbox=SANDBOX.format(network=network),
        donor_species="".join(f"    {d}\n" for d in donors),
        donor_column="".join(f"    {d:<10s} 1.d-4   T\n" for d in donors),
        donor_inlet="".join(f"    {d:<10s} 1.d-10  T\n" for d in donors))


def merge_donor_database(donor_lines, out_path, base_db=None):
    """The build's lambda.dat with ITS donor rows swapped for these.

    PFLOTRAN needs every primary species in the database, so a generated
    network whose donors are not in lambda.dat fails at read time. The
    generated database file carries ONLY the donor rows; everything else —
    H2O, HCO3-, the secondary species, the A(s) mineral, and the `null`
    section separators PFLOTRAN's parser depends on — has to come from the
    build's own file.
    """
    from pathlib import Path as _P
    base = _P(base_db or DB).read_text().splitlines()
    keep = [ln for ln in base if not re.match(r"\s*'C\d+-DONOR'", ln)]
    # Back where the originals were: after BIOMASS, before the first 'null'.
    at = next((i for i, ln in enumerate(keep)
               if ln.strip().startswith("'null'")), len(keep))
    merged = keep[:at] + list(donor_lines) + keep[at:]
    out_path = _P(out_path)
    out_path.write_text("\n".join(merged) + "\n")
    return out_path


def build(src_deck, dst_dir, col, recharge_mm_yr, network, years=20.0,
          donors=None, db=None):
    dst_dir.mkdir(parents=True, exist_ok=True)
    text = src_deck.read_text()

    # 1. add the transport process model alongside RICHARDS flow
    text = text.replace(
        "    SUBSURFACE_FLOW flow\n      MODE RICHARDS\n    /\n",
        "    SUBSURFACE_FLOW flow\n      MODE RICHARDS\n    /\n"
        "    SUBSURFACE_TRANSPORT transport\n      MODE GIRT\n    /\n")

    # 2. transport needs a numerical-methods block and a diffusion coefficient
    text = text.replace(
        "#=========================== discretization ",
        "NUMERICAL_METHODS TRANSPORT\n  NEWTON_SOLVER\n    NUMERICAL_JACOBIAN\n  /\n"
        "  LINEAR_SOLVER\n    SOLVER DIRECT\n  /\nEND\n\n"
        "FLUID_PROPERTY\n  DIFFUSION_COEFFICIENT 1.d-9\nEND\n\n"
        "#=========================== discretization ", 1)

    # 3. chemistry (with the sandbox nested inside it) before the flow conditions
    chem = chemistry_block(network, donors=donors, db=db)
    anchor = "FLOW_CONDITION initial"
    text = text.replace(anchor, chem + anchor, 1)

    # 4. set this scenario's recharge flux (m/y)
    text = re.sub(r"(LIQUID_FLUX LIST.*?0\.d0\s+)[\d.d+-]+",
                  lambda m: m.group(1) + f"{recharge_mm_yr / 1000.0:.4f}d0",
                  text, flags=re.S)

    # 5. couplers: give every flow condition its transport partner
    text = text.replace(
        "INITIAL_CONDITION\n  FLOW_CONDITION initial\n  REGION all\nEND",
        "INITIAL_CONDITION\n  FLOW_CONDITION initial\n"
        "  TRANSPORT_CONDITION column_ic\n  REGION all\nEND")
    text = text.replace(
        "BOUNDARY_CONDITION top_recharge\n  FLOW_CONDITION recharge\n  REGION top\nEND",
        "BOUNDARY_CONDITION top_recharge\n  FLOW_CONDITION recharge\n"
        "  TRANSPORT_CONDITION inlet_bc\n  REGION top\nEND")
    text = text.replace(
        "BOUNDARY_CONDITION bottom_wt\n  FLOW_CONDITION water_table\n  REGION bottom\nEND",
        "BOUNDARY_CONDITION bottom_wt\n  FLOW_CONDITION water_table\n"
        "  TRANSPORT_CONDITION outlet_bc\n  REGION bottom\nEND")

    name = f"{col}_r{int(recharge_mm_yr)}"
    deck = dst_dir / f"{name}.in"
    deck.write_text(text)
    return deck, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--column", default="col_01")
    ap.add_argument("--recharge", type=float, nargs="+", default=[100.0, 10.0])
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()

    src = SRC_RUN / args.column / f"{args.column}.in"
    if not src.exists():
        raise SystemExit(f"source deck not found: {src}")
    OUT.mkdir(parents=True, exist_ok=True)
    # network file travels with the decks so paths stay absolute and stable
    network = OUT / "reaction_network_lambda.txt"
    if not network.exists():
        shutil.copy2(RT / "reaction_network_lambda.txt", network)

    manifest = []
    for r in args.recharge:
        d = OUT / f"r{int(r)}"
        deck, name = build(src, d, args.column, r, network)
        rec = {"column": args.column, "recharge_mm_yr": r, "deck": str(deck)}
        print(f"  built {deck.relative_to(ROOT)}")
        if args.run:
            t0 = time.time()
            p = subprocess.run([str(EXE), "-pflotranin", deck.name],
                               cwd=deck.parent, capture_output=True,
                               text=True, timeout=1800)
            log = (deck.parent / f"{name}.log")
            log.write_text(p.stdout + p.stderr)
            ok = p.returncode == 0
            rec.update(run_ok=ok, seconds=round(time.time() - t0, 1))
            print(f"  {'OK ' if ok else 'FAIL'} {name}: {rec['seconds']}s")
            if not ok:
                tail = [l for l in (p.stdout + p.stderr).splitlines()
                        if "ERROR" in l.upper()][:3]
                for l in tail:
                    print("      " + l.strip())
        manifest.append(rec)
    (OUT / "reactive_demo.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n-> {OUT}/reactive_demo.json")


if __name__ == "__main__":
    main()
