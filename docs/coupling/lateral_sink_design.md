# Lateral sink for the coupled column: design and contract

Date 2026-09-04. Approved scope (user): a lateral sink as the outlet, with QCHARGE as
the forward flux in scope. This file is the contract the implementation is built
against, in both repositories. Every physical claim below was read in the PFLOTRAN
v7.0 source on Compy, cited by file and line; nothing is from memory.

## 1. Why

A sealed column has nowhere to put the drainage ELM sends it, so its water table
climbs without bound (Naches 1979: the five wettest columns filled to the surface
during snowmelt and left the walk at days 90 to 151). In a real hillslope the
missing destination is sideways: groundwater flows toward the nearest stream,
faster the higher the water table stands above the stream level. That gives a
self-limiting balance and produces a quantity the sealed column cannot: a
baseflow per column, which can be compared (never "validated") in basin sum
against the USGS gauges the framework already fetches.

## 2. Mechanism in PFLOTRAN (verified in source)

Boundary condition, written by the deck builder:

    REGION lateral
      FACE WEST
      COORDINATES
        0.d0 0.d0 <z_lo>
        0.d0 1.d0 <z_hi>
      /
    END
    FLOW_CONDITION lateral_drain
      TYPE
        LIQUID_PRESSURE DIRICHLET_CONDUCTANCE
      /
      CONDUCTANCE <C>
      LIQUID_PRESSURE 101325.d0
    END
    BOUNDARY_CONDITION lateral_sink
      FLOW_CONDITION lateral_drain
      REGION lateral
    END

Facts that make this the right block:

- Flux per face: q = C * (kr/mu)_cell * (p_cell - p_boundary) * area, with C
  replacing perm/dist (richards_common.F90:743-748, 781-788). C therefore has
  units of metres (permeability per unit distance); in-tree decks use 1e-12 and
  1e-13.
- Outflow only: the seepage clamp zeroes inflow whenever the boundary pressure is
  at or below the reference pressure 101325 Pa (richards_common.F90:769-779,
  option_flow.F90:143). With LIQUID_PRESSURE 101325 the face leaks only while the
  cell has positive pressure head, always outward.
- No gravity term on a side face: dist(1:3) is along x, so dphi = p_boundary -
  p_cell exactly (grid_structured.F90:1153-1167).
- The region box selects whole cells whose span contains the (nudged) edges
  (grid.F90:1842-1958); put z_lo on a cell face.
- Mass balance: with OUTPUT / MASS_BALANCE_FILE the file <cid>-mas.dat carries,
  per named coupler, "lateral_sink Water Mass [kg]" (cumulative) and
  "lateral_sink Water Mass [kg/y]" (rate), outflow NEGATIVE
  (output_observation.F90:2598-2622, 3285-3320). All couplers must be named
  (patch.F90:298-303); ours are.
- Checkpoint: the cumulative BC mass is not checkpointed and an existing
  -mas.dat is appended on restart (checkpoint.F90:1073-1157,
  output_observation.F90:53-64). Every walk window is its own case directory, so
  each window's file starts at zero: the last cumulative value in a window's
  file is that window's total.
- Restart chain: the BC set may differ between legs (precedent: anchored spin,
  then sealed windows from its checkpoint). A constant CONDUCTANCE carries no
  dataset, so nothing needs the t = 0 pad.

## 3. Band and the mapping onto a recession timescale

The sink acts on a thin band of side faces from z_lo = H - datum_depth up to
z_hi = z_lo + B (B = sink_band_m, default 1.0 m, two 0.5 m cells). For a water
table standing h above the datum with h >> B and saturated band (kr = 1):

    Q = C * (rho g / mu) * B * h        [m^3/s per 1 m^2 plan area]

which is a linear reservoir, the standard baseflow model, with a smooth
quadratic onset for h < B. Matching the observed recession Q = Sy * h / tau gives

    C = Sy * mu / (tau * rho * g * B)   mu = 1.002e-3 Pa s, rho = 998 kg/m^3, g = 9.81

Magnitudes (outflow at 1 m of head, in mm/day, equals Sy * 1000 / tau):
Naches tau 30 d: Sy 0.05 -> C 1.97e-15 m, 1.67 mm/day; Sy 0.10 -> 3.95e-15 m,
3.33 mm/day; Sy 0.20 -> 7.90e-15 m, 6.67 mm/day. Brandywine tau 35 d: Sy 0.10 ->
3.38e-15 m, 2.86 mm/day. Against ELM drainage of 0.1 to 3 mm/day this puts the
equilibrium water table 0.03 to 1 m above the datum.

Why not a hillslope (Dupuit) conductance k / L^2 from the column's own
permeability and the path length to the stream: at 1-km path lengths it yields
0.003 mm/day at 5 m of head, three orders of magnitude below the recession the
gauges show. The recession is the observed truth for the basin and is used as
such. The per-column implied timescale is still recorded for inspection.

## 4. Parameters per column (all recorded on columns.json and walk.json)

- sink_datum_m: depth below the surface of the drainage datum. Two sources, both
  computed and recorded, one chosen by the dial --sink-datum {hand,conus2}:
  * hand: height above nearest drainage from D8 flow routing on the 1-km CONUS
    TOPO already on /compyfs (the surfdata the warm start is cut from;
    /compyfs/bish218/conus1k/netcdf/surfdata_conus_1k_small_lat{N}_*.nc, TOPO
    (240, 6960) per band, 1/120 degree). Priority-flood pit fill, D8 receivers,
    elevation-ordered accumulation, stream = accumulation >= A km^2, HAND =
    TOPO(column cell) - TOPO(first stream cell along the path). A is fixed ONCE
    per basin by the gauge check (the in-basin gauges must land on stream cells
    and the stream-cell TOPO must match their altitude_m within the cell-mean
    error); A is recorded. Floor: sink_datum_m = max(HAND, 0.5 * STD_ELEV),
    STD_ELEV being the column's own surfdata sub-grid relief, because a 1-km cell
    mean cannot resolve the channel incision of a valley cell (four Naches
    columns, including the wettest, col_10, sit on stream cells with HAND 0).
    The floor is a stated assumption, not a measurement.
  * conus2: the CONUS2 steady-state water-table depth already sampled per column
    (columns.json water_table_m from wtd_conus2.tif).
- sink_datum_applied_m = min(max(sink_datum_m, B), H - B): the band must lie
  inside the column. sink_datum_clipped_to_domain is true when the deep clamp
  binds (the column is shorter than its height above the stream; draining at
  the base is the best a short column can do; this is where the parked
  domain-depth rule shows). sink_datum_raised_to_band is true when the shallow
  clamp binds: a datum above B would put the band's top above the surface and
  thin it below the B the conductance was mapped with. CONUS2 gives exactly
  that in valley cells (job 778434, 2026-09-13: nine of the 18 Naches columns
  at 0 to 0.05 m, and the deck builder refused the 0.0), so with the conus2
  source those columns drain through their top metre: a discharge cell whose
  water table cannot stand above 1 m for long.
- sink_band_m: B, default 1.0.
- sink_sy: default min(0.2, porosity of the column's own material at the datum
  depth); 0.2 is ELM's own aquifer specific yield (set_water_table.py:23).
  Source recorded; dial --sink-sy overrides.
- sink_tau_days: REQUIRED when a sink is requested, no hidden default. Computed
  by tools/recession_tau.py from the run's gauge records: master recession on
  falling segments (rain days and, where SNOTEL is on disk, melt days excluded;
  regulated gauges excluded by name), both the log-linear fit and the dQ/dt
  method reported. Reference values from the reader pass: Naches 30 d (American
  River, unregulated; the main stem is regulated and excluded), Brandywine 35 d.
  Source recorded.
- sink_conductance_m: C from the formula above, recorded with every input.
- Also recorded: hand_m, hand_path_km, hand_stream_area_km2, hand_threshold_km2,
  std_elev_m, conus2_wtd_m, sink_implied_tau_days (at the column's current head).

## 5. Forward flux: QCHARGE

- --forward {qdrai,qcharge} on walk_setup, stored as walk.json forward_var, read
  once in walk_job beside pf_bottom. Default stays qdrai so existing walk
  directories keep their meaning; the new experiments pass qcharge explicitly.
- ELM sign: QCHARGE positive = into the aquifer (SoilWaterMovementMod.F90:759).
  It is already extracted 3-hourly to daily in mm/day for every run.
- Spin rate: the mean of the forward variable over the ELM source run's
  extracted.json (after the negative policy), not the PF run's daily_flux_mm_day
  (that array is QDRAI).
- Negative days: --negative-forward {clip,pass}, default clip. A negative top
  flux extracts water at an unsaturated top face, double-counting the capillary
  rise ELM already resolved, and works against the solver; clip records the
  clipped total per window per column (forward_clipped_mm). pass exists for the
  experiment that wants it.
- Empty series (the Naches col_04 case: QCHARGE absent from the extract): a
  recorded departure before the window loop, reason "ELM wrote no QCHARGE for
  this column", never a traceback. The framework leg's empty-series raise
  (pflotran_exp_manager.py:369-374) moves onto the existing skip path (7825938).

## 6. Fields, meanings and prose (in code, never behind a tool call)

- Server extract_column_series adds, per column that has a sink, a block
  lateral_outflow = {times_y, cumulative_kg (as written, negative = out),
  rate_kg_y, outflow_mm_day (sign flipped: positive = leaving), window_total_mm}
  with a units entry, read from <cid>-mas.dat columns "lateral_sink Water Mass
  [kg]" and "[kg/y]". The column's plan area is 1 m^2, so 1 kg of water is
  1 mm; kg/y to mm/day divides by 365.25. Absent file or column: block absent,
  not zero.
- Framework FIELD_SEMANTICS gains lateral_outflow_mm_day and
  lateral_outflow_window_mm with their origin; _not_computed.drainage_flux keeps
  its entry with the note that lateral outflow is computed when a sink is
  present. The two stale sentences calling the bottom a fixed head
  (FIELD_SEMANTICS.water_table_m; compare/water_table._boundary_condition) are
  rewritten to describe the three bottom states: anchored, sealed, sealed with
  lateral sink.
- Keysets: every new columns.json scalar is classified keep+optional; arrays go
  to drop with a reason, like daily_flux_mm_day.
- Capability prose that a sink makes stale (prompts encode capabilities):
  server _constraints (the sentence claiming bottom='none' is free-draining is
  already wrong: 'none' seals), the conceptual menu's always_true "anchored, not
  held", and planner_capability_probe.txt lines 62-63 and 116 ("no lateral
  transport"): amend to "a per-column lateral sink to a drainage datum is
  available; there is still no column-to-column transport".

## 7. Files and ownership

Server, reaction_sandbox_mcp-upstream (branch compy-port):
tools/column_builder.py (build_column_deck kwargs lateral_sink_depth_m,
lateral_sink_band_m, lateral_sink_conductance; refusals in the 409-413 style;
region, condition, coupler appended AFTER set_layer_order at line 400 and after
the bottom branch at 447; mass_balance True whenever a sink exists; result
["lateral_sink"] with boundary_name and the two -mas.dat column names),
tools/pflotran_input_agent.py (_write_flow_conditions: a CONDUCTANCE branch),
server.py (create_column_deck signature, docstring and forwarding;
create_decks_from_columns per-column keys lateral_sink_depth_m /
lateral_sink_band_m / lateral_sink_conductance forwarded like recharge_mm_yr;
extract_column_series -mas.dat block plus units; _constraints prose),
tools/conceptual.py prose, tests/test_column_tools.py (guards beside the
sealed-bottom test; one PFLOTRAN_EXECUTABLE-gated short run allowed).

Framework, multi-agent-framework (branch compy-port-agentic-pipeline):
D: tools/drainage_datum.py, tools/recession_tau.py, tests/test_drainage_datum.py,
tests/test_recession_tau.py (prototypes in the session scratchpad:
hand_estimate.py, recession.py).
W: tools/walk_setup.py, tools/walk_job.py, tools/walk_lib.py,
mcp/pflotran-mcp/pflotran_exp_manager.py, src/agents/analysis/compare (prose),
src/agents/prompts/planner_capability_probe.txt, tests/test_walk.py,
tests/test_pflotran_coupling.py.

Contract keys, columns.json per column: sink_datum_source, sink_datum_m,
sink_datum_applied_m, sink_datum_clipped_to_domain, sink_datum_raised_to_band,
sink_band_m, sink_sy,
sink_sy_source, sink_tau_days, sink_tau_source, sink_conductance_m,
sink_implied_tau_days, hand_m, hand_path_km, hand_stream_area_km2,
hand_threshold_km2, std_elev_m, conus2_wtd_m.
walk.json dials: forward_var, negative_forward, pf_bottom, sink (a dict of the
dials as given). Per window per column row: forward_mm_window,
forward_clipped_mm, lateral_outflow_window_mm, water_table_m.

## 8. Experiment plan (every submission is asked first; every job id announced)

1. Brandywine 2010, monthly, sealed + sink, --forward qcharge; twin with qdrai
   if cheap. Compare with walk_monthly_brandywine_sealed: water table bounded,
   outflow seasonal, basin sum of column outflow against the gauges' low-flow
   months (a comparison).
2. Naches 1979, monthly, sealed + sink, qcharge: the five columns that filled to
   the surface should hold; the nine dead columns receive QCHARGE and are
   watched for revival.

## 9. Stated open items

- The 1-km HAND cannot see the channel a valley column drains to; the STD_ELEV
  floor is the assumption that covers it, and the conus2 dial is the alternative.
- The domain-depth rule is still parked; sink_datum_clipped_to_domain marks
  every column where it binds.
- Brandywine tau rests on one year; the Naches main-stem gauges are regulated
  and excluded from tau.
