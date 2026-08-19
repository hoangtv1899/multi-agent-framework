#!/usr/bin/env python3
"""
PFLOTRAN Experiment Manager — declarations, and three pass-throughs
mcp/pflotran-mcp/pflotran_exp_manager.py

    in   the sampler's columns (id, lat, lon, elevation, band)
    out  decks built, run, and read back — by the PFLOTRAN server, one tool
         call per stage

WHERE THIS FILE LIVES, AND WHY (moved 2026-08-18). It sat in src/core/ as the
one model-named file left in the framework's core. ELM's manager had already
moved beside its server (mcp/elm-mcp/src/elm_exp_manager.py); this is the same
move for the same reason: the framework's per-server code lives in one place
per server, and src/core/ names no model. The PFLOTRAN server itself is a
separate repo (reaction_sandbox_mcp-upstream, launched as the `pflotran-mcp`
console script named in mcp_config.json); this directory holds only what the
framework needs to drive it. Imported by path — see core/resumable._manager_for.

WHAT IS NOT HERE ANY MORE. The per-column join — subsurface profile, water
table and daily rain looked up off the files reception wrote — used to be
_to_run_plan, 117 lines, and was the reason this file kept growing. It is the
server's now: `create_decks_from_columns` takes the run directory and does the
lookup itself (tools/site_data.py there), exactly as ELM's build tool takes
coordinates and reads its own data. One call at materialize builds every deck
and hands back the run plan; the stages after it read what that call wrote.

WHAT IS HERE, and why it stays on this side of the boundary:

    FIELD_SEMANTICS   what each number the server extracts MEANS. Read by
                      the Analyzer as the authority on the run's outputs. The
                      framework reads what came out; the meaning of a field
                      does not go behind a tool call (decision of 2026-08-06).
    _refine_columns   the one build call, at the moment the base allows it
    _to_run_plan      hands the server's run plan back to the base
    _build_case_inputs / _run / _extract
                      the base's stage slots, each one tool call

NO SNAP. `_refine_columns` builds decks and returns the columns unchanged —
ELM's warm start moves every column onto its donor gridcell; this model has no
donor. The columns stay where the plan put them.

Reception is the only component that reaches outside the framework; every
stage here reads what reception wrote and fetches nothing.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.exp_manager_base import ExperimentManagerBase, _is_factor_sweep
from core.keyset import KeySet


class PFLOTRANExpManager(ExperimentManagerBase):
	"""Declarations plus one build call; the base does the rest."""

	MODEL = "pflotran"

	# THE SERVER THIS MANAGER DRIVES — decks, runs, extraction all go to it.
	MCP_NAME = "pflotran"

	# NEITHER STAGE APPLIES. There is no case to compile — a deck is a text file
	# — and no scheduler to wait on: a 786-cell column runs in 86 s and a 36-cell
	# one in 5 s, measured. The base reads both off the class to decide which
	# stages to walk.
	NEEDS_CASE_BUILD = False
	NEEDS_SCHEDULER = False

	# WHAT THE NUMBERS MEAN, and why this cannot be left empty. _package writes
	# it into experiment.json and step 2 hands it to an LLM as THE AUTHORITY on
	# what the run produced. The base default is {} on purpose — PFLOTRAN used
	# to inherit ELM's, so a saturation study came back described as a water
	# budget with QOVER-derived runoff fractions it never computed. An empty
	# dict is safer than a wrong one and still leaves a reader to guess, which
	# is what this fills.
	#
	# EVERY ENTRY IS A RAW FIELD. Nothing derived is listed because nothing
	# derived is written: _extract reads what PFLOTRAN wrote and stops. Front
	# depth, transit time and drainage flux are the Analyzer's to compute FROM
	# these, and a catalogue entry for a field the file does not contain would
	# be an invitation to report one that was never calculated.
	FIELD_SEMANTICS = {
		"saturation": {
			"units": "-", "from": ["Liquid Saturation"],
			"note": ("fraction of PORE SPACE filled with water, 0 to 1 — NOT a "
					 "volumetric water content. Multiply by porosity for water "
					 "content. 1.0 means saturated, and a column reading 1.0 "
					 "at every depth has no unsaturated zone to study"),
		},
		"liquid_pressure_Pa": {
			"units": "Pa", "from": ["Liquid Pressure"],
			"note": ("ABSOLUTE pressure, not head and not suction. Atmospheric "
					 "is ~101325 Pa, so a value below that is unsaturated and "
					 "one above it is below the water table"),
		},
		"depth_m": {
			"units": "m", "from": ["Z"],
			"note": ("BELOW THE LAND SURFACE, positive downward. PFLOTRAN "
					 "writes elevation upward from the domain bottom; the "
					 "extractor converts it once so nothing downstream has to"),
		},
		"times_y": {
			"units": "y", "from": ["SOLUTIONTIME"],
			"note": ("years since the simulation START, which INCLUDES the "
					 "steady spin-up. A transient column runs spin_years at "
					 "the forcing mean first, so the first snapshot is the "
					 "spun-up state and the transient period begins after it"),
		},
		"water_table_m": {
			"units": "m", "from": ["ParFlow CONUS2 ss_water_table_depth (site run)",
								   "the design's level or held_fixed (controlled sweep)",
								   "the driving model's solved mean water table (coupled run)"],
			"note": ("the INITIAL condition, not a result — the column is "
					 "initialised hydrostatic about it and its bottom face is "
					 "anchored to it. Comparing it to the final saturation "
					 "profile compares an input with an output"),
		},
		"unsaturated_m": {
			"units": "m", "from": ["water_table_m", "domain_depth_m"],
			"note": ("how much unsaturated column the run STARTED with. An "
					 "input, and the thing the study is about: a column with "
					 "0 m of it has no vertical transit to observe, however "
					 "cleanly it ran"),
		},
		"wt_in_domain": {
			"units": "true/false", "from": ["water_table_m", "domain_depth_m"],
			"note": ("whether that initial water table is INSIDE the column "
					 "at all. False means the domain was capped above it, so "
					 "water_table_m describes a place the model never "
					 "represented and nothing in the profile is measured "
					 "against it"),
		},
		"transient": {
			"units": "true/false", "from": ["the deck's FLOW_CONDITION"],
			"note": ("true when the top boundary follows the daily series, "
					 "false when it is one constant rate. Which one decides "
					 "what the output times mean: a steady column's snapshots "
					 "are one state approached, a transient column's are the "
					 "year being walked through"),
		},
		"n_forcing_steps": {
			"units": "count", "from": ["Daymet daily precipitation (site run)",
									   "the written year of rain (controlled sweep)"],
			"note": ("how many boundary values the column was driven with — "
					 "365 for a year of days. Null or 0 means a constant "
					 "boundary"),
		},
		"rain_borrowed_km": {
			"units": "km", "from": ["the nearest reception grid point's Daymet series"],
			"note": ("NULL when the column sits on a grid point of its own and "
					 "was driven with the rain fetched there. A NUMBER means "
					 "this column was placed off the grid (pinned to a station) "
					 "and took the nearest grid point's daily series from this "
					 "far away — always within one grid spacing, and said here "
					 "so nobody reads the column's rain as measured at the "
					 "station. Beyond one spacing no rain is borrowed and the "
					 "column is steady (transient false)"),
		},
		"recharge_mm_yr": {
			"units": "mm/y", "from": ["the deck tool's steady rate (site run)",
									  "the design's recharge_mm_yr (controlled sweep)"],
			"note": ("the CONSTANT top-boundary flux, for a column driven by "
					 "one. NULL ON A TRANSIENT COLUMN, where the boundary is "
					 "the 365-step series and no single rate describes it — "
					 "an absence with a reason, not a gap. Either way it is "
					 "an input: no flux is written out, so the rate that "
					 "actually reached the water table is not in this run"),
		},
		# A CONTROLLED SWEEP'S OWN INPUTS — absent on a site run.
		"soil": {
			"units": "class name", "from": ["the design's soil level or held_fixed"],
			"note": ("a USDA texture class; what it BECOMES is the Carsel & "
					 "Parrish (1988) class means (porosity, permeability, van "
					 "Genuchten alpha and n, residual saturation), one material "
					 "down the whole column unless soil_depth_m puts a "
					 "substrate under it. A textbook material, not a soil "
					 "anyone measured"),
		},
		"substrate": {
			"units": "class name", "from": ["held_fixed.substrate"],
			"note": ("the material from soil_depth_m down, from the same "
					 "table; NULL when the column is one material throughout"),
		},
		"soil_depth_m": {
			"units": "m", "from": ["the design's soil_depth_m level or held_fixed"],
			"note": ("where soil ends and substrate begins; NULL when there is "
					 "no substrate"),
		},
		"soil_source": {
			"units": "text", "from": ["build_conceptual_columns"],
			"note": "the one sentence saying where the material came from",
		},
		"weather": {
			"units": "spec", "from": ["the design's rain level or held_fixed.rain"],
			"note": ("the WRITTEN rain: {fill: uniform|seasonal|storms, mm_yr, "
					 "peak_doy|n_storms}. The 365 daily values the deck was "
					 "driven with are rebuilt from exactly this; the shape is "
					 "what a rain sweep varies at one total. NULL on a site "
					 "run, whose rain is Daymet's"),
		},
		# WHAT IS ABSENT, said explicitly. A reader looking for these will not
		# find them, and the reason is a design decision rather than a gap.
		"_not_computed": {
			"fields": ["wetting_front_depth", "transit_time", "residence_time",
					   "drainage_flux", "recharge_fraction", "water_budget"],
			"note": ("the extraction is RAW SERIES ONLY. These answer what the "
					 "study asked and are the Analyzer's to derive from the "
					 "series above — the server has particle-trajectory, "
					 "residence-time and breakthrough-curve tools for it. "
					 "Nothing in this file is any of them"),
		},
		"_forcing": {
			"note": ("the top boundary was driven with PRECIPITATION applied as "
					 "recharge — Daymet's on a site run, the design's written "
					 "rain or steady rate on a controlled sweep: no snow "
					 "storage, no evapotranspiration removed, no runoff "
					 "generated, because this model has none of them. In a "
					 "snow-dominated basin the snowpack therefore infiltrates "
					 "in winter rather than at melt, so seasonal TIMING is "
					 "wrong by construction. See the assumptions ledger, which "
					 "says which of the two this run was"),
		},
	}

	# ─────────────────────────────────────────────────────────
	# STEP 0 — BUILD THE DECKS, inside materialize
	# ─────────────────────────────────────────────────────────
	# What the framework may override on the server's build. Anything not in
	# config takes the server's own default, and the server reports the values
	# it used in run_plan.PFLOTRAN_CONFIG — so the record says what was built,
	# not what was asked for.
	BUILD_KNOBS = ("max_cell_m", "cap_m", "min_depth_m", "spin_years",
				   "recharge_mm_yr")

	def _build_sweep_columns(self, design: Dict[str, Any],
							 config: Dict[str, Any]) -> Dict[str, Any]:
		"""A controlled sweep's columns, built by the server that runs them.

		The same forty lines ELM's manager has, for the same reason: the base
		knows a sweep is one column per level combination; only this server
		knows that a `soil` level of "loam" is five van Genuchten numbers, or
		that a water table at 2 m makes a 7 m column. The design crosses the
		MCP boundary and comes back as columns in the shape the deck tool
		reads — so from here on the sweep IS the site path.

		CHECKED FIRST, AND SEPARATELY. build_conceptual_columns refuses an
		unbuildable design on its own, but calling check first means the run
		stops with every reason named at once, including the ones that would
		NOT have refused (unusual: a saturated water table, 0 mm/yr) — which
		belong in the log before compute rather than after.

		`deck_knobs` — held_fixed settings that are arguments to the deck
		build (cap_m, spin_years, max_cell_m) rather than fields of a column —
		are kept for _refine_columns, which passes them to the deck tool.
		"""
		client = self._mcp(config)
		if client is None:
			raise RuntimeError(
				f"the {self.MCP_NAME!r} MCP is required to build a controlled "
				f"sweep — what a factor level becomes is this server's "
				f"knowledge. Register a `{self.MCP_NAME}` client in mcp_config.json.")

		verdict = self._mcp_call(client, "check_conceptual_design",
								 {"design": design}) or {}
		for f in (verdict.get("facts") or []):
			print(f"   · {f.get('what')}: {f.get('detail')}")
		for u in (verdict.get("unusual") or []):
			print(f"   ⚠️  {u.get('factor')}: {u.get('why')}")
		if not verdict.get("buildable", False):
			why = "\n".join(f"     - {w.get('factor')}: {w.get('why')}"
							for w in (verdict.get("wont_build") or []))
			raise RuntimeError(
				f"the {self.MCP_NAME} server refused this sweep design:\n{why}")

		out = self._mcp_call(client, "build_conceptual_columns",
							 {"design": design}) or {}
		if not out.get("columns"):
			raise RuntimeError(
				"build_conceptual_columns returned no columns for a design the "
				"server had just called buildable — the check and the builder "
				"disagree, which they share code precisely to prevent: "
				+ str(out.get("error") or "")[:200])
		self._deck_knobs = dict(out.get("deck_knobs") or {})
		return out

	def _build_coupled_columns(self, config: Dict[str, Any]) -> Dict[str, Any]:
		"""The prior run's columns, each carrying its own driving flux.

		A COUPLED STUDY REUSES THE UPSTREAM RUN'S COLUMNS VERBATIM — same ids,
		coordinates, elevations, bands, pinned stations — so the Analyzer can
		line the two models up column by column. What this backend adds is its
		own boundary, read off the prior run's RECORD (experiment.json and
		extracted.json — framework files, no model server touched and no
		NetCDF reopened):

		    daily_flux_mm_day   the prior model's daily series for the
		                        coupling variable (default QDRAI, ELM's
		                        sub-surface drainage), mm/day as extracted
		    water_table_m       the prior model's own solved water table
		                        (metrics.water_table_depth_m) — NOT CONUS2's,
		                        so the column is anchored where the driving
		                        model says the water stands
		    flux_description    one sentence naming all of it; the server
		                        writes it into the deck's forcing_caveat

		Raises, with the reason named, on anything that would otherwise be
		guessed: no prior run, a column without the variable, units that are
		not mm/day, a column the prior never solved a water table for.
		"""
		brief = (config or {}).get("brief") or {}
		# MERGED, THE BRIEF'S VALUES WINNING. Reception resolved the prior run
		# and counted its columns INTO the brief's block; the planner's copy
		# in the strategy restates the design and has neither. Taking the
		# strategy's block alone — the first version did — read a coupling
		# with no prior_run_dir and refused a run reception had fully resolved.
		coupling = {**((config.get("strategy") or {}).get("coupling") or {}),
					**(brief.get("coupling") or {})}
		src = (coupling.get("prior_run_dir") or config.get("coupling_source"))
		if not src or not Path(src).is_dir():
			raise RuntimeError(
				"a coupled study needs the prior run's directory — reception "
				"resolves it into brief.coupling.prior_run_dir; got "
				f"{src!r}")
		src = Path(src).resolve()

		cols_f = src / "columns.json"
		exp_f = src / "experiment.json"
		ext_f = src / "03_results" / "extracted.json"
		for f in (cols_f, exp_f, ext_f):
			if not f.is_file():
				raise RuntimeError(
					f"the prior run {src.name} has no {f.name} — it must have "
					f"finished its extract before it can drive another model")
		prior = json.loads(cols_f.read_text())
		prior = prior.get("columns", prior) if isinstance(prior, dict) else prior
		wtd = {r.get("case_name") or r.get("id"):
			   ((r.get("metrics") or {}).get("water_table_depth_m"))
			   for r in json.loads(exp_f.read_text()).get("columns") or []}
		data = (json.loads(ext_f.read_text()) or {}).get("data") or {}

		# THE VARIABLE, READ AGAINST WHAT THE EXTRACT HOLDS. The brief's
		# coupling_variable is settled in conversation and often arrives as a
		# sentence ("ELM sub-surface drainage QDRAI -> top recharge"), so the
		# name is taken as the first ALL-CAPS token that the prior's extract
		# actually carries — never invented, and refused by name when nothing
		# matches.
		import re as _re
		have = sorted(((next(iter(data.values()), {}) or {})
					   .get("variables") or {}).keys())
		tokens = _re.findall(r"[A-Z][A-Z0-9_]{2,}",
							 str(coupling.get("coupling_variable") or "QDRAI"))
		var = next((t for t in tokens if t in have), None)
		if var is None:
			raise RuntimeError(
				f"the coupling names no variable the prior run extracted — "
				f"asked for {tokens or ['(nothing)']}, the extract holds {have}")

		# THE FIELDS THAT TRAVEL are the framework's sampler keys. The prior
		# model's own extras (ELM's soil_*, forcing_cell) stay behind: this
		# run's column metadata is composed for THIS model, and the merge
		# raises on a key nobody named — correctly, since ELM's donor soil is
		# not a fact about a PFLOTRAN column.
		KEEP = ("id", "lat", "lon", "elevation_m", "band", "band_range_m",
				"pinned", "station_id", "station_variable")
		columns: List[Dict[str, Any]] = []
		for c in prior:
			cid = c.get("id")
			block = (data.get(cid) or {})
			v = ((block.get("variables") or {}).get(var)) or {}
			vals = v.get("values") or []
			if not vals:
				raise RuntimeError(
					f"{cid}: the prior run's extract carries no {var} series "
					f"— it has {sorted((block.get('variables') or {}).keys())}")
			units = str(v.get("units") or "")
			if units != "mm/day":
				raise RuntimeError(
					f"{cid}: {var} is in {units!r}, and the deck tool takes "
					f"mm/day — convert at the extract, not here")
			wt = wtd.get(cid)
			if not isinstance(wt, (int, float)):
				raise RuntimeError(
					f"{cid}: the prior run reports no water_table_depth_m in "
					f"its metrics — the coupled column has nothing to anchor to")
			dates = block.get("dates") or []
			col: Dict[str, Any] = {k: c.get(k) for k in KEEP if c.get(k) is not None}
			col["water_table_m"] = round(float(wt), 3)
			col["daily_flux_mm_day"] = [float(x) for x in vals]
			col["coupled_from"] = src.name
			col["coupling_variable"] = var
			col["flux_description"] = (
				f"{var} from {src.name} ({len(vals)} daily values, mm/day) "
				f"applied as the top boundary: the driving model already "
				f"stored snow, removed evapotranspiration and generated "
				f"runoff, so this flux is what left ITS soil column downward. "
				f"The water table ({col['water_table_m']} m) is the driving "
				f"model's own solved mean, not CONUS2's.")
			if dates:
				col["forcing_start"] = int(str(dates[0])[:4])
				col["forcing_end"] = int(str(dates[-1])[:4])
			columns.append(col)

		print(f"   {len(columns)} column(s) from {src.name}, each with its "
			  f"own {var} series and solved water table")
		return {
			"approach": "coupled",
			"coupled_from": str(src),
			"coupling_variable": var,
			"n_columns": len(columns),
			"bands": [],
			"columns": columns,
		}

	def _refine_columns(self, columns, config: Dict[str, Any]) -> Dict[str, Any]:
		"""ONE MCP call: join, decks, run plan.

		Runs inside the base's materialize, before columns.json is written —
		the slot ELM uses for its warm start. There is nothing to snap here, so
		the columns go back exactly as they came; what this returns is the
		server's answer, and _to_run_plan hands it on.

		THE SERVER READS THE SITE ITSELF. `site_dir` is this run's directory,
		where reception wrote the CONUS2 subsurface, the water table raster and
		the daily rain. The server looks each column up by coordinate and
		reports the ones it could not complete rather than dropping them.

		A CONTROLLED SWEEP HAS NO SITE. Its columns already carry a profile,
		a water table and their water (from build_conceptual_columns), so no
		site_dir is passed and nothing is looked up; the design's deck knobs
		(cap_m, spin_years, max_cell_m) go to the deck tool instead. Everything
		else in this call — the decks, the run plan, the columns replaced in
		place — is identical.
		"""
		client = self._mcp(config)
		if client is None:
			raise RuntimeError(
				f"no {self.MCP_NAME!r} client in config['mcp_clients'] — the "
				f"decks are written by the model server, so materialize "
				f"cannot finish without one")
		sweep = _is_factor_sweep(config.get("strategy") or {}, config.get("brief"))
		args = {
			"columns":  columns,
			"out_dir":  str(self.input_dir / "decks"),
		}
		if not sweep:
			args["site_dir"] = str(self.run_dir)
		for k in self.BUILD_KNOBS:
			if (config or {}).get(k) is not None:
				args[k] = float(config[k])
		# THE DESIGN'S KNOBS WIN over the framework's config: held_fixed is
		# what the user settled, and the run must be the study described.
		for k, v in (getattr(self, "_deck_knobs", None) or {}).items():
			args[k] = float(v)
		print(f"   {len(columns)} column(s) -> decks in {args['out_dir']}"
			  + ("  (controlled sweep: no site, the columns carry their own "
				 "soil and water)" if sweep else ""))
		res = self._mcp_call(client, "create_decks_from_columns", args,
							 budget=1800.0) or {}
		decks = res.get("decks") or []
		if not decks or not res.get("run_plan"):
			raise RuntimeError(f"the server built no decks: "
							   f"{str(res.get('error') or res)[:200]}")
		built = [d for d in decks if d.get("status") == "built"]
		sat = [d for d in decks
			   if d.get("unsaturated_m") is not None and d["unsaturated_m"] < 0.5]
		print(f"   ✓ {len(built)}/{len(decks)} deck(s) built, "
			  f"{sum(d.get('n_cells') or 0 for d in built)} cells total")
		if sat:
			print(f"   ⚠️  {len(sat)} column(s) have no unsaturated zone: "
				  f"{', '.join(d['id'] for d in sat)}")
		for d in decks:
			if d.get("status") != "built":
				print(f"   ⚠️  {d.get('id')}: {str(d.get('reason'))[:120]}")
		borrowed = [r for r in (res["run_plan"].get("CONDITIONS_COUPLERS") or [])
					if r.get("rain_borrowed_km") is not None]
		if borrowed:
			print(f"   ↪ {len(borrowed)} off-grid column(s) took the nearest grid "
				  f"point's rain (grid spacing "
				  f"{res['run_plan'].get('grid_spacing_km')} km): "
				  + ", ".join(f"{r.get('id')} {r['rain_borrowed_km']} km"
							  for r in borrowed))
		inc = (res["run_plan"].get("incomplete") or [])
		if inc:
			print(f"   ⚠️  {len(inc)} column(s) incomplete — "
				  + "; ".join(f"{i.get('id')} lacks "
							  f"{', '.join(i.get('missing') or [])}"
							  for i in inc))
		# THE COLUMNS ARE REPLACED IN PLACE WITH THE COLUMNS AS BUILT — the
		# same list, each entry now carrying what the server learned about it:
		# the joined scalars (water table, forcing years, borrowed rain) and
		# the deck's own facts (unsaturated depth, transient or steady, its
		# warning). This is what ELM does with its snapped columns, and it is
		# what makes columns.json the ONE seam the package reads: the base
		# carries these onto the rows through COLUMN_METADATA_EXTRA below and
		# asserts on any key nobody named. Until 2026-08-18 they travelled by
		# a second, unchecked list in _extract, and `warning` — "this column
		# is essentially saturated" — vanished on the way to experiment.json.
		columns[:] = [dict(r) for r in res["run_plan"]["CONDITIONS_COUPLERS"]]
		self._mcp_inputs = res
		return {"mcp_inputs": res}

	# WHAT create_decks_from_columns ADDS TO A COLUMN, and what the base's
	# package carries from columns.json onto the row (composed with the
	# framework's own COLUMN_METADATA; a key neither names raises). Every kept
	# key is an INPUT to the column and FIELD_SEMANTICS says what it means;
	# they are carried so an output can be read against what produced it.
	COLUMN_METADATA_EXTRA = KeySet(
		"PFLOTRAN_COLUMN_METADATA",
		keep = ("water_table_m", "unsaturated_m", "wt_in_domain",
				"transient", "n_forcing_steps", "recharge_mm_yr",
				"rain_borrowed_km", "rain_borrowed_from",
				# A CONTROLLED SWEEP'S OWN FACTS (build_conceptual_columns):
				# which texture class the column is, what sits under it and
				# from what depth, and where the soil came from. Absent on a
				# site column, whose subsurface is CONUS2's and is described
				# by the ledger, not per row.
				"soil", "substrate", "soil_depth_m", "soil_source",
				# A COUPLED RUN'S OWN FACTS (build_coupled_columns): which
				# run drove this one, with which variable, and the sentence
				# the deck's forcing_caveat carries. Absent elsewhere.
				"coupled_from", "coupling_variable", "flux_description",
				# THE SERVER'S OWN SENTENCE ABOUT THIS COLUMN — "only 0.01 m
				# of unsaturated column", "built steady, no daily rain". Step 0
				# of the Analyzer turns a row's `warning` into a caveat naming
				# the columns that carry it; that is the whole route by which
				# a per-column fact reaches a claim.
				"warning",
				# which of the three inputs the join could not find, when any.
				"incomplete"),
		# Absent on a column that is on the grid, saturated-or-not, complete:
		# a normal run produces none of these on most rows.
		optional = ("recharge_mm_yr", "rain_borrowed_km", "rain_borrowed_from",
					"warning", "incomplete",
					"soil", "substrate", "soil_depth_m", "soil_source",
					"coupled_from", "coupling_variable", "flux_description"),
		drop = {
			"deck_status": "the run stage's `status` is the word the framework "
						   "counts on; a deck that did not build has no output "
						   "and is `failed` there",
			"reason": "why a deck did not build — printed at materialize and "
					  "kept in run_plan.json; the row's status carries the fact",
			"case_dir": "the extractor's row already carries it, from the run",
			"input_file": "the deck's path; case_dir locates it",
			"n_cells": "the extractor reports it from the output itself",
			"depth_m": "the extractor reports it as domain_depth_m, read off "
					   "the output",
			"domain_why": "the rule is stated once in the server's constraints "
						  "report; wt_in_domain on the row says whether it held",
			"forcing_caveat": "the transient wording is the ledger's first "
							  "assumption; the steady case is said in `warning`",
			"cell_dz_m": "deck numerics; the .in file is the record",
			"max_cell_m": "deck numerics; PFLOTRAN_CONFIG in run_plan.json",
			"n_material_zones": "deck numerics; the .in file is the record",
			"source": "the same sentence on every column — the ledger's second "
					  "assumption says it once",
			# A SWEEP ROW KEEPS ITS ARRAYS (the design is their only source);
			# the design figure reads them off columns.json. They do not
			# belong on an experiment row: the deck encodes both, and the row
			# already says which class (`soil`) and which rain (`weather`).
			"subsurface_profile": "the material layers a sweep column was "
								  "built from; the deck holds them and `soil` "
								  "/ `substrate` name them",
			"precipitation_mm_day": "the written year of rain a sweep column "
									"was driven with; `weather` is the spec "
									"that made it, n_forcing_steps its length",
			"daily_flux_mm_day": "the driving model's series a coupled column "
								 "was driven with; the deck encodes it, "
								 "flux_description says what it is, and the "
								 "prior run's extract is its source of truth",
		},
		source = "columns.json -> columns[*], as create_decks_from_columns wrote them",
		where  = "mcp/pflotran-mcp/pflotran_exp_manager.py :: COLUMN_METADATA_EXTRA",
	)

	def _draw_design(self, res: Dict[str, Any], config: Dict[str, Any]) -> None:
		"""sampling_design.png — the columns PFLOTRAN will actually integrate.

		Six panels from columns.json AS BUILT and reception.json: the columns
		over the terrain, the year's rain at each and through the year, the
		CONUS2 water table each was given, each column's depth on the CONUS2
		layer ladder, and how much unsaturated column each starts with. Drawn
		after columns.json is persisted, by code beside this manager
		(sampling_design.py), the way ELM's is drawn by code beside its server.
		A design figure: no model output is on it. Non-fatal in the base.
		"""
		sampling_design = self._sibling_module("sampling_design")
		png = sampling_design.render_run(
			self.run_dir,
			reception=self.run_dir / "reception.json",
			out=self.run_dir / "sampling_design.png")
		print(f"✓ sampling design → {Path(png).name}")

	def _to_run_plan(self, plan, columns, config, refine) -> Dict[str, Any]:
		"""The run plan the server already built, handed back — as ELM does."""
		out = (refine or {}).get("mcp_inputs") or getattr(self, "_mcp_inputs", None)
		if not out or "run_plan" not in out:
			raise RuntimeError(
				"no run plan from the pflotran server — _refine_columns must "
				"run first")
		return out["run_plan"]

	# ─────────────────────────────────────────────────────────
	# STEP 1 — CASE INPUTS
	# ─────────────────────────────────────────────────────────
	def _build_case_inputs(self, plan: Dict[str, Any],
						   config: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Read what materialize already built. Computes nothing.

		The decks were written by the one call in _refine_columns and the run
		plan carries one record per column — the column, its joined scalars
		and its deck. This stage exists because the base's pipeline has a slot
		for it; on a resume it is the same read off run_plan.json.
		"""
		rows = (plan or {}).get("CONDITIONS_COUPLERS") or []
		if not rows:
			raise RuntimeError(
				"the run plan carries no CONDITIONS_COUPLERS — materialize "
				"must run first; it is where the server builds the decks")
		built = sum(1 for r in rows if r.get("deck_status") == "built")
		print(f"   {len(rows)} column(s), {built} with a built deck")
		return [dict(r) for r in rows]

	# ─────────────────────────────────────────────────────────
	# STEP 3 — RUN
	# ─────────────────────────────────────────────────────────
	def _run(self, experiments: List[Dict[str, Any]],
			 config: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""Run every deck, inline. No scheduler, no job id, no polling.

		NEEDS_SCHEDULER IS FALSE and this is where that shows. ELM's _run
		submits to SLURM and hands back a job id, because a case takes long
		enough that a tool must return before it finishes. A PFLOTRAN column
		measured 5 s at 36 cells and 86 s at 786, so the whole 17-column
		ensemble finishes inside one call and there is nothing to wait for.

		ONE DECK PER CALL, AND THAT IS NOT AN OVERSIGHT. `run_pflotran_simulation`
		accepts `input_file` as a list, and in its default mode it runs only the
		FIRST one — measured 2026-08-17: three decks in, one .out produced, one
		exit code returned, `validation_status: "success"`, and no
		`results_by_input`. Sixteen columns would have been reported as run
		without a process ever starting. A loop costs ~4 s of session setup per
		column against a 5-86 s run and is attributable by construction.

		A FAILED COLUMN DOES NOT STOP THE ENSEMBLE. The others are real results
		and the failure is recorded against the column it belongs to, because
		"this column did not converge" and "the study did not run" are
		different findings.
		"""
		client = self._mcp(config)
		if client is None:
			raise RuntimeError(f"no {self.MCP_NAME!r} client to run the decks with")

		runnable = [e for e in experiments if e.get("input_file")]
		if not runnable:
			raise RuntimeError("no deck has an input_file — _build_case_inputs "
							   "produced nothing runnable")
		print(f"   {len(runnable)} deck(s), inline — no scheduler")

		outcomes: Dict[str, Dict[str, Any]] = {}
		for i, e in enumerate(runnable, 1):
			res = self._mcp_call(client, "run_pflotran_simulation",
								 {"input_file": e["input_file"],
								  "num_cores": 1, "timeout": 3000},
								 budget=3600.0) or {}
			codes = res.get("exit_codes") or []
			outcomes[e["input_file"]] = {
				"exit_code": codes[0] if codes else None,
				"stdout": (res.get("stdout_paths") or [None])[0],
				"output_files": res.get("output_files") or [],
				"seconds": res.get("execution_time"),
				"error": res.get("error") or res.get("stderr"),
			}
			mark = "✓" if outcomes[e["input_file"]]["exit_code"] == 0 else "✗"
			print(f"   {mark} [{i}/{len(runnable)}] {e.get('id')} "
				  f"({e.get('n_cells')} cells)")

		# WRITTEN ONTO THE EXPERIMENTS IN PLACE, as ELM's collect does. The
		# base builds RUN_SUMMARY.json from the EXPERIMENTS list, not from what
		# this returns — so a fresh dict per case left every runtime at 0 and
		# every status unrecorded there, while the values sat in a list nobody
		# persisted. `runtime_seconds` is the base's word for it.
		out, ok = [], 0
		for e in experiments:
			r = e
			one = outcomes.get(e.get("input_file") or "") or {}
			code = one.get("exit_code")
			r["exit_code"] = code
			r["ran"] = (code == 0)
			# THE WORD THE FRAMEWORK COUNTS ON. _outcome_map reads `status`
			# and _package's FAILED set reads it too; "completed" is the one
			# spelling both agree on.
			r["status"] = "completed" if code == 0 else "failed"
			r["stdout"] = one.get("stdout")
			r["output_files"] = one.get("output_files")
			r["runtime_seconds"] = one.get("seconds")
			if e.get("input_file") and code != 0:
				r["run_error"] = str(one.get("error")
									 or "the deck produced no exit code")[:300]
			ok += 1 if r["ran"] else 0
			out.append(r)
		print(f"   ✓ {ok}/{len(runnable)} column(s) ran clean")
		for r in out:
			if r.get("input_file") and not r.get("ran"):
				print(f"   ⚠️  {r.get('id')}: {str(r.get('run_error'))[:120]}")
		return out

	# ─────────────────────────────────────────────────────────
	# STEP 4 — EXTRACT
	# ─────────────────────────────────────────────────────────
	EXTRACTED = "extracted.json"

	def _extract(self, experiments: List[Dict[str, Any]],
				 plan: Dict[str, Any] = None,
				 config: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Read the finished columns into depth-resolved series on disk.

		THIN, BY THE SAME RULE ELM FOLLOWS. `extract_column_series` returns the
		numbers PFLOTRAN wrote — saturation and liquid pressure against depth
		at each output time — and nothing derived from them. No wetting-front
		depth, no transit time, no drainage flux, no ratios. Those answer the
		question the study asked, and answering it is the Analyzer's job; a
		stage that both reads a file format and decides what it means is the
		one place a wrong interpretation becomes unreviewable, because the raw
		numbers stop being written down.

		That is a real restraint here rather than a nominal one: this server
		carries compute_particle_trajectories, residence-time distributions and
		breakthrough curves, any of which would have answered "how deep and how
		fast" directly. They are left for the Analyzer to call against the
		series this writes.

		THE SERIES GO TO DISK, not into the return. Seventeen columns at up to
		786 cells and five output times is tens of thousands of numbers.
		03_results/extracted.json is the artifact; this returns the rows, the
		units and the path, which is the extract stage's contract.
		"""
		client = self._mcp(config or {})
		if client is None:
			raise RuntimeError(f"no {self.MCP_NAME!r} client — reading the "
							   f"model's own output format is the server's job")

		# WHAT RAN IS ESTABLISHED HERE, FROM DISK — not read off a status field.
		# execute_plan hands this stage the CASE INPUTS, not the run results,
		# and that is deliberate: a run record is written before the model
		# starts, so a column can be marked pending and have finished, or
		# marked done and have produced nothing. The output files are the
		# ground truth. So every case directory is offered to the extractor and
		# a column with no .tec comes back ok=false with its reason, rather
		# than being filtered out here on a field this stage cannot verify.
		cases = [{"id": e.get("id"), "case_dir": e.get("case_dir")}
				 for e in experiments if e.get("case_dir")]
		if not cases:
			raise RuntimeError("no column has a case directory — "
							   "_build_case_inputs produced nothing to read")

		out_file = str(self.results_dir / self.EXTRACTED)
		res = self._mcp_call(client, "extract_column_series",
							 {"cases": cases, "out_file": out_file},
							 budget=1800.0) or {}
		rows = res.get("rows") or []
		read = [r for r in rows if r.get("ok")]
		print(f"   ✓ {len(read)}/{len(rows)} column(s) read — "
			  f"{res.get('n_values', 0):,} values -> {out_file}")
		for r in rows:
			if not r.get("ok"):
				print(f"   ⚠️  {r.get('id')}: {str(r.get('reason'))[:120]}")

		# THE SERIES COME BACK OFF DISK AND ONTO THE ROWS. The tool writes them
		# to out_file and returns summaries, because a 786-cell column at five
		# output times is ~8,000 numbers and a tool result is not the place for
		# seventeen of those. But the Analyzer reads experiment.json and only
		# experiment.json — its step 0 says so, and _package copies the rows
		# verbatim — so a row without its series arrives at the Analyzer as a
		# column that ran and produced nothing. That is exactly what happened:
		# 17 of 17 columns packaged, every frame empty, and the run declined
		# with "no daily series, no soil layers and no depth profiles".
		#
		# So the file stays the raw record and the rows carry a copy, which is
		# what ELM's extractor already does by returning its daily values
		# inline. Attached under `profiles` because that is the frame the
		# Analyzer builds from a depth x time grid — one value per (output
		# time, depth), which is neither a daily series nor a soil layer.
		#
		# READ, NOT REBUILT. The block is passed through as the server wrote
		# it; this stage does not know what saturation means and does not need
		# to. It also makes a resumed run identical to a fresh one, since
		# _rehydrate_extract reads these same rows back out of experiment.json.
		attached = 0
		try:
			series = ((json.loads(Path(out_file).read_text()) or {})
					  .get("columns") or {})
		except Exception as e:                                  # noqa: BLE001
			series = {}
			print(f"   ⚠️  could not read back {out_file} ({e}) — the rows "
				  f"carry no series, so the Analyzer will find no frame")
		# THE COLUMN'S OWN SETUP IS NOT ATTACHED HERE ANY MORE (2026-08-18).
		# It used to be copied from the case inputs by a hand-written tuple,
		# and a key not on the tuple vanished in silence. The build call now
		# writes it onto columns.json, and the base's package carries it onto
		# the row through COLUMN_METADATA_EXTRA — the one asserted seam.
		for r in rows:
			blk = series.get(r.get("id"))
			if blk:
				r["profiles"] = blk
				attached += 1
		if series and attached < len(read):
			print(f"   ⚠️  {len(read) - attached} column(s) read but not "
				  f"attached — their id is not a key of the series file")
		if attached:
			print(f"   ✓ {attached} column(s) carry their depth x time series "
				  f"into experiment.json")

		return {
			"rows": rows,
			"units": res.get("units") or {},
			"extra_summary": {
				"extracted_path": res.get("path"),
				"n_with_series": res.get("n_with_series"),
				"n_values": res.get("n_values"),
				# NAMED, NOT COUNTED. A column absent from the extraction is
				# absent from every figure downstream, and "17 planned, 15
				# read" is a fact about the study rather than a detail.
				"no_output": [r.get("id") for r in rows if not r.get("ok")],
				"what_is_here": ("raw saturation and liquid pressure against "
								 "depth, per column, per output time. Nothing "
								 "derived — front depth, transit time and "
								 "fluxes are the Analyzer's to compute"),
			},
			# NO SPIN-UP WAS DROPPED, and saying so is different from saying
			# nothing. A transient column runs spin_years steady at the series
			# mean first, and those snapshots are IN the file: the reader needs
			# to know they are there rather than assume they were removed.
			"spinup_dropped": 0,
		}
