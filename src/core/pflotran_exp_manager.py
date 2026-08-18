#!/usr/bin/env python3
"""
PFLOTRAN Experiment Manager — the join, and nothing else
src/core/pflotran_exp_manager.py

    in   the sampler's columns (id, lat, lon, elevation, band)
    out  a run plan where each column carries what a deck needs

DELIBERATELY THIN, AND THAT IS THE DESIGN. The base class already samples,
checks the strategy against reception, clips to the watershed and persists
columns.json; none of that is model-specific and none of it is repeated here.
What PFLOTRAN adds is one step the base cannot do: saying WHICH server owns the
pinning rules, and joining the per-column subsurface, water table and forcing
onto the columns the sampler placed.

NO SNAP. `_refine_columns` is inherited from the base and returns {} — ELM's
warm start moves every column 200-700 m onto its donor gridcell, and this model
has no donor. The columns stay exactly where the plan put them, which is why
the subsurface and rain lookups below hit the same coordinates the sampler
chose.

WHERE THE NUMBERS COME FROM, and why none of them is fetched here:

    subsurface   ParFlow CONUS2, five parameter fields over the basin, written
                 once by reception. Read at any point by
                 core.conus2_subsurface.sample — no network, no PIN.
    water table  the CONUS2 steady-state field, same pattern, read by
                 core.static_wtd.sample.
    forcing      Daymet daily precipitation at the grid points, in the
                 reception package.

Reception is the only component that reaches outside the framework, so this
reads what reception wrote and fetches nothing.

UNITS ARE NOT CONVERTED HERE. Each column carries CONUS2's own vocabulary —
conductivity in m/h, van Genuchten alpha in 1/m, `n` rather than `m`. Turning
those into a deck is the PFLOTRAN server's job (`create_decks_from_columns`),
and it is the only place that knows what a deck wants.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.exp_manager_base import ExperimentManagerBase


class PFLOTRANExpManager(ExperimentManagerBase):
	"""Sampling from the base; the per-column join is the whole of what is here."""

	MODEL = "pflotran"

	# WHICH SERVER OWNS THE PINNING RULES. `_pinning_rules` prefers the model
	# the brief names and falls back to this, so a manager driven directly by a
	# tool or a test still asks the right server.
	MCP_NAME = "pflotran"
	CAPABILITIES_TOOL = "describe_pflotran_capabilities"

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
			"units": "m", "from": ["ParFlow CONUS2 ss_water_table_depth"],
			"note": ("the INITIAL condition, not a result — the column is "
					 "initialised hydrostatic about it. Comparing it to the "
					 "final saturation profile compares an input with an "
					 "output"),
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
			"units": "count", "from": ["Daymet daily precipitation"],
			"note": ("how many boundary values the column was driven with — "
					 "365 for a year of days. Null or 0 means a constant "
					 "boundary"),
		},
		"recharge_mm_yr": {
			"units": "mm/y", "from": ["Daymet daily precipitation"],
			"note": ("the CONSTANT top-boundary flux, for a column driven by "
					 "one. NULL ON A TRANSIENT COLUMN, where the boundary is "
					 "the 365-step series and no single rate describes it — "
					 "an absence with a reason, not a gap. Either way it is "
					 "an input: no flux is written out, so the rate that "
					 "actually reached the water table is not in this run"),
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
			"note": ("the top boundary was driven with Daymet PRECIPITATION "
					 "applied as recharge: no snow storage, no "
					 "evapotranspiration removed, no runoff generated, because "
					 "this model has none of them. In a snow-dominated basin "
					 "the snowpack therefore infiltrates in winter rather than "
					 "at melt, so seasonal TIMING is wrong by construction. "
					 "See the assumptions ledger"),
		},
	}

	def _to_run_plan(self, plan: Dict[str, Any], columns, config: Dict[str, Any],
					 refine: Dict[str, Any]) -> Dict[str, Any]:
		"""Columns -> the executable payload, by joining what reception wrote.

		THE JOIN IS A LOOKUP, NOT A FETCH. Reception wrote the subsurface and
		the water table as FIELDS over the basin and the rain as a series per
		grid point, all keyed by coordinate. Every column sits exactly where the
		sampler placed it, so each one is a dictionary lookup away from
		everything it needs — no interpolation, no nearest-neighbour, and no
		request.

		A COLUMN THAT CANNOT BE COMPLETED IS REPORTED, NOT DROPPED. A missing
		profile or water table means this location is outside the fetched box
		or over a no-data cell, and that is a fact about the study worth
		carrying — a silently shorter ensemble is a design nobody chose.
		"""
		from core import conus2_subsurface as cs
		from core import static_wtd

		run_dir = self.run_dir
		reception = config.get("reception") or {}
		lats = [c["lat"] for c in columns]
		lons = [c["lon"] for c in columns]

		profiles = cs.sample(run_dir, lats, lons)
		tables = static_wtd.sample(run_dir, lats, lons)
		precip = reception.get("precipitation") or {}
		rain = precip.get("series") or {}
		# WHICH YEARS THE COLUMN WAS ACTUALLY DRIVEN WITH, read off the same
		# calendar the series is on rather than copied from the plan. The
		# Analyzer asks "was the run driven over the period that was asked
		# for?" and answers it from the columns; a column that never records
		# its years answers `null`, and null reads as a period that was
		# planned and not honoured. Seventeen PFLOTRAN columns said exactly
		# that about a run driven with the 1988 Daymet year they were joined
		# to. Taken from the plan instead, the check would compare the plan
		# with itself and could never fail.
		cal_years = [y for y in ((precip.get("calendar") or {}).get("year") or [])
					 if isinstance(y, (int, float))]
		yr_first = int(min(cal_years)) if cal_years else None
		yr_last  = int(max(cal_years)) if cal_years else None

		out: List[Dict[str, Any]] = []
		incomplete: List[Dict[str, Any]] = []
		for col, prof, wt in zip(columns, profiles, tables):
			key = f"{round(col['lat'], 5)},{round(col['lon'], 5)}"
			row = dict(col)
			row["subsurface_profile"] = prof
			row["water_table_m"] = wt
			row["precipitation_mm_day"] = rain.get(key)
			# Only a column that GOT a series was driven by it. One that fell
			# outside the fetched box has no forcing and must not claim years.
			if rain.get(key) is not None and yr_first is not None:
				row["forcing_start"], row["forcing_end"] = yr_first, yr_last
			missing = [n for n, v in (("subsurface_profile", prof),
									  ("water_table_m", wt),
									  ("precipitation_mm_day", rain.get(key)))
					   if v is None]
			if missing:
				row["incomplete"] = missing
				incomplete.append({"id": col.get("id"), "missing": missing})
			out.append(row)

		# THE FORCING ASSUMPTION TRAVELS WITH THE PLAN. Daymet reports
		# PRECIPITATION and the deck's top boundary is a RECHARGE flux; between
		# them sit snow storage, evapotranspiration and runoff, none of which
		# this model has. In a snow-dominated basin the winter snowpack
		# therefore enters the ground in winter rather than at melt, so seasonal
		# timing is wrong by construction. Recorded here so the run carries it
		# whether or not anyone reads the server's copy.
		ledger = [
			{"assumption": "precipitation used as recharge",
			 "why": ("Daymet gives precipitation; the top boundary takes a "
					 "recharge flux. This model has no snowpack, no "
					 "evapotranspiration and no runoff generation, so nothing "
					 "stands between them"),
			 "cost": ("seasonal timing is wrong where snow matters — the "
					  "snowpack infiltrates in winter instead of at melt"),
			 "source": "design decision, 2026-08-17"},
			{"assumption": "subsurface from ParFlow CONUS2",
			 "why": ("a soil survey stops at about 1.5 m; these columns run to "
					 "tens or hundreds of metres. CONUS2 parameterises the "
					 "whole 392 m — SSURGO-derived above 2 m, GLHYMPS "
					 "hydrogeologic units below"),
			 "cost": ("properties are per-geologic-unit constants on a 1 km "
					  "grid, not site measurements; below 2 m the van Genuchten "
					  "curve shape is uniform basin-wide, so transit is "
					  "controlled by porosity and permeability alone"),
			 "source": "https://essd.copernicus.org/articles/13/3263/2021/"},
			{"assumption": "residual saturation as published",
			 "why": ("the water table these columns start from is an OUTPUT of "
					 "this same parameterisation; substituting a "
					 "texture-derived residual would make the initial "
					 "condition disagree with the material it initialises"),
			 "cost": ("CONUS2's sres is 1e-5 to 1e-4 against the 0.04-0.11 a "
					  "texture-derived profile gives, so these columns drain "
					  "far more completely than a real soil would"),
			 "source": "user decision, 2026-08-17"},
		]

		return {
			"CONDITIONS_COUPLERS": out,
			"PFLOTRAN_CONFIG": {
				# 0.5 m: measured. Refining a 12.85 m unsaturated column from
				# 2.0 m to 0.1 m moved the water-table front 1.30 m and the
				# stored water 1.5%; from 0.5 m to 0.1 m moved them 0.05 m and
				# 0.1%. Finer than 0.5 m buys nothing and costs cells.
				"max_cell_m": float(config.get("max_cell_m", 0.5)),
				"cap_m": float(config.get("cap_m", cs.TOTAL_DEPTH_M)),
				"min_depth_m": float(config.get("min_depth_m", cs.MIN_DEPTH_M)),
				"spin_years": float(config.get("spin_years", 10.0)),
			},
			"assumptions_ledger": ledger,
			"n_columns": len(out),
			"n_incomplete": len(incomplete),
			"incomplete": incomplete,
		}

	# ─────────────────────────────────────────────────────────
	# STEP 1 — BUILD THE DECKS
	# ─────────────────────────────────────────────────────────
	def _build_case_inputs(self, plan: Dict[str, Any],
						   config: Dict[str, Any]) -> List[Dict[str, Any]]:
		"""The run plan's columns become 17 runnable decks, written by the server.

		A THIN PASS-THROUGH, AND DELIBERATELY SO. Everything this stage needs is
		already on the columns — _to_run_plan joined the subsurface, the water
		table and the forcing — so there is nothing to compute here. The deck
		itself is written by `create_decks_from_columns`, because knowing what a
		deck wants is the server's job: unit conversion, the layer-boundary
		domain depth, subdividing material zones into cells.

		NEEDS_CASE_BUILD IS FALSE, so the base skips _build_cases entirely. For
		ELM that stage compiles a CIME case and takes eight minutes; a PFLOTRAN
		deck is a text file, so writing it IS the build and there is no second
		step to wait on.
		"""
		client = self._mcp(config)
		if client is None:
			raise RuntimeError(
				f"no {self.MCP_NAME!r} client in config['mcp_clients'] — the "
				f"decks are written by the model server, so this stage cannot "
				f"run without one")

		cols = (plan or {}).get("CONDITIONS_COUPLERS") or []
		if not cols:
			raise RuntimeError(
				"the run plan carries no CONDITIONS_COUPLERS — _to_run_plan "
				"must run first, and it is what joins the subsurface, the "
				"water table and the forcing onto the sampled columns")
		cfg = dict((plan or {}).get("PFLOTRAN_CONFIG") or {})
		out_dir = str(self.run_dir / "01_inputs" / "decks")

		print(f"   {len(cols)} column(s) -> decks in {out_dir}")
		# BUDGETED, because this writes one deck per column and a 786-cell
		# column is not a quick call. The ceiling belongs to the CALL, not to
		# the server, whose registered timeout is sized for its ordinary tool.
		res = self._mcp_call(client, "create_decks_from_columns",
							 {"columns": cols, "out_dir": out_dir, **cfg},
							 budget=1800.0) or {}
		decks = res.get("decks") or []
		if not decks:
			raise RuntimeError(f"the server built no decks: "
							   f"{str(res.get('error') or res)[:200]}")

		built = [d for d in decks if d.get("status") == "built"]
		sat = [d for d in decks if d.get("warning")]
		print(f"   ✓ {len(built)}/{len(decks)} deck(s) built, "
			  f"{sum(d.get('n_cells') or 0 for d in built)} cells total")
		if sat:
			# SAID OUT LOUD, because a saturated column runs perfectly and
			# answers nothing about vertical transit. It is a fact about the
			# site, not a failure, so it is reported rather than dropped.
			print(f"   ⚠️  {len(sat)} column(s) have no unsaturated zone: "
				  f"{', '.join(d['id'] for d in sat)}")
		if len(built) < len(decks):
			for d in decks:
				if d.get("status") != "built":
					print(f"   ⚠️  {d.get('id')}: {str(d.get('reason'))[:120]}")

		# THE COLUMN AND ITS DECK TRAVEL TOGETHER. The base persists whatever
		# this returns as case_inputs.json and resumes from it, so each record
		# has to carry enough to re-run without re-deriving anything.
		# `status` IS THE FRAMEWORK'S WORD FOR THE RUN'S OUTCOME, not the
		# deck's. _outcome_map counts a case successful when status ==
		# "completed", so leaving the builder's "built" here reported a
		# finished 17-column ensemble as 0/17 — after the compute, in the final
		# line. The build outcome keeps its own key.
		by_id = {c.get("id"): c for c in cols}
		out = []
		for d in decks:
			src = {k: v for k, v in (by_id.get(d.get("id")) or {}).items()
				   if k not in ("subsurface_profile", "precipitation_mm_day")}
			rec = {**src, **d}
			rec["deck_status"] = rec.pop("status", None)
			out.append(rec)
		return out

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

		out, ok = [], 0
		for e in experiments:
			r = dict(e)
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
			r["run_seconds"] = one.get("seconds")
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
		# The column's OWN SETUP travels the same way, from the case inputs
		# this stage was handed. It is the manager's to carry: the server read
		# .tec files and has no idea what drove them or what they started from.
		#
		# water_table_m is the one that had to be here. FIELD_SEMANTICS
		# declares it, so the Analyzer went looking and reported "the ParFlow
		# CONUS2 reference water_table_m is finite for 0 of the 17 columns" —
		# about a run where every column was initialised hydrostatic about a
		# finite one. A name in FIELD_SEMANTICS with no number on any row is a
		# capability the package claims and does not have, and the claim it
		# produced was a false statement about the study.
		#
		# EVERY ONE OF THESE IS AN INPUT, and FIELD_SEMANTICS says so for each.
		# They are carried so an output can be read against what produced it —
		# not so they can be reported as results.
		SETUP = ("forcing_start", "forcing_end", "water_table_m",
				 "unsaturated_m", "wt_in_domain",
				 "transient", "n_forcing_steps", "recharge_mm_yr")
		by_id = {e.get("id"): e for e in (experiments or []) if e.get("id")}
		for r in rows:
			blk = series.get(r.get("id"))
			if blk:
				r["profiles"] = blk
				attached += 1
			src = by_id.get(r.get("id")) or {}
			for k in SETUP:
				if src.get(k) is not None and r.get(k) is None:
					r[k] = src[k]
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
