#!/usr/bin/env python3
"""
Experiment Manager — the model-agnostic half
src/core/exp_manager_base.py

The third box. It takes reception.json and strategy.json and turns them into
compute, then hands a results package to the Analyzer.

What lives HERE is everything true of any land-surface or subsurface model:

    check         compare the strategy against reception before spending
                  anything — the last moment both are in hand
    materialize   strategy → concrete columns (coordinates, elevation bands,
                  soil, water table) and the sampling-design figure
    package       run summary + the LLM input payload

What lives in a BACKEND subclass is everything that knows a model's file
formats and job model: how to warm-start it, how to build a case, how to
submit it, and how to read its history files.

The split matters because ELM and PFLOTRAN differ in every one of those and in
none of the first set. A PFLOTRAN study of the same watershed samples the same
columns from the same DEM against the same strategy; only the case directory
and the input deck differ. Sharing the sampler is also what makes coupling
meaningful — a PFLOTRAN run over columns ELM already produced is only a
comparison if both used one materialisation.

Two ordering constraints are structural, not stylistic, and the base enforces
them so no backend can get them wrong:

  1. _refine_columns() runs BEFORE columns.json is written. A warm start snaps
     each column to its donor gridcell, so a file written earlier describes a
     plan that does not match the run — every column of the 2026-07-28 run was
     200-700 m from where columns.json claimed, and both the spatial maps and
     the nearest-station matching read that file.
  2. The sampling-design figure is drawn AFTER the refinement, for the same
     reason: it was showing a design that had already been superseded.

Subclasses implement:
    _refine_columns(columns, config) -> Dict   backend edits to the columns
                                               (warm start, donor soil); the
                                               returned dict is passed on to
                                               _to_run_plan. Default: no-op.
    _to_run_plan(plan, columns, config, refine) -> Dict   columns → the
                                               backend's executable plan.
    _build / _prepare / _run / _extract        the compute stages.
"""
import json
import sys
from pathlib  import Path
from datetime import datetime
from typing   import Any, Dict, List

sys.path.insert(0, "src")


class ExperimentManagerBase:
	"""Sampling, the strategy gate, and packaging — no model in sight."""

	MODEL = "experiment"          # backends override; names the run dir

	def __init__(self,
				 base_output_dir: str = "./workflow_outputs",
				 run_dir: str = None):
		"""
		run_dir  an EXISTING directory to run in. The coordinator creates it and
		         writes reception.json and strategy.json into it as each stage
		         finishes, so every file is written by whatever produced it and
		         this stage only ever READS its inputs. It also means a crash
		         here leaves those two intact.

		         Omitted (standalone / CLI use), one is minted as before.
		"""
		self.base_output_dir = Path(base_output_dir)
		if run_dir:
			self.run_dir = Path(run_dir)
		else:
			timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
			self.run_dir = self.base_output_dir / f"{self.MODEL}_run_{timestamp}"
		self.run_dir.mkdir(parents=True, exist_ok=True)

		# Four numbered subdirs — the same names for every backend, so a
		# coupled study has one directory layout rather than two.
		self.input_dir       = self.run_dir / "01_inputs"
		self.setup_plots_dir = self.run_dir / "02_setup_plots"
		self.results_dir     = self.run_dir / "03_results"
		self.analysis_dir    = self.run_dir / "04_analysis"

		for d in [self.input_dir, self.setup_plots_dir,
				  self.results_dir, self.analysis_dir]:
			d.mkdir(exist_ok=True)

		self.strategy_report = None

	# ─────────────────────────────────────────────────────────
	# STEP 0a — THE GATE
	# ─────────────────────────────────────────────────────────
	def check(self,
			  plan:   Dict[str, Any],
			  config: Dict[str, Any]) -> Dict[str, Any]:
		"""Compare strategy against reception. Returns a possibly-updated config.

		Here rather than at the planner hand-off because this is where cost
		begins: everything upstream is one LLM call, everything downstream is a
		case build and a queue slot per column. STOP conditions raise;
		corrections are recorded in the run and the run continues.
		"""
		from core.strategy_check import check as _check, render as _render
		reception = config.get("reception") or {"brief": config.get("brief") or {}}
		report, fixed = _check(reception, config.get("strategy") or plan)
		print("🔎 STEP 0: Checking strategy against reception")
		print(_render(report))
		if not report["ok"]:
			raise ValueError("strategy does not agree with reception: "
							 + "; ".join(report["stop"]))
		if report["corrections"]:
			config = {**config, "strategy": fixed}
		self.strategy_report = report
		return config

	# ─────────────────────────────────────────────────────────
	# BACKEND HOOKS
	# ─────────────────────────────────────────────────────────
	def _refine_columns(self, columns, config: Dict[str, Any]) -> Dict[str, Any]:
		"""Backend edits to the sampled columns, before they are persisted.

		ELM warm-starts here, which snaps coordinates and adopts the donor
		cell's soil. A backend with nothing to add returns {}.
		"""
		return {}

	def _to_run_plan(self, plan, columns, config, refine) -> Dict[str, Any]:
		raise NotImplementedError(
			f"{type(self).__name__} must turn columns into an executable plan")

	# ─────────────────────────────────────────────────────────
	# STEP 0b — MATERIALIZE SAMPLING (strategy → concrete columns)
	# ─────────────────────────────────────────────────────────
	def _materialize(self,
					 plan:   Dict[str, Any],
					 config: Dict[str, Any]) -> Dict[str, Any]:
		"""Turn a planner strategy into concrete columns, then into a run plan.

		No-op when the plan already carries an executable payload (the legacy
		single-site shape), so old plans keep working unchanged.

		Needs, via config:
			brief        — reception brief (domain bbox + optional HUC)
			mcp_clients  — dict of live MCP clients (terrain/fan_wtd/geology)
		Optional: yr_start, yr_end, soil_config, substrate, n_columns, n_bands.
		"""
		if self._already_executable(plan):
			return plan

		exp     = _load_tool("expand_sampling")
		brief   = config.get("brief") or {}
		clients = config.get("mcp_clients") or {}

		bbox = exp._bbox_from_brief(brief)
		if not bbox:
			raise ValueError(
				"Cannot materialize sampling: no domain bbox in the reception "
				"brief, and the plan has no executable payload. Pass "
				"config['brief'] with domain.bbox, or supply an explicit plan.")
		if not clients.get("terrain"):
			raise ValueError(
				"Cannot materialize sampling: the 'terrain' MCP client is "
				"required. Pass config['mcp_clients'].")

		n_total = config.get("n_columns") or exp._n_from_plan(plan)
		if not n_total:
			raise ValueError(
				"Cannot materialize sampling: no column count in the plan "
				"and none in config.")
		bands = config.get("n_bands") or len(
			(brief.get("heterogeneity") or {}).get("elevation_bands") or []) or 4

		config = self.check(plan, config)

		print("\n🗺️  STEP 0: Materializing Sampling")
		print("-" * 40)
		print(f"   bbox={bbox} N={n_total} bands={bands}")

		# Clip the rectangular DEM sample to the real basin when we know the HUC.
		# Without a boundary the sample is the raw bbox, so columns can land
		# OUTSIDE the basin — the run still completes, but the ensemble no
		# longer represents the watershed that was asked about. That is a
		# scientific difference, so say so loudly rather than degrade silently.
		boundary = None
		huc = (brief.get("domain") or {}).get("huc")
		if huc:
			try:
				b = clients["terrain"].call_tool_json(
					"get_watershed_boundary",
					{"huc": huc, "huc_level": len(str(huc))}) or {}
				boundary = b.get("rings")
			except Exception as e:
				print(f"   ⚠️  boundary lookup FAILED for HUC {huc} ({e})")
		if not boundary:
			why = ("no HUC in the reception brief"
				   if not huc else f"boundary lookup returned nothing for HUC {huc}")
			name = (brief.get("domain") or {}).get("name") or "the watershed"
			print(f"   ⚠️  WARNING: sampling the RAW BBOX, not the watershed — {why}.")
			print(f"   ⚠️  Columns may fall outside the basin; the ensemble is "
				  f"a bounding-box sample, not '{name}'.")
			print(f"   ⚠️  Fix: ensure reception resolves a HUC, or pass "
				  f"config['boundary'] explicitly.")
		boundary = config.get("boundary", boundary)

		res = exp.expand(clients, bbox, n_total, bands, boundary=boundary)
		if res.get("error"):
			raise RuntimeError(f"Sampling expansion failed: {res['error']}")
		columns = res.get("columns", [])
		if not columns:
			raise RuntimeError("Sampling expansion produced no columns.")

		# Keep the WBD polygon in the result: plot_columns() draws it as the
		# basin outline in panel (a), and it makes columns.json self-contained
		# (grid + boundary) so a later re-plot needs no MCP fetch.
		if boundary:
			res["boundary"] = boundary

		# Provenance: was this a true watershed sample or a bbox fallback?
		res["sampling_domain"] = {
			"clipped_to_watershed": bool(boundary),
			"huc": huc,
			"name": (brief.get("domain") or {}).get("name"),
			"bbox": bbox,
			"caveat": None if boundary else (
				"Columns were sampled from the bounding box, NOT clipped to the "
				"watershed boundary (no HUC resolved). Some columns may lie "
				"outside the basin; treat the ensemble as a bbox sample."),
		}

		yr_start = int(config.get("yr_start", 1995))
		yr_end   = int(config.get("yr_end",   yr_start))

		# Backend refinement BEFORE persisting — see the module docstring. The
		# ELM warm start snaps every column to its donor gridcell.
		refine = self._refine_columns(columns, config) or {}
		res["columns"] = columns

		(self.input_dir / "columns.json").write_text(json.dumps(res, indent=2))
		(self.run_dir  / "columns.json").write_text(json.dumps(res, indent=2))

		# The sampling-design figure: domain map + watershed outline,
		# hypsometry with band edges, soil configs, Fan WTD vs elevation,
		# per-band allocation, NLDAS precip gradient. Same function
		# tools/expand_sampling.py --plot uses; forcing_year populates the
		# precip-vs-elevation panel that is otherwise blank.
		try:
			png = exp.plot_columns(
				res, str(self.run_dir / "sampling_design.png"),
				forcing_year = yr_start)
			print(f"✓ sampling design → {Path(png).name}")
		except Exception as e:
			print(f"   ⚠️  sampling_design.png failed ({e}) — non-fatal")

		executable = self._to_run_plan(plan, columns, config, refine)
		merged     = {**plan, **executable}

		# The honesty payload reads this back at extraction; without it an
		# integrated run shipped an empty ledger.
		(self.run_dir / "assumptions.json").write_text(
			json.dumps(executable.get("assumptions_ledger", []), indent=2))
		(self.run_dir / "run_plan.json").write_text(json.dumps(merged, indent=2))
		# Make the run dir self-describing, with the SAME filenames the
		# standalone tools expect. validate_run.build_validation() needs
		# reception_brief.json (domain bbox) and interpret_run.interpret()
		# needs plan.json (goals + feasibility verdict). Writing them here
		# means an integrated run is consumable by every existing tool.
		# reception.json itself is the coordinator's to write.
		if not (self.run_dir / "reception_brief.json").exists():
			(self.run_dir / "reception_brief.json").write_text(
				json.dumps(brief, indent=2))
		(self.run_dir / "plan.json").write_text(json.dumps(plan, indent=2))
		print(f"✓ {len(columns)} column(s) materialized ({yr_start}-{yr_end})")
		return merged

	def _already_executable(self, plan: Dict[str, Any]) -> bool:
		"""Does this plan already carry a backend payload? Backends override."""
		return False


	# ─────────────────────────────────────────────────────────
	# PACKAGE — the one file the Analyzer reads
	# ─────────────────────────────────────────────────────────
	#
	# FIELD_SEMANTICS is the point of this stage. `precip_mm_yr` once held
	# RAIN alone, so every fraction computed against it was inflated and a
	# run was reported as draining more water than fell on it — a semantic
	# bug, not a structural one, and no schema check could have caught it.
	# Naming the source variables next to the number is what makes the next
	# one visible.
	FIELD_SEMANTICS = {
		"precip_mm_yr":            {"units": "mm/yr", "from": ["RAIN", "SNOW"],
									"note": "TOTAL precipitation: rain + snow"},
		"rainfall_mm_yr":          {"units": "mm/yr", "from": ["RAIN"]},
		"snowfall_mm_yr":          {"units": "mm/yr", "from": ["SNOW"]},
		"annual_runoff_mm_yr":     {"units": "mm/yr", "from": ["QOVER", "QDRAI"],
									"note": "surface + subsurface drainage, "
											"the streamflow-comparable total"},
		"annual_recharge_mm_yr":   {"units": "mm/yr", "from": ["QCHARGE"]},
		"runoff_fraction":         {"units": "1", "from": ["annual_runoff_mm_yr",
														   "precip_mm_yr"]},
		"recharge_fraction":       {"units": "1", "from": ["annual_recharge_mm_yr",
														   "precip_mm_yr"]},
		"water_table_depth_m":     {"units": "m", "from": ["ZWT"],
									"note": "positive downward from the surface"},
		"peak_swe_mm":             {"units": "mm", "from": ["H2OSNO"]},
		"tws_seasonal_range_mm":   {"units": "mm", "from": ["TWS"]},
	}

	def _package(self,
				 plan:    Dict[str, Any],
				 results: Any,
				 config:  Dict[str, Any] = None) -> Dict[str, Any]:
		"""experiment.json — everything the Analyzer needs, and nothing else.

		Deliberately simple. The expensive stages are the build and the run;
		this one is cheap to revise, so it carries what is known now rather
		than waiting for a schema the Analyzer has not yet asked for.

		The per-column rows come straight from the backend's extraction, so
		this stage stays model-agnostic: it labels and assembles, it does not
		compute.
		"""
		config = config or {}
		rows   = getattr(results, "results", None) or []
		if isinstance(rows, dict):
			rows = list(rows.values())

		brief  = config.get("brief") or {}
		period = ((brief.get("run_settings") or {}).get("resolved_period")) or {}
		# Enumerate FAILURE, not success. Backends spell success differently
		# ("ok", "success", "completed") and a new spelling must not silently
		# report a good run as zero columns; a column with metrics and no
		# failure status produced numbers, which is the thing being counted.
		FAILED = {"failed", "error", "timeout", "cancelled"}
		ok = [r for r in rows
			  if str((r or {}).get("status", "")).lower() not in FAILED
			  and (r or {}).get("metrics")]

		pkg = {
			"model":   self.MODEL,
			"run_dir": str(self.run_dir),
			"created": datetime.now().isoformat(timespec="seconds"),
			"domain": {
				"name": (brief.get("domain") or {}).get("name"),
				"huc":  (brief.get("domain") or {}).get("huc"),
				"bbox": (brief.get("domain") or {}).get("bbox"),
			},
			"period": {"yr_start": period.get("yr_start") or config.get("yr_start"),
					   "yr_end":   period.get("yr_end")   or config.get("yr_end"),
					   "source":   period.get("source")},
			"columns_total":     len(rows),
			"columns_succeeded": len(ok),
			# What the numbers MEAN — see FIELD_SEMANTICS above.
			"field_semantics":   self.FIELD_SEMANTICS,
			"variable_units":    getattr(results, "units", None)
								 or (getattr(results, "summary", {}) or {}).get("units"),
			"strategy_check":    self.strategy_report,
			"goals":             plan.get("goals"),
			"columns":           rows,
			"artifacts": {
				"sampling_design": "sampling_design.png",
				"columns":         "columns.json",
				"run_plan":        "run_plan.json",
				"extracted":       "04_analysis/hydro_summary.json",
			},
		}
		(self.run_dir / "experiment.json").write_text(
			json.dumps(pkg, indent=2, default=str))
		print(f"✓ experiment.json — {len(ok)}/{len(rows)} column(s) "
			  f"→ the Analyzer's only input")
		return pkg


	# ─────────────────────────────────────────────────────────
	# PACKAGE — run summary + the payload the Analyzer reads
	# ─────────────────────────────────────────────────────────

	# ─────────────────────────────────────────────────────────
	# STEP 5 — PACKAGE LLM INPUT (top level)
	# ─────────────────────────────────────────────────────────
	def _save_llm_input(self,
						plan:     Dict[str, Any],
						analyzer: Any) -> None:
		"""Save LLM_ANALYSIS_INPUT.json at the top level of run_dir.

		An ALIAS. experiment.json is the manager's real product; this exists
		because the standalone report tools open it by this name.

		Non-fatal, and that matters: it runs AFTER the compute has succeeded,
		so anything raising here throws away a finished ensemble over a
		convenience file. get_llm_analysis_input() is also specific to ELM's
		results object — a backend without it must still finish its run.
		"""
		try:
			llm_input = analyzer.get_llm_analysis_input()
		except Exception as e:
			print(f"   ⚠️  LLM_ANALYSIS_INPUT.json skipped ({e}) — "
				  f"experiment.json is unaffected")
			return
		llm_input['experiment_plan'] = plan
		llm_input['run_directory']   = str(self.run_dir)

		# get_llm_analysis_input() omits extra_summary, so the honesty payload
		# and the observation verdicts never reached the report agent. Attach
		# them here so the written report can be held to the same standard as
		# 04_analysis/interpretation.md.
		if getattr(analyzer, 'extra_summary', None):
			llm_input['limitations'] = analyzer.extra_summary.get('limitations')
			llm_input['assumptions_ledger'] = \
				analyzer.extra_summary.get('assumptions_ledger')
		vp = self.analysis_dir / "validation.json"
		if vp.exists():
			try:
				val = json.loads(vp.read_text())
				llm_input['validation'] = {
					"targets": val.get("targets"),
					"observation_inventory": val.get("observation_inventory"),
					"hydrograph_metrics": {
						k: v for k, v in (val.get("hydrograph") or {}).items()
						if k not in ("days", "obs", "mod")},
				}
			except Exception as e:
				print(f"   ⚠️  could not attach validation to LLM input: {e}")
		ip = self.analysis_dir / "interpretation.md"
		if ip.exists():
			llm_input['interpretation_md'] = ip.read_text()

		try:
			llm_file = self.run_dir / "LLM_ANALYSIS_INPUT.json"
			with open(llm_file, 'w') as f:
				json.dump(llm_input, f, indent=2, default=str)
			print(f"✓ LLM_ANALYSIS_INPUT.json saved")
		except Exception as e:
			print(f"   ⚠️  LLM_ANALYSIS_INPUT.json not written ({e})")

	# ─────────────────────────────────────────────────────────
	# RUN SUMMARY
	# ─────────────────────────────────────────────────────────

	# ─────────────────────────────────────────────────────────
	# RUN SUMMARY
	# ─────────────────────────────────────────────────────────
	def _create_run_summary(self,
							plan:        Dict[str, Any],
							experiments: List[Dict],
							results:     Dict[str, bool],
							start_time:  datetime,
							end_time:    datetime
							) -> Dict[str, Any]:
		n_total   = len(experiments)
		n_success = sum(results.values())

		# .get(), not [], deliberately. This is the LAST step: the ensemble is
		# already computed and experiment.json already written, so a missing
		# key here would throw away a finished run over a summary field.
		exp_details = [
			{
				'name':              e.get('scenario_name') or e.get('case_name'),
				'case_name':         e.get('case_name'),
				'status':            'completed'
									 if results.get(e.get('case_name'))
									 else 'failed',
				'forcing_period':    e.get('forcing_period'),
				'forcing_start':     e.get('forcing_start'),
				'forcing_end':       e.get('forcing_end'),
				'model_type':        self.MODEL,
				'runtime_seconds':   0,
				'timesteps':         0,
				'newton_iterations': 0,
			}
			for e in experiments
		]

		return {
			'run_directory':         str(self.run_dir),
			'start_time':            start_time.isoformat(),
			'end_time':              end_time.isoformat(),
			'total_runtime_seconds': (
				end_time - start_time
			).total_seconds(),
			'experiments_total':     n_total,
			'experiments_success':   n_success,
			'experiments_failed':    n_total - n_success,
			'experiments':           exp_details,
			'convergence_warnings':  [],
			'output_files': {
				'inputs':             str(self.input_dir),
				'setup_plots':        str(self.setup_plots_dir),
				'results':            str(self.results_dir),
				'analysis':           str(self.analysis_dir),
				'experiment_summary': str(
					self.input_dir / "experiment_summary.json"),
				'execution_report':   str(
					self.results_dir / "execution_report.txt"),
				'results_csv':        str(
					self.results_dir / "results_summary.csv"),
				'hydro_summary':      str(
					self.analysis_dir / "hydro_summary.json"),
				'llm_input':          str(
					self.run_dir / "LLM_ANALYSIS_INPUT.json"),
			},
			'model_type': 'elm',
		}


	def _save_run_summary(self,
						  run_summary: Dict[str, Any]) -> None:
		summary_file = self.run_dir / "RUN_SUMMARY.json"
		with open(summary_file, 'w') as f:
			json.dump(run_summary, f, indent=2, default=str)
		print(f"✓ RUN_SUMMARY.json saved")


	def _save_error(self,
					error:      Exception,
					start_time: datetime) -> None:
		import traceback
		error_log = {
			'run_directory': str(self.run_dir),
			'start_time':    start_time.isoformat(),
			'end_time':      datetime.now().isoformat(),
			'status':        'failed',
			'error':         str(error),
			'traceback':     traceback.format_exc(),
			'model_type':    'elm',
		}
		error_file = self.run_dir / "ERROR_LOG.json"
		with open(error_file, 'w') as f:
			json.dump(error_log, f, indent=2)


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("ELMExpManager — call via workflow.py or test directly:")
    print("  from core.elm_exp_manager import ELMExpManager")
    print("  mgr = ELMExpManager()")
    print("  run_summary = mgr.execute_plan(plan, {})")


def _load_tool(name: str):
	"""Import tools/<name>.py by path — tools/ is a script dir, not a package."""
	import importlib.util
	if name in sys.modules:
		return sys.modules[name]
	path = Path(__file__).resolve().parents[2] / "tools" / f"{name}.py"
	spec = importlib.util.spec_from_file_location(name, path)
	mod  = importlib.util.module_from_spec(spec)
	sys.modules[name] = mod
	spec.loader.exec_module(mod)
	return mod
