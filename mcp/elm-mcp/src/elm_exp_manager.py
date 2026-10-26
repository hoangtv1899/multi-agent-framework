#!/usr/bin/env python3
"""
ELM Experiment Manager
mcp/elm-mcp/src/elm_exp_manager.py

Single responsibility: orchestrate the ELM execution pipeline — the
Experiment Manager box of the framework (see ARCHITECTURE.md).

NOT AN MCP TOOL, despite living in the server's src/. The seven tools are
registered in mcp/elm-mcp/main.py; this is the framework's per-model manager
class, which moved here with the rest of the ELM code in phase 1b and is
imported back across the boundary by core/backends.py. It is deleted after
phase 4 (docs/ELM_MCP_PLAN.md §1e) — the input half is already a shim around
one MCP call, and what remains is the SLURM job-A/job-B orchestration that has
to live outside a tool, because a tool must return fast or return an id.

Output directory structure:

    elm_run_YYYYMMDD_HHMMSS/
        ├── columns.json  run_plan.json  plan.json  reception_brief.json
        ├── assumptions.json                        (the honesty ledger)
        ├── sampling_design.png                     (step 0, materialize)
        ├── warmstart/                              (step 0b, only if warm)
        │   ├── Warmstart_<col>.nc
        │   └── warmstart.json
        ├── 01_inputs/
        │   └── experiment_summary.json
        ├── 03_results/
        │   ├── execution_report.txt  results_summary.csv
        │   └── extracted.json          (the record of what was READ)
        ├── 04_analysis/
        │   ├── comparison.json  investigation.json  interpretation.json
        │   ├── analysis.json           (the Analyzer's boundary file)
        │   ├── partitioning.png  controls.png  soil_control.png
        │   └── wtd_columns.png
        │   ├── <col>/<col>.in + outputs
        ├── experiment.json             (this box's product)
        └── RUN_SUMMARY.json

ANALYSIS_REPORT.json and LLM_ANALYSIS_INPUT.json used to sit at the top level.
Both belonged to the one-shot report agent, deleted 2026-08-13; the Analyzer
writes 04_analysis/analysis.json instead. validation.json is listed by no
current writer either.

ELM cases live at: $PSCRATCH/E3SMv3/1D_ELM.*/
"""
import csv
import glob
import os
import json
import sys
from pathlib  import Path
from datetime import datetime
from typing   import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from framework_path import ensure_on_path, framework_dir
# require: this subclasses core.exp_manager_base, the base class ELM and
# PFLOTRAN share. It is deliberately not vendored, so this needs a checkout.
ensure_on_path(require=True)

from core.exp_manager_base import ExperimentManagerBase, Pending
from core.keyset import KeySet


# _ROOT and _load_tool ARE GONE (2026-08-19) with step 4d: their only user
# was the PFLOTRAN coupling CLIs, and a coupling is a run of the PFLOTRAN
# manager now — see exp_manager_base._is_coupling.


# ─────────────────────────────────────────────────────────────────────
# ELM EXPERIMENT MANAGER
# ─────────────────────────────────────────────────────────────────────
class ELMExpManager(ExperimentManagerBase):
	"""
	Executes ELM experiment plans. EVERY ELM COMPUTATION IS THE MCP'S.

	What is left here is the four things that are not ELM knowledge: WHEN to
	call the server (the stage order is the base's), WHERE the run directory
	is, WHAT the derived fields mean (FIELD_SEMANTICS), and arranging the
	framework's own follow-up job. The stages read:

	    materialize        one MCP call — warm start, donor soil, surfaces,
	                       case_inputs.json, the run plan  (_refine_columns)
	    build_case_inputs  read back what that call wrote
	    build_cases        submit JOB A (build + run), submit JOB B, stop
	    run                collect from disk — A already ran the columns
	    extract            read the history NetCDFs
	    couple             optional one-way handoff to PFLOTRAN

	THERE IS NO LOCAL PATH. It was deleted 2026-08-10: _refine_columns requires
	the `elm` client, so every branch that ran ELM by import was unreachable
	from the first stage onward and only looked like a fallback. With it went
	_run_batch, _submit_via_mcp, the ELMExperimentBuilder handle, and the last
	use of tools/submit_cases.sh from this side of the boundary.

	Figures, observation validation and interpretation are NOT here — they are
	the Analyzer (src/agents/analyzer.py), which execute_plan invokes after
	extraction.
	"""
	MODEL = "elm"

	def __init__(self,
				 base_output_dir: str = "./workflow_outputs",
				 run_dir: str = None):
		super().__init__(base_output_dir, run_dir)
		print(f"\n{'=' * 60}")
		print(f"ELM Experiment Manager")
		print(f"Run dir : {self.run_dir}")
		print(f"Cases   : $PSCRATCH/E3SMv3/")
		print(f"{'=' * 60}\n")

	# ─────────────────────────────────────────────────────────
	# BACKEND HOOKS — what the base cannot know about ELM
	# ─────────────────────────────────────────────────────────
	def _already_executable(self, plan: Dict[str, Any]) -> bool:
		"""CONDITIONS_COUPLERS is ELM's executable payload (the legacy shape),
		DOMAIN_CASE a whole-domain study's."""
		return bool(plan.get("CONDITIONS_COUPLERS") or plan.get("DOMAIN_CASE"))

	def _build_sweep_columns(self, design: Dict[str, Any],
							 config: Dict[str, Any]) -> Dict[str, Any]:
		"""A controlled sweep's columns, built by the server that runs them.

		The base knows a sweep is one column per level combination. Only this
		server knows that a soil_texture level of 27 becomes a synthetic
		surface dataset with those percentages in it, or that two coordinates
		in one NLDAS cell would share their weather exactly — so the design
		crosses the MCP boundary and comes back as columns.

		CHECKED FIRST, AND SEPARATELY. build_conceptual_columns refuses an
		unbuildable design on its own, but its refusal is an exception with a
		list in it. Calling check first means the run stops with every reason
		named at once, including the ones that would NOT have refused — a level
		outside the fitted range builds fine and still belongs in the log
		before an hour of compute rather than after.
		"""
		client = self._mcp(config)
		if client is None:
			raise RuntimeError(
				"the elm MCP is required to build a controlled sweep — what a "
				"factor level becomes is this server's knowledge. Register an "
				"`elm` client in mcp_config.json.")

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
				f"the elm server refused this sweep design:\n{why}")

		out = self._mcp_call(client, "build_conceptual_columns",
							 {"design": design}) or {}
		if not out.get("columns"):
			raise RuntimeError(
				"build_conceptual_columns returned no columns for a design the "
				"server had just called buildable — the check and the builder "
				"disagree, which they share code precisely to prevent.")
		return out

	def _build_coupled_columns(self, config: Dict[str, Any]) -> Dict[str, Any]:
		"""The prior PFLOTRAN run's columns, each carrying its next start.

		THE RETURN LEG OF TWO-WAY COUPLING. The forward leg drove PFLOTRAN
		with ELM's drainage; this one hands PFLOTRAN's SOLVED water table
		back as each column's initial state. The columns are the prior run's
		verbatim (which are the original ELM run's — same ids, coordinates,
		bands), so an iteration is the same ensemble walked to consistency:

		    initial_water_table_m   PFLOTRAN's pressure-crossing depth at its
		                            final output time, read by the ONE public
		                            function that owns that definition
		                            (compare/water_table.solved_water_table_m,
		                            loaded by path from beside the PFLOTRAN
		                            server — the meaning of that model's
		                            output stays on that model's side)
		    water_table_prior_m     what ELM said last time (the anchor the
		                            PFLOTRAN column was given)
		    water_table_delta_m     solved minus prior — the number an
		                            iteration watches; near zero means the
		                            two models agree and the loop is done

		The warm start then stamps each column's fresh finidat with the new
		water table (inputs.py -> set_water_table.py), reporting clamps —
		ELM's aquifer ends at 28.802 m, and a deeper PFLOTRAN water table is
		honestly clamped, not silently believed.
		"""
		brief = (config or {}).get("brief") or {}
		coupling = {**((config.get("strategy") or {}).get("coupling") or {}),
					**(brief.get("coupling") or {})}
		src = coupling.get("prior_run_dir") or config.get("coupling_source")
		if not src or not Path(src).is_dir():
			raise RuntimeError(
				"a coupled study needs the prior run's directory — reception "
				f"resolves it into brief.coupling.prior_run_dir; got {src!r}")
		src = Path(src).resolve()
		cols_f = src / "columns.json"
		ext_f = src / "03_results" / "extracted.json"
		for f in (cols_f, ext_f):
			if not f.is_file():
				raise RuntimeError(
					f"the prior run {src.name} has no {f.name} — it must have "
					f"finished its extract before it can drive another model")

		# the one function that owns "where PFLOTRAN's water table IS"
		import importlib.util
		# Resolved through the framework, not by counting parents, so this still
		# points at the sibling server when elm-mcp is checked out on its own.
		pkg = framework_dir() / "mcp" / "pflotran-mcp" / "compare"
		if not (pkg / "__init__.py").is_file():
			raise RuntimeError(
				f"two-way coupling needs the PFLOTRAN server's compare package "
				f"at {pkg}, which is not there. It ships with "
				f"multi-agent-framework; set IDEAS_FRAMEWORK_DIR to a checkout "
				f"that has mcp/pflotran-mcp/.")
		spec = importlib.util.spec_from_file_location(
			"compare_pflotran", pkg / "__init__.py",
			submodule_search_locations=[str(pkg)])
		mod = importlib.util.module_from_spec(spec)
		sys.modules.setdefault("compare_pflotran", mod)
		spec.loader.exec_module(mod)
		from compare_pflotran import water_table as _pwt

		prior = json.loads(cols_f.read_text())
		prior = prior.get("columns", prior) if isinstance(prior, dict) else prior
		data = (json.loads(ext_f.read_text()) or {}).get("columns") or {}

		KEEP = ("id", "lat", "lon", "elevation_m", "band", "band_range_m",
				"pinned", "station_id", "station_variable")
		columns: List[Dict[str, Any]] = []
		for c in prior:
			cid = c.get("id")
			solved = _pwt.solved_water_table_m(data.get(cid) or {})
			if solved is None:
				raise RuntimeError(
					f"{cid}: the prior run's final profile has no water table "
					f"in the domain (bottom cell unsaturated) — there is "
					f"nothing to hand back for this column")
			prior_wt = c.get("water_table_m")
			col: Dict[str, Any] = {k: c.get(k) for k in KEEP
								   if c.get(k) is not None}
			col["initial_water_table_m"] = float(solved)
			col["coupled_from"] = src.name
			col["coupling_variable"] = "water_table"
			if isinstance(prior_wt, (int, float)):
				col["water_table_prior_m"] = round(float(prior_wt), 3)
				col["water_table_delta_m"] = round(float(solved) - float(prior_wt), 3)
			# THE WHOLE PROFILE, NOT JUST THE CROSSING. PFLOTRAN also solved
			# where the water sits above the table; carried UNINTERPRETED —
			# ELM's server converts it with ELM's own porosity at the stamp
			# (set_water_table.apply_profile), so each model's meaning stays
			# on its own side and the column starts from one coherent state.
			blk = data.get(cid) or {}
			deps, sats = blk.get("depth_m"), blk.get("saturation")
			if (isinstance(deps, list) and len(deps) >= 2
					and isinstance(sats, list) and sats
					and isinstance(sats[-1], list)):
				col["initial_saturation_profile"] = {
					"depth_m": [round(float(x), 3) for x in deps],
					"saturation": [round(float(s), 4) for s in sats[-1]],
					"at_time_y": (blk.get("times_y") or [None])[-1],
				}
			columns.append(col)

		deltas = [c.get("water_table_delta_m") for c in columns
				  if c.get("water_table_delta_m") is not None]
		print(f"   {len(columns)} column(s) from {src.name}, each taking "
			  f"PFLOTRAN's solved water table as its next start"
			  + (f"; moved vs ELM's last answer: "
				 + ", ".join(f"{d:+.3f} m" for d in deltas) if deltas else ""))
		return {
			"approach": "coupled",
			"coupled_from": str(src),
			"coupling_variable": "water_table",
			"n_columns": len(columns),
			"bands": [],
			"columns": columns,
		}

	def _refine_columns(self, columns, config: Dict[str, Any]) -> Dict[str, Any]:
		"""ONE MCP call: warm start, donor soil, surfaces, case_inputs.json.

		Runs inside the base's materialize, BEFORE columns.json is written and
		before the design figure is drawn — the warm start snaps each column to
		its donor gridcell, so anything persisted earlier describes a plan the
		run will not follow.

		WHY THE WHOLE INPUT BUILD HAPPENS HERE, in a hook named for refinement.
		The MCP does all six input steps in one call and the manager used to
		split them across two stages, with the columns.json write wedged
		between. That split is why routing this through the MCP was impossible
		for a week: calling the tool from the later stage re-ran the warm start,
		and calling it from here could not work while the tool's only input was
		a file that does not exist yet at this moment. The tool now takes the
		columns as DATA, so the call lands here — the one moment that satisfies
		the ordering constraint — and the two stages after it have nothing left
		to compute.

		THE COLUMNS ARE REPLACED IN PLACE. The base persists the same list
		object it handed us, so returning a new one would silently persist the
		pre-snap coordinates.

		A whole-domain case has nothing to warm-start: its saved state is the
		recipe's finidat, and _build_domain_case has checked it already.
		"""
		if self._is_domain_rows(columns):
			return {"whole_domain": True}
		client = self._mcp(config)
		if client is None:
			raise RuntimeError(
				"the elm MCP is required to build ELM inputs — the local "
				"by-import path was deleted 2026-08-10 (docs/EXP_MANAGER_ELM.md "
				"§4). Register an `elm` client in mcp_config.json.")
		yr_start = int((config or {}).get("yr_start", 1995))
		ws = (config or {}).get("warm_start", True)
		out = self._mcp_call(client, "build_elm_inputs_from_location", {
			"run_dir":       str(self.run_dir),
			"columns":       columns,
			"yr_start":      yr_start,
			"yr_end":        int((config or {}).get("yr_end", yr_start)),
			"soil_config":   str((config or {}).get("soil_config", "native")),
			"substrate":     str((config or {}).get("substrate", "extrapolate")),
			# FORWARDED EXPLICITLY. Only conus_restart used to cross, so
			# warm_start=False could not be expressed at all and a cold sweep
			# was built warm while its columns claimed otherwise.
			"warm_start":    ws is not False,
			"conus_restart": str((ws or {}).get("conus_restart", "")
								 if isinstance(ws, dict) else ""),
		}, budget=900)
		if not out.get("ok"):
			raise RuntimeError(
				f"build_elm_inputs_from_location failed: {out.get('error')}")
		columns[:] = out["columns"]          # snapped — see the docstring
		self._mcp_inputs = out
		print(f"✓ {out['n_cases']} case input(s) built via the elm MCP")
		return {"mcp_inputs": out}

	def _draw_design(self, res: Dict[str, Any], config: Dict[str, Any]) -> None:
		"""sampling_design.png — what ELM will actually integrate.

		Six panels: the columns over a hillshade, NLDAS forcing against donor
		elevation, the initial soil water read out of each column's finidat, and
		the sand / clay / organic profiles the donor gridcell handed over. Every
		value is POST-warm-start, which is why this can only run after
		_refine_columns — the snap moves every column and swaps its soil.

		IT LIVES HERE BECAUSE IT IS ELM KNOWLEDGE. Reading a finidat and knowing
		what a donor gridcell is are this server's business; docs/ELM_MCP_PLAN.md
		said so on 2026-08-07 and listed the figure among the input tool's
		returns, but `plots.sampling_design` was never written and the framework
		kept drawing its own from tools/expand_sampling.plot_columns — unstyled,
		and describing the columns as SAMPLED rather than as run. Moved
		2026-08-13; that plotter is deleted.
		"""
		if self._is_domain_rows(res.get("columns")):
			print("   (no design figure for a whole-domain case yet)")
			return
		sampling_design = self._sibling_module("sampling_design")
		png = sampling_design.render_run(
			self.run_dir,
			reception = self.run_dir / "reception.json",
			year      = int((config or {}).get("yr_start", 0) or 0),
			out       = self.run_dir / "sampling_design.png")
		print(f"✓ sampling design → {Path(png).name}")

	def _to_run_plan(self, plan, columns, config, refine) -> Dict[str, Any]:
		"""The plan the MCP already built, handed back.

		Rebuilding it here would be the same computation twice with two chances
		to disagree; the base still persists it as run_plan.json and resumes
		from it, so it is returned by the call rather than recomputed.
		"""
		if self._is_domain_rows(columns):
			return {"DOMAIN_CASE": {"recipe": f"01_inputs/{self.DOMAIN_RECIPE}",
									"chain": f"01_inputs/{self.DOMAIN_CHAIN}",
									"record": self.DOMAIN_RECORD},
					"assumptions_ledger": []}
		out = (refine or {}).get("mcp_inputs") or getattr(self, "_mcp_inputs", None)
		if not out or "run_plan" not in out:
			raise RuntimeError(
				"no run plan from the elm MCP — _refine_columns must run first")
		return out["run_plan"]

	# ─────────────────────────────────────────────────────────
	# WHOLE-DOMAIN MODE: one case, a chain of gated year jobs
	# ─────────────────────────────────────────────────────────
	# A whole-domain study (exp_manager_base._is_whole_domain) is ONE row: the
	# case. Its recipe and the chain it asks for go to 01_inputs/; the server
	# checks the input set, writes the scripts and submits the chain
	# (submit_domain_chain), and every job leaves its verdict as a check record
	# in domain_case/checks/. The rows this mode extracts are the gates' year
	# numbers over vegetated cells; slicing a run into regions for the Analyzer
	# is the next framework piece, and until then the Analyzer is skipped with
	# that reason rather than run on rows it was not built for.
	DOMAIN_RECIPE = "domain_recipe.json"
	DOMAIN_CHAIN  = "domain_chain.json"
	DOMAIN_RECORD = "domain_case"
	KIND_DOMAIN   = "whole_domain"
	# The year gate's numbers that become a row's metrics, and their units.
	DOMAIN_YEAR_UNITS = {
		"gpp_gC_m2_yr": "gC m-2 yr-1", "peak_tlai": "m2 m-2", "precip_mm": "mm yr-1",
		"et_mm": "mm yr-1", "runoff_mm": "mm yr-1", "tws_change_mm": "mm",
		"water_closure_frac_of_precip": "fraction of the year's precipitation",
		"swe_apr1_mm": "mm", "totecosysc_change_frac": "fraction of the stock",
	}

	@classmethod
	def _is_domain_rows(cls, rows) -> bool:
		rows = list(rows or [])
		return bool(rows) and all(isinstance(r, dict) and r.get("kind") == cls.KIND_DOMAIN
								  for r in rows)

	def _domain_client(self, config):
		client = self._mcp(config or {})
		if client is None:
			raise RuntimeError(
				"the elm MCP is required for a whole-domain case: the server "
				"checks the input set and submits the chain. Register an `elm` "
				"client in mcp_config.json.")
		return client

	def _build_domain_case(self, config: Dict[str, Any]) -> Dict[str, Any]:
		"""The one row of a whole-domain study, from strategy['domain_case'].

		domain_case names the case, either as a full `recipe` or as an
		existing case to copy (`from_case`, with the new `case` name and an
		optional `case_root`), and the chain: `years` (default the study's
		period), `smoke_days`, `smoke`, `with_case`, `partition`, `tag`,
		`walls`, and a `reference` run ({run_dir, stream, pattern,
		gate_years}). The input set is checked HERE, before any job: a set
		that fails stops the study with the check's reasons.
		"""
		cfg = config or {}
		if (cfg.get("strategy") or {}).get("domain_experiment"):
			return self._build_domain_experiment(cfg)
		dc = dict(((cfg.get("strategy") or {}).get("domain_case")) or {})
		client = self._domain_client(cfg)
		if dc.get("from_case") and not dc.get("case"):
			raise RuntimeError("domain_case copies a case (from_case) but names no new "
							   "case: the copy would be the case it copies")
		if dc.get("recipe"):
			recipe = dict(dc["recipe"])
		elif dc.get("from_case"):
			out = self._mcp_call(client, "read_domain_recipe", {
				"case_dir": str(dc["from_case"]), "case": str(dc.get("case") or ""),
				"case_root": str(dc.get("case_root") or "")})
			if not out.get("ok"):
				raise RuntimeError(f"read_domain_recipe: {out.get('error')}")
			if out.get("problems"):
				raise RuntimeError("the case's recipe cannot be used: "
								   + "; ".join(out["problems"]))
			recipe = out["recipe"]
		else:
			raise RuntimeError("domain_case names no case: give a recipe or from_case")
		if dc.get("notes"):
			recipe["notes"] = list(recipe.get("notes") or []) + [str(n) for n in dc["notes"]]
		DC = self._sibling_module("domain_case")
		r = DC.Recipe.from_json(json.dumps(recipe))
		bad = r.check()
		if bad:
			raise RuntimeError("the recipe cannot be used: " + "; ".join(bad))
		if dc.get("years"):
			years = [int(y) for y in dc["years"]]
		elif cfg.get("yr_start"):
			y0 = int(cfg["yr_start"])
			years = list(range(y0, int(cfg.get("yr_end") or y0) + 1))
		else:
			years = [r.first_year]
		chain = {"years": years, "smoke_days": int(dc.get("smoke_days", 5)),
				 "smoke": bool(dc.get("smoke", True)),
				 "with_case": bool(dc.get("with_case", True)),
				 "partition": str(dc.get("partition") or cfg.get("queue") or "short"),
				 "tag": dc.get("tag") or r.case, "reference": dc.get("reference"),
				 "walls": dc.get("walls") or {}}
		if chain["with_case"] and r.case_dir.exists():
			raise RuntimeError(f"the chain would create {r.case_dir}, which exists: name a "
							   f"new case, or re-arm later years with with_case and smoke "
							   f"false")
		inputs = r.inputs()
		print(f"   case {r.case} in {r.case_root}, years {years[0]}-{years[-1]}")
		chk = self._mcp_call(client, "check_domain_inputs", {
			"surface": str(inputs["surface"]), "domain": str(inputs["domain"]),
			"finidat": str(inputs["finidat"]), "run_dir": str(self.run_dir)}, budget=600)
		if not chk.get("ok"):
			raise RuntimeError(f"check_domain_inputs: {chk.get('error')}")
		if not chk.get("pass"):
			raise RuntimeError("the input set fails its check, so nothing was submitted: "
							   + "; ".join(chk.get("reasons") or []))
		print("   ✓ the input set passes its check (ELM's rules and the framework's)")
		(self.input_dir / self.DOMAIN_RECIPE).write_text(json.dumps(recipe, indent=1))
		(self.input_dir / self.DOMAIN_CHAIN).write_text(json.dumps(chain, indent=1))
		row = {"case_name": r.case, "kind": self.KIND_DOMAIN,
			   "case_dir": str(r.case_dir), "years": years,
			   "forcing_start": years[0], "forcing_end": years[-1],
			   "inputs": inputs, "inputs_check": chk.get("path")}
		(self.input_dir / self.CASE_INPUTS).write_text(json.dumps([row], indent=2))
		return {"columns": [row],
				"domain_case": {"recipe": f"01_inputs/{self.DOMAIN_RECIPE}",
								"chain": f"01_inputs/{self.DOMAIN_CHAIN}",
								"inputs_check": chk.get("path")}}

	def _submit_domain(self, experiments, config) -> "Pending":
		"""The chain through the server, then job B on every one of its jobs."""
		if self._is_experiment_rows(experiments):
			return self._submit_experiment(experiments, config)
		client = self._domain_client(config)
		email = self._notify_email(config)
		out = self._mcp_call(client, "submit_domain_chain",
							 {"run_dir": str(self.run_dir)}, budget=600)
		if not out.get("ok"):
			raise RuntimeError(f"submit_domain_chain: {out.get('error')}"
							   + (": " + "; ".join(out["reasons"]) if out.get("reasons")
								  else ""))
		jobs = [{"stage": j["stage"], "job_id": str(j["job_id"])} for j in out["jobs"]]
		for j in jobs:
			print(f"   {j['stage']:>8}  job {j['job_id']}")
		ids = [j["job_id"] for j in jobs]
		# JOB B WAITS ON EVERY JOB, the chain's report included: started on the
		# last year alone it could find the report still queued, read the chain
		# as active, and stop one line before the analysis.
		job_b = self._submit_job_b(":".join(ids), config, email)
		return Pending(job_b or ids[-1], scope="domain", n_cases=len(experiments or []),
					   chain=jobs, job_ids=ids, job_id_b=job_b or None, via="mcp",
					   notify=email or None)

	def _domain_records(self) -> Dict[str, Any]:
		DC = self._sibling_module("domain_case")
		return DC.chain_records(self.run_dir / self.DOMAIN_RECORD)

	def _attach_chain(self, experiments, recs) -> bool:
		"""The case succeeded only when its whole chain did: every planned
		year passed. Some years passed is 'incomplete', none 'failed'."""
		planned, passed = recs.get("years_planned") or [], recs.get("years_passed") or []
		ok = recs.get("first_unpassed") is None and bool(planned) and \
			sorted(planned) == sorted(passed)
		for e in experiments:
			e["chain"] = {"years_planned": planned, "years_passed": passed,
						  "first_unpassed": recs.get("first_unpassed")}
			e["status"] = "completed" if ok else ("incomplete" if passed else "failed")
		return ok

	def _poll_domain(self, record, experiments, config):
		# Job B finalizes the study itself once the chain has ended; a resume
		# that collected it meanwhile would write the same files at once.
		jb = record.get("job_id_b")
		if jb and (self._slurm_state(jb) or "") in self.ACTIVE_JOB_STATES:
			print(f"   job {jb} (the finalize) is still queued or running; it will "
				  f"collect the chain")
			return None
		if self._is_experiment_rows(experiments):
			return self._poll_experiment(experiments, config)
		st = self._mcp_call(self._domain_client(config), "check_domain_chain",
							{"run_dir": str(self.run_dir)})
		if not st.get("ok"):
			raise RuntimeError(f"check_domain_chain: {st.get('error')}")
		if st.get("active"):
			running = [f"{j['stage']}={j['state']}" for j in st.get("jobs") or []
					   if j.get("state") not in ("COMPLETED",)]
			print(f"   the chain is still going ({', '.join(running)}): nothing to "
				  f"collect yet")
			return None
		self._attach_chain(experiments, self._domain_records())
		print(f"   ✓ the chain ended: years passed {st.get('years_passed')}, "
			  f"first failure {st.get('failed_stage')}")
		return experiments

	def _adopt_domain(self, experiments) -> Dict[str, Any]:
		if self._is_experiment_rows(experiments):
			return self._adopt_experiment(experiments)
		recs = self._domain_records()
		ok = self._attach_chain(experiments, recs)
		self._save_case_inputs(experiments)
		self._mark("build_cases", n_experiments=len(experiments), n_ok=int(ok),
				   adopted_from=f"{self.DOMAIN_RECORD}/chain.json")
		self._mark("run", n_results=len(experiments), n_ok=int(ok),
				   adopted_from=f"{self.DOMAIN_RECORD}/checks")
		return {"build_cases": len(experiments), "run": len(experiments)}

	def _collect_domain(self, experiments) -> Dict[str, bool]:
		if self._is_experiment_rows(experiments):
			return self._collect_experiment(experiments)
		recs = self._domain_records()
		ok = self._attach_chain(experiments, recs)
		lines = [f"whole-domain case {e['case_name']}: years passed "
				 f"{recs.get('years_passed')}, first stage without a passing check "
				 f"{recs.get('first_unpassed')}" for e in experiments]
		lines += [f"  {s['stage']:>8} job {s['job_id']} check "
				  f"{'none' if s['check_pass'] is None else 'PASS' if s['check_pass'] else 'FAIL'}"
				  for s in recs.get("stages") or []]
		(self.results_dir / "execution_report.txt").write_text("\n".join(lines) + "\n")
		print("\n".join(lines))
		return {e["case_name"]: ok for e in experiments}

	def _extract_domain(self, experiments) -> Dict[str, Any]:
		"""One row per model year whose check record exists: the year gate's
		numbers over vegetated cells and, where a reference run was compared,
		its per-field bias and correlation and the elevation-band totals."""
		if self._is_experiment_rows(experiments):
			return self._extract_experiment(experiments)
		import re
		recs = self._domain_records()
		case = (experiments or [{}])[0].get("case_name")
		rows = []
		for st in recs.get("stages") or []:
			if not re.fullmatch(r"y\d{4}", st["stage"]) or not st.get("check"):
				continue
			y = st["stage"][1:]
			doc = json.loads(Path(st["check"]).read_text())
			yr = (doc.get("years") or {}).get(y) or {}
			row = {"case_name": case, "kind": self.KIND_DOMAIN, "year": int(y),
				   "region": "vegetated cells (PCT_NATVEG = 100)",
				   "status": "completed" if yr.get("pass") else "failed",
				   "job_id": st["job_id"], "check": st["check"],
				   "cells": yr.get("cells"), "days": yr.get("days"),
				   "metrics": {k: yr[k] for k in self.DOMAIN_YEAR_UNITS if k in yr},
				   "reasons": yr.get("reasons") or []}
			ref = yr.get("reference")
			if ref:
				row["reference"] = {
					"pass": ref.get("pass"), "gated": ref.get("gated"),
					"reference": ref.get("reference"),
					"fields": {f: {k: v.get(k) for k in ("bias_pct", "daily_r", "ok")}
							   for f, v in (ref.get("fields") or {}).items()
							   if "missing" not in v},
					"bands": ref.get("bands")}
			rows.append(row)
		have = {r["year"] for r in rows}
		for y in recs.get("years_planned") or []:
			if y not in have:
				rows.append({"case_name": case, "kind": self.KIND_DOMAIN, "year": int(y),
							 "region": "vegetated cells (PCT_NATVEG = 100)",
							 "status": "not run", "metrics": {},
							 "reasons": [f"the chain stopped before this year (first stage "
										 f"without a passing check: "
										 f"{recs.get('first_unpassed')})"]})
		rows.sort(key=lambda r: r["year"])
		(self.results_dir / "extracted.json").write_text(json.dumps(
			{"kind": self.KIND_DOMAIN, "rows": rows,
			 "source": f"{self.DOMAIN_RECORD}/checks/y<YEAR>_<job>.json, the year gates' "
					   f"own records"}, indent=1))
		print(f"   ✓ {len(rows)} model year(s) read from the chain's check records")
		return {"rows": rows, "units": dict(self.DOMAIN_YEAR_UNITS),
				"extra_summary": {}, "spinup_dropped": {}}

	def _package_domain(self, plan, results, config) -> Dict[str, Any]:
		"""experiment.json for a whole-domain case: the case, its input check,
		its chain and the year rows (under `columns`, where a resumed run reads
		them back)."""
		if self._is_experiment_rows(self._rehydrate_case_inputs() or []):
			return self._package_experiment(plan, results, config)
		cfg = config or {}
		rows = [dict(r) for r in self._extract_rows(results) if isinstance(r, dict)]
		recipe, recipe_path = {}, None
		for rp in (self.run_dir / self.DOMAIN_RECORD / self.DOMAIN_RECIPE,
				   self.input_dir / self.DOMAIN_RECIPE):
			if rp.is_file():
				recipe, recipe_path = json.loads(rp.read_text()), rp
				break
		recs = self._domain_records() if (self.run_dir / self.DOMAIN_RECORD /
										  "chain.json").is_file() else None
		built_from = None
		for st in (recs or {}).get("stages") or []:
			if st["stage"] == "case" and st.get("check"):
				built_from = (json.loads(Path(st["check"]).read_text()) or {}).get("source")
		chk = {}
		cp = self.run_dir / self.DOMAIN_RECORD / "inputs_check.json"
		if cp.is_file():
			d = json.loads(cp.read_text())
			chk = {"pass": d.get("pass"), "reasons": d.get("reasons"),
				   "checks": {k: (d.get("checks") or {}).get(k) for k in (
					   "domain_vs_surface_coord_max_diff_deg", "layout_matches",
					   "pft_weight_max_abs_diff", "restart_grid_max_offset_deg",
					   "expected_dims", "active_cells")}}
		brief = cfg.get("brief") or {}
		pkg = {
			"model": self.MODEL, "kind": self.KIND_DOMAIN, "archetype": self.KIND_DOMAIN,
			"run_dir": str(self.run_dir),
			"created": datetime.now().isoformat(timespec="seconds"),
			"domain": {k: (brief.get("domain") or {}).get(k) for k in ("name", "huc", "bbox")},
			"case": {k: recipe.get(k) for k in ("case", "case_root", "compset", "res",
												"start", "notes", "source", "copied_from")}
					| {"sourcemods": {k: v.get("sha256") for k, v in
									  (recipe.get("sourcemods") or {}).items()},
					   "user_files": {k: v.get("sha256") for k, v in
									  (recipe.get("user_files") or {}).items()},
					   "built_from": built_from},
			"inputs_check": chk,
			"chain": recs,
			"columns_total": len(rows),
			"columns_succeeded": sum(1 for r in rows if r.get("status") == "completed"),
			"variable_units": dict(self.DOMAIN_YEAR_UNITS),
			"strategy_check": self.strategy_report, "goals": (plan or {}).get("goals"),
			"columns": rows,
			"artifacts": {"recipe": str(recipe_path.relative_to(self.run_dir))
						  if recipe_path else None,
						  "chain": f"{self.DOMAIN_RECORD}/chain.json",
						  "status": f"{self.DOMAIN_RECORD}/STATUS.log",
						  "extracted": "03_results/extracted.json"},
		}
		(self.run_dir / "experiment.json").write_text(json.dumps(pkg, indent=1, default=str))
		print(f"✓ experiment.json: {pkg['columns_succeeded']}/{len(rows)} model year(s)")
		return pkg

	def _package(self, plan, results, config=None):
		if self._is_domain_rows(self._rehydrate_case_inputs() or []):
			return self._package_domain(plan, results, config)
		return super()._package(plan, results, config)

	def _analysis_skip_reason(self, experiments):
		if self._has_backend_report(experiments):
			return None
		if self._is_domain_rows(experiments):
			return ("the Analyzer reads sampled-column rows, and this whole-domain "
					"run has no regional numbers (they come from an experiment's "
					"analysis, and its plan names none), so experiment.json holds "
					"the year gates' numbers only")
		return None

	def _has_backend_report(self, experiments) -> bool:
		"""A whole-domain experiment whose plan named an analysis: its report
		is rendered from the regional numbers (agents.analysis.domain_report),
		no LLM. Numbers that could not be made are a failed report, not a skip:
		the plan asked for them (03_results/regional.json then holds the error,
		and the renderer raises with it)."""
		return self._is_experiment_rows(experiments) and bool(self._regional_from_disk())

	def _backend_report(self, experiments, results, config) -> Dict[str, Any]:
		from agents.analysis.domain_report import build
		return build(self.run_dir)

	# ─────────────────────────────────────────────────────────
	# WHOLE-DOMAIN EXPERIMENTS: a control and runs of named changes
	# ─────────────────────────────────────────────────────────
	# docs/ELM_DOMAIN_EXPERIMENTS.md. strategy['domain_experiment'] names a
	# control (a prepared one the server lists, reused: its output is compared
	# and nothing is spent on it; or a recipe that runs too) and runs that
	# differ from it by changes from the server's list. The server checks the
	# experiment and writes every run's inputs (build_domain_runs); each run
	# that executes is a piece-3 chain in its own record directory,
	# domain_case/<run>/. One row per run, the control's included, each with
	# its "run" name: that key is what marks an experiment's rows.
	DOMAIN_EXPERIMENT = "domain_experiment.json"
	DOMAIN_CONTROL_FACTS = "domain_control_facts.json"
	# The year numbers a changed run is set against its control on.
	DIFF_KEYS = ("gpp_gC_m2_yr", "peak_tlai", "et_mm", "runoff_mm", "swe_apr1_mm",
				 "totecosysc_change_frac")

	@classmethod
	def _is_experiment_rows(cls, rows) -> bool:
		rows = list(rows or [])
		return cls._is_domain_rows(rows) and all("run" in r for r in rows)

	def _run_record(self, row) -> Path:
		return self.run_dir / self.DOMAIN_RECORD / row["run"]

	@staticmethod
	def _executed(rows):
		return [r for r in rows if not r.get("reused")]

	def _control_recipe(self, ctl, client) -> Dict[str, Any]:
		"""The recipe of a control that runs too: given whole, or read from an
		existing case under a new name, as piece 3's domain_case is."""
		if ctl.get("recipe"):
			return dict(ctl["recipe"])
		if not (ctl.get("from_case") and ctl.get("case")):
			raise RuntimeError("an experiment whose control runs too needs the control's "
							   "recipe, or from_case with a new case name")
		out = self._mcp_call(client, "read_domain_recipe", {
			"case_dir": str(ctl["from_case"]), "case": str(ctl["case"]),
			"case_root": str(ctl.get("case_root") or "")})
		if not out.get("ok"):
			raise RuntimeError(f"read_domain_recipe: {out.get('error')}")
		if out.get("problems"):
			raise RuntimeError("the control's recipe cannot be used: "
							   + "; ".join(out["problems"]))
		return out["recipe"]

	def _build_domain_experiment(self, config: Dict[str, Any]) -> Dict[str, Any]:
		"""The rows of a whole-domain experiment, one per run: checked by the
		server against its change list and the control's facts, then each
		run's inputs written by it (build_domain_runs). The input set, which
		every run shares, is checked here, before any job."""
		cfg = config or {}
		exp = dict((cfg.get("strategy") or {}).get("domain_experiment") or {})
		client = self._domain_client(cfg)
		ctl = dict(exp.get("control") or {})
		reuse = bool(ctl.get("reuse"))
		if not isinstance(exp.get("runs"), list):
			raise RuntimeError("domain_experiment's runs must be a list: runs that differ "
							   "from the control by named changes, or none to read the "
							   "control alone")
		if reuse and not ctl.get("name"):
			raise RuntimeError("a reused control is a prepared one: name it "
							   "(describe_domain_controls)")
		if not reuse:
			(self.input_dir / self.DOMAIN_RECIPE).write_text(
				json.dumps(self._control_recipe(ctl, client), indent=1))
		if exp.get("years"):
			years = [int(y) for y in exp["years"]]
		elif cfg.get("yr_start"):
			y0 = int(cfg["yr_start"])
			years = list(range(y0, int(cfg.get("yr_end") or y0) + 1))
		else:
			raise RuntimeError("domain_experiment names no years and the study has no "
							   "period")
		chain = {k: exp[k] for k in ("smoke_days", "smoke", "partition", "tag", "walls",
									 "allow_forcing_cycle") if k in exp}
		if "partition" not in chain and cfg.get("queue"):
			chain["partition"] = str(cfg["queue"])
		record = {"control": {"name": ctl.get("name"), "reuse": reuse},
				  "runs": exp["runs"], "years": years, "chain": chain}
		(self.input_dir / self.DOMAIN_EXPERIMENT).write_text(json.dumps(record, indent=1))
		chk = self._mcp_call(client, "check_domain_experiment", {"experiment": record},
							 budget=600)
		if chk.get("error") or chk.get("wont_build"):
			raise RuntimeError("the experiment will not build: " + "; ".join(
				[str(chk["error"])] if chk.get("error") else
				[f"{w['where']}: {w['what']}" for w in chk["wont_build"]]))
		facts = None
		if ctl.get("name"):
			d = self._mcp_call(client, "describe_domain_controls", {}, budget=120)
			facts = next((c for c in d.get("controls") or [] if c.get("name") == ctl["name"]),
						 None)
		built = self._mcp_call(client, "build_domain_runs", {"run_dir": str(self.run_dir)},
							   budget=600)
		if not built.get("ok"):
			raise RuntimeError(f"build_domain_runs: {built.get('error')}")
		(self.input_dir / self.DOMAIN_CONTROL_FACTS).write_text(json.dumps(
			{"control": built["control"], "facts": facts, "check": chk}, indent=1, default=str))
		DC = self._sibling_module("domain_case")
		inputs, ic = None, {}
		if built["runs"]:
			# The input set every executed run shares; a control read alone ran
			# with its inputs already and spends nothing now.
			first = built["runs"][0]["name"]
			inputs = DC.Recipe.from_json((self.input_dir / "runs" / first /
										  self.DOMAIN_RECIPE).read_text()).inputs()
			ic = self._mcp_call(client, "check_domain_inputs", {
				"surface": str(inputs["surface"]), "domain": str(inputs["domain"]),
				"finidat": str(inputs["finidat"]), "run_dir": str(self.run_dir)}, budget=600)
			if not ic.get("ok"):
				raise RuntimeError(f"check_domain_inputs: {ic.get('error')}")
			if not ic.get("pass"):
				raise RuntimeError("the input set fails its check, so nothing was submitted: "
								   + "; ".join(ic.get("reasons") or []))
		base = {"kind": self.KIND_DOMAIN, "years": years, "forcing_start": years[0],
				"forcing_end": years[-1], "inputs": inputs, "inputs_check": ic.get("path")}
		rows = []
		if reuse:
			c = built["control"]
			rows.append(dict(base, case_name=Path(c["case_dir"]).name, run="control",
							 reused=True, case_dir=c["case_dir"], run_dir=c["run_dir"],
							 changes=[], after=None))
		for r in built["runs"]:
			rows.append(dict(base, case_name=r["case"], run=r["name"], reused=False,
							 case_dir=r["case_dir"], changes=r["changes"], after=r["after"]))
		(self.input_dir / self.CASE_INPUTS).write_text(json.dumps(rows, indent=2))
		print(f"   experiment on {ctl.get('name') or 'its own control'}: "
			  + ", ".join(r["run"] + (" (reused)" if r.get("reused") else "") for r in rows)
			  + f"; years {years[0]}-{years[-1]}; about "
			  f"{(chk.get('cost') or {}).get('node_hours')} node-hours")
		return {"columns": rows,
				"domain_experiment": {"experiment": f"01_inputs/{self.DOMAIN_EXPERIMENT}",
									  "runs": "01_inputs/runs",
									  "control_facts": f"01_inputs/{self.DOMAIN_CONTROL_FACTS}"}}

	def _submit_experiment(self, experiments, config) -> "Pending":
		"""Each executed run's chain, the control's first; a run compared with
		a control that runs too waits for the control's case job. Job B waits
		on every job of every run."""
		if not self._executed(experiments):
			# THE CONTROL ALONE: nothing to run. The rows go straight on to
			# extraction, as a finished run's would.
			self._attach_runs(experiments)
			print("   nothing to run: the experiment reads the reused control alone")
			return experiments
		client = self._domain_client(config)
		email = self._notify_email(config)
		jobs: List[Dict[str, str]] = []
		case_job: Dict[str, str] = {}
		for e in self._executed(experiments):
			after = case_job.get(e.get("after")) if e.get("after") else ""
			if e.get("after") and not after:
				raise RuntimeError(f"run {e['run']} waits for {e['after']}, whose case job "
								   f"was not submitted")
			out = self._mcp_call(client, "submit_domain_chain", {
				"run_dir": str(self.run_dir), "run": e["run"], "after_job": after or ""},
				budget=600)
			if not out.get("ok"):
				raise RuntimeError(
					f"submit_domain_chain for run {e['run']}: {out.get('error')}"
					+ (": " + "; ".join(out["reasons"]) if out.get("reasons") else "")
					+ (f"; already queued for earlier runs: "
					   f"{', '.join(j['run'] + '/' + j['stage'] + '=' + j['job_id'] for j in jobs)}"
					   if jobs else ""))
			for j in out["jobs"]:
				jobs.append({"run": e["run"], "stage": j["stage"], "job_id": str(j["job_id"])})
				print(f"   {e['run']:>12} {j['stage']:>8}  job {j['job_id']}")
			case_job[e["run"]] = next((str(j["job_id"]) for j in out["jobs"]
									   if j["stage"] == "case"), "")
		ids = [j["job_id"] for j in jobs]
		job_b = self._submit_job_b(":".join(ids), config, email)
		return Pending(job_b or ids[-1], scope="domain", n_cases=len(experiments),
					   chain=jobs, job_ids=ids, job_id_b=job_b or None, via="mcp",
					   notify=email or None)

	def _attach_runs(self, experiments) -> bool:
		"""Every executed run's chain record attached to its row (piece 3's
		rule per run: it succeeded only when every planned year passed). A
		reused control spent nothing and is marked so."""
		DC = self._sibling_module("domain_case")
		ok = True
		for e in experiments:
			if e.get("reused"):
				e["status"] = "reused"
				continue
			rd = self._run_record(e)
			if not (rd / DC.CHAIN_FILE).is_file():
				e["status"], e["chain"] = "not submitted", None
				ok = False
				continue
			ok = self._attach_chain([e], DC.chain_records(rd)) and ok
		return ok

	def _poll_experiment(self, experiments, config):
		client = self._domain_client(config)
		going = []
		for e in self._executed(experiments):
			st = self._mcp_call(client, "check_domain_chain",
								{"run_dir": str(self.run_dir), "run": e["run"]})
			if not st.get("ok"):
				raise RuntimeError(f"check_domain_chain for run {e['run']}: {st.get('error')}")
			if st.get("active"):
				going.append(e["run"])
		if going:
			print(f"   runs still going: {', '.join(going)}; nothing to collect yet")
			return None
		self._attach_runs(experiments)
		print("   ✓ every run's chain ended: " + ", ".join(
			f"{e['run']} {e.get('status')}" for e in experiments))
		return experiments

	def _adopt_experiment(self, experiments) -> Dict[str, Any]:
		self._attach_runs(experiments)
		self._save_case_inputs(experiments)
		n_ok = sum(1 for e in experiments if e.get("status") in ("completed", "reused"))
		self._mark("build_cases", n_experiments=len(experiments), n_ok=n_ok,
				   adopted_from=f"{self.DOMAIN_RECORD}/<run>/chain.json")
		self._mark("run", n_results=len(experiments), n_ok=n_ok,
				   adopted_from=f"{self.DOMAIN_RECORD}/<run>/checks")
		return {"build_cases": len(experiments), "run": len(experiments)}

	def _collect_experiment(self, experiments) -> Dict[str, bool]:
		DC = self._sibling_module("domain_case")
		self._attach_runs(experiments)
		lines = []
		for e in experiments:
			if e.get("reused"):
				lines.append(f"run control: reused {e['case_dir']} (nothing spent)")
				continue
			recs = DC.chain_records(self._run_record(e)) if \
				(self._run_record(e) / DC.CHAIN_FILE).is_file() else {}
			lines.append(f"run {e['run']} (case {e['case_name']}): years passed "
						 f"{recs.get('years_passed')}, first stage without a passing check "
						 f"{recs.get('first_unpassed')}")
			lines += [f"  {s['stage']:>8} job {s['job_id']} check "
					  f"{'none' if s['check_pass'] is None else 'PASS' if s['check_pass'] else 'FAIL'}"
					  for s in recs.get("stages") or []]
		(self.results_dir / "execution_report.txt").write_text("\n".join(lines) + "\n")
		print("\n".join(lines))
		return {e["case_name"]: e.get("status") in ("completed", "reused") for e in experiments}

	def _control_facts(self) -> Dict[str, Any]:
		p = self.input_dir / self.DOMAIN_CONTROL_FACTS
		return json.loads(p.read_text()) if p.is_file() else {}

	def _extract_experiment(self, experiments, config=None) -> Dict[str, Any]:
		"""One row per run and model year: an executed run's from its year
		checks, the reused control's from its facts; every changed run's row
		carries its difference from the control in that year."""
		import re
		DC = self._sibling_module("domain_case")
		cf = ((self._control_facts().get("facts") or {}).get("facts") or {}).get("per_year") or {}
		rows, ctl_rows = [], {}
		for e in experiments:
			if e.get("reused"):
				for y in e["years"]:
					m = cf.get(str(y)) or {}
					row = {"case_name": e["case_name"], "kind": self.KIND_DOMAIN, "run": "control",
						   "year": int(y), "region": "vegetated cells (PCT_NATVEG = 100)",
						   "status": "reused" if m else "not available",
						   "metrics": {k: m[k] for k in self.DOMAIN_YEAR_UNITS if m.get(k) is not None},
						   "source": "the reused control's facts (describe_domain_controls)",
						   "reasons": [] if m else [f"the control's facts hold no {y}"]}
					rows.append(row)
					ctl_rows[int(y)] = row
				continue
			rd = self._run_record(e)
			recs = DC.chain_records(rd) if (rd / DC.CHAIN_FILE).is_file() else {"stages": []}
			have = set()
			for st in recs.get("stages") or []:
				if not re.fullmatch(r"y\d{4}", st["stage"]) or not st.get("check"):
					continue
				y = st["stage"][1:]
				yr = (json.loads(Path(st["check"]).read_text()).get("years") or {}).get(y) or {}
				row = {"case_name": e["case_name"], "kind": self.KIND_DOMAIN, "run": e["run"],
					   "year": int(y), "region": "vegetated cells (PCT_NATVEG = 100)",
					   "status": "completed" if yr.get("pass") else "failed",
					   "job_id": st["job_id"], "check": st["check"],
					   "metrics": {k: yr[k] for k in self.DOMAIN_YEAR_UNITS if k in yr},
					   "expected": yr.get("expected"),
					   "reasons": yr.get("reasons") or []}
				rows.append(row)
				have.add(int(y))
				if e["run"] == "control":
					ctl_rows[int(y)] = row
			for y in e["years"]:
				if int(y) not in have:
					rows.append({"case_name": e["case_name"], "kind": self.KIND_DOMAIN,
								 "run": e["run"], "year": int(y), "status": "not run",
								 "metrics": {}, "reasons": [
									 f"the chain stopped before this year (first stage without "
									 f"a passing check: {recs.get('first_unpassed')})"]})
		for row in rows:
			c = ctl_rows.get(row["year"])
			if row["run"] != "control" and c and c["metrics"] and row["metrics"]:
				row["difference_from_control"] = {
					k: row["metrics"][k] - c["metrics"][k] for k in self.DIFF_KEYS
					if row["metrics"].get(k) is not None and c["metrics"].get(k) is not None}
		rows.sort(key=lambda r: (r["run"] != "control", r["run"], r["year"]))
		(self.results_dir / "extracted.json").write_text(json.dumps(
			{"kind": self.KIND_DOMAIN, "experiment": True, "rows": rows,
			 "source": f"{self.DOMAIN_RECORD}/<run>/checks/y<YEAR>_<job>.json, and the reused "
					   f"control's facts"}, indent=1, default=str))
		print(f"   ✓ {len(rows)} run-year row(s) read")
		regional = self._regional(experiments, config)
		return {"rows": rows, "units": dict(self.DOMAIN_YEAR_UNITS), "extra_summary": {},
				"spinup_dropped": {}, "regional": regional}

	def _experiment_analysis(self, config) -> Dict[str, Any]:
		"""The plan's analysis of a whole-domain experiment: from the study's
		strategy, or strategy.json beside the run when a finalize job has no
		strategy in its config."""
		st = (config or {}).get("strategy")
		if not st and (self.run_dir / "strategy.json").is_file():
			st = json.loads((self.run_dir / "strategy.json").read_text())
		return ((st or {}).get("domain_experiment") or {}).get("analysis") or {}

	def _regional(self, experiments, config) -> Optional[Dict[str, Any]]:
		"""The experiment's numbers by region (domain_analysis), for every run
		with output, written to 03_results/regional.json. None when the plan
		names no analysis; the reason, when they cannot be made."""
		analysis = self._experiment_analysis(config)
		if not analysis.get("variables"):
			return None
		A = self._sibling_module("domain_analysis")
		DC = self._sibling_module("domain_case")
		DCT = self._sibling_module("domain_controls")
		try:
			ctl = self._control_facts().get("control") or {}
			control, runs = None, {}
			for e in experiments:
				if e.get("reused"):
					entry = DCT.get(ctl.get("name"))
					control, runs["control"] = entry, entry["run_dir"]
					continue
				rd = self._run_record(e)
				if not (rd / DC.CHAIN_FILE).is_file():
					continue
				checks = [s["check"] for s in DC.chain_records(rd)["stages"]
						  if s["stage"].startswith("y") and s.get("check")]
				if not checks:
					continue
				runs[e["run"]] = json.loads(Path(checks[0]).read_text())["run_dir"]
				if e["run"] == "control":
					control = {"case_dir": e["case_dir"], "regions": {}, "observations": {}}
			if control is None:
				raise RuntimeError("the control has no output to read")
			table = A.experiment(control, runs, analysis,
								 [int(y) for y in experiments[0]["years"]])
		except Exception as e:                                      # noqa: BLE001
			table = {"error": f"{type(e).__name__}: {e}"}
			print(f"   ⚠️  the regional numbers could not be made: {table['error']}")
		(self.results_dir / "regional.json").write_text(json.dumps(table, indent=1, default=str))
		if "error" not in table:
			print(f"   ✓ regional numbers: {len(table['regions'])} region(s), "
				  f"{len(table['runs'])} run(s), {len(table['problems'])} problem(s)")
		return table

	def _regional_from_disk(self) -> Optional[Dict[str, Any]]:
		p = self.results_dir / "regional.json"
		return json.loads(p.read_text()) if p.is_file() else None

	def _package_experiment(self, plan, results, config) -> Dict[str, Any]:
		"""experiment.json for a whole-domain experiment: the control and every
		run with its changes (what each means and does not, as the server's
		list says), the run-year rows with their differences from the control,
		each run's chain, and one chain summary across the executed runs."""
		DC = self._sibling_module("domain_case")
		cfg = config or {}
		rows = [dict(r) for r in self._extract_rows(results) if isinstance(r, dict)]
		experiments = self._rehydrate_case_inputs() or []
		cf = self._control_facts()
		runs, chains = [], {}
		for e in experiments:
			ep = self.input_dir / "runs" / e["run"] / "domain_expectations.json"
			changes = (json.loads(ep.read_text()).get("changes") if ep.is_file() else []) or []
			rp = self._run_record(e) / DC.RECIPE_FILE
			recipe = json.loads(rp.read_text()) if rp.is_file() else {}
			built = None
			if not e.get("reused") and (self._run_record(e) / DC.CHAIN_FILE).is_file():
				chains[e["run"]] = DC.chain_records(self._run_record(e))
				for st in chains[e["run"]]["stages"]:
					if st["stage"] == "case" and st.get("check"):
						built = (json.loads(Path(st["check"]).read_text()) or {}).get("source")
			runs.append({"run": e["run"], "case": e["case_name"], "case_dir": e.get("case_dir"),
						 "reused": bool(e.get("reused")), "status": e.get("status"),
						 "changes": changes, "source": recipe.get("source"),
						 "built_from": built, "after": e.get("after")})
		planned = sorted({int(y) for e in experiments for y in e.get("years") or []})
		executed = [c for c in chains.values()]
		if executed:
			passed = sorted(set.intersection(*[set(c["years_passed"]) for c in executed]))
		else:
			# THE CONTROL ALONE: its years are there when its facts hold them.
			passed = sorted({r["year"] for r in rows
							 if r.get("run") == "control" and r.get("status") == "reused"})
		first = next((f"{r}: {c['first_unpassed']}" for r, c in chains.items()
					  if c.get("first_unpassed")), None)
		if len(chains) < len(self._executed(experiments)):
			first = first or "a run was not submitted"
		missing = sorted(r["year"] for r in rows
						 if r.get("run") == "control" and r.get("status") != "reused")
		if not executed and missing:
			# The facts are stale (a history file is newer than them) or were
			# never computed for these years: refresh_domain_control_facts.
			first = first or f"control: its facts hold no {missing}"
		# WHAT THE RUNS COST, as each chain's report stage read it back from
		# the scheduler (STATUS.log); a reused control spent nothing.
		compute = {"runs": {}, "seconds": 0, "note": None}
		for r, c in chains.items():
			jobs = DC.report_lines(c["record_dir"])
			secs = [j["seconds"] for j in jobs if isinstance(j.get("seconds"), int)]
			compute["runs"][r] = {"jobs": jobs, "seconds": sum(secs) if secs else None}
			compute["seconds"] += sum(secs)
		if any(not v["jobs"] for v in compute["runs"].values()):
			compute["note"] = ("a run's report stage has not written its job times "
							   "(it runs after every other job of the chain)")
		if any(e.get("reused") for e in experiments):
			compute["control"] = "reused: its output was read, nothing was spent on it"
		brief = cfg.get("brief") or {}
		pkg = {
			"model": self.MODEL, "kind": self.KIND_DOMAIN, "archetype": self.KIND_DOMAIN,
			"run_dir": str(self.run_dir), "created": datetime.now().isoformat(timespec="seconds"),
			"domain": {k: (brief.get("domain") or {}).get(k) for k in ("name", "huc", "bbox")},
			"experiment": {"control": cf.get("control"), "runs": runs,
						   "check": {k: (cf.get("check") or {}).get(k)
									 for k in ("facts", "unusual", "cost")}},
			"control_facts": (cf.get("facts") or {}).get("facts"),
			"chain": {"years_planned": planned, "years_passed": passed,
					  "first_unpassed": first, "runs": chains, "compute": compute},
			# What a regional number of a whole-domain run cannot support, from
			# the server's list; step 0 of the Analyzer reads it as it reads a
			# site run's limitations.
			"limitations": json.loads(json.dumps(
				self._sibling_module("domain_analysis").LIMITATIONS)),
			"columns_total": len(rows),
			"columns_succeeded": sum(1 for r in rows if r.get("status") in ("completed", "reused")),
			"variable_units": dict(self.DOMAIN_YEAR_UNITS),
			"strategy_check": self.strategy_report, "goals": (plan or {}).get("goals"),
			"columns": rows,
			"regional": (results or {}).get("regional") if isinstance(results, dict)
						 and "regional" in results else self._regional_from_disk(),
			"artifacts": {"experiment": f"01_inputs/{self.DOMAIN_EXPERIMENT}",
						  "runs": "01_inputs/runs", "records": f"{self.DOMAIN_RECORD}/<run>",
						  "control_facts": f"01_inputs/{self.DOMAIN_CONTROL_FACTS}",
						  "extracted": "03_results/extracted.json",
						  "regional": "03_results/regional.json"},
		}
		(self.run_dir / "experiment.json").write_text(json.dumps(pkg, indent=1, default=str))
		print(f"✓ experiment.json: {len(runs)} run(s), {pkg['columns_succeeded']}/{len(rows)} "
			  f"run-year row(s)")
		return pkg

	# ─────────────────────────────────────────────────────────
	# MAIN ENTRY POINT
	# ─────────────────────────────────────────────────────────
	# execute_plan now lives in ExperimentManagerBase. ELM keeps the stages
	# and the declarations; the sequence and its ordering guarantees are the
	# base's, so a second backend cannot re-implement them differently.
	NEEDS_CASE_BUILD   = True          # CIME case build, ~8 min for the first
	NEEDS_SCHEDULER = True          # sbatch + wait

	# WHAT build_elm_inputs_from_location ADDS TO A COLUMN, and what the base's
	# package carries from columns.json onto the row (composed with the
	# framework's own COLUMN_METADATA; a key neither names raises). Moved here
	# from the base list on 2026-08-18: which soil a column got and from where
	# is ELM knowledge, so its declaration lives beside the server.
	#
	#   forcing_cell, soil_*   WHICH FORCING CELL, and WHAT SOIL — two
	#                          independent facts about where the column ended
	#                          up, both settled at input time and neither
	#                          derivable from the row without them. Columns
	#                          sharing forcing_cell got the same rain, so a
	#                          difference between them is soil or terrain;
	#                          that reading is the reader's to make, and
	#                          nothing here precomputes it. The
	#                          soil-attribution figure and any claim that soil
	#                          explains a gradient rest on soil_*.
	#   warm_start             cold or warm. Decides which initialisation
	#                          caveat applies and how much of the early record
	#                          is the start rather than the soil —
	#                          extract.resolve_spinup reads it and trims 14
	#                          days for one and a year for the other.
	COLUMN_METADATA_EXTRA = KeySet(
		"ELM_COLUMN_METADATA",
		keep = ("forcing_cell", "soil_summary", "soil_top_texture",
				"soil_layers", "soil_source", "soil_profile",
				"warm_start",
				# A COUPLED FOLLOW-UP'S OWN FACTS (_build_coupled_columns +
				# the warm start's override): which run drove this one, the
				# water table it asked for, the one the finidat actually got
				# (clamps differ), the sentence saying so, and how far the
				# driving model moved it from ELM's own last answer.
				"coupled_from", "coupling_variable",
				"initial_water_table_m", "initial_water_table_written_m",
				"initial_water_table_note", "initial_soil_moisture_note",
				"water_table_prior_m", "water_table_delta_m"),
		# A site run starts warm by default and its columns do not say so
		# per column; a sweep writes it. Neither absence is a hole — nor is
		# any coupling key on a run nothing drove.
		optional = ("warm_start",
					"coupled_from", "coupling_variable",
					"initial_water_table_m", "initial_water_table_written_m",
					"initial_water_table_note", "initial_soil_moisture_note",
					"water_table_prior_m", "water_table_delta_m"),
		drop = {
			"outside_design_band": "set by warm_start when the snapped donor "
								   "leaves the band the sampler drew. A "
								   "sampling-design fact, read by the design "
								   "figure, not a property of the results",
			"initial_saturation_profile": "the driving model's profile, "
								   "already stamped into the finidat with "
								   "ELM's own porosity; a table-sized design "
								   "input — stays on columns.json, its note "
								   "travels instead",
		},
		source = "columns.json -> columns[*], as build_elm_inputs_from_location wrote them",
		where  = "mcp/elm-mcp/src/elm_exp_manager.py :: COLUMN_METADATA_EXTRA",
	)

	# What ELM's derived metrics MEAN. Lives here, not on the base, because
	# every entry names an ELM history variable: on the base it was silently
	# inherited by PFLOTRAN, whose experiment.json then advertised a surface
	# water budget it never computed. See ExperimentManagerBase.FIELD_SEMANTICS.
	FIELD_SEMANTICS = {
		"precip_mm_yr":            {"units": "mm/yr", "from": ["RAIN", "SNOW"],
									"note": "TOTAL precipitation: rain + snow"},
		"rainfall_mm_yr":          {"units": "mm/yr", "from": ["RAIN"]},
		"snowfall_mm_yr":          {"units": "mm/yr", "from": ["SNOW"]},
		# FIVE ENTRIES WENT ON 2026-08-13, and the note each of them needed is
		# the reason. `runoff_fraction` and `recharge_fraction` were the
		# recharge-vs-runoff SPLIT, denominator (QCHARGE + QOVER), and needed a
		# paragraph here saying they were not fractions of precipitation —
		# because the name says they are. `recharge_to_runoff_ratio` reached
		# 4,288,570 on a column with no runoff. `precip_total_mm_yr` was
		# `precip_mm_yr` under a second name. `annual_runoff_mm_yr` and
		# `annual_recharge_mm_yr` duplicated the water-budget terms at a
		# different rounding.
		#
		# Everything they answered is in `water_budget`, where every term
		# carries `_frac_of_P` and the denominator is IN THE NAME. A catalogue
		# entry that has to warn the reader what its key does not mean is a key
		# that should be renamed, not documented.
		"water_budget":            {"units": "mm/yr and fractions of P",
									"from": ["RAIN", "SNOW", "QOVER", "QDRAI",
											 "QCHARGE", "QINFL", "QSOIL",
											 "QVEGE", "QVEGT", "TWS"],
									"note": "each term as <name>_mm_yr AND "
											"<name>_frac_of_P. recharge is "
											"INTERNAL to TWS, not an export — "
											"see recharge_is_internal. "
											"unaccounted_mm_yr is what the "
											"balance cannot place, and it is "
											"0-14% of P on real runs"},
		"water_table_depth_m":     {"units": "m", "from": ["ZWT"],
									"note": "positive downward from the surface"},
		"peak_swe_modelled_mm":    {"units": "mm", "from": ["H2OSNO"],
									"note": "MODELLED peak. `peak_swe_mm` "
											"elsewhere is the OBSERVED peak at "
											"a snow pillow — different thing"},
		"tws_seasonal_range_mm":   {"units": "mm", "from": ["TWS"]},
		"n_days_in_record":        {"units": "days", "from": [],
									"note": "every mm/yr above is a daily rate "
											"x 365.25, so a short record is an "
											"EXTRAPOLATION by 365.25/this"},
	}

	# ─────────────────────────────────────────────────────────
	# THE MCP — the only path there is
	# ─────────────────────────────────────────────────────────
	# Not a default with a fallback behind it: _refine_columns raises without a
	# client, so a study that reaches any stage below has one. run_via_mcp=False
	# now fails at the first stage instead of quietly running a second
	# implementation of ELM that nobody has exercised since the move.
	MCP_NAME = "elm"

	def _announce(self, experiments, config) -> str:
		"""Say what is about to happen. PRINTS ONLY — it never asks.

		Asking belongs to the coordinator: it owns every other question put to
		the user (interactive_reception, the --resume prompt) and it owns the
		TTY guard that keeps those from hanging a scripted run. A backend that
		called input() here would be a compute stage holding the terminal open,
		and it only ended up here because the run directory and the column
		count happened to be in scope.

		The address is whatever the coordinator, the environment, or the
		remembered preference already settled on.
		"""
		n = len(experiments or [])
		email = self._notify_email(config)
		print()
		print(f"⚙️  This ELM study runs unattended.")
		print(f"    {n} column(s) · build + run + analysis, ~25-40 min in the queue.")
		print(f"    Nothing to watch. Results will appear in:")
		print(f"      {self.analysis_dir}")
		print(f"    ✉  {email}" if email else
			  "    (no notification address — nothing will tell you when it lands)")
		print()
		return email

	# ─────────────────────────────────────────────────────────
	# STEP 2 — JOBS A AND B  (the whole of the compute)
	# ─────────────────────────────────────────────────────────
	def _build_cases(self, experiments: List[Dict], config: Dict[str, Any] = None):
		"""JOB A through the MCP, then JOB B by dependency. Returns a Pending.

		Named build_cases because that is the base's slot for "a stage that may
		hand back a job id"; job A does the build AND the run, so _run has
		nothing left to submit by the time it is reached. ONE JOB, ALWAYS — the
		split flow cost three invocations for a 25-40 minute study (submit the
		build, come back and submit the run, come back and analyze), which made
		the person the scheduler.

		A builds the cases and runs every column, and stops. B is submitted HERE,
		with --dependency=afterany on A's id, and does the analysis. Then this
		returns and the framework exits — SLURM starts B by itself.

		WHY B IS SUBMITTED BY THE FRAMEWORK AND NOT BY A. The tool this replaces,
		run_elm_study, ended its job by running `workflow.py --finalize`: the
		server's job executing the client's code, which is the dependency the
		boundary rule forbids and the reason the two could not be separated.
		Submitting B from this side inverts it — the framework asks for the model
		run and arranges its own follow-up, and nothing in the server calls back.

		afterany, NEVER afterok. With afterok a failed ensemble means B never runs
		and no mail is ever sent — the silent failure of 2026-08-06, twice. afterany
		means B always runs and always reports, including "the ensemble failed".
		"""
		config = config or {}
		if self._is_domain_rows(experiments):
			return self._submit_domain(experiments, config)
		client = self._mcp(config)
		if client is None:
			raise RuntimeError(
				"the elm MCP is required to build and run ELM cases — the "
				"local by-import path was deleted 2026-08-10 "
				"(docs/EXP_MANAGER_ELM.md §4). Register an `elm` client in "
				"mcp_config.json.")
		# The server reads 01_inputs/case_inputs.json and nothing else. It does
		# not generate inputs and never opens a surfdata file: FSURDAT and
		# FINIDAT — the warm start's two products — crossed the boundary as
		# data when _refine_columns wrote this file.
		src = self.input_dir / self.CASE_INPUTS
		if not src.is_file():
			raise RuntimeError(
				f"{src} is missing — _refine_columns should have written it "
				f"via the elm MCP")
		# ── ALREADY INSIDE AN ALLOCATION? RUN IN PLACE. ─────────────────
		# The coupling iteration submits ONE job and runs its legs — ELM,
		# PFLOTRAN, ELM again — inside it; a nested sbatch from a compute
		# node would rejoin the queue and defeat the allocation. The wrapper
		# is srun-based, so the ensemble lands on this job's own cores. The
		# whole pipeline then continues synchronously (no Pending, no job B —
		# this very process does the analysis next). IDEAS_FORCE_SBATCH=1
		# restores the submit for someone deliberately queueing from a job.
		if os.environ.get("SLURM_JOB_ID") and \
				os.environ.get("IDEAS_FORCE_SBATCH") != "1":
			jid = os.environ["SLURM_JOB_ID"]
			wall = str(config.get("study_walltime", "02:00:00"))
			h, m, sec = (list(map(int, wall.split(":"))) + [0, 0])[:3]
			budget = h * 3600 + m * 60 + sec + 900
			print(f"   ⚙  inside allocation {jid} — running the ensemble IN "
				  f"PLACE (srun on this job's cores), no new submission")
			out = self._mcp_call(client, "run_elm_ensemble", {
				"run_dir": str(self.run_dir), "in_allocation": True,
			}, budget=float(budget))
			if out.get("error"):
				raise RuntimeError(f"run_elm_ensemble (in allocation): "
								   f"{out['error']}")
			by_name = {c.get("case_name"): c.get("case_dir")
					   for c in (out.get("cases") or [])}
			for e in experiments:
				if by_name.get(e.get("case_name")):
					e["case_dir"] = by_name[e["case_name"]]
			print(f"   ✓ {out.get('n_built')}/{out.get('n_cases')} case(s) "
				  f"built and run in {out.get('elapsed_s')} s "
				  f"(allocation {jid})")
			return experiments

		email = self._announce(experiments, config)
		out = self._mcp_call(client, "run_elm_ensemble", {
			"run_dir":  str(self.run_dir),
			"queue":    str(config.get("queue", "")),
			"walltime": str(config.get("study_walltime", "02:00:00")),
		}, budget=600)
		if out.get("error"):
			raise RuntimeError(f"run_elm_ensemble: {out['error']}")
		job_a = str(out["job_id"])
		job_b = self._submit_job_b(job_a, config, email)
		if email:
			print(f"   ✉  {email} will be mailed when job B lands")
		# BOTH IDS. `job_id` is what the framework POLLS — job B when there is
		# one, because B ending means the study is over. But B is the reporter;
		# A is the job that built and ran the columns, and its id was dropped
		# here and recorded nowhere else, so the run state named B's id beside
		# A's log and nothing could say how long the model actually took.
		# Measured 2026-08-13: the only runtime any artifact carried was the
		# framework's own 593.8 s tail against an 18 m 23 s ensemble.
		return Pending(job_b or job_a, n_cases=len(experiments or []),
					   job_id_a=job_a, job_id_b=job_b or None,
					   log=out.get("log_path"), via="mcp", scope="study",
					   notify=email or None)

	def _submit_job_b(self, job_a: str, config: Dict[str, Any],
					  email: str = "") -> Optional[str]:
		"""JOB B — the analysis, held until A ends. Returns its id, or None.

		Non-fatal: A is already queued and its model output does not depend on B.
		Losing B costs the analysis, which can be run by hand; raising here would
		strand a running ensemble with nothing recording that it exists.
		"""
		import shutil, subprocess
		if not shutil.which("sbatch"):
			print("   ⚠️  no sbatch — job B not submitted; analyse by hand")
			return None
		# REFUSE A NODE-LOCAL RUN DIRECTORY. A compute node cannot see this node's
		# /tmp: the first A/B attempt (770938/770939) put its scripts there and both
		# jobs died in two seconds with completely empty logs, which is
		# indistinguishable from a scheduler fault.
		#
		# It also makes this method safe to reach from a test. pytest's tmp_path is
		# under /tmp, and `sbatch` exists on a login node, so a test that got past
		# the mocked client would otherwise submit a real job into the queue.
		#
		# IT FAILS OPEN, AND SAYS SO. `df` not answering is not evidence that the
		# directory is node-local, so refusing on it would break every machine
		# where df is absent or slow. But an unmade check that prints nothing is
		# indistinguishable from a check that passed — which is exactly how this
		# went unnoticed once already, when a harness broke `df` and the bare
		# `except` turned the failure into a silent green light.
		try:
			fs = subprocess.check_output(
				["df", "-P", str(self.run_dir)], text=True,
				timeout=20).splitlines()[-1].split()[0]
		except Exception as e:                                  # noqa: BLE001
			print(f"   ⚠️  could not check whether {self.run_dir} is on shared "
				  f"storage ({type(e).__name__}: {e}) — submitting job B anyway. "
				  f"If it dies in seconds with an empty log, this is why.")
			fs = ""
		if fs.startswith("/dev/"):
			print(f"   ⚠️  {self.run_dir} is on {fs}, a node-local filesystem — "
				  f"job B not submitted. Put the run directory on $PSCRATCH.")
			return None
		fw = framework_dir()
		sb = self.run_dir / "ensemble_B.sbatch"
		mail = (f"#SBATCH --mail-user={email}\n#SBATCH --mail-type=END,FAIL"
				if email else "")
		sb.write_text(f"""#!/bin/bash
#SBATCH -J elm_B
#SBATCH -N 1
#SBATCH -p {config.get("queue") or os.environ.get("IDEAS_SLURM_QUEUE", "short")}
#SBATCH -A {config.get("account") or os.environ.get("IDEAS_SLURM_ACCOUNT", "e3sm")}
#SBATCH -t {config.get("analysis_walltime", "00:30:00")}
#SBATCH -o {self.run_dir}/ensemble_B.log
{mail}
cd {fw}
# JOB B RUNS THE ANALYSIS. It was deferred while the Analyzer was rebuilt, and
# the default stayed 0 long after — so every run since has stopped at the
# ensemble and been finished by hand, while SLURM recorded job B as FAILED in
# 2 s (notify_study exits non-zero on an incomplete study). Default flipped
# 2026-08-13, once the Analyzer had been verified end to end on both archived
# studies. IDEAS_RUN_ANALYSIS=0 still defers it.
if [ "${{IDEAS_RUN_ANALYSIS:-1}}" = "1" ]; then
  # --finalize, NOT --resume. This runs INSIDE job B, so --resume polls the
  # scheduler, finds this very job active, and stops one line short of the
  # analysis it was submitted to produce — "Job <B> is still running, N
  # column(s) queued", exit 1, 40 seconds. finalize_run exists precisely for
  # this and its docstring describes the failure; the template just never used
  # it. Observed on 773411 and 773413 (2026-08-13), both after job A had
  # COMPLETED and the output was on disk.
  {sys.executable} workflow.py --finalize {self.run_dir}
  {sys.executable} tools/notify_study.py {self.run_dir}
else
  {sys.executable} tools/notify_study.py {self.run_dir} --deferred
fi
exit $?
""")
		try:
			jid = subprocess.check_output(
				["sbatch", "--parsable", f"--dependency=afterany:{job_a}", str(sb)],
				text=True, timeout=120).strip().split(";")[0]
		except Exception as e:                                  # noqa: BLE001
			print(f"   ⚠️  job B not submitted ({e}) — A is running; analyse by hand")
			return None
		print(f"   job A {job_a} (build+run) → job B {jid} (afterany, reports)")
		return jid

	def _poll(self, record: Dict[str, Any], experiments, config):
		"""Has job A landed? The case directories if so, None if not.

		ONE SHAPE, because there is one job. build_cases is the only stage that
		returns a Pending now, so the dispatch on record['stage'] is gone with
		the split flow — and with it the way that dispatch went wrong: polling a
		build+run job down the `build_cases` branch handed back case dirs and let
		execute_plan walk into _run, which submitted the ensemble A HAD ALREADY
		RUN. Every column would have run twice, and the second pass would have
		overwritten the history files of the first.

		The case directories come back from check_elm_job itself. A stage that
		hands back a job id needs somewhere to hand back its ANSWER, and a few
		short strings can travel inline — unlike results, which is why reading
		those stayed on this side.
		"""
		if record.get("scope") == "domain":
			return self._poll_domain(record, experiments, config)
		client = self._mcp(config or {})
		if client is None:
			raise RuntimeError(
				f"job {record.get('job_id')} is recorded as pending but there "
				f"is no `elm` client to ask about it — register one in "
				f"mcp_config.json, or check it by hand with squeue")
		st = self._mcp_call(client, "check_elm_job",
							{"job_id": str(record.get("job_id")),
							 "run_dir": str(self.run_dir)})
		if st.get("active", True):
			print(f"   job {record.get('job_id')} is "
				  f"{st.get('state') or 'unanswered'} — nothing to collect yet")
			return None
		if not st.get("ok"):
			raise RuntimeError(
				f"the case build failed: {st.get('error')}"
				+ (f"\n{st.get('log_tail')}" if st.get("log_tail") else ""))
		by_name = {c.get("case_name"): c.get("case_dir")
				   for c in (st.get("cases") or [])}
		for e in experiments:
			if by_name.get(e.get("case_name")):
				e["case_dir"] = by_name[e["case_name"]]
		print(f"   ✓ {st.get('n_ok')}/{st.get('n_total')} case(s) built")
		return experiments

	def _build_case_inputs(self,
			   plan:   Dict[str, Any],
			   config: Dict[str, Any]) -> List[Dict]:
		"""Read what the MCP already wrote. Computes nothing.

		_refine_columns made the one call that produced case_inputs.json, so
		this stage exists only because the base's pipeline has a slot for it.
		The rows come back as PLAIN DATA with no live ELMAgent, and that is now
		the ONLY state they are ever in — not a resume-only special case. The
		--keepexe builder handle went with the local path; the job side does
		the cloning, on the compute node, out of case_inputs.json.
		"""
		rows = self._rehydrate_case_inputs()
		if not rows:
			raise RuntimeError(
				f"{self.input_dir / self.CASE_INPUTS} is missing or empty — "
				f"_refine_columns should have written it via the elm MCP")
		return rows

	# NO BUILD-TIME SURFACE FIGURE (deleted 2026-08-14, by request).
	# column_surfaces.png was drawn by the MCP's build job because it read each
	# case's generated FSURDAT through its `run/lnd_in`, which exists only after
	# the build. sampling_design.png shows the same soil from the donor data,
	# post-warm-start, and is the figure that gets looked at.

	BUILT_CASES = "built_cases.json"

	def adopt_completed_run(self) -> Dict[str, Any]:
		"""ELM also adopts the case build, and re-attaches where each landed.

		built_cases.json is the one thing the finalize cannot work out for
		itself: case directories carry a creation timestamp, so they cannot be
		derived from the case names. Without re-attaching them, _extract globs
		`{case_dir}/run/*.elm.h0.*.nc` against a case_dir of None and every
		column reports zero history files — a finished ensemble read as a
		total failure.
		"""
		experiments = self._rehydrate_case_inputs() or []
		if self._is_domain_rows(experiments):
			return self._adopt_domain(experiments)
		src = self.input_dir / self.BUILT_CASES
		adopted: Dict[str, Any] = {}
		if src.is_file() and experiments:
			try:
				built = json.loads(src.read_text())
				by_name = {c.get("case_name"): c.get("case_dir")
						   for c in (built.get("cases") or [])}
				n = 0
				for e in experiments:
					if by_name.get(e.get("case_name")):
						e["case_dir"] = by_name[e["case_name"]]
						n += 1
				self._save_case_inputs(experiments)
				self._mark("build_cases", n_experiments=len(experiments),
						   n_ok=n, adopted_from=self.BUILT_CASES)
				adopted["build_cases"] = n
			except Exception as e:                              # noqa: BLE001
				print(f"   ⚠️  could not adopt {self.BUILT_CASES} ({e}) — "
					  f"the case directories will be missing")
		adopted.update(super().adopt_completed_run())
		return adopted

	# ─────────────────────────────────────────────────────────
	# STEP 3 — RUN (writes to 03_results/)
	# ─────────────────────────────────────────────────────────
	def _run(self,
			 experiments: List[Dict],
			 config:      Dict[str, Any]):
		"""COLLECT. Job A already ran the columns; this never submits anything.

		The stage is still here because the base's sequence has a slot for it
		and because everything it does after the columns stop — the run
		summaries, execution_report.txt, results_summary.csv — still has to
		happen. What it does NOT do is run ELM: by the time execute_plan reaches
		this line, either job A has just been polled (the case dirs came back
		from _poll) or the run state says build_cases is done, and both mean the
		ensemble has already been through the queue.

		SUBMITTING HERE WAS A BUG, not a simplification removed. A --resume of a
		landed A+B study walked out of _poll into this stage and re-submitted
		every column, over the top of the history files the first pass wrote.
		Nothing had exercised it only because the analysis is deferred.

		Success is what is ON DISK: a column succeeded if ELM wrote it a history
		file. The scheduler being finished is what ends the wait; the files
		decide the outcome.
		"""
		if self._is_domain_rows(experiments):
			return self._collect_domain(experiments)
		return self._collect(experiments, self._outcomes_from_disk(experiments))

	def _collect(self,
				 experiments: List[Dict],
				 results:     Dict[str, bool]) -> Dict[str, bool]:
		"""Everything that happens once the columns have stopped running."""
		for exp in experiments:
			try:
				exp['run_summary'] = self._run_summary_for(exp)
			except Exception as e:                              # noqa: BLE001
				print(f"   ⚠️  run summary failed for {exp['case_name']}: {e}")

		n_ok   = sum(1 for v in results.values() if v)
		n_fail = len(results) - n_ok
		print(f"✓ {n_ok} succeeded, {n_fail} failed")

		# Post-run reporting → 03_results/
		self._write_execution_report(experiments, results)
		self._write_results_csv(experiments, results)

		return results

	@staticmethod
	def _run_summary_for(exp: Dict) -> Dict[str, Any]:
		"""The case's run summary. Off disk, always.

		There is no live agent to ask any more: the cases are built inside job
		A, and what comes back to this side is JSON. Asking the string repr of
		an ELMAgent for a summary is how this once reported zero history files
		for cases that had plenty — the branch that did it is gone rather than
		guarded, so there is nothing left to fall back FROM.
		"""
		cd = exp.get('case_dir')
		return {'history_files': sorted(glob.glob(f"{cd}/run/*.elm.h0.*.nc"))
								 if cd else []}

	def _outcomes_from_disk(self, experiments: List[Dict]) -> Dict[str, bool]:
		"""Which columns produced output — the one definition of success: a
		column succeeded if ELM wrote it a history file. A whole-domain case
		succeeded if at least one model year passed its check."""
		if self._is_experiment_rows(experiments):
			rows = [dict(e) for e in experiments]
			self._attach_runs(rows)
			return {e['case_name']: e.get('status') in ('completed', 'reused') for e in rows}
		if self._is_domain_rows(experiments):
			ok = self._attach_chain([dict(e) for e in experiments], self._domain_records())
			return {e['case_name']: ok for e in experiments}
		return {e['case_name']: bool(e.get('case_dir') and
									 glob.glob(f"{e['case_dir']}/run/*.elm.h0.*.nc"))
				for e in experiments}

	# ONE definition, in inputs.py, because two lists that must agree are two
	# lists that will not. A key added there and forgotten here would drop a
	# field from case_inputs.json silently — the file would exist, parse, and be
	# missing something the build needs.
	@property
	def CASE_KEYS(self):                                    # noqa: N802
		import inputs
		return inputs.CASE_KEYS

	def _serialise_case_inputs(self, experiments):
		"""Delegates to inputs.serialise_case_inputs — see there. A
		whole-domain row is plain data already and goes as it is."""
		if self._is_domain_rows(experiments):
			return [dict(e) for e in experiments]
		import inputs
		return inputs.serialise_case_inputs(experiments)

	def _rehydrate_handles(self, experiments, plan, config) -> None:
		"""Drop the ELMAgent string reprs a reloaded case_inputs.json carries.

		An agent is an object and the manifest is JSON, so what comes back is a
		string: truthy, with no methods, and therefore a thing that fails at the
		call site rather than here. Nothing on this side calls one any more —
		the build happens inside job A — so they are dropped and the key stays
		absent.
		"""
		stale = [e for e in experiments if isinstance(e.get('elm_agent'), str)]
		for e in stale:
			e.pop('elm_agent', None)
		if stale:
			print(f"   ↻ {len(stale)} case handle(s) are not JSON — working "
				  f"from the case directories on disk")

	def _write_execution_report(self,
								experiments: List[Dict],
								results:     Dict[str, bool]) -> None:
		"""Write a plain-text run report to 03_results/execution_report.txt."""
		report_file = self.results_dir / "execution_report.txt"

		n_total   = len(experiments)
		n_success = sum(results.values())

		lines = []
		lines.append("=" * 60)
		lines.append("ELM EXECUTION REPORT")
		lines.append("=" * 60)
		lines.append(f"Generated:         {datetime.now().isoformat()}")
		lines.append(f"Run directory:     {self.run_dir}")
		lines.append(f"Total experiments: {n_total}")
		lines.append(f"Successful:        {n_success}")
		lines.append(f"Failed:            {n_total - n_success}")
		lines.append("")

		for exp in experiments:
			case_name   = exp['case_name']
			status      = ('completed' if results.get(case_name)
						   else 'failed')
			run_summary = exp.get('run_summary', {}) or {}
			n_hist      = len(run_summary.get('history_files', []))

			lines.append(f"── {case_name}")
			lines.append(f"   status:           {status}")
			lines.append(f"   scenario:         {exp.get('scenario_name', '?')}")
			lines.append(f"   forcing_period:   {exp.get('forcing_period', '?')}")
			lines.append(f"   forcing_years:    "
						 f"{exp.get('forcing_start', '?')}-"
						 f"{exp.get('forcing_end', '?')}")
			lines.append(f"   soil_config:      {exp.get('soil_config', '?')}")
			if exp.get('substrate'):
				lines.append(f"   substrate:        {exp['substrate']}")
			lines.append(f"   case_dir:         {exp.get('case_dir', '?')}")
			lines.append(f"   history_files:    {n_hist}")
			lines.append("")

		report_file.write_text('\n'.join(lines))
		print(f"✓ execution_report.txt saved")

	def _write_results_csv(self,
						   experiments: List[Dict],
						   results:     Dict[str, bool]) -> None:
		"""Write a tabular summary to 03_results/results_summary.csv."""
		csv_file = self.results_dir / "results_summary.csv"
		fields = ['case_name', 'scenario_name', 'forcing_period',
				  'forcing_start', 'forcing_end', 'soil_config',
				  'substrate', 'status', 'history_file_count',
				  'case_dir']

		with open(csv_file, 'w', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=fields)
			writer.writeheader()
			for exp in experiments:
				case_name   = exp['case_name']
				run_summary = exp.get('run_summary', {}) or {}
				writer.writerow({
					'case_name':          case_name,
					'scenario_name':      exp.get('scenario_name', ''),
					'forcing_period':     exp.get('forcing_period', ''),
					'forcing_start':      exp.get('forcing_start', ''),
					'forcing_end':        exp.get('forcing_end', ''),
					'soil_config':        exp.get('soil_config', ''),
					'substrate':          exp.get('substrate', '') or '',
					'status':             ('completed'
										   if results.get(case_name)
										   else 'failed'),
					'history_file_count': len(run_summary.get(
						'history_files', [])),
					'case_dir':           exp.get('case_dir', ''),
				})
		print(f"✓ results_summary.csv saved")

	# ─────────────────────────────────────────────────────────
	# STEP 4 — ANALYZE (writes to 04_analysis/)
	# ─────────────────────────────────────────────────────────
	def _extract(self,
				 experiments,
				 plan:   Dict[str, Any] = None,
				 config: Dict[str, Any] = None) -> Dict[str, Any]:
		"""Read the ELM history NetCDFs, then build the rows FROM what was read.

		TWO STEPS, IN THIS ORDER, AND NOTHING BETWEEN THEM:

		    extract_run   history files -> 03_results/extracted.json
		    build_rows    extracted.json -> the per-column rows

		This used to be one object, ELMResultsAnalyzer, which read and computed
		in the same pass and called write_extracted itself. That made TWO
		writers of one artifact — it and extract_run — reached by two paths.
		They did not disagree, but the shape is the one that already cost a
		day: an extracted.json from one morning beside an experiment.json from
		the next, 351 days against 352, nothing saying so. One writer now, and
		the rows are reconstructible from the file it wrote.

		OVERWRITE, deliberately. This stage has just run the model, so it is
		the authoritative read; reuse belongs to the tool and to the comparison,
		which may legitimately open an artifact somebody else made. A resumed
		run does not pay for it twice — execute_plan skips a completed stage
		outright, so this body does not run at all.

		This is extraction, not analysis: it knows ELM's output format and
		nothing about what the numbers mean. Figures, observation comparison
		and interpretation belong to the Analyzer (src/agents/analyzer.py).

		Also attaches the honesty payload (structural + configuration
		limitations, assumptions ledger), which _ensemble_blocks folds into
		experiment.json and step 0 turns into the caveats that bind the
		interpreter.
		"""
		if self._is_experiment_rows(experiments):
			return self._extract_experiment(experiments, config)
		if self._is_domain_rows(experiments):
			return self._extract_domain(experiments)
		cfg  = config or {}
		plan = plan or {}
		honesty = {}
		try:
			from core.limitations import select_limitations
			couplers = plan.get("CONDITIONS_COUPLERS") or [{}]
			y0 = int(couplers[0].get("DATM_CLMNCEP_YR_START", cfg.get("yr_start", 1995)))
			y1 = int(couplers[0].get("DATM_CLMNCEP_YR_END",   cfg.get("yr_end", y0)))
			honesty = {
				"limitations": select_limitations(
					n_years       = max(1, y1 - y0 + 1),
					warm_start    = bool(couplers[0].get("FINIDAT")),
					forcing       = cfg.get("forcing", "nldas"),
					spinup_years  = int(cfg.get("spinup_years", 0)),
					# WHICH warm start, and whether the soil was kept with it.
					# Without these the caveat cannot tell an equilibrated,
					# self-consistent state from the mismatch that made year
					# one a relaxation — and it defaulted to warning about both.
					# ALWAYS "conus" — there is one warm start (2026-08-12).
					# This used to dig a `source` out of the config, which by
					# then could be True, a dict, or a string; with one source
					# the question is answered here instead of re-derived.
					warm_source   = "conus",
					# WHAT THE RUN ACTUALLY DID, not what a site run always
					# does. Both of these were hardcoded on the premise that a
					# warm start is required and the donor's surfdata is always
					# what ELM runs on. A conceptual sweep is cold and
					# prescribes its own soil, so both were simply false there,
					# and the caveat list described a different run.
					soil_source   = str(couplers[0].get("SOIL_SOURCE") or "conus"),
					archetype     = str((cfg.get("strategy") or {}).get("archetype")
										or (cfg.get("brief") or {}).get("design_archetype")
										or ""),
					written_weather = bool(couplers[0].get("PRESCRIBED_WEATHER")),
				),
				"assumptions_ledger": (
					json.loads((self.run_dir / "assumptions.json").read_text())
					if (self.run_dir / "assumptions.json").exists() else []),
			}
		except (NameError, AttributeError, TypeError) as e:
			# A BUG HERE IS NOT A MISSING PAYLOAD, and the handler used to
			# report it as one. A bare `except Exception` swallowed a NameError
			# — a stale variable left by a refactor — and printed "limitations
			# payload unavailable", which reads as "this run has no
			# limitations". It does not: it means this code did not run. The
			# consequence is silent and large, because extra_summary is what
			# _ensemble_blocks folds into experiment.json and step 0 turns into
			# the caveats that bind the interpreter. Found 2026-08-13 by
			# re-running the stage over an archived study.
			print(f"   ✗ BUG in the limitations payload — {type(e).__name__}: "
				  f"{e}. This is a code fault, not a run without caveats; "
				  f"experiment.json will carry no limitations.")
			raise
		except Exception as e:
			# Genuinely absent input: no assumptions.json, an unreadable one, a
			# plan with no couplers. The run stands; the caveats do not.
			print(f"   ⚠️  limitations payload unavailable ({type(e).__name__}: "
				  f"{e}) — experiment.json will carry no limitations")

		from extract import VARIABLE_UNITS, extract_run
		from column_rows import build_rows, spinup_dropped

		res = extract_run(str(self.run_dir), overwrite=True)
		if not res.get("ok"):
			# NOT fatal here. _package still runs and records that the run
			# produced nothing readable, which is the only account of what
			# happened; raising would throw away a finished ensemble.
			print(f"   ✗ extraction failed: {res.get('error')}")
			return {"rows": [], "units": dict(VARIABLE_UNITS),
					"extra_summary": honesty}
		print(f"   ✓ {res['n_ok']}/{res['n_columns']} column(s) read "
			  f"-> 03_results/extracted.json ({res.get('size_mb')} MB)")

		rows = build_rows(self.run_dir)
		print(f"   ✓ {len(rows)} row(s) built from the artifact")

		# READ OFF THE ARTIFACT, not remembered from the extraction pass — the
		# same rule as the rows. It travels in the stage contract so the base
		# can fold it into experiment.json, where step 4 has always looked for
		# it and never found it.
		drop = spinup_dropped(self.run_dir)
		if drop:
			span = (f", from {drop['from']} to {drop['to']}"
					if drop.get("applied") else "")
			print(f"   ✓ {drop.get('basis', 'start-up')} trim recorded: "
				  f"{drop['days']} d, {drop['timesteps_dropped']} timestep(s)"
				  f"{span}")
			if drop.get("note"):
				print(f"   ! {drop['note']}")

		# THE STAGE'S OUTPUT IS DATA — a plain dict in the contract's shape, not
		# a live object the base has to read with getattr. That was the last
		# holdout, and it is what makes this stage crossable by a tool call.
		return {"rows": list(rows.values()),
				"units": dict(VARIABLE_UNITS),
				"extra_summary": honesty,
				"spinup_dropped": drop}

	# ─────────────────────────────────────────────────────────
	# STEP 4d — ONE-WAY ELM → PFLOTRAN COUPLING
	# ─────────────────────────────────────────────────────────
