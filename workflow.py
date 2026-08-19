#!/usr/bin/env python3
"""
IDEAS workflow coordinator — the four agents of the framework, in order.

    User request
      → Reception          brief   (MCP tool loop: what/where/when)
      → Planner            plan    (sampling strategy + feasibility verdict)
      → Experiment Manager run     (materialize → build → prepare → run)
      → Analyzer           report  (metrics → validation → interpretation)

WHICH MODEL RUNS IS RECEPTION'S CHOICE. Reception asks each MCP server what
it is (`describe_*_capabilities`) and the brief names the model; _adopt_model
checks the name against core/model_servers.RUNNABLE, and
core/resumable._manager_for turns it into the Experiment Manager class that
lives beside that server (mcp/elm-mcp/src, mcp/pflotran-mcp). The run
directory is named for that model, which is why it is minted only once the
brief exists — see process_request. Wired 2026-08-18; until then every run
directory was elm_run_* and _execute named ELM's manager by hand.
"""
import os
import sys
import traceback
import json
from pathlib import Path
from typing  import Optional
sys.path.insert(0, "src")

# THE MANAGER FOR A MODEL NAME comes from core/resumable._manager_for, which
# puts the server's directory on the path and imports the class from beside
# the server. Both the fresh path (_execute) and the resume/finalize paths use
# it, so there is one place that knows where a manager lives.
from core.resumable import _manager_for


def _config_for(model: str, base: dict, period: dict = None,
				initialization: dict = None) -> dict:
	"""The manager's config for `model`: ELM's has settings of its own.

	The base config — brief, reception, strategy, mcp_clients — is what every
	manager takes. ELM adds a period in years and a warm start (see
	_elm_config); any other model takes the base plus the resolved period's
	years, which the base class prints on the columns it materialises.
	"""
	if model == "elm":
		return _elm_config(base, period=period, initialization=initialization)
	cfg = dict(base)
	if period and period.get("yr_start"):
		cfg["yr_start"] = int(period["yr_start"])
		cfg["yr_end"] = int(period.get("yr_end") or period["yr_start"])
	return cfg


def _elm_config(base: dict, period: dict = None,
				initialization: dict = None) -> dict:
	"""backends.config_for("elm", ...), with the other backends' branches gone.

	`base` holds what the manager always takes — brief, reception, strategy,
	mcp_clients — and is never modified in place.

	ELM KNOWLEDGE IN THE FRAMEWORK, and it should not stay here. Both settings
	below are facts about ELM: that a period is years, and that a single-column
	run wants a warm start. They sat in backends.py for the same bad reason —
	it was the file that already knew which model was which. The right home is
	the ELM server, beside build_elm_inputs_from_location, which is the thing
	that actually reads them.
	"""
	cfg = dict(base)

	# The period reception resolved. Recorded because it is what was ASKED
	# about, which is not always what was simulated.
	if period:
		if period.get("yr_start"):
			cfg["yr_start"] = int(period["yr_start"])
		cfg["yr_end"] = int(period.get("yr_end")
							or period.get("yr_start") or 1995)

	# WARM IS THE DEFAULT, cold is an explicit opt-out. A cold single-column
	# year starts from ELM's generic state and spends the run relaxing out of
	# it — measured on this framework, recharge came out -0.18 mm/yr cold
	# against 309 warm on the SAME column.
	if (initialization or {}).get("mode") != "cold":
		cfg["warm_start"] = True
	return cfg


class _NothingToReportOn(Exception):
	"""Not a failure: the run finished and produced nothing to interpret.

	Its own type so the skip reads as a skip. Folded into the generic handler
	it would print "written report failed", which describes a broken reporter
	rather than an empty ensemble.
	"""


def _save_reception(run_dir, result: dict) -> None:
	"""Persist the reception package the moment it exists.

	CALLED TWICE, AND THE SECOND CALLER IS THE POINT. The normal path writes
	this on the way into planning. The other is a run that STOPS at dispatch
	because reception named a model this framework cannot drive yet — three
	minutes of watershed, soil, rain and observation fetching, and the reason
	the run stopped is a missing wrapper rather than anything wrong with what
	was gathered. Throwing it away would make the honest answer the expensive
	one, and the next attempt would fetch all of it again.

	`trace` and `raw` are dropped in both: the tool trace is the conversation,
	not the package, and `raw` is the unparsed reply.
	"""
	run_dir = Path(run_dir)
	run_dir.mkdir(parents=True, exist_ok=True)
	(run_dir / "reception.json").write_text(
		json.dumps({k: v for k, v in result.items()
					if k not in ("trace", "raw")}, indent=2, default=str))
	# alias for the standalone tools that open it by this name
	(run_dir / "reception_brief.json").write_text(
		json.dumps(result.get("brief") or {}, indent=2, default=str))


def _drop_if_empty(d) -> None:
	"""Remove a run directory only if nothing was ever written into it.

	The directory is minted DURING reception (between its LLM phase and its
	gather, named for the model the brief chose), and a failure between the
	mint and the first write can leave it untouched — so it is removed again
	here. rmdir refuses a non-empty directory, which is the safety: this can
	never take a real run with it.
	"""
	if not d:
		return
	try:
		Path(d).rmdir()
	except OSError:
		pass


from agents.planner               import Planner
from core.mcp_manager             import MCPManager

class WorkflowCoordinator:
	"""Wires Reception → Planner → Experiment Manager → Analyzer."""

	# WHICH MODEL, as a class attribute so it has a value even on an instance
	# built without __init__ — the test harness does exactly that, and so does
	# anything that reconstructs a coordinator to inspect it. __init__ replaces
	# it with the validated name.
	model = "elm"

	def __init__(self,
				 reception_model:      str = "claude-opus-4-8-project",
				 planner_model:        str = "claude-opus-4-8-project",
				 # ACCEPTED AND UNUSED. The Analyzer is the five-step box in
				 # src/agents/analysis/, and each step picks its own model —
				 # step 2 and step 3 have different jobs and different budgets,
				 # so one name here could only ever be wrong for one of them.
				 # Kept in the signature because callers and tests pass it.
				 analyzer_model:       str = None,
				 default_output_dir:   str = "./workflow_outputs",
				 mcp_config_file:      str = "mcp_config.json",
				 interactive_reception: bool = False,
				 model:                str = None):

		# WHICH MODEL THIS SESSION RUNS. Validated here, at construction,
		# rather than at _execute — that is minutes of reception and planning
		# later, and a typo'd name should not cost an LLM call to discover.
		# THE DEFAULT BEFORE RECEPTION SPEAKS. Reception names the model that
		# will run; this is only what stands if it names none. Checked against
		# RUNNABLE here so a name that cannot run fails now rather than
		# minutes later at _execute.
		from core.model_servers import RUNNABLE
		self.model = (model or "elm").strip().lower()
		if self.model not in RUNNABLE:
			raise ValueError(
				f"unknown model {self.model!r} — runnable: "
				f"{', '.join(sorted(RUNNABLE))}")

		# ── MCP Manager ───────────────────────────────────────
		print("\n" + "=" * 70)
		print("Initializing MCP Tools")
		print("=" * 70)
		try:
			self.mcp_manager = MCPManager(mcp_config_file)
			mcp_clients      = self.mcp_manager.get_all_clients()
			if mcp_clients:
				print(f"✓ Loaded {len(mcp_clients)} MCP server(s)")
				for name in mcp_clients:
					print(f"  - {name}")
				# The servers' own INFO logging would otherwise interleave with
				# the reception trace between every prompt. Say where it went.
				from core.mcp_client import mcp_errlog
				_log = getattr(mcp_errlog(), "name", "<stderr>")
				if _log != "<stderr>":
					print(f"  server logs \u2192 {_log}  "
					      f"(IDEAS_MCP_LOG=stderr to show them here)")
			else:
				print("⚠️  No MCP servers configured")
		except Exception as e:
			print(f"⚠️  MCP initialization failed: {e}")
			mcp_clients = {}
		print("=" * 70 + "\n")
	
		# Keep the live clients: the ELM manager's materialize stage needs
		# terrain/fan_wtd/geology to turn a sampling strategy into columns.
		self.mcp_clients = mcp_clients

		# ── Agents ────────────────────────────────────────────
		# Reception is an LLM-driven tool loop over all MCP servers. It emits
		# `domain: {name, huc, bbox}`, which the Experiment Manager's
		# materialize stage needs to turn a strategy into columns.
		from agents.reception_llm import LLMReceptionAgent
		self.reception = LLMReceptionAgent(
			model       = reception_model,
			mcp_clients = mcp_clients,
			interactive = interactive_reception,
		)
		# NO PINNING RULES YET, AND THAT IS THE POINT. This used to ask here,
		# with a comment claiming "this is where the backend is known" — it was
		# not. Construction happens before the request is read, so the only
		# model available is the default, and the answer was ELM's whatever
		# reception went on to choose. It also spent an MCP call on every
		# start, including the runs that end in a clarifying question.
		# The rules are read in the design path instead, once reception has
		# named a model. Not fatal if they are missing: the sampler asks the
		# same server for the same answer and refuses to place a column
		# without it, so a plan made without them cannot quietly reach compute.
		self.planner = Planner(model=planner_model, pinning=None)

		# NO ANALYZER OBJECT. The Analyzer is not an agent this class holds; it
		# is a box that runs over a finished run directory, constructed where it
		# is used (execute_plan stage 4c, and _workflow_analyze_existing).

		# ── Defaults ──────────────────────────────────────────
		# ABSOLUTE, ALWAYS (2026-08-18). The run directory is minted under this
		# and every path the coordinator hands a model server — the run dir, a
		# deck's input file — is derived from it. The servers are separate
		# processes with their own working directories: the CLI default
		# "./workflow_outputs" reached the PFLOTRAN server as a relative deck
		# path and PFLOTRAN answered "File not found" on all 17 columns of a
		# Gunnison study, while every driver that had passed an absolute path
		# ran clean. Resolved once, here, so no caller has to remember.
		self.default_output_dir   = str(Path(default_output_dir).resolve())
	
		# ── Conversation state ────────────────────────────────
		self.conversation_context = {
			'last_run_dir':  None,
			'last_plan':     None,
			'last_analysis': None,
			'last_focus':    None,
		}
	
	def _adopt_model(self, brief: dict) -> None:
		"""Take the model from the brief, or say why the default stands.

		RECEPTION CHOOSES, THIS ONE CHECKS. The brief names a model that a
		server described; whether this framework can drive a STUDY with it is
		a different question, answered by core/model_servers.RUNNABLE — ELM
		and, since 2026-08-18, PFLOTRAN.

		Refuses rather than falling back silently. A study that ran ELM
		because PFLOTRAN was unavailable, and said so nowhere, is a wrong
		answer wearing a completed run directory.

		THIS IS THE ONLY PLACE THE LIMIT IS ENFORCED (2026-08-17). It used to
		be enforced twice: here, and again in reception's own prompt, which
		told the model to pick something runnable. Two enforcements of one rule
		is one too many, and the prompt's copy was the harmful one — it made
		reception report a forced choice as a scientific one. Reception now
		names the right instrument and says separately that it cannot be
		driven; the refusal below is what stops the run, and its caller turns
		it into a message rather than a traceback.
		"""
		from core.model_servers import RUNNABLE
		want = str((brief or {}).get("model") or "").strip().lower()
		if not want:
			print(f"   model: {self.model} (reception named none)")
			return
		if want == self.model:
			print(f"   model: {want} (reception)")
			return
		if want not in RUNNABLE:
			raise RuntimeError(
				f"reception chose {want!r}, and its server does describe "
				f"itself — but this framework cannot run a study with it "
				f"yet. Nothing turns a sampling strategy into {want} decks, "
				f"writes its experiment.json, or names what its numbers "
				f"mean. Runnable today: {', '.join(sorted(RUNNABLE))}.")
		# The run directory is already named for `want`: process_request
		# mints it from the brief, between reception's LLM phase and its
		# gather, so nothing needs renaming here.
		print(f"   model: {want} (reception, was {self.model})")
		self.model = want

	def _pinning_block(self, mcp_clients: dict, model: str) -> Optional[dict]:
		"""What a column of `model` may be compared against, asked of ITS server.

		ASKED OF THE SERVER BY NAME, NOT VIA A MANAGER CLASS (2026-08-17). It
		used to go through the manager, which carried MCP_NAME and a
		CAPABILITIES_TOOL — and since the only manager left was ELM's, every
		study was handed ELM's pinning rules under a heading that told the
		planner they came from "the model server that will run this study".
		For a PFLOTRAN study those rules are not merely stale, they are
		INVERTED: ELM offers swe and et and withholds water_table; PFLOTRAN
		computes no snow and no ET and offers water_table alone. A planner
		reading the wrong block spends its validation columns on snow pillows
		for a model with no snow, and every downstream check agrees with it.

		Routing through the manager also tied this to a study wrapper. Whether
		a column may be pinned to a well is a property of the MODEL; whether
		this framework can drive a study is a property of this repository.
		PFLOTRAN can answer the first today and cannot do the second, and a
		manager class conflates them — which is why the name goes straight to
		describe_server, the same call reception uses to choose.

		THIS IS THE ONLY FETCH (2026-08-18). The block is written into
		strategy.json beside the plan, and the sampler reads it from there
		rather than asking the server again — so the sample is checked against
		exactly the rules the design was made under.

		Returns None and says so rather than raising: the planner without the
		block writes a plan, strategy.json is written without a `pinning` key,
		and the sampler refuses to place a column until one is there. Failing
		at the later point is better than failing at the earlier one, because
		the later point is where a wrong answer would start costing compute.
		"""
		try:
			block = (self._model_report(mcp_clients, model) or {}).get("pinning")
			if not block:
				raise RuntimeError("its report carries no `pinning` block")
			names = [e.get("variable") for e in (block.get("pinnable") or [])]
			print(f"✓ pinning rules from {model}: {', '.join(names) or '(none)'}")
			return block
		except Exception as e:                                  # noqa: BLE001
			print(f"⚠️  could not read the pinning rules from the {model} "
				  f"server ({e}) — the planner will not be told what it may "
				  f"pin to, and the sampler will refuse to place a pinned "
				  f"column until it can ask.")
			return None

	def _model_report(self, mcp_clients: dict, model: str) -> Optional[dict]:
		"""The chosen model server's capability report, fetched once per run.

		MEMOIZED because two blocks are read out of it — `pinning` and
		`constraints` — and the report costs a real MCP round trip (2.5 s for
		ELM, measured). Asking twice for one answer is how the pinning call
		came to run at construction AND again at plan time.
		"""
		cached = getattr(self, "_report_cache", None)
		if cached and cached[0] == model:
			return cached[1]
		from core.model_servers import describe_server
		got = describe_server(mcp_clients or {}, model)
		if got.get("error"):
			raise RuntimeError(got["error"])
		if got.get("describes_itself") is False:
			raise RuntimeError(f"{model} exposes no describe tool")
		rep = got.get("report") or {}
		self._report_cache = (model, rep)
		return rep

	def _constraints_block(self, mcp_clients: dict,
						   model: str) -> Optional[dict]:
		"""What a study using `model` may NOT vary — the design fence.

		THE OTHER HALF OF THE SAME MOVE AS _pinning_block. These limits used to
		be four hardcoded bullets in planner.txt, stated as facts about "the
		framework": NLDAS-2 only, soil from the CONUS donor gridcell, warm
		start by default, compset fixed. All four are facts about ELM. A
		PFLOTRAN study reading them is told its soil is unchoosable while its
		SSURGO profiles sit in the reception package, and is downgraded for
		limits it does not have.

		Returns None rather than raising, and the prompt says what to do with
		that: judge feasibility on what was asked and the observations, and say
		the fence could not be read. A missing fence must not silently become
		an absent one — that would make every study come back `full`.
		"""
		try:
			block = (self._model_report(mcp_clients, model) or {}).get(
				"constraints")
			if not block:
				raise RuntimeError("its report carries no `constraints` block")
			off = [k for k in ("forcing", "soil", "initial_state")
				   if (block.get(k) or {}).get("available") is False]
			print(f"✓ design limits from {model}"
				  + (f" — UNAVAILABLE HERE: {', '.join(off)}" if off else ""))
			return block
		except Exception as e:                                  # noqa: BLE001
			print(f"⚠️  could not read the design limits from the {model} "
				  f"server ({e}) — the planner will judge feasibility without "
				  f"knowing what this model cannot vary, so a study it calls "
				  f"feasible may not be.")
			return None

	# ═════════════════════════════════════════════════════════
	# MAIN ENTRY POINT
	# ═════════════════════════════════════════════════════════
	def _plan(self, result: dict, run_dir) -> dict:
		"""reception.json -> strategy.json, with the rulebook it was made under.

		ONE METHOD, so the step can be driven on its own — against an archived
		reception, without the execute stage behind it — and so what it writes
		is what a test reads. It was inlined in _workflow_design_and_run until
		2026-08-18, which meant the only way to see strategy.json produced was
		to run a study.

		WHAT MAY BE PINNED TO is asked of the model reception just chose — by
		name, so this is the chosen server's answer rather than the only manager
		the framework happens to have. `self.model` is what _adopt_model settled
		on, so the two cannot disagree. AND WHAT IT MAY NOT VARY comes from the
		same report: one round trip, _model_report memoizes between the two.

		THE RULEBOOK TRAVELS WITH THE DESIGN. The pinning block was fetched for
		the planner's prompt and used to be discarded after it; the sampler
		then asked the server the same question a second time, and two fetches
		of one rule can disagree — a server updated between plan and run, a
		resume a week later, a manager naming a different server than the
		brief. Written here, beside the plan, it is what the sampler enforces:
		the exact block this design was made under, the same object, read off
		the same file. Written by this code and not by the planner, because an
		LLM restating a rule it was given is a second copy of it.
		"""
		self.planner.pinning = self._pinning_block(self.mcp_clients, self.model)
		self.planner.constraints = self._constraints_block(
			self.mcp_clients, self.model)
		plan = self.planner.plan(result)
		if self.planner.pinning:
			plan["pinning"] = self.planner.pinning
		(Path(run_dir) / "strategy.json").write_text(
			json.dumps(plan, indent=2, default=str))
		return plan

	def _reception_context(self) -> Optional[dict]:
		"""What reception should know about earlier turns — and nothing more.

		The context is JSON-dumped verbatim into reception's prompt, so this
		is a token budget, not just a convenience. conversation_context holds
		last_analysis and last_plan in FULL; sending those would put an entire
		analysis report and strategy into every subsequent request.

		What reception actually needs is enough to resolve a follow-up: which
		run to look at for "analyze that again", and what was being studied so
		"the same basin" resolves. Returns None on the first turn so no
		CONVERSATION CONTEXT block is emitted at all.
		"""
		cc = self.conversation_context or {}
		ctx = {}
		if cc.get('last_run_dir'):
			ctx['prior_run_dir'] = str(cc['last_run_dir'])
		if cc.get('last_focus'):
			ctx['prior_focus'] = cc['last_focus']
		plan = cc.get('last_plan') or {}
		if isinstance(plan, dict):
			samp = plan.get('sampling') or {}
			if samp.get('n_columns'):
				ctx['prior_n_columns'] = samp['n_columns']
			if plan.get('archetype'):
				ctx['prior_archetype'] = plan['archetype']

		# WHAT COULD BE RESUMED — the list reception is allowed to pick from,
		# and nothing else. Discovery is a filesystem scan plus one squeue, so
		# it is deterministic and cheap; the model's job is to choose, not to
		# work out whether a job has finished.
		#
		# The REQUEST TEXT is what makes the list usable: run dirs are named by
		# timestamp, so "resume my Gunnison run" matches no directory name at
		# all. Truncated because this goes into every prompt.
		try:
			from core.resumable import find_resumable
			rows = find_resumable(self.default_output_dir, check_jobs=True)
			if rows:
				ctx['resumable_runs'] = [{
					'run_dir': r['run_dir'],
					'model':   r.get('model'),
					'stage':   r.get('stage'),
					'status':  r.get('why'),
					'age':     r.get('age'),
					'request': (r.get('request') or '')[:160],
				} for r in rows[:10]]
		except Exception as e:                                  # noqa: BLE001
			# Never fatal: a broken scan must not stop someone asking a
			# question that has nothing to do with resuming.
			print(f"   ⚠️  could not scan for resumable runs ({e})")

		return ctx or None

	def process_request(self,
						user_request: str,
						output_dir:   Optional[str] = None
						) -> str:
		print("\n" + "=" * 70)
		print("COORDINATOR")
		print("=" * 70)
		print(f"Request: {user_request[:80]}...")
		print("=" * 70 + "\n")
	
		# THE RUN DIRECTORY IS MADE DURING RECEPTION — after its LLM phase,
		# before its gather (2026-08-18). Reception writes things that are not
		# JSON — the modelled water table as a GeoTIFF, the CONUS2 subsurface
		# arrays — and they have to land BESIDE reception.json, so the directory
		# must exist before the gather. But it is NAMED FOR THE MODEL THAT
		# RUNS, and only the brief knows that; minted before reception, every
		# directory was elm_run_* whatever ran. So the coordinator hands
		# reception a function that mints the directory from the brief, and
		# reception calls it exactly between its two phases, on the design
		# route only. The coordinator still owns the directory — it is this
		# function that makes it — and a clarification mints nothing.
		#
		# THE NAME MUST BE UNIQUE, and a one-second timestamp is not.
		#
		# Two studies launched in the same second got the SAME directory —
		# `exist_ok=True` adopts whatever is already there — and then overwrote
		# each other's reception.json, strategy.json and columns.json while both
		# were running. One died in materialize against the other's columns and
		# the survivor's directory held a mixture of the two. Neither failure
		# named the collision: what surfaced was a warm start with no donors for
		# columns the OTHER study had sampled.
		#
		# mkdir(exist_ok=False) in a loop, so the FILESYSTEM decides who wins
		# rather than a check-then-create that races just as badly.
		from datetime import datetime as _dt
		base = Path(output_dir or self.default_output_dir).resolve()
		minted: dict = {}

		def _mint(brief: dict) -> Path:
			# The model reception named, else the default that stands when it
			# names none (_adopt_model prints which). Not yet checked against
			# RUNNABLE: an unrunnable choice still gets its package saved,
			# under its own name, so nothing gathered is lost or mislabelled.
			name = str((brief or {}).get("model") or self.model).strip().lower()
			stamp = f"{_dt.now():%Y%m%d_%H%M%S}"
			for suffix in ("", *(f"_{i}" for i in range(2, 100))):
				d = base / f"{name}_run_{stamp}{suffix}"
				try:
					d.mkdir(parents=True, exist_ok=False)
					minted["dir"] = d
					return d
				except FileExistsError:
					continue
			raise RuntimeError(
				f"could not mint a run directory under {base} — 99 names "
				f"already taken for {stamp}")

		# Step 1 — Reception
		#
		# ON ANY FAILURE THE EMPTY DIRECTORY GOES WITH IT. Reception can raise
		# after minting and before writing anything — an unreachable server.
		# Without this every failed request left a dated, empty run directory
		# behind, and those accumulate in exactly the place someone looks for
		# real runs. Before the mint there is nothing to drop.
		try:
			result = self.reception.process(
				user_request = user_request,
				context      = self._reception_context(),
				run_dir      = _mint,
			)
		except BaseException:
			if minted.get("dir"):
				_drop_if_empty(minted["dir"])
			raise
		run_dir = minted.get("dir")
		# Reception returns the whole package now: route (dispatch), brief
		# (science), observations + grid (what was fetched), provenance.
		# The adapter that used to flatten this into a dataclass is gone — it
		# forwarded only `brief`, which is how observations, grid and
		# provenance were being dropped before anything downstream saw them.
		result["user_request"] = user_request
		action = (result.get("route") or {}).get("action", "clarify")
		print(f"🧠 Route: {action}\n")

		# ── WHICH MODEL, from the brief ──────────────────────────────
		# Reception chose it against what the servers said they are (STEP 2a),
		# so this is the first point in the run where the model is known. It
		# used to be fixed by a --model flag before the request was read.
		#
		# A REFUSAL HERE IS A NORMAL OUTCOME, NOT A CRASH (2026-08-17). Reception
		# is now told to name the RIGHT instrument even when this framework
		# cannot drive it, so "the model that fits cannot be run yet" is an
		# answer the pipeline is expected to produce. It used to reach the user
		# as an unhandled traceback with the reception package unwritten. Now
		# the package is saved first and the reason is stated in full: nothing
		# has been spent on compute, and everything gathered is on disk.
		if action == "design":
			try:
				self._adopt_model(result.get("brief") or {})
			except RuntimeError as e:
				_save_reception(run_dir, result)
				brief = result.get("brief") or {}
				return (
					f"⏹  STOPPED AT DISPATCH — nothing was run, nothing was "
					f"queued.\n\n{e}\n\n"
					f"WHY THIS MODEL: {brief.get('model_rationale') or '(none given)'}\n\n"
					f"WHAT RECEPTION FOUND: {brief.get('model_availability') or '(none given)'}\n\n"
					f"The reception package is saved and complete — the "
					f"watershed, the grid, the soil, the rain and the "
					f"observations are all in\n"
					f"    {Path(run_dir) / 'reception.json'}\n"
					f"Nothing in it needs fetching again. There is no button "
					f"that picks it up yet, though: --resume reads "
					f"run_state.json and strategy.json, which are written "
					f"AFTER planning, and this run stopped before that. Until "
					f"{brief.get('model')!r} has a study wrapper, driving the "
					f"planner against this file is a manual step.")

		# Only the design route uses the directory made above. Reception returns
		# before it gathers anything on the other three, so no directory was
		# minted at all (the mint runs on the design route only) — but an
		# older reception could still hand one back, so the drop stays.
		if action != 'design' and run_dir:
			_drop_if_empty(run_dir)

		# Step 2 — Route
		if action == 'clarify':
			return self._workflow_clarification(result)
		elif action == 'analyze_existing':
			return self._workflow_analyze_existing(result)
		elif action == 'resume':
			return self._workflow_resume(result)
		elif action == 'design':
			return self._workflow_design_and_run(
				result     = result,
				output_dir = str(Path(output_dir or self.default_output_dir).resolve()),
				run_dir    = run_dir,
			)
		else:
			return f"❌ Unknown route: {action}"
	
	# ═════════════════════════════════════════════════════════
	# INTERACTIVE MODE
	# ═════════════════════════════════════════════════════════
	def run_interactive(self):
		print("\n" + "=" * 70)
		print("COORDINATOR - INTERACTIVE MODE")
		print("=" * 70)
		print("\nCommands:")
		print("  - Type your request naturally")
		print("  - 'quit' or 'exit' → stop")
		print("  - 'status'         → show context")
		print("  - 'clear'          → clear history")
		print("=" * 70 + "\n")
	
		while True:
			try:
				user_input = input("You: ").strip()
				if not user_input:
					continue
				if user_input.lower() in ['quit', 'exit', 'q']:
					print("\n👋 Goodbye!\n")
					break
				if user_input.lower() == 'status':
					self._print_status()
					continue
				if user_input.lower() == 'clear':
					self._clear_context()
					print("✓ Conversation history cleared\n")
					continue
				response = self.process_request(user_input)
				print(f"\n{response}\n")
			except KeyboardInterrupt:
				print("\n\n👋 Interrupted. Goodbye!\n")
				break
			except Exception as e:
				print(f"\n❌ Error: {e}\n")
				traceback.print_exc()
	
	# ═════════════════════════════════════════════════════════
	# WORKFLOW 1 — CLARIFICATION
	# ═════════════════════════════════════════════════════════
	def _workflow_clarification(self, result) -> str:
		print("💬 WORKFLOW: Clarification Needed\n")
		lines = ["I need some clarification:\n"]
		for i, q in enumerate((result.get('route') or {}).get('questions') or [], 1):
			lines.append(f"{i}. {q}")
		return "\n".join(lines)
	
	# ═════════════════════════════════════════════════════════
	# WORKFLOW 2 — ANALYZE EXISTING
	# ═════════════════════════════════════════════════════════
	def _workflow_analyze_existing(self, result) -> str:
		print("📊 WORKFLOW: Analyze Existing Results\n")
		run_dir = (
			(result.get('route') or {}).get('prior_run_dir') or
			self.conversation_context.get('last_run_dir')
		)
		if not run_dir or not Path(run_dir).exists():
			return ("❌ Please specify a valid run directory or "
					"ensure a previous run exists.")
	
		print(f"📂 Analyzing: {run_dir}\n")
		# THE ANALYZER BOX, over a run directory. This used to require
		# LLM_ANALYSIS_INPUT.json and hand it to a one-shot report agent, which
		# meant "analyze that run" failed on any archived study that predated
		# the alias or never wrote it. The Analyzer reads the packaged run off
		# disk, so a directory is the only thing it needs.
		try:
			from agents.analyzer import Analyzer
			status = Analyzer(str(run_dir)).run() or {}
			if status.get("error"):
				return f"❌ Analysis failed: {status['error']}"
			report = json.loads(
				(Path(run_dir) / "04_analysis" / "analysis.json").read_text())
			self.conversation_context['last_analysis'] = report
			self.conversation_context['last_run_dir']  = run_dir
			return self._format_analysis_response(report, run_dir)
		except Exception as e:
			return f"❌ Analysis failed: {e}"
	
	# ═════════════════════════════════════════════════════════
	# WORKFLOW 3 — DESIGN & RUN
	# ═════════════════════════════════════════════════════════
	def _workflow_design_and_run(self, result, output_dir: str, run_dir) -> str:
		print("🚀 WORKFLOW: Design & Run\n")
		try:
			# The COORDINATOR owns the run directory, and each stage's file is
			# written when that stage finishes. Reception and the planner then
			# only ever produce, and the Experiment Manager only ever reads —
			# which also means a failure downstream leaves both intact.
			#
			# IT IS MADE IN process_request NOW, not here, because reception
			# writes the modelled water table into it as a GeoTIFF and needs
			# somewhere to put it. It is still NAMED FOR THE MODEL THAT RAN:
			# every run directory used to be elm_run_*, which was accurate while
			# ELM was the only backend and becomes a mislabel the moment it is
			# not — archived PFLOTRAN studies would all claim to be ELM.
			run_dir = Path(run_dir)
			_save_reception(run_dir, result)

			# Step 1 — Plan
			print("📋 STEP 1: Planning Experiments")
			print("-" * 50)
			plan = self._plan(result, run_dir)
			# The capability-aware planner emits a STRATEGY, never
			# CONDITIONS_COUPLERS — those are materialized against real data in
			# the Experiment Manager's step 0. Counting them here printed
			# "0 experiments" on every successful plan, which reads as a failure.
			strat = plan.get('sampling') or {}
			n_req = strat.get('n_columns')
			verdict = (plan.get('feasibility') or {}).get('verdict', '?')
			print(f"✓ Strategy: {strat.get('approach', 'n/a')}, "
				  f"N={n_req if n_req is not None else 'expander decides'}, "
				  f"feasibility={verdict}")
			print("  (columns are materialized from real data in step 0)\n")
	
			self.conversation_context['last_plan']  = plan
			self.conversation_context['last_focus'] = (
				((result.get('brief') or {}).get('scientific_framing') or {}).get('goals', [None])[0]
			)
	
			# Step 2 — Execute
			print("⚙️  STEP 2: Executing Experiments")
			print("-" * 50)
			run_summary = self._execute(
				plan         = plan,
				output_dir   = output_dir,
				run_dir      = run_dir,
				reception    = result,
				brief        = result.get('brief') or {},
				period       = ((result.get('brief') or {}).get('run_settings') or {}).get('resolved_period'),
				initialization = ((result.get('brief') or {}).get('run_settings') or {}).get('initialization'),
			)
			# QUEUED IS NOT FAILED, AND THIS LINE SAID IT WAS. A detached run
			# reaches here about three seconds after submitting, when nothing
			# has executed — and "✓ Execution: 0/19 succeeded" reads as an
			# ensemble that ran and produced nothing. It is the same failure
			# the data path keeps having: NOT YET rendering identically to
			# NOTHING. The manager already knows the difference; it sets
			# status="pending" and records both job ids.
			submitted = run_summary.get('status') == 'pending'
			if submitted:
				print(f"⏳ Submitted: "
					  f"{run_summary.get('experiments_pending') or run_summary['experiments_total']}"
					  f" column(s) queued — nothing has run yet\n")
			else:
				print(f"✓ Execution: "
					  f"{run_summary['experiments_success']}/"
					  f"{run_summary['experiments_total']} "
					  f"succeeded\n")
	
			self.conversation_context['last_run_dir'] = (
				run_summary['run_directory']
			)
	
			# NOTHING TO ANALYSE YET IS NOT NOTHING TO ANALYSE. On a detached
			# run the analysis is job B's, chained behind the ensemble with
			# --dependency=afterany. Printing "STEP 3: Analyzing Results"
			# followed by "report skipped — 0/19 columns produced output" told
			# the user their study had failed, minutes before it succeeded.
			if submitted:
				jid_a = run_summary.get('job_id_a')
				jid_b = run_summary.get('job_id_b') or run_summary.get('job_id')
				print("📊 STEP 3: Analysis — deferred to the scheduler")
				print("-" * 50)
				if jid_b and jid_b != jid_a:
					print(f"   job {jid_a or '?'} runs the ensemble; job {jid_b} "
						  f"analyses it when that finishes.")
				else:
					print(f"   job {jid_b or jid_a or '?'} is queued.")
				print(f"   You can close this terminal.\n")
				return (
					f"⏳ Submitted — the run is queued, not finished.\n"
					f"   {run_summary['experiments_total']} column(s) · "
					f"{run_summary['run_directory']}\n\n"
					f"   watch    squeue -u $USER\n"
					f"            tail -f {run_summary['run_directory']}"
					f"/ensemble_A.log\n"
					f"   answer   {run_summary['run_directory']}"
					f"/04_analysis/analysis.json  (written by job "
					f"{run_summary.get('job_id_b') or run_summary.get('job_id')})\n\n"
					f"   If the analysis job never runs:\n"
					f"            {run_summary.get('resume_command')}")

			# Step 3 — Analyze
			print("📊 STEP 3: Analyzing Results")
			print("-" * 50)
			# THE ANALYSIS ALREADY RAN. execute_plan's stage 4c runs the
			# Analyzer over this directory and writes 04_analysis/analysis.json.
			# This step used to spend TWO MORE LLM calls on a second, separate
			# interpreter that never saw the comparison, the caveats or the
			# figures — and its answer, not the pipeline's, was what got printed.
			# The worse of two analyses was the one the user read. Deleted
			# 2026-08-13; this step now reads what 4c wrote.
			#
			# STILL NON-FATAL, for the original reason: the ensemble and
			# experiment.json exist by the time we get here, so a missing or
			# half-written report must not be reported as a failed pipeline.
			run_directory = Path(run_summary['run_directory'])
			report_path = run_directory / "04_analysis" / "analysis.json"
			analysis = None
			try:
				# 4c skips itself on a dead ensemble rather than paying an LLM
				# to conclude it has no data, so the file is legitimately absent
				# here. RUN_SUMMARY.json already says that for free.
				if not run_summary.get('experiments_success'):
					raise _NothingToReportOn(
						f"0/{run_summary['experiments_total']} columns produced "
						f"output")
				if not report_path.exists():
					raise FileNotFoundError(
						f"04_analysis/analysis.json was not written; "
						f"experiment.json holds the results")
				analysis = json.loads(report_path.read_text())
				print("✓ Analysis complete\n")
				self.conversation_context['last_analysis'] = analysis
			except _NothingToReportOn as e:
				print(f"   ⏭  report skipped — {e}\n")
			except Exception as e:                              # noqa: BLE001
				print(f"   ⚠️  written report unavailable ({e}) — the run and "
					  f"its results are intact\n")

			if analysis is None:
				return (f"✅ Run complete: {run_summary['experiments_success']}"
						f"/{run_summary['experiments_total']} columns → "
						f"{run_summary['run_directory']}\n"
						f"   (the written report was not produced; "
						f"experiment.json holds the results)")
			return self._format_full_pipeline_response(
				run_summary, analysis
			)
	
		except Exception as e:
			return (f"❌ Pipeline failed: {e}\n\n"
					f"{traceback.format_exc()}")
	
	def _workflow_resume(self, result) -> str:
		"""Reception asked to continue a run. It may CHOOSE; it may not INVENT.

		Deterministic discovery, LLM routing. The scan finds what is resumable
		— a mechanical question that squeue answers exactly — and reception's
		only job is to pick one of those, because "which of my runs did you
		mean" is genuinely conversational and "is job 770696 finished" is not.

		The allowlist below is the whole point of the split. A run directory
		reception produced rather than selected would either crash or, worse,
		resume a different study and report it as the one that was asked about.
		Run dirs are named by timestamp, so two studies of the same watershed
		an hour apart are indistinguishable to a model working from the name.

		Same discipline as the strategy gate and the 2-tool allowlist: the
		model chooses among real options rather than manufacturing one.
		"""
		from core.resumable import find_resumable, describe

		rows = find_resumable(self.default_output_dir, check_jobs=True)
		if not rows:
			return ("Nothing is waiting to be resumed. "
					"`python workflow.py --resume` lists every run and says "
					"why each one cannot be continued.")

		asked = (result.get("route") or {}).get("prior_run_dir")
		allowed = {r["run_dir"]: r for r in rows}
		chosen = allowed.get(asked) if asked else None

		if asked and chosen is None:
			# Reception named something that is not on the list. Say so rather
			# than guess: this is the failure the allowlist exists to catch.
			print(f"   ⚠️  reception named a run that is not resumable: {asked}")

		if chosen is None:
			if len(rows) == 1:
				chosen = rows[0]
				print(f"   only one run is resumable — {chosen['name']}")
			else:
				listing = "\n".join(f"  [{i}] {describe(r)}"
									for i, r in enumerate(rows, 1))
				return (f"{len(rows)} runs could be continued. Which one?\n\n"
						f"{listing}\n"
						f"Or run it directly:\n"
						f"    python workflow.py --resume <run_dir>")

		print(f"\n{describe(chosen)}")
		return self.resume_run(chosen["run_dir"])

	# ═════════════════════════════════════════════════════════
	# RESUME — continue a run that was interrupted or submitted
	# ═════════════════════════════════════════════════════════
	def finalize_run(self, run_dir: str) -> str:
		"""Run the pipeline tail from inside the batch job that produced the output.

		Called as the last line of the combined study job, after the cases are
		built and the columns have stopped. It differs from resume_run in one
		respect, and that respect is the whole reason it exists: it must NOT
		poll the scheduler. The job that produced this output is the job
		running this code, so `squeue` reports it RUNNING, `_poll` returns
		None, and the run stops one line short of the analysis it was
		submitted to produce.

		So the finished stages are adopted from the filesystem first — the
		output exists or it does not — and the ordinary resume then carries on
		from extract. Everything after that is the same code that runs on a
		login node; the only thing that changed is which machine it runs on.
		"""
		from core.resumable import inspect_run

		rd = Path(run_dir).resolve()      # absolute: the servers have their own cwd
		if not rd.is_dir():
			return f"❌ No such run directory: {rd}"

		rec = inspect_run(rd, check_jobs=False)
		model = rec.get("model") or self.model
		Manager = _manager_for(model)
		if Manager is None:
			return (f"❌ The run state says this run used model '{model}', which "
					f"this build has no Experiment Manager for")
		self.model = model

		print("\n" + "=" * 70)
		print(f"FINALIZING IN PLACE — {rd.name}")
		print("=" * 70)
		mgr = Manager(base_output_dir=str(rd.parent), run_dir=str(rd))
		try:
			adopted = mgr.adopt_completed_run()
		except Exception as e:                                  # noqa: BLE001
			return (f"❌ could not adopt the finished stages ({e}) — the "
					f"output is on disk; finish with "
					f"`python workflow.py --resume {rd}`")
		for stage, n in (adopted or {}).items():
			print(f"   ✓ adopted {stage} from disk ({n})")
		if not adopted:
			print("   ⚠️  nothing adopted — resuming will poll the scheduler")

		# From here it is the ordinary resume, which now finds run done and
		# starts at extract. Kept as a delegation rather than a copy so the
		# tail cannot drift between the two entry points.
		return self.resume_run(str(rd))

	def resume_run(self, run_dir: str) -> str:
		"""Re-enter an existing run directory and carry on from its run state.

		Everything needed is already ON DISK, written by the stage that
		produced it: run_state.json says what is done and which model ran,
		reception.json and strategy.json are the inputs, run_plan.json is the
		materialized plan. Nothing is reconstructed from conversation, which is
		what makes this work across sessions — and across machines.

		THE MODEL COMES FROM THE LEDGER, not from --model. Resuming an ELM run
		with the default backend would build PFLOTRAN decks in an ELM
		directory; the run itself is the authority on what it is.
		"""
		# _read_json rather than json.loads: these files were written by an
		# earlier session and a half-written one must degrade to "no reception
		# package", not take the resume down.
		from core.resumable import inspect_run, describe, _read_json

		rd = Path(run_dir).resolve()      # absolute: the servers have their own cwd
		if not rd.is_dir():
			return f"❌ No such run directory: {rd}"

		rec = inspect_run(rd, check_jobs=True)
		print("\n" + "=" * 70)
		print("RESUMING")
		print("=" * 70)
		print(describe(rec))

		if not rec["resumable"]:
			return (f"❌ {rd.name} cannot be resumed: {rec['why']}\n"
					f"   Run `python workflow.py --resume` to see what can.")

		model = rec.get("model") or self.model
		if _manager_for(model) is None:
			return (f"❌ The run state says this run used model '{model}', which "
					f"this build has no Experiment Manager for")
		if model != self.model:
			print(f"   model: {model} (from the run state, not --model)")
			self.model = model

		reception = _read_json(rd / "reception.json") or {}
		strategy  = _read_json(rd / "strategy.json")  or {}
		brief     = reception.get("brief") or {}
		settings  = brief.get("run_settings") or {}

		try:
			summary = self._execute(
				plan           = strategy,
				output_dir     = str(rd.parent),
				run_dir        = rd,
				reception      = reception,
				brief          = brief,
				period         = settings.get("resolved_period"),
				initialization = settings.get("initialization"),
				resume         = True,
			)
		except Exception as e:                                  # noqa: BLE001
			return f"❌ Resume failed: {e}\n\n{traceback.format_exc()}"

		self.conversation_context['last_run_dir'] = summary['run_directory']

		# Still queued is a NORMAL outcome, not a failure — that is the whole
		# point of the pending state, and it must not read as a broken run.
		if summary.get("status") == "pending":
			return (f"⏳ Job {summary.get('job_id')} is still running — "
					f"{summary['experiments_pending']} column(s) queued.\n"
					f"   Check again: {summary['resume_command']}")
		return (f"✅ Resumed: {summary['experiments_success']}"
				f"/{summary['experiments_total']} columns → "
				f"{summary['run_directory']}")

	def _execute(self,
				 plan:           dict,
				 output_dir:     str,
				 run_dir:        Path,
				 reception:      dict,
				 brief:          dict = None,
				 period:         dict = None,
				 initialization: dict = None,
				 resume:         bool = False) -> dict:
		"""Hand the plan to the Experiment Manager.

		run_dir and reception are PASSED IN, not rediscovered. The coordinator
		created the directory and holds the reception package; reaching for
		either as a free variable is how this method came to reference two
		names that only existed in its caller.
		"""
		# WHICH MODEL: the one _adopt_model settled from the brief. Its manager
		# lives beside its server and is imported by path; the base's
		# execute_plan reads the stage declarations (NEEDS_CASE_BUILD,
		# NEEDS_SCHEDULER) off the class.
		Manager = _manager_for(self.model)
		if Manager is None:
			raise RuntimeError(
				f"no Experiment Manager for model {self.model!r} — "
				f"core/resumable._manager_for names the ones that exist")
		executor = Manager(base_output_dir=output_dir, run_dir=str(run_dir))
		# brief + mcp_clients feed the manager's materialize stage, which turns
		# the planner's sampling_strategy into the backend's own run plan.
		cfg = _config_for(
			self.model,
			{
				'brief':       brief or {},
				'reception':   reception,
				'strategy':    plan,
				'mcp_clients': self.mcp_clients,
				# Warm start edits a completed run's restart files, so the
				# carrier is whatever this session ran last. That makes "now
				# warm-start it" work as a plain follow-up, with no paths for
				# the user to supply.
				'last_run_dir': self.conversation_context.get('last_run_dir'),
			},
			period=period,
			initialization=initialization)
		# Opt-in, and only ever set by resume_run. A fresh run must not inherit
		# it: re-entering a directory usually means "do it again", and guessing
		# wrong that way silently reuses stale compute.
		if resume:
			cfg['resume'] = True
		# ASKING IS THE COORDINATOR'S JOB. It already owns every other question
		# put to the user, and the TTY guard that stops those from hanging a
		# scripted run. A backend that prompted would be a compute stage
		# holding the terminal open.
		email = self._ask_notify_email(Manager)
		if email:
			cfg['notify_email'] = email
		return executor.execute_plan(plan, cfg)

	def _ask_notify_email(self, Manager) -> str:
		"""Where to mail the result, for a backend whose runs outlive the session.

		Only asked for a scheduler-backed model: PFLOTRAN finishes in seconds
		while you watch, and offering to email you about it would be noise.

		TTY-ONLY, like every other prompt here. With no terminal it returns
		whatever the environment or the remembered preference already supplies,
		so a scripted run is never blocked waiting on an answer nobody is there
		to give.
		"""
		import sys as _sys
		from core.notify_prefs import remembered_email, remember_email

		if not getattr(Manager, 'NEEDS_SCHEDULER', False):
			return ''
		current = (os.environ.get('IDEAS_NOTIFY_EMAIL', '').strip()
				   or remembered_email() or '')
		if not (_sys.stdin and _sys.stdin.isatty()):
			return current

		prompt = (f"\n✉  Email when this study finishes?  [{current}]\n"
				  f"   (Enter to accept · type an address · '-' for none) "
				  if current else
				  "\n✉  Email when this study finishes? "
				  "(address, or Enter for none) ")
		try:
			reply = input(prompt).strip()
		except (EOFError, KeyboardInterrupt):
			print()
			return current

		if not reply:
			return current
		chosen = '' if reply == '-' else reply
		try:
			remember_email(chosen or None)
			print(f"   remembered in ~/.ideas/notify.json")
		except Exception as e:					# noqa: BLE001
			print(f"   (could not remember it: {e})")
		return chosen
	
	# ═════════════════════════════════════════════════════════
	# UTILITIES
	# ═════════════════════════════════════════════════════════
	def _print_status(self):
		print("\n" + "-" * 70)
		print("CONVERSATION STATUS")
		print("-" * 70)
		print(f"Last Run:     "
			  f"{self.conversation_context.get('last_run_dir', 'None')}")
		print(f"Last Focus:   "
			  f"{self.conversation_context.get('last_focus', 'None')}")
		print(f"Has Plan:     "
			  f"{'Yes' if self.conversation_context.get('last_plan') else 'No'}")
		print(f"Has Analysis: "
			  f"{'Yes' if self.conversation_context.get('last_analysis') else 'No'}")
		print("-" * 70 + "\n")
	
	def _clear_context(self):
		self.conversation_context = {
			'last_run_dir':  None,
			'last_plan':     None,
			'last_analysis': None,
			'last_focus':    None,
		}
	
	# ─────────────────────────────────────────────────────────
	# RENDERING analysis.json — the Analyzer's boundary file
	# ─────────────────────────────────────────────────────────
	# Both formatters below read `answer` / `verdict` / `claims` / `withheld`,
	# the schema step 4 writes. They used to read `answer_to_user_question` and
	# `key_findings` — the deleted report agent's shape, in which a claim was a
	# sentence and nothing else. A claim now travels with the figure and the
	# script that produced it, so the terminal can show where a number came
	# from, and `withheld` shows what the audit struck rather than hiding it.

	@staticmethod
	def _claim_lines(analysis: dict, limit: int = 3) -> list:
		out = []
		claims = (analysis.get('claims') or [])[:limit]
		if claims:
			out += ["🔍 CLAIMS:", "-" * 70]
			for i, c in enumerate(claims, 1):
				out.append(f"{i}. {c.get('claim', 'N/A')}")
				fig = c.get('figure')
				if fig:
					out.append(f"   figure: {fig}")
			out.append("")
		withheld = analysis.get('withheld') or []
		if withheld:
			out += ["🚫 WITHHELD (the audit struck these):", "-" * 70]
			for w in withheld[:3]:
				out.append(f"• {w.get('claim', 'N/A')}")
				out.append(f"  because: {w.get('struck_because', 'N/A')}")
			out.append("")
		return out

	def _format_analysis_response(self, analysis: dict,
								  run_dir: str = None) -> str:
		run_dir = run_dir or self.conversation_context.get('last_run_dir')
		lines = ["=" * 70, "ANALYSIS RESULTS", "=" * 70, ""]
		lines += ["📌 ANSWER:", "-" * 70,
				  analysis.get('answer') or 'N/A',
				  "",
				  f"verdict: {analysis.get('verdict')}", ""]
		lines += self._claim_lines(analysis)
		lines += [
			"=" * 70,
			f"Full report: {run_dir}/04_analysis/analysis.json",
			"=" * 70,
		]
		return "\n".join(lines)
	
	def _format_full_pipeline_response(self,
										run_summary: dict,
										analysis:    dict) -> str:
		lines = ["=" * 70, "WORKFLOW COMPLETE", "=" * 70, ""]
		lines += ["📌 ANSWER:", "-" * 70,
				  analysis.get('answer') or 'N/A',
				  "",
				  f"verdict: {analysis.get('verdict')}", ""]
		lines += ["⚙️  EXECUTION:", "-" * 70,
				  f"• Experiments: "
				  f"{run_summary['experiments_success']}/"
				  f"{run_summary['experiments_total']} succeeded",
				  f"• Runtime:     "
				  f"{run_summary['total_runtime_seconds']:.1f}s",
				  f"• Output:      "
				  f"{run_summary['run_directory']}", ""]
		lines += self._claim_lines(analysis)
		# CAVEATS REPLACE "RECOMMENDATIONS". The old agent invented next steps;
		# the pipeline carries what actually limits the run, which is the thing
		# a reader has to know before quoting any number above.
		caveats = [c for c in (analysis.get('caveats') or [])
				   if c.get('severity') in ('blocking', 'qualify')]
		if caveats:
			lines += ["⚠️  CAVEATS:", "-" * 70]
			for c in caveats[:3]:
				lines.append(f"  • [{c.get('severity')}] "
							 f"{c.get('statement') or c.get('id')}")
			lines.append("")
		lines += ["=" * 70,
				  f"Full details: {run_summary['run_directory']}"
				  f"/04_analysis/analysis.json",
				  "=" * 70]
		return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="IDEAS workflow: Reception → Planner → "
                    "Experiment Manager → Analyzer"
    )
    parser.add_argument(
        '--interactive', '-i',
        action = 'store_true',
        help   = 'Run in interactive mode'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default = './workflow_outputs',
        help    = 'Output directory'
    )
    parser.add_argument(
        '--mcp-config', '-m',
        default = 'mcp_config.json',
        help    = 'MCP configuration file'
    )
    parser.add_argument(
        '--resume',
        nargs   = '?',
        const   = '',            # bare --resume: list what can be resumed
        default = None,          # absent: not a resume at all
        metavar = 'RUN_DIR',
        help    = 'continue an interrupted or submitted run. With a run '
                  'directory, resumes that one; on its own, lists the runs '
                  'that can be resumed and why, and asks which.'
    )
    parser.add_argument(
        '--finalize',
        metavar = 'RUN_DIR',
        default = None,
        help    = 'run the pipeline tail (extract, package, analyze) against a '
                  'run directory whose output is already on disk, WITHOUT '
                  'polling the scheduler. This is the last line of the '
                  'combined study job — the job that produced the output is '
                  'the one running this, so polling would find itself active '
                  'and stop. Use --resume from a login node instead.'
    )
    parser.add_argument(
        '--request', '-r',
        metavar = 'TEXT',
        default = None,
        help    = 'run ONE study from this request and exit — the whole '
                  'pipeline, reception through the Analyzer, unattended. '
                  'Reception does not ask questions unless --ask is also '
                  'given (which needs a terminal); put the basin, the model '
                  'and the years in the sentence.'
    )
    parser.add_argument(
        '--no-ask',
        action = 'store_true',
        help   = 'do NOT let reception ask clarifying questions — it resolves '
                 'the period and other gaps itself and records them as '
                 'conflicts (the old default)'
    )
    parser.add_argument(
        '--ask', '-a',
        action = 'store_true',
        help   = argparse.SUPPRESS       # implied by --interactive; kept so
                                         # existing commands keep working
    )
    args = parser.parse_args()

    # ── --finalize, before anything expensive ────────────────────────
    # Runs on a COMPUTE NODE as the tail of the study job, so it must not
    # depend on a TTY and must not poll the scheduler. Placed ahead of
    # --resume because it delegates there once the finished stages are
    # adopted, and the two flags must not both fire.
    if args.finalize:
        coordinator = WorkflowCoordinator(
            default_output_dir    = args.output_dir,
            mcp_config_file       = args.mcp_config,
            interactive_reception = False,
        )
        print(coordinator.finalize_run(args.finalize))
        return

    # ── --resume, before anything expensive ──────────────────────────
    # Listing is a filesystem scan and needs no LLM, no MCP servers and no
    # reception. Building the coordinator first would spend all three to print
    # a directory listing — and would fail outright if the gateway were down,
    # which is precisely a moment someone might be trying to recover a run.
    if args.resume is not None:
        from core.resumable import find_resumable, print_listing
        target = args.resume
        if not target:
            rows = find_resumable(args.output_dir, check_jobs=True,
                                  include_all=True)
            print_listing(rows, args.output_dir)
            live = [r for r in rows if r.get("resumable")]
            if not live:
                return
            if sys.stdin.isatty():
                # THE CHOICE IS THE USER'S, including when there is only one.
                # Auto-resuming a lone candidate looks helpful and is not: a
                # bare --resume is often someone asking what is outstanding,
                # and answering it by starting work is not what was asked.
                # Enter means "I was only looking".
                try:
                    pick = input(f"Which one? [1-{len(live)}, or Enter to "
                                 f"quit] ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return
                if not pick.isdigit() or not 1 <= int(pick) <= len(live):
                    print("Nothing resumed.")
                    return
                target = live[int(pick) - 1]["run_dir"]
            else:
                # No TTY: print the commands rather than guessing.
                print("Pick one and pass it explicitly:")
                for r in live:
                    print(f"    python workflow.py --resume {r['run_dir']}")
                return

        coordinator = WorkflowCoordinator(
            default_output_dir    = args.output_dir,
            mcp_config_file       = args.mcp_config,
            interactive_reception = False,
        )
        print(coordinator.resume_run(target))
        return

    # --interactive means the user is sitting at a TTY answering prompts, so
    # reception may ask them things. Requiring a second --ask flag to enable
    # that made "interactive" mode silently non-interactive: it resolved the
    # simulation period on its own and went straight to the planner, which is
    # the one decision a user most wants to make.
    coordinator = WorkflowCoordinator(
        default_output_dir    = args.output_dir,
        mcp_config_file       = args.mcp_config,
        interactive_reception = (args.interactive or args.ask) and not args.no_ask,
    )

    if args.request:
        # ONE STUDY, UNATTENDED (2026-08-18). The end-to-end test that used to
        # need a driver script or a typed line at the interactive prompt:
        # reception -> planner -> the model's Experiment Manager -> Analyzer,
        # into a directory named for the model reception chose.
        print(coordinator.process_request(args.request))
        return

    if args.interactive:
        coordinator.run_interactive()
    else:
        print("=" * 70)
        print("Run with --interactive for interactive mode, or --request TEXT for one study")
        print("\nExamples:")
        print("  python workflow.py --interactive")
        print('  python workflow.py --request "How deep does water move through the '
              'unsaturated soil in the Naches River basin, Washington? Use PFLOTRAN. '
              'One year, 1988."')
        print("  python workflow.py --interactive --ask")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()