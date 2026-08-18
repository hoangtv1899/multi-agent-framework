#!/usr/bin/env python3
"""
LAMBDA-PFLOTRAN — flow plus reactive transport
src/core/lambda_pflotran_exp_manager.py

    in   the same sampled columns every backend gets
    out  experiment.json with the flow metrics PFLOTRAN reports, plus the
         organic-matter chemistry the LAMBDA reaction sandbox adds

A SUBCLASS OF PFLOTRANExpManager, because the flow half is not merely similar
to a standalone PFLOTRAN run — it IS one. The reactive deck is built by taking
the flow deck this manager's parent already produces and grafting a CHEMISTRY
block onto it, so materialize, run and the flow half of extract are inherited
unchanged. What this class adds is three things the parent has no notion of:

    the reaction network   which organic-matter pools exist, and the
                           stoichiometry that turns each into biomass
    the deck graft         transport process model, chemistry, and a transport
                           partner for every flow condition
    a shorter clock        see YEARS below — this is the constraint that most
                           changes what the run MEANS

YEARS, AND WHY IT IS 1 AND NOT THE PARENT'S 20. The standalone demo's note
says the deck NaNs at t = 9.31 y on ITS column. On the framework's sampled
columns it fails far earlier, and I measured where:

    col_01  Dt -> 3.6e-16 y at t = 1.02 y      (at a 50 m AND a 12 m domain)
    col_02  Dt -> 7.1e-16 y at t = 1.35 y      (at a 50 m AND a 12 m domain)
    col_02  Dt -> 7.1e-16 y at t = 1.79 y      (regression network, stock db)

The timestep collapses to machine epsilon and the solve grinds without
advancing. Three things it is NOT: not the domain depth (identical collapse
time at 50 m and 12 m), not the saturation (columns with a water table fail
too), and not the generated reaction network (the build's own verified
regression network, with its stock database, fails on the same column). It is
the demonstration's reaction parameterisation meeting a real sampled column.

Every collapse observed is above 1.0 y, and at 1.0 y every water-bearing
column completes — measured, not assumed. So the cap is 1 y, and the ledger
records both the cap and that it is a different experiment from the parent's
20-year flow relaxation: those two must never be compared as if they were the
same run.

WHAT ONE YEAR STILL SHOWS. CH2O depletion of 0.19-0.30 across the ensemble,
ordered by water table depth. Biomass washes out to ~0 within the year under
this parameterisation, which is the same washout the demo's note describes,
arriving sooner — so biomass here is a diagnostic of the configuration, not a
prediction about the watershed.

WHERE THE NETWORK COMES FROM, in order: the reaction MCP's run_lambda_binning
tool, then the same binning called in-process, then the build's verified
regression fixture. Every candidate is validated before a deck is built from
it, and the one actually used is recorded per column.

TWO THINGS MEASURED ABOUT THAT MCP TOOL, both worth knowing before trusting a
binning choice made through it:

  * `binning_method` HAS NO EFFECT. "uniform", "cumulative", "class_based" and
    "bulk" return byte-identical networks at every bin count tested. The
    server is routed to the reconstructed binning package, which implements
    neither the four methods nor their semantics — so selecting one through
    the tool is a no-op, not a choice.
  * the donor NAMES can collide. A bin is named for its mean carbon number,
    ROUNDED, so two bins whose means round to the same integer become one
    species. See N_BINS.

WHAT THIS IS AND IS NOT. The sandbox parameters come verbatim from the build's
own verified regression case. They are a demonstration configuration, not a
calibration to any site — and the reaction network, when generated, carries its
own header saying the binning is RECONSTRUCTED. Both facts reach the Analyzer
through the ledger, because the figures this run produces would otherwise look
exactly like the figures of a calibrated study.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.pflotran_exp_manager import PFLOTRANExpManager, _load_tool


class LambdaPFLOTRANExpManager(PFLOTRANExpManager):
    """steps 0-5 for coupled Richards flow + LAMBDA reactive transport."""

    MODEL = "lambda_pflotran"

    # Its own key, for the reason the parent documents: two backends sharing a
    # plan key means one silently accepts the other's plan and builds nothing.
    PLAN_KEY = "LAMBDA_CASES"

    # Every timestep collapse measured on framework columns is above 1.0 y, and
    # at 1.0 y every water-bearing column completes. See the module docstring
    # for the measurements — this number is empirical, not the demo's 9.31.
    YEARS_CAP = 1.0

    # 3-5 bins is the useful range: below 3 the pools stop being distinguishable,
    # and each one is a primary species PFLOTRAN must transport.
    #
    # FIVE, NOT THREE, and the reason is the donor NAMES. Bins are named for
    # their mean carbon number, rounded, so bins whose means round to the same
    # integer collide. Swept against the reaction MCP for SPS_0001:
    #
    #     n=3   C21, C24, C21          collision
    #     n=4   C21, C23, C22, C23     two collisions
    #     n=5   C21, C22, C24, C23, C20   distinct
    #
    # This is a property of THIS sample's carbon distribution, not a rule, so
    # _validate_network still checks every network and the fallback still
    # exists. Five is simply the setting that works here.
    N_BINS = 5

    # Shorter than the parent's 900 s. A reactive column that is going to
    # finish does so in seconds; one whose timestep has collapsed to machine
    # epsilon will never finish, and waiting 15 minutes per column to learn
    # that costs an ensemble's worth of wall time to no purpose.
    RUN_TIMEOUT_S = 300

    FIELD_SEMANTICS = {
        **PFLOTRANExpManager.FIELD_SEMANTICS,
        "ch2o_final_M": {
            "units": "M", "from": ["Total CH2O(s)"],
            "note": "depth-mean solid organic carbon at the last output time. "
                    "The substrate the sandbox consumes"},
        "ch2o_depleted_frac": {
            "units": "1", "from": ["Total CH2O(s)"],
            "note": "fraction of the INITIAL CH2O consumed over the run. A "
                    "rate would be misleading: consumption is not linear in "
                    "time once O2 limits it"},
        "biomass_final_M": {
            "units": "M", "from": ["Total BIOMASS"],
            "note": "depth-mean microbial biomass at the last output time. "
                    "OBSERVED TO WASH OUT to ~0 within the 1 y run under this "
                    "demonstration parameterisation — a diagnostic of the "
                    "configuration, not a prediction about the watershed"},
        "o2_final_M": {
            "units": "M", "from": ["Total O2(aq)"],
            "note": "depth-mean dissolved oxygen; the electron acceptor, and "
                    "what limits oxidation once it is drawn down"},
        "ph_final": {
            "units": "1", "from": ["pH"], "note": "depth-mean pH"},
        "reaction_network_source": {
            "units": "text", "from": [],
            "note": "'binned[mcp]' — generated by the reaction MCP's "
                    "run_lambda_binning tool; 'binned' — the same binning "
                    "called in-process when the MCP is unavailable; or "
                    "'regression' — the build's verified fixture, used when "
                    "binning fails OR produces a network a correct deck "
                    "cannot be built from. The binning is RECONSTRUCTED and "
                    "none of the three is calibrated to this watershed"},
    }

    # ─────────────────────────────────────────────────────────
    # PLAN
    # ─────────────────────────────────────────────────────────
    def _to_run_plan(self, plan, columns, config, refine) -> Dict[str, Any]:
        """The parent's flow plan, on a shorter clock and with the chemistry
        assumptions recorded."""
        asked = float(config.get("years", self.YEARS_CAP))
        years = min(asked, self.YEARS_CAP)
        out = super()._to_run_plan(
            plan, columns, {**config, "years": years}, refine)

        out["pflotran_settings"]["years"] = years
        out["pflotran_settings"]["n_bins"] = int(
            config.get("n_bins", self.N_BINS))
        out["pflotran_settings"]["reactive"] = True

        ledger = out.setdefault("assumptions_ledger", [])
        if asked > self.YEARS_CAP:
            ledger.append({
                "key": "years_capped",
                "value": f"{asked} y requested, {years} y run",
                "why": f"the timestep collapses to machine epsilon between "
                       f"t=1.02 and t=1.79 y on sampled columns — measured at "
                       f"two domain depths, with and without a water table, "
                       f"and with the build's own regression network. Running "
                       f"the request as asked would have hung every column"})
        # THE REACTIVE ENSEMBLE IS SMALLER THAN THE FLOW ENSEMBLE, on purpose.
        # A column whose Fan water table lies below the domain runs fully
        # unsaturated, and aqueous-phase reactive transport through one is not
        # a hard problem — it is the wrong problem. There is almost no water to
        # carry solutes, so the GIRT solve drives its timestep to machine
        # epsilon: measured here, col_01 (Fan WTD 251 m) ground to
        # Dt = 3.6e-16 y at t = 1.017 y and never advanced, at BOTH a 50 m and
        # a 12 m domain — so this is the saturation, not the depth.
        dry = [c["id"] for c in out[self.PLAN_KEY] if not c.get("wt_in_domain")]
        if dry:
            ledger.append({
                "key": "columns_without_reactive_transport",
                "value": f"{len(dry)} of {len(out[self.PLAN_KEY])}: "
                         f"{', '.join(dry)}",
                "why": "no water table in the domain, so the column is fully "
                       "unsaturated. There is no aqueous phase to transport "
                       "solutes through; these columns are NOT part of the "
                       "reactive ensemble and no chemistry is reported for "
                       "them"})
        # MEASURED, and it bounds what this run can be used to claim. Running
        # the SAME columns with a 3-pool network and a 5-pool one changed only
        # pH, in the sixth decimal (7.79179957 -> 7.79179204); CH2O depletion,
        # biomass and O2 were byte-identical. The cause is in the
        # demonstration's own constraints: donors start at 1.d-4 M against
        # CH2O(s) at 1.1d2 M, six orders of magnitude apart, so the binned
        # pools act as tracers and the bulk substrate drives everything
        # reported. Without this on the record an interpretation could credit
        # the binning for a gradient the binning did not produce.
        ledger.append({
            "key": "reaction_network_influence",
            "value": "negligible on every reported metric except pH (6th dp)",
            "why": "the demonstration constraints initialise the donor pools "
                   "at 1e-4 M against CH2O(s) at 110 M. Differences between "
                   "binnings do not propagate to ch2o_depleted_frac, "
                   "biomass_final_M or o2_final_M — those describe the bulk "
                   "substrate, NOT the organic-matter pools the binning "
                   "resolves"})
        ledger.append({
            "key": "reaction_parameters",
            "value": "the build's verified regression configuration",
            "why": "a DEMONSTRATION of reactive transport, not a calibration "
                   "to this watershed — no rate here was fitted to any "
                   "measurement taken in the basin"})
        ledger.append({
            "key": "comparability_to_flow_only",
            "value": f"{years} y reactive vs {PFLOTRANExpManager.__name__}'s "
                     f"20 y flow relaxation",
            "why": "different durations answer different questions; the water "
                   "table in a 5-year reactive column has not relaxed as far "
                   "as one in a 20-year flow run and the two must not be "
                   "compared as if they had"})
        return out

    # ─────────────────────────────────────────────────────────
    # THE REACTION NETWORK
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _validate_network(network: Path, donors: List[str]) -> Optional[str]:
        """Why this network is unusable, or None if it is fine.

        THE DONOR NAMES ARE NOT GUARANTEED UNIQUE. Each bin is named for its
        MEAN CARBON NUMBER, rounded — so two bins whose means round to the
        same integer produce the same species name. Observed: the `cumulative`
        binning of SPS_0001 at n_bins=3 emits C21-DONOR, C24-DONOR, C21-DONOR,
        with different molar masses (438.9 and 464.9) for the two C21 rows.

        Left alone that is silently wrong rather than loudly broken: the deck
        declares the DEDUPLICATED species list, so PFLOTRAN sees two species
        for three reactions and the database gets two rows with one key. Two
        physically distinct organic-matter pools become one, and the run still
        produces plausible chemistry.
        """
        text = Path(network).read_text()
        reactions = [ln for ln in text.splitlines()
                     if ln.strip() and not ln.lstrip().startswith("#")]
        seen, dupes = set(), []
        for m in re.finditer(r"\b(C\d+-DONOR)\b", text):
            if m.group(1) in seen:
                dupes.append(m.group(1))
            seen.add(m.group(1))
        if dupes:
            return (f"duplicate donor species {sorted(set(dupes))} — two bins "
                    f"share a rounded carbon number, so distinct pools would "
                    f"silently merge into one")
        if len(donors) != len(reactions):
            return (f"{len(reactions)} reaction(s) but {len(donors)} donor "
                    f"species — every reaction must name its own pool")
        return None

    def _reaction_network(self, out_dir: Path, n_bins: int, brd,
                          clients: Optional[Dict[str, Any]] = None):
        """(network path, donor names, database path, source).

        Tries the binning pipeline first, because a network generated from the
        sample's own FTICR table is the point of the LAMBDA coupling. Falls
        back to the build's verified regression fixture — and RECORDS which
        was used, because the two are not equivalent and nothing in the output
        files would otherwise say which one produced them.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        net, donors, source = None, None, None

        # THE MCP FIRST. The reaction sandbox server exposes run_lambda_binning
        # as a tool, and going through it is the point of having registered the
        # server: the framework then uses the same entry point a person would,
        # rather than reaching past it into the package's internals.
        for attempt, fn in (("binned[mcp]", lambda: self._mcp_network(
                                out_dir, n_bins, (clients or {}).get("pflotran"))),
                            ("binned", lambda: self._binned_network(
                                out_dir, n_bins))):
            try:
                net, donors = fn()
            except Exception as e:                              # noqa: BLE001
                print(f"   ⚠️  {attempt} unavailable ({e})")
                net = None
                continue
            bad = self._validate_network(net, donors)
            if bad:
                # NOT a fallback to silence — the network is real, it is just
                # not one a correct deck can be built from.
                print(f"   ⚠️  {attempt} rejected: {bad}")
                net = None
                continue
            source = attempt
            break

        if net is None:
            import shutil
            src = Path(brd.RT) / "reaction_network_lambda.txt"
            if not src.exists():
                raise RuntimeError(
                    f"no reaction network: binning produced none and the "
                    f"regression fixture is missing at {src}")
            net = out_dir / "reaction_network_lambda.txt"
            shutil.copy2(src, net)
            donors = list(brd.DEFAULT_DONORS)
            source = "regression"

        # The donors must exist in the thermodynamic database or PFLOTRAN
        # fails at read time. The regression fixture's already do; a generated
        # network's do not.
        db = Path(brd.DB)
        if source and source.startswith("binned"):
            rows = self._donor_db_rows(out_dir, donors)
            if rows:
                db = brd.merge_donor_database(rows, out_dir / "lambda.dat")
        print(f"   reaction network [{source}]: {len(donors)} donor(s) "
              f"— {', '.join(donors)}")
        return net, donors, db, source

    # The sample the reaction MCP bins. One sample's FTICR table is what the
    # server ships with; a real study would name its own.
    SAMPLE_ID = "SPS_0001"

    # How compounds are grouped. The MCP's own vocabulary, which is NOT the
    # reconstruction package's ("lambda"/"nosc"/"quantile") — another reason
    # to go through the tool rather than the internals.
    BINNING_METHOD = "uniform"

    def _mcp_network(self, out_dir: Path, n_bins: int, client):
        """Bin through the reaction MCP's run_lambda_binning tool.

        The server is registered in mcp_config.json as `pflotran` and exposes
        39 tools; this is the one that turns an FTICR table into a PFLOTRAN
        reaction network. Using it keeps the framework on the same entry point
        a person calling the server by hand would use.
        """
        if client is None:
            raise RuntimeError("no `pflotran` MCP client in config['mcp_clients']")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        r = client.call_tool_json("run_lambda_binning", {
            "sample_id": self.SAMPLE_ID,
            "binning_method": self.BINNING_METHOD,
            "n_bins": int(n_bins),
            "output_dir": str(out_dir / "mcp")}) or {}
        if r.get("error"):
            raise RuntimeError(str(r["error"])[:200])

        src = r.get("reaction_network_file")
        if not src or not Path(src).exists():
            raise RuntimeError(
                f"the tool reported no reaction network (status="
                f"{r.get('validation_status')!r})")

        net = out_dir / "reaction_network.txt"
        net.write_text(Path(src).read_text())
        db_src = r.get("reaction_database_file")
        if db_src and Path(db_src).exists():
            (out_dir / "lambda_database.dat").write_text(
                Path(db_src).read_text())

        donors = self._donors_in(net)
        if not donors:
            raise RuntimeError("the tool's network names no donor species")
        return net, donors

    def _binned_network(self, out_dir: Path, n_bins: int):
        """Run the binning pipeline; return (network path, donor names)."""
        import importlib.util
        import sys
        import tempfile

        root = Path(self.LAMBDA_REFACTOR)
        if not (root / "src" / "binning").is_dir():
            raise FileNotFoundError(f"no binning package under {root}")
        sys.path.insert(0, str(root / "src"))
        try:
            from binning.binning_pipeline import BinningPipeline
            from binning.config import BinningConfig
        finally:
            pass

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)   # own its destination
        tmp = Path(tempfile.mkdtemp(prefix="lambda_bins_"))
        cfg = BinningConfig.from_dict({
            "n_bins": int(n_bins),
            "paths": {"preprocessing_dir": str(root / "analysis" / "preprocessing"),
                      "output_dir": str(tmp)}})
        BinningPipeline(cfg).run()

        net_src = next(tmp.rglob("reaction_network.txt"), None)
        if net_src is None:
            raise RuntimeError("binning wrote no reaction_network.txt")
        net = out_dir / "reaction_network.txt"
        net.write_text(net_src.read_text())

        db_src = next(tmp.rglob("lambda_database.dat"), None)
        if db_src is not None:
            (out_dir / "lambda_database.dat").write_text(db_src.read_text())

        donors = self._donors_in(net)
        if not donors:
            raise RuntimeError("the generated network names no donor species")
        return net, donors

    @staticmethod
    def _donors_in(network: Path) -> List[str]:
        """Donor species named in the network, in the order they appear.

        Read back OUT of the file rather than carried alongside it: the deck
        and the network must agree about the species list, and the only way to
        be sure is to take both from the same source.
        """
        seen, out = set(), []
        for m in re.finditer(r"\b(C\d+-DONOR)\b",
                             Path(network).read_text()):
            if m.group(1) not in seen:
                seen.add(m.group(1))
                out.append(m.group(1))
        return out

    @staticmethod
    def _donor_db_rows(out_dir: Path, donors: List[str]) -> List[str]:
        """The generated database's rows for these donors, if it wrote any."""
        p = out_dir / "lambda_database.dat"
        if not p.exists():
            return []
        want = set(donors)
        return [ln for ln in p.read_text().splitlines()
                if any(f"'{d}'" in ln for d in want)]

    # Where the LAMBDA refactor package lives. Outside this repo, so it is a
    # class attribute a test or a different install can point elsewhere.
    LAMBDA_REFACTOR = ("/qfs/people/tran289/IDEAS/reaction_sandbox_mcp-upstream/"
                       "lambda_pflotran_refactor")

    # ─────────────────────────────────────────────────────────
    # BUILD — flow decks from the parent, then the graft
    # ─────────────────────────────────────────────────────────
    def _build_case_inputs(self, plan: Dict[str, Any], config: Dict[str, Any]) -> List[Dict]:
        """Flow decks, then a reactive deck grafted onto each.

        The reactive decks go in their own directory rather than beside the
        flow decks, because _run takes the single .in it finds in a case dir —
        two decks in one directory and it would execute whichever the glob
        returned first.
        """
        flow = super()._build_case_inputs(plan, config)

        brd = _load_tool("build_reactive_demo")
        settings = plan.get("pflotran_settings") or {}
        out = self.run_dir / "01_inputs" / "lambda"
        net, donors, db, source = self._reaction_network(
            out, int(settings.get("n_bins", self.N_BINS)), brd,
            clients=config.get("mcp_clients"))

        # The flow decks already carry this recharge; passing the same value
        # keeps the graft from re-writing a flux the flow build chose.
        recharge = float(settings.get("recharge_mm_yr", 100.0))

        # Which columns hold water. _to_run_plan records why the others are
        # excluded; this is where they actually are.
        wet = {c["id"] for c in (plan.get(self.PLAN_KEY) or [])
               if c.get("wt_in_domain")}

        cases, failed, dry = [], [], []
        for e in flow:
            cid = e.get("id") or e.get("case_name")
            if wet and cid not in wet:
                dry.append(cid)
                continue
            src = next(Path(e.get("case_dir") or "").glob("*.in"), None)
            if src is None:
                failed.append(cid)
                continue
            case_dir = out / cid
            deck, _ = brd.build(src, case_dir, cid, recharge,
                                net.resolve(), donors=donors, db=db)
            cases.append({**e, "case_dir": case_dir, "flow_deck": str(src),
                          "deck": str(deck), "reaction_network": str(net),
                          "reaction_network_source": source,
                          "donors": donors})

        if dry:
            print(f"   {len(dry)} column(s) have no water table in the domain "
                  f"and get NO reactive deck — fully unsaturated, so there is "
                  f"no aqueous phase to transport through: "
                  f"{', '.join(map(str, dry))}")
        if failed:
            print(f"   ⚠️  {len(failed)} column(s) had no flow deck to graft "
                  f"onto and are absent from the ensemble: "
                  f"{', '.join(map(str, failed))}")
        if not cases:
            raise RuntimeError(
                f"no reactive decks built from {len(flow)} flow deck(s)"
                + (f" — all {len(dry)} column(s) were fully unsaturated, so "
                   f"this domain supports no reactive transport at all"
                   if dry else ""))
        print(f"✓ {len(cases)} reactive deck(s) → {out}")
        return cases

    # ─────────────────────────────────────────────────────────
    # EXTRACT — the parent's flow metrics, plus the chemistry
    # ─────────────────────────────────────────────────────────
    def _extract(self, experiments, plan=None, config=None):
        """Flow metrics from the parent, chemistry added on top.

        The parent's _read_tec indexes X, Y, Z, pressure, saturation
        positionally, and a reactive .tec inserts the chemistry columns AFTER
        saturation and before Material ID — so those five positions are
        unchanged and the inherited water-table and saturation metrics are
        correct here without modification. Chemistry needs the VARIABLES line,
        because its column order depends on the species list.
        """
        results = super()._extract(experiments, plan=plan, config=config)

        by_id = {(e.get("id") or e.get("case_name")): e
                 for e in (experiments or [])}
        n_chem = 0
        for row in results.results:
            e = by_id.get(row.get("case_name")) or {}
            row["reaction_network_source"] = e.get("reaction_network_source")
            row["donors"] = e.get("donors")
            if row.get("status") != "ok":
                continue
            tecs = sorted(Path(e.get("case_dir") or "").glob("*.tec"))
            chem = self._chemistry(tecs)
            if chem:
                row["metrics"].update(chem["metrics"])
                row.setdefault("variables", {}).update(chem["variables"])
                row.setdefault("profiles", {}).update(chem["profiles"])
                n_chem += 1

        results.units.update({"Total CH2O(s)": "M", "Total BIOMASS": "M",
                              "Total O2(aq)": "M", "pH": "-"})
        print(f"✓ chemistry on {n_chem}/{len(results.results)} column(s)")
        return results

    # Species pulled off the reactive output, and the metric each becomes.
    CHEM_VARS = {"Total CH2O(s)": "ch2o", "Total BIOMASS": "biomass",
                 "Total O2(aq)": "o2", "pH": "ph"}

    @classmethod
    def _chemistry(cls, tecs) -> Dict[str, Any]:
        """Depth-mean chemistry at the first and last output times."""
        if not tecs:
            return {}
        first = cls._read_tec_named(tecs[0])
        last = cls._read_tec_named(tecs[-1])
        if not last:
            return {}

        def mean(frame, col):
            v = frame.get(col) or []
            return (sum(v) / len(v)) if v else None

        metrics: Dict[str, Any] = {}
        for col, short in cls.CHEM_VARS.items():
            m = mean(last, col)
            if m is None:
                continue
            metrics[f"{short}_final_M" if short != "ph" else "ph_final"] = \
                round(m, 8)

        # Depletion needs the FIRST output time, not the constraint value: the
        # deck's initial condition is equilibrated before t=0 is written.
        c0, c1 = mean(first, "Total CH2O(s)"), mean(last, "Total CH2O(s)")
        if c0 and c1 is not None and c0 > 0:
            metrics["ch2o_depleted_frac"] = round(max(0.0, (c0 - c1) / c0), 6)
            metrics["ch2o_initial_M"] = round(c0, 8)

        variables = {}
        for col in cls.CHEM_VARS:
            v = last.get(col) or []
            if v:
                variables[col] = {"units": "-" if col == "pH" else "M",
                                  "min": round(min(v), 10),
                                  "max": round(max(v), 10),
                                  "mean": round(sum(v) / len(v), 10)}

        profiles = {}
        for col, short in cls.CHEM_VARS.items():
            series = []
            for t in tecs:
                f = cls._read_tec_named(t)
                if f.get(col):
                    series.append(f[col])
            if series:
                profiles[short] = series
        return {"metrics": metrics, "variables": variables,
                "profiles": profiles}

    @staticmethod
    def _read_tec_named(path: Path) -> Dict[str, List[float]]:
        """{variable name -> column}, from the Tecplot VARIABLES line.

        By NAME, not position: the chemistry column order follows the species
        list, so a run with different bins has a different layout. The parent's
        positional reader stays correct for the flow columns, which never move.
        """
        lines = Path(path).read_text().splitlines()
        if len(lines) < 4:
            return {}
        names = re.findall(r'"([^"]+)"', lines[1])
        if not names:
            return {}
        # Strip the unit suffix so callers ask for "Total CH2O(s)", not
        # "Total CH2O(s) [M]".
        clean = [re.sub(r"\s*\[[^\]]*\]\s*$", "", n).strip() for n in names]
        cols: Dict[str, List[float]] = {c: [] for c in clean}
        for line in lines[3:]:
            f = line.split()
            if len(f) < len(clean):
                continue
            try:
                vals = [float(x) for x in f[:len(clean)]]
            except ValueError:
                continue
            for c, v in zip(clean, vals):
                cols[c].append(v)
        return cols
