#!/usr/bin/env python3
"""
Plan Validator
Pure Python structural checks for PFLOTRAN experiment plans.
No LLM involved — fast, deterministic, free.

Checks:
    1. Layer ordering        → auto-fix if inverted
    2. Z continuity          → flag gaps
    3. Material count        → flag mismatch
    4. Datum bounds          → flag out of domain
    5. Recharge positive     → flag negative flux
    6. Condition references  → flag undefined conditions
    7. Recharge syntax       → flag FLUX_LIST usage
"""
from typing import Any, Dict, List, Tuple


class PlanValidator:
    """
    Structural validator for PFLOTRAN experiment plans.
    Auto-fixes what it can, flags what it cannot.
    """

    def check(self, plan: Dict) -> Tuple[Dict, List[str]]:
        """Run all structural checks. Returns (fixed_plan, issues)."""
        issues = []

        plan, order_issues = self._check_layer_order(plan)
        issues.extend(order_issues)
        issues.extend(self._check_z_continuity(plan))
        issues.extend(self._check_material_count(plan))
        issues.extend(self._check_datum_bounds(plan))
        issues.extend(self._check_recharge(plan))
        issues.extend(self._check_condition_references(plan))
        issues.extend(self._check_recharge_syntax(plan))

        if not issues:
            issues.append("✓ All structural checks passed")

        return plan, issues

    # ─────────────────────────────────────────────────────────────────
    # CHECK 1: Layer ordering
    # ─────────────────────────────────────────────────────────────────

    def _check_layer_order(self,
                           plan: Dict) -> Tuple[Dict, List[str]]:
        """Layers must be ordered z_min ascending. Auto-fixes."""
        issues = []
        layers = plan.get("LAYER_REGIONS_AND_COORDINATES", [])
        if len(layers) < 2:
            return plan, issues
        z_mins = [
            self._parse_z(l.get("coordinates", {}).get("z_min", 0))
            for l in layers
        ]
        if z_mins != sorted(z_mins):
            plan = self._sort_layers(plan)
            issues.append(
                "⚠️  Layer order was inverted → auto-fixed"
            )
        return plan, issues

    # ─────────────────────────────────────────────────────────────────
    # CHECK 2: Z continuity
    # ─────────────────────────────────────────────────────────────────

    def _check_z_continuity(self, plan: Dict) -> List[str]:
        """z_max[i] must equal z_min[i+1]. Tolerance 1e-4 m."""
        issues = []
        layers = plan.get("LAYER_REGIONS_AND_COORDINATES", [])
        for i in range(len(layers) - 1):
            z_max_i  = self._parse_z(
                layers[i].get("coordinates", {}).get("z_max", 0)
            )
            z_min_i1 = self._parse_z(
                layers[i + 1].get("coordinates", {}).get("z_min", 0)
            )
            if abs(z_max_i - z_min_i1) > 1e-4:
                issues.append(
                    f"⚠️  Gap between layer {i + 1} and {i + 2}: "
                    f"z_max={z_max_i:.4f} ≠ z_min={z_min_i1:.4f}"
                )
        return issues

    # ─────────────────────────────────────────────────────────────────
    # CHECK 3: Material count
    # ─────────────────────────────────────────────────────────────────

    def _check_material_count(self, plan: Dict) -> List[str]:
        """Number of materials must equal number of layers."""
        n_layers = len(plan.get("LAYER_REGIONS_AND_COORDINATES",    []))
        n_mats   = len(plan.get(
            "MATERIAL_PROPERTIES_AND_CHARACTERISTIC_CURVES", []
        ))
        if n_layers != n_mats:
            return [f"⚠️  Layer count ({n_layers}) ≠ "
                    f"material count ({n_mats})"]
        return []

    # ─────────────────────────────────────────────────────────────────
    # CHECK 4: Datum bounds
    # ─────────────────────────────────────────────────────────────────

    def _check_datum_bounds(self, plan: Dict) -> List[str]:
        """Every HYDROSTATIC datum_z must be within (0, total_depth)."""
        issues = []
        layers = plan.get("LAYER_REGIONS_AND_COORDINATES", [])
        if not layers:
            return issues
        total_depth     = self._parse_z(
            layers[-1].get("coordinates", {}).get("z_max", 0)
        )
        flow_conditions = plan.get("FLOW_CONDITIONS", {})
        for coupler in plan.get("CONDITIONS_COUPLERS", []):
            ic_name = coupler.get("INITIAL_CONDITION", "")
            ic      = flow_conditions.get(ic_name, {})
            datum   = self._extract_datum(ic)
            if datum is not None:
                if not (0 < datum < total_depth):
                    issues.append(
                        f"⚠️  Datum {datum:.3f} m out of bounds "
                        f"[0, {total_depth:.3f}] in '{ic_name}'"
                    )
        return issues

    # ─────────────────────────────────────────────────────────────────
    # CHECK 5: Recharge positive
    # ─────────────────────────────────────────────────────────────────

    def _check_recharge(self, plan: Dict) -> List[str]:
        """All LIQUID_FLUX values must be positive."""
        issues = []
        for cond_name, cond in plan.get("FLOW_CONDITIONS", {}).items():
            flux_val = cond.get("LIQUID_FLUX")
            if flux_val is not None:
                flux = self._parse_z(
                    str(flux_val).split()[0]
                )
                if flux < 0:
                    issues.append(
                        f"⚠️  Negative recharge in '{cond_name}': {flux}"
                    )
        return issues

    # ─────────────────────────────────────────────────────────────────
    # CHECK 6: Condition references
    # ─────────────────────────────────────────────────────────────────

    def _check_condition_references(self, plan: Dict) -> List[str]:
        """
        Every condition referenced in CONDITIONS_COUPLERS
        must exist in FLOW_CONDITIONS.
        This is the most common LLM mistake [1].
        """
        issues  = []
        defined = set(plan.get("FLOW_CONDITIONS", {}).keys())

        for coupler in plan.get("CONDITIONS_COUPLERS", []):
            exp_name = coupler.get("EXPERIMENT", "unknown")
            refs = {
                "INITIAL_CONDITION":
                    coupler.get("INITIAL_CONDITION"),
                "BOUNDARY_CONDITION_SURFACE":
                    coupler.get("BOUNDARY_CONDITION_SURFACE"),
                "BOUNDARY_CONDITION_DEEP_BOUNDARY":
                    coupler.get("BOUNDARY_CONDITION_DEEP_BOUNDARY"),
            }
            for ref_type, ref_name in refs.items():
                if ref_name and ref_name not in defined:
                    issues.append(
                        f"❌ [{exp_name}] {ref_type} "
                        f"'{ref_name}' not defined in FLOW_CONDITIONS"
                    )
        return issues

    # ─────────────────────────────────────────────────────────────────
    # CHECK 7: Recharge syntax
    # ─────────────────────────────────────────────────────────────────

    def _check_recharge_syntax(self, plan: Dict) -> List[str]:
        """
        Recharge conditions must use LIQUID_FLUX NEUMANN.
        Flag if FLUX_LIST is used instead.
        """
        issues = []
        for cond_name, cond in plan.get("FLOW_CONDITIONS", {}).items():
            if "FLUX_LIST" in cond:
                issues.append(
                    f"⚠️  '{cond_name}' uses FLUX_LIST — "
                    f"use LIQUID_FLUX NEUMANN instead"
                )
        return issues

    # ─────────────────────────────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────────────────────────────

    def _sort_layers(self, plan: Dict) -> Dict:
        """Sort layers and materials by z_min ascending."""
        layers = plan.get("LAYER_REGIONS_AND_COORDINATES", [])
        mats   = plan.get(
            "MATERIAL_PROPERTIES_AND_CHARACTERISTIC_CURVES", []
        )
        if len(layers) == len(mats):
            paired = sorted(
                zip(layers, mats),
                key=lambda p: self._parse_z(
                    p[0].get("coordinates", {}).get("z_min", 0)
                )
            )
            sorted_layers, sorted_mats = zip(*paired)
            plan["LAYER_REGIONS_AND_COORDINATES"] = list(sorted_layers)
            for i, mat in enumerate(sorted_mats, 1):
                mat["ID"] = i
            plan["MATERIAL_PROPERTIES_AND_CHARACTERISTIC_CURVES"] = (
                list(sorted_mats)
            )
        else:
            plan["LAYER_REGIONS_AND_COORDINATES"] = sorted(
                layers,
                key=lambda l: self._parse_z(
                    l.get("coordinates", {}).get("z_min", 0)
                )
            )
        return plan

    @staticmethod
    def _parse_z(val: Any) -> float:
        """Convert Fortran or standard float notation to Python float."""
        try:
            return float(
                str(val).replace("d", "e").replace("D", "E")
            )
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _extract_datum(condition: Dict) -> Any:
        """Extract z datum from HYDROSTATIC condition."""
        datum_str = condition.get("DATUM", "")
        if not datum_str:
            return None
        parts = str(datum_str).split()
        if len(parts) >= 3:
            try:
                return float(
                    parts[2].replace("d", "e").replace("D", "E")
                )
            except ValueError:
                return None
        return None