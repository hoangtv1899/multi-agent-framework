#!/bin/bash
# run_watershed.sh — drive the watershed pipeline end to end.
#
#   bash tools/run_watershed.sh [-i] "<question with HUC8 ...>"
#       steps 1-4 (plan -> materialize+plot -> adapter -> build/clone) + a
#       per-column soil pre-flight plot, then STOP and print the salloc run
#       block (step 5) + the analyze command (step 6). The safe default:
#       inspect 04_analysis/debug_surfaces.png before spending node time.
#
#   bash tools/run_watershed.sh --execute [-i] "<question ...>"
#       the same, but also run step 5 on a salloc node and analyze (step 6) —
#       the whole pipeline in one go.
#
#   bash tools/run_watershed.sh --analyze <run-dir>
#       step 6 only: analyze + result/timeseries plots (after a manual run).
#
#   -i forwards interactive clarification to reception (ambiguous names).
#   Env overrides: REF (reference case dir), YR_START, YR_END.
set -euo pipefail
cd "$(dirname "$0")/.."                  # project root

REF=${REF:-/pscratch/sd/h/hvtran/E3SMv3/1D_ELM.3c13216be8.2026-06-19-150916.elm_phase0}
YR_START=${YR_START:-1995}
YR_END=${YR_END:-1995}
PY=python3
SALLOC="salloc -N 1 -t 60:00 -q interactive -C cpu -A m3780"

analyze() {                              # step 6
    local RD="$1"
    [ -f "$RD/cases.json" ] || { echo "no $RD/cases.json — build/run first"; exit 1; }
    echo "==> [6/6] analyze $RD"
    $PY tools/analyze_run.py --run-dir "$RD" --cases-file cases.json --plan-file run_plan.json --plot
    $PY tools/plot_columns.py --run-dir "$RD" --cases-file cases.json --timeseries || true
    echo ""
    echo "results -> $RD/04_analysis/ (elevation_gradient.png, soil_control.png, hydro_summary.json)"
}

# ── --analyze mode (step 6 only) ─────────────────────────────────────────────
if [ "${1:-}" = "--analyze" ]; then
    [ -n "${2:-}" ] || { echo "usage: $0 --analyze <run-dir>"; exit 1; }
    analyze "$2"; exit 0
fi

# ── flags ────────────────────────────────────────────────────────────────────
INTERACTIVE=""; EXECUTE=""
while [ "${1:-}" ]; do
    case "$1" in
        -i)        INTERACTIVE="--interactive"; shift;;
        --execute) EXECUTE=1; shift;;
        -*)        echo "unknown flag: $1"; exit 1;;
        *)         break;;
    esac
done
Q="${1:-}"
[ -n "$Q" ] || { echo "usage: $0 [--execute] [-i] \"<question with HUC8 ...>\""; exit 1; }

RD="workflow_outputs/pipeline_$(date +%Y%m%d_%H%M%S)"
echo "==> run dir: $RD"

echo "==> [1/6] plan (reception -> planner)"
$PY tools/run_pipeline.py $INTERACTIVE --out "$RD" "$Q"
if [ ! -f "$RD/plan.json" ]; then
    echo ""
    echo "✗ no plan.json — reception did not return a runnable site design"
    echo "  (likely clarification_needed or a non-site question)."
    echo "  See $RD/reception_brief.json; refine the question or rerun with -i."
    exit 1
fi

echo "==> [2/6] materialize + planning plot"
$PY tools/expand_sampling.py --run-dir "$RD" --plot

echo "==> [3/6] adapter (columns -> executable plan)"
$PY src/core/columns_to_plan.py "$RD/columns.json" \
    --yr-start "$YR_START" --yr-end "$YR_END" --out "$RD/run_plan.json"

echo "==> [4/6] build/clone cases from $(basename "$REF")"
$PY tools/build_cases.py --plan "$RD/run_plan.json" --ref "$REF" --out-dir "$RD"

echo "==> pre-flight: per-column soil profiles"
$PY tools/plot_columns.py --run-dir "$RD" --cases-file cases.json --surfaces || true
NCOL=$($PY -c "import json;print(len(json.load(open('$RD/cases.json'))))")

# ── step 5 + 6 ───────────────────────────────────────────────────────────────
if [ "$EXECUTE" ]; then
    echo "==> [5/6] run $NCOL columns on a salloc node (this blocks until the node is granted)"
    EXE=$(cat "$RD/exe_path.txt")
    CASES=$($PY -c "import json;print(' '.join(json.load(open('$RD/cases.json'))))")
    $SALLOC bash tools/run_cases.sh "$EXE" $CASES | tee "$RD/run.log"
    analyze "$RD"
else
    cat <<EOF

────────────────────────────────────────────────────────────────────────
✓ built $NCOL columns.  First check $RD/04_analysis/debug_surfaces.png
  (distinct soil profiles per column?), then:

NEXT — [5/6] run on a compute node (copy-paste):

  $SALLOC
  EXE=\$(cat $RD/exe_path.txt)
  CASES=\$($PY -c "import json;print(' '.join(json.load(open('$RD/cases.json'))))")
  bash tools/run_cases.sh "\$EXE" \$CASES | tee $RD/run.log
  exit

THEN — [6/6] analyze:

  bash tools/run_watershed.sh --analyze $RD
────────────────────────────────────────────────────────────────────────
EOF
fi
