#!/bin/bash
# run_watershed.sh — drive the watershed pipeline end to end.
#
#   bash tools/run_watershed.sh "<question with HUC8 ...>"
#       [1/6] plan (reception may ask clarifying questions on a terminal),
#       then a GREEN-LIGHT GATE: the resolved run settings (period, forcing,
#       columns, feasibility) are shown and you confirm before anything is
#       built. Steps 2-4 (materialize -> adapter -> build/clone) follow, plus
#       a per-column soil pre-flight plot, then STOP and print the salloc run
#       block (step 5) + the analyze command (step 6). The safe default:
#       inspect 04_analysis/debug_surfaces.png before spending node time.
#
#   bash tools/run_watershed.sh --execute "<question ...>"
#       the same, but also run step 5 on a salloc node and analyze (step 6).
#
#   bash tools/run_watershed.sh --yes "<question ...>"
#       batch mode: no clarifying questions, no gate (for scripted runs).
#
#   bash tools/run_watershed.sh --analyze <run-dir>
#       step 6 only: analyze + result/timeseries plots (after a manual run).
#
#   The simulation period comes from the question (reception resolves it
#   against forcing availability). Env YR_START/YR_END override it.
#   Env overrides: REF (reference case dir), YR_START, YR_END.
set -euo pipefail
cd "$(dirname "$0")/.."                  # project root

REF=${REF:-/pscratch/sd/h/hvtran/E3SMv3/1D_ELM.3c13216be8.2026-06-19-150916.elm_phase0}
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
INTERACTIVE=""; EXECUTE=""; YES=""
while [ "${1:-}" ]; do
    case "$1" in
        -i)        INTERACTIVE="--interactive"; shift;;   # kept for compat
        --yes)     YES=1; shift;;
        --execute) EXECUTE=1; shift;;
        -*)        echo "unknown flag: $1"; exit 1;;
        *)         break;;
    esac
done
Q="${1:-}"

# A controlling terminal enables interactive clarification + the gate by
# default; --yes turns both off. Test /dev/tty is openable (not just [ -t 0 ]):
# the MCP/LLM step disturbs fd 0, so we read prompts from /dev/tty later.
HAVE_TTY=""
if { : < /dev/tty; } 2>/dev/null; then HAVE_TTY=1; fi
if [ -z "$YES" ] && [ -n "$HAVE_TTY" ]; then INTERACTIVE="--interactive"; fi
if [ -n "$YES" ]; then INTERACTIVE=""; fi

# No question on the command line? Ask for it (needs a terminal).
if [ -z "$Q" ]; then
    if [ -n "$HAVE_TTY" ]; then
        echo "Enter your scientific question (one line, then Enter):"
        IFS= read -r Q < /dev/tty
    fi
    [ -n "$Q" ] || { echo "usage: $0 [--execute] [--yes] \"<question ...>\""; exit 1; }
fi

RD="workflow_outputs/pipeline_$(date +%Y%m%d_%H%M%S)"
echo "==> run dir: $RD"

echo "==> [1/6] plan (reception -> planner)"
# Connect the pipeline's stdin to the terminal so reception's clarifying
# questions can read your answer (the MCP subprocess churn breaks a later
# re-open of /dev/tty, so we hand it a real terminal fd from the start).
if [ -n "$HAVE_TTY" ]; then
    $PY tools/run_pipeline.py $INTERACTIVE --out "$RD" "$Q" < /dev/tty
else
    $PY tools/run_pipeline.py $INTERACTIVE --out "$RD" "$Q"
fi
if [ ! -f "$RD/plan.json" ]; then
    echo ""
    echo "✗ no plan.json — reception did not return a runnable site design"
    echo "  (likely clarification_needed or a non-site question)."
    echo "  See $RD/reception_brief.json; refine the question or rerun with -i."
    exit 1
fi

# ── resolve the simulation period: env override > reception brief > default ──
if [ -n "${YR_START:-}" ] || [ -n "${YR_END:-}" ]; then
    YR_START=${YR_START:-${YR_END}}; YR_END=${YR_END:-${YR_START}}
    PERIOD_SOURCE="user"
else
    read -r YR_START YR_END PERIOD_SOURCE <<< "$($PY - "$RD/reception_brief.json" <<'PYEOF'
import json, sys
rp = (json.load(open(sys.argv[1])).get("run_settings") or {}).get("resolved_period") or {}
y0, y1 = rp.get("yr_start"), rp.get("yr_end")
src = {"user": "user", "user-clamped": "user-clamped"}.get(rp.get("source"), "DEFAULT")
print(f"{y0} {y1} {src}" if isinstance(y0, int) and isinstance(y1, int) else "1995 1995 DEFAULT")
PYEOF
)"
fi

# ── green-light gate: confirm the resolved settings before building ─────────
$PY - "$RD" "$YR_START" "$YR_END" "$PERIOD_SOURCE" <<'PYEOF'
import json, sys
rd, y0, y1, src = sys.argv[1:5]
brief = json.load(open(f"{rd}/reception_brief.json"))
plan = json.load(open(f"{rd}/plan.json"))
fd = brief.get("forcing_data") or {}
fe = plan.get("feasibility") or {}
summ = plan.get("experiment_summary") or {}
dom = brief.get("domain") or {}
ny = int(y1) - int(y0) + 1
print("\n" + "─" * 72)
print("PROPOSED RUN  (nothing built yet)")
print("─" * 72)
area = dom.get("area_km2")
print(f"  domain      : {dom.get('name', '?')} (HUC {dom.get('huc', '?')})"
      + (f", {area:,.0f} km2" if isinstance(area, (int, float)) else ""))
print(f"  period      : {y0}-{y1}  ({ny} yr)   [source: {src}]")
print(f"  forcing     : {fd.get('source', 'NLDAS-2 (build default)')} "
      f"(available {fd.get('available_start_year', '?')}-"
      f"{fd.get('available_end_year', '?')})")
print(f"  columns     : {summ.get('total_columns', '?')} "
      f"({summ.get('exploratory', '?')} exploratory + "
      f"{summ.get('validation', '?')} validation)")
print(f"  feasibility : {fe.get('verdict', '?')}  "
      f"({len(fe.get('not_answerable') or [])} aspects need missing capabilities)")
for c in (brief.get("run_settings") or {}).get("conflicts") or []:
    print(f"  ⚠️  {c}")
print("─" * 72)
PYEOF
if [ -z "$YES" ] && [ -n "$HAVE_TTY" ]; then
    # Read from /dev/tty, not fd 0 — the reception step leaves stdin at EOF.
    read -r -p "Proceed to build with these settings? [Y/n/edit] " ANS < /dev/tty
    case "${ANS:-y}" in
        [nN]*) echo "Stopped. Plan is saved in $RD — rerun with YR_START/YR_END set,"
               echo "or refine the question."; exit 0;;
        [eE]*) read -r -p "  start year: " YR_START < /dev/tty
               read -r -p "  end year  : " YR_END < /dev/tty
               PERIOD_SOURCE="user";;
    esac
fi

echo "==> [2/6] materialize + planning plot"
$PY tools/expand_sampling.py --run-dir "$RD" --plot

echo "==> [3/6] adapter (columns -> executable plan)"
$PY src/core/columns_to_plan.py "$RD/columns.json" \
    --yr-start "$YR_START" --yr-end "$YR_END" \
    --period-source "$PERIOD_SOURCE" --out "$RD/run_plan.json"

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
