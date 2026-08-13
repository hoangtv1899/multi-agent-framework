#!/bin/bash
# submit_cases.sh — run a study's columns as ONE batch job, all columns
# CONCURRENTLY on a single node (each is tiny: 1 task / 2 cores, so ~60 fit on a
# 128-core node). Default is submit-and-forget; --wait blocks until the job
# finishes and prints a per-column summary; --analyze runs step 6 after that.
#
#   bash tools/submit_cases.sh <run-dir> [-q short|slurm] [-t 00:30:00]
#                              [--dry] [--wait] [--analyze]
#                              [--analyze-in-job] [-m <email>]
#
# Reads <run-dir>/exe_path.txt + cases.json; the job writes <run-dir>/run.log.
# --dry writes the sbatch script but does not submit (inspect it first).
# --analyze-in-job appends the analysis to the batch job itself (runs right
#   after the columns finish, on the already-allocated node) — submit and
#   walk away; results appear without any process waiting on a login node.
# -m <email> adds Slurm mail (END,FAIL) so you are notified when it is done.
# --wait/--analyze remain for the synchronous flow; do not combine --analyze
#   with --analyze-in-job (the analysis would run twice).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

RD="${1:?usage: submit_cases.sh <run-dir> [-q short|slurm] [-t 00:30:00] [--dry] [--wait] [--analyze] [--analyze-in-job] [-m <email>]}"; shift || true
# Compy partitions: 'short' (2 h limit) or 'slurm' (4 days).
QUEUE=short; TLIMIT=00:30:00; DRY=""; WAIT=""; ANALYZE=""; AJOB=""; MAIL=""
while [ "${1:-}" ]; do
    case "$1" in
        -q) QUEUE="$2"; shift 2;;
        -t) TLIMIT="$2"; shift 2;;
        -m) MAIL="$2"; shift 2;;
        --dry) DRY=1; shift;;
        --wait) WAIT=1; shift;;
        --analyze) ANALYZE=1; shift;;
        --analyze-in-job) AJOB=1; shift;;
        *) echo "unknown arg: $1"; exit 1;;
    esac
done
if [ -n "$ANALYZE" ] && [ -n "$AJOB" ]; then
    echo "use --analyze OR --analyze-in-job, not both"; exit 1
fi

[ -f "$RD/exe_path.txt" ] && [ -f "$RD/cases.json" ] || { echo "need $RD/exe_path.txt + cases.json (build first)"; exit 1; }
EXE=$(cat "$RD/exe_path.txt")
CASES=$(python3 -c "import json;print(' '.join(json.load(open('$RD/cases.json'))))")
N=$(python3 -c "import json;print(len(json.load(open('$RD/cases.json'))))")
ABS_RD=$(readlink -f "$RD")
SB="$RD/submit_cases.sbatch"

MAILLINES=""
[ -n "$MAIL" ] && MAILLINES="#SBATCH --mail-user=$MAIL
#SBATCH --mail-type=END,FAIL"

AJOBLINES=""
[ -n "$AJOB" ] && AJOBLINES="
echo \"── in-job analysis ──\"
source /qfs/people/tran289/IDEAS/env_compy.sh 2>/dev/null || true
cd $ROOT
python3 mcp/elm-mcp/scripts/analyze_run.py --run-dir $ABS_RD --plot \\
  || echo \"analysis failed — rerun with: python3 mcp/elm-mcp/scripts/analyze_run.py --run-dir $ABS_RD --plot\""

cat > "$SB" <<SBATCH
#!/bin/bash
#SBATCH -J elm_cols
#SBATCH -N 1
#SBATCH -p $QUEUE
#SBATCH -A e3sm
#SBATCH -t $TLIMIT
#SBATCH -o $ABS_RD/run.log
$MAILLINES
# Run every column concurrently: each srun step takes 1 task / 2 cores and
# --exclusive keeps steps from sharing CPUs, so they pack onto the node
# instead of serialising.
#   NOTE: --exclusive, not --exact. Compy runs Slurm 18.08, and --exact was
#   only added in Slurm 20.11 ("srun: unrecognized option '--exact'", every
#   column exits instantly with 0 history files). On 18.08 the step-level
#   spelling of "don't share CPUs between steps" is --exclusive.
EXE="$EXE"
echo "running $N columns concurrently on \$SLURM_NODELIST"
t0=\$SECONDS
for C in $CASES; do
  (
    cd "\$C/run" || exit
    source ../.env_mach_specific.sh 2>/dev/null
    mkdir -p timing/checkpoints
    srun --mpi=pmi2 --exclusive --ntasks=1 --cpus-per-task=2 --cpu_bind=cores --mem=4G "\$EXE" > srun.out 2>&1
    RC=\$?   # capture immediately: \$(basename ...) below would reset \$?
    echo "  \$(basename \$C): rc=\$RC history=\$(ls *.elm.h0.*.nc 2>/dev/null | wc -l)"
  ) &
done
wait
echo "ALL_DONE in \$(( (SECONDS - t0) / 60 ))min"
$AJOBLINES
SBATCH

echo "wrote $SB  ($N columns, queue=$QUEUE, t=$TLIMIT)"
if [ "$DRY" ]; then echo "(--dry — not submitted; inspect the script above)"; exit 0; fi

if [ -z "$WAIT" ]; then
    JID=$(sbatch --parsable "$SB")
    echo "submitted job $JID  →  log: $RD/run.log"
    echo "  watch:   squeue -j $JID    |    tail -f $RD/run.log"
    if [ -n "$AJOB" ]; then
        echo "  analysis runs inside the job after the columns finish"
        echo "  results will appear in $RD/04_analysis/"
    else
        echo "  analyze: python3 mcp/elm-mcp/scripts/analyze_run.py --run-dir $RD --plot"
    fi
    [ -n "$MAIL" ] && echo "  email notification (END,FAIL) → $MAIL"
    exit 0
fi

# ── --wait: block until the job finishes, then summarize per-column results ──
echo "submitting and waiting (sbatch --wait) ..."
RC=0
JID=$(sbatch --parsable --wait "$SB") || RC=$?
echo "job ${JID:-?} finished (sbatch rc=$RC)"
if command -v sacct >/dev/null && [ -n "${JID:-}" ]; then
    sacct -j "$JID" --format=JobID,State,Elapsed,MaxRSS -n 2>/dev/null | head -3 || true
fi

echo ""
echo "── per-column results ($RD/run.log) ──"
if [ ! -f "$RD/run.log" ]; then
    echo "✗ no run.log — job produced no output (check queue limits / sacct above)"
    exit 1
fi
grep "rc=" "$RD/run.log" || true
NOK=$(awk '/rc=/{ if ($0 ~ /rc=0( |$)/) n++ } END{print n+0}' "$RD/run.log")
NBAD=$(awk '/rc=/{ if ($0 !~ /rc=0( |$)/) n++ } END{print n+0}' "$RD/run.log")
if ! grep -q "ALL_DONE" "$RD/run.log"; then
    echo "⚠️  job ended WITHOUT ALL_DONE — likely hit the $TLIMIT wall time."
    echo "    Resubmit with more time, e.g.: bash tools/submit_cases.sh $RD -q slurm -t 02:00:00 --wait"
    exit 1
fi
if [ "$NBAD" -gt 0 ]; then
    echo "── $NOK/$N columns succeeded, $NBAD failed (see srun.out in each failed case's run/) ──"
else
    echo "── $NOK/$N columns succeeded ──"
fi

if [ -n "$ANALYZE" ]; then
    if [ "${NOK:-0}" -eq 0 ]; then
        echo "✗ nothing succeeded — skipping analysis"; exit 1
    fi
    echo ""
    python3 mcp/elm-mcp/scripts/analyze_run.py --run-dir "$RD" --plot
fi
