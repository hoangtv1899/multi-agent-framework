#!/bin/bash
# submit_cases.sh — run a study's columns as ONE batch job, all columns
# CONCURRENTLY on a single node (each is tiny: 1 task / 2 cores, so ~60 fit on a
# 128-core node). Submit-and-forget — no interactive salloc, ~3 min wall time.
#
#   bash tools/submit_cases.sh <run-dir> [-q debug|regular] [-t 00:30:00] [--dry]
#
# Reads <run-dir>/exe_path.txt + cases.json; the job writes <run-dir>/run.log.
# --dry writes the sbatch script but does not submit (inspect it first).
set -euo pipefail
cd "$(dirname "$0")/.."

RD="${1:?usage: submit_cases.sh <run-dir> [-q debug|regular] [-t 00:30:00] [--dry]}"; shift || true
QUEUE=debug; TLIMIT=00:30:00; DRY=""
while [ "${1:-}" ]; do
    case "$1" in
        -q) QUEUE="$2"; shift 2;;
        -t) TLIMIT="$2"; shift 2;;
        --dry) DRY=1; shift;;
        *) echo "unknown arg: $1"; exit 1;;
    esac
done

[ -f "$RD/exe_path.txt" ] && [ -f "$RD/cases.json" ] || { echo "need $RD/exe_path.txt + cases.json (build first)"; exit 1; }
EXE=$(cat "$RD/exe_path.txt")
CASES=$(python3 -c "import json;print(' '.join(json.load(open('$RD/cases.json'))))")
N=$(python3 -c "import json;print(len(json.load(open('$RD/cases.json'))))")
ABS_RD=$(readlink -f "$RD")
SB="$RD/submit_cases.sbatch"

cat > "$SB" <<SBATCH
#!/bin/bash
#SBATCH -J elm_cols
#SBATCH -N 1
#SBATCH -q $QUEUE
#SBATCH -C cpu
#SBATCH -A m3780
#SBATCH -t $TLIMIT
#SBATCH -o $ABS_RD/run.log
# Run every column concurrently: each srun step takes exactly 1 task / 2 cores
# (--exact), so they pack onto the node instead of serialising.
EXE="$EXE"
echo "running $N columns concurrently on \$SLURM_NODELIST"
t0=\$SECONDS
for C in $CASES; do
  (
    cd "\$C/run" || exit
    source ../.env_mach_specific.sh 2>/dev/null
    mkdir -p timing/checkpoints
    srun --exact --ntasks=1 --cpus-per-task=2 --cpu-bind=cores --mem=4G "\$EXE" > srun.out 2>&1
    echo "  \$(basename \$C): rc=\$? history=\$(ls *.elm.h0.*.nc 2>/dev/null | wc -l)"
  ) &
done
wait
echo "ALL_DONE in \$(( (SECONDS - t0) / 60 ))min"
SBATCH

echo "wrote $SB  ($N columns, queue=$QUEUE, t=$TLIMIT)"
if [ "$DRY" ]; then echo "(--dry — not submitted; inspect the script above)"; exit 0; fi
JID=$(sbatch --parsable "$SB")
echo "submitted job $JID  →  log: $RD/run.log"
echo "  watch:   squeue -j $JID    |    tail -f $RD/run.log"
echo "  analyze: bash tools/run_watershed.sh --analyze $RD"
