#!/bin/bash
# run_study.sh — the WHOLE study as ONE batch job: build the CIME cases, run
# every column, then run the framework's own tail (extract → package →
# analyze) on the same allocation. Slurm mails you when it is over, and by
# then the analysis is already written.
#
#   bash tools/run_study.sh <run-dir> [-q short|slurm] [-t 02:00:00]
#                           [-m <email>] [--dry]
#
# Reads <run-dir>/01_inputs/case_inputs.json — the case list the framework
# wrote. Everything else is discovered INSIDE the job, which is the reason
# this is one script rather than three: the case directories do not exist
# until the build step has run, so they cannot be templated in beforehand.
#
# Why it exists: the split build/run/finalize flow cost the user three
# separate invocations, each waiting on a queue. This costs one.
#
# Restartable. The build step is skipped when 01_inputs/built_cases.json
# already reports success, so a job killed at the wall time can be resubmitted
# without paying for the ~8-10 min compile a second time.
#
# -m <email> uses Slurm's own mail rather than sendmail on the node, so it
# does not depend on a compute node having a working MTA. Confirmed
# delivering to @pnnl.gov from Compy on 2026-08-03 (job 770794).
#
# Compare tools/submit_cases.sh, which runs the columns only and is still the
# right tool when the cases are already built and you want nothing else.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

RD="${1:?usage: run_study.sh <run-dir> [-q short|slurm] [-t 02:00:00] [-m <email>] [--dry]}"; shift || true
QUEUE=short; TLIMIT=02:00:00; MAIL=""; DRY=""
while [ "${1:-}" ]; do
    case "$1" in
        -q) QUEUE="$2"; shift 2;;
        -t) TLIMIT="$2"; shift 2;;
        -m) MAIL="$2"; shift 2;;
        --dry) DRY=1; shift;;
        *) echo "unknown arg: $1"; exit 1;;
    esac
done

ABS_RD=$(readlink -f "$RD")
[ -f "$ABS_RD/01_inputs/case_inputs.json" ] || {
    echo "need $ABS_RD/01_inputs/case_inputs.json — the framework writes it"; exit 1; }
N=$(python3 -c "import json;print(len(json.load(open('$ABS_RD/01_inputs/case_inputs.json'))))")
# The job gets a login shell with NO conda environment, so the interpreter
# must be named explicitly — `command -v python3` there resolves to the
# system Python, which cannot import the framework. This is the same
# reason the MCP's build_elm_cases passes sys.executable.
PY="${IDEAS_PYTHON:-/qfs/people/tran289/.conda/envs/ideas/bin/python3}"
[ -x "$PY" ] || PY=$(command -v python3)
SB="$ABS_RD/run_study.sbatch"

MAILLINES=""
[ -n "$MAIL" ] && MAILLINES="#SBATCH --mail-user=$MAIL
#SBATCH --mail-type=END,FAIL"

cat > "$SB" <<SBATCH
#!/bin/bash
#SBATCH -J elm_study
#SBATCH -N 1
#SBATCH -p $QUEUE
#SBATCH -A e3sm
#SBATCH -t $TLIMIT
#SBATCH -o $ABS_RD/study.log
$MAILLINES
# NOT set -e. A failed column must not abort the job before the finalize step:
# a 17/19 ensemble is a result, and losing the analysis over it would be worse
# than the two columns.
set -u
export IDEAS_FRAMEWORK_DIR=$ROOT
export PSCRATCH=\${PSCRATCH:-/compyfs/tran289}
export LC_ALL=en_US.utf8
export LANG=en_US.utf8
cd $ROOT
echo "study: $N column(s) in $ABS_RD on \$(hostname)"
T0=\$SECONDS
BUILT=$ABS_RD/01_inputs/built_cases.json

# -- 1. build the CIME cases -----------------------------------------
if $PY -c "import json,sys;d=json.load(open('\$BUILT'));sys.exit(0 if d.get('ok') else 1)" 2>/dev/null; then
  echo "-- build: reusing \$BUILT --"
else
  echo "-- build --"
  $PY $ROOT/mcp/elm-mcp/build_cases_job.py $ABS_RD || echo "build step returned nonzero"
fi
$PY -c "import json,sys;d=json.load(open('\$BUILT'));sys.exit(0 if d.get('ok') else 1)" 2>/dev/null || {
  echo "BUILD FAILED -- no cases to run; stopping before the ensemble"; exit 1; }

# -- 2. run every column concurrently --------------------------------
# Case dirs and EXEROOT are read HERE, not templated above: they did not
# exist when this script was written.
CASES=\$($PY -c "import json;print(' '.join(c['case_dir'] for c in json.load(open('\$BUILT'))['cases'] if c.get('case_dir')))")
FIRST=\$(echo \$CASES | awk '{print \$1}')
EXE=\$(cd \$FIRST && ./xmlquery EXEROOT --value)/e3sm.exe
[ -x "\$EXE" ] || { echo "no executable at \$EXE"; exit 1; }
echo "-- run: \$(echo \$CASES | wc -w) column(s), exe=\$EXE --"
# --exclusive, not --exact: Compy runs Slurm 18.08 and --exact arrived in
# 20.11. With --exact every column exits instantly with 0 history files.
for C in \$CASES; do
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
echo "-- columns done in \$(( (SECONDS - T0) / 60 ))min --"

# -- 3. the framework's own tail, on this same allocation ------------
# --finalize, not --resume: the job that produced this output is the job
# running this line, so resume would poll itself, find it RUNNING, and stop
# one step short of the analysis it was submitted to produce.
echo "-- finalize (extract, package, analyze) --"
source /qfs/people/tran289/IDEAS/env_compy.sh 2>/dev/null || true
cd $ROOT
$PY workflow.py --finalize $ABS_RD \\
  || echo "finalize failed -- the model output stands; finish with: python workflow.py --resume $ABS_RD"
echo "STUDY_DONE in \$(( (SECONDS - T0) / 60 ))min"
SBATCH

echo "wrote $SB  ($N columns, queue=$QUEUE, t=$TLIMIT)"
if [ "$DRY" ]; then echo "(--dry -- not submitted; inspect the script above)"; exit 0; fi

JID=$(sbatch --parsable "$SB")
echo "submitted job $JID  ->  log: $ABS_RD/study.log"
echo "  watch:   squeue -j $JID    |    tail -f $ABS_RD/study.log"
[ -n "$MAIL" ] && echo "  email (END,FAIL) -> $MAIL -- by then the analysis is written"
echo "  results: $ABS_RD/04_analysis/"
exit 0
