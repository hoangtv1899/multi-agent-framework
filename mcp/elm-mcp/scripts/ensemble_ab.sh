#!/bin/bash
# JOB A — build the CIME cases, then run every column. Nothing else.
#
#     bash ensemble_ab.sh <run_dir> <python>
#
# The whole of what needs the compute node, and none of what does not. The
# analysis is job B's, submitted by the framework with --dependency=afterany on
# this job's id, so it starts by itself when this ends and this script never
# calls back into the framework to trigger it.
#
# THAT DIRECTION IS THE POINT. The script this replaces ended by running
# `workflow.py --finalize` — the server's job executing the client's code, and a
# circular dependency the boundary rule forbids. Jobs A and B invert it: the
# framework submits B and exits, SLURM presses the button.
#
# `afterany` on B's side, never `afterok`: with afterok a failed ensemble means B
# never runs and NO MAIL IS EVER SENT, which is the silent failure that happened
# twice on 2026-08-06. afterany means B always runs and always reports, including
# "the ensemble failed, here is why".
#
# Lives in the MCP because building and running ELM is the MCP's job. It is a
# FILE rather than a heredoc so it can be run by hand when a build fails, read
# without untangling shell quoting, and syntax-checked without a queue wait.
set -uo pipefail
RD="${1:?usage: ensemble_ab.sh <run_dir> <python>}"
PY="${2:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BUILT="$RD/01_inputs/built_cases.json"
T0=$SECONDS

ok_built () { $PY -c "import json,sys;d=json.load(open('$BUILT'));sys.exit(0 if d.get('ok') else 1)" 2>/dev/null; }

# -- 1. build ---------------------------------------------------------
if ok_built; then
  echo "-- build: reusing $BUILT --"
else
  echo "-- build --"
  $PY "$ROOT/mcp/elm-mcp/scripts/ensemble_job.py" "$RD" || echo "build step returned nonzero"
fi
ok_built || { echo "BUILD FAILED -- no cases to run; stopping before the ensemble"; exit 1; }

# -- 2. run every column concurrently ---------------------------------
# Case dirs and EXEROOT are read HERE, not templated in: they did not exist when
# this script was written. EXEROOT rather than <case>/build, so a --keepexe clone
# finds the reference case's executable instead of an empty directory of its own.
CASES=$($PY -c "import json;print(' '.join(c['case_dir'] for c in json.load(open('$BUILT'))['cases'] if c.get('case_dir')))")
FIRST=$(echo $CASES | awk '{print $1}')
EXE=$(cd "$FIRST" && ./xmlquery EXEROOT --value)/e3sm.exe
[ -x "$EXE" ] || { echo "no executable at $EXE"; exit 1; }
echo "-- run: $(echo $CASES | wc -w) column(s), exe=$EXE --"

# --exclusive, not --exact: Compy runs Slurm 18.08 and --exact arrived in 20.11.
# With --exact every column exits instantly with 0 history files.
FAIL=0
for C in $CASES; do
  (
    cd "$C/run" || exit 1
    source ../.env_mach_specific.sh 2>/dev/null
    mkdir -p timing/checkpoints
    srun --mpi=pmi2 --exclusive --ntasks=1 --cpus-per-task=2 --cpu_bind=cores --mem=4G "$EXE" > srun.out 2>&1
    RC=$?      # capture immediately: the $(basename ...) below would reset $?
    echo "  $(basename $C): rc=$RC history=$(ls *.elm.h0.*.nc 2>/dev/null | wc -l)"
    exit $RC
  ) &
done
wait
for job in $(jobs -p); do :; done

# Count what actually landed rather than trusting the loop: a column that wrote
# no history is a failure however srun exited.
N_OK=$($PY -c "
import json,glob,os
cs=[c['case_dir'] for c in json.load(open('$BUILT'))['cases'] if c.get('case_dir')]
print(sum(1 for c in cs if glob.glob(os.path.join(c,'run','*.elm.h0.*.nc'))))")
N=$(echo $CASES | wc -w)
echo "ENSEMBLE_DONE $N_OK/$N in $(( (SECONDS - T0) / 60 ))min"
[ "$N_OK" = "$N" ] || FAIL=1
exit $FAIL
