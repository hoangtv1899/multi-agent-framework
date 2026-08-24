#!/bin/bash
# The two-way ELM <-> PFLOTRAN iteration, inside ONE allocation.
# tools/coupling_loop.sh
#
#     sbatch -N 1 -p short -A e3sm -t 04:00:00 \
#            -o workflow_outputs/coupling_loop_%j.log \
#            tools/coupling_loop.sh <prior_pflotran_run_dir> [max_iters] [tol_m]
#
# ONE JOB, ONE QUEUE WAIT, THE WHOLE PICARD LOOP. Inside the allocation each
# leg is a plain `workflow.py --request`: the ELM manager notices SLURM_JOB_ID
# and runs its ensemble IN PLACE (srun on this job's own cores — the wrapper
# has always been srun-based; the sbatch was only an envelope), and the
# PFLOTRAN legs were always inline. The Analyzer's LLM calls work from compute
# nodes (job B has made them for weeks).
#
# CONVERGENCE IS CHECKED FIRST, off the prior PFLOTRAN leg's own record
# (tools/coupling_delta.py: how far each column's solved water table MOVED
# since the leg before it). Converged -> exit before spending a single ELM
# minute. Otherwise: ELM leg (PFLOTRAN's solved water table stamped into each
# column's finidat) -> PFLOTRAN leg (the new ELM run's QDRAI) -> check again.
#
# A SEED RUN HAS NOTHING TO HAVE MOVED FROM, so the first check of a fresh
# chain comes back UNDECIDABLE (exit 1), and that is a reason to iterate, not
# to stop: the loop treats it as "not converged" on the FIRST pass only. An
# undecidable check later in the loop is a real fault (a column that left the
# domain) and still stops.
#
# Size -t for max_iters * (one ELM ensemble + minutes of overhead); the
# Brandywine 3-column legs ran ~20 min each.
set -uo pipefail
PRIOR="${1:?usage: coupling_loop.sh <prior_pflotran_run_dir> [max_iters] [tol_m]}"
MAX_ITERS="${2:-3}"
TOL="${3:-0.10}"
# UNDER SBATCH THE SCRIPT IS A COPY: SLURM spools it to node-local /tmp, so
# BASH_SOURCE points there and a path derived from it is wrong (learned from
# job 773714, which looked for tools/ under /tmp/slurmd). SLURM_SUBMIT_DIR is
# where sbatch was RUN — the repo root, since that is how the header says to
# submit. Interactive use still resolves from the script's own location.
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
source /qfs/people/tran289/IDEAS/env_compy.sh
PY="${IDEAS_PYTHON:-python3}"

newest() { ls -td workflow_outputs/${1}_run_* 2>/dev/null | head -1; }

echo "== coupling loop in allocation ${SLURM_JOB_ID:-<none>} =="
echo "   prior PFLOTRAN leg: $PRIOR   max_iters=$MAX_ITERS   tol=${TOL} m"

for i in $(seq 1 "$MAX_ITERS"); do
  echo; echo "== iteration $i: convergence check on $PRIOR =="
  "$PY" tools/coupling_delta.py "$PRIOR" --tol "$TOL"; rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "== the two models agree — stopping after $((i-1)) iteration(s) =="
    exit 0
  elif [ "$rc" -eq 1 ] && [ "$i" -eq 1 ]; then
    # the seed leg of a fresh chain: no previous iteration exists, so there
    # is nothing it can have moved from. Iterate — that is what makes one.
    echo "   (seed leg — no previous iteration to compare against; iterating)"
  elif [ "$rc" -ne 3 ]; then
    echo "== convergence undecidable (see above) — stopping ==" ; exit 1
  fi

  echo; echo "== iteration $i: ELM leg (start at PFLOTRAN's solved water tables) =="
  "$PY" workflow.py --request "A coupling follow-up: re-run ELM from the prior PFLOTRAN run $(basename "$PRIOR"), reusing its columns exactly and starting each column at PFLOTRAN's solved water table and soil moisture." \
        --output-dir ./workflow_outputs || { echo "ELM leg failed"; exit 1; }
  ELM_LEG="$(newest elm)"
  echo "   ELM leg -> $ELM_LEG"

  echo; echo "== iteration $i: PFLOTRAN leg (drive with the new QDRAI) =="
  "$PY" workflow.py --request "A coupling follow-up: drive PFLOTRAN with the sub-surface drainage (QDRAI) from the prior ELM run $(basename "$ELM_LEG"). Reuse that run's columns exactly, anchored at ELM's own solved water table." \
        --output-dir ./workflow_outputs || { echo "PFLOTRAN leg failed"; exit 1; }
  PRIOR="$(newest pflotran)"
  echo "   PFLOTRAN leg -> $PRIOR"
done

echo; echo "== iteration budget spent — final check =="
"$PY" tools/coupling_delta.py "$PRIOR" --tol "$TOL"
