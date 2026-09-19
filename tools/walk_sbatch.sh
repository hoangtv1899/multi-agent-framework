#!/bin/bash
# Submit a prepared walk directory as one SLURM job.
#
#   tools/walk_sbatch.sh WALK_DIR [HOURS] [PARTITION]
#
# WALK_DIR   the directory tools/walk_setup.py wrote (holds walk.json)
# HOURS      wall-clock limit, default 2 (partition short allows up to 2;
#            pass PARTITION=slurm for anything longer, up to 4 days)
# PARTITION  short (default) or slurm
#
# The job log lands beside the walk directory on /compyfs, never on the login
# node's /tmp (job 774957 died in 4 s on a /tmp -o path). walk_job.py runs
# ELM through srun inside this allocation and PFLOTRAN serially, as the
# previous walks (774961, 774967, 775061) did on one node.
set -euo pipefail
WALK_DIR=$(readlink -f "${1:?usage: walk_sbatch.sh WALK_DIR [HOURS] [PARTITION]}")
HOURS=${2:-2}
PART=${3:-short}
[ -f "$WALK_DIR/walk.json" ] || { echo "no walk.json in $WALK_DIR" >&2; exit 2; }
NAME=$(basename "$WALK_DIR")
LOG_DIR=$(dirname "$WALK_DIR")
REPO=/qfs/people/tran289/IDEAS/multi-agent-framework
sbatch --job-name="$NAME" --account=e3sm --partition="$PART" --nodes=1 \
       --time="${HOURS}:00:00" --output="$LOG_DIR/${NAME}_%j.out" \
       --mail-type=END,FAIL --mail-user="${IDEAS_NOTIFY_EMAIL:-$USER@pnnl.gov}" \
       --wrap="source /qfs/people/tran289/IDEAS/env_compy.sh && cd $REPO && python tools/walk_job.py $WALK_DIR"
