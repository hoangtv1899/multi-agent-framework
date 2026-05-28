#!/bin/bash
#SBATCH -J elm_workflow_test
#SBATCH -N 1
#SBATCH -t 30:00
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -A m3780
#SBATCH -o slurm_logs/elm_workflow_test_%j.out
#SBATCH -e slurm_logs/elm_workflow_test_%j.err
#
# Submit the full ELM workflow test as a batch job.
# Replaces the interactive `salloc + python3 tests/test_workflow_elm.py`
# dance, so you can submit and walk away.
#
# Wallclock budget (3 experiments):
#   Build × 3   ~25-30 min
#   Run × 3     ~10-15 min   (longer than the 1-yr smoke test because
#                              the request asks for 5-yr runs)
#   LLM calls   ~1-2 min
#   Buffer     ~20 min
#   ─────────────────────────
#   Total      ~60-70 min   (90 min wallclock gives headroom)
#
# USAGE
# ─────
#   cd ~/RCSFA/multi-agent
#   mkdir -p slurm_logs
#   sbatch tests/submit_workflow_test.sh
#
#   # Watch progress:
#   squeue -u $USER
#   tail -f slurm_logs/elm_workflow_test_<JOBID>.out

set -e

# ── Working directory ────────────────────────────────────────────────
cd $HOME/RCSFA/multi-agent
mkdir -p slurm_logs

# ── Environment ──────────────────────────────────────────────────────
# Uncomment / modify whatever you usually run before `python3` in your
# interactive sessions. Common patterns at NERSC:
#
#   module load python
#   source $HOME/miniconda3/bin/activate <your_env>
#   source $HOME/.bashrc            # if env activation lives there

# ── Diagnostics ──────────────────────────────────────────────────────
echo "──────────────────────────────────────────────────────────"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURM_NODELIST"
echo "Start   : $(date)"
echo "Pwd     : $(pwd)"
echo "Python  : $(which python3)"
echo "──────────────────────────────────────────────────────────"
echo

# ── Run the workflow ─────────────────────────────────────────────────
module load pytorch
python3 tests/test_workflow_elm.py
RC=$?

# ── Done ─────────────────────────────────────────────────────────────
echo
echo "──────────────────────────────────────────────────────────"
echo "End     : $(date)"
echo "Exit RC : $RC"
echo "──────────────────────────────────────────────────────────"

exit $RC