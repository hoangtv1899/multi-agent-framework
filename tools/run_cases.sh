#!/bin/bash
# Run the cloned ELM columns (shared exe). Args: <exe> <case_dir>...
EXE="$1"; shift
echo "exe: $EXE ($([ -f "$EXE" ] && echo present || echo MISSING))"
for C in "$@"; do
  name=$(basename "$C")
  echo "=== running $name ==="
  cd "$C" || { echo "  case dir missing"; continue; }
  source ./.env_mach_specific.sh 2>/dev/null
  mkdir -p run/timing/checkpoints
  cd run
  t0=$SECONDS
  srun --label -n 1 -N 1 -c 2 --cpu_bind=cores "$EXE" > srun.out 2>&1
  rc=$?
  nh=$(ls *.elm.h0.*.nc 2>/dev/null | wc -l)
  echo "  $name: rc=$rc  $(( (SECONDS - t0) / 60 ))min  history_files=$nh"
  grep -iE "SUCCESSFUL TERMINATION|forrtl|ERROR|abort|NetCDF" srun.out | tail -2
done
echo "ALL_DONE"
