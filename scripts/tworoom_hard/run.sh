#!/bin/bash
#
# LeWAM on TwoRoom hard (manifest mode, goal offset as a deadline): ARMS x SEEDS, one job at a time on one
# UUID-pinned GPU and 8 pinned cores. Episodes = the stable-worldmodel cem_vs_icem_seeds25 runs for each seed
# (export them first with export_manifests.py). Run from playground/lewam:
#   ARMS="native_d25 native_d50 constant" SEEDS="8902 10748 17401" GPU=5 bash scripts/tworoom_hard/run.sh
#   NUM_EVAL=4 OUT=data/tworoom_hard/smoke ...   smoke test on the first rows of each manifest
#   TEST_RUN=1 ...                               print the commands, run nothing

LW=$(cd "$(dirname "$0")/../.." && pwd)
ARMS=${ARMS:?arms: native_d25 native_d50 constant}
SEEDS=${SEEDS:-8902 10748 17401}
GPU=${GPU:?one nvidia-smi GPU index}
CORES=${CORES:-256-263}
NUM_EVAL=${NUM_EVAL:-30}
OUT=${OUT:-$LW/data/tworoom_hard}
LOGS=${LOGS:-$LW/logs/tworoom_hard}
TEST_RUN=${TEST_RUN:-0}
[[ $GPU == 2 || $GPU == 8 ]] && { echo "nvidia-smi GPU $GPU is off limits" >&2; exit 1; }
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$GPU" '$1 == i {print $2}')
[[ -n $UUID ]] || { echo "no nvidia-smi GPU '$GPU'" >&2; exit 1; }

# P2 LeWAM planner settings (PushT), unchanged
COMMON="policy=lewam-tworoom eval.mode=lewam_plan eval.plan_mode=grad eval.grad_all_k=true eval.grad_tr=0.01 \
eval.exec_actions=5 eval.save_video=false"

arm_args() {
  case $1 in
    native_d25) echo "eval.horizon_schedule=adaptive eval.h0=5 eval.goal_offset_steps=25 eval.rollout_steps=5" ;;
    native_d50) echo "eval.horizon_schedule=adaptive eval.h0=10 eval.goal_offset_steps=50 eval.rollout_steps=10" ;;
    constant) echo "eval.horizon_schedule=constant eval.rollout_steps=5" ;;
    *) echo "unknown arm $1" >&2; return 1 ;;
  esac
}

# the exported manifests hold 30 rows (same-room first); a smoke test (NUM_EVAL < 30) evaluates the first and
# last NUM_EVAL / 2 rows, so both same-room and cross-wall pairs are run
if (( NUM_EVAL < 30 )); then
  MANIFESTS=$OUT/manifests
  [[ $TEST_RUN == 1 ]] || /home/eben/.conda/envs/lewam/bin/python -c "
import sys, numpy as np
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
for seed in sys.argv[4:]:
    m = np.load(f'{src}/seed{seed}_n30.npz')
    rows = np.r_[:n // 2, 30 - (n - n // 2):30]
    np.savez(f'{dst}/seed{seed}_n{n}.npz', **{k: m[k][rows] for k in m.files})
" "$LW/data/tworoom_hard/manifests" "$(mkdir -p "$MANIFESTS" && echo "$MANIFESTS")" "$NUM_EVAL" $SEEDS
else
  MANIFESTS=$LW/data/tworoom_hard/manifests
fi

mkdir -p "$LOGS"
for arm in $ARMS; do
  for seed in $SEEDS; do
    dir=$OUT/$arm/$seed
    cmd="cd $LW && /home/eben/.conda/envs/lewam/bin/python scripts/tworoom_hard/eval_lewam_tworoom_hard.py \
seed=$seed eval.num_eval=$NUM_EVAL eval.manifest_dir=$MANIFESTS output_dir=$dir $COMMON $(arm_args "$arm")"
    if [[ $TEST_RUN == 1 ]]; then echo "$cmd"; continue; fi
    mkdir -p "$dir"
    printf '%s\t%s\t%s\n' "$UUID" "$CORES" "$(date +%FT%T)" > "$dir/gpu.txt"
    echo "===== $(date +%T) $arm seed $seed -> $dir"
    env CUDA_VISIBLE_DEVICES="$UUID" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c "$CORES" \
      bash -c "$cmd" > "$LOGS/${arm}_${seed}.log" 2>&1 || echo "FAILED $arm $seed"
    grep -E "^success_rate" "$LOGS/${arm}_${seed}.log"
  done
done
echo "===== $(date +%T) done"
