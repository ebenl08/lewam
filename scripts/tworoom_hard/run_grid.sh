#!/bin/bash
#
# LeWAM native (deadline 50) on TwoRoom hard over the CEM grid: cost window {w1, wH} x h {2, 5, 10} x rh {1, 2, 5, 10}
# (rh <= h) x SEEDS, 30 episodes per run (the episodes of the stable-worldmodel CEM runs for each seed). Jobs are split
# round-robin over GPUS, one worker per GPU in its own tmux window (session $SESSION), one job at a time per GPU.
# Finished jobs (episodes.csv present) are skipped, so a rerun resumes. Run from playground/lewam:
#   GPUS="1 6" bash scripts/tworoom_hard/run_grid.sh             launch the tmux workers
#   GPUS="1 6" TEST_RUN=1 bash scripts/tworoom_hard/run_grid.sh  print each worker's commands, run nothing

LW=$(cd "$(dirname "$0")/../.." && pwd)
GPUS=${GPUS:?nvidia-smi GPU indices, e.g. "1 6"}
SEEDS=${SEEDS:-8902 10748 17401 18714 28967 30531 36136 37182 47259 60813}
CORES=(${CORES:-256-263 264-271 272-279 280-287})
OUT=${OUT:-$LW/data/tworoom_hard/grid}
LOGS=${LOGS:-$LW/logs/tworoom_hard/grid}
SESSION=${SESSION:-lewam_tworoom_grid}
TEST_RUN=${TEST_RUN:-0}
CELLS="2:1 2:2 5:1 5:2 5:5 10:1 10:2 10:5 10:10"

# P2 LeWAM planner settings (PushT), as scripts/tworoom_hard/run.sh
COMMON="policy=lewam-tworoom eval.mode=lewam_plan eval.plan_mode=grad eval.grad_all_k=true eval.grad_tr=0.01 \
eval.save_video=false eval.num_eval=30 eval.manifest_dir=$LW/data/tworoom_hard/manifests"

grid_args() {  # window h rh
  local window_cost=1
  [[ $1 == wH ]] && window_cost=$2
  echo "eval.horizon_schedule=adaptive eval.h0=10 eval.goal_offset_steps=50 eval.rollout_steps=$2 \
eval.exec_actions=$(( 5 * $3 )) eval.plan_cost_window=$window_cost"
}

jobs() {  # one line per job: window h rh seed
  for window in w1 wH; do
    for cell in $CELLS; do
      for seed in $SEEDS; do echo "$window ${cell%:*} ${cell#*:} $seed"; done
    done
  done
}

worker() {  # worker index, GPU index, cores
  local k=$1 gpu=$2 cores=$3 n=$4 i=0 uuid
  [[ $gpu == 2 || $gpu == 8 ]] && { echo "nvidia-smi GPU $gpu is off limits" >&2; return 1; }
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$gpu" '$1 == g {print $2}')
  [[ -n $uuid ]] || { echo "no nvidia-smi GPU '$gpu'" >&2; return 1; }
  while read -r window h rh seed; do
    (( i++ % n == k )) || continue
    local dir=$OUT/native_d50_$window/h${h}_rh${rh}/$seed log=$LOGS/${window}_h${h}_rh${rh}_${seed}.log
    local cmd="cd $LW && /home/eben/.conda/envs/lewam/bin/python scripts/tworoom_hard/eval_lewam_tworoom_hard.py \
seed=$seed output_dir=$dir $COMMON $(grid_args "$window" "$h" "$rh")"
    if [[ $TEST_RUN == 1 ]]; then echo "[gpu $gpu] $cmd"; continue; fi
    [[ -f $dir/episodes.csv ]] && { echo "skip $window h$h rh$rh $seed (done)"; continue; }
    mkdir -p "$dir"
    printf '%s\t%s\t%s\n' "$uuid" "$cores" "$(date +%FT%T)" > "$dir/gpu.txt"
    echo "===== $(date +%T) $window h$h rh$rh seed $seed"
    env CUDA_VISIBLE_DEVICES="$uuid" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c "$cores" \
      bash -c "$cmd" > "$log" 2>&1 || echo "FAILED $window h$h rh$rh $seed"
    grep -E "^success_rate" "$log"
  done < <(jobs)
  echo "===== $(date +%T) worker $k done"
}

gpus=($GPUS)
if [[ -n $WORKER ]]; then
  worker "$WORKER" "${gpus[$WORKER]}" "${CORES[$WORKER]}" "${#gpus[@]}"
  exit
fi
mkdir -p "$LOGS"
for k in "${!gpus[@]}"; do
  if [[ $TEST_RUN == 1 ]]; then WORKER=$k bash "$0"; continue; fi
  cmd="WORKER=$k GPUS='$GPUS' bash $LW/scripts/tworoom_hard/run_grid.sh > $LOGS/worker_gpu${gpus[$k]}.log 2>&1"
  if (( k == 0 )); then tmux new-session -d -s "$SESSION" -n "gpu${gpus[$k]}" "$cmd"
  else tmux new-window -t "$SESSION" -n "gpu${gpus[$k]}" "$cmd"; fi
  echo "worker $k on GPU ${gpus[$k]}, cores ${CORES[$k]}: tmux $SESSION:gpu${gpus[$k]}, log $LOGS/worker_gpu${gpus[$k]}.log"
done
