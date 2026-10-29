#!/bin/bash
#
# Paper reproduction (docs/REPRODUCE.md): goal-reaching eval of the released checkpoints, TASKS x arms {reactive,
# best-of-32, gradient} x SEEDS, 50 episodes per run, each task config's own goal offset and budget. Jobs are split
# round-robin over PER_GPU workers per GPU (stacked jobs), each worker in its own tmux window (session $SESSION) with
# 8 pinned cores from CORE_BASE up. Finished jobs (results.txt present) are skipped, so a rerun resumes. Run from
# playground/lewam:
#   GPUS="1 5 8" PER_GPU=3 bash scripts/paper_repro/run.sh             launch the tmux workers
#   GPUS="1 5 8" PER_GPU=3 TEST_RUN=1 bash scripts/paper_repro/run.sh  print each worker's commands, run nothing
#   NUM_EVAL=2 OUT=data/paper_repro/smoke ...                          smoke test

LW=$(cd "$(dirname "$0")/../.." && pwd)
GPUS=${GPUS:?nvidia-smi GPU indices, e.g. "1 5 8"}
PER_GPU=${PER_GPU:-1}
TASKS=${TASKS:-puzzle transport}
ARMS=${ARMS:-reactive best_of_k grad}
SEEDS=${SEEDS:-42 0 1}
NUM_EVAL=${NUM_EVAL:-50}
VIDEO_SEED=${VIDEO_SEED:-}  # save one mp4 per episode for this seed only; empty = no videos
VARIANT=${VARIANT:-}        # empty = paper protocol | reddot_max = PointMaze, reach the red marker from step 0
CORE_BASE=${CORE_BASE:-256}
OUT=${OUT:-$LW/data/paper_repro}
LOGS=${LOGS:-$LW/logs/paper_repro}
SESSION=${SESSION:-lewam_${TASKS// /_}${VARIANT:+_$VARIANT}}
TEST_RUN=${TEST_RUN:-0}
# nvidia-smi index -> EGL device id (measured 2026-10-22). robosuite (transport) cannot parse a UUID in
# CUDA_VISIBLE_DEVICES, so it renders on MUJOCO_EGL_DEVICE_ID, whose order differs from nvidia-smi's
EGL_ID=(3 4 0 1 2 8 9 5 6 7)

arm_args() {  # an arm may carry a deadline suffix, e.g. grad_d200 (pointmaze_harness)
  if [[ $1 =~ ^(.+)_d([0-9]+)$ ]]; then
    echo "$(arm_args "${BASH_REMATCH[1]}") eval.goal_offset_steps=${BASH_REMATCH[2]}"
    return
  fi
  case $1 in
    reactive) echo "eval.mode=lewam_policy" ;;
    best_of_k) echo "eval.mode=lewam_plan eval.plan_mode=best_of_k" ;;
    grad) echo "eval.mode=lewam_plan eval.plan_mode=grad eval.grad_all_k=true eval.grad_tr=0.01" ;;
    *) echo "unknown arm $1" >&2; return 1 ;;
  esac
}

policy() {  # released checkpoint (run name under $STABLEWM_HOME/checkpoints) of a task
  case $1 in
    pointmaze | pointmaze_harness) echo "lewam-pointmaze-large" ;;
    *) echo "lewam-$1" ;;
  esac
}

variant_args() {  # eval script and extra overrides of a variant (task, variant)
  case $2 in
    "") [[ $1 == pointmaze_harness ]] && echo "scripts/paper_repro/eval_lewam_pointmaze_harness.py" \
          || echo "scripts/eval_lewam.py" ;;
    reddot_max) echo "scripts/paper_repro/eval_lewam_reddot.py eval.full_traj=true" ;;
    *) echo "unknown variant $1" >&2; return 1 ;;
  esac
}

jobs() {  # one line per job: task arm seed
  for task in $TASKS; do
    for arm in $ARMS; do
      for seed in $SEEDS; do echo "$task $arm $seed"; done
    done
  done
}

worker() {  # worker index, GPU index, cores, number of workers
  local k=$1 gpu=$2 cores=$3 n=$4 i=0 uuid
  [[ $gpu == 2 ]] && { echo "nvidia-smi GPU $gpu is off limits" >&2; return 1; }
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$gpu" '$1 == g {print $2}')
  [[ -n $uuid ]] || { echo "no nvidia-smi GPU '$gpu'" >&2; return 1; }
  while read -r task arm seed; do
    (( i++ % n == k )) || continue
    local label=$task${VARIANT:+_$VARIANT}
    local dir=$OUT/$label/$arm/$seed log=$LOGS/${label}_${arm}_${seed}.log save_video=false
    [[ $seed == "$VIDEO_SEED" ]] && save_video=true
    local variant=($(variant_args "$task" "$VARIANT"))
    local cmd="cd $LW && STABLEWM_HOME=/home/eben/.stable_worldmodel /home/eben/.conda/envs/lewam/bin/python \
${variant[0]} --config-name $task policy=$(policy "$task") seed=$seed eval.num_eval=$NUM_EVAL \
eval.save_video=$save_video output_dir=$dir $(arm_args "$arm") ${variant[*]:1}"
    if [[ $TEST_RUN == 1 ]]; then echo "[gpu $gpu] $cmd"; continue; fi
    [[ -f $dir/results.txt ]] && { echo "skip $task $arm $seed (done)"; continue; }
    mkdir -p "$dir"
    printf '%s\t%s\t%s\n' "$uuid" "$cores" "$(date +%FT%T)" > "$dir/gpu.txt"
    echo "===== $(date +%T) $task $arm seed $seed"
    env CUDA_VISIBLE_DEVICES="$uuid" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MUJOCO_GL=egl \
      MUJOCO_EGL_DEVICE_ID="${EGL_ID[$gpu]}" taskset -c "$cores" \
      bash -c "$cmd" > "$log" 2>&1 || echo "FAILED $task $arm $seed"
    grep -E "success_rate" "$dir/results.txt" 2>/dev/null
  done < <(jobs)
  echo "===== $(date +%T) worker $k done"
}

slots=()  # one GPU index per worker, PER_GPU workers per GPU
for gpu in $GPUS; do
  for _ in $(seq "$PER_GPU"); do slots+=("$gpu"); done
done
if [[ -n $WORKER ]]; then
  first_core=$(( CORE_BASE + 8 * WORKER ))
  worker "$WORKER" "${slots[$WORKER]}" "$first_core-$(( first_core + 7 ))" "${#slots[@]}"
  exit
fi
mkdir -p "$LOGS"
for k in "${!slots[@]}"; do
  if [[ $TEST_RUN == 1 ]]; then WORKER=$k bash "$0"; continue; fi
  name=gpu${slots[$k]}_w$k
  log=$LOGS/worker_${TASKS// /_}${VARIANT:+_$VARIANT}_$name.log
  cmd="WORKER=$k GPUS='$GPUS' PER_GPU=$PER_GPU CORE_BASE=$CORE_BASE TASKS='$TASKS' ARMS='$ARMS' SEEDS='$SEEDS' \
NUM_EVAL=$NUM_EVAL VIDEO_SEED='$VIDEO_SEED' VARIANT='$VARIANT' OUT='$OUT' LOGS='$LOGS' bash $LW/scripts/paper_repro/run.sh > $log 2>&1"
  if (( k == 0 )); then tmux new-session -d -s "$SESSION" -n "$name" "$cmd"
  else tmux new-window -t "$SESSION" -n "$name" "$cmd"; fi
  echo "worker $k on GPU ${slots[$k]}, cores $(( CORE_BASE + 8 * k ))-$(( CORE_BASE + 8 * k + 7 )): tmux $SESSION:$name, log $log"
done
