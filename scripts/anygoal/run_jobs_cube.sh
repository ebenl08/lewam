#!/bin/bash
#
# AnyGoal Cube-variant runs scored with AnyGoal's Cube metrics (scripts/anygoal/eval_lewam_anygoal_cube.py). Same
# arms, offset 25 / budget 50 and overrides as scripts/beam_tree/run_jobs.sh; one worker runs JOB_FILE ("env arm seed")
# on one UUID-pinned GPU. Run from playground/lewam inside tmux:
#   GPU=3 CORES=272-279 JOB_FILE=logs/beam_smoke/jobs_anygoal_c3_gpu3.txt GROUP=anygoal_c3 bash scripts/anygoal/run_jobs_cube.sh
# Data: data/beam_smoke/<env>/<GROUP>/<arm>/<seed>/; logs: logs/beam_smoke/<env>/<GROUP>/<arm>_<seed>.log

LW=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=/home/eben/.conda/envs/lewam/bin/python
GPU=${GPU:?nvidia-smi GPU index}
CORES=${CORES:?cores, e.g. 272-279}
JOB_FILE=${JOB_FILE:?file of "env arm seed" lines}
GROUP=${GROUP:?output subfolder, e.g. anygoal_c3}
[[ $GPU == 2 ]] && { echo "nvidia-smi GPU 2 is off limits" >&2; exit 1; }
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$GPU" '$1 == i {print $2}')
[[ -n $UUID ]] || { echo "no nvidia-smi GPU '$GPU'" >&2; exit 1; }

COMMON="eval.num_eval=30 eval.save_video=false eval.mode=lewam_plan eval.exec_actions=5 eval.horizon_schedule=adaptive \
eval.plan_cost_window=1 eval.plan_mode=beam"

task_args() {  # env
  case $1 in
    cube_hop|cube_diagonal) echo "--config-name=$1 policy=lewam-cube eval.rollout_steps=5" ;;
    *) echo "not a Cube variant: $1" >&2; return 1 ;;
  esac
}

arm_args() {  # arm (copied from scripts/beam_tree/run_jobs.sh)
  local b8c4="eval.beam_width=8 eval.beam_children=4" soft="eval.beam_prune=soft eval.beam_elite=true"
  case $1 in
    chains_b32c1) echo "eval.beam_width=32 eval.beam_children=1 eval.beam_score=goal eval.beam_prune=topk" ;;
    topk_goal) echo "$b8c4 eval.beam_score=goal eval.beam_prune=topk" ;;
    soft_t1_goal) echo "$b8c4 eval.beam_score=goal $soft eval.beam_tau=1.0" ;;
    native_grad) echo "eval.plan_mode=grad" ;;                                         # overrides COMMON's beam
    *) echo "unknown arm $1" >&2; return 1 ;;
  esac
}

cd "$LW" || exit 1
while read -r env arm seed; do
  [[ -z $env ]] && continue
  out=$LW/data/beam_smoke/$env/$GROUP/$arm/$seed
  log=$LW/logs/beam_smoke/$env/$GROUP/${arm}_$seed.log
  mkdir -p "$out" "$(dirname "$log")"
  echo "===== $(date +%T) $env $arm $seed"
  env CUDA_VISIBLE_DEVICES="$UUID" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MUJOCO_GL=egl taskset -c "$CORES" \
    "$PYTHON" scripts/anygoal/eval_lewam_anygoal_cube.py $(task_args "$env") seed="$seed" output_dir="$out" \
    eval.beam_spread_path="$out/beam_spread.jsonl" $COMMON $(arm_args "$arm") ${EXTRA:-} > "$log" 2>&1 \
    || echo "FAILED $env $arm $seed"
done < "$JOB_FILE"
echo "===== $(date +%T) done"
