#!/bin/bash
#
# Beam-search smoke runs: one worker runs the jobs of JOB_FILE (lines "env arm seed") one at a time on one
# UUID-pinned GPU, with spread logging on. Run from playground/lewam inside tmux:
#   GPU=1 CORES=272-279 JOB_FILE=logs/beam_smoke/jobs_gpu1.txt GROUP=spread bash scripts/beam_tree/run_jobs.sh
# Data: data/beam_smoke/<env>/<GROUP>/<arm>/<seed>/ (with beam_spread.jsonl); logs: logs/beam_smoke/<env>/<GROUP>/<arm>_<seed>.log

LW=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=/home/eben/.conda/envs/lewam/bin/python
GPU=${GPU:?nvidia-smi GPU index}
CORES=${CORES:?cores, e.g. 272-279}
JOB_FILE=${JOB_FILE:?file of "env arm seed" lines}
GROUP=${GROUP:?output subfolder, e.g. spread}
[[ $GPU == 2 || $GPU == 8 ]] && { echo "nvidia-smi GPU $GPU is off limits" >&2; exit 1; }
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$GPU" '$1 == i {print $2}')
[[ -n $UUID ]] || { echo "no nvidia-smi GPU '$GPU'" >&2; exit 1; }

COMMON="eval.num_eval=30 eval.save_video=false eval.mode=lewam_plan eval.exec_actions=5 eval.horizon_schedule=adaptive \
eval.plan_cost_window=1 eval.plan_mode=beam"

task_args() {  # env
  case $1 in
    tworoom_hard) echo "scripts/tworoom_hard/eval_lewam_tworoom_hard.py policy=lewam-tworoom \
eval.manifest_dir=$LW/data/tworoom_hard/manifests eval.h0=10 eval.goal_offset_steps=50 eval.rollout_steps=10" ;;
    puzzle) echo "scripts/eval_lewam.py --config-name=puzzle policy=lewam-puzzle eval.rollout_steps=5" ;;
    pusht) echo "scripts/eval_lewam.py --config-name=pusht policy=lewam-pusht eval.dataset_name=pusht_expert_train \
eval.rollout_steps=5" ;;
    *) echo "unknown env $1" >&2; return 1 ;;
  esac
}

arm_args() {  # arm
  local b8c4="eval.beam_width=8 eval.beam_children=4" soft="eval.beam_prune=soft eval.beam_elite=true"
  case $1 in
    chains_b32c1) echo "eval.beam_width=32 eval.beam_children=1 eval.beam_score=goal eval.beam_prune=topk" ;;
    topk_goal) echo "$b8c4 eval.beam_score=goal eval.beam_prune=topk" ;;
    soft_t1_goal) echo "$b8c4 eval.beam_score=goal $soft eval.beam_tau=1.0" ;;
    soft_t3_goal) echo "$b8c4 eval.beam_score=goal $soft eval.beam_tau=3.0" ;;
    topk_astar_l0p1) echo "$b8c4 eval.beam_score=astar eval.beam_g_weight=0.1 eval.beam_prune=topk" ;;
    topk_astar_l1) echo "$b8c4 eval.beam_score=astar eval.beam_g_weight=1.0 eval.beam_prune=topk" ;;
    soft_t3_astar_l0p1) echo "$b8c4 eval.beam_score=astar eval.beam_g_weight=0.1 $soft eval.beam_tau=3.0" ;;
    soft_t3_astar_l1) echo "$b8c4 eval.beam_score=astar eval.beam_g_weight=1.0 $soft eval.beam_tau=3.0" ;;
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
    "$PYTHON" $(task_args "$env") seed="$seed" output_dir="$out" eval.beam_spread_path="$out/beam_spread.jsonl" \
    $COMMON $(arm_args "$arm") > "$log" 2>&1 || echo "FAILED $env $arm $seed"
  grep success_rate "$log" | tail -1 | cut -c1-90
done < "$JOB_FILE"
echo "===== $(date +%T) done"
