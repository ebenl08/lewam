#!/bin/bash
#
# P2 planner latency on PushT (notes/pending_implementation/lewam-planner-comparison/p2-latency.md). One worker per
# GPU pops a whole config (arm h) from a shared queue and runs its 10 seeds back to back, pinned to its own 8 CPU
# cores. GPUs must be otherwise idle: a shared GPU inflates latency.
#
# Protocol as PG scripts/latency/measure_cem_latency.sh: 10 envs, eval_budget 50, rh 1, action_block 5, batch 1
# (one env per plan call), CUDA events, first replan dropped.
#   LeWAM arms (lewam conda env, this repo): native / grid_w1 / grid_wH, eval.timing=single
#   our arms (swm conda env, PG entries unchanged, read-only configs): cem 300x10, icem_faithful 300x10 (goal_mse),
#     leflow_cem / leflow_icem 300x5 fitted prior, warm k16, clamp (rollout_goal), as the PushT SR runs
#   throughput: LeWAM native paper cell batched (eval.timing=batched, 50 envs), seeds 42 0 1, vs paper Table 6
# Outputs: data/planner_comparison/p2/latency/pusht/<arm>/h<h>/<seed>/ (timing.json, results.txt, config.json).
# Run from playground/lewam/:
#   bash scripts/planner_comparison/latency.sh --jobs > logs/planner_comparison/latency_jobs.tsv
#   GPUS="3 4 6" WAIT=1 bash scripts/planner_comparison/latency.sh   (nvidia-smi indices, never 2 or 8)
#   TEST_RUN=1   print the command of the first seed of every config, run nothing
#   SEED_LIST="10247 15783"   run a round on a seed subset (finished runs are skipped)
#   CALIBRATE=1  run cem h5 on every GPU in parallel (seeds of SEED_LIST) into calibration/<gpu>/ to compare GPUs
#   DETAIL=planner|model|mot   LeWAM nested stage timers (eval.timing_detail); use with a separate ROOT
# Job lines: arm<TAB>h[<TAB>envs per plan call B[<TAB>LeWAM proposals N]]; B > 1 plans B envs in one call
# (num_eval = B) and writes to <arm>/h<h>_b<B>[_n<N>]/; B = 1 / no N keeps the P2 layout <arm>/h<h>/.

LW=$PWD
PG=$(cd "$LW/../stable-worldmodel" && pwd)
LW_PYTHON=${LW_PYTHON:-/home/eben/.conda/envs/lewam/bin/python}
PG_PYTHON=${PG_PYTHON:-/home/eben/.conda/envs/swm/bin/python}
GROUP=planner_comparison
JOBS=${JOBS:-logs/${GROUP}/latency_jobs.tsv}
QUEUE=logs/${GROUP}/latency_queue.tsv
FAILED=logs/${GROUP}/latency_failed.tsv
ROOT=${ROOT:-$LW/data/${GROUP}/p2/latency/pusht}  # stage-breakdown runs: ROOT=$PWD/data/planner_comparison/p2/breakdown/pusht
CORE_BASE=${CORE_BASE:-256}  # worker k uses cores CORE_BASE + 8k .. + 8k + 7
TEST_RUN=${TEST_RUN:-0}
read -ra SEEDS <<< "${SEED_LIST:-10247 15783 56783 77596 85661 90789 68975 23471 27901 91991}"  # a round = a subset
read -ra THROUGHPUT_SEEDS <<< "${THROUGHPUT_SEED_LIST:-42 0 1}"
ARMS=(native grid_w1 grid_wH cem icem_faithful leflow_cem leflow_icem)
mkdir -p logs/${GROUP}

if [[ ${1:-} == --jobs ]]; then
  for arm in "${ARMS[@]}"; do for h in 2 5 10; do printf '%s\t%s\n' "$arm" "$h"; done; done
  printf 'throughput\t5\n'
  exit 0
fi

gpu_uuid() {  # UUID of nvidia-smi GPU $1; CUDA indices skip smi GPU 2, so pin by UUID
  local uuid
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$1" '$1 == i {print $2}')
  [[ -n $uuid ]] || { echo "no nvidia-smi GPU '$1'" >&2; return 1; }
  echo "$uuid"
}

read -ra gpus <<< "${GPUS:?set GPUS to idle nvidia-smi GPU indices, one worker each}"
(( ${#gpus[@]} <= 4 )) || { echo "at most 4 GPUs" >&2; exit 1; }
for i in "${!gpus[@]}"; do
  [[ ${gpus[$i]} == 2 || ${gpus[$i]} == 8 ]] && { echo "nvidia-smi GPU ${gpus[$i]} is off limits" >&2; exit 1; }
  gpus[$i]=$(gpu_uuid "${gpus[$i]}") || exit 1
done

lewam_cmd() {  # $1 arm, $2 h, $3 seed, $4 output folder, $5 envs per plan call (1 = single), $6 proposals N (optional)
  local arm=$1 h=$2 seed=$3 out=$4 batch=${5:-1} samples=${6:-}
  local args=(--config-name=pusht policy=lewam-pusht eval.dataset_name=pusht_expert_train eval.save_video=false
    eval.mode=lewam_plan eval.plan_mode=grad eval.grad_all_k=true eval.grad_tr=0.01 eval.exec_actions=5)
  case $arm in
    native) args+=(eval.rollout_steps="$h") ;;
    grid_w1) args+=(eval.rollout_steps="$h" eval.horizon_schedule=constant) ;;
    grid_wH) args+=(eval.rollout_steps="$h" eval.horizon_schedule=constant eval.plan_cost_window="$h") ;;
    throughput) args+=(eval.num_eval=50 eval.timing=batched) ;;  # paper cell: rollout_steps from h0 = 5
  esac
  if [[ $arm != throughput ]]; then
    if (( batch > 1 )); then args+=(eval.num_eval="$batch" eval.timing=batched)  # one plan call over all envs
    else args+=(eval.num_eval=10 eval.timing=single); fi
  fi
  [[ -n $samples ]] && args+=(eval.num_proposals="$samples")
  [[ -n ${DETAIL:-} ]] && args+=(eval.timing_detail="$DETAIL")  # planner | model | mot stage timers
  echo "cd $LW && $LW_PYTHON scripts/eval_lewam.py ${args[*]} seed=$seed output_dir=$out"
}

pg_cmd() {  # $1 arm, $2 h, $3 seed, $4 output folder, $5 envs per plan call
  local arm=$1 h=$2 seed=$3 out=$4 batch=${5:-1} num_eval=10
  (( batch > 1 )) && num_eval=$batch  # one solve over all envs
  local common=(seed="$seed" eval.num_eval="$num_eval" eval.eval_budget=50 ++eval.save_episode_video=false
    plan_config.horizon="$h" plan_config.receding_horizon=1 plan_config.action_block=5 solver.batch_size="$batch"
    +output_folder="$out")
  local quiet=(++callbacks.solver_progress.enabled=false ++callbacks.observers.rollout=false
    ++callbacks.observers.trajectory=false ++callbacks.observers.reward_cost=false
    ++callbacks.observers.embedding_mse=false ++callbacks.observers.pca_features=false
    ++callbacks.observers.goal_mismatch=false ++decoder.enabled=false)
  local args
  case $arm in
    cem|icem_faithful|cem_300x30)
      local solver=cem iterations=10
      [[ $arm == icem_faithful ]] && solver=icem2/icem_faithful
      [[ $arm == cem_300x30 ]] && iterations=30  # the paper's Table 6 CEM budget
      args=(src/eval.py --config-name=pusht policy=lewm/weights_epoch_10.pt eval.dataset_name=pusht_expert_train.h5
        ++eval.manifest_path=null solver="$solver" solver.num_samples=300 solver.n_steps="$iterations" objective=goal_mse
        ++callbacks.timer.enabled=true ++callbacks.timer.warmup_solves=1 "${quiet[@]}" "${common[@]}") ;;
    leflow_cem|leflow_icem)
      local refine=(solver.icem_alpha=0.1 solver.icem_noise_beta=2.0 solver.icem_n_elite_keep=5)
      [[ $arm == leflow_cem ]] && refine=(solver.icem_alpha=0 solver.icem_noise_beta=0 solver.icem_n_elite_keep=0)
      args=(src/leflow/eval.py --config-name=pusht.yaml solver=latent_flow_icem objective=rollout_goal
        policy="latent_planner/latent_planner_pusht_h${h}_ep20/latent_planner_epoch_10.pt" ++objective.window="$h"
        ++observer.latent_drift=false solver.num_samples=64 solver.flow_steps=16 ++solver.warm_samples=16
        ++solver.warm_t0=0.6 ++solver.clamp_actions=true solver.prior_mode=fitted solver.icem_num_samples=300
        solver.icem_n_steps=5 solver.icem_var_scale=1.0 solver.prior_elites=30 "${refine[@]}" timing=true
        eval.goal_offset_steps=25 "${common[@]}") ;;
  esac
  echo "cd $PG && PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PG_PYTHON ${args[*]}"
}

run_config() {  # $1 gpu, $2 worker index, $3 arm, $4 h, $5 optional output root, $6 envs per call, $7 proposals N
  local batch=${6:-1} samples=${7:-}
  local tag=h$4
  (( batch > 1 )) && tag+=_b$batch
  [[ -n $samples ]] && tag+=_n$samples
  local gpu=$1 worker=$2 arm=$3 h=$4 root=${5:-$ROOT/$3/$tag} seeds=("${SEEDS[@]}")
  [[ $arm == throughput ]] && seeds=("${THROUGHPUT_SEEDS[@]}")
  local cores="$((CORE_BASE + 8 * worker))-$((CORE_BASE + 8 * worker + 7))"
  for seed in "${seeds[@]}"; do
    local out=$root/$seed cmd
    case $arm in
      cem|icem_faithful|cem_300x30|leflow_cem|leflow_icem) cmd=$(pg_cmd "$arm" "$h" "$seed" "$out" "$batch") ;;
      *) cmd=$(lewam_cmd "$arm" "$h" "$seed" "$out" "$batch" "$samples") ;;
    esac
    if [[ $TEST_RUN == 1 ]]; then echo "$cmd"; return; fi
    [[ -f $out/timing.json ]] && continue
    mkdir -p "$out"
    printf '%s\t%s\t%s\n' "$gpu" "$cores" "$(date +%FT%T)" > "$out/gpu.txt"
    echo "===== [worker $worker] $(date +%T) $arm h=$h seed=$seed on $gpu cores $cores ====="
    env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MUJOCO_GL=egl \
      taskset -c "$cores" bash -c "$cmd" > "$out/log.txt" 2>&1 \
      || printf '%s\t%s\t%s\n' "$arm" "$h" "$seed" >> "$FAILED"
  done
}

if [[ $TEST_RUN == 1 ]]; then
  while IFS=$'\t' read -r arm h batch samples; do run_config "${gpus[0]}" 0 "$arm" "$h" "" "$batch" "$samples"; done < "$JOBS"
  exit 0
fi

if [[ ${CALIBRATE:-0} == 1 ]]; then
  for i in "${!gpus[@]}"; do
    run_config "${gpus[$i]}" "$i" cem 5 "$LW/data/${GROUP}/p2/latency/calibration/${gpus[$i]}" \
      > "logs/${GROUP}/latency_calibration_${i}.log" 2>&1 &
  done
  wait
  echo "calibration done"
  exit 0
fi

worker() {  # $1 gpu, $2 worker index
  local job
  while job=$(flock "$QUEUE.lock" bash -c 'line=$(head -n1 "$1"); [[ -n $line ]] && sed -i 1d "$1"; echo "$line"' _ "$QUEUE") \
      && [[ -n $job ]]; do
    IFS=$'\t' read -r arm h batch samples <<< "$job"
    run_config "$1" "$2" "$arm" "$h" "" "$batch" "$samples"
  done
}

cp "$JOBS" "$QUEUE"
for i in "${!gpus[@]}"; do
  worker "${gpus[$i]}" "$i" > "logs/${GROUP}/latency_worker_${i}.log" 2>&1 &
  sleep 2
done
echo "launched ${#gpus[@]} latency workers, $(wc -l < "$QUEUE") configs"
[[ "${WAIT:-0}" == 1 ]] && wait
