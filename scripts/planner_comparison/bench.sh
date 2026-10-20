#!/bin/bash
#
# P2c capture-and-replay benchmark runner (PushT h5, P2 planner settings). One line of ARMS per run, one job at a
# time on one UUID-pinned GPU and 8 pinned cores. MODE=time: timed calls + one counted call per batch (FLOPs, bytes,
# memory) -> $OUT/<arm>/time.json. MODE=nsys: timed calls only, under nsys -> $OUT/<arm>/nsys.json + .nsys-rep/.sqlite.
# Run from playground/lewam:
#   ARMS="cem leflow_cem grid_w1 native" MODE=time GPU=5 BATCHES="1 4 16 64 256 1024" \
#     bash scripts/planner_comparison/bench.sh
#   TEST_RUN=1 ...   print the commands, run nothing

LW=$(cd "$(dirname "$0")/../.." && pwd)
PG=$(cd "$LW/../stable-worldmodel" && pwd)
ARMS=${ARMS:?arms: cem leflow_cem grid_w1 native}
MODE=${MODE:-time}
GPU=${GPU:?one nvidia-smi GPU index}
CORES=${CORES:-256-263}
BATCHES=${BATCHES:-1 4 16 64 256 1024}
STEPS_LEFT=${STEPS_LEFT:-5 4 3 2 1}
CAPTURE_ENVS=${CAPTURE_ENVS:-16}
SEED=${SEED:-10247}
OUT=${OUT:-$LW/data/planner_comparison/p2/bench}
TEST_RUN=${TEST_RUN:-0}
[[ $GPU == 2 || $GPU == 8 ]] && { echo "nvidia-smi GPU $GPU is off limits" >&2; exit 1; }
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$GPU" '$1 == i {print $2}')
[[ -n $UUID ]] || { echo "no nvidia-smi GPU '$GPU'" >&2; exit 1; }

LEWAM_ARGS="--config-name=pusht policy=lewam-pusht eval.dataset_name=pusht_expert_train eval.save_video=false \
eval.mode=lewam_plan eval.plan_mode=grad eval.grad_all_k=true eval.grad_tr=0.01 eval.exec_actions=5 \
eval.rollout_steps=5 eval.num_eval=$CAPTURE_ENVS seed=$SEED"
PG_COMMON="eval.num_eval=$CAPTURE_ENVS eval.eval_budget=50 ++eval.save_episode_video=false plan_config.horizon=5 \
plan_config.receding_horizon=1 plan_config.action_block=5 solver.batch_size=$CAPTURE_ENVS seed=$SEED"
CEM_ARGS="--config-name=pusht policy=lewm/weights_epoch_10.pt eval.dataset_name=pusht_expert_train.h5 \
++eval.manifest_path=null solver=cem solver.num_samples=300 solver.n_steps=10 objective=goal_mse \
++callbacks.timer.enabled=false ++callbacks.solver_progress.enabled=false ++callbacks.observers.rollout=false \
++callbacks.observers.trajectory=false ++callbacks.observers.reward_cost=false ++callbacks.observers.embedding_mse=false \
++callbacks.observers.pca_features=false ++callbacks.observers.goal_mismatch=false ++decoder.enabled=false $PG_COMMON"
LEFLOW_ARGS="--config-name=pusht.yaml solver=latent_flow_icem objective=rollout_goal \
policy=latent_planner/latent_planner_pusht_h5_ep20/latent_planner_epoch_10.pt ++objective.window=5 \
++observer.latent_drift=false solver.num_samples=64 solver.flow_steps=16 ++solver.warm_samples=16 ++solver.warm_t0=0.6 \
++solver.clamp_actions=true solver.prior_mode=fitted solver.icem_num_samples=300 solver.icem_n_steps=5 \
solver.icem_var_scale=1.0 solver.prior_elites=30 solver.icem_alpha=0 solver.icem_noise_beta=0 \
solver.icem_n_elite_keep=0 timing=false eval.goal_offset_steps=25 $PG_COMMON"

bench_cmd() {  # arm out_json nsys_flag
  local arm=$1 out=$2 flag=$3 scratch="$(dirname "$2")/eval_scratch"
  case $arm in
    cem) echo "cd $PG && PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/eben/.conda/envs/swm/bin/python \
$LW/scripts/planner_comparison/bench_pg.py --entry src/eval.py --out $out --batches $BATCHES $flag -- $CEM_ARGS \
+output_folder=$scratch" ;;
    leflow_cem) echo "cd $PG && PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/eben/.conda/envs/swm/bin/python \
$LW/scripts/planner_comparison/bench_pg.py --entry src/leflow/eval.py --out $out --batches $BATCHES $flag -- \
$LEFLOW_ARGS +output_folder=$scratch" ;;
    grid_w1) echo "cd $LW && /home/eben/.conda/envs/lewam/bin/python scripts/planner_comparison/bench_lewam.py --out $out \
--batches $BATCHES $flag -- $LEWAM_ARGS eval.horizon_schedule=constant" ;;
    native) echo "cd $LW && /home/eben/.conda/envs/lewam/bin/python scripts/planner_comparison/bench_lewam.py --out $out \
--batches $BATCHES --steps-left $STEPS_LEFT $flag -- $LEWAM_ARGS" ;;
    *) echo "unknown arm $arm" >&2; return 1 ;;
  esac
}

for arm in $ARMS; do
  dir=$OUT/$arm
  if [[ $MODE == nsys ]]; then cmd=$(bench_cmd "$arm" "$dir/nsys.json" --nsys); else cmd=$(bench_cmd "$arm" "$dir/time.json"); fi
  if [[ $TEST_RUN == 1 ]]; then echo "$cmd"; continue; fi
  mkdir -p "$dir"
  printf '%s\t%s\t%s\n' "$UUID" "$CORES" "$(date +%FT%T)" > "$dir/gpu_$MODE.txt"
  echo "===== $(date +%T) $arm $MODE -> $dir"
  run=(env CUDA_VISIBLE_DEVICES="$UUID" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MUJOCO_GL=egl taskset -c "$CORES")
  if [[ $MODE == nsys ]]; then
    "${run[@]}" nsys profile --trace=cuda,nvtx,cublas,cudnn --sample=none --cpuctxsw=none --force-overwrite=true \
      --output="$dir/nsys_bench_$arm" bash -c "$cmd" > "$dir/log_nsys.txt" 2>&1 || echo "FAILED $arm $MODE"
    nsys export --type=sqlite --force-overwrite=true --output="$dir/nsys_bench_$arm.sqlite" \
      "$dir/nsys_bench_$arm.nsys-rep" >> "$dir/log_nsys.txt" 2>&1 || echo "EXPORT FAILED $arm"
  else
    "${run[@]}" bash -c "$cmd" > "$dir/log_time.txt" 2>&1 || echo "FAILED $arm $MODE"
  fi
done
echo "===== $(date +%T) bench done"
