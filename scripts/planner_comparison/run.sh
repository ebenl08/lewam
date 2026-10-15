#!/bin/bash
#
# LeWAM gradient planner, P1 of notes/pending_implementation/lewam-planner-comparison/. Workers pop jobs
# (phase env arm h rh seed, from make_jobs.py) from a shared queue; a GPU listed k times runs k workers.
#   repro: native paper cell, paper protocol (num_eval 50, the task config's offset and budget).
#   grid:  arm x (h, rh) cell on the PushT protocol of the stable-worldmodel baselines: pusht_expert_train.h5,
#          offset 25, budget 50, 30 episodes. Cell: eval.rollout_steps=h, eval.exec_actions=5*rh.
# Arms: native = horizon_schedule adaptive (LeWAM default); grid_w1 = constant; grid_wH = constant + cost window h.
# Run from playground/lewam/:
#   python scripts/planner_comparison/make_jobs.py --phase repro > logs/planner_comparison/jobs.tsv
#   GPUS="1 1 1 1 3 3 3 3" bash scripts/planner_comparison/run.sh   (nvidia-smi indices, never 2 or 8)
#   TEST_RUN=1              print the command of the first job only

PYTHON=${PYTHON:-/home/eben/.conda/envs/lewam/bin/python}
GROUP=${GROUP:-planner_comparison}
JOBS=${JOBS:-logs/${GROUP}/jobs.tsv}
QUEUE=${QUEUE:-logs/${GROUP}/queue.tsv}
FAILED=logs/${GROUP}/failed.tsv
TEST_RUN=${TEST_RUN:-0}
# Reacher repro: extra overrides of the reacher task config (P1 open question 1)
REACHER_EXTRA=${REACHER_EXTRA:-eval.dataset_name=reacher}
mkdir -p logs/${GROUP}

gpu_uuid() {  # UUID of nvidia-smi GPU $1; CUDA indices skip smi GPU 2, so pin by UUID
  local uuid
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$1" '$1 == i {print $2}')
  [[ -n $uuid ]] || { echo "no nvidia-smi GPU '$1'" >&2; return 1; }
  echo "$uuid"
}

read -ra gpus <<< "${GPUS:?set GPUS to idle nvidia-smi GPU indices, repeated for several workers per GPU}"
if (( $(printf '%s\n' "${gpus[@]}" | sort -u | wc -l) > 4 )); then
  echo "more than 4 distinct GPUs in GPUS" >&2; exit 1
fi
for i in "${!gpus[@]}"; do
  [[ ${gpus[$i]} == 2 || ${gpus[$i]} == 8 ]] && { echo "nvidia-smi GPU ${gpus[$i]} is off limits" >&2; exit 1; }
  gpus[$i]=$(gpu_uuid "${gpus[$i]}") || exit 1
done

run_job() {  # $1 gpu, then the job columns
  local gpu=$1 phase=$2 env=$3 arm=$4 h=$5 rh=$6 seed=$7
  local plan=(eval.mode=lewam_plan eval.plan_mode=grad eval.grad_all_k=true eval.grad_tr=0.01)
  local task out
  if [[ $phase == repro ]]; then
    out=data/${GROUP}/p1/repro/${env}/${seed}
    task=(--config-name="$env" policy="lewam-${env}" eval.num_eval=50)
    [[ $env == pusht ]] && task+=(eval.dataset_name=pusht_expert_train)  # byte-identical to the HF pusht data
    [[ $env == reacher ]] && task+=($REACHER_EXTRA)
  else
    out=data/${GROUP}/p1/runs/${env}/${arm}/h${h}_rh${rh}/${seed}
    task=(--config-name=pusht policy=lewam-pusht eval.dataset_name=pusht_expert_train eval.num_eval=30
      eval.save_video=false eval.rollout_steps="$h" eval.exec_actions=$((5 * rh)))
    case $arm in
      native) ;;
      grid_w1) task+=(eval.horizon_schedule=constant) ;;
      grid_wH) task+=(eval.horizon_schedule=constant eval.plan_cost_window="$h") ;;
      *) echo "unknown arm $arm" >&2; return 1 ;;
    esac
  fi

  local cmd=(env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MUJOCO_GL=egl
    "$PYTHON" scripts/eval_lewam.py "${task[@]}" "${plan[@]}" seed="$seed" output_dir="$PWD/$out")
  if [[ $TEST_RUN == 1 ]]; then
    printf '%q ' "${cmd[@]}"; echo
    return
  fi
  mkdir -p "$out"
  "${cmd[@]}" > "$out/log.txt" 2>&1 || printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$phase" "$env" "$arm" "$h" "$rh" "$seed" >> "$FAILED"
}

worker() {  # $1 gpu, $2 worker index
  local gpu=$1 job
  while job=$(flock "$QUEUE.lock" bash -c 'line=$(head -n1 "$1"); [[ -n $line ]] && sed -i 1d "$1"; echo "$line"' _ "$QUEUE") \
      && [[ -n $job ]]; do
    IFS=$'\t' read -r phase env arm h rh seed <<< "$job"
    echo "===== [worker $2] $(date +%T) $phase $env $arm h=$h rh=$rh seed=$seed on $gpu ====="
    run_job "$gpu" "$phase" "$env" "$arm" "$h" "$rh" "$seed"
  done
}

cp "$JOBS" "$QUEUE"
if [[ $TEST_RUN == 1 ]]; then
  IFS=$'\t' read -r phase env arm h rh seed < "$QUEUE"
  run_job "${gpus[0]}" "$phase" "$env" "$arm" "$h" "$rh" "$seed"
  exit 0
fi
for i in "${!gpus[@]}"; do
  worker "${gpus[$i]}" "$i" > "logs/${GROUP}/worker_${i}.log" 2>&1 &
  sleep 2
done
echo "launched ${#gpus[@]} workers, $(wc -l < "$QUEUE") jobs (${GROUP})"
[[ "${WAIT:-0}" == 1 ]] && wait
