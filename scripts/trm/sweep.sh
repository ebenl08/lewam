#!/bin/bash
#
# TRM head hyperparameter sweep (Δ ≤ 50, 80/10/10 train/val/test episodes) or the wandb rerun of the existing
# baseline heads, for one env on one UUID-pinned GPU. Every run logs to wandb (ebenl_08/lewam-trm). Run from
# playground/lewam inside tmux:
#   GPU=4 CORES=400-407 bash scripts/trm/sweep.sh tworoom sweep
#   GPU=4 CORES=400-407 bash scripts/trm/sweep.sh tworoom baseline
# sweep:    heads data/trm/sweep/<env>/<run>/, logs logs/trm/<env>/sweep/<run>.log, wandb group <env>_sweep_d50
# baseline: heads data/trm/heads_wandb/<run>/, logs logs/trm/<env>/baseline_wandb/<run>.log, wandb group <env>_baseline
# Runs whose metrics.json exists are skipped, so a stopped sweep resumes.

LW=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=/home/eben/.conda/envs/lewam/bin/python
ENV_NAME=${1:?env: tworoom | puzzle | pointmaze}
MODE=${2:?mode: sweep | baseline}
GPU=${GPU:?nvidia-smi GPU index}
CORES=${CORES:?cores, e.g. 272-279}
[[ $GPU == 2 || $GPU == 8 ]] && { echo "nvidia-smi GPU $GPU is off limits" >&2; exit 1; }
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$GPU" '$1 == i {print $2}')
[[ -n $UUID ]] || { echo "no nvidia-smi GPU '$GPU'" >&2; exit 1; }

case $ENV_NAME in
  tworoom) DATASET=tworoom.h5; CKPT=lewam-tworoom; CAPS="null 50" ;;
  puzzle) DATASET=puzzle_expert.h5; CKPT=lewam-puzzle; CAPS="null 50 100" ;;
  pointmaze) DATASET=pointmaze_large.h5; CKPT=lewam-pointmaze-large; CAPS="null 50 100" ;;
  *) echo "unknown env $ENV_NAME" >&2; exit 1 ;;
esac
cd "$LW" || exit 1

train() {  # out_root, log dir, run name, wandb group, then hydra overrides
  local out_root=$1 log_dir=$2 name=$3 group=$4
  shift 4
  [[ -f $out_root/$name/metrics.json ]] && { echo "skip $name (done)"; return; }
  mkdir -p "$log_dir"
  env CUDA_VISIBLE_DEVICES="$UUID" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c "$CORES" \
    "$PYTHON" -m lewam.trm.train dataset="$DATASET" checkpoint="$CKPT" output_model_name="$name" \
    out_root="$out_root" hydra.run.dir="$LW/outputs/trm/$(basename "$out_root")/$name" wandb.group="$group" "$@" \
    > "$log_dir/$name.log" 2>&1 || { echo "FAILED $name, see $log_dir/$name.log"; return; }
  echo "$name  $(grep -h 'test:\|final:' "$log_dir/$name.log" | tail -1 | grep -o '\(val\|test\)/spearman=[-0-9.]*')"
}

echo "===== $(date +%T) $ENV_NAME $MODE"
if [[ $MODE == sweep ]]; then
  for lr in 3e-4 1e-3 3e-3; do
    for n_pairs in 100000 1000000; do
      for wd in 1e-4 1e-2; do
        for dropout in 0.0 0.1; do
          for epochs in 10 20; do
            name=lr${lr}_np$((n_pairs / 1000))k_wd${wd}_do${dropout}_ep$epochs
            train "$LW/data/trm/sweep/$ENV_NAME" "$LW/logs/trm/$ENV_NAME/sweep" "$name" "${ENV_NAME}_sweep_d50" \
              sampler.max_delta=50 sampler.test_frac=0.1 optimizer.lr=$lr sampler.n_pairs=$n_pairs \
              optimizer.weight_decay=$wd head.dropout=$dropout trainer.max_epochs=$epochs
          done
        done
      done
    done
  done
elif [[ $MODE == baseline ]]; then
  for cap in $CAPS; do
    tag=""; [[ $cap != null ]] && tag=_d$cap
    for variant in "" _shuff; do
      shuffle=false; [[ $variant == _shuff ]] && shuffle=true
      train "$LW/data/trm/heads_wandb" "$LW/logs/trm/$ENV_NAME/baseline_wandb" "trm_${ENV_NAME}_v1$tag$variant" \
        "${ENV_NAME}_baseline" sampler.max_delta=$cap shuffle_labels=$shuffle
    done
  done
else
  echo "unknown mode $MODE" >&2; exit 1
fi
echo "===== $(date +%T) $ENV_NAME $MODE done"
