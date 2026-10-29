#!/bin/bash
#
# For the chosen sweep config of one env: the shuffled-label control, then the imagined-latent probe on the head
# and on the control. One UUID-pinned GPU. Run from playground/lewam inside tmux:
#   GPU=4 CORES=400-407 bash scripts/trm/best_control_probe.sh tworoom
# control head: data/trm/sweep/<env>/<config>_shuff/; probes: data/trm/probe/trm_<env>_best{,_shuff}/;
# logs: logs/trm/<env>/best/{train_shuff,probe,probe_shuff}.log

LW=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=/home/eben/.conda/envs/lewam/bin/python
ENV_NAME=${1:?env: tworoom | puzzle | pointmaze}
CONFIG=lr1e-3_np1000k_wd1e-2_do0.1_ep20
GPU=${GPU:?nvidia-smi GPU index}
CORES=${CORES:?cores, e.g. 272-279}
[[ $GPU == 2 || $GPU == 8 ]] && { echo "nvidia-smi GPU $GPU is off limits" >&2; exit 1; }
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$GPU" '$1 == i {print $2}')
[[ -n $UUID ]] || { echo "no nvidia-smi GPU '$GPU'" >&2; exit 1; }

case $ENV_NAME in
  tworoom) DATASET=tworoom.h5; CKPT=lewam-tworoom ;;
  puzzle) DATASET=puzzle_expert.h5; CKPT=lewam-puzzle ;;
  pointmaze) DATASET=pointmaze_large.h5; CKPT=lewam-pointmaze-large ;;
  *) echo "unknown env $ENV_NAME" >&2; exit 1 ;;
esac
SWEEP=$LW/data/trm/sweep/$ENV_NAME
LOGS=$LW/logs/trm/$ENV_NAME/best
mkdir -p "$LOGS"
cd "$LW" || exit 1

run() {  # log name, then the python args
  local log=$LOGS/$1.log
  shift
  echo "===== $(date +%T) $ENV_NAME $(basename "$log" .log)"
  env CUDA_VISIBLE_DEVICES="$UUID" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c "$CORES" \
    "$PYTHON" "$@" > "$log" 2>&1 || { echo "FAILED, see $log"; exit 1; }
}

run train_shuff -m lewam.trm.train dataset="$DATASET" checkpoint="$CKPT" output_model_name="${CONFIG}_shuff" \
  out_root="$SWEEP" hydra.run.dir="$LW/outputs/trm/$ENV_NAME/${CONFIG}_shuff" wandb.group="${ENV_NAME}_sweep_d50" \
  sampler.max_delta=50 sampler.test_frac=0.1 optimizer.lr=1e-3 sampler.n_pairs=1000000 optimizer.weight_decay=1e-2 \
  head.dropout=0.1 trainer.max_epochs=20 shuffle_labels=true
grep -h "test:" "$LOGS/train_shuff.log" | sed 's/.*test: //' | cut -c1-120
for variant in "" _shuff; do
  run "probe$variant" -m lewam.trm.probe_imagined --dataset "$DATASET" --checkpoint "$CKPT" \
    --head "$SWEEP/$CONFIG$variant/weights_epoch_20.pt" --out "$LW/data/trm/probe/trm_${ENV_NAME}_best$variant"
  grep -h "rho_" "$LOGS/probe$variant.log" | sed 's/.*| //'
done
echo "===== $(date +%T) $ENV_NAME done"
