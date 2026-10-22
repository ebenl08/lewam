#!/bin/bash
#
# TRM head for one env on LeWAM latents: latent cache (verified), real head, shuffled-label control, imagined-latent
# probe. One UUID-pinned GPU. Run from playground/lewam inside tmux:
#   GPU=5 CORES=272-279 bash scripts/trm/cache_train.sh tworoom
# Caches: ~/.stable_worldmodel/latents/<dataset>__<checkpoint>.npz; heads: data/trm/heads/trm_<env>_v1{,_shuff}/;
# probe: data/trm/probe/trm_<env>_v1/; logs: logs/trm/<env>/{cache,train,train_shuff,probe}.log

LW=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=/home/eben/.conda/envs/lewam/bin/python
ENV_NAME=${1:?env: tworoom | puzzle | pointmaze}
GPU=${GPU:?nvidia-smi GPU index}
CORES=${CORES:?cores, e.g. 272-279}
[[ $GPU == 2 || $GPU == 8 ]] && { echo "nvidia-smi GPU $GPU is off limits" >&2; exit 1; }
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$GPU" '$1 == i {print $2}')
[[ -n $UUID ]] || { echo "no nvidia-smi GPU '$GPU'" >&2; exit 1; }

case $ENV_NAME in
  tworoom) DATASET=tworoom.h5; CKPT=lewam-tworoom; SUBSET="" ;;
  puzzle) DATASET=puzzle_expert.h5; CKPT=lewam-puzzle; SUBSET="--n-episodes 5000 --episode-seed 0" ;;
  pointmaze) DATASET=pointmaze_large.h5; CKPT=lewam-pointmaze-large; SUBSET="" ;;
  *) echo "unknown env $ENV_NAME" >&2; exit 1 ;;
esac
RUN=trm_${ENV_NAME}_v1
LOGS=$LW/logs/trm/$ENV_NAME
mkdir -p "$LOGS"
cd "$LW" || exit 1

run() {  # log name, then the python args
  local log=$LOGS/$1.log
  shift
  echo "===== $(date +%T) $ENV_NAME $(basename "$log" .log)"
  env CUDA_VISIBLE_DEVICES="$UUID" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c "$CORES" \
    "$PYTHON" "$@" > "$log" 2>&1 || { echo "FAILED, see $log"; exit 1; }
}

run cache -m lewam.trm.cache --dataset "$DATASET" --checkpoint "$CKPT" $SUBSET --verify
grep -h "fidelity\|written" "$LOGS/cache.log" | cut -c1-160
for variant in "" "_shuff"; do
  shuffle=false; [[ $variant == _shuff ]] && shuffle=true
  run "train$variant" -m lewam.trm.train dataset="$DATASET" checkpoint="$CKPT" output_model_name="$RUN$variant" \
    shuffle_labels=$shuffle hydra.run.dir="$LW/outputs/trm/$RUN$variant"
  grep -h "final:" "$LOGS/train$variant.log" | sed 's/.*final: //' | cut -c1-160
done
run probe -m lewam.trm.probe_imagined --dataset "$DATASET" --checkpoint "$CKPT" \
  --head "$LW/data/trm/heads/$RUN/weights_epoch_20.pt"
grep -h "rho_" "$LOGS/probe.log" | sed 's/.*| //'
echo "===== $(date +%T) $ENV_NAME done"
