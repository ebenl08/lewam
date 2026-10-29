#!/bin/bash
#
# TRM head for one env on LeWAM latents: latent cache (verified), real head, shuffled-label control, imagined-latent
# probe. One UUID-pinned GPU. Run from playground/lewam inside tmux:
#   GPU=5 CORES=272-279 [MAX_DELTA=50] bash scripts/trm/cache_train.sh tworoom [--build-cache]
# The latent cache is built only with --build-cache; otherwise the existing cache is reused.
# MAX_DELTA caps the pair gap (sampler.max_delta) and tags every run and log with _d<MAX_DELTA>.
# Caches: ~/.stable_worldmodel/latents/<dataset>__<checkpoint>.npz; heads: data/trm/heads/trm_<env>_v1[_d50]{,_shuff}/;
# probe: data/trm/probe/trm_<env>_v1[_d50]/; logs: logs/trm/<env>/{cache,train,train_shuff,probe}[_d50].log

LW=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=/home/eben/.conda/envs/lewam/bin/python
ENV_NAME=${1:?env: tworoom | puzzle | pointmaze}
BUILD_CACHE=false
[[ ${2:-} == --build-cache ]] && BUILD_CACHE=true
[[ -n ${2:-} && $BUILD_CACHE == false ]] && { echo "unknown option '$2'" >&2; exit 1; }
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
TAG="" DELTA_ARGS=""
[[ -n ${MAX_DELTA:-} ]] && { TAG=_d$MAX_DELTA; DELTA_ARGS="sampler.max_delta=$MAX_DELTA"; }
RUN=trm_${ENV_NAME}_v1$TAG
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

if $BUILD_CACHE; then
  run cache -m lewam.trm.cache --dataset "$DATASET" --checkpoint "$CKPT" $SUBSET --verify
  grep -h "fidelity\|written" "$LOGS/cache.log" | cut -c1-160
else
  CACHE=${STABLEWM_HOME:-$HOME/.stable_worldmodel}/latents/${DATASET}__$CKPT.npz
  [[ -f $CACHE ]] || { echo "no latent cache $CACHE; pass --build-cache" >&2; exit 1; }
  echo "===== $(date +%T) $ENV_NAME reusing $CACHE"
fi
for variant in "" "_shuff"; do
  shuffle=false; [[ $variant == _shuff ]] && shuffle=true
  run "train$variant$TAG" -m lewam.trm.train dataset="$DATASET" checkpoint="$CKPT" output_model_name="$RUN$variant" \
    shuffle_labels=$shuffle $DELTA_ARGS hydra.run.dir="$LW/outputs/trm/$RUN$variant"
  grep -h "final:" "$LOGS/train$variant$TAG.log" | sed 's/.*final: //' | cut -c1-160
done
run "probe$TAG" -m lewam.trm.probe_imagined --dataset "$DATASET" --checkpoint "$CKPT" \
  --head "$LW/data/trm/heads/$RUN/weights_epoch_20.pt"
grep -h "rho_" "$LOGS/probe$TAG.log" | sed 's/.*| //'
echo "===== $(date +%T) $ENV_NAME done"
