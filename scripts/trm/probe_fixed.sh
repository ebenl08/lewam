#!/bin/bash
#
# Rerun the imagined-latent probe (after the bucket-order fix) for the v1 and v1_d50 heads of one env.
# Run from playground/lewam inside tmux:
#   GPU=5 CORES=272-279 bash scripts/trm/probe_fixed.sh tworoom
# Probe: data/trm/probe/<head run>_fixed/; logs: logs/trm/<env>/probe_fixed{,_d50}.log

LW=$(cd "$(dirname "$0")/../.." && pwd)
PYTHON=/home/eben/.conda/envs/lewam/bin/python
ENV_NAME=${1:?env: tworoom | puzzle | pointmaze}
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
cd "$LW" || exit 1

for TAG in "" _d50; do
  RUN=trm_${ENV_NAME}_v1$TAG
  log=$LW/logs/trm/$ENV_NAME/probe_fixed$TAG.log
  echo "===== $(date +%T) $ENV_NAME probe $RUN"
  env CUDA_VISIBLE_DEVICES="$UUID" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c "$CORES" \
    "$PYTHON" -m lewam.trm.probe_imagined --dataset "$DATASET" --checkpoint "$CKPT" \
    --head "$LW/data/trm/heads/$RUN/weights_epoch_20.pt" --out "$LW/data/trm/probe/${RUN}_fixed" \
    > "$log" 2>&1 || { echo "FAILED, see $log"; exit 1; }
  grep -h "rho_" "$log" | sed 's/.*| //'
done
echo "===== $(date +%T) $ENV_NAME done"
