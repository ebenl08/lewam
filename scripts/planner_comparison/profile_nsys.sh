#!/bin/bash
#
# Nsight Systems profile of PushT latency runs: is the GPU busy, how many kernels and CPU-GPU syncs per plan.
# Each line of CMDS is one eval command from `latency.sh` with TEST_RUN=1; it runs under `nsys profile` on one GPU
# (UUID-pinned) and 8 pinned cores, then `nsys stats` writes kernel / API / memcpy summaries next to the run.
# Run from playground/lewam/:
#   CMDS=logs/planner_comparison/profile_cmds.txt GPU=6 bash scripts/planner_comparison/profile_nsys.sh

CMDS=${CMDS:?file of eval commands, one per line}
GPU=${GPU:?one nvidia-smi GPU index}
CORES=${CORES:-256-263}
ALLOW_GPU=${ALLOW_GPU:-}  # one-off user grant for an off-limits GPU (2 or 8), e.g. ALLOW_GPU=8
[[ ($GPU == 2 || $GPU == 8) && $GPU != "$ALLOW_GPU" ]] && { echo "nvidia-smi GPU $GPU is off limits" >&2; exit 1; }
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$GPU" '$1 == i {print $2}')
[[ -n $UUID ]] || { echo "no nvidia-smi GPU '$GPU'" >&2; exit 1; }

while IFS= read -r cmd; do
  out=$(grep -oE '(output_dir|\+output_folder)=[^ ]+' <<< "$cmd" | cut -d= -f2)
  name=nsys_$(basename "$(dirname "$(dirname "$out")")")_$(basename "$(dirname "$out")")_$(basename "$out")
  cmd=${cmd/&& /\&\& exec env }  # python replaces the bash wrapper (escaped: bash 5.2 expands & to the match)
  mkdir -p "$out"
  printf '%s\t%s\t%s\n' "$UUID" "$CORES" "$(date +%FT%T)" > "$out/gpu.txt"
  echo "===== $(date +%T) $out/$name"
  env CUDA_VISIBLE_DEVICES="$UUID" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MUJOCO_GL=egl taskset -c "$CORES" \
    nsys profile --trace=cuda,nvtx,cublas,cudnn --sample=none --cpuctxsw=none --force-overwrite=true \
      --output="$out/$name" bash -c "$cmd" > "$out/log.txt" 2>&1 || echo "FAILED $out"
  nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,nvtx_sum,nvtx_kern_sum --format csv \
    --force-export=true --force-overwrite=true --output="$out/nsys" "$out/$name.nsys-rep" > "$out/nsys_stats.log" 2>&1 || echo "STATS FAILED $out"
done < "$CMDS"
echo "===== $(date +%T) profiling done"
