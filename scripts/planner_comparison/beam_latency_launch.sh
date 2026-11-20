#!/bin/bash
#
# Beam-tree latency on PushT beside the P2 arms, on one idle GPU, run from playground/lewam/ inside tmux:
#   GPU=8 bash scripts/planner_comparison/beam_latency_launch.sh   (LOAD_MAX=100: waits for server load average below it)
# Finished runs (timing.json present) are skipped, so a relaunch resumes.
# 1. CEM h5 calibration on GPU (seeds 10247 15783), to compare with the P2 GPUs in p2/latency/calibration/
# 2. end-to-end: 3 beam arms x h {2, 5, 10} x 10 seeds, batch 1 (p2/latency/pusht/<arm>/h<h>/)
# 3. amortized: 3 beam arms + icem_faithful + leflow_icem x B {2..64} x 3 seeds, h5 (p2/sweeps/pusht/batch/)
# 4. plots and table: analysis/planner_comparison/beam_latency.py -> data/planner_comparison/analysis/p2/beam_latency*

GPU=${GPU:?nvidia-smi GPU index}
L=logs/planner_comparison
uuid=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$uuid"; then
  echo "GPU $GPU is busy" >&2
  exit 1
fi

LOAD_MAX=${LOAD_MAX:-100}  # host launch time inflates under server CPU load: wait for 3 calm minutes in a row
calm=0
while (( calm < 3 )); do
  load=$(cut -d' ' -f1 /proc/loadavg)
  if (( ${load%.*} < LOAD_MAX )); then calm=$((calm + 1)); else calm=0; fi
  echo "$(date +%T) load $load, calm minutes $calm"
  (( calm < 3 )) && sleep 60
done

echo "$(date +%T) calibration on GPU $GPU"
GPUS=$GPU CALIBRATE=1 TAG=beam_latency SEED_LIST="10247 15783" bash scripts/planner_comparison/latency.sh
echo "$(date +%T) end-to-end"
GPUS=$GPU WAIT=1 TAG=beam_latency_e2e JOBS=$L/beam_latency_e2e_jobs.tsv bash scripts/planner_comparison/latency.sh
echo "$(date +%T) e2e done; failed: $(cat $L/beam_latency_e2e_failed.tsv 2>/dev/null | wc -l)"
GPUS=$GPU WAIT=1 TAG=beam_latency_batch JOBS=$L/beam_latency_batch_jobs.tsv \
  ROOT=$PWD/data/planner_comparison/p2/sweeps/pusht/batch SEED_LIST="10247 15783 56783" \
  bash scripts/planner_comparison/latency.sh
echo "$(date +%T) batch done; failed: $(cat $L/beam_latency_batch_failed.tsv 2>/dev/null | wc -l)"
/home/eben/.conda/envs/lewam/bin/python analysis/planner_comparison/beam_latency.py > $L/beam_latency_analysis.log 2>&1
echo "$(date +%T) ALL DONE"
