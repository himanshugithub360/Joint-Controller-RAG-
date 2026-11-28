#!/usr/bin/env bash
# Stage 4 on Kaggle, one round per call, from the repo root:
#   bash kaggle/run_stage4.sh 1        # then 2, then 3 (each after the workstation step)
# Needs Accelerator "GPU T4 x2" and Internet on. Writes /kaggle/working/stage4_r<N>.zip.
#
# One process per tier (frees each model's GPU memory). Tiers that need one GPU run two at
# a time, one per T4 (CUDA_VISIBLE_DEVICES), split into two queues of similar model size;
# tiers that need both GPUs run afterwards. Outputs do not depend on this scheduling.
set -euo pipefail
ROUND=${1:?round number 1, 2 or 3}
JOBS=kaggle/inputs/stage4_r${ROUND}.jsonl.gz
META=${JOBS%.jsonl.gz}.meta.json
OUT=/kaggle/working/stage4_r${ROUND}
PY=/tmp/vllm-env/bin/python

nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
git log --oneline -1
test -f "$JOBS" || { echo "missing $JOBS: pull the repo after the workstation step"; exit 1; }

if [ ! -x "$PY" ]; then
  pip install -q uv
  uv venv -q --python 3.12 /tmp/vllm-env
  uv pip install -q --python "$PY" -r envs/kaggle-requirements.txt
fi
"$PY" -c "import vllm, torch; print('vllm', vllm.__version__, '| cuda', torch.cuda.is_available())"

rm -rf "$OUT" "$OUT.zip"
mkdir -p "$OUT/logs"
# prints three lines: GPU-0 queue, GPU-1 queue, two-GPU tiers
mapfile -t Q < <("$PY" - "$META" <<'EOF'
import json, re, sys
m = json.load(open(sys.argv[1]))
size = lambda t: float(re.search(r"([\d.]+)B", m["models"][t]).group(1))
one = sorted([t for t in m["run_tiers"] if m["tensor_parallel"][t] == 1], key=size, reverse=True)
qs, w = [[], []], [0.0, 0.0]
for t in one:
    i = w.index(min(w)); qs[i].append(t); w[i] += size(t)
print(" ".join(qs[0])); print(" ".join(qs[1]))
print(" ".join(t for t in m["run_tiers"] if m["tensor_parallel"][t] > 1))
EOF
)
echo "GPU 0: ${Q[0]:-} | GPU 1: ${Q[1]:-} | both GPUs: ${Q[2]:-}"

run_queue() {   # $1 = GPU id, rest = tiers
  local gpu=$1; shift
  for tier in "$@"; do
    CUDA_VISIBLE_DEVICES=$gpu "$PY" kaggle/run_jobs.py --jobs "$JOBS" --tier "$tier" --out-dir "$OUT" \
      > "$OUT/logs/$tier.log" 2>&1 || { echo "tier $tier FAILED"; tail -30 "$OUT/logs/$tier.log"; return 1; }
    grep -E "^(tier|  [0-9]+ prompts)" "$OUT/logs/$tier.log"
  done
}
run_queue 0 ${Q[0]:-} & P0=$!
run_queue 1 ${Q[1]:-} & P1=$!
wait $P0 && wait $P1
for tier in ${Q[2]:-}; do
  "$PY" kaggle/run_jobs.py --jobs "$JOBS" --tier "$tier" --out-dir "$OUT" > "$OUT/logs/$tier.log" 2>&1 \
    || { echo "tier $tier FAILED"; tail -30 "$OUT/logs/$tier.log"; exit 1; }
  grep -E "^(tier|  [0-9]+ prompts)" "$OUT/logs/$tier.log"
done

(cd "$OUT" && zip -q -r "$OUT.zip" .)
ls -l "$OUT" "$OUT.zip"
echo "STAGE4 ROUND ${ROUND} DONE"
