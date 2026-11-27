#!/usr/bin/env bash
# Stage 4 on Kaggle, one round per call, from the repo root:
#   bash kaggle/run_stage4.sh 1        # then 2, then 3 (each after the workstation step)
# Needs Accelerator "GPU T4 x2" and Internet on. Writes /kaggle/working/stage4_r<N>.zip.
set -euo pipefail
ROUND=${1:?round number 1, 2 or 3}
JOBS=kaggle/inputs/stage4_r${ROUND}.jsonl.gz
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
TIERS=$("$PY" -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]))['run_tiers']))" "${JOBS%.jsonl.gz}.meta.json")
for tier in $TIERS; do         # one process per tier frees each model's GPU memory
  "$PY" kaggle/run_jobs.py --jobs "$JOBS" --tier "$tier" --out-dir "$OUT"
done

(cd "$OUT" && zip -q -r "$OUT.zip" .)
ls -l "$OUT" "$OUT.zip"
echo "STAGE4 ROUND ${ROUND} DONE"
