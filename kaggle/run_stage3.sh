#!/usr/bin/env bash
# Stage 3 on Kaggle in one command (the same steps as kaggle/stage3.ipynb), from the repo root:
#   bash kaggle/run_stage3.sh
# Needs Accelerator "GPU T4 x2" and Internet on. Writes /kaggle/working/stage3_out/ and stage3_out.zip.
set -euo pipefail
OUT=/kaggle/working/stage3_out
PY=/tmp/vllm-env/bin/python

nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
git log --oneline -1

if [ ! -x "$PY" ]; then
  pip install -q uv
  uv venv -q --python 3.12 /tmp/vllm-env
  uv pip install -q --python "$PY" -r envs/kaggle-requirements.txt
fi
"$PY" -c "import vllm, torch; print('vllm', vllm.__version__, '| cuda', torch.cuda.is_available())"

rm -rf "$OUT" /kaggle/working/stage3_out.zip
for tier in S M L; do          # one process per tier frees each model's GPU memory
  "$PY" kaggle/stage3_generate.py --tier "$tier" --out-dir "$OUT"
done

(cd "$OUT" && zip -q -r /kaggle/working/stage3_out.zip .)
ls -l "$OUT" /kaggle/working/stage3_out.zip
echo "STAGE3 DONE"
