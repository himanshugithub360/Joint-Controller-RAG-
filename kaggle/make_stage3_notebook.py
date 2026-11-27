"""Writes kaggle/stage3.ipynb (kept as code so the notebook's cells are reviewable in diffs)."""

import json
from pathlib import Path

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

cells = [
    md("# Stage 3 on Kaggle: all three tiers answer the validation prompts\n\n"
       "Settings before running: **Accelerator = GPU T4 x2**, **Internet = on**. Then *Run all*.\n\n"
       "Runs the 2,000 prompts in `kaggle/inputs/stage3_jobs.jsonl.gz` (500 validation questions x answer, "
       "closed-book, tentative, sub-query) on each tier, plus 200 answer prompts a second time for the "
       "determinism check. About 30 minutes, mostly model downloads and loading.\n\n"
       "At the end, download **stage3_out.zip** from the output panel (or the Output tab after a saved run) and "
       "unzip it into `data/stage3/` on the workstation, then run `python scripts/s03_score.py`."),
    code("!nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv"),
    code("import os, subprocess\n"
         "%cd /kaggle/working\n"
         "REPO = 'himanshugithub360/Joint-Controller-RAG-'\n"
         "if os.path.exists('/kaggle/working/repo'):\n"
         "    subprocess.run(['git', '-C', '/kaggle/working/repo', 'pull', '--ff-only'], check=True)\n"
         "else:\n"
         "    subprocess.run(['git', 'clone', '--depth', '1', f'https://github.com/{REPO}.git', '/kaggle/working/repo'], check=True)\n"
         "%cd /kaggle/working/repo\n"
         "!git log --oneline -1"),
    code("# vLLM 0.6.6 (last release that runs on T4) needs Python <= 3.12; Kaggle's image is 3.13.\n"
         "if not os.path.exists('/tmp/vllm-env/bin/python'):\n"
         "    !pip install -q uv\n"
         "    !uv venv -q --python 3.12 /tmp/vllm-env\n"
         "    !uv pip install -q --python /tmp/vllm-env/bin/python -r envs/kaggle-requirements.txt\n"
         "!/tmp/vllm-env/bin/python -c \"import vllm, torch; print('vllm', vllm.__version__, '| cuda', torch.cuda.is_available())\""),
    code("# One process per tier, so each model's GPU memory is fully freed before the next.\n"
         "PY = '/tmp/vllm-env/bin/python'\n"
         "!rm -rf /kaggle/working/stage3_out\n"
         "!{PY} kaggle/stage3_generate.py --tier S\n"
         "!{PY} kaggle/stage3_generate.py --tier M\n"
         "!{PY} kaggle/stage3_generate.py --tier L"),
    code("import json, pandas as pd, shutil\n"
         "out = '/kaggle/working/stage3_out'\n"
         "for t in 'SML':\n"
         "    p = f'{out}/{t}.meta.json'\n"
         "    if os.path.exists(p):\n"
         "        m = json.load(open(p))\n"
         "        print(t, m['model'], f\"load {m['load_s']:.0f} s, generate {m['generate_s']:.0f} s, {m['n_rows']} rows\")\n"
         "    else:\n"
         "        print(t, 'MISSING: read the errors printed above')\n"
         "shutil.make_archive('/kaggle/working/stage3_out', 'zip', out)\n"
         "df = pd.read_json(f'{out}/S.jsonl', lines=True)\n"
         "df[df.kind == 'answer'][['qid', 'text', 'n_in', 'n_out']].head(10)"),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
Path(__file__).with_name("stage3.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote kaggle/stage3.ipynb")
