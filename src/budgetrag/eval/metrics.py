"""EM and token-level F1, imported from the reused repositories (not rewritten).

- EM: Search-R1's em_check (verl/utils/reward_score/qa_em.py).
- F1: FlashRAG's F1_Score.token_level_scores (flashrag/evaluator/metrics.py).

Both use the same SQuAD normalisation (lowercase, drop punctuation and articles, squeeze
spaces) and take the max over all gold answers. The files are loaded straight from
third_party/ (fetched by scripts/s00_fetch_third_party.py) so their packages' heavy
dependencies are never imported.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from functools import lru_cache
from pathlib import Path

THIRD_PARTY = Path(__file__).resolve().parents[3] / "third_party"
QA_EM = THIRD_PARTY / "Search-R1" / "verl" / "utils" / "reward_score" / "qa_em.py"
FR_EVAL = THIRD_PARTY / "FlashRAG" / "flashrag" / "evaluator"


def _load(name: str, path: Path) -> types.ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run python scripts/s00_fetch_third_party.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=None)
def _search_r1():
    return _load("_reused_search_r1_qa_em", QA_EM)


@lru_cache(maxsize=None)
def _flashrag_f1():
    # metrics.py does `from flashrag.evaluator.utils import normalize_answer`; register just
    # that module (FlashRAG's package __init__ pulls in torch, faiss, vLLM...), then clean up.
    saved = {k: sys.modules.get(k) for k in ("flashrag", "flashrag.evaluator", "flashrag.evaluator.utils")}
    try:
        for pkg in ("flashrag", "flashrag.evaluator"):
            sys.modules[pkg] = types.ModuleType(pkg)
        _load("flashrag.evaluator.utils", FR_EVAL / "utils.py")
        metrics = _load("_reused_flashrag_metrics", FR_EVAL / "metrics.py")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return metrics.F1_Score({"dataset_name": "budgetrag"})


def normalize_answer(s: str) -> str:
    return _search_r1().normalize_answer(s)


def exact_match(prediction: str, golds: list[str]) -> float:
    return float(_search_r1().em_check(prediction, list(golds)))


def f1(prediction: str, golds: list[str]) -> float:
    return float(_flashrag_f1().token_level_scores(prediction, list(golds))["f1"])
