"""Turn raw model output into the answer string that gets scored (plan Stage 3, step 6).

First line only, then strip answer prefixes ("Answer:", "The answer is"), wrapping quotes or
markdown, and trailing punctuation. Tentative answers use AutoSearch's <answer> tags; the
text inside them is taken first. Scoring itself is budgetrag.eval.metrics (reused code).
"""

from __future__ import annotations

import re

from budgetrag.eval import metrics

_PREFIX = re.compile(
    r"^\s*(?:(?:the\s+)?(?:final\s+|short\s+)?answer\s*(?:is\s*)?[:\-]?|a\s*:|search\s+query\s*:|query\s*:)\s*",
    re.IGNORECASE,
)
_WRAP = "\"'`*“”‘’«»"
_TRAIL = ".,;:!"


def _strip(s: str) -> str:
    prev = None
    while s != prev:
        prev = s
        s = s.strip().rstrip(_TRAIL).strip()
        if len(s) >= 2 and s[0] in _WRAP and s[-1] in _WRAP:
            s = s[1:-1]
        elif s and s[0] in "*`" :
            s = s.lstrip("*`")
        elif s and s[-1] in "*`":
            s = s.rstrip("*`")
    return s


def extract_answer(raw: str | None) -> str:
    """Scorable answer text; '' when the output holds none."""
    if not raw:
        return ""
    text = raw
    if "<answer>" in text:
        text = text.rsplit("<answer>", 1)[1]
    text = text.split("</answer>", 1)[0]
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    line = lines[0]
    line = _PREFIX.sub("", line, count=1)
    return _strip(line)


def extract_query(raw: str | None) -> str:
    """Search query text from a subquery call ('' when empty)."""
    return extract_answer(raw)


def score(raw: str | None, golds: list[str]) -> dict:
    pred = extract_answer(raw)
    return {"pred": pred, "em": metrics.exact_match(pred, golds), "f1": metrics.f1(pred, golds)}
