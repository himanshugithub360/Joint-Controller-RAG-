"""Prompt templates in configs/prompts/<version>/ and Search-R1's passage format.

Prompts are rendered to plain strings on the workstation (Qwen chat template included), so
Kaggle only runs generate() on fixed text: the same bytes every time, which keeps vLLM's
prefix cache and the outcome cache keyed on exactly what the model saw.
"""

from __future__ import annotations

from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[3] / "configs" / "prompts"


def load_template(name: str, version: str = "v1", prompt_dir: Path = PROMPT_DIR) -> str:
    """Template text with its leading '#' comment lines removed."""
    lines = (prompt_dir / version / name).read_text(encoding="utf-8").splitlines()
    while lines and lines[0].startswith("#"):
        lines.pop(0)
    return "\n".join(lines).strip("\n")


def format_passages(passages: list[dict]) -> str:
    """Search-R1's _passages2string: 'Doc i(Title: <title>) <text>' per line, i from 1.

    `passages` are {title, text} dicts (title as stored in wiki-18, quotes included).
    """
    return "".join(f"Doc {i}(Title: {p['title']}) {p['text']}\n" for i, p in enumerate(passages, 1))


def fill(template: str, question: str, passages: list[dict] | None = None) -> str:
    # str.replace, not str.format: passages contain braces.
    out = template.replace("{passages}", format_passages(passages or []))
    return out.replace("{question}", question.strip())


def render_chat(tokenizer, user_text: str) -> str:
    """One user turn in the model's chat template, ready for generation."""
    return tokenizer.apply_chat_template([{"role": "user", "content": user_text}],
                                         tokenize=False, add_generation_prompt=True)
