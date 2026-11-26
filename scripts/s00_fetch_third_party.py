"""Stage 0: fetch the reused repositories into third_party/ at the commits pinned in THIRD_PARTY.md.

    python scripts/s00_fetch_third_party.py            # all of them
    python scripts/s00_fetch_third_party.py FlashRAG   # only some

Reused code is never edited in place: thesis code imports from it or copies a function
with a citation in the file header.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE_ROW = re.compile(r"^\|\s*([\w.-]+)\s*\|\s*\[([\w.-]+/[\w.-]+)\]\(.*?\)\s*\|\s*`([0-9a-f]{40})`")


def pinned() -> list[tuple[str, str, str]]:
    rows = []
    for line in (ROOT / "THIRD_PARTY.md").read_text(encoding="utf-8").splitlines():
        m = TABLE_ROW.match(line)
        if m:
            rows.append(m.groups())
    return rows


def git(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def fetch(name: str, repo: str, sha: str) -> None:
    dest = ROOT / "third_party" / name
    if not (dest / ".git").exists():
        dest.mkdir(parents=True, exist_ok=True)
        git("init", "-q", cwd=dest)
        git("remote", "add", "origin", f"https://github.com/{repo}.git", cwd=dest)
    git("fetch", "-q", "--depth", "1", "origin", sha, cwd=dest)
    git("checkout", "-q", "--detach", sha, cwd=dest)
    print(f"{name:14s} {repo:28s} {sha[:12]}  ok")


def main(names: list[str]) -> None:
    rows = pinned()
    if not rows:
        sys.exit("no pinned repositories found in THIRD_PARTY.md")
    for name, repo, sha in rows:
        if not names or name in names:
            fetch(name, repo, sha)


if __name__ == "__main__":
    main(sys.argv[1:])
