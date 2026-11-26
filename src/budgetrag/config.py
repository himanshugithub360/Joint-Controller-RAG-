"""Load the Hydra config from configs/ without taking over the script's entry point."""

from __future__ import annotations

import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """Compose configs/config.yaml with optional overrides such as ["profile=full"]."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides or [])
    OmegaConf.resolve(cfg)
    return cfg


def repo_path(p: str | Path) -> Path:
    """Resolve a config path against the repo root unless it is already absolute."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def save_run_config(cfg: DictConfig, out_dir: Path) -> Path:
    """Save the resolved config and git hash next to a run's outputs (plan convention)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_config.yaml"
    data = OmegaConf.to_container(cfg, resolve=True)
    data["_git_hash"] = git_hash()
    path.write_text(OmegaConf.to_yaml(OmegaConf.create(data)), encoding="utf-8")
    return path
