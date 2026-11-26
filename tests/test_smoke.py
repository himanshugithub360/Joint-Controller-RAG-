"""Stage 0 smoke tests: the local environment imports and the config composes."""

import importlib

import pytest

from budgetrag.config import load_config


@pytest.mark.parametrize("module", ["torch", "faiss", "bm25s", "gymnasium", "omnisafe", "pandas", "pyarrow"])
def test_import(module):
    importlib.import_module(module)


@pytest.mark.parametrize("profile", ["lite", "full"])
def test_config_profiles(profile):
    cfg = load_config([f"profile={profile}"])
    assert cfg.profile_name == profile
    assert set(cfg.models.tiers) == {"S", "M", "L"}
    assert cfg.paths.raw.endswith("raw")
