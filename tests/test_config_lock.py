"""If this fails, a price-moving number changed. Review the golden diff, then run
`python scripts/update_lock.py` and commit config/model.lock with the change."""

import json

from livepricing import config


def test_config_matches_lock():
    lock = json.loads((config.CONFIG_DIR / "model.lock").read_text())
    assert config.config_hash() == lock["config_sha256"], \
        "config/model.json changed without updating config/model.lock"


def test_golden_prices_unchanged():
    lock = json.loads((config.CONFIG_DIR / "model.lock").read_text())
    now = config.golden_outputs(config.ModelConfig.load())
    for state, old, new in zip(config.GOLDEN_STATES, lock["golden"], now):
        for k in old:
            assert abs(old[k] - new[k]) < 1e-6, f"{k} moved at state {state}: {old[k]} -> {new[k]}"
