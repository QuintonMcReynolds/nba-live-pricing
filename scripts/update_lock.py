"""Regenerate config/model.lock after a reviewed change to config/model.json."""

import json

from livepricing import config

if __name__ == "__main__":
    print(json.dumps(config.write_lock(), indent=2))
