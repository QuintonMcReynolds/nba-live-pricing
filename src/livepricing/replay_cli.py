"""Stream a recorded slate (artifacts/slate_events.jsonl) onto Kafka."""

from __future__ import annotations

import argparse
import json

from .bus import KafkaBus
from .replay import publish
from .state import Event


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", required=True)
    ap.add_argument("--file", default="artifacts/slate_events.jsonl")
    ap.add_argument("--speed", type=float, default=None)
    args = ap.parse_args()
    with open(args.file) as fh:
        events = [Event(**json.loads(line)) for line in fh]
    n = publish(KafkaBus(args.bootstrap), events, speed=args.speed)
    print(f"published {n} events")


if __name__ == "__main__":
    main()
