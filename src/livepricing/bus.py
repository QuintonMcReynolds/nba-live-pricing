"""Message bus: an in-memory bus for tests/replays and a Kafka bus for deployment.

Both expose publish(topic, key, value) and consume(topics) -> iterator of (topic, value).
Events are keyed by game_id so one game's actions stay ordered within a partition.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Iterator

TOPIC_EVENTS = "pbp.events"
TOPIC_PRICES = "prices.live"
TOPIC_ALERTS = "trading.alerts"


class InMemoryBus:
    def __init__(self):
        self.queues: dict[str, deque] = defaultdict(deque)
        self.log: dict[str, list] = defaultdict(list)

    def publish(self, topic: str, key: str, value: dict) -> None:
        self.queues[topic].append(value)
        self.log[topic].append(value)

    def consume(self, topics: list[str]) -> Iterator[tuple[str, dict]]:
        while any(self.queues[t] for t in topics):
            for t in topics:
                while self.queues[t]:
                    yield t, self.queues[t].popleft()

    def flush(self) -> None:
        pass


class KafkaBus:
    """confluent-kafka producer/consumer with JSON values (works with Kafka or Redpanda)."""

    def __init__(self, bootstrap: str, group_id: str = "pricing-engine", idle_timeout: float = 5.0):
        from confluent_kafka import Consumer, Producer

        self.producer = Producer({"bootstrap.servers": bootstrap, "linger.ms": 1,
                                  "enable.idempotence": True})
        self.consumer_conf = {"bootstrap.servers": bootstrap, "group.id": group_id,
                              "auto.offset.reset": "earliest", "enable.auto.commit": True}
        self._consumer_cls = Consumer
        self.idle_timeout = idle_timeout

    def publish(self, topic: str, key: str, value: dict) -> None:
        self.producer.produce(topic, key=str(key), value=json.dumps(value).encode())
        self.producer.poll(0)

    def flush(self) -> None:
        self.producer.flush(10)

    def consume(self, topics: list[str]) -> Iterator[tuple[str, dict]]:
        """Yield messages until the topics have been idle for idle_timeout seconds."""
        import time

        consumer = self._consumer_cls(self.consumer_conf)
        consumer.subscribe(topics)
        last = time.monotonic()
        try:
            while time.monotonic() - last < self.idle_timeout:
                msg = consumer.poll(0.2)
                if msg is None:
                    continue
                if msg.error():
                    raise RuntimeError(msg.error())
                last = time.monotonic()
                yield msg.topic(), json.loads(msg.value())
        finally:
            consumer.close()
