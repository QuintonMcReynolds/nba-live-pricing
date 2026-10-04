from fastapi.testclient import TestClient

from livepricing.engine import PricingEngine
from livepricing.service import create_app


def test_service_endpoints(cfg, pregame, scripted_game):
    eng = PricingEngine(cfg, pregame)
    for e in scripted_game[:10]:
        eng.on_event(e)
    c = TestClient(create_app(eng))
    assert c.get("/health").json()["ok"]
    assert c.get("/markets/1").json()["game_id"] == 1
    assert c.get("/markets/2").status_code == 404
    r = c.post("/overrides", json={"game_id": 1, "points": 9, "trader": "t", "reason": "r"})
    assert r.status_code == 422
    r = c.post("/overrides", json={"game_id": 1, "points": 1, "trader": "t", "reason": "news"})
    assert r.status_code == 200
    assert "livepricing_events 10" in c.get("/metrics").text
    assert c.get("/audit").json()[-1]["accepted"]
