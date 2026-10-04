import pytest

from livepricing import pricing


@pytest.mark.parametrize("p", [0.1, 0.35, 0.5, 0.62, 0.9])
def test_american_roundtrip(p):
    assert pricing.implied(pricing.american(p)) == pytest.approx(p, abs=0.003)


def test_two_way_overround():
    h, a = pricing.two_way(0.6, 0.045)
    assert pricing.implied(h) + pricing.implied(a) == pytest.approx(1.045, abs=0.005)


def test_half_point_lines():
    assert pricing.half_point(4.0) == 4.5
    assert pricing.half_point(-3.2) == -3.5
    line, p = pricing.spread_quote(6.2, 12)
    assert line == -6.5 and 0.45 < p < 0.55
