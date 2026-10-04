from livepricing.state import GameState, possession_after

from .conftest import make_event


def test_made_shot_flips_possession():
    assert possession_after(make_event(1, 2000, 2, 0, side=1, poss=1)) == -1


def test_missed_shot_keeps_shooting_team_until_rebound():
    e = make_event(1, 2000, 0, 0, side=1, poss=1, desc="MISS Booker 18' Jump Shot")
    assert possession_after(e) == 1
    assert possession_after(make_event(2, 1998, 0, 0, action="rebound", side=-1, poss=1)) == -1


def test_free_throws_flip_only_after_the_last():
    first = make_event(1, 2000, 1, 0, action="freethrow", sub="1 of 2", side=1, poss=1)
    last = make_event(2, 2000, 2, 0, action="freethrow", sub="2 of 2", side=1, poss=1)
    assert possession_after(first) == 1
    assert possession_after(last) == -1


def test_turnover_flips():
    assert possession_after(make_event(1, 2000, 0, 0, action="turnover", side=1, poss=1)) == -1


def test_duplicate_and_out_of_order_events_are_dropped():
    st = GameState(1, "PHX", "DEN", 0, 228)
    assert st.apply(make_event(5, 2000, 10, 8))
    assert not st.apply(make_event(5, 2000, 99, 0))
    assert not st.apply(make_event(4, 2010, 99, 0))
    assert (st.home_score, st.away_score) == (10, 8)
