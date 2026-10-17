"""Type heuristics over synthetic tracks (#24). No model, no video: tracks are built by hand at 4 fps."""

import pandas as pd
import yaml

from src.config import RULES_YAML
from src.heuristics import TRACKS_SCHEMA, detect, find_roles
from src.heuristics.detect import CASH_ON_DESK_NEXT_CLIENT, COUNT_WITHOUT_CONTROLLER, WORKPLACE_LEFT

FPS = 4
SEAT = (200.0, 400.0)  # where the cashier sits in the 464x848 frame
COUNTER = (200.0, 700.0)  # where a client stands, one body height or more away from the seat
DESK = (200.0, 450.0)  # where cash lies, next to the seat


def part(track_id, kind, t0, t1, cx, cy, *, h=200.0, wobble=0.0):
    """Rows of one track between t0 and t1 seconds; `wobble` moves the box +-wobble px each frame."""
    label = "person" if kind == "person" else "cash"
    rows = []
    for i in range(round(t0 * FPS), round(t1 * FPS)):
        dx = wobble if i % 2 else -wobble
        rows.append(
            {
                "track_id": track_id,
                "kind": kind,
                "label": label,
                "t": i / FPS,
                "x1": cx + dx - h / 4,
                "y1": cy - h / 2,
                "x2": cx + dx + h / 4,
                "y2": cy + h / 2,
                "conf": 0.9,
            }
        )
    return rows


def tracks(*parts):
    rows = [row for p in parts for row in p]
    return TRACKS_SCHEMA.validate(pd.DataFrame(rows, columns=list(TRACKS_SCHEMA.columns)))


def test_cashier_is_the_long_track_not_a_still_visitor():
    df = tracks(
        part(1, "person", 0, 30, *SEAT, wobble=3),
        part(2, "person", 5, 20, *COUNTER),  # perfectly still, but present only half as long
    )

    roles = find_roles(df)

    assert roles.cashier == 1


def test_cashier_is_the_stillest_of_the_long_tracks():
    df = tracks(
        part(1, "person", 0, 30, *COUNTER, wobble=40),  # present all clip, but pacing around
        part(2, "person", 0, 30, *SEAT, wobble=3),
    )

    roles = find_roles(df)

    assert roles.cashier == 2
    assert roles.seat == SEAT
    assert roles.scale == 200.0


def test_no_person_tracks_means_no_roles():
    df = tracks(part(1, "cash", 0, 10, *SEAT, h=20))

    assert find_roles(df) is None


def hits_of(type_id, df):
    return [h for h in detect(df) if h.type_id == type_id]


def workplace_left(*, away=(10, 20), client=(5, 20), client_at=COUNTER, cash=(2, 20)):
    """Cashier at the seat, leaves for `away` seconds; a client and cash on the desk as given (None = absent)."""
    stays = [(0, 30)] if away is None else [(0, away[0]), (away[1], 30)]
    parts = [part(1, "person", a, b, *SEAT, wobble=3) for a, b in stays]
    if client:
        parts.append(part(2, "person", *client, *client_at))
    if cash:
        parts.append(part(10, "cash", *cash, *DESK, h=20))
    return tracks(*parts)


def test_type1_cashier_leaves_with_cash_on_desk_and_client_at_counter():
    hits = hits_of("workplace_left", workplace_left())

    assert [(h.start_sec, h.end_sec) for h in hits] == [(10.0, 19.75)]


def test_type1_silent_when_no_client_is_visible():
    assert hits_of("workplace_left", workplace_left(client=None)) == []


def test_type1_silent_when_no_cash_is_on_the_desk():
    assert hits_of("workplace_left", workplace_left(cash=None)) == []


def test_type1_silent_when_the_cashier_only_steps_away_for_a_second():
    assert hits_of("workplace_left", workplace_left(away=(10, 11))) == []


def test_type1_silent_when_the_cashier_never_leaves():
    assert hits_of("workplace_left", workplace_left(away=None)) == []


def test_type1_silent_before_the_cashier_has_arrived():
    assert hits_of("workplace_left", workplace_left(away=(0, 8))) == []


def client_change(*, a=(3, 10), b=(14, 25), b_at=COUNTER, cash=(1, 22)):
    """Cashier stays at the seat; client A, then client B, at the counter; cash on the desk as given."""
    parts = [part(1, "person", 0, 30, *SEAT, wobble=3), part(2, "person", *a, *COUNTER)]
    if b:
        parts.append(part(3, "person", *b, *b_at))
    if cash:
        parts.append(part(10, "cash", *cash, *DESK, h=20))
    return tracks(*parts)


def test_type2_cash_stays_on_the_desk_after_the_client_changes():
    hits = hits_of("cash_on_desk_next_client", client_change())

    assert [(h.start_sec, h.end_sec) for h in hits] == [(14.0, 21.75)]


def test_type2_silent_when_the_cash_is_gone_before_the_next_client_arrives():
    assert hits_of("cash_on_desk_next_client", client_change(cash=(1, 12))) == []


def test_type2_silent_when_there_is_no_next_client():
    assert hits_of("cash_on_desk_next_client", client_change(b=None)) == []


def test_type2_silent_when_the_next_client_arrives_before_the_first_one_leaves():
    assert hits_of("cash_on_desk_next_client", client_change(a=(3, 16))) == []


def test_type2_silent_when_the_cash_is_put_away_right_after_the_next_client_arrives():
    assert hits_of("cash_on_desk_next_client", client_change(cash=(1, 15))) == []


def counting(*, cash=(5, 20), cashier=(0, 30), others=()):
    """Cashier with cash on the desk; `others` is a list of (track_id, t0, t1, cx, cy) for other people."""
    parts = [part(1, "person", *cashier, *SEAT, wobble=3)]
    if cash:
        parts.append(part(10, "cash", *cash, *DESK, h=20))
    parts += [part(i, "person", t0, t1, cx, cy) for i, t0, t1, cx, cy in others]
    return tracks(*parts)


def test_type4_cashier_counts_cash_alone():
    hits = hits_of("count_without_controller", counting())

    assert [(h.start_sec, h.end_sec) for h in hits] == [(5.0, 19.75)]


def test_type4_silent_when_a_client_is_at_the_counter():
    df = counting(others=[(2, 3, 25, *COUNTER)])

    assert hits_of("count_without_controller", df) == []


def test_type4_silent_when_a_second_worker_is_in_the_frame():
    df = counting(others=[(2, 3, 25, 200.0, 100.0)])  # behind the cashier, far from the counter

    assert hits_of("count_without_controller", df) == []


def test_type4_silent_when_there_is_no_cash():
    assert hits_of("count_without_controller", counting(cash=None)) == []


def test_type4_silent_when_the_cash_lies_there_but_the_cashier_is_away():
    assert hits_of("count_without_controller", counting(cashier=(0, 10), cash=(12, 20))) == []


def test_type4_silent_when_the_cash_is_handled_only_for_two_seconds():
    assert hits_of("count_without_controller", counting(cash=(5, 7))) == []


def test_type2_silent_when_the_cash_appears_only_after_the_previous_client_left():
    assert hits_of("cash_on_desk_next_client", client_change(cash=(11, 22))) == []


def test_silent_during_normal_service():
    df = tracks(
        part(1, "person", 0, 30, *SEAT, wobble=3),
        part(2, "person", 5, 15, *COUNTER),
        part(10, "cash", 6, 12, *DESK, h=20),  # the client's money while being served
    )

    assert detect(df) == []


def test_silent_when_nothing_is_tracked():
    assert detect(tracks()) == []


def test_every_type_the_heuristics_can_report_exists_in_rules_yaml():
    known = {t["id"] for t in yaml.safe_load(RULES_YAML.read_text(encoding="utf-8"))["types"]}

    assert {WORKPLACE_LEFT, CASH_ON_DESK_NEXT_CLIENT, COUNT_WITHOUT_CONTROLLER} <= known


def test_type1_silent_when_the_only_other_person_is_far_from_the_counter():
    df = workplace_left(client_at=(440.0, 800.0))  # 2.3 cashier heights from the seat

    assert hits_of("workplace_left", df) == []


def test_type2_silent_when_the_next_person_is_far_from_the_counter():
    df = client_change(b_at=(440.0, 800.0))

    assert hits_of("cash_on_desk_next_client", df) == []


def test_type4_silent_when_a_client_stands_right_beside_the_cashier():
    df = counting(others=[(2, 3, 25, 200.0, 500.0)])  # 0.5 cashier heights from the seat

    assert hits_of("count_without_controller", df) == []


def test_a_cashier_who_returns_under_a_new_track_id_is_not_someone_else():
    df = tracks(
        part(1, "person", 0, 10, *SEAT, wobble=3),
        part(5, "person", 20, 30, *SEAT, wobble=3),  # the tracker lost the cashier and gave a new id
        part(10, "cash", 2, 30, *DESK, h=20),
    )

    hits = [(h.type_id, h.start_sec, h.end_sec) for h in detect(df)]

    assert hits == [("count_without_controller", 2.0, 9.75), ("count_without_controller", 20.0, 29.75)]


def test_type1_silent_when_the_cashier_is_never_seen_at_the_seat():
    df = tracks(
        part(1, "person", 0, 15, 100.0, 200.0),
        part(1, "person", 15, 30, 300.0, 600.0),  # paces between two spots, nobody stands at the median seat
        part(2, "person", 5, 20, *COUNTER),
        part(10, "cash", 2, 20, *DESK, h=20),
    )

    assert hits_of("workplace_left", df) == []
