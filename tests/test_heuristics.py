"""Type heuristics over synthetic tracks (#24). No model, no video: tracks are built by hand at 4 fps."""

import pandas as pd
import pytest
import yaml
from pandera.errors import SchemaError

from src.config import RULES_YAML
from src.heuristics import TRACKS_SCHEMA, Thresholds, detect, find_roles
from src.heuristics.detect import (
    CASH_ON_DESK_NEXT_CLIENT,
    COUNT_WITHOUT_CONTROLLER,
    ID_NOT_PRESENTED,
    WORKPLACE_LEFT,
)

FPS = 4
SEAT = (200.0, 400.0)  # where the cashier sits in the 464x848 frame
COUNTER = (200.0, 700.0)  # where a client stands, one body height or more away from the seat
DESK = (200.0, 450.0)  # where cash lies, next to the seat


def part(track_id, kind, t0, t1, cx, cy, *, h=200.0, wobble=0.0, drop_every=0):
    """Rows of one track between t0 and t1 seconds; `wobble` moves the box +-wobble px each frame,
    `drop_every=8` leaves out every 8th frame (a detector miss)."""
    label = kind  # in these scenes every kind carries its own name as the label
    rows = []
    for i in range(round(t0 * FPS), round(t1 * FPS)):
        if drop_every and i % drop_every == drop_every - 1:
            continue
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


def hits_of(type_id, df, **kwargs):
    return [h for h in detect(df, **kwargs) if h.type_id == type_id]


def workplace_left(*, away=(10, 20), client=(5, 20), client_at=COUNTER, cash=(2, 20), cash_drop=0):
    """Cashier at the seat, leaves for `away` seconds; a client and cash on the desk as given (None = absent)."""
    stays = [(0, 30)] if away is None else [(0, away[0]), (away[1], 30)]
    parts = [part(1, "person", a, b, *SEAT, wobble=3) for a, b in stays]
    if client:
        parts.append(part(2, "person", *client, *client_at))
    if cash:
        parts.append(part(10, "cash", *cash, *DESK, h=20, drop_every=cash_drop))
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


def client_change(*, a=(3, 10), b=(14, 25), b_at=COUNTER, cash=(1, 22), cash_drop=0):
    """Cashier stays at the seat; client A, then client B, at the counter; cash on the desk as given."""
    parts = [part(1, "person", 0, 30, *SEAT, wobble=3), part(2, "person", *a, *COUNTER)]
    if b:
        parts.append(part(3, "person", *b, *b_at))
    if cash:
        parts.append(part(10, "cash", *cash, *DESK, h=20, drop_every=cash_drop))
    return tracks(*parts)


def test_type2_cash_stays_on_the_desk_after_the_client_changes():
    hits = hits_of("cash_on_desk_next_client", client_change())

    assert [(h.start_sec, h.end_sec) for h in hits] == [(14.0, 21.75)]


def test_type2_silent_when_the_cash_is_gone_before_the_next_client_arrives():
    assert hits_of("cash_on_desk_next_client", client_change(cash=(1, 12))) == []


def test_type2_silent_when_there_is_no_next_client():
    assert hits_of("cash_on_desk_next_client", client_change(b=None)) == []


def test_type2_counts_from_the_moment_the_previous_client_leaves_when_the_next_one_steps_up_early():
    df = client_change(a=(0, 10), b=(9.5, 25), cash=(5, 30))  # B steps up half a second before A leaves

    hits = hits_of("cash_on_desk_next_client", df)

    assert [(h.start_sec, h.end_sec) for h in hits] == [(10.0, 29.75)]


def passer_by_during_a_visit(*, b=(18, 25)):
    """Client A at the counter 3-16 s; someone steps up beside A 5-8 s and leaves first; the cash goes on the
    desk at 9 s and stays until 22 s; client B as given (None = absent)."""
    parts = [
        part(1, "person", 0, 30, *SEAT, wobble=3),
        part(2, "person", 3, 16, *COUNTER),
        part(4, "person", 5, 8, 260.0, 700.0),
        part(10, "cash", 9, 22, *DESK, h=20),
    ]
    if b:
        parts.append(part(3, "person", *b, *COUNTER))
    return tracks(*parts)


def test_type2_a_passer_by_during_the_previous_visit_does_not_hide_the_client_change():
    hits = hits_of("cash_on_desk_next_client", passer_by_during_a_visit())

    assert [(h.start_sec, h.end_sec) for h in hits] == [(18.0, 21.75)]


def test_type2_silent_when_the_only_other_person_comes_and_goes_during_the_visit():
    assert hits_of("cash_on_desk_next_client", passer_by_during_a_visit(b=None)) == []


def test_type2_silent_when_the_cash_is_put_away_right_after_the_next_client_arrives():
    assert hits_of("cash_on_desk_next_client", client_change(cash=(1, 15))) == []


def counting(*, cash=(5, 20), cashier=(0, 30), others=(), cash_drop=0):
    """Cashier with cash on the desk; `others` is a list of (track_id, t0, t1, cx, cy) for other people."""
    parts = [part(1, "person", *cashier, *SEAT, wobble=3)]
    if cash:
        parts.append(part(10, "cash", *cash, *DESK, h=20, drop_every=cash_drop))
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
        part(11, "document", 5, 7, *COUNTER, h=20),  # the client shows an identity document
    )

    assert detect(df) == []


def test_silent_when_nothing_is_tracked():
    assert detect(tracks()) == []


def test_every_type_the_heuristics_can_report_exists_in_rules_yaml():
    known = {t["id"] for t in yaml.safe_load(RULES_YAML.read_text(encoding="utf-8"))["types"]}

    assert {
        WORKPLACE_LEFT,
        CASH_ON_DESK_NEXT_CLIENT,
        COUNT_WITHOUT_CONTROLLER,
        ID_NOT_PRESENTED,
    } <= known


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


def cashier_away_for_a_while():
    """Cashier at the seat 0-10 s and 22-30 s, a still client at the counter 5-20 s, cash on the desk."""
    return tracks(
        part(1, "person", 0, 10, *SEAT, wobble=3),
        part(1, "person", 22, 30, *SEAT, wobble=3),
        part(2, "person", 5, 20, *COUNTER),
        part(10, "cash", 2, 30, *DESK, h=20),
    )


def test_the_seat_is_where_one_track_stays_longest_not_where_the_stillest_one_stands():
    roles = find_roles(cashier_away_for_a_while())

    assert roles.cashier == 1
    assert roles.seat == SEAT


def test_type1_is_reported_when_the_cashier_is_away_for_40_percent_of_the_clip():
    df = cashier_away_for_a_while()

    assert [(h.start_sec, h.end_sec) for h in hits_of("workplace_left", df)] == [(10.0, 19.75)]
    assert [
        h for h in hits_of("count_without_controller", df) if h.end_sec >= 10.0 and h.start_sec < 22.0
    ] == []


def clients_back_to_back(client_wobble):
    """Cashier at the seat 0-25 s and 35-60 s; client A at the counter 0-30 s, client B at the same spot
    30-60 s; cash on the desk 20-40 s. Together the clients stand longer than the cashier sits."""
    return tracks(
        part(1, "person", 0, 25, *SEAT, wobble=3),
        part(1, "person", 35, 60, *SEAT, wobble=3),
        part(2, "person", 0, 30, *COUNTER, wobble=client_wobble),
        part(3, "person", 30, 60, *COUNTER, wobble=client_wobble),
        part(10, "cash", 20, 40, *DESK, h=20),
    )


BACK_TO_BACK = pytest.mark.parametrize(
    "client_wobble", [0.0, 10.0], ids=["still clients", "clients wobble more than the cashier"]
)


@BACK_TO_BACK
def test_clients_who_come_one_after_another_do_not_make_the_counter_the_seat(client_wobble):
    roles = find_roles(clients_back_to_back(client_wobble))

    assert (roles.cashier, roles.seat) == (1, SEAT)


@BACK_TO_BACK
def test_type1_is_reported_when_clients_come_one_after_another(client_wobble):
    df = clients_back_to_back(client_wobble)

    assert [(h.start_sec, h.end_sec) for h in hits_of("workplace_left", df)] == [(25.0, 34.75)]
    assert hits_of("count_without_controller", df) == []


def test_one_missed_cash_detection_every_two_seconds_does_not_hide_a_type1_episode():
    hits = hits_of("workplace_left", workplace_left(cash_drop=8))

    assert [h.start_sec for h in hits] == [10.0]


def test_one_missed_cash_detection_every_two_seconds_does_not_hide_a_type2_episode():
    hits = hits_of("cash_on_desk_next_client", client_change(cash_drop=8))

    assert [h.start_sec for h in hits] == [14.0]


def test_one_missed_cash_detection_every_two_seconds_does_not_hide_a_type4_episode():
    hits = hits_of("count_without_controller", counting(cash=(5, 25), cash_drop=8))

    assert [h.start_sec for h in hits] == [5.0]


def test_a_gap_longer_than_a_second_still_splits_the_episode():
    df = tracks(
        part(1, "person", 0, 30, *SEAT, wobble=3),
        part(10, "cash", 5, 10, *DESK, h=20),
        part(10, "cash", 11.25, 20, *DESK, h=20),  # missing for 1.25 s
    )

    hits = hits_of("count_without_controller", df)

    assert [(h.start_sec, h.end_sec) for h in hits] == [(5.0, 9.75), (11.25, 19.75)]


def test_the_track_schema_rejects_a_box_without_area():
    with pytest.raises(SchemaError):
        tracks(part(1, "person", 0, 5, *SEAT, h=0.0))


@pytest.mark.parametrize(
    ("type_id", "scene", "stricter"),
    [
        ("workplace_left", lambda: workplace_left(), Thresholds(min_away_sec=11.0)),
        ("workplace_left", lambda: workplace_left(), Thresholds(counter_radius=1.0)),
        ("workplace_left", lambda: workplace_left(cash_drop=8), Thresholds(gap_sec=0.0)),
        ("cash_on_desk_next_client", lambda: client_change(), Thresholds(min_hold_sec=9.0)),
        ("count_without_controller", lambda: counting(), Thresholds(min_count_sec=16.0)),
        ("count_without_controller", lambda: counting(), Thresholds(desk_radius=0.1)),
        ("id_not_presented", lambda: payout(), Thresholds(min_payout_sec=14.0)),
    ],
    ids=[
        "min_away_sec",
        "counter_radius",
        "gap_sec",
        "min_hold_sec",
        "min_count_sec",
        "desk_radius",
        "min_payout_sec",
    ],
)
def test_every_threshold_can_be_passed_in(type_id, scene, stricter):
    df = scene()

    assert hits_of(type_id, df) != []
    assert [h for h in detect(df, thresholds=stricter) if h.type_id == type_id] == []


def test_the_seat_radius_decides_whether_two_nearby_spots_are_one_seat():
    df = tracks(part(1, "person", 0, 30, *SEAT, wobble=50))  # alternates between x=150 and x=250

    assert find_roles(df).seat[0] == 200.0
    assert find_roles(df, Thresholds(seat_radius=0.25)).seat[0] in (150.0, 250.0)


def test_the_seat_radius_decides_who_counts_as_standing_at_the_seat():
    df = tracks(
        part(1, "person", 0, 10, *SEAT, wobble=3),
        part(1, "person", 20, 30, *SEAT, wobble=3),
        part(
            2, "person", 12, 18, 200.0, 500.0
        ),  # 0.5 cashier heights from the seat while the cashier is away
        part(10, "cash", 2, 30, *DESK, h=20),
    )

    default = [h for h in detect(df) if h.type_id == "workplace_left"]
    narrow = [h for h in detect(df, thresholds=Thresholds(seat_radius=0.4)) if h.type_id == "workplace_left"]

    assert default == []  # taken for the cashier's stand-in at the seat
    assert [(h.start_sec, h.end_sec) for h in narrow] == [(12.0, 17.75)]  # too far from the seat: a client


def payout(*, client=(5, 20), cash=(6, 19), document=None, document_at=COUNTER):
    """Cashier at the seat, one client at the counter, cash on the desk; `document` = (t0, t1) when shown."""
    parts = [part(1, "person", 0, 30, *SEAT, wobble=3), part(2, "person", *client, *COUNTER)]
    if cash:
        parts.append(part(10, "cash", *cash, *DESK, h=20))
    if document:
        parts.append(part(11, "document", *document, *document_at, h=20))
    return tracks(*parts)


def test_type7_cash_in_front_of_a_client_and_no_document_shown():
    hits = hits_of("id_not_presented", payout())

    assert [(h.start_sec, h.end_sec) for h in hits] == [(6.0, 18.75)]


def test_type7_silent_when_the_client_shows_a_document():
    assert hits_of("id_not_presented", payout(document=(7, 9))) == []


def test_type7_silent_without_cash_because_p31_is_about_paying_out():
    assert hits_of("id_not_presented", payout(cash=None)) == []


def test_type7_silent_when_the_client_only_passes_by_the_counter():
    assert hits_of("id_not_presented", payout(client=(5, 9), cash=(6, 9))) == []


def test_type7_a_document_far_from_the_counter_does_not_count_as_shown():
    df = payout(document=(7, 9), document_at=(440.0, 800.0))  # 2.3 cashier heights from the seat

    assert [(h.start_sec, h.end_sec) for h in hits_of("id_not_presented", df)] == [(6.0, 18.75)]


def test_type7_a_document_of_the_previous_client_does_not_cover_the_next_one():
    df = tracks(
        part(1, "person", 0, 30, *SEAT, wobble=3),
        part(2, "person", 3, 12, *COUNTER),
        part(11, "document", 4, 6, *COUNTER, h=20),  # only client A shows one
        part(3, "person", 14, 26, *COUNTER),
        part(10, "cash", 4, 26, *DESK, h=20),
    )

    assert [(h.start_sec, h.end_sec) for h in hits_of("id_not_presented", df)] == [(14.0, 25.75)]


@pytest.mark.parametrize(("collapsed", "to"), [("y2", "y1"), ("x2", "x1")], ids=["no height", "no width"])
def test_a_box_without_area_that_skipped_the_schema_is_ignored_not_a_crash(collapsed, to):
    broken = tracks(part(99, "person", 5, 6, *COUNTER))
    broken[collapsed] = broken[to]  # as if TRACKS_SCHEMA.validate had not run
    mixed = pd.concat([workplace_left(), broken])

    assert find_roles(broken) is None
    assert detect(broken) == []
    assert detect(mixed) == detect(workplace_left())


def test_the_cashier_is_the_track_most_present_at_the_seat_not_the_lowest_id():
    df = tracks(
        part(1, "person", 0, 5, *SEAT, wobble=3),  # a short earlier fragment
        part(7, "person", 10, 30, *SEAT, wobble=3),
    )

    assert find_roles(df).cashier == 7


def test_missed_detections_of_the_cashier_do_not_hand_the_seat_to_a_client():
    df = tracks(
        part(1, "person", 0, 60, *SEAT, drop_every=4),  # 45 s of boxes over 60 s at the seat
        part(2, "person", 5, 55, *COUNTER),  # 50 s of boxes
    )

    assert (find_roles(df).cashier, find_roles(df).seat) == (1, SEAT)
    assert find_roles(df, Thresholds(gap_sec=0.0)).cashier == 2  # misses no longer closed


@pytest.mark.parametrize(
    ("fragments", "client_until", "seat_by"),
    [
        # With "track" the client waits less than the cashier's longest fragment; a longer wait takes the seat.
        ([(1, 0, 12), (5, 12, 30)], 15, "track"),  # fragments of 12 s and 18 s, the client waits 15 s
        ([(1, 0, 10), (5, 10, 20), (6, 20, 30)], 8, "track"),  # three fragments of 10 s, the client waits 8 s
        # "boxes" keeps the seat through the longer waits from the review of #58.
        ([(1, 0, 12), (5, 12, 30)], 20, "boxes"),
        ([(1, 0, 10), (5, 10, 20), (6, 20, 30)], 15, "boxes"),
    ],
    ids=["two ids", "three ids", "two ids, long wait, boxes", "three ids, long wait, boxes"],
)
def test_a_cashier_split_into_several_ids_keeps_the_seat_while_a_client_waits(
    fragments, client_until, seat_by
):
    df = tracks(
        *(part(i, "person", a, b, *SEAT, wobble=3) for i, a, b in fragments),
        part(2, "person", 0, client_until, *COUNTER),
        part(10, "cash", 2, 30, *DESK, h=20),
    )
    th = Thresholds(seat_by=seat_by)

    roles = find_roles(df, th)

    assert roles.seat == SEAT
    assert roles.cashier != 2
    assert hits_of("workplace_left", df, thresholds=th) == []
    assert hits_of("cash_on_desk_next_client", df, thresholds=th) == []
