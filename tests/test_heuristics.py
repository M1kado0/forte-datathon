"""Type heuristics over synthetic tracks (#24). No model, no video: tracks are built by hand at 4 fps."""

import numpy as np
import pandas as pd
import pytest
import yaml
from pandera.errors import SchemaError

from src.config import RULES_YAML
from src.heuristics import TRACKS_SCHEMA, Thresholds, detect, find_roles
from src.heuristics.detect import (
    CASH_LEFT_ON_DESK,
    CASH_ON_DESK_NEXT_CLIENT,
    CLIENT_LEFT_DURING_COUNT,
    FOREIGN_OBJECT_AT_DESK,
    FOREIGN_OBJECTS,
    ID_NOT_PRESENTED,
    LAPTOP_LEFT_OPEN,
    OUTSIDER_IN_CASH_ROOM,
    VALUABLES_LEFT_ON_DESK,
    VALUABLES_OBJECTS,
    Hit,
)
from src.heuristics.vote import VOTING_TYPES, vote
from src.rules.describe import OBJECT_NAMES, describe_violation

FPS = 4
SEAT = (200.0, 400.0)  # where the cashier sits in the 464x848 frame
COUNTER = (200.0, 700.0)  # where a client stands, one body height or more away from the seat
DESK = (200.0, 450.0)  # where cash lies, next to the seat


def part(track_id, kind, t0, t1, cx, cy, *, h=200.0, wobble=0.0, drop_every=0, label=None, conf=0.9):
    """Rows of one track between t0 and t1 seconds; `wobble` moves the box +-wobble px each frame,
    `drop_every=8` leaves out every 8th frame (a detector miss). The label is the kind unless given
    (a COCO class for kind "object")."""
    label = label or kind
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
                "conf": conf,
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


def cash_left(*, away=(10, 20), client=(5, 20), client_at=COUNTER, cash=(2, 20), cash_drop=0):
    """Cashier at the seat, leaves for `away` seconds; a client and cash on the desk as given (None = absent)."""
    stays = [(0, 30)] if away is None else [(0, away[0]), (away[1], 30)]
    parts = [part(1, "person", a, b, *SEAT, wobble=3) for a, b in stays]
    if client:
        parts.append(part(2, "person", *client, *client_at))
    if cash:
        parts.append(part(10, "cash", *cash, *DESK, h=20, drop_every=cash_drop))
    return tracks(*parts)


def test_cash_left_on_desk_cashier_leaves_with_cash_on_the_desk_and_a_client_at_the_counter():
    hits = hits_of(CASH_LEFT_ON_DESK, cash_left())

    assert [(h.start_sec, h.end_sec) for h in hits] == [(10.0, 19.75)]


def test_cash_left_on_desk_needs_no_client_at_the_counter():
    hits = hits_of(CASH_LEFT_ON_DESK, cash_left(client=None))

    assert [(h.start_sec, h.end_sec) for h in hits] == [(10.0, 19.75)]


def test_cash_left_on_desk_does_not_care_where_a_client_stands():
    far = cash_left(client_at=(440.0, 800.0))  # 2.3 cashier heights from the seat

    assert [(h.start_sec, h.end_sec) for h in hits_of(CASH_LEFT_ON_DESK, far)] == [(10.0, 19.75)]


def test_cash_left_on_desk_silent_when_no_cash_is_on_the_desk():
    assert hits_of(CASH_LEFT_ON_DESK, cash_left(cash=None)) == []


def test_cash_left_on_desk_silent_when_the_cashier_only_steps_away_for_a_second():
    assert hits_of(CASH_LEFT_ON_DESK, cash_left(away=(10, 11))) == []


def test_cash_left_on_desk_silent_when_the_cashier_never_leaves():
    assert hits_of(CASH_LEFT_ON_DESK, cash_left(away=None)) == []


def test_cash_left_on_desk_silent_before_the_cashier_has_arrived():
    assert hits_of(CASH_LEFT_ON_DESK, cash_left(away=(0, 8))) == []


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
        CASH_LEFT_ON_DESK,
        CASH_ON_DESK_NEXT_CLIENT,
        LAPTOP_LEFT_OPEN,
        ID_NOT_PRESENTED,
        OUTSIDER_IN_CASH_ROOM,
        FOREIGN_OBJECT_AT_DESK,
        VALUABLES_LEFT_ON_DESK,
        CLIENT_LEFT_DURING_COUNT,
    } <= known


def test_type2_silent_when_the_next_person_is_far_from_the_counter():
    df = client_change(b_at=(440.0, 800.0))

    assert hits_of("cash_on_desk_next_client", df) == []


def test_a_cashier_who_returns_under_a_new_track_id_is_not_someone_else():
    df = tracks(
        part(1, "person", 0, 10, *SEAT, wobble=3),
        part(5, "person", 20, 30, *SEAT, wobble=3),  # the tracker lost the cashier and gave a new id
        part(10, "cash", 2, 30, *DESK, h=20),
    )

    hits = [(h.type_id, h.start_sec, h.end_sec) for h in detect(df)]

    assert hits == [(CASH_LEFT_ON_DESK, 10.0, 19.75)]  # the new id at the seat from 20 s is the cashier back


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


def test_cash_left_on_desk_is_reported_when_the_cashier_is_away_for_40_percent_of_the_clip():
    df = cashier_away_for_a_while()

    assert [(h.start_sec, h.end_sec) for h in hits_of(CASH_LEFT_ON_DESK, df)] == [(10.0, 21.75)]


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
def test_cash_left_on_desk_is_reported_when_clients_come_one_after_another(client_wobble):
    df = clients_back_to_back(client_wobble)

    assert [(h.start_sec, h.end_sec) for h in hits_of(CASH_LEFT_ON_DESK, df)] == [(25.0, 34.75)]


def test_one_missed_cash_detection_every_two_seconds_does_not_hide_a_cash_left_on_desk_episode():
    hits = hits_of(CASH_LEFT_ON_DESK, cash_left(cash_drop=8))

    assert [h.start_sec for h in hits] == [10.0]


def test_one_missed_cash_detection_every_two_seconds_does_not_hide_a_type2_episode():
    hits = hits_of("cash_on_desk_next_client", client_change(cash_drop=8))

    assert [h.start_sec for h in hits] == [14.0]


def test_a_gap_longer_than_a_second_still_splits_the_episode():
    df = tracks(
        part(1, "person", 0, 4, *SEAT, wobble=3),  # the cashier leaves at 4 s
        part(10, "cash", 5, 10, *DESK, h=20),
        part(10, "cash", 11.25, 20, *DESK, h=20),  # missing for 1.25 s
    )

    hits = hits_of(CASH_LEFT_ON_DESK, df)

    assert [(h.start_sec, h.end_sec) for h in hits] == [(5.0, 9.75), (11.25, 19.75)]


def test_the_track_schema_rejects_a_box_without_area():
    with pytest.raises(SchemaError):
        tracks(part(1, "person", 0, 5, *SEAT, h=0.0))


@pytest.mark.parametrize(
    ("type_id", "scene", "stricter"),
    [
        (CASH_LEFT_ON_DESK, lambda: cash_left(), Thresholds(min_away_sec=11.0)),
        (CASH_LEFT_ON_DESK, lambda: cash_left(), Thresholds(desk_radius=0.1)),
        (CASH_LEFT_ON_DESK, lambda: cash_left(cash_drop=8), Thresholds(gap_sec=0.0)),
        (LAPTOP_LEFT_OPEN, lambda: laptop_scene(), Thresholds(min_away_sec=13.0)),
        (LAPTOP_LEFT_OPEN, lambda: laptop_scene(), Thresholds(laptop_radius=0.5)),
        (LAPTOP_LEFT_OPEN, lambda: laptop_scene(conf=0.6), Thresholds(laptop_conf=0.7)),
        ("cash_on_desk_next_client", lambda: client_change(), Thresholds(min_hold_sec=9.0)),
        ("id_not_presented", lambda: payout(), Thresholds(min_payout_sec=14.0)),
    ],
    ids=[
        "cash-min_away_sec",
        "cash-desk_radius",
        "cash-gap_sec",
        "laptop-min_away_sec",
        "laptop-laptop_radius",
        "laptop-laptop_conf",
        "min_hold_sec",
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

    default = hits_of(CASH_LEFT_ON_DESK, df)
    narrow = hits_of(CASH_LEFT_ON_DESK, df, thresholds=Thresholds(seat_radius=0.4))

    # the client is taken for the cashier's stand-in, so the seat is never empty for 3 s
    assert default == []
    assert [(h.start_sec, h.end_sec) for h in narrow] == [(10.0, 19.75)]  # too far from the seat: a client


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
    mixed = pd.concat([cash_left(), broken])

    assert find_roles(broken) is None
    assert detect(broken) == []
    assert detect(mixed) == detect(cash_left())


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
    assert hits_of(CASH_LEFT_ON_DESK, df, thresholds=th) == []
    assert hits_of("cash_on_desk_next_client", df, thresholds=th) == []


# --- part 2 (#33): types 3, 5, 6, 8 and the heuristics' vote -------------------------------------------

BESIDE = (
    200.0,
    520.0,
)  # 0.6 cashier heights from the seat: a second person on the cashier's side of the desk


def outsider(*, who=BESIDE, t=(5, 15), h=200.0, cashier=(0, 30)):
    return tracks(part(1, "person", *cashier, *SEAT, wobble=3), part(2, "person", *t, *who, h=h))


def test_type3_a_second_person_stands_beside_the_cashier():
    hits = hits_of(OUTSIDER_IN_CASH_ROOM, outsider())

    assert [(h.start_sec, h.end_sec) for h in hits] == [(5.0, 14.75)]


def test_type3_silent_for_a_single_frame_beside_the_seat():
    assert hits_of(OUTSIDER_IN_CASH_ROOM, outsider(t=(5, 5.25))) == []


def test_type3_silent_for_a_client_at_the_counter():
    assert hits_of(OUTSIDER_IN_CASH_ROOM, outsider(who=COUNTER)) == []


def test_type3_silent_for_a_small_box_a_client_across_the_counter_looks_like():
    # the same spot, but the box is 0.6 of the cashier's height
    assert hits_of(OUTSIDER_IN_CASH_ROOM, outsider(h=120.0)) == []


def test_type3_silent_for_the_cashier_seen_twice():
    # a second box almost on top of the cashier's: a tracker split or a duplicate detection
    assert hits_of(OUTSIDER_IN_CASH_ROOM, outsider(who=(205.0, 405.0))) == []


def test_type3_silent_when_the_cashier_is_away_from_the_seat():
    # someone else at the seat while the cashier's own track is gone is the cashier coming back
    assert hits_of(OUTSIDER_IN_CASH_ROOM, outsider(cashier=(0, 4))) == []


def test_type3_silent_for_the_cashier_alone_at_the_seat():
    assert hits_of(OUTSIDER_IN_CASH_ROOM, tracks(part(1, "person", 0, 30, *SEAT, wobble=3))) == []


def test_type3_silent_for_the_cashier_returning_under_a_new_track_id():
    df = tracks(
        part(1, "person", 0, 10, *SEAT, wobble=3),
        part(2, "person", 12, 30, *SEAT, wobble=3),  # at the seat: the seat is the cashier's
    )

    assert hits_of(OUTSIDER_IN_CASH_ROOM, df) == []


def counted_with_controller(*, cash=(5, 15), who=BESIDE, t=(5, 15)):
    """Cashier at the seat, a second worker beside it, cash on the desk: a count with a controller."""
    return tracks(
        part(1, "person", 0, 30, *SEAT, wobble=3),
        part(2, "person", *t, *who),
        part(10, "cash", *cash, *DESK, h=20),
    )


EXEMPT = Thresholds(controller_exempt=True)


def test_type3_silent_for_a_second_worker_who_controls_the_count_when_the_exemption_is_on():
    assert hits_of(OUTSIDER_IN_CASH_ROOM, counted_with_controller(), thresholds=EXEMPT) == []


def test_type3_the_count_exemption_is_off_by_default_because_it_would_hide_a_real_outsider():
    assert hits_of(OUTSIDER_IN_CASH_ROOM, counted_with_controller()) != []


def test_type3_with_the_exemption_on_it_still_fires_for_the_part_of_the_stay_without_cash():
    df = counted_with_controller(cash=(5, 9), t=(5, 20))

    hits = hits_of(OUTSIDER_IN_CASH_ROOM, df, thresholds=EXEMPT)

    assert [(h.start_sec, h.end_sec) for h in hits] == [(9.0, 19.75)]


def on_desk(label, *, t=(5, 15), at=DESK, conf=0.9):
    return tracks(
        part(1, "person", 0, 30, *SEAT, wobble=3),
        part(10, "object", *t, *at, h=40, label=label, conf=conf),
    )


@pytest.mark.parametrize("label", ["cup", "cell phone", "bottle"])
def test_type5_cup_phone_or_bottle_on_the_desk(label):
    hits = hits_of(FOREIGN_OBJECT_AT_DESK, on_desk(label))

    assert [(h.start_sec, h.end_sec, h.label) for h in hits] == [(5.0, 14.75, label)]
    assert hits_of(VALUABLES_LEFT_ON_DESK, on_desk(label)) == []


@pytest.mark.parametrize("label", ["cup", "cell phone", "bottle"])
def test_type5_silent_for_a_single_frame(label):
    assert hits_of(FOREIGN_OBJECT_AT_DESK, on_desk(label, t=(5, 5.25))) == []


def test_type5_each_class_has_its_own_confidence_floor():
    assert hits_of(FOREIGN_OBJECT_AT_DESK, on_desk("cell phone", conf=0.3)) == []
    assert hits_of(FOREIGN_OBJECT_AT_DESK, on_desk("cell phone", conf=0.6)) != []
    assert hits_of(FOREIGN_OBJECT_AT_DESK, on_desk("cup", conf=0.4)) == []
    assert hits_of(FOREIGN_OBJECT_AT_DESK, on_desk("bottle", conf=0.3)) != []


def test_type5_silent_for_an_object_far_from_the_desk():
    assert hits_of(FOREIGN_OBJECT_AT_DESK, on_desk("cup", at=(200.0, 1500.0))) == []


def test_type5_silent_for_a_class_that_is_not_in_the_list():
    assert hits_of(FOREIGN_OBJECT_AT_DESK, on_desk("laptop")) == []


BOOK_ON = Thresholds(book_conf=0.5)


def test_type6_is_off_until_a_book_confidence_is_set():
    assert hits_of(VALUABLES_LEFT_ON_DESK, on_desk("book")) == []


def test_type6_a_book_on_the_desk():
    hits = hits_of(VALUABLES_LEFT_ON_DESK, on_desk("book"), thresholds=BOOK_ON)

    assert [(h.start_sec, h.end_sec, h.label) for h in hits] == [(5.0, 14.75, "book")]
    assert hits_of(FOREIGN_OBJECT_AT_DESK, on_desk("book"), thresholds=BOOK_ON) == []


def test_type6_silent_for_a_single_frame():
    assert hits_of(VALUABLES_LEFT_ON_DESK, on_desk("book", t=(5, 5.25)), thresholds=BOOK_ON) == []


def cup_from_the_start(*, until=15, then=None):
    parts = [
        part(1, "person", 0, 30, *SEAT, wobble=3),
        part(10, "object", 0, until, *DESK, h=40, label="cup"),  # there from the first second
    ]
    if then:
        parts.append(part(11, "object", *then, *DESK, h=40, label="cup"))
    return tracks(*parts)


def test_type5_an_object_there_from_the_first_frame_is_reported_by_default():
    assert hits_of(FOREIGN_OBJECT_AT_DESK, cup_from_the_start()) != []


def test_type5_the_arrival_test_skips_a_track_that_was_there_before_the_shift_started():
    arrival = Thresholds(arrive_after_sec=2.0)

    assert hits_of(FOREIGN_OBJECT_AT_DESK, cup_from_the_start(), thresholds=arrival) == []


def test_type5_the_arrival_test_judges_each_track_not_the_whole_class():
    arrival = Thresholds(arrive_after_sec=2.0)
    df = cup_from_the_start(until=1, then=(10, 20))  # a stray box at the start, a real cup later

    hits = hits_of(FOREIGN_OBJECT_AT_DESK, df, thresholds=arrival)

    assert [(h.start_sec, h.end_sec) for h in hits] == [(10.0, 19.75)]


def test_type5_silent_for_an_object_that_stays_put_in_almost_every_frame_when_the_mask_is_on():
    df = tracks(
        part(1, "person", 0, 30, *SEAT, wobble=3),
        part(10, "object", 3, 30, *DESK, h=40, label="cup"),  # 90% of the clip, never moves
    )
    mask = Thresholds(static_frac=0.8)

    assert hits_of(FOREIGN_OBJECT_AT_DESK, df) != []  # off by default
    assert hits_of(FOREIGN_OBJECT_AT_DESK, df, thresholds=mask) == []
    assert hits_of(FOREIGN_OBJECT_AT_DESK, on_desk("cup"), thresholds=mask) != []  # 33% of the clip


def test_cash_window_drops_a_lone_cash_box_and_keeps_a_steady_run():
    from src.heuristics.detect import _dense

    mask = np.zeros(40, dtype=bool)
    mask[5] = True  # one box
    mask[15:30] = True  # a steady run
    dense = _dense(mask, 8, 0.5)  # 2 s at 4 fps

    assert not dense[5]
    assert dense[15:30].all()
    assert not dense[:3].any()
    assert not dense[35:].any()


def client_leaves(*, client=(3, 12), cash=(2, 20), cashier=(0, 30), next_client=None):
    """Cashier at the seat, a client at the counter, cash on the desk as given (None = absent)."""
    parts = [part(1, "person", *cashier, *SEAT, wobble=3), part(2, "person", *client, *COUNTER)]
    if cash:
        parts.append(part(10, "cash", *cash, *DESK, h=20))
    if next_client:
        parts.append(part(3, "person", *next_client, *COUNTER))
    return tracks(*parts)


def test_type8_the_client_leaves_and_the_cash_stays_on_the_desk():
    hits = hits_of(CLIENT_LEFT_DURING_COUNT, client_leaves())

    assert [(h.start_sec, h.end_sec) for h in hits] == [(12.0, 19.75)]


def test_type8_silent_when_the_cash_is_gone_after_a_single_frame():
    assert hits_of(CLIENT_LEFT_DURING_COUNT, client_leaves(cash=(2, 12.25))) == []


def test_type8_silent_when_the_cash_is_put_away_before_the_client_leaves():
    assert hits_of(CLIENT_LEFT_DURING_COUNT, client_leaves(cash=(2, 10))) == []


def test_type8_silent_when_there_was_no_cash_while_the_client_was_there():
    assert hits_of(CLIENT_LEFT_DURING_COUNT, client_leaves(cash=(13, 20))) == []


def test_type8_silent_when_the_next_client_steps_up_before_the_cash_lay_alone_long_enough():
    df = client_leaves(next_client=(14, 25))  # 2 s alone, below min_left_sec

    assert hits_of(CLIENT_LEFT_DURING_COUNT, df) == []


def test_type8_fires_for_the_time_the_cash_lay_alone_even_if_the_next_client_came_later():
    df = client_leaves(client=(3, 12), next_client=(21, 30), cash=(2, 22.5))

    hits = hits_of(CLIENT_LEFT_DURING_COUNT, df)

    assert [(h.start_sec, h.end_sec) for h in hits] == [(12.0, 20.75)]


def test_type8_survives_a_passer_by_in_the_frame_right_after_the_client_left():
    df = tracks(
        part(1, "person", 0, 30, *SEAT, wobble=3),
        part(2, "person", 3, 12, *COUNTER),
        part(3, "person", 12, 12.5, *COUNTER),  # a passer-by, two frames
        part(10, "cash", 2, 20, *DESK, h=20),
    )

    hits = hits_of(CLIENT_LEFT_DURING_COUNT, df)

    assert [(h.start_sec, h.end_sec) for h in hits] == [(12.5, 19.75)]


def test_type8_silent_when_the_cashier_is_away_after_the_client_leaves():
    assert hits_of(CLIENT_LEFT_DURING_COUNT, client_leaves(cashier=(0, 11))) == []


def test_type8_silent_when_the_client_is_still_at_the_counter_at_the_end():
    assert hits_of(CLIENT_LEFT_DURING_COUNT, client_leaves(client=(3, 30))) == []


@pytest.mark.parametrize(
    ("type_id", "scene", "stricter"),
    [
        (OUTSIDER_IN_CASH_ROOM, outsider, Thresholds(min_outsider_sec=11.0)),
        (OUTSIDER_IN_CASH_ROOM, outsider, Thresholds(room_radius=0.5)),
        (FOREIGN_OBJECT_AT_DESK, lambda: on_desk("cup"), Thresholds(min_object_sec=11.0)),
        (FOREIGN_OBJECT_AT_DESK, lambda: on_desk("cell phone", conf=0.6), Thresholds(phone_conf=0.7)),
        (CLIENT_LEFT_DURING_COUNT, client_leaves, Thresholds(min_left_sec=9.0)),
    ],
    ids=[
        "min_outsider_sec",
        "room_radius",
        "min_object_sec",
        "phone_conf",
        "min_left_sec",
    ],
)
def test_every_threshold_of_part_2_can_be_passed_in(type_id, scene, stricter):
    df = scene()

    assert hits_of(type_id, df) != []
    assert hits_of(type_id, df, thresholds=stricter) == []


def test_the_book_floor_and_min_object_sec_apply_to_type6():
    df = on_desk("book")

    assert hits_of(VALUABLES_LEFT_ON_DESK, df, thresholds=BOOK_ON) != []
    assert hits_of(VALUABLES_LEFT_ON_DESK, df, thresholds=Thresholds(book_conf=0.95)) == []
    assert (
        hits_of(VALUABLES_LEFT_ON_DESK, df, thresholds=Thresholds(book_conf=0.5, min_object_sec=11.0)) == []
    )


LAPTOP_AT = (200.0, 600.0)  # 1 cashier height from the seat, as the open laptop sits in X-28


def laptop_scene(*, away=(8, 20), laptop=(0, 30), at=LAPTOP_AT, conf=0.9, laptop_drop=0):
    """Cashier at the seat, away for `away` seconds (None = never); a laptop box as given (None = absent)."""
    stays = [(0, 30)] if away is None else [(0, away[0]), (away[1], 30)]
    parts = [part(1, "person", a, b, *SEAT, wobble=3) for a, b in stays if b > a]
    if laptop:
        parts.append(
            part(20, "object", *laptop, *at, h=60.0, label="laptop", conf=conf, drop_every=laptop_drop)
        )
    return tracks(*parts)


def test_laptop_left_open_fires_while_the_laptop_is_on_the_desk_and_the_cashier_is_away():
    hits = hits_of(LAPTOP_LEFT_OPEN, laptop_scene())

    assert [(h.start_sec, h.end_sec) for h in hits] == [(8.0, 19.75)]


def test_laptop_left_open_silent_when_the_cashier_stays_at_the_seat():
    assert hits_of(LAPTOP_LEFT_OPEN, laptop_scene(away=None)) == []


def test_laptop_left_open_silent_when_the_cashier_only_steps_away_for_a_second():
    assert hits_of(LAPTOP_LEFT_OPEN, laptop_scene(away=(8, 9))) == []


def test_laptop_left_open_silent_without_a_laptop_on_the_desk():
    assert hits_of(LAPTOP_LEFT_OPEN, laptop_scene(laptop=None)) == []


def test_laptop_left_open_silent_when_the_laptop_is_seen_for_less_than_the_absence_threshold():
    assert hits_of(LAPTOP_LEFT_OPEN, laptop_scene(laptop=(12, 14))) == []


def test_laptop_left_open_silent_for_a_laptop_far_from_the_cashiers_desk():
    assert hits_of(LAPTOP_LEFT_OPEN, laptop_scene(at=(200.0, 1200.0))) == []  # 4 cashier heights away


def test_laptop_left_open_silent_for_a_doubtful_laptop_box():
    assert hits_of(LAPTOP_LEFT_OPEN, laptop_scene(conf=0.3)) == []


def test_laptop_left_open_silent_before_the_cashier_has_arrived():
    assert hits_of(LAPTOP_LEFT_OPEN, laptop_scene(away=(0, 8))) == []


def test_laptop_left_open_survives_a_missed_laptop_detection_every_two_seconds():
    hits = hits_of(LAPTOP_LEFT_OPEN, laptop_scene(laptop_drop=8))

    assert [h.start_sec for h in hits] == [8.0]


def test_a_laptop_on_the_desk_does_not_make_it_a_foreign_object():
    assert hits_of(FOREIGN_OBJECT_AT_DESK, laptop_scene()) == []


NEXT_CLIENT = {CASH_ON_DESK_NEXT_CLIENT}


def test_the_vote_says_violation_for_a_series_of_n_seconds_and_is_silent_for_a_shorter_one():
    series = [Hit(CASH_ON_DESK_NEXT_CLIENT, 10.0, 19.75)]  # 40 frames at 4 fps = 10 s

    assert vote(series, 10.0, types=NEXT_CLIENT) is True
    assert vote(series, 10.25, types=NEXT_CLIENT) is False
    assert vote([Hit(CASH_ON_DESK_NEXT_CLIENT, 5.0, 5.0)], 1.0, types=NEXT_CLIENT) is False  # one frame
    assert vote([], 1.0, types=NEXT_CLIENT) is False


def test_the_vote_takes_the_longest_hit_not_the_sum():
    two_short = [Hit(CASH_ON_DESK_NEXT_CLIENT, 0.0, 2.75), Hit(CASH_ON_DESK_NEXT_CLIENT, 5.0, 7.75)]

    assert vote(two_short, 4.0, types=NEXT_CLIENT) is False
    assert vote(two_short, 3.0, types=NEXT_CLIENT) is True


def test_only_a_type_inside_the_false_positive_budget_votes_by_default():
    long_hit = {
        t: [Hit(t, 0.0, 30.0)] for t in (*VOTING_TYPES, CASH_ON_DESK_NEXT_CLIENT, CLIENT_LEFT_DURING_COUNT)
    }

    assert VOTING_TYPES <= {OUTSIDER_IN_CASH_ROOM, FOREIGN_OBJECT_AT_DESK}
    assert all(vote(long_hit[t], 3.0) for t in VOTING_TYPES)
    assert not vote(long_hit[CASH_ON_DESK_NEXT_CLIENT], 3.0)  # 7/19 clean videos
    assert not vote(long_hit[CLIENT_LEFT_DURING_COUNT], 3.0)  # 4/19
    assert vote(long_hit[CLIENT_LEFT_DURING_COUNT], 3.0, types={CLIENT_LEFT_DURING_COUNT})


def test_the_vote_follows_detect_end_to_end():
    assert vote(detect(client_change()), 8.0, types=NEXT_CLIENT) is True  # the hit lasts 8 s
    assert vote(detect(client_change()), 8.25, types=NEXT_CLIENT) is False
    assert vote(detect(client_change(cash=(1, 12))), 1.0, types=NEXT_CLIENT) is False


def test_every_object_class_the_heuristics_look_for_has_a_name_for_the_description():
    assert set(FOREIGN_OBJECTS + VALUABLES_OBJECTS) <= set(OBJECT_NAMES)


@pytest.mark.parametrize(
    "type_id",
    [OUTSIDER_IN_CASH_ROOM, FOREIGN_OBJECT_AT_DESK, VALUABLES_LEFT_ON_DESK, CLIENT_LEFT_DURING_COUNT],
)
def test_descriptions_of_part_2_never_claim_hiding_a_phone_camera_or_valuables(type_id):
    # the template only: the cited act's title ends up in the full description and says "ценностей"
    template = next(
        t["description"]
        for t in yaml.safe_load(RULES_YAML.read_text(encoding="utf-8"))["types"]
        if t["id"] == type_id
    )
    texts = [template.format(object=name).lower() for name in OBJECT_NAMES.values()]

    for name in OBJECT_NAMES.values():
        assert name in describe_violation(type_id, {"object": name}).description or "{object}" not in template
    for text in texts:
        for banned in ("сокрыт", "скрыт", "камер", "ценност"):
            assert banned not in text
