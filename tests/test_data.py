"""Label normalization, label contract, label check; stub-pipeline and feature tests."""

import pandas as pd
import pandera.errors
import pytest
import yaml

from contracts.data import LabelsSchema, StubSchema
from src.config import FEATURES, ID_COL, RULES_YAML, TARGET
from src.data.check_labels import check_labels, main
from src.data.labels import (
    build_labels,
    count_arrangements,
    load_labels,
    normalize_name,
    parse_timecode,
    read_arrangements,
    read_types,
)
from src.data.stub import make_stub
from src.features.build import to_model_frame
from src.models.explain import fmt_kzt


def test_stub_matches_contract():
    StubSchema.validate(make_stub(200))


def test_contract_rejects_bad_rows():
    df = make_stub(10)
    df.loc[0, "age"] = 5
    with pytest.raises(pandera.errors.SchemaError):
        StubSchema.validate(df)


def test_no_leakage_columns_in_features():
    assert TARGET not in FEATURES and ID_COL not in FEATURES


def test_model_frame_categories_fixed():
    X = to_model_frame(pd.DataFrame([make_stub(1).iloc[0]]))
    assert list(X.columns) == FEATURES
    assert len(X["region"].cat.categories) == 6


def test_fmt_kzt_thin_space():
    assert fmt_kzt(1250000) == "1 250 000 ₸"


# \u0425 = Cyrillic Х, \u0058 = Latin X: the two look identical in an editor, so spell them out.
CYR_X, LAT_X = "\u0425", "\u0058"


def test_normalize_name_unifies_cyrillic_and_latin_x():
    assert normalize_name(f"{CYR_X}-47") == normalize_name(f"{LAT_X}-47") == "X-47"


def test_normalize_name_strips_space_and_case():
    assert normalize_name("  \u0445-5 ") == "X-5"
    assert normalize_name("x-1") == "X-1"


@pytest.mark.parametrize(
    ("raw", "seconds"),
    [
        ("22 секунда", 22),
        ("1минута 8 секунда", 68),  # sic: the xlsx has no space after the number
        ("1 минута 8 секунд", 68),
        ("2 минуты", 120),
        (None, None),
        ("", None),
    ],
)
def test_parse_timecode(raw, seconds):
    assert parse_timecode(raw) == seconds


@pytest.mark.parametrize("raw", ["скоро", "8", "1 час", "-5 секунда"])
def test_parse_timecode_rejects_unknown_text(raw):
    with pytest.raises(ValueError):
        parse_timecode(raw)


def raw_labels(*rows):
    """Rows as (file name, timestamp text, verdict text) = the xlsx columns."""
    return pd.DataFrame(rows, columns=["file_name", "timestamp", "verdict_ru"])


def tiny_xlsx_rows():
    return raw_labels(
        (f"{LAT_X}-1", "22 секунда", "нарушение"),
        (f"{CYR_X}-3", None, "без нарушения"),
        (f"{CYR_X}-4", "57 секунда", "нарушение"),
        (f"{CYR_X}-5", None, "без нарушения"),
        (f"{CYR_X}-47", "1минута 8 секунда", "нарушение"),
        (f"{CYR_X}-48", None, "без нарушения"),
    )


def test_build_labels_merges_duplicate_videos():
    out = build_labels(tiny_xlsx_rows())
    assert out["video_id"].tolist() == ["X-1", "X-3", "X-4", "X-4", "X-5"]
    assert out["verdict"].tolist() == ["violation", "no_violation", "violation", "violation", "no_violation"]
    assert out["second"].tolist() == [22, pd.NA, 57, 68, pd.NA]
    assert out["video_id"].nunique() == 4  # 6 xlsx rows, 2 duplicate pairs merged -> 4 videos


def test_build_labels_points_at_the_distributed_file_name():
    files = build_labels(tiny_xlsx_rows()).groupby("video_id")["file"].agg(set)
    assert files["X-1"] == {f"{LAT_X}-1.mov"}  # Latin on disk: normalization must not rewrite it
    assert files["X-4"] == {f"{CYR_X}-4.mov"}  # merged pair -> the canonical (lower number) file only


def test_build_labels_second_is_integer_not_float():
    csv = build_labels(tiny_xlsx_rows()).to_csv(index=False)
    assert ",68\n" in csv.replace("\r\n", "\n") and "68.0" not in csv


def test_build_labels_rejects_merging_violation_with_clean_video():
    rows = raw_labels(
        (f"{CYR_X}-4", "57 секунда", "нарушение"),
        (f"{CYR_X}-47", None, "без нарушения"),
    )
    with pytest.raises(ValueError, match="X-4"):
        build_labels(rows)


def test_build_labels_rejects_unknown_verdict():
    with pytest.raises(ValueError):
        build_labels(raw_labels((f"{CYR_X}-3", None, "возможно")))


def good_labels():
    return pd.DataFrame(
        {
            "video_id": ["X-1", "X-3", "X-4", "X-4"],
            "file": [f"{LAT_X}-1.mov", f"{CYR_X}-3.mov", f"{CYR_X}-4.mov", f"{CYR_X}-4.mov"],
            "verdict": ["violation", "no_violation", "violation", "violation"],
            "second": pd.array([22, None, 57, 68], dtype="Int64"),
        }
    )


def test_labels_contract_accepts_build_output():
    LabelsSchema.validate(build_labels(tiny_xlsx_rows()))
    LabelsSchema.validate(good_labels())


@pytest.mark.parametrize(
    ("row", "column", "value"),
    [
        (0, "verdict", "maybe"),  # exactly two verdict values
        (0, "video_id", f"{CYR_X}-1"),  # not normalized (Cyrillic X)
        (0, "second", pd.NA),  # violation without a second
        (1, "second", 5),  # no_violation with a second
        (0, "second", -3),  # negative second
        (3, "second", 57),  # same second twice for one video
    ],
)
def test_labels_contract_rejects_bad_rows(row, column, value):
    df = good_labels()
    df.loc[row, column] = value
    with pytest.raises(pandera.errors.SchemaError):
        LabelsSchema.validate(df)


def typed_labels():
    types = [
        "cash_partially_given",
        None,
        "cash_on_desk_next_client",
        "cash_on_desk_next_client",
    ]
    return good_labels().assign(violation_type=types)


def test_labels_contract_accepts_violation_type():
    LabelsSchema.validate(typed_labels())


@pytest.mark.parametrize(
    ("row", "value"),
    [
        (0, None),  # violation without a type
        (1, "cash_on_desk_next_client"),  # no_violation with a type
        (0, " "),  # blank type
        (0, "worker left the key on a safe"),  # free text, not an id from rules.yaml
        (0, "cash_on_desk_next_client "),  # id with trailing space
    ],
)
def test_labels_contract_rejects_bad_violation_type(row, value):
    df = typed_labels()
    df.loc[row, "violation_type"] = value
    with pytest.raises(pandera.errors.SchemaError):
        LabelsSchema.validate(df)


def tiny_types():
    return pd.DataFrame(
        {
            "video_id": ["X-1", "X-4", "X-4"],
            "second": [22, 57, 68],
            "violation_type": [
                "cash_partially_given",
                "cash_on_desk_next_client",
                "cash_on_desk_next_client",
            ],
        }
    )


def test_build_labels_adds_types_and_keeps_row_order():
    out = build_labels(tiny_xlsx_rows(), tiny_types())
    assert out["video_id"].tolist() == ["X-1", "X-3", "X-4", "X-4", "X-5"]
    assert out["violation_type"].fillna("-").tolist() == [
        "cash_partially_given",
        "-",
        "cash_on_desk_next_client",
        "cash_on_desk_next_client",
        "-",
    ]


def test_build_labels_rejects_violation_without_type():
    with pytest.raises(ValueError, match=r"without a type.*X-4.*68"):
        build_labels(tiny_xlsx_rows(), tiny_types().iloc[:2])


def test_build_labels_rejects_type_for_no_label():
    orphan = pd.concat([tiny_types(), tiny_types().iloc[[0]].assign(second=23)], ignore_index=True)
    with pytest.raises(ValueError, match=r"match no violation label.*X-1.*23"):
        build_labels(tiny_xlsx_rows(), orphan)


def test_build_labels_rejects_two_types_for_one_label():
    twice = pd.concat([tiny_types(), tiny_types().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="two types"):
        build_labels(tiny_xlsx_rows(), twice)


def test_build_labels_rejects_unknown_type_id():
    with pytest.raises(pandera.errors.SchemaError):
        build_labels(tiny_xlsx_rows(), tiny_types().assign(violation_type="left the door open"))


def arranged_labels():
    return good_labels().assign(arrangement=["back", "back", "facing", "facing"])


def test_labels_contract_accepts_arrangement():
    LabelsSchema.validate(arranged_labels())


@pytest.mark.parametrize(
    ("row", "value"),
    [
        (0, None),  # no arrangement
        (0, "sideways"),  # not one of the two values
        (0, "Back"),  # wrong case
        (3, "back"),  # X-4 has two rows with two arrangements
    ],
)
def test_labels_contract_rejects_bad_arrangement(row, value):
    df = arranged_labels()
    df.loc[row, "arrangement"] = value
    with pytest.raises(pandera.errors.SchemaError):
        LabelsSchema.validate(df)


def tiny_arrangements():
    return pd.DataFrame(
        {"video_id": ["X-1", "X-3", "X-4", "X-5"], "arrangement": ["back", "back", "facing", "facing"]}
    )


def test_build_labels_adds_arrangement_to_every_row_of_a_video():
    out = build_labels(tiny_xlsx_rows(), tiny_types(), tiny_arrangements())
    assert list(out.columns)[-2:] == ["violation_type", "arrangement"]
    assert out["video_id"].tolist() == ["X-1", "X-3", "X-4", "X-4", "X-5"]
    assert out["arrangement"].tolist() == ["back", "back", "facing", "facing", "facing"]


def test_build_labels_rejects_video_without_arrangement():
    with pytest.raises(ValueError, match=r"without an arrangement.*X-5"):
        build_labels(tiny_xlsx_rows(), tiny_types(), tiny_arrangements().iloc[:3])


def test_build_labels_rejects_arrangement_for_no_video():
    orphan = pd.concat([tiny_arrangements(), pd.DataFrame({"video_id": ["X-47"], "arrangement": ["back"]})])
    with pytest.raises(ValueError, match=r"match no video.*X-47"):  # alias ids are merged away
        build_labels(tiny_xlsx_rows(), tiny_types(), orphan)


def test_build_labels_rejects_two_arrangements_for_one_video():
    twice = pd.concat([tiny_arrangements(), pd.DataFrame({"video_id": ["X-1"], "arrangement": ["facing"]})])
    with pytest.raises(ValueError, match=r"two arrangements.*X-1"):
        build_labels(tiny_xlsx_rows(), tiny_types(), twice)


def test_build_labels_rejects_unknown_arrangement():
    with pytest.raises(pandera.errors.SchemaError):
        build_labels(tiny_xlsx_rows(), tiny_types(), tiny_arrangements().assign(arrangement="sideways"))


def test_count_arrangements_counts_videos_not_label_rows():
    # X-4 has two violation rows but is one video
    assert count_arrangements(arranged_labels()) == {
        "back": {"violation": 1, "no_violation": 1},
        "facing": {"violation": 1, "no_violation": 0},
    }


@pytest.fixture
def labels_dir(tmp_path):
    """A labels.csv plus a videos folder holding exactly the files it names."""
    labels = good_labels()
    labels.to_csv(tmp_path / "labels.csv", index=False)
    videos = tmp_path / "videos"
    videos.mkdir()
    for name in labels["file"].unique():
        (videos / name).write_bytes(b"")
    return tmp_path


def test_check_labels_counts_videos_and_violations(labels_dir):
    counts = check_labels(labels_dir / "labels.csv", labels_dir / "videos")
    assert counts == {"videos": 3, "violation_videos": 2, "violation_labels": 3, "clean_videos": 1}


def test_check_labels_fails_on_missing_file(labels_dir):
    (labels_dir / "videos" / f"{CYR_X}-3.mov").unlink()
    with pytest.raises(FileNotFoundError, match="-3.mov"):
        check_labels(labels_dir / "labels.csv", labels_dir / "videos")


def test_check_labels_fails_on_contract_violation(labels_dir):
    bad = good_labels().assign(verdict="maybe")
    bad.to_csv(labels_dir / "labels.csv", index=False)
    with pytest.raises(pandera.errors.SchemaError):
        check_labels(labels_dir / "labels.csv", labels_dir / "videos")


def test_check_labels_cli_prints_counts_and_exits_nonzero_on_missing_file(labels_dir, capsys):
    argv = [str(labels_dir / "labels.csv"), str(labels_dir / "videos")]
    main(argv)
    assert "3 видео" in capsys.readouterr().out
    (labels_dir / "videos" / f"{LAT_X}-1.mov").unlink()
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code not in (0, None)


def test_check_labels_cli_prints_violations_per_type_with_zeros(labels_dir, capsys):
    typed_labels().to_csv(labels_dir / "labels.csv", index=False)
    main([str(labels_dir / "labels.csv"), str(labels_dir / "videos")])
    lines = capsys.readouterr().out.splitlines()
    assert "cash_on_desk_next_client: 2" in lines
    assert "cash_partially_given: 1" in lines
    assert "other: 0" in lines  # a type with no label is still listed


def test_check_labels_cli_prints_videos_per_arrangement_and_verdict(labels_dir, capsys):
    arranged_labels().to_csv(labels_dir / "labels.csv", index=False)
    main([str(labels_dir / "labels.csv"), str(labels_dir / "videos")])
    lines = capsys.readouterr().out.splitlines()
    assert "back: 2 (1 / 1)" in lines
    assert "facing: 1 (1 / 0)" in lines


def test_committed_arrangements_cover_the_committed_labels_exactly():
    """data/arrangements.csv is what a rebuild reads; labels.csv must carry exactly it (hand-checked: 14 back, 26 facing)."""
    arrangements = read_arrangements()
    per_video = load_labels().groupby("video_id")["arrangement"].first()
    assert per_video.to_dict() == arrangements.set_index("video_id")["arrangement"].to_dict()
    assert arrangements["arrangement"].value_counts().to_dict() == {"facing": 26, "back": 14}


def test_committed_labels_csv_matches_the_issue():
    """Counts come from issue #21, not from the code: 40 videos, 21 violating (22 labels), 19 clean."""
    labels = load_labels()
    violations = labels[labels["verdict"] == "violation"]
    assert labels["video_id"].nunique() == 40
    assert violations["video_id"].nunique() == 21 and len(violations) == 22
    assert labels["video_id"].nunique() - violations["video_id"].nunique() == 19
    assert labels.loc[labels["video_id"] == "X-4", "second"].tolist() == [57, 68]
    assert not labels["video_id"].isin(["X-47", "X-48"]).any()  # merged into X-4 and X-5


def test_committed_types_file_is_what_labels_csv_holds():
    """Rebuilding labels.csv must not lose or change a type: the types file is the source, the csv a copy."""
    labels = load_labels()
    violations = labels[labels["verdict"] == "violation"]
    types = read_types()
    assert len(types) == len(violations) == 22
    assert set(zip(types["video_id"], types["second"], types["violation_type"], strict=True)) == set(
        zip(violations["video_id"], violations["second"], violations["violation_type"], strict=True)
    )


def test_labelled_types_match_the_videos_of_each_rules_type():
    """Every violation label carries the type whose `videos` lists its clip, and every listed clip is labelled so."""
    rules = yaml.safe_load(RULES_YAML.read_text(encoding="utf-8"))
    by_rules = {(v, t["id"]) for t in rules["types"] for v in t.get("videos", [])}
    labels = load_labels()
    violations = labels[labels["verdict"] == "violation"]
    assert set(zip(violations["video_id"], violations["violation_type"], strict=True)) == by_rules
