"""Label normalization, label contract, label check; stub-pipeline and feature tests."""

import pandas as pd
import pandera.errors
import pytest

from contracts.data import LabelsSchema, StubSchema
from src.config import FEATURES, ID_COL, TARGET
from src.data.check_labels import check_labels, main
from src.data.labels import build_labels, load_labels, normalize_name, parse_timecode
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
    assert ",68\n" in csv and "68.0" not in csv


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
        "[cash partially given to client]",
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
        (0, "Cash on desk"),  # neither an id nor [...]
        (0, "[]"),  # brackets without a scenario
    ],
)
def test_labels_contract_rejects_bad_violation_type(row, value):
    df = typed_labels()
    df.loc[row, "violation_type"] = value
    with pytest.raises(pandera.errors.SchemaError):
        LabelsSchema.validate(df)


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


def test_committed_labels_csv_matches_the_issue():
    """Counts come from issue #21, not from the code: 40 videos, 21 violating (22 labels), 19 clean."""
    labels = load_labels()
    violations = labels[labels["verdict"] == "violation"]
    assert labels["video_id"].nunique() == 40
    assert violations["video_id"].nunique() == 21 and len(violations) == 22
    assert labels["video_id"].nunique() - violations["video_id"].nunique() == 19
    assert labels.loc[labels["video_id"] == "X-4", "second"].tolist() == [57, 68]
    assert not labels["video_id"].isin(["X-47", "X-48"]).any()  # merged into X-4 and X-5
