import pytest

from mlops import extract


def test_the_committed_extract_matches_the_expected_schema(raw_frame):
    assert extract.ID in raw_frame.columns
    assert extract.LABEL in raw_frame.columns
    assert len(extract.feature_columns(raw_frame)) == 30
    assert len(raw_frame) == 569


def test_the_extract_is_clean_to_begin_with(raw_frame):
    assert set(raw_frame[extract.LABEL].unique()) == extract.VALID_LABELS
    assert not raw_frame[extract.ID].duplicated().any()
    assert not raw_frame.isna().any().any()


def test_feature_columns_excludes_the_id_and_the_label(tiny_frame):
    assert extract.feature_columns(tiny_frame) == ["mean_radius", "mean_area"]


def test_ingest_writes_a_snapshot_for_the_run(staging, tmp_path, tiny_frame):
    source = tmp_path / "wdbc.csv"
    tiny_frame.to_csv(source, index=False)

    meta = extract.ingest(source, staging, "2026-08-25")
    assert meta["rows"] == 4
    assert (staging / "2026-08-25" / "raw.parquet").exists()


def test_ingest_refuses_a_missing_extract(staging, tmp_path):
    with pytest.raises(FileNotFoundError):
        extract.ingest(tmp_path / "absent.csv", staging, "2026-08-25")


def test_ingest_refuses_an_extract_without_the_label(staging, tmp_path, tiny_frame):
    source = tmp_path / "bad.csv"
    tiny_frame.drop(columns=[extract.LABEL]).to_csv(source, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        extract.ingest(source, staging, "2026-08-25")


def test_rerunning_a_date_reuses_that_dates_folder(staging):
    first = extract.run_dir(staging, "2026-08-25")
    second = extract.run_dir(staging, "2026-08-25")
    assert first == second
