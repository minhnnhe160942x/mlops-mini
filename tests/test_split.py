from mlops import extract, split


def test_the_same_id_always_lands_in_the_same_bucket():
    assert split.bucket("WDBC-0001") == split.bucket("WDBC-0001")


def test_different_ids_spread_across_buckets():
    buckets = {split.bucket(f"WDBC-{i:04d}") for i in range(200)}
    assert len(buckets) > 50


def test_the_split_is_roughly_the_requested_size(raw_frame):
    is_test = split.test_mask(raw_frame, test_fraction=0.20)
    assert 0.15 < is_test.mean() < 0.25


def test_the_split_does_not_move_between_runs(raw_frame):
    first = split.test_mask(raw_frame, 0.20)
    second = split.test_mask(raw_frame, 0.20)
    assert first.equals(second)


def test_adding_rows_does_not_reshuffle_existing_ones(raw_frame):
    before = split.test_mask(raw_frame, 0.20)
    grown = raw_frame.copy()
    grown.loc[len(grown)] = grown.iloc[0].copy()
    grown.loc[len(grown) - 1, extract.ID] = "WDBC-9999"
    after = split.test_mask(grown, 0.20)
    assert after.iloc[: len(raw_frame)].equals(before)


def test_split_writes_both_halves(snapshot, staging):
    sizes = split.split(snapshot, staging, "2026-08-25", test_fraction=0.20)
    assert sizes["train"] + sizes["test"] == 569
    assert (staging / "2026-08-25" / "train_unscaled.parquet").exists()
    assert (staging / "2026-08-25" / "test_unscaled.parquet").exists()
