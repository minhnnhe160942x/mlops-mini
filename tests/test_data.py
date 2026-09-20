from mlops import data


def test_reference_has_a_target_and_no_spaces_in_column_names():
    frame = data.load_reference()
    assert data.TARGET in frame.columns
    assert not any(" " in c for c in frame.columns)
    assert len(frame) > 0


def test_split_is_disjoint_and_covers_the_frame():
    frame = data.load_reference()
    reference, pool = data.split_reference_and_pool(frame)
    assert len(reference) + len(pool) == len(frame)
    assert not reference.empty and not pool.empty


def test_batch_respects_the_requested_size():
    _, pool = data.split_reference_and_pool(data.load_reference())
    assert len(data.make_batch(pool, n_rows=50)) == 50


def test_batch_is_capped_by_the_pool_size():
    _, pool = data.split_reference_and_pool(data.load_reference())
    assert len(data.make_batch(pool, n_rows=10**6)) == len(pool)


def test_shift_leaves_the_label_untouched():
    _, pool = data.split_reference_and_pool(data.load_reference())
    batch = data.make_batch(pool, n_rows=100, shift=2.0)
    assert set(batch[data.TARGET].unique()) <= {0, 1}


def test_feature_columns_excludes_the_target():
    frame = data.load_reference()
    assert data.TARGET not in data.feature_columns(frame)
