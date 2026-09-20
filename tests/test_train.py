from mlops import data
from mlops.train import PRIMARY_METRIC, build_estimator, evaluate


def test_estimator_trains_and_scores_sensibly():
    frame = data.load_reference()
    features = data.feature_columns(frame)
    split = int(len(frame) * 0.75)

    estimator = build_estimator()
    estimator.fit(frame[features][:split], frame[data.TARGET][:split])
    metrics = evaluate(estimator, frame[features][split:], frame[data.TARGET][split:])

    assert {"accuracy", "f1", PRIMARY_METRIC} <= metrics.keys()
    assert 0.0 <= metrics[PRIMARY_METRIC] <= 1.0
    assert metrics["accuracy"] > 0.8
