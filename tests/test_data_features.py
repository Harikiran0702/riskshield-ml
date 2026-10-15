import numpy as np
import pandas as pd
import pytest

from riskshield.api.schemas import Transaction
from riskshield.data.generator import generate
from riskshield.features.engineering import FEATURE_NAMES, build_features
from riskshield.models.training import temporal_split


def test_generator_deterministic():
    pd.testing.assert_frame_equal(generate(100), generate(100))
    assert not generate(100, 1).equals(generate(100, 2))


def test_future_events_cannot_change_prefix():
    pd.testing.assert_frame_equal(generate(100), generate(200).iloc[:100])


def test_generated_schema_and_imbalance(frame):
    assert frame.transaction_id.is_unique
    assert 0 < frame.fraud_label.mean() < 0.15
    for row in frame.drop(columns="fraud_label").iloc[:100].to_dict("records"):
        Transaction.model_validate(row)


def test_invalid_generation():
    with pytest.raises(ValueError):
        generate(0)


def test_history_excludes_current_and_future(frame):
    times = pd.to_datetime(frame.timestamp, utc=True)
    for _, customer in frame.groupby("customer_id"):
        for position, (i, row) in enumerate(customer.iterrows()):
            prior = customer.iloc[:position]
            assert row.prior_transaction_count == position
            if len(prior):
                assert row.historical_avg_amount == pytest.approx(prior.transaction_amount.mean())
                in_hour = prior.loc[times[prior.index] >= times[i] - pd.Timedelta(hours=1)]
                assert row.transactions_1h == len(in_hour)
                assert row.spend_1h == pytest.approx(in_hour.transaction_amount.sum())


def test_feature_schema_and_values(frame):
    features = build_features(frame)
    assert features.columns.tolist() == FEATURE_NAMES
    assert np.isfinite(features.to_numpy()).all()
    assert features.iloc[0].log_amount == pytest.approx(np.log1p(frame.iloc[0].transaction_amount))
    assert "fraud_label" not in features
    assert "customer_id" not in features


def test_labels_and_identifiers_do_not_change_features(frame):
    changed = frame.copy()
    changed["fraud_label"] = 1 - changed.fraud_label
    changed["customer_id"] = "IGNORE"
    pd.testing.assert_frame_equal(build_features(frame), build_features(changed))


def test_single_and_offline_features_match(frame):
    pd.testing.assert_frame_equal(build_features(frame).iloc[-1:], build_features(frame.iloc[-1:]))


def test_unknown_category_rejected(frame):
    changed = frame.iloc[:1].copy()
    changed["merchant_category"] = "unknown"
    with pytest.raises(ValueError):
        build_features(changed)


def test_temporal_boundaries(frame):
    train, val, test = temporal_split(frame.sample(frac=1, random_state=3))
    assert train.timestamp.max() < val.timestamp.min() < test.timestamp.min()
    assert set(train.transaction_id).isdisjoint(test.transaction_id)


def test_one_class_split_rejected(frame):
    with pytest.raises(ValueError):
        temporal_split(frame.assign(fraud_label=0))
