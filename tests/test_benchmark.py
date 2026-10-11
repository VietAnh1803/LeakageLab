import builtins
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss, brier_score_loss
from threadpoolctl import threadpool_limits
from leakagelab.common import read_config, frame_hash
from leakagelab.generate_data import generate
from leakagelab.prepare import features, split_masks, prepare, HISTORY
from leakagelab.solver import pipeline, solve, select_threshold
from leakagelab.grade import validate_submission, grade
from leakagelab.metrics import metrics


@pytest.fixture
def config():
    cfg = read_config(Path(__file__).parents[1] / "configs/config.yaml")
    # Account/merchant novelty pools must fit this smaller test configuration.
    cfg.update(n_transactions=1800, n_accounts=180, n_merchants=60, bootstrap_replicates=10)
    return cfg


def test_generator_and_split_hashes_reproduce(config, tmp_path):
    a, b = prepare(config, tmp_path / "a"), prepare(config, tmp_path / "b")
    assert a == b
    assert a["purged_rows"] > 0
    assert a["generator_hash"] == frame_hash(generate(config))


def test_boundaries_maturity_and_partition_disjoint(config):
    raw = generate(config)
    masks = split_masks(raw, config)
    start = pd.Timestamp(config["start"])
    train_cut = start + pd.Timedelta(days=108)
    test_cut = start + pd.Timedelta(days=144)
    assert (raw.loc[masks["train"], "label_available_at"] < train_cut).all()
    assert (raw.loc[masks["validation"], "label_available_at"] < test_cut).all()
    assert (raw.loc[masks["validation"], "transaction_time"] >= train_cut).all()
    assert (raw.loc[masks["test"], "transaction_time"] >= test_cut).all()
    assert not (masks["train"] & masks["validation"]).any()
    assert not (masks["validation"] & masks["test"]).any()
    # Explicit exact-cutoff fixture checks strict inequality.
    toy = raw.iloc[:3].copy()
    toy["transaction_time"] = [train_cut - pd.Timedelta(days=7), train_cut, test_cut]
    toy["label_available_at"] = toy.transaction_time + pd.Timedelta(days=7)
    toy_masks = split_masks(toy, config)
    assert not toy_masks["train"].iloc[0]
    assert toy_masks["validation"].iloc[1]
    assert toy_masks["test"].iloc[2]


def test_future_rows_do_not_change_past_features(config):
    raw = generate(config)
    freeze = pd.Timestamp(config["start"]) + pd.Timedelta(days=108)
    past = features(raw.iloc[:900].copy(), freeze)
    corrupted = raw.copy()
    corrupted.loc[900:, "amount"] *= 1000
    corrupted.loc[900:, "target"] = 1 - corrupted.loc[900:, "target"]
    all_x = features(corrupted, freeze)
    pd.testing.assert_frame_equal(past, all_x.iloc[:900].reset_index(drop=True))


def test_history_strictly_prior_and_matured_labels_only(config):
    raw = generate(config).iloc[:3].copy()
    t = pd.Timestamp("2025-01-01T00:00:00Z")
    raw["transaction_time"] = [t, t + pd.Timedelta(days=7), t + pd.Timedelta(days=7, seconds=1)]
    raw["label_available_at"] = raw.transaction_time + pd.Timedelta(days=7)
    raw["account_id"] = "same"
    raw["merchant_id"] = "same"
    raw["amount"] = [10., 20., 30.]
    raw["target"] = [1, 0, 1]
    x = features(raw, t + pd.Timedelta(days=100))
    assert x.account_past_count.tolist() == [0, 1, 2]
    assert x.merchant_matured_count.tolist() == [0, 0, 1]
    assert x.account_amount_7d.tolist() == [0, 10, 20]
    assert x.merchant_matured_rate.iloc[2] == pytest.approx(2/21)
    frozen = features(raw, t + pd.Timedelta(days=7))
    assert frozen.merchant_matured_count.tolist() == [0, 0, 0]


def test_preprocessing_fit_only_training(config):
    raw = generate(config)
    x = features(raw, pd.Timestamp(config["start"]) + pd.Timedelta(days=108))
    masks = split_masks(raw, config)
    train, validation = x.loc[masks["train"]], x.loc[masks["validation"]].copy()
    validation["channel"] = "validation_only_category"
    validation["device_score"] = 1e9
    model = pipeline("logistic", 1803)
    with threadpool_limits(2):
        model.fit(train, raw.loc[masks["train"], "target"])
        imputer = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
        median_before = imputer.statistics_.copy()
        model.predict_proba(validation)
    assert median_before[2] == pytest.approx(train.device_score.median())
    np.testing.assert_array_equal(imputer.statistics_, median_before)
    categories = model.named_steps["preprocess"].named_transformers_["category"].categories_[0]
    assert "validation_only_category" not in categories


@pytest.fixture
def labels_submission():
    return (pd.DataFrame({"transaction_id": ["a", "b", "c", "d"], "target": [0, 1, 0, 1]}),
            pd.DataFrame({"transaction_id": ["a", "b", "c", "d"], "probability": [.1, .6, .8, .9]}))


def test_shuffled_submission_and_metric_parity(labels_submission):
    labels, sub = labels_submission
    a = validate_submission(sub, labels)
    b = validate_submission(sub.sample(frac=1, random_state=4), labels)
    pd.testing.assert_frame_equal(a, b)
    m = metrics(a.target, a.probability, .5)
    assert m["average_precision"] == average_precision_score(a.target, a.probability)
    assert m["roc_auc"] == roc_auc_score(a.target, a.probability)
    assert m["log_loss"] == log_loss(a.target, a.probability)
    assert m["brier"] == brier_score_loss(a.target, a.probability)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "extra", "schema", "nan", "inf", "negative", "above_one", "strings", "null_id"])
def test_invalid_submission_rejected(labels_submission, mutation):
    labels, sub = labels_submission
    if mutation == "duplicate": sub.loc[1, "transaction_id"] = "a"
    elif mutation == "missing": sub = sub.iloc[:-1]
    elif mutation == "extra": sub.loc[4] = ["extra", .5]
    elif mutation == "schema": sub["unexpected"] = 1
    elif mutation == "null_id": sub.loc[0, "transaction_id"] = None
    else:
        bad = {"nan": np.nan, "inf": np.inf, "negative": -.01, "above_one": 1.01, "strings": "bad"}[mutation]
        if mutation == "strings": sub["probability"] = sub.probability.astype(object)
        sub.loc[0, "probability"] = bad
    with pytest.raises(ValueError):
        validate_submission(sub, labels)


def test_solver_visible_only_end_to_end(config, tmp_path, monkeypatch):
    prepare(config, tmp_path / "benchmark")
    bundle = tmp_path / "benchmark/solver"
    allowed = {str((bundle / name).resolve()) for name in
               ["train.csv", "validation.csv", "test.csv", "sample_submission.csv"]}
    original = pd.read_csv
    opened = []
    def guarded(path, *args, **kwargs):
        resolved = str(Path(path).resolve())
        assert resolved in allowed, f"Solver attempted unauthorized input: {resolved}"
        opened.append(resolved)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", guarded)
    with threadpool_limits(2):
        solve(bundle, tmp_path / "solution")
    assert set(opened) == allowed
    monkeypatch.setattr(pd, "read_csv", original)
    result = grade(tmp_path / "benchmark/author/test_labels.csv", tmp_path / "solution/submission.csv",
                   tmp_path / "solution/selection_lock.json")
    assert 0 <= result["test"]["average_precision"] <= 1
    source = Path(__file__).parents[1] / "src/leakagelab/solver.py"
    assert "from .generate_data" not in source.read_text()
    assert "from .grade" not in source.read_text()


def test_single_class_auc_and_validation_threshold():
    assert metrics([0, 0], [.2, .3], .5)["roc_auc"] is None
    threshold = select_threshold([0, 1, 0, 1], [.1, .7, .2, .9])
    assert threshold == .7
