"""AUTHOR ONLY: strict ID-joined scoring, never positional scoring."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from .metrics import metrics
from .common import write_json, sha256


def validate_submission(submission, labels):
    if list(submission.columns) != ["transaction_id", "probability"]:
        raise ValueError("Expected exactly transaction_id,probability in that order")
    if labels.transaction_id.isna().any() or labels.transaction_id.duplicated().any():
        raise ValueError("Invalid author ID set")
    if submission.transaction_id.isna().any() or submission.transaction_id.duplicated().any():
        raise ValueError("Missing or duplicate transaction IDs")
    if set(submission.transaction_id) != set(labels.transaction_id):
        raise ValueError("Missing or extra transaction IDs")
    if not pd.api.types.is_numeric_dtype(submission.probability):
        raise ValueError("Probabilities must be numeric")
    p = submission.probability.to_numpy(dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and within [0,1]")
    return labels.merge(submission, on="transaction_id", validate="one_to_one")


def account_bootstrap(joined, replicates, seed):
    # Whole account trajectories resampled, retaining within-account dependence.
    # Does not capture shared merchant/time shocks; conditional synthetic CI.
    rng = np.random.default_rng(seed)
    groups = list(joined.groupby("account_id", sort=True).indices.values())
    scores = []
    for _ in range(replicates):
        ix = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        y = joined.target.to_numpy()[ix]
        if len(np.unique(y)) == 2:
            scores.append(average_precision_score(y, joined.probability.to_numpy()[ix]))
    return {"method": "account cluster percentile bootstrap, conditional on one synthetic seed",
            "replicates": replicates, "valid_replicates": len(scores),
            "ap_95_interval": np.quantile(scores, [.025, .975]).tolist(),
            "limitation": "Shared merchant/time shocks and generator uncertainty are not covered."}


def grade(labels_path, submission_path, lock_path, test_features_path=None, train_path=None, replicates=200):
    lock = json.loads(Path(lock_path).read_text())
    # Preserve the saved float exactly: one ULP can flip a locked-threshold tie.
    joined = validate_submission(pd.read_csv(submission_path, float_precision="round_trip"), pd.read_csv(labels_path))
    result = {"test": metrics(joined.target, joined.probability, lock["threshold"]),
              "model": lock["model"], "selection_lock_sha256": sha256(lock_path),
              "submission_sha256": sha256(submission_path), "secondary": {}}
    if test_features_path is not None:
        features = pd.read_csv(test_features_path)
        joined = joined.merge(features[["transaction_id", "account_id", "merchant_id"]],
                               on="transaction_id", validate="one_to_one")
        result["uncertainty"] = account_bootstrap(joined, replicates, lock["seed"])
        if train_path is not None:
            train = pd.read_csv(train_path)
            for entity in ["account_id", "merchant_id"]:
                unseen = ~joined[entity].isin(train[entity])
                for name, mask in [("unseen", unseen), ("seen", ~unseen)]:
                    subset = joined.loc[mask]
                    if len(subset):
                        result["secondary"][f"{name}_{entity}"] = metrics(subset.target, subset.probability, lock["threshold"])
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--submission", required=True)
    p.add_argument("--lock", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--test-features")
    p.add_argument("--train")
    args = p.parse_args()
    result = grade(args.labels, args.submission, args.lock, args.test_features, args.train)
    write_json(args.output, result)
    print(json.dumps(result["test"], indent=2))


if __name__ == "__main__":
    main()
