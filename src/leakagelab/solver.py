"""REFERENCE SOLVER: only solver-visible CSVs, no generator or grader imports."""
import argparse
import importlib.metadata
import json
import os
import platform
import threading
import time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from threadpoolctl import threadpool_limits
from .common import write_json, sha256
from .metrics import metrics

HISTORY = ["account_past_count", "account_past_mean_amount", "account_count_7d",
           "account_amount_7d", "account_hours_since_last", "merchant_matured_count",
           "merchant_matured_rate"]
NUMERIC = ["amount", "foreign", "device_score", "hour", "day_of_week"] + HISTORY
CATEGORICAL = ["channel", "merchant_category"]


def pipeline(kind, seed, history=True):
    numeric = NUMERIC if history else [n for n in NUMERIC if n not in HISTORY]
    if kind == "logistic":
        num = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler())])
        cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        model = LogisticRegression(C=1, max_iter=600, random_state=seed)
    else:
        num = "passthrough"
        cat = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
        if kind == "dummy":
            model = DummyClassifier(strategy="prior")
        else:
            model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
                learning_rate=.08 if kind == "hgb_008" else .15,
                l2_regularization=2, early_stopping=False, random_state=seed,
                categorical_features=list(range(len(numeric), len(numeric) + len(CATEGORICAL))))
    pre = ColumnTransformer([("numeric", num, numeric), ("category", cat, CATEGORICAL)],
                            remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("preprocess", pre), ("model", model)])


def select_threshold(y, probabilities):
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[np.argmax(f1)])


class PeakMemory:
    """Sample process RSS at 20 ms, including preprocessing; excludes child RSS."""
    def __enter__(self):
        self.peak = psutil.Process().memory_info().rss
        self.stop = threading.Event()
        def sample():
            process = psutil.Process()
            while not self.stop.wait(.02):
                self.peak = max(self.peak, process.memory_info().rss)
        self.worker = threading.Thread(target=sample, daemon=True)
        self.worker.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.worker.join()


def fit_evaluate(kind, train, validation, seed, history=True):
    model = pipeline(kind, seed, history)
    start = time.perf_counter()
    with PeakMemory() as memory:
        model.fit(train, train.target)
        probabilities = model.predict_proba(validation)[:, 1]
    threshold = select_threshold(validation.target, probabilities)
    result = {"model": kind, "history": history, "seed": seed,
              "fit_predict_seconds": time.perf_counter() - start,
              "sampled_peak_rss_mb": memory.peak / 1024**2,
              "validation": metrics(validation.target, probabilities, threshold),
              "parameters": model.named_steps["model"].get_params()}
    return model, probabilities, result


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "cpu_logical": os.cpu_count(),
            "packages": {m: importlib.metadata.version(m) for m in
                         ["numpy", "pandas", "scikit-learn", "matplotlib", "psutil", "PyYAML"]},
            "gpu_requested": False, "memory_method": "process RSS sampled every 20 ms"}


def solve(bundle, output, seed=1803, threads=2):
    started = time.perf_counter()
    bundle, output = Path(bundle), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # The full read allowlist. No traversal of parent folders or author files.
    train = pd.read_csv(bundle / "train.csv")
    validation = pd.read_csv(bundle / "validation.csv")
    test = pd.read_csv(bundle / "test.csv")
    sample = pd.read_csv(bundle / "sample_submission.csv")
    assert "target" not in test.columns
    assert set(sample.transaction_id) == set(test.transaction_id)
    experiments, fitted = [], {}
    with threadpool_limits(limits=threads):
        for kind in ["dummy", "logistic", "hgb_008", "hgb_015"]:
            model, probabilities, result = fit_evaluate(kind, train, validation, seed)
            experiments.append(result)
            fitted[kind] = (model, probabilities, result)
        winner = max(experiments, key=lambda r: r["validation"]["average_precision"])
        kind = winner["model"]
        model, vp, _ = fitted[kind]
        # Diagnostics excluded from candidate selection.
        _, _, ablation = fit_evaluate(kind, train, validation, seed, history=False)
        ablation["track"] = "temporal_no_history_ablation"
        experiments.append(ablation)
        pool = pd.concat([train, validation], ignore_index=True)
        rt, rv = train_test_split(pool, test_size=.2, random_state=seed, stratify=pool.target)
        _, _, random_result = fit_evaluate(kind, rt, rv, seed)
        random_result["track"] = "random_split_diagnostic_INVALID_for_future_deployment"
        experiments.append(random_result)
        # Freeze model config and validation threshold before test is graded.
        write_json(output / "selection_lock.json", {"model": kind, "seed": seed,
            "threshold": winner["validation"]["threshold"], "selected_by": "highest validation AP",
            "parameters": winner["parameters"], "refit": False,
            "validation_ids_sha256": sha256(bundle / "validation.csv"),
            "test_ids_sha256": sha256(bundle / "test.csv")})
        pd.DataFrame({"transaction_id": test.transaction_id,
                      "probability": model.predict_proba(test)[:, 1]}).to_csv(output / "submission.csv", index=False)
    pd.DataFrame({"transaction_id": validation.transaction_id, "target": validation.target,
                  "probability": vp}).to_csv(output / "validation_predictions.csv", index=False)
    write_json(output / "experiments.json", experiments)
    pd.DataFrame([{**{k:r[k] for k in ["model", "history", "seed", "fit_predict_seconds", "sampled_peak_rss_mb"]},
                   "track": r.get("track", "temporal_candidate"), **r["validation"]} for r in experiments]).to_csv(output / "experiments.csv", index=False)
    write_json(output / "environment.json", environment())
    write_json(output / "solver_run.json", {"total_seconds": time.perf_counter() - started,
        "bundle": str(bundle), "threads": threads, "selection_sha256": sha256(output / "selection_lock.json"),
        "submission_sha256": sha256(output / "submission.csv"),
        "allowed_inputs": ["train.csv", "validation.csv", "test.csv", "sample_submission.csv"]})
    print(f"Selected {kind}; validation AP={winner['validation']['average_precision']:.6f}; solver wall={time.perf_counter()-started:.2f}s")
    return winner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=1803)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    solve(args.bundle, args.output, args.seed, args.threads)


if __name__ == "__main__":
    main()
