"""AUTHOR ONLY: point-in-time features and purged temporal partitions."""
import argparse
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd
from .common import read_config, write_json, frame_hash, sha256
from .generate_data import generate

HISTORY = ["account_past_count", "account_past_mean_amount", "account_count_7d",
           "account_amount_7d", "account_hours_since_last", "merchant_matured_count",
           "merchant_matured_rate"]


def features(raw, label_freeze):
    raw = raw.sort_values("transaction_time").reset_index(drop=True)
    ts = pd.to_datetime(raw.transaction_time, utc=True).astype("int64").to_numpy()
    available = pd.to_datetime(raw.label_available_at, utc=True).astype("int64").to_numpy()
    freeze = pd.Timestamp(label_freeze).value
    counts, sums, last = defaultdict(int), defaultdict(float), {}
    queues = defaultdict(deque)
    label_count, label_sum = defaultdict(int), defaultdict(int)
    order = np.argsort(available, kind="stable")
    pointer = 0
    rows = []
    for i, r in enumerate(raw.itertuples(index=False)):
        now = ts[i]
        # Strictly earlier than prediction time AND author training cutoff.
        while pointer < len(raw) and available[order[pointer]] < min(now, freeze):
            j = order[pointer]
            merchant = raw.iloc[j].merchant_id
            label_count[merchant] += 1
            label_sum[merchant] += int(raw.iloc[j].target)
            pointer += 1
        q = queues[r.account_id]
        while q and q[0][0] < now - 7 * 86400 * 10**9:
            q.popleft()
        c = counts[r.account_id]
        mc = label_count[r.merchant_id]
        rows.append([c, sums[r.account_id] / c if c else np.nan, len(q),
                     sum(x[1] for x in q),
                     (now - last[r.account_id]) / (3600 * 10**9) if c else np.nan,
                     mc, (label_sum[r.merchant_id] + 1) / (mc + 20)])
        counts[r.account_id] += 1
        sums[r.account_id] += r.amount
        last[r.account_id] = now
        q.append((now, r.amount))
    visible = raw.drop(columns=["target", "label_available_at", "adverse_event_at"]).copy()
    visible["hour"] = pd.to_datetime(visible.transaction_time).dt.hour
    visible["day_of_week"] = pd.to_datetime(visible.transaction_time).dt.dayofweek
    for j, name in enumerate(HISTORY):
        visible[name] = np.asarray(rows)[:, j]
    return visible


def split_masks(raw, config):
    start = pd.Timestamp(config["start"])
    train_cut = start + pd.Timedelta(days=config["train_cutoff_day"])
    test_cut = start + pd.Timedelta(days=config["test_cutoff_day"])
    time = pd.to_datetime(raw.transaction_time, utc=True)
    available = pd.to_datetime(raw.label_available_at, utc=True)
    return {"train": (time < train_cut) & (available < train_cut),
            "validation": (time >= train_cut) & (time < test_cut) & (available < test_cut),
            "test": time >= test_cut}


def prepare(config, root):
    root = Path(root)
    author, bundle = root / "author", root / "solver"
    author.mkdir(parents=True, exist_ok=True)
    bundle.mkdir(parents=True, exist_ok=True)
    raw = generate(config)
    freeze = pd.Timestamp(config["start"]) + pd.Timedelta(days=config["train_cutoff_day"])
    x = features(raw, freeze)
    masks = split_masks(raw, config)
    manifest = {"config": config, "generator_hash": frame_hash(raw), "features_hash": frame_hash(x),
                "label_freeze": freeze.isoformat(), "partitions": {}, "files": {}}
    raw.to_csv(author / "raw.csv", index=False)
    for name, mask in masks.items():
        part = x.loc[mask].copy()
        labeled = raw.loc[mask, ["transaction_id", "target"]]
        if name != "test":
            part["target"] = labeled.target.to_numpy()
        part.to_csv(bundle / f"{name}.csv", index=False)
        if name == "test":
            labeled.to_csv(author / "test_labels.csv", index=False)
        manifest["partitions"][name] = {
            "n": int(mask.sum()), "positives": int(labeled.target.sum()),
            "prevalence": float(labeled.target.mean()),
            "start": raw.loc[mask, "transaction_time"].min().isoformat(),
            "end": raw.loc[mask, "transaction_time"].max().isoformat(),
            "max_label_availability": raw.loc[mask, "label_available_at"].max().isoformat(),
            "ids_hash": frame_hash(labeled[["transaction_id"]]),
        }
    pd.DataFrame({"transaction_id": x.loc[masks["test"], "transaction_id"],
                  "probability": 0.5}).to_csv(bundle / "sample_submission.csv", index=False)
    for path in sorted(root.rglob("*.csv")):
        manifest["files"][str(path.relative_to(root)).replace("\\", "/")] = sha256(path)
    manifest["purged_rows"] = int(len(raw) - sum(m.sum() for m in masks.values()))
    write_json(author / "split_manifest.json", manifest)
    # Solver-facing manifest deliberately excludes class balance / ground truth.
    write_json(bundle / "bundle_manifest.json", {"task": "adverse event within seven days",
        "label_freeze": freeze.isoformat(), "files": {k:v for k,v in manifest["files"].items() if k.startswith("solver/")}})
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--output", default="data")
    args = p.parse_args()
    print(prepare(read_config(args.config), args.output)["partitions"])


if __name__ == "__main__":
    main()
