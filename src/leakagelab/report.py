"""Author-side plots, only after selection has been frozen and test graded."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve


def plots(root):
    root = Path(root)
    out = root / "results"
    experiments = pd.read_csv(out / "experiments.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    names = experiments.model + experiments.history.map({True: " + history", False: " no history"})
    names.iloc[-1] += " (random diagnostic)"
    ax.barh(names, experiments.average_precision, color="#246887")
    ax.set_xlabel("Average precision on each protocol's validation partition")
    ax.set_title("LeakageLab: validation comparison (not matched test sets)")
    fig.tight_layout()
    fig.savefig(out / "validation_ap.png", dpi=160)
    plt.close(fig)
    submission = pd.read_csv(out / "submission.csv")
    labels = pd.read_csv(root / "data/author/test_labels.csv")
    joined = labels.merge(submission, on="transaction_id", validate="one_to_one")
    precision, recall, _ = precision_recall_curve(joined.target, joined.probability)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label="Selected model: final temporal test")
    ax.axhline(joined.target.mean(), linestyle="--", color="gray", label="Test prevalence")
    ax.set(xlabel="Recall (fraction)", ylabel="Precision (fraction)", title="Synthetic temporal test: precision–recall")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "test_precision_recall.png", dpi=160)
    plt.close(fig)
    raw = pd.read_csv(root / "data/author/raw.csv")
    raw["week"] = pd.to_datetime(raw.transaction_time).dt.strftime("%Y-W%U")
    weekly = raw.groupby("week").target.agg(["size", "mean"])
    weekly.to_csv(out / "weekly_prevalence.csv")
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(range(len(weekly)), weekly["mean"], color="#246887")
    ax.set(xlabel="Week index since 2025-01-01", ylabel="Positive label fraction",
           title="Generated label prevalence over time (author audit)")
    fig.tight_layout()
    fig.savefig(out / "weekly_prevalence.png", dpi=160)
    plt.close(fig)
