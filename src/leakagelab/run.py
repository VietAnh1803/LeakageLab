"""Author orchestration: prepare -> separate solver process -> grader -> reports."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from .common import read_config, write_json, sha256
from .prepare import prepare
from .grade import grade


def run(config_path, root, smoke=False):
    start = time.perf_counter()
    config = read_config(config_path)
    if smoke:
        config.update(n_transactions=3000, n_accounts=120, n_merchants=60, bootstrap_replicates=20)
    root = Path(root).resolve()
    manifest = prepare(config, root / "data")
    source_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(source_root),
               OMP_NUM_THREADS=str(config["threads"]), OPENBLAS_NUM_THREADS=str(config["threads"]),
               MKL_NUM_THREADS=str(config["threads"]))
    # Separate working directory contains only solver inputs; input access also
    # audited in tests. This is separation of responsibilities, not a secure sandbox.
    subprocess.run([sys.executable, "-m", "leakagelab.solver", "--bundle", str(root / "data/solver"),
                    "--output", str(root / "results"), "--seed", str(config["seed"]),
                    "--threads", str(config["threads"])], cwd=root / "data/solver", env=env,
                   check=True, timeout=config["resource_budget_seconds"])
    result = grade(root / "data/author/test_labels.csv", root / "results/submission.csv",
                   root / "results/selection_lock.json", root / "data/solver/test.csv",
                   root / "data/solver/train.csv", config["bootstrap_replicates"])
    write_json(root / "results/metrics.json", result)
    write_json(root / "results/split_manifest.json", manifest)
    from .report import plots
    plots(root)
    write_json(root / "results/run_manifest.json", {"mode": "smoke" if smoke else "full",
        "wall_seconds": time.perf_counter() - start, "config": config,
        "order": ["prepare", "solver subprocess / lock", "author grade", "report"],
        "source_hashes": {str(p.relative_to(source_root)).replace("\\", "/"): sha256(p)
                           for p in sorted((source_root / "leakagelab").glob("*.py"))},
        "result_hashes": {p.name: sha256(p) for p in sorted((root / "results").glob("*.json"))}})
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--output", default="artifacts/local")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    run(args.config, args.output, args.smoke)


if __name__ == "__main__":
    main()
