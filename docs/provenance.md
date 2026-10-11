# Provenance and public release

LeakageLab is an independent portfolio project by Nguyen Viet Anh, developed with
Codex AI assistance in implementation, tests, and documentation. Labels and scores
come from executable NumPy/sklearn code, not an LLM.

All transactions, accounts, merchants, timestamps, and outcomes are synthetic.
No third-party competition data or company records are included. The user authorized
publication of source, notebooks, and experiment results on GitHub on 11 October 2026.

The public input bundle includes labeled train/validation, unlabeled future test,
sample submission, and checksum manifest. It excludes final-test ground truth and
raw author data. The generator is public and can reproduce labels, so the holdout
is not secret against reverse engineering.

The original author experiment completed in private Kaggle notebook version 2,
run 357285783. Its full working directory and private source input remain private.
The public reference notebook reads only the solver-visible bundle and provides
a runnable demonstration without private-account access.

No CV, company documents, credentials, or tool caches are included in the release.
No leaderboard submission, job application, Eris runtime compatibility, or Eris
acceptance is claimed. Official reuse would require checking the current contributor
rules on originality, AI assistance, licensing, and runtime contracts.

Original external tooling references:
- [Kaggle kernels CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)
- [Kaggle kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)

Recorded metrics, source/data hashes, environment details, and partition counts are
retained in the public verification record and `artifacts/kaggle/results/`.
