# LeakageLab — Temporal Risk Benchmark & Evaluation Harness

This is an independent, AI-assisted portfolio benchmark on synthetic transactions.
It is not a Kaggle competition, an Eris submission, or a validated real-world fraud system.

## Task and prediction contract

For every transaction, predict the probability of at least one synthetic adverse
event in `(transaction_time, transaction_time + 7 days]`. Prediction occurs at
`transaction_time` in UTC. A binary label is mature only after the complete
seven-day observation horizon. Even a positive event that occurs earlier is not
released to the modeling process until the horizon has finished. This conservative
policy avoids outcome-dependent observation delays.

Current transaction amount, foreign indicator, channel, merchant category, noisy
device score, and calendar features are known at prediction time. Account histories
use strictly earlier transactions, including those in purge gaps. A rolling window
includes prior transactions with timestamps `>= t - 7 days` and `< t`. Merchant
target statistics use only labels with `label_available_at < min(t, 2025-04-19 UTC)`.
No target-history updates from validation or test outcomes are permitted. The prior
is fixed, `(positive_count + 1) / (matured_count + 20)`, and does not estimate a
global prior from future labels.

## Deployment assumption and split

The main track predicts future transactions from both returning and newly arriving
entities. Accounts may repeat across partitions. Unseen-account and unseen-merchant
scores are separate secondary diagnostics; they do not replace the temporal track.

Training cutoff: 2025-04-19 00:00 UTC. Train transactions must have their labels
strictly available before this cutoff, so the final seven days before it are purged.
Validation starts at that cutoff; its labels must be available before the test
cutoff, 2025-05-25 00:00 UTC. The seven days immediately before test are purged.
Test starts on May 25 and runs to the end of the generated transaction period.
Authors observe test labels through July 6. Exact counts, realized ranges, positive
counts, and hashes appear in `artifacts/local/results/split_manifest.json`.

## Participant inputs

The solver-visible bundle contains `train.csv` and `validation.csv` with target,
`test.csv` without target, `sample_submission.csv`, and an informational bundle
manifest. The validation subset is explicitly designated for model and threshold
selection. No raw generator data, adverse-event timestamps, label-availability
timestamps, latent factors, or final-test labels are solver inputs. The reference
solver reads exactly the four CSVs. It ignores entity IDs as predictors but retains
them for output matching and author subgroup analysis.

The author notebook regenerates data and invokes the solver in another process.
This tests the intended input contract; it is not an OS-level adversarial sandbox.
The standalone reference notebook embeds only solver modules. A public generator
can reproduce all labels, so removing ground-truth CSVs does not prevent reverse
engineering. No secure private leaderboard or anti-cheating guarantee is claimed.

## Submission and scoring

Submit UTF-8 CSV with exactly `transaction_id,probability`, in that column order.
Each test ID must appear once and probabilities must be numeric, finite, and in
`[0,1]`. Rows may be shuffled. The grader joins by ID and rejects duplicates,
missing/extra/null IDs, incorrect schema, NaN/Inf, strings, and out-of-range values.

Primary metric: **average precision (AP), higher is better**. It is the sklearn
non-interpolated sum `AP = sum_n (R_n - R_(n-1)) P_n`, where `P_n` and `R_n` are
precision and recall at distinct score thresholds. This is not trapezoidal PR-AUC.
Secondary metrics: ROC-AUC (null if only one class), natural-log binary log loss,
mean squared probability error (Brier), and precision/recall at a threshold chosen
by maximum validation F1. AP depends on prevalence; scores on different partitions
must not be interpreted as an isolated measure of degradation or improvement.

Candidate selection, seed, hyperparameters, input hashes, and threshold are written
to `selection_lock.json` before the author grades final test. The selected model is
not refit on validation, so its validation-selected threshold remains attached to
the exact fitted model. The final test must not be used to tune it.

## Resource budget

CPU only, two numerical threads, 1,800 seconds maximum per solver process, no paid
compute, network access unnecessary. Four bounded candidates plus two diagnostics;
no unrestricted tuning. Kaggle outputs go under `/kaggle/working`. Optional mounted
inputs are read from `/kaggle/input`. Without a mount, the self-contained author
notebook generates synthetic inputs under the working directory. PyTorch/GPU is
omitted because the tabular baselines and harness complete cheaply on CPU.

## Approach-neutral rubric

1. **Honor the prediction contract.** Rationale: future observations invalidate a
   deployment estimate. Evidence: timestamps, strict maturity checks, the
   future-row perturbation test, and documented permitted inputs.
2. **Preserve temporal partition boundaries.** Rationale: immature outcomes cannot
   train a model deployed at a historical cutoff. Evidence: split manifest and
   exact-cutoff tests; account repetition is allowed in the main track.
3. **Fit learned preprocessing on training only.** Rationale: fitting medians,
   vocabularies, or scalers on validation/test leaks distribution information.
   Evidence: inspect fit calls and tests with extreme validation values and a
   validation-only category. No particular estimator is required.
4. **Select without final-test feedback.** Rationale: repeated holdout tuning biases
   reported performance. Evidence: selection lock created before grading,
   validation experiment log, unchanged lock hash, and no final labels in solver.
5. **Produce valid probability predictions.** Rationale: the deliverable must be
   machine-gradeable independent of row order. Evidence: complete unique ID set,
   valid schema/range, shuffled-row parity, and adversarial submission tests.
6. **Use the specified metric exactly.** Rationale: interpolated PR-AUC is a
   different quantity from AP. Evidence: sklearn metric parity; report prevalence,
   handle one-class ROC-AUC, and apply the locked threshold for secondary metrics.
7. **Make experiments reproducible and bounded.** Rationale: reviewers must be
   able to reproduce the result within the budget. Evidence: fixed seed/config,
   package versions, data/source hashes, wall time, sampled memory, clean-state
   notebook run, and successful CLI reproduction. GPU is not rewarded.
8. **Report limitations and diagnostics honestly.** Rationale: synthetic evidence
   cannot establish production impact. Evidence: synthetic/AI-assistance labels,
   separate unseen-entity scores, history ablation, invalid random-split warning,
   and conditional account-bootstrap interpretation. A complex model need not win.

## Eris context

The [official Eris application page](https://shipd.ai/quests/eris/join?mode=apply)
was checked on 2026-10-11 (Asia/Saigon). It lists building datasets, problem
statements, and evaluation harnesses. It does not provide the full contributor
runtime or originality/AI-assistance specification. No compatibility or acceptance
is asserted. Before any official reuse, check current contributor terms, AI
assistance, originality, and licensing requirements. No application was submitted.
