# LeakageLab interview notes

This is an AI-assisted portfolio project. Explain the code and measured evidence;
do not describe it as an Eris-accepted task, production fraud system, or evaluation
of frontier agents. The useful contribution is a reproducible task contract and
evaluation harness, not an unusually large AP number.

## Decisions to explain

Prediction is at transaction time; labels become observable after seven days.
History features consume raw prior transactions even in purge gaps, but merchant
label features consume only mature labels and freeze at the original training
cutoff. This intentionally avoids online feedback from validation/test outcomes.
Negative labels also wait the full horizon. The temporal split simulates a model
trained on April 19, selected after validation labels mature by May 25, and scored
on future transactions. Training is not refreshed after validation.

Raw account/merchant IDs are excluded as model predictors, although returning
entities are allowed. Logistic preprocessing uses training medians, missingness
indicators, standardization, and unknown-safe one-hot encoding. HGB uses missing
values and categorical ordinal codes marked as categorical, with unknown codes
represented as NaN. Its early stopping is disabled so it cannot silently create
a random internal temporal-validation split. Four fixed candidates are compared;
the history ablation and random split never select the winner.

The solver imports no generator or grader. Its read allowlist is tested. Separate
processes and input contracts prevent accidental label access in the reference
solution but are not an adversarial security boundary. Grading joins unique IDs;
strict schema and range checks fail before scoring. A public synthetic generator
can reproduce holdout labels, so the benchmark is not a secure leaderboard.

## Ten questions and answers

1. **Why purge seven days?** A transaction immediately before training cutoff has
   an incomplete label horizon. `split_masks` requires availability strictly before
   the cutoff, removing 3,924 rows across the two boundaries in this realization.
   Purge does not mean deleting their already-observed raw transaction histories.
2. **Is a group split necessary?** It depends on deployment. Future transactions
   from existing accounts are the primary assumption, so returning accounts are
   allowed. Holding out all accounts would answer a different question. This run
   separately scores 1,216 unseen-account and 942 unseen-merchant test transactions.
3. **How do you prove history is causal?** `features` emits history before updating
   the current transaction. Delayed-label updates require strict earlier availability
   and the freeze cutoff. Tests corrupt all later targets/amounts and compare past
   features exactly, plus a three-row fixture checks the exact seven-day boundary.
4. **Can the target aggregate leak even if shifted one row?** Yes. A prior target
   can still be immature. The code uses `label_available_at`, not simply row order;
   a label exactly available at prediction time is conservatively excluded.
5. **Why AP instead of accuracy or ROC-AUC alone?** The task ranks uncommon adverse
   outcomes. AP summarizes precision at recovered-positive increments; an all-zero
   classifier can have high accuracy. ROC-AUC is useful but does not show precision
   under the observed prevalence. Report AP with class balance and calibration metrics.
6. **Why did logistic regression beat HGB?** Validation AP was 0.1323 for logistic,
   versus 0.1193/0.1147 for bounded HGB settings. Plausible explanation: the latent
   mechanism is substantially additive and observed signals are weak/noisy; a simple
   model may have lower estimation variance. This is an inference, not a causal proof.
   The limited grid establishes only what happened under these fixed settings.
7. **What did the ablation/random comparison show?** Removing all history reduced
   logistic validation AP from 0.1323 to 0.0877 on the same 7,971 rows. Random-split
   AP was 0.1206 on a different 7,205-row set, lower here. Random training can use
   future-era examples, so it is invalid for the deployment claim. Neither comparison
   was forced to produce a predetermined ordering.
8. **Why is test AP higher while ROC-AUC is lower?** Validation prevalence is 6.02%,
   test prevalence 11.82%; AP depends on prevalence. Validation ROC-AUC is 0.6763,
   final-test ROC-AUC 0.6525. Test AP 0.1931 is not evidence that drift improved ranking.
   Log loss also rises, and the threshold was not recalibrated on test.
9. **How trustworthy is the confidence interval?** A 200-replicate account-cluster
   bootstrap gave AP interval [0.1766, 0.2128]. Resampling whole account trajectories
   preserves within-account dependence. It does not cover shared time/merchant
   shocks, generator-seed variation, or real-population uncertainty. It is conditional
   on this synthetic realization; a multi-seed temporal study would be stronger.
10. **How was end-to-end execution verified?** The local CLI generated data, ran the
    isolated solver, froze selection, graded by ID, and saved metrics/hashes. A private
    Kaggle Save & Run All completed on CPU in 49.7 seconds and passed the same 18 tests.
    Downloaded executed notebook and logs provide evidence. No Kaggle medals, ranking,
    commercial impact, or Eris acceptance were obtained or claimed.

## Follow-up work that would improve the benchmark

Add several generator seeds and expanding-window folds with preprocessing refit
inside each training fold. Introduce gradual drift in addition to the abrupt shift,
evaluate calibration using only validation, and compare time-block uncertainty.
Preserve a fresh final holdout for any subsequent tuning. A truly private benchmark
would need separate author data and a sandboxed grading service; publishing the
generator cannot provide that secrecy. These are future improvements, not completed
features of the current result.
