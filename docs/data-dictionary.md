# Data dictionary and synthetic generation

All identifiers represent generated entities. Amount is in arbitrary synthetic
currency units. No real customer, merchant, financial, or company records are used.

- `transaction_id`: stable unique string; joins submissions to labels. Not a model feature.
- `transaction_time`: UTC prediction timestamp; unique at second resolution.
- `account_id`, `merchant_id`: generated entity IDs. Reference model excludes raw IDs.
- `amount`: lognormal transaction amount; mean shifts upward in the final period.
- `foreign`: binary transaction-context indicator; probability shifts from 0.12 to 0.24.
- `channel`: web/mobile/terminal; wallet is introduced only during the test period.
- `merchant_category`: retail/travel/services/digital; based on merchant assignment.
- `device_score`: noisy continuous observation of a hidden device factor; approximately
  7% missing before shift and 12% after shift. It is not the hidden factor itself.
- `hour`, `day_of_week`: calendar features derived from prediction time in UTC.
- `account_past_count`: count of strictly prior account transactions, all observed history.
- `account_past_mean_amount`: mean amount over strictly prior account transactions;
  missing for a first transaction. Current amount is excluded.
- `account_count_7d`: account count in the prior seven-day rolling window.
- `account_amount_7d`: total amount in that same window; current transaction excluded.
- `account_hours_since_last`: time since prior account transaction; missing for first event.
- `merchant_matured_count`: number of merchant labels strictly available before both
  prediction time and the fixed training cutoff.
- `merchant_matured_rate`: fixed-prior smoothed rate `(positives + 1)/(count + 20)`
  over those mature labels. It never uses the current row target.
- `target`: binary adverse-event outcome; only present in solver train/validation.
- `label_available_at` (**author only**): transaction time plus seven days for every row.
- `adverse_event_at` (**author only**): a random time inside the horizon for positives;
  missing for negatives. It never enters the reference feature matrix.

Generator seed: 1803. 50,000 transactions over 180 days beginning 2025-01-01 UTC,
1,500 possible accounts and 300 possible merchants. Ten percent of entity IDs are
reserved for the final period. New-entity transactions arrive with independent
12% account / 10% merchant selection probabilities; realized counts are reported
in author subgroup metrics. The latent adverse-event probability combines hidden
account and merchant risks, device condition, observed context, time/category
interaction, event noise, and Bernoulli sampling. No observed field determines
the target directly; no LLM generated labels or benchmark scores.

Final-period shift changes amount, foreign rate, missingness, introduces a channel
category and new IDs, increases the latent intercept by 0.65, and adds a digital
category interaction. These are deliberately controlled synthetic changes, not
measurements from a real payment network. The model does not receive the shift
flag or the latent adverse-event probability.

The author split manifest records exact UTC ranges, mature-label boundaries,
class balance, purge counts, canonical frame hashes, and byte hashes. CSV byte
hashes can differ across OS line endings. Canonical frame hashes serialize numeric
values with 12 significant digits and LF newlines; they are reproducibility checks,
not cryptographic secrecy. Generation is reproducible for an identical config.
Increasing the requested dataset size changes RNG consumption, so prefix stability
across different generator sizes is not promised. Future-invariance is tested on
the feature algorithm using a fixed event stream.
