"""AUTHOR ONLY: reproducible latent-factor generator. Never imported by solver."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .common import read_config, write_json, frame_hash


def generate(config):
    rng = np.random.default_rng(config["seed"])
    n, na, nm = (config[k] for k in ("n_transactions", "n_accounts", "n_merchants"))
    seconds = np.sort(rng.integers(0, config["duration_days"] * 86400, n))
    # Unique prediction timestamps remove tie-order ambiguity.
    seconds = np.maximum.accumulate(seconds - np.arange(n)) + np.arange(n)
    t = pd.Timestamp(config["start"]) + pd.to_timedelta(seconds, unit="s")
    day = seconds / 86400
    shift = day >= config["test_cutoff_day"]
    account_risk = rng.normal(0, 1, na)
    merchant_risk = rng.normal(0, 0.8, nm)
    new_account_start = max(1, int(na * .9))
    new_merchant_start = max(1, int(nm * .9))
    account = rng.integers(0, new_account_start, n)
    merchant = rng.integers(0, new_merchant_start, n)
    # New entities only start arriving in the final period.
    new_a = shift & (rng.random(n) < .12)
    new_m = shift & (rng.random(n) < .10)
    account[new_a] = rng.integers(new_account_start, na, new_a.sum())
    merchant[new_m] = rng.integers(new_merchant_start, nm, new_m.sum())
    latent_device = rng.normal(size=n)
    amount = rng.lognormal(3.5 + .25 * shift + .12 * account_risk[account], .9, n)
    foreign = rng.binomial(1, .12 + .12 * shift, n)
    channel = rng.choice(["web", "mobile", "terminal"], n, p=[.35, .45, .2])
    channel[shift & (rng.random(n) < .12)] = "wallet"
    category = np.array(["retail", "travel", "services", "digital"])[merchant % 4]
    hour = t.hour.to_numpy()
    # No observed feature determines target: hidden entity factors, interactions,
    # hidden device condition, per-event noise, and Bernoulli sampling all matter.
    logit = (-3.8 + .9 * account_risk[account] + .8 * merchant_risk[merchant]
             + .45 * latent_device + .6 * foreign + .3 * (hour < 5)
             + .25 * np.log1p(amount / 50) + .65 * shift
             + .5 * shift * (category == "digital") + rng.normal(0, .55, n))
    prob = 1 / (1 + np.exp(-logit))
    target = rng.binomial(1, prob)
    device_score = latent_device + rng.normal(0, 1.2, n)
    device_score[rng.random(n) < (.07 + .05 * shift)] = np.nan
    event_delay = rng.uniform(.05, config["horizon_days"], n)
    return pd.DataFrame({
        "transaction_id": [f"tx_{i:07d}" for i in range(n)],
        "transaction_time": t,
        "account_id": [f"a_{x:04d}" for x in account],
        "merchant_id": [f"m_{x:04d}" for x in merchant],
        "amount": amount, "foreign": foreign, "channel": channel,
        "merchant_category": category, "device_score": device_score,
        "target": target,
        "label_available_at": t + pd.Timedelta(days=config["horizon_days"]),
        "adverse_event_at": pd.Series(t + pd.to_timedelta(event_delay, unit="D")).where(target == 1),
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--output", default="data/author/raw.csv")
    args = p.parse_args()
    df = generate(read_config(args.config))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    write_json(Path(args.output).with_suffix(".json"), {"rows": len(df), "data_hash": frame_hash(df)})


if __name__ == "__main__":
    main()
