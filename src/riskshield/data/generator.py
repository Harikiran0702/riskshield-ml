"""Chronological event simulation. Labels are sampled AFTER history snapshots."""

from collections import deque

import numpy as np
import pandas as pd

from riskshield.config.settings import CATEGORY_RISK, SETTINGS


def generate(n: int = SETTINGS["rows"], seed: int = SETTINGS["seed"]) -> pd.DataFrame:
    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    count = SETTINGS["customers"]
    averages = rng.lognormal(3.8, 0.6, count)
    ages = rng.integers(1, 2000, count)
    histories = [deque() for _ in range(count)]
    sums, squares, counts = np.zeros(count), np.zeros(count), np.zeros(count, dtype=int)
    last = np.full(count, -86400.0)
    devices = [set() for _ in range(count)]
    merchants = rng.uniform(0.02, 0.85, 100)
    rows = []
    now = 0.0
    previous_customer = 0
    for i in range(n):
        now += float(rng.exponential(60))
        c = previous_customer if rng.random() < 0.16 else int(rng.integers(count))
        previous_customer = c
        ts = pd.Timestamp("2025-01-01T00:00:00Z") + pd.Timedelta(seconds=now)
        history = histories[c]
        while history and history[0][0] < now - 3600:
            history.popleft()
        avg = sums[c] / counts[c] if counts[c] else averages[c]
        std = max(10.0, np.sqrt(max(0, squares[c] / counts[c] - avg**2))) if counts[c] else avg
        amount = round(
            float(
                min(
                    1e7,
                    rng.lognormal(np.log(averages[c]), 0.8)
                    * (rng.uniform(3, 7) if rng.random() < 0.07 else 1),
                )
            ),
            2,
        )
        device = f"DEV-{c}-{int(rng.integers(1, 100)) if rng.random() < 0.12 else 0}"
        country = str(rng.choice(["US", "GB", "CA", "DE"], p=[0.89, 0.04, 0.04, 0.03]))
        ip_country = country if rng.random() < 0.95 else "SG"
        m = int(rng.integers(100))
        category = str(rng.choice(list(CATEGORY_RISK)))
        tx10 = sum(t > now - 600 for t, _ in history)
        row = dict(
            transaction_id=f"TXN-{i:07d}",
            customer_id=f"CUS-{c:04d}",
            timestamp=ts.isoformat(),
            transaction_amount=max(0.01, amount),
            merchant_id=f"MER-{m:03d}",
            merchant_category=category,
            payment_channel=str(rng.choice(["pos", "web", "mobile"])),
            country=country,
            home_country="US",
            device_id=device,
            ip_country=ip_country,
            account_age_days=int(ages[c] + now // 86400),
            historical_avg_amount=float(avg),
            historical_std_amount=float(std),
            prior_transaction_count=int(counts[c]),
            transactions_10m=tx10,
            transactions_1h=len(history),
            spend_1h=float(sum(a for _, a in history)),
            seconds_since_previous=float(now - last[c]),
            new_device=device not in devices[c],
            merchant_risk=float(merchants[m]),
        )
        # Overlapping classes; latent noise is deliberately unavailable to the model.
        logit = (
            -6.3
            + 0.65 * min(amount / avg, 8)
            + 0.8 * row["new_device"]
            + 0.8 * (country != "US")
            + 0.6 * (ip_country != country)
            + 0.6 * min(tx10, 4)
            + 0.7 * (ts.hour < 6)
            + 0.8 * merchants[m]
            + 0.4 * CATEGORY_RISK[category]
            + 0.6 * (ages[c] < 30)
            + rng.normal(0, 0.6)
        )
        row["fraud_label"] = int(rng.random() < 1 / (1 + np.exp(-logit)))
        rows.append(row)
        # Only observable transaction activity updates state, never fraud labels.
        history.append((now, amount))
        sums[c] += amount
        squares[c] += amount**2
        counts[c] += 1
        last[c] = now
        devices[c].add(device)
    return pd.DataFrame(rows)
