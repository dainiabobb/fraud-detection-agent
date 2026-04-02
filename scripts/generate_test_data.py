"""
Generate synthetic test transactions for the fraud detection pipeline.

Produces a mix of:
  - 85% normal transactions (reasonable amounts, consistent locations, regular hours)
  - 10% fraud-like transactions (geo anomalies, velocity spikes, off-hours, high amounts)
  -  5% AML-suspicious transactions (structuring near $10k, round numbers, high-risk countries)

Usage:
    python scripts/generate_test_data.py [--count N] [--output FILE] [--seed SEED]

    # 100 transactions to stdout
    python scripts/generate_test_data.py

    # 5000 transactions to a file
    python scripts/generate_test_data.py --count 5000 --output data/synthetic_test.jsonl

    # Deterministic output
    python scripts/generate_test_data.py --count 500 --seed 42
"""

import argparse
import json
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from typing import TextIO


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

NORMAL_CITIES: list[tuple[str, str, str]] = [
    ("Dallas", "TX", "US"),
    ("Austin", "TX", "US"),
    ("Houston", "TX", "US"),
    ("Chicago", "IL", "US"),
    ("New York", "NY", "US"),
    ("Los Angeles", "CA", "US"),
    ("San Francisco", "CA", "US"),
    ("Seattle", "WA", "US"),
    ("Denver", "CO", "US"),
    ("Phoenix", "AZ", "US"),
    ("Atlanta", "GA", "US"),
    ("Boston", "MA", "US"),
    ("Miami", "FL", "US"),
    ("Portland", "OR", "US"),
    ("Nashville", "TN", "US"),
]

HIGH_RISK_COUNTRIES: list[tuple[str, str]] = [
    ("Lagos", "NG"),
    ("Accra", "GH"),
    ("Nairobi", "KE"),
    ("Kyiv", "UA"),
    ("Minsk", "BY"),
    ("Tashkent", "UZ"),
    ("Panama City", "PA"),
    ("Belize City", "BZ"),
    ("Yangon", "MM"),
    ("Vientiane", "LA"),
]

NORMAL_CATEGORIES: list[str] = [
    "Groceries",
    "Gas Stations",
    "Restaurants",
    "Pharmacies",
    "Retail",
    "Digital Goods",
    "Subscriptions",
    "Services",
]

FRAUD_CATEGORIES: list[str] = [
    "Electronics",
    "Jewelry",
    "Gift Cards",
    "Wire Transfer",
    "Digital Goods",
]

AML_CATEGORIES: list[str] = [
    "Wire Transfer",
    "Services",
    "Consulting",
]

NORMAL_USER_IDS: list[str] = [f"user-{i}" for i in range(100, 200)]
FRAUD_USER_IDS: list[str] = [f"user-{i}" for i in range(700, 750)]
AML_USER_IDS: list[str] = [f"user-{i}" for i in range(400, 430)]

CHANNELS: list[str] = ["online", "mobile", "in-store"]
FRAUD_CHANNELS: list[str] = ["online", "mobile"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_ip(rng: random.Random) -> str:
    return f"{rng.randint(1, 254)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"


def _rand_device(prefix: str, rng: random.Random) -> str:
    return f"dev-{prefix}-{rng.randint(1000, 9999)}"


def _timestamp_in_range(
    start: datetime, end: datetime, rng: random.Random
) -> str:
    delta = end - start
    offset_seconds = rng.randint(0, int(delta.total_seconds()))
    dt = start + timedelta(seconds=offset_seconds)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _business_hours_timestamp(rng: random.Random, base_date: datetime) -> str:
    """Return a timestamp during business/active hours (7am-10pm local)."""
    hour = rng.randint(7, 21)
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    dt = base_date.replace(hour=hour, minute=minute, second=second, tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _off_hours_timestamp(rng: random.Random, base_date: datetime) -> str:
    """Return a timestamp during off-hours (12am-5am) when fraud is more common."""
    hour = rng.randint(0, 4)
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    dt = base_date.replace(hour=hour, minute=minute, second=second, tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Transaction generators
# ---------------------------------------------------------------------------

def _make_normal_transaction(rng: random.Random, base_date: datetime) -> dict:
    city, state, country = rng.choice(NORMAL_CITIES)
    category = rng.choice(NORMAL_CATEGORIES)
    user_id = rng.choice(NORMAL_USER_IDS)

    # Amount shaped by category
    if category == "Groceries":
        amount = round(rng.uniform(20.0, 180.0), 2)
    elif category == "Gas Stations":
        amount = round(rng.uniform(30.0, 90.0), 2)
    elif category == "Restaurants":
        amount = round(rng.uniform(12.0, 120.0), 2)
    elif category == "Pharmacies":
        amount = round(rng.uniform(15.0, 150.0), 2)
    elif category == "Subscriptions":
        amount = rng.choice([9.99, 12.99, 14.99, 17.99, 29.99])
    elif category == "Digital Goods":
        amount = round(rng.uniform(5.0, 80.0), 2)
    else:
        amount = round(rng.uniform(25.0, 300.0), 2)

    return {
        "transaction_id": f"txn-{uuid.uuid4().hex[:12]}",
        "user_id": user_id,
        "amount": amount,
        "merchant_category": category,
        "merchant_name": f"{category} Merchant {rng.randint(1, 999):03d}",
        "merchant_city": city,
        "merchant_state": state,
        "merchant_country": country,
        "channel": rng.choice(CHANNELS),
        "timestamp": _business_hours_timestamp(rng, base_date),
        "device_id": _rand_device("usr", rng),
        "ip_address": _rand_ip(rng),
        "currency": "USD",
        "_label": "normal",
    }


def _make_fraud_transaction(rng: random.Random, base_date: datetime) -> dict:
    """Generate a fraud-like transaction with one or more anomaly signals."""
    fraud_type = rng.choice(["geo_anomaly", "velocity_spike", "off_hours_high_value", "new_device"])
    user_id = rng.choice(FRAUD_USER_IDS)
    category = rng.choice(FRAUD_CATEGORIES)

    if fraud_type == "geo_anomaly":
        # Transaction in a high-risk country
        city, country = rng.choice(HIGH_RISK_COUNTRIES)
        state = None
        amount = round(rng.uniform(500.0, 4000.0), 2)
        channel = rng.choice(FRAUD_CHANNELS)
        timestamp = _off_hours_timestamp(rng, base_date)
    elif fraud_type == "velocity_spike":
        # Domestic but unusually high amount for the category
        city, state, country = rng.choice(NORMAL_CITIES)
        amount = round(rng.uniform(1500.0, 6000.0), 2)
        channel = "online"
        timestamp = _business_hours_timestamp(rng, base_date)
    elif fraud_type == "off_hours_high_value":
        # High value purchase in the middle of the night
        city, state, country = rng.choice(NORMAL_CITIES)
        state = state
        amount = round(rng.uniform(800.0, 3500.0), 2)
        channel = rng.choice(FRAUD_CHANNELS)
        timestamp = _off_hours_timestamp(rng, base_date)
    else:
        # New device (dev-new- prefix) making a high-value purchase
        city, state, country = rng.choice(NORMAL_CITIES)
        amount = round(rng.uniform(600.0, 2500.0), 2)
        channel = rng.choice(FRAUD_CHANNELS)
        timestamp = _business_hours_timestamp(rng, base_date)

    txn: dict = {
        "transaction_id": f"txn-{uuid.uuid4().hex[:12]}",
        "user_id": user_id,
        "amount": amount,
        "merchant_category": category,
        "merchant_name": f"{category} Store {rng.randint(1, 999):03d}",
        "merchant_city": city,
        "merchant_country": country,
        "channel": channel,
        "timestamp": timestamp,
        "device_id": (
            f"dev-new-{rng.randint(1000, 9999)}"
            if fraud_type == "new_device"
            else _rand_device("frd", rng)
        ),
        "ip_address": _rand_ip(rng),
        "currency": "USD",
        "_label": "fraud",
        "_fraud_type": fraud_type,
    }
    # merchant_state is optional; omit for international transactions
    if fraud_type == "geo_anomaly":
        txn["merchant_state"] = None
    else:
        txn["merchant_state"] = state  # type: ignore[assignment]

    return txn


def _make_aml_transaction(rng: random.Random, base_date: datetime) -> dict:
    """Generate an AML-suspicious transaction (structuring / round numbers / high-risk)."""
    aml_type = rng.choice(["structuring", "round_number_wire", "high_risk_country_wire"])
    user_id = rng.choice(AML_USER_IDS)
    category = rng.choice(AML_CATEGORIES)

    if aml_type == "structuring":
        # Amount just under the $10,000 CTR reporting threshold
        amount = round(rng.uniform(8200.0, 9800.0), 2)
        city, state, country = rng.choice([("Miami", "FL", "US"), ("New York", "NY", "US"), ("Los Angeles", "CA", "US")])
        channel = "online"
        timestamp = _business_hours_timestamp(rng, base_date)
    elif aml_type == "round_number_wire":
        # Round-number wire transfers are an AML signal
        amount = float(rng.choice([5000, 7500, 10000, 15000, 20000, 25000]))
        city, state, country = rng.choice(NORMAL_CITIES)
        channel = "online"
        timestamp = _business_hours_timestamp(rng, base_date)
    else:
        # Wire to a high-risk jurisdiction
        city, country = rng.choice(HIGH_RISK_COUNTRIES)
        state = None
        amount = round(rng.uniform(2000.0, 15000.0), 2)
        channel = "online"
        timestamp = _business_hours_timestamp(rng, base_date)

    txn: dict = {
        "transaction_id": f"txn-{uuid.uuid4().hex[:12]}",
        "user_id": user_id,
        "amount": amount,
        "merchant_category": category,
        "merchant_name": f"Wire Services {rng.randint(1, 99):02d}",
        "merchant_city": city,
        "merchant_country": country,
        "channel": channel,
        "timestamp": timestamp,
        "device_id": _rand_device("aml", rng),
        "ip_address": _rand_ip(rng),
        "currency": "USD",
        "_label": "aml",
        "_aml_type": aml_type,
    }
    if aml_type == "high_risk_country_wire":
        txn["merchant_state"] = None
    else:
        txn["merchant_state"] = state  # type: ignore[assignment]

    return txn


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_transactions(count: int, rng: random.Random) -> list[dict]:
    """Generate `count` synthetic transactions with the configured mix."""
    # Reference window: last 30 days leading up to 2026-03-31
    end_date = datetime(2026, 3, 31, tzinfo=timezone.utc)
    start_date = end_date - timedelta(days=30)

    transactions: list[dict] = []

    for _ in range(count):
        base_date = start_date + timedelta(
            seconds=rng.randint(0, int((end_date - start_date).total_seconds()))
        )
        roll = rng.random()
        if roll < 0.85:
            txn = _make_normal_transaction(rng, base_date)
        elif roll < 0.95:
            txn = _make_fraud_transaction(rng, base_date)
        else:
            txn = _make_aml_transaction(rng, base_date)

        transactions.append(txn)

    return transactions


def write_jsonlines(transactions: list[dict], out: TextIO) -> None:
    for txn in transactions:
        out.write(json.dumps(txn) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic test transactions for the fraud detection pipeline."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of transactions to generate (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (JSON lines). Defaults to stdout.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic output.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    transactions = generate_transactions(args.count, rng)

    # Tally labels for a summary
    label_counts: dict[str, int] = {}
    for txn in transactions:
        label = txn.get("_label", "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1

    print(
        f"Generated {args.count} transactions: "
        + ", ".join(f"{k}={v}" for k, v in sorted(label_counts.items())),
        file=sys.stderr,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            write_jsonlines(transactions, f)
        print(f"Wrote {args.count} transactions to {args.output}", file=sys.stderr)
    else:
        write_jsonlines(transactions, sys.stdout)


if __name__ == "__main__":
    main()
