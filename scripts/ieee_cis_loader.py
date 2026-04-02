"""
Load IEEE-CIS Fraud Detection dataset and inject synthetic AML patterns.

Reads train_transaction.csv and train_identity.csv from the data/ directory,
applies column mapping per the project README, injects synthetic AML patterns
for ~2% of users, and returns the processed dataset.

Usage (CLI):
    python scripts/ieee_cis_loader.py [--sample-size N] [--output FILE]

    # Load full dataset, output JSON lines
    python scripts/ieee_cis_loader.py --output data/processed_transactions.jsonl

    # Load 5000-transaction sample (cheaper for test runs)
    python scripts/ieee_cis_loader.py --sample-size 5000 --output data/sample.jsonl

Programmatic usage:
    from scripts.ieee_cis_loader import load_ieee_cis

    transactions, ground_truth, aml_injected = load_ieee_cis(
        data_dir="data/",
        sample_size=5000,
    )
    # transactions: list[dict]
    # ground_truth: dict[str, bool]   transaction_id -> is_fraud
    # aml_injected: dict[str, str]    user_id -> typology name
"""

import argparse
import json
import os
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Reference epoch: TransactionDT is seconds elapsed since this date
REFERENCE_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)

# ProductCD -> merchant_category
PRODUCT_CD_MAP: dict[str, str] = {
    "W": "Digital Goods",
    "H": "Hotels",
    "C": "Services",
    "S": "Subscriptions",
    "R": "Retail",
}

# addr2 country code -> ISO country code (top values from the dataset)
COUNTRY_CODE_MAP: dict[float, str] = {
    87.0: "US",
    96.0: "GB",
    166.0: "AU",
    60.0: "CA",
    76.0: "DE",
    226.0: "FR",
    1.0: "IN",
    65.0: "JP",
    150.0: "SG",
}

# Simplified zip-to-city mapping (addr1 in IEEE-CIS is a zip-like code 0-500)
# We bucket ranges into representative US cities for realism.
def _zip_to_city(addr1: float) -> tuple[str, str]:
    """Map IEEE-CIS addr1 range to (city, state)."""
    if pd.isna(addr1):
        return ("Unknown", "XX")
    z = int(addr1)
    if z <= 50:
        return ("New York", "NY")
    elif z <= 100:
        return ("Los Angeles", "CA")
    elif z <= 150:
        return ("Chicago", "IL")
    elif z <= 200:
        return ("Houston", "TX")
    elif z <= 250:
        return ("Phoenix", "AZ")
    elif z <= 300:
        return ("Philadelphia", "PA")
    elif z <= 350:
        return ("San Antonio", "TX")
    elif z <= 400:
        return ("San Diego", "CA")
    elif z <= 450:
        return ("Dallas", "TX")
    else:
        return ("San Jose", "CA")


# AML injection parameters per typology
AML_TYPOLOGIES = {
    "structuring": 0.40,      # 40% of injections
    "smurfing": 0.20,         # 20%
    "layering": 0.15,         # 15%
    "round_tripping": 0.10,   # 10%
    "profile_mismatch": 0.15, # 15%
}

HIGH_RISK_CITIES: list[tuple[str, str]] = [
    ("Lagos", "NG"),
    ("Accra", "GH"),
    ("Panama City", "PA"),
    ("Belize City", "BZ"),
    ("Yangon", "MM"),
]


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------

def _map_transaction_row(row: pd.Series) -> dict:
    """Map a single IEEE-CIS row to the internal transaction schema."""
    txn_id = str(int(row["TransactionID"]))
    card1 = row.get("card1", 0)
    user_id = f"user-{int(card1)}" if not pd.isna(card1) else f"user-unknown-{txn_id}"

    amount = float(row["TransactionAmt"]) if not pd.isna(row.get("TransactionAmt")) else 0.0

    product_cd = str(row.get("ProductCD", "")).strip().upper()
    category = PRODUCT_CD_MAP.get(product_cd, "Retail")

    city, state = _zip_to_city(row.get("addr1"))
    addr2 = row.get("addr2")
    country = COUNTRY_CODE_MAP.get(float(addr2), "US") if not pd.isna(addr2) else "US"

    txn_dt = row.get("TransactionDT")
    if not pd.isna(txn_dt):
        timestamp_dt = REFERENCE_DATE + timedelta(seconds=float(txn_dt))
    else:
        timestamp_dt = REFERENCE_DATE
    timestamp = timestamp_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    device_type = str(row.get("DeviceType", "")).strip().lower()
    if device_type == "mobile":
        channel = "mobile"
    elif device_type == "desktop":
        channel = "online"
    else:
        channel = "online"

    return {
        "transaction_id": txn_id,
        "user_id": user_id,
        "amount": round(amount, 2),
        "merchant_category": category,
        "merchant_name": f"{category} Merchant",
        "merchant_city": city,
        "merchant_state": state,
        "merchant_country": country,
        "channel": channel,
        "timestamp": timestamp,
        "device_id": f"dev-ieee-{txn_id}",
        "ip_address": "0.0.0.0",  # not available in IEEE-CIS; placeholder
        "currency": "USD",
    }


# ---------------------------------------------------------------------------
# AML injection
# ---------------------------------------------------------------------------

def _inject_structuring(
    user_id: str,
    base_time: datetime,
    rng: random.Random,
) -> list[dict]:
    """3-6 deposits of $8,200-$9,800 over 2-4 weeks."""
    count = rng.randint(3, 6)
    total_days = rng.randint(14, 28)
    txns = []
    for i in range(count):
        offset_days = int(i * total_days / count) + rng.randint(0, 2)
        dt = base_time + timedelta(days=offset_days, hours=rng.randint(9, 17))
        txns.append({
            "transaction_id": f"txn-aml-struct-{uuid.uuid4().hex[:10]}",
            "user_id": user_id,
            "amount": round(rng.uniform(8200.0, 9800.0), 2),
            "merchant_category": "Wire Transfer",
            "merchant_name": "Wire Services Inc",
            "merchant_city": "Miami",
            "merchant_state": "FL",
            "merchant_country": "US",
            "channel": "online",
            "timestamp": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "device_id": f"dev-aml-{user_id}",
            "ip_address": f"174.{rng.randint(1,254)}.{rng.randint(1,254)}.{rng.randint(1,254)}",
            "currency": "USD",
            "_aml_injected": True,
            "_aml_typology": "structuring",
        })
    return txns


def _inject_smurfing(
    user_id: str,
    base_time: datetime,
    rng: random.Random,
) -> list[dict]:
    """4-8 feeder users sending $2k-$4k to the target within 10 days."""
    feeder_count = rng.randint(4, 8)
    txns = []
    for i in range(feeder_count):
        feeder_id = f"user-smurf-{rng.randint(10000, 99999)}"
        dt = base_time + timedelta(days=rng.randint(0, 10), hours=rng.randint(8, 20))
        txns.append({
            "transaction_id": f"txn-aml-smurf-{uuid.uuid4().hex[:10]}",
            "user_id": feeder_id,
            "amount": round(rng.uniform(2000.0, 4000.0), 2),
            "merchant_category": "Wire Transfer",
            "merchant_name": "P2P Transfer",
            "merchant_city": "New York",
            "merchant_state": "NY",
            "merchant_country": "US",
            "channel": "online",
            "timestamp": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "device_id": f"dev-smurf-{i}",
            "ip_address": f"45.{rng.randint(1,254)}.{rng.randint(1,254)}.{rng.randint(1,254)}",
            "currency": "USD",
            "_aml_injected": True,
            "_aml_typology": "smurfing",
            "_smurfing_target": user_id,
        })
    return txns


def _inject_layering(
    user_id: str,
    base_time: datetime,
    rng: random.Random,
) -> list[dict]:
    """$20k-$50k splits into 3-5 outbound transfers in 24-48 hours."""
    total_amount = round(rng.uniform(20000.0, 50000.0), 2)
    hop_count = rng.randint(3, 5)
    base_per_hop = total_amount / hop_count
    txns = []
    city, country = rng.choice(HIGH_RISK_CITIES)
    for i in range(hop_count):
        offset_hours = int(i * 48 / hop_count) + rng.randint(0, 3)
        dt = base_time + timedelta(hours=offset_hours)
        hop_amount = round(base_per_hop * rng.uniform(0.85, 1.15), 2)
        txns.append({
            "transaction_id": f"txn-aml-layer-{uuid.uuid4().hex[:10]}",
            "user_id": user_id,
            "amount": hop_amount,
            "merchant_category": "Wire Transfer",
            "merchant_name": f"Offshore Transfer {i+1}",
            "merchant_city": city,
            "merchant_state": None,
            "merchant_country": country,
            "channel": "online",
            "timestamp": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "device_id": f"dev-aml-{user_id}",
            "ip_address": f"197.{rng.randint(1,254)}.{rng.randint(1,254)}.{rng.randint(1,254)}",
            "currency": "USD",
            "_aml_injected": True,
            "_aml_typology": "layering",
            "_layering_hop": i + 1,
            "_layering_total_amount": total_amount,
        })
    return txns


def _inject_round_tripping(
    user_id: str,
    base_time: datetime,
    rng: random.Random,
) -> list[dict]:
    """$15k-$40k cycle through intermediaries and return to same account within 30 days."""
    amount = round(rng.uniform(15000.0, 40000.0), 2)
    intermediary_id = f"user-intermediary-{rng.randint(10000, 99999)}"
    txns = [
        # Outbound: user -> intermediary
        {
            "transaction_id": f"txn-aml-rt-out-{uuid.uuid4().hex[:10]}",
            "user_id": user_id,
            "amount": amount,
            "merchant_category": "Wire Transfer",
            "merchant_name": "Offshore Intermediary",
            "merchant_city": "Panama City",
            "merchant_state": None,
            "merchant_country": "PA",
            "channel": "online",
            "timestamp": base_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "device_id": f"dev-aml-{user_id}",
            "ip_address": f"200.{rng.randint(1,254)}.{rng.randint(1,254)}.{rng.randint(1,254)}",
            "currency": "USD",
            "_aml_injected": True,
            "_aml_typology": "round_tripping",
            "_rt_direction": "outbound",
        },
        # Inbound: intermediary -> user (30 days later, slightly reduced)
        {
            "transaction_id": f"txn-aml-rt-in-{uuid.uuid4().hex[:10]}",
            "user_id": user_id,
            "amount": round(amount * rng.uniform(0.88, 0.97), 2),
            "merchant_category": "Services",
            "merchant_name": "Consulting Payment",
            "merchant_city": "New York",
            "merchant_state": "NY",
            "merchant_country": "US",
            "channel": "online",
            "timestamp": (base_time + timedelta(days=rng.randint(25, 30))).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "device_id": f"dev-aml-{user_id}",
            "ip_address": f"200.{rng.randint(1,254)}.{rng.randint(1,254)}.{rng.randint(1,254)}",
            "currency": "USD",
            "_aml_injected": True,
            "_aml_typology": "round_tripping",
            "_rt_direction": "inbound",
        },
    ]
    return txns


def _inject_profile_mismatch(
    user_id: str,
    base_time: datetime,
    rng: random.Random,
) -> list[dict]:
    """Low-activity user receives $10k-$50k 'Consulting' transaction."""
    amount = round(rng.uniform(10000.0, 50000.0), 2)
    dt = base_time + timedelta(days=rng.randint(0, 7), hours=rng.randint(9, 17))
    return [
        {
            "transaction_id": f"txn-aml-mismatch-{uuid.uuid4().hex[:10]}",
            "user_id": user_id,
            "amount": amount,
            "merchant_category": "Consulting",
            "merchant_name": "Global Consulting Partners",
            "merchant_city": "New York",
            "merchant_state": "NY",
            "merchant_country": "US",
            "channel": "online",
            "timestamp": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "device_id": f"dev-aml-{user_id}",
            "ip_address": f"72.{rng.randint(1,254)}.{rng.randint(1,254)}.{rng.randint(1,254)}",
            "currency": "USD",
            "_aml_injected": True,
            "_aml_typology": "profile_mismatch",
        }
    ]


def _inject_aml_patterns(
    transactions: list[dict],
    user_ids: list[str],
    injection_rate: float,
    rng: random.Random,
) -> tuple[list[dict], dict[str, str]]:
    """
    Inject AML patterns into ~injection_rate fraction of unique users.

    Returns:
        (augmented_transactions, aml_injected)
        aml_injected: {user_id: typology_name}
    """
    aml_injected: dict[str, str] = {}
    injected_txns: list[dict] = []

    unique_users = list(set(user_ids))
    target_count = max(1, int(len(unique_users) * injection_rate))
    selected_users = rng.sample(unique_users, min(target_count, len(unique_users)))

    # Pick typology according to distribution
    typology_names = list(AML_TYPOLOGIES.keys())
    typology_weights = list(AML_TYPOLOGIES.values())

    # Place AML-injected transactions near the END of the timeline so they
    # land in the test set when an 80/20 chronological split is used.
    # IEEE-CIS TransactionDT spans ~15M seconds (~173 days from REFERENCE_DATE).
    base_time = datetime(2024, 6, 1, tzinfo=timezone.utc)

    for user_id in selected_users:
        typology = rng.choices(typology_names, weights=typology_weights, k=1)[0]
        aml_injected[user_id] = typology

        user_base_time = base_time + timedelta(days=rng.randint(0, 30))

        if typology == "structuring":
            new_txns = _inject_structuring(user_id, user_base_time, rng)
        elif typology == "smurfing":
            new_txns = _inject_smurfing(user_id, user_base_time, rng)
        elif typology == "layering":
            new_txns = _inject_layering(user_id, user_base_time, rng)
        elif typology == "round_tripping":
            new_txns = _inject_round_tripping(user_id, user_base_time, rng)
        else:  # profile_mismatch
            new_txns = _inject_profile_mismatch(user_id, user_base_time, rng)

        injected_txns.extend(new_txns)

    augmented = transactions + injected_txns
    return augmented, aml_injected


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_ieee_cis(
    data_dir: str = "data/",
    sample_size: Optional[int] = None,
    aml_injection_rate: float = 0.02,
    seed: int = 42,
) -> tuple[list[dict], dict[str, bool], dict[str, str]]:
    """
    Load IEEE-CIS dataset, map columns, and inject synthetic AML patterns.

    Args:
        data_dir: Path to directory containing train_transaction.csv and train_identity.csv.
        sample_size: If set, sample this many rows from the transaction CSV before processing.
        aml_injection_rate: Fraction of users to inject AML patterns into (~0.02 = 2%).
        seed: Random seed for deterministic sampling and injection.

    Returns:
        transactions: List of transaction dicts (pipeline input format).
        ground_truth: {transaction_id: is_fraud}  -- NOT passed to pipeline.
        aml_injected: {user_id: typology_name}    -- NOT passed to pipeline.

    Raises:
        FileNotFoundError: If train_transaction.csv is not found in data_dir.
    """
    txn_path = os.path.join(data_dir, "train_transaction.csv")
    identity_path = os.path.join(data_dir, "train_identity.csv")

    if not os.path.exists(txn_path):
        raise FileNotFoundError(
            f"train_transaction.csv not found at {txn_path}. "
            "Download the IEEE-CIS Fraud Detection dataset from Kaggle and place it in the data/ directory."
        )

    # Load transaction CSV; identity is optional (provides DeviceType)
    print(f"Loading {txn_path}...", file=sys.stderr)
    txn_df = pd.read_csv(txn_path)

    if os.path.exists(identity_path):
        print(f"Loading {identity_path}...", file=sys.stderr)
        identity_df = pd.read_csv(identity_path)
        # Merge on TransactionID to pull in DeviceType
        txn_df = txn_df.merge(
            identity_df[["TransactionID", "DeviceType"]],
            on="TransactionID",
            how="left",
        )
    else:
        print(
            f"Warning: {identity_path} not found. DeviceType will default to 'online'.",
            file=sys.stderr,
        )
        txn_df["DeviceType"] = "desktop"

    # Optionally sample before processing (saves time/cost)
    if sample_size is not None and sample_size < len(txn_df):
        txn_df = txn_df.sample(n=sample_size, random_state=seed)
        print(f"Sampled {sample_size} rows from {len(txn_df)} total.", file=sys.stderr)

    print(f"Mapping {len(txn_df)} rows to transaction schema...", file=sys.stderr)

    # Extract ground truth BEFORE mapping (isFraud is the label column)
    ground_truth: dict[str, bool] = {}
    for _, row in txn_df.iterrows():
        txn_id = str(int(row["TransactionID"]))
        ground_truth[txn_id] = bool(row.get("isFraud", 0))

    # Map rows to internal schema
    transactions: list[dict] = []
    for _, row in txn_df.iterrows():
        try:
            txn = _map_transaction_row(row)
            transactions.append(txn)
        except Exception as exc:
            print(
                f"Warning: skipped TransactionID {row.get('TransactionID')}: {exc}",
                file=sys.stderr,
            )

    # Inject AML patterns
    rng = random.Random(seed)
    user_ids = [t["user_id"] for t in transactions]
    transactions, aml_injected = _inject_aml_patterns(
        transactions, user_ids, aml_injection_rate, rng
    )

    print(
        f"Done. {len(transactions)} transactions total "
        f"({len(ground_truth)} original, {len(transactions) - len(ground_truth)} AML-injected). "
        f"{len(aml_injected)} users received AML injection.",
        file=sys.stderr,
    )

    return transactions, ground_truth, aml_injected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load IEEE-CIS dataset, map columns, inject AML patterns, and output JSON lines."
    )
    parser.add_argument(
        "--data-dir",
        default="data/",
        help="Directory containing train_transaction.csv and train_identity.csv (default: data/)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of rows to sample from the CSV. Omit to load all rows.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for transaction JSON lines. Defaults to stdout.",
    )
    parser.add_argument(
        "--aml-injection-rate",
        type=float,
        default=0.02,
        help="Fraction of users to inject AML patterns into (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and injection (default: 42)",
    )
    args = parser.parse_args()

    try:
        transactions, ground_truth, aml_injected = load_ieee_cis(
            data_dir=args.data_dir,
            sample_size=args.sample_size,
            aml_injection_rate=args.aml_injection_rate,
            seed=args.seed,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Write transactions as JSON lines
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for txn in transactions:
                f.write(json.dumps(txn) + "\n")
        print(f"Wrote {len(transactions)} transactions to {args.output}", file=sys.stderr)
    else:
        for txn in transactions:
            sys.stdout.write(json.dumps(txn) + "\n")

    # Always write ground truth and AML injection metadata alongside the output
    if args.output:
        gt_path = args.output.replace(".jsonl", "") + "_ground_truth.json"
        aml_path = args.output.replace(".jsonl", "") + "_aml_injected.json"
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(ground_truth, f, indent=2)
        with open(aml_path, "w", encoding="utf-8") as f:
            json.dump(aml_injected, f, indent=2)
        print(f"Ground truth written to {gt_path}", file=sys.stderr)
        print(f"AML injection map written to {aml_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
