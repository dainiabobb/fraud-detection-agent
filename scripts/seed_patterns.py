"""
Seed initial fraud patterns into DynamoDB FraudPatterns table.

Usage:
    python scripts/seed_patterns.py [--table TABLE_NAME] [--region REGION]

The FraudPatterns table uses PK=patternName (no SK).
Patterns are consumed by the Sentinel for Tier 1 detection and refined by
the Pattern Discovery Lambda over time.
"""

import argparse
import sys
from decimal import Decimal

import boto3


def get_seed_patterns() -> list[dict]:
    """Return 3 seed fraud patterns."""
    return [
        {
            # Multiple small purchases under $50 from different merchants within 1 hour.
            # Classic card-testing or credential-stuffing pattern where attackers validate
            # stolen card numbers with low-value purchases before escalating.
            "patternName": "rapid-fire-small-purchases",
            "description": (
                "Multiple small purchases (<$50) within 1 hour from different merchants. "
                "Indicates card-testing or credential-stuffing attack pattern."
            ),
            "signals": {
                "max_amount_per_txn": Decimal("50.00"),
                "min_txn_count": 3,
                "time_window_minutes": 60,
                "requires_different_merchants": True,
            },
            "thresholds": {
                "escalation_score": Decimal("0.78"),
                "auto_block_score": Decimal("0.95"),
            },
            "precision": Decimal("0.82"),
            "recall": Decimal("0.71"),
            "f1_score": Decimal("0.76"),
            "sample_count": 247,
            "false_positive_rate": Decimal("0.12"),
            "status": "ACTIVE",
            "discovered_at": "2026-01-15T08:00:00Z",
            "last_updated_at": "2026-03-01T06:00:00Z",
            "version": 3,
        },
        {
            # Transactions from locations more than 500 miles apart within 2 hours.
            # Physically impossible without air travel and indicates compromised credentials
            # being used from a geographically distant attacker.
            "patternName": "geo-impossible-travel",
            "description": (
                "Transactions from locations >500 miles apart within 2 hours. "
                "Physically impossible travel window indicates account takeover."
            ),
            "signals": {
                "min_distance_miles": 500,
                "max_time_window_minutes": 120,
                "requires_different_ip_asn": True,
            },
            "thresholds": {
                "escalation_score": Decimal("0.70"),
                "auto_block_score": Decimal("0.92"),
            },
            "precision": Decimal("0.91"),
            "recall": Decimal("0.68"),
            "f1_score": Decimal("0.78"),
            "sample_count": 183,
            "false_positive_rate": Decimal("0.06"),
            "status": "ACTIVE",
            "discovered_at": "2026-01-15T08:00:00Z",
            "last_updated_at": "2026-03-15T06:00:00Z",
            "version": 2,
        },
        {
            # A device making its first-ever appearance on the account initiates a
            # transaction over $500. New device combined with high value is a strong
            # indicator of account takeover after credential compromise.
            "patternName": "new-device-high-value",
            "description": (
                "First-time device making a transaction >$500. "
                "New device combined with high-value purchase strongly indicates account takeover."
            ),
            "signals": {
                "min_amount": Decimal("500.00"),
                "device_seen_before": False,
                "account_age_days_min": 30,
            },
            "thresholds": {
                "escalation_score": Decimal("0.72"),
                "auto_block_score": Decimal("0.90"),
            },
            "precision": Decimal("0.76"),
            "recall": Decimal("0.85"),
            "f1_score": Decimal("0.80"),
            "sample_count": 412,
            "false_positive_rate": Decimal("0.18"),
            "status": "ACTIVE",
            "discovered_at": "2026-01-15T08:00:00Z",
            "last_updated_at": "2026-02-20T06:00:00Z",
            "version": 4,
        },
    ]


def seed_patterns(table_name: str, region: str) -> None:
    """Write all seed patterns to DynamoDB."""
    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.Table(table_name)

    patterns = get_seed_patterns()
    succeeded = 0
    failed = 0

    for pattern in patterns:
        name = pattern["patternName"]
        try:
            table.put_item(Item=pattern)
            print(f"  Seeded pattern: {name}")
            succeeded += 1
        except Exception as exc:
            print(f"  ERROR seeding '{name}': {exc}", file=sys.stderr)
            failed += 1

    print(f"\nDone. Seeded {succeeded} patterns, {failed} failed.")
    if failed > 0:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed initial fraud patterns into the FraudPatterns DynamoDB table."
    )
    parser.add_argument(
        "--table",
        default="FraudPatterns",
        help="DynamoDB table name (default: FraudPatterns)",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region (default: us-east-1)",
    )
    args = parser.parse_args()

    print(f"Seeding patterns into table '{args.table}' in region '{args.region}'...")
    seed_patterns(table_name=args.table, region=args.region)


if __name__ == "__main__":
    main()
