"""
Seed sample behavioral personas into DynamoDB FraudPersonas table.

Usage:
    python scripts/seed_personas.py [--table TABLE_NAME] [--region REGION]

The FraudPersonas table uses PK=userId, SK=version.
Each persona matches the Behavioral Persona schema produced by the Archaeologist.
"""

import argparse
import json
import sys
from decimal import Decimal

import boto3
from boto3.dynamodb.types import TypeSerializer


def get_sample_personas() -> list[dict]:
    """Return 5 realistic behavioral personas for seeding."""
    return [
        {
            # user-123: Dallas TX regular shopper, groceries/gas, low risk
            "userId": "user-123",
            "version": "CURRENT",
            "user_id": "user-123",
            "geo_footprint": [
                {"city": "Dallas", "state": "TX", "frequency": Decimal("0.65")},
                {"city": "Fort Worth", "state": "TX", "frequency": Decimal("0.25")},
                {"city": "Plano", "state": "TX", "frequency": Decimal("0.10")},
            ],
            "category_anchors": [
                {
                    "category": "Groceries",
                    "avg_amount": Decimal("120.00"),
                    "frequency": "weekly",
                    "std_deviation": Decimal("35.00"),
                },
                {
                    "category": "Gas Stations",
                    "avg_amount": Decimal("55.00"),
                    "frequency": "weekly",
                    "std_deviation": Decimal("12.00"),
                },
                {
                    "category": "Restaurants",
                    "avg_amount": Decimal("38.50"),
                    "frequency": "daily",
                    "std_deviation": Decimal("22.00"),
                },
            ],
            "velocity": {
                "daily_txn_count": Decimal("4.2"),
                "daily_spend_amount": Decimal("185.00"),
                "max_single_txn": Decimal("450.00"),
                "hourly_burst_limit": 3,
            },
            "anomaly_history": {
                "false_positive_triggers": ["travel-to-miami"],
                "confirmed_fraud_count": 0,
            },
            "ip_footprint": [
                {
                    "region": "Dallas-Fort Worth, TX",
                    "frequency": Decimal("0.65"),
                    "typical_asn": "AS7018 AT&T",
                },
                {
                    "region": "Plano, TX",
                    "frequency": Decimal("0.25"),
                    "typical_asn": "AS7922 Comcast",
                },
            ],
            "temporal_profile": {
                "active_hours": [7, 22],
                "peak_hour": 12,
                "weekend_ratio": Decimal("0.28"),
                "timezone_estimate": "America/Chicago",
            },
            "aml_profile": {
                "deposit_pattern": {
                    "avg_amount": Decimal("850.00"),
                    "pct_near_threshold": Decimal("0.02"),
                },
                "transfer_pattern": {
                    "avg_count": 3,
                    "unique_counterparties": 12,
                    "pct_high_risk_jurisdictions": Decimal("0.00"),
                },
                "round_trip_score": Decimal("0.00"),
                "economic_profile_match": True,
                "typology_flags": [],
            },
        },
        {
            # user-456: NYC business traveler, frequent flyer, hotels/restaurants
            "userId": "user-456",
            "version": "CURRENT",
            "user_id": "user-456",
            "geo_footprint": [
                {"city": "New York", "state": "NY", "frequency": Decimal("0.72")},
                {"city": "Chicago", "state": "IL", "frequency": Decimal("0.15")},
                {"city": "Los Angeles", "state": "CA", "frequency": Decimal("0.08")},
                {"city": "Miami", "state": "FL", "frequency": Decimal("0.05")},
            ],
            "category_anchors": [
                {
                    "category": "Hotels",
                    "avg_amount": Decimal("320.00"),
                    "frequency": "weekly",
                    "std_deviation": Decimal("110.00"),
                },
                {
                    "category": "Restaurants",
                    "avg_amount": Decimal("85.00"),
                    "frequency": "daily",
                    "std_deviation": Decimal("45.00"),
                },
                {
                    "category": "Airlines",
                    "avg_amount": Decimal("620.00"),
                    "frequency": "monthly",
                    "std_deviation": Decimal("200.00"),
                },
            ],
            "velocity": {
                "daily_txn_count": Decimal("6.8"),
                "daily_spend_amount": Decimal("780.00"),
                "max_single_txn": Decimal("1200.00"),
                "hourly_burst_limit": 4,
            },
            "anomaly_history": {
                "false_positive_triggers": ["travel-to-chicago", "large-hotel-purchase"],
                "confirmed_fraud_count": 0,
            },
            "ip_footprint": [
                {
                    "region": "New York City, NY",
                    "frequency": Decimal("0.70"),
                    "typical_asn": "AS7922 Comcast",
                },
                {
                    "region": "Chicago, IL",
                    "frequency": Decimal("0.14"),
                    "typical_asn": "AS7018 AT&T",
                },
            ],
            "temporal_profile": {
                "active_hours": [6, 23],
                "peak_hour": 14,
                "weekend_ratio": Decimal("0.35"),
                "timezone_estimate": "America/New_York",
            },
            "aml_profile": {
                "deposit_pattern": {
                    "avg_amount": Decimal("4200.00"),
                    "pct_near_threshold": Decimal("0.04"),
                },
                "transfer_pattern": {
                    "avg_count": 8,
                    "unique_counterparties": 22,
                    "pct_high_risk_jurisdictions": Decimal("0.00"),
                },
                "round_trip_score": Decimal("0.00"),
                "economic_profile_match": True,
                "typology_flags": [],
            },
        },
        {
            # user-789: San Francisco tech worker, online shopping, high single-txn
            "userId": "user-789",
            "version": "CURRENT",
            "user_id": "user-789",
            "geo_footprint": [
                {"city": "San Francisco", "state": "CA", "frequency": Decimal("0.80")},
                {"city": "San Jose", "state": "CA", "frequency": Decimal("0.12")},
                {"city": "Oakland", "state": "CA", "frequency": Decimal("0.08")},
            ],
            "category_anchors": [
                {
                    "category": "Digital Goods",
                    "avg_amount": Decimal("92.00"),
                    "frequency": "weekly",
                    "std_deviation": Decimal("60.00"),
                },
                {
                    "category": "Electronics",
                    "avg_amount": Decimal("380.00"),
                    "frequency": "monthly",
                    "std_deviation": Decimal("280.00"),
                },
                {
                    "category": "Restaurants",
                    "avg_amount": Decimal("65.00"),
                    "frequency": "daily",
                    "std_deviation": Decimal("30.00"),
                },
            ],
            "velocity": {
                "daily_txn_count": Decimal("5.1"),
                "daily_spend_amount": Decimal("320.00"),
                "max_single_txn": Decimal("2800.00"),
                "hourly_burst_limit": 5,
            },
            "anomaly_history": {
                "false_positive_triggers": ["large-electronics-purchase", "gaming-subscription"],
                "confirmed_fraud_count": 0,
            },
            "ip_footprint": [
                {
                    "region": "San Francisco Bay Area, CA",
                    "frequency": Decimal("0.88"),
                    "typical_asn": "AS7922 Comcast",
                },
                {
                    "region": "San Jose, CA",
                    "frequency": Decimal("0.10"),
                    "typical_asn": "AS16591 Google Fiber",
                },
            ],
            "temporal_profile": {
                "active_hours": [9, 2],
                "peak_hour": 20,
                "weekend_ratio": Decimal("0.42"),
                "timezone_estimate": "America/Los_Angeles",
            },
            "aml_profile": {
                "deposit_pattern": {
                    "avg_amount": Decimal("2100.00"),
                    "pct_near_threshold": Decimal("0.01"),
                },
                "transfer_pattern": {
                    "avg_count": 2,
                    "unique_counterparties": 5,
                    "pct_high_risk_jurisdictions": Decimal("0.00"),
                },
                "round_trip_score": Decimal("0.00"),
                "economic_profile_match": True,
                "typology_flags": [],
            },
        },
        {
            # user-1000: Miami retiree, small consistent transactions
            "userId": "user-1000",
            "version": "CURRENT",
            "user_id": "user-1000",
            "geo_footprint": [
                {"city": "Miami", "state": "FL", "frequency": Decimal("0.90")},
                {"city": "Fort Lauderdale", "state": "FL", "frequency": Decimal("0.10")},
            ],
            "category_anchors": [
                {
                    "category": "Groceries",
                    "avg_amount": Decimal("65.00"),
                    "frequency": "weekly",
                    "std_deviation": Decimal("18.00"),
                },
                {
                    "category": "Pharmacies",
                    "avg_amount": Decimal("42.00"),
                    "frequency": "monthly",
                    "std_deviation": Decimal("15.00"),
                },
                {
                    "category": "Restaurants",
                    "avg_amount": Decimal("28.00"),
                    "frequency": "weekly",
                    "std_deviation": Decimal("10.00"),
                },
            ],
            "velocity": {
                "daily_txn_count": Decimal("2.1"),
                "daily_spend_amount": Decimal("95.00"),
                "max_single_txn": Decimal("200.00"),
                "hourly_burst_limit": 2,
            },
            "anomaly_history": {
                "false_positive_triggers": [],
                "confirmed_fraud_count": 0,
            },
            "ip_footprint": [
                {
                    "region": "Miami, FL",
                    "frequency": Decimal("0.92"),
                    "typical_asn": "AS11427 Cox Communications",
                },
            ],
            "temporal_profile": {
                "active_hours": [8, 20],
                "peak_hour": 10,
                "weekend_ratio": Decimal("0.30"),
                "timezone_estimate": "America/New_York",
            },
            "aml_profile": {
                "deposit_pattern": {
                    "avg_amount": Decimal("1200.00"),
                    "pct_near_threshold": Decimal("0.00"),
                },
                "transfer_pattern": {
                    "avg_count": 1,
                    "unique_counterparties": 3,
                    "pct_high_risk_jurisdictions": Decimal("0.00"),
                },
                "round_trip_score": Decimal("0.00"),
                "economic_profile_match": True,
                "typology_flags": [],
            },
        },
        {
            # user-2000: Chicago student, low velocity, weekend-heavy
            "userId": "user-2000",
            "version": "CURRENT",
            "user_id": "user-2000",
            "geo_footprint": [
                {"city": "Chicago", "state": "IL", "frequency": Decimal("0.85")},
                {"city": "Evanston", "state": "IL", "frequency": Decimal("0.12")},
                {"city": "Oak Park", "state": "IL", "frequency": Decimal("0.03")},
            ],
            "category_anchors": [
                {
                    "category": "Restaurants",
                    "avg_amount": Decimal("22.00"),
                    "frequency": "weekly",
                    "std_deviation": Decimal("12.00"),
                },
                {
                    "category": "Digital Goods",
                    "avg_amount": Decimal("15.00"),
                    "frequency": "monthly",
                    "std_deviation": Decimal("8.00"),
                },
                {
                    "category": "Groceries",
                    "avg_amount": Decimal("48.00"),
                    "frequency": "weekly",
                    "std_deviation": Decimal("20.00"),
                },
            ],
            "velocity": {
                "daily_txn_count": Decimal("1.8"),
                "daily_spend_amount": Decimal("55.00"),
                "max_single_txn": Decimal("120.00"),
                "hourly_burst_limit": 2,
            },
            "anomaly_history": {
                "false_positive_triggers": [],
                "confirmed_fraud_count": 0,
            },
            "ip_footprint": [
                {
                    "region": "Chicago, IL",
                    "frequency": Decimal("0.85"),
                    "typical_asn": "AS7018 AT&T",
                },
                {
                    "region": "Evanston, IL",
                    "frequency": Decimal("0.12"),
                    "typical_asn": "AS7018 AT&T",
                },
            ],
            "temporal_profile": {
                "active_hours": [10, 2],
                "peak_hour": 19,
                "weekend_ratio": Decimal("0.58"),
                "timezone_estimate": "America/Chicago",
            },
            "aml_profile": {
                "deposit_pattern": {
                    "avg_amount": Decimal("300.00"),
                    "pct_near_threshold": Decimal("0.00"),
                },
                "transfer_pattern": {
                    "avg_count": 1,
                    "unique_counterparties": 2,
                    "pct_high_risk_jurisdictions": Decimal("0.00"),
                },
                "round_trip_score": Decimal("0.00"),
                "economic_profile_match": True,
                "typology_flags": [],
            },
        },
    ]


def seed_personas(table_name: str, region: str) -> None:
    """Write all sample personas to DynamoDB."""
    dynamodb = boto3.resource("dynamodb", region_name=region)
    table = dynamodb.Table(table_name)

    personas = get_sample_personas()
    succeeded = 0
    failed = 0

    for persona in personas:
        user_id = persona["userId"]
        try:
            table.put_item(Item=persona)
            print(f"  Seeded persona for {user_id}")
            succeeded += 1
        except Exception as exc:
            print(f"  ERROR seeding {user_id}: {exc}", file=sys.stderr)
            failed += 1

    print(f"\nDone. Seeded {succeeded} personas, {failed} failed.")
    if failed > 0:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed sample behavioral personas into the FraudPersonas DynamoDB table."
    )
    parser.add_argument(
        "--table",
        default="FraudPersonas",
        help="DynamoDB table name (default: FraudPersonas)",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region (default: us-east-1)",
    )
    args = parser.parse_args()

    print(f"Seeding personas into table '{args.table}' in region '{args.region}'...")
    seed_personas(table_name=args.table, region=args.region)


if __name__ == "__main__":
    main()
