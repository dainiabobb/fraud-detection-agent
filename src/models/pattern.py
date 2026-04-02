"""
Pattern data model.

FraudPattern records are produced by the Pattern Discovery Lambda
(daily EventBridge schedule) and stored in the FraudPatterns DynamoDB
table (PK=patternName, no SK).

Active patterns are pulled into the Sentinel's gray-zone decision path to
enable faster Tier 1 detection without a round-trip to Tier 2.
"""

from __future__ import annotations

from pydantic import BaseModel


class FraudPattern(BaseModel):
    """
    An emergent attack pattern discovered by clustering recent fraud decisions.

    Lifecycle:
      ACTIVE  — pattern is loaded into the Sentinel's pattern cache (Redis)
                and matched against incoming transactions.
      RETIRED — pattern's false-positive rate exceeded the refinement threshold
                or no matches have occurred for an extended period.

    A pattern is created only when a cluster of 3+ confirmed fraud cases
    shares >70 % feature overlap (precision threshold).
    """

    # Short, human-readable identifier, e.g. "rapid-geo-shift-high-value"
    pattern_name: str
    description: str
    # Natural language rule passed verbatim to the Sentinel prompt
    detection_rule: str
    # Fraction of Sentinel matches that were subsequently confirmed as fraud
    precision: float
    # Fraction of actual fraud cases this pattern catches; None until measured
    recall: float | None = None
    # Representative transaction IDs from the discovery cluster
    sample_transaction_ids: list[str]
    # ISO 8601 UTC timestamp when Pattern Discovery first wrote this record
    discovered_at: str
    # ISO 8601 UTC timestamp of the most recent Sentinel match; None if never matched
    last_matched_at: str | None = None
    # Cumulative Sentinel match count since the pattern was created
    match_count: int = 0
    # ACTIVE | RETIRED
    status: str = "ACTIVE"
    # Running estimate; Pattern Discovery refines or retires when this is too high
    false_positive_rate: float = 0.0
