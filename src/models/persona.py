"""
Behavioral Persona data models.

The Archaeologist (Tier 3) compresses 24 months of raw transaction logs
into one BehavioralPersona per user and writes it to the FraudPersonas
DynamoDB table (PK=userId, SK=version).

The nested model hierarchy mirrors the JSON schema documented in the README
exactly so that DynamoDB item serialisation and deserialisation are lossless.
"""

from __future__ import annotations

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Geo footprint
# ---------------------------------------------------------------------------

class GeoEntry(BaseModel):
    """A city the user has transacted from, with a relative frequency weight."""

    city: str
    state: str
    # Fraction of all transactions originating from this city (0.0 – 1.0)
    frequency: float


# ---------------------------------------------------------------------------
# Spending category anchors
# ---------------------------------------------------------------------------

class CategoryAnchor(BaseModel):
    """Typical spending behaviour for one merchant category."""

    category: str
    avg_amount: float
    # Human-readable cadence string, e.g. "weekly", "monthly"
    frequency: str
    std_deviation: float


# ---------------------------------------------------------------------------
# Transaction velocity
# ---------------------------------------------------------------------------

class VelocityProfile(BaseModel):
    """Rolling averages and hard limits derived from historical velocity."""

    daily_txn_count: float
    daily_spend_amount: float
    max_single_txn: float
    # Maximum transactions the user has ever made within a single clock hour
    hourly_burst_limit: int


# ---------------------------------------------------------------------------
# Historical anomaly context
# ---------------------------------------------------------------------------

class AnomalyHistory(BaseModel):
    """
    Lightweight record of past anomalies to reduce Sentinel false positives.

    false_positive_triggers: pattern or event labels that previously fired
                             but turned out to be legitimate behaviour
                             (e.g. "travel-to-miami").
    """

    false_positive_triggers: list[str]
    confirmed_fraud_count: int


# ---------------------------------------------------------------------------
# IP / network footprint
# ---------------------------------------------------------------------------

class IPEntry(BaseModel):
    """A network region the user typically connects from."""

    # Human-readable metro area, e.g. "Dallas-Fort Worth, TX"
    region: str
    frequency: float
    # Representative ASN for this region, e.g. "AS7018 AT&T"
    typical_asn: str


# ---------------------------------------------------------------------------
# Temporal behaviour
# ---------------------------------------------------------------------------

class TemporalProfile(BaseModel):
    """When the user is typically active."""

    # [start_hour, end_hour] in 24-hour format (local time)
    active_hours: list[int]
    peak_hour: int
    # Fraction of transactions that occur on weekends (0.0 – 1.0)
    weekend_ratio: float
    # IANA timezone string, e.g. "America/Chicago"
    timezone_estimate: str


# ---------------------------------------------------------------------------
# AML sub-models
# ---------------------------------------------------------------------------

class DepositPattern(BaseModel):
    """Aggregate statistics for inbound deposit transactions."""

    avg_amount: float
    # Fraction of deposits within $500 of the $10,000 CTR threshold
    pct_near_threshold: float


class TransferPattern(BaseModel):
    """Aggregate statistics for outbound transfer transactions."""

    avg_count: float
    unique_counterparties: int
    # Fraction of transfers to FATF high-risk jurisdictions (0.0 – 1.0)
    pct_high_risk_jurisdictions: float


class AMLProfile(BaseModel):
    """AML-specific behavioural signals synthesised by the Archaeologist."""

    deposit_pattern: DepositPattern
    transfer_pattern: TransferPattern
    # 0.0 = no round-trip signal; 1.0 = strong round-trip signal
    round_trip_score: float
    # True when transaction volume is consistent with KYC-declared income
    economic_profile_match: bool
    # AML typology labels flagged by prior analysis runs
    typology_flags: list[str]


# ---------------------------------------------------------------------------
# Root persona
# ---------------------------------------------------------------------------

class BehavioralPersona(BaseModel):
    """
    Complete behavioral profile for one user.

    Written by the Archaeologist to FraudPersonas (PK=userId, SK=version)
    and cached in Redis for sub-millisecond Sentinel reads.
    """

    user_id: str
    geo_footprint: list[GeoEntry]
    category_anchors: list[CategoryAnchor]
    velocity: VelocityProfile
    anomaly_history: AnomalyHistory
    ip_footprint: list[IPEntry]
    temporal_profile: TemporalProfile
    aml_profile: AMLProfile
    # Schema version — bump when the shape changes to allow forward migration
    version: str = "v1"
