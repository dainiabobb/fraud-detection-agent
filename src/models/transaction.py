"""
Transaction data models.

Transaction is the raw event ingested from Kinesis.
EnrichedTransaction extends it with GeoIP, VPN detection,
and temporal fields added during Sentinel pre-processing.
"""

from __future__ import annotations

from pydantic import BaseModel


class Transaction(BaseModel):
    """Raw transaction event as received from the ingestion stream."""

    transaction_id: str
    user_id: str
    amount: float
    currency: str = "USD"
    merchant_name: str | None = None
    merchant_category: str | None = None
    # Physical or inferred location of the merchant
    merchant_city: str | None = None
    merchant_country: str = "US"
    # Channel values: "online" | "mobile" | "in-store"
    channel: str
    # ISO 8601 UTC timestamp, e.g. "2026-03-31T14:05:00Z"
    timestamp: str
    ip_address: str | None = None
    device_id: str | None = None
    card_last_four: str | None = None


class EnrichedTransaction(Transaction):
    """
    Transaction augmented by the Sentinel pre-processing step.

    Added fields come from:
    - MaxMind GeoLite2 (geo_* fields, is_vpn)
    - Temporal expansion (local_hour derived from geo_lon + timestamp)
    """

    # GeoIP-resolved location for the originating IP address
    geo_city: str | None = None
    geo_country: str | None = None
    geo_lat: float | None = None
    geo_lon: float | None = None
    # Local hour at the cardholder's estimated location (0-23)
    local_hour: int | None = None
    # True when MaxMind flags the IP as belonging to a VPN/proxy ASN
    is_vpn: bool = False
