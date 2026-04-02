"""
GeoIP enrichment utilities for the fraud detection agent.

Looks up geographic metadata for an IP address using the MaxMind GeoLite2
City database.  The database is expected at /opt/GeoLite2-City.mmdb when
running inside a Lambda layer; the GEOIP_DB_PATH environment variable
provides a fallback for local development.

Also provides a helper to convert a UTC ISO-8601 timestamp to the local hour
in a given IANA timezone — used to populate the `local_hour` feature for
embedding and rule evaluation.
"""

import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)

# Prefer the Lambda layer path; fall back to an env-var-controlled local path.
_LAMBDA_DB_PATH = "/opt/GeoLite2-City.mmdb"
_FALLBACK_DB_PATH = os.environ.get("GEOIP_DB_PATH", "")


def _get_db_path() -> str | None:
    """Return the first GeoLite2 DB path that exists on disk, or None."""
    for path in (_LAMBDA_DB_PATH, _FALLBACK_DB_PATH):
        if path and os.path.exists(path):
            return path
    return None


def enrich_geoip(ip_address: str | None) -> dict:
    """Look up geographic metadata for an IP address.

    Args:
        ip_address: IPv4 or IPv6 address string, or None.

    Returns:
        Dict with keys geo_city, geo_country, geo_lat, geo_lon, is_vpn.
        Returns an empty dict if ip_address is None, the database cannot be
        found, or any lookup error occurs — callers must treat enrichment as
        best-effort.
    """
    if not ip_address:
        return {}

    db_path = _get_db_path()
    if db_path is None:
        logger.warning(
            "GeoIP database not found at %s or GEOIP_DB_PATH; skipping enrichment",
            _LAMBDA_DB_PATH,
        )
        return {}

    try:
        # Import here so the module can be imported in environments that do
        # not have geoip2 installed without raising an ImportError at the top.
        import geoip2.database  # type: ignore[import-untyped]
        import geoip2.errors  # type: ignore[import-untyped]

        with geoip2.database.Reader(db_path) as reader:
            response = reader.city(ip_address)

        city: str = response.city.name or ""
        country: str = response.country.iso_code or ""
        lat: float | None = response.location.latitude
        lon: float | None = response.location.longitude

        return {
            "geo_city": city,
            "geo_country": country,
            "geo_lat": lat,
            "geo_lon": lon,
            # GeoLite2 City does not carry a VPN flag; default False.
            # A commercial MaxMind product or separate dataset would be needed
            # for reliable VPN/proxy detection.
            "is_vpn": False,
        }

    except Exception as exc:  # noqa: BLE001
        # Intentionally broad: network errors, bad IP strings, corrupt DB, etc.
        logger.warning(
            "GeoIP lookup failed for ip=%s: %s",
            ip_address,
            exc,
            exc_info=True,
        )
        return {}


def calculate_local_hour(timestamp_iso: str, timezone_str: str | None) -> int | None:
    """Convert a UTC ISO-8601 timestamp to the local hour in the given timezone.

    Args:
        timestamp_iso: ISO-8601 datetime string, e.g. "2024-03-15T14:30:00Z"
                       or "2024-03-15T14:30:00+00:00".
        timezone_str:  IANA timezone name, e.g. "America/Chicago".  If None
                       or empty, the function returns None.

    Returns:
        Local hour as an integer in [0, 23], or None if the timezone is
        unknown or the timestamp cannot be parsed.
    """
    if not timezone_str:
        return None

    try:
        tz = ZoneInfo(timezone_str)
    except (ZoneInfoNotFoundError, KeyError):
        logger.warning("Unknown timezone string: %s", timezone_str)
        return None

    try:
        # datetime.fromisoformat handles both 'Z' suffix (Python 3.11+) and
        # '+00:00' offset notation.
        ts = timestamp_iso.replace("Z", "+00:00")
        dt_utc = datetime.fromisoformat(ts)
        dt_local = dt_utc.astimezone(tz)
        return dt_local.hour
    except (ValueError, OverflowError) as exc:
        logger.warning(
            "Failed to parse timestamp '%s': %s", timestamp_iso, exc
        )
        return None
