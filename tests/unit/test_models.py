"""
Unit tests for all Pydantic data models.

Tests cover construction with valid data, validation errors for missing or
invalid fields, serialisation round-trips, nested structures, and default
values.  No AWS clients or services are touched.
"""

import pytest
from pydantic import ValidationError

from src.models.aml import AMLRiskScore, AMLSignal, InvestigationCase, ScoreUpdate
from src.models.decision import FraudDecision
from src.models.pattern import FraudPattern
from src.models.persona import (
    AMLProfile,
    AnomalyHistory,
    BehavioralPersona,
    CategoryAnchor,
    DepositPattern,
    GeoEntry,
    IPEntry,
    TemporalProfile,
    TransferPattern,
    VelocityProfile,
)
from src.models.transaction import EnrichedTransaction, Transaction


# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------


class TestTransaction:
    def test_valid_transaction_creation(self):
        txn = Transaction(
            transaction_id="txn-001",
            user_id="user-001",
            amount=99.99,
            channel="online",
            timestamp="2026-03-31T14:00:00Z",
        )
        assert txn.transaction_id == "txn-001"
        assert txn.amount == 99.99
        # Defaults applied correctly
        assert txn.currency == "USD"
        assert txn.merchant_country == "US"
        assert txn.ip_address is None

    def test_transaction_missing_required_field_raises(self):
        # 'channel' is required with no default
        with pytest.raises(ValidationError) as exc_info:
            Transaction(
                transaction_id="txn-002",
                user_id="user-002",
                amount=50.0,
                # channel omitted
                timestamp="2026-03-31T14:00:00Z",
            )
        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "channel" in field_names

    def test_transaction_missing_amount_raises(self):
        with pytest.raises(ValidationError):
            Transaction(
                transaction_id="txn-003",
                user_id="user-003",
                channel="mobile",
                timestamp="2026-03-31T14:00:00Z",
            )

    def test_transaction_optional_fields_default_to_none(self):
        txn = Transaction(
            transaction_id="txn-004",
            user_id="user-004",
            amount=10.0,
            channel="in-store",
            timestamp="2026-03-31T09:00:00Z",
        )
        assert txn.merchant_name is None
        assert txn.merchant_city is None
        assert txn.device_id is None
        assert txn.card_last_four is None


# ---------------------------------------------------------------------------
# EnrichedTransaction
# ---------------------------------------------------------------------------


class TestEnrichedTransaction:
    def test_enriched_inherits_transaction_fields(self):
        enriched = EnrichedTransaction(
            transaction_id="txn-005",
            user_id="user-005",
            amount=200.0,
            channel="online",
            timestamp="2026-03-31T20:00:00Z",
            geo_city="Dallas",
            geo_country="US",
            geo_lat=32.78,
            geo_lon=-96.8,
            local_hour=14,
            is_vpn=False,
        )
        # Inherited required fields present
        assert enriched.transaction_id == "txn-005"
        assert enriched.amount == 200.0
        # Enrichment fields present
        assert enriched.geo_city == "Dallas"
        assert enriched.local_hour == 14
        assert enriched.is_vpn is False

    def test_enriched_geo_fields_default_to_none(self):
        enriched = EnrichedTransaction(
            transaction_id="txn-006",
            user_id="user-006",
            amount=50.0,
            channel="mobile",
            timestamp="2026-03-31T10:00:00Z",
        )
        assert enriched.geo_city is None
        assert enriched.geo_country is None
        assert enriched.local_hour is None
        assert enriched.is_vpn is False


# ---------------------------------------------------------------------------
# FraudDecision
# ---------------------------------------------------------------------------


class TestFraudDecision:
    def _make_decision(self, **overrides) -> dict:
        base = {
            "transaction_id": "txn-007",
            "user_id": "user-007",
            "verdict": "APPROVE",
            "tier": "SENTINEL",
            "confidence": 0.92,
            "reasoning": "High similarity to legitimate transactions.",
            "timestamp": "2026-03-31T14:05:00+00:00",
        }
        return {**base, **overrides}

    def test_valid_decision_creation(self):
        decision = FraudDecision(**self._make_decision())
        assert decision.verdict == "APPROVE"
        assert decision.tier == "SENTINEL"
        assert decision.confidence == 0.92
        # Defaults
        assert decision.escalation_target is None
        assert decision.pattern_matches == []
        assert decision.tokens_used == 0
        assert decision.latency_ms == 0

    def test_ttl_field_is_optional(self):
        # No ttl supplied — should default to None
        decision = FraudDecision(**self._make_decision())
        assert decision.ttl is None

    def test_ttl_field_accepted_when_provided(self):
        decision = FraudDecision(**self._make_decision(ttl=1_800_000_000))
        assert decision.ttl == 1_800_000_000

    def test_serialisation_round_trip(self):
        original = FraudDecision(**self._make_decision(verdict="ESCALATE", escalation_target="FRAUD_ONLY"))
        dumped = original.model_dump()
        restored = FraudDecision.model_validate(dumped)
        assert restored.verdict == original.verdict
        assert restored.escalation_target == original.escalation_target
        assert restored.timestamp == original.timestamp

    def test_pattern_matches_stored_correctly(self):
        decision = FraudDecision(
            **self._make_decision(
                verdict="ESCALATE",
                escalation_target="FRAUD_ONLY",
                pattern_matches=["rapid-geo-shift", "high-value-new-merchant"],
            )
        )
        assert len(decision.pattern_matches) == 2
        assert "rapid-geo-shift" in decision.pattern_matches


# ---------------------------------------------------------------------------
# BehavioralPersona (full nested structure)
# ---------------------------------------------------------------------------


class TestBehavioralPersona:
    def test_full_persona_construction(self, sample_persona):
        persona = BehavioralPersona(**sample_persona)
        assert persona.user_id == "user-7890"
        assert len(persona.geo_footprint) == 1
        assert persona.geo_footprint[0].city == "Dallas"
        assert persona.velocity.daily_txn_count == 4.2
        assert persona.temporal_profile.timezone_estimate == "America/Chicago"
        assert persona.aml_profile.economic_profile_match is True
        assert persona.version == "v1"

    def test_persona_aml_profile_nested_models(self, sample_persona):
        persona = BehavioralPersona(**sample_persona)
        aml = persona.aml_profile
        assert aml.deposit_pattern.avg_amount == 850.0
        assert aml.transfer_pattern.unique_counterparties == 12
        assert aml.round_trip_score == 0.0
        assert aml.typology_flags == []

    def test_persona_default_version(self):
        persona = BehavioralPersona(
            user_id="user-999",
            geo_footprint=[GeoEntry(city="Austin", state="TX", frequency=1.0)],
            category_anchors=[
                CategoryAnchor(
                    category="Retail",
                    avg_amount=80.0,
                    frequency="monthly",
                    std_deviation=20.0,
                )
            ],
            velocity=VelocityProfile(
                daily_txn_count=2.0,
                daily_spend_amount=100.0,
                max_single_txn=300.0,
                hourly_burst_limit=2,
            ),
            anomaly_history=AnomalyHistory(
                false_positive_triggers=[],
                confirmed_fraud_count=0,
            ),
            ip_footprint=[
                IPEntry(
                    region="Austin, TX",
                    frequency=1.0,
                    typical_asn="AS7018 AT&T",
                )
            ],
            temporal_profile=TemporalProfile(
                active_hours=[8, 21],
                peak_hour=10,
                weekend_ratio=0.3,
                timezone_estimate="America/Chicago",
            ),
            aml_profile=AMLProfile(
                deposit_pattern=DepositPattern(avg_amount=500.0, pct_near_threshold=0.0),
                transfer_pattern=TransferPattern(
                    avg_count=1.0,
                    unique_counterparties=3,
                    pct_high_risk_jurisdictions=0.0,
                ),
                round_trip_score=0.0,
                economic_profile_match=True,
                typology_flags=[],
            ),
        )
        assert persona.version == "v1"


# ---------------------------------------------------------------------------
# AMLRiskScore and ScoreUpdate
# ---------------------------------------------------------------------------


class TestAMLRiskScore:
    def test_aml_risk_score_creation(self):
        update = ScoreUpdate(
            timestamp="2026-03-31T14:00:00Z",
            delta=15.0,
            reason="Structuring detected",
            transaction_id="txn-100",
        )
        score = AMLRiskScore(
            user_id="user-200",
            current_score=15.0,
            last_updated="2026-03-31T14:00:00Z",
            score_history=[update],
        )
        assert score.current_score == 15.0
        assert len(score.score_history) == 1
        assert score.investigation_case_id is None

    def test_score_history_multiple_entries(self):
        updates = [
            ScoreUpdate(
                timestamp=f"2026-03-{20 + i:02d}T10:00:00Z",
                delta=float(10 * (i + 1)),
                reason=f"Event {i}",
                transaction_id=f"txn-{i}",
            )
            for i in range(3)
        ]
        score = AMLRiskScore(
            user_id="user-201",
            current_score=60.0,
            last_updated="2026-03-22T10:00:00Z",
            score_history=updates,
            investigation_case_id="CASE-001",
        )
        assert len(score.score_history) == 3
        assert score.investigation_case_id == "CASE-001"


# ---------------------------------------------------------------------------
# InvestigationCase
# ---------------------------------------------------------------------------


class TestInvestigationCase:
    def test_open_case_creation(self):
        case = InvestigationCase(
            case_id="CASE-001",
            user_id="user-300",
            status="OPEN",
            opened_at="2026-03-31T10:00:00Z",
            risk_score_at_open=52.0,
            typologies_detected=["STRUCTURING"],
            transactions=["txn-A", "txn-B"],
            notes=["Initial structuring pattern detected."],
        )
        assert case.status == "OPEN"
        assert case.escalated_at is None
        assert case.closed_at is None
        assert "txn-A" in case.transactions

    def test_escalated_case_has_escalated_at(self):
        case = InvestigationCase(
            case_id="CASE-002",
            user_id="user-301",
            status="ESCALATED",
            opened_at="2026-03-01T10:00:00Z",
            escalated_at="2026-03-15T10:00:00Z",
            risk_score_at_open=55.0,
            typologies_detected=["LAYERING"],
            transactions=["txn-C"],
            notes=["Escalated after layering confirmed."],
        )
        assert case.status == "ESCALATED"
        assert case.escalated_at == "2026-03-15T10:00:00Z"
        assert case.closed_at is None


# ---------------------------------------------------------------------------
# AMLSignal
# ---------------------------------------------------------------------------


class TestAMLSignal:
    @pytest.mark.parametrize(
        "signal_type,confidence,details",
        [
            ("STRUCTURING", 0.85, {"amount": 9200.0}),
            ("SMURFING", 0.70, {"feeder_count": 4}),
            ("LAYERING", 0.75, {"hop_count": 3}),
            ("ROUND_TRIPPING", 0.65, {"amount": 5000.0}),
            ("U_TURN", 0.90, {"merchant_country": "IR"}),
            ("PROFILE_MISMATCH", 0.75, {"source": "persona"}),
        ],
    )
    def test_aml_signal_creation_each_typology(self, signal_type, confidence, details):
        signal = AMLSignal(
            signal_type=signal_type,
            confidence=confidence,
            details=details,
        )
        assert signal.signal_type == signal_type
        assert signal.confidence == confidence
        assert signal.details == details


# ---------------------------------------------------------------------------
# FraudPattern
# ---------------------------------------------------------------------------


class TestFraudPattern:
    def test_pattern_with_defaults(self):
        pattern = FraudPattern(
            pattern_name="rapid-geo-shift",
            description="Card used across two distant cities within 30 minutes.",
            detection_rule="Transaction within 30 min of prior txn >500 miles away.",
            precision=0.82,
            sample_transaction_ids=["txn-X", "txn-Y", "txn-Z"],
            discovered_at="2026-03-01T06:00:00Z",
        )
        # Defaults
        assert pattern.recall is None
        assert pattern.last_matched_at is None
        assert pattern.match_count == 0
        assert pattern.status == "ACTIVE"
        assert pattern.false_positive_rate == 0.0

    def test_pattern_retired_status(self):
        pattern = FraudPattern(
            pattern_name="old-pattern",
            description="Stale pattern.",
            detection_rule="Some rule.",
            precision=0.51,
            sample_transaction_ids=["txn-1"],
            discovered_at="2025-01-01T00:00:00Z",
            status="RETIRED",
            false_positive_rate=0.35,
        )
        assert pattern.status == "RETIRED"
        assert pattern.false_positive_rate == 0.35
