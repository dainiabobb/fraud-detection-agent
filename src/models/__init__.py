"""
src.models — Pydantic v2 data models for the fraud detection pipeline.

All public model classes are re-exported here so that the rest of the
codebase can import from a single location:

    from src.models import Transaction, EnrichedTransaction, FraudDecision, ...
"""

from src.models.transaction import EnrichedTransaction, Transaction
from src.models.decision import FraudDecision
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
from src.models.aml import AMLRiskScore, AMLSignal, InvestigationCase, ScoreUpdate
from src.models.pattern import FraudPattern

__all__ = [
    # transaction.py
    "Transaction",
    "EnrichedTransaction",
    # decision.py
    "FraudDecision",
    # persona.py
    "GeoEntry",
    "CategoryAnchor",
    "VelocityProfile",
    "AnomalyHistory",
    "IPEntry",
    "TemporalProfile",
    "DepositPattern",
    "TransferPattern",
    "AMLProfile",
    "BehavioralPersona",
    # aml.py
    "ScoreUpdate",
    "AMLRiskScore",
    "InvestigationCase",
    "AMLSignal",
    # pattern.py
    "FraudPattern",
]
