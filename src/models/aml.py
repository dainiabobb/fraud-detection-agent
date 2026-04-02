"""
AML (Anti-Money Laundering) data models.

AMLRiskScore  — continuous accumulator stored in AMLRiskScores DynamoDB table.
InvestigationCase — permanent compliance record in InvestigationCases table.
AMLSignal     — ephemeral signal produced by the Sentinel or AML Specialist
                and passed through the pipeline; not persisted directly.
"""

from __future__ import annotations

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# AML Risk Score
# ---------------------------------------------------------------------------

class ScoreUpdate(BaseModel):
    """One entry in the score_history ledger."""

    # ISO 8601 UTC timestamp of the update
    timestamp: str
    # Signed delta applied to current_score (positive = risk increase)
    delta: float
    # Human-readable explanation produced by the AML Specialist
    reason: str
    # Transaction that triggered this update
    transaction_id: str


class AMLRiskScore(BaseModel):
    """
    Continuous AML risk accumulator for one user.

    Stored in the AMLRiskScores table (PK=userId).
    Thresholds (from README):
      < 30  → Normal
      30-50 → Elevated
      50-80 → Investigation case opened automatically
      > 80  → Auto-escalated to compliance
    """

    user_id: str
    # Current cumulative risk score (0.0 – 100.0)
    current_score: float
    # ISO 8601 UTC timestamp of the most recent score update
    last_updated: str
    # Full audit trail of every delta applied to this score
    score_history: list[ScoreUpdate]
    # Set when an InvestigationCase has been opened for this user
    investigation_case_id: str | None = None


# ---------------------------------------------------------------------------
# Investigation Case
# ---------------------------------------------------------------------------

class InvestigationCase(BaseModel):
    """
    Permanent compliance record opened when a user's AML risk score
    crosses the investigation threshold (50+).

    Stored in InvestigationCases (PK=caseId, SK=userId).
    Lifecycle: OPEN → ESCALATED → CLOSED (managed by the compliance team).
    Records are never deleted — regulatory retention requirement.
    """

    case_id: str
    user_id: str
    # OPEN | ESCALATED | CLOSED
    status: str
    # ISO 8601 UTC
    opened_at: str
    # Populated when status transitions to ESCALATED
    escalated_at: str | None = None
    # Populated when status transitions to CLOSED
    closed_at: str | None = None
    risk_score_at_open: float
    # AML typology labels that contributed to opening this case
    typologies_detected: list[str]
    # transaction_id values associated with this investigation
    transactions: list[str]
    # Free-text audit notes appended by compliance analysts or the AML Specialist
    notes: list[str]


# ---------------------------------------------------------------------------
# AML Signal (ephemeral, pipeline-internal)
# ---------------------------------------------------------------------------

class AMLSignal(BaseModel):
    """
    Structural AML signal detected by the Sentinel or AML Specialist.

    This model is NOT persisted to DynamoDB directly; it travels as part
    of the escalation payload to the Swarm Orchestrator and into the
    AML Specialist's context window.

    signal_type values: STRUCTURING | SMURFING | LAYERING |
                        ROUND_TRIPPING | U_TURN | PROFILE_MISMATCH
    """

    signal_type: str
    # 0.0 – 1.0 posterior confidence for this signal
    confidence: float
    # Arbitrary key/value evidence bag (amounts, counts, jurisdiction codes …)
    details: dict
