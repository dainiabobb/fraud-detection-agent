"""
FraudDecision data model.

Written to DynamoDB FraudDecisions table (PK=transactionId, SK=timestamp).
The optional `ttl` field holds an epoch-second timestamp used by DynamoDB's
TTL mechanism to auto-expire records after 90 days.
"""

from __future__ import annotations

from pydantic import BaseModel


class FraudDecision(BaseModel):
    """
    Final adjudication record produced by any tier of the pipeline.

    Verdict values : APPROVE | BLOCK | ESCALATE
    Tier values    : SENTINEL | FRAUD_ANALYST | AML_SPECIALIST
    escalation_target (only set when verdict == ESCALATE):
                     FRAUD_ONLY | AML_ONLY | BOTH
    """

    transaction_id: str
    user_id: str
    # APPROVE | BLOCK | ESCALATE
    verdict: str
    # Which tier emitted this decision
    tier: str
    # 0.0 – 1.0 posterior confidence from the deciding model
    confidence: float
    # Free-text chain-of-thought produced by the LLM
    reasoning: str
    # Populated only when verdict == ESCALATE
    escalation_target: str | None = None
    # Names of PatternDiscovery patterns that fired on this transaction
    pattern_matches: list[str] = []
    # ISO 8601 UTC timestamp of when the decision was recorded
    timestamp: str
    # Bedrock token consumption for this decision (input + output combined)
    tokens_used: int = 0
    # Wall-clock milliseconds from transaction receipt to decision write
    latency_ms: int = 0
    # Epoch seconds; DynamoDB TTL attribute — None means no expiry
    ttl: int | None = None
