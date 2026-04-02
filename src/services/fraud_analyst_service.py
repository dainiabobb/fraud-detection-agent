"""
FraudAnalystService — Tier 2 fraud adjudication agent.

Receives transactions escalated by the Sentinel (FRAUD_ONLY or BOTH routing)
and applies Claude Sonnet to produce a final BLOCK or APPROVE verdict.

Responsibilities:
  1. Format the fraud_analyst.txt prompt with transaction, persona, similar
     transactions, and the Sentinel's escalation context.
  2. Invoke Sonnet via BedrockClient.
  3. Parse the JSON response and build a FraudDecision.
  4. Persist the decision to DynamoDB (FraudDecisions table).
  5. Publish CloudWatch metrics (block / approve, token spend).
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

from src.clients.bedrock_client import BedrockClient
from src.clients.dynamodb_client import DynamoDBClient
from src.clients.opensearch_client import OpenSearchClient
from src.config import Config
from src.models.decision import FraudDecision
from src.utils.metrics import FraudMetrics
from src.utils.sanitizer import sanitize_llm_input

logger = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")


def _extract_json(text: str) -> str:
    """Extract the first JSON object or array from *text*.

    Finds the outermost { ... } or [ ... ] and returns that substring.
    Raises ValueError if neither delimiter is found.
    """
    obj_start = text.find("{")
    arr_start = text.find("[")

    if obj_start == -1 and arr_start == -1:
        raise ValueError(f"No JSON object or array found in LLM response: {text!r}")

    if obj_start == -1:
        start, end_char = arr_start, "]"
    elif arr_start == -1:
        start, end_char = obj_start, "}"
    else:
        start = min(obj_start, arr_start)
        end_char = "}" if start == obj_start else "]"

    end = text.rfind(end_char)
    if end == -1 or end < start:
        raise ValueError(
            f"No closing '{end_char}' found in LLM response: {text!r}"
        )
    return text[start : end + 1]


class FraudAnalystService:
    """Tier 2 fraud adjudication: deep analysis via Claude Sonnet."""

    def __init__(
        self,
        config: Config,
        dynamodb_client: DynamoDBClient,
        opensearch_client: OpenSearchClient,
        bedrock_client: BedrockClient,
        metrics: FraudMetrics,
    ) -> None:
        self._config = config
        self._dynamo = dynamodb_client
        self._opensearch = opensearch_client
        self._bedrock = bedrock_client
        self._metrics = metrics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        transaction_dict: dict,
        persona_dict: dict | None,
        similar_transactions: list[dict],
        escalation_context: str,
    ) -> FraudDecision:
        """Perform deep fraud analysis and return a final verdict.

        Args:
            transaction_dict:     Enriched transaction data (JSON-serialisable dict).
            persona_dict:         Behavioral persona, or None if unavailable.
            similar_transactions: kNN results from the Sentinel's vector search.
            escalation_context:   Sentinel's reasoning string explaining why this
                                  transaction was escalated.

        Returns:
            FraudDecision with verdict BLOCK or APPROVE, tier=FRAUD_ANALYST.

        Raises:
            Exception: Propagated from Bedrock or DynamoDB on unrecoverable failure.
        """
        start_ms = int(time.monotonic() * 1000)
        now_iso = datetime.now(timezone.utc).isoformat()
        ttl_epoch = int(time.time()) + 90 * 86_400  # 90-day retention

        transaction_id: str = transaction_dict.get("transaction_id", "unknown")
        user_id: str = transaction_dict.get("user_id", "unknown")

        # Build and format the Fraud Analyst prompt.
        prompt = self._load_prompt(
            "fraud_analyst",
            transaction_json=json.dumps(transaction_dict, default=str),
            persona_json=json.dumps(persona_dict, default=str) if persona_dict else "null",
            similar_transactions=json.dumps(similar_transactions, default=str),
            escalation_context=sanitize_llm_input(escalation_context),
        )

        # Invoke Sonnet — higher token budget for chain-of-thought reasoning.
        sonnet_response = self._bedrock.invoke_sonnet(prompt, max_tokens=2048)
        tokens_used: int = sonnet_response.get("tokens_used", 0)
        self._metrics.record_token_spend(tokens_used)

        # Parse the JSON verdict from the model response.
        parsed = self._parse_llm_response(
            sonnet_response.get("text", ""),
            context={"transaction_id": transaction_id, "user_id": user_id},
        )

        verdict: str = parsed.get("verdict", "BLOCK")
        confidence: float = float(parsed.get("confidence", 0.5))
        reasoning: str = sanitize_llm_input(parsed.get("reasoning", ""))

        # Normalise verdict: only BLOCK or APPROVE are valid at Tier 2.
        if verdict not in {"BLOCK", "APPROVE"}:
            logger.warning(
                "FraudAnalyst returned unexpected verdict — defaulting to BLOCK",
                extra={
                    "transaction_id": transaction_id,
                    "user_id": user_id,
                    "raw_verdict": verdict,
                },
            )
            verdict = "BLOCK"

        latency_ms = int(time.monotonic() * 1000) - start_ms

        decision = FraudDecision(
            transaction_id=transaction_id,
            user_id=user_id,
            verdict=verdict,
            tier="FRAUD_ANALYST",
            confidence=confidence,
            reasoning=reasoning,
            escalation_target=None,  # Tier 2 always issues a terminal verdict
            pattern_matches=[],
            timestamp=now_iso,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            ttl=ttl_epoch,
        )

        # Persist decision to DynamoDB.
        self._store_decision(decision)

        # Publish outcome metrics.
        if verdict == "BLOCK":
            self._metrics.record_block()
        else:
            self._metrics.record_auto_approve()

        logger.info(
            "FraudAnalyst decision recorded",
            extra={
                "transaction_id": transaction_id,
                "user_id": user_id,
                "verdict": verdict,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "tokens_used": tokens_used,
            },
        )
        return decision

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_prompt(self, template_name: str, **kwargs: object) -> str:
        """Load and format a prompt template from the prompts/ directory.

        Args:
            template_name: File name without the .txt extension.
            **kwargs:       Placeholder values to substitute.

        Returns:
            Formatted prompt string.

        Raises:
            FileNotFoundError: If the template file is missing.
        """
        path = os.path.join(_PROMPTS_DIR, f"{template_name}.txt")
        with open(path, encoding="utf-8") as fh:
            template = fh.read()
        return template.format(**kwargs)

    def _parse_llm_response(self, text: str, context: dict) -> dict:
        """Extract and parse the JSON payload from a Sonnet response.

        Falls back to a BLOCK verdict on parse error so that a malformed model
        response never silently approves a potentially fraudulent transaction.
        """
        try:
            json_str = _extract_json(text)
            return json.loads(json_str)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error(
                "Failed to parse FraudAnalyst LLM response — defaulting to BLOCK",
                extra={
                    **context,
                    "errorType": type(exc).__name__,
                    "error_detail": str(exc),
                    "raw_text": text[:500],
                },
            )
            # Fail closed: an unparseable response is treated as a BLOCK.
            return {
                "verdict": "BLOCK",
                "confidence": 0.5,
                "reasoning": "Parse error in FraudAnalyst response; blocked for safety.",
            }

    def _store_decision(self, decision: FraudDecision) -> None:
        """Write the FraudDecision to the DynamoDB decisions table.

        The item uses PK=DECISION#<transaction_id> so point-gets are O(1).
        The TTL attribute enables automatic expiry after 90 days.
        """
        item = decision.model_dump()
        # Map to DynamoDB key schema used by the verdict-timestamp GSI.
        item["PK"] = f"DECISION#{decision.transaction_id}"
        try:
            self._dynamo.put_decision(
                table_name=self._config.dynamo_table_decisions,
                decision=item,
            )
        except Exception:
            # Log but do not re-raise — a write failure must not silently
            # discard the in-memory decision already returned to the caller.
            logger.error(
                "Failed to persist FraudDecision to DynamoDB",
                extra={
                    "errorType": "DynamoDBWriteFailure",
                    "transaction_id": decision.transaction_id,
                    "user_id": decision.user_id,
                    "verdict": decision.verdict,
                },
                exc_info=True,
            )
