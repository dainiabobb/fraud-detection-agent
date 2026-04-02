"""
AMLSpecialistService — Tier 2 Anti-Money Laundering analysis agent.

The AML Specialist does NOT block transactions ("tipping off" doctrine).
Its role is to:
  1. Assess structural AML typologies against the user's transaction history.
  2. Update the continuous AML risk score in DynamoDB (clamped to [0, 100]).
  3. Open an InvestigationCase when the score crosses the investigation threshold.
  4. Escalate an existing case to compliance when the compliance threshold is crossed.
  5. Publish CloudWatch metrics for AML risk updates and case lifecycle events.

The service is invoked asynchronously (fire-and-forget) by the Swarm Orchestrator
when escalation_target is AML_ONLY or BOTH, so it never blocks a payment flow.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

from src.clients.bedrock_client import BedrockClient
from src.clients.dynamodb_client import DynamoDBClient
from src.config import Config
from src.models.aml import AMLRiskScore, InvestigationCase, ScoreUpdate
from src.utils.metrics import FraudMetrics
from src.utils.sanitizer import sanitize_llm_input

logger = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")


def _extract_json(text: str) -> str:
    """Extract the first JSON object or array from *text*.

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


def _generate_case_id() -> str:
    """Generate a time-ordered investigation case ID using a ULID-style prefix."""
    # Use current epoch millis as a sortable prefix to avoid ulid dependency here;
    # the actual ulid library is used for log IDs per project conventions.
    import uuid
    return f"CASE-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8].upper()}"


class AMLSpecialistService:
    """Tier 2 AML analysis: typology detection, score update, case management."""

    def __init__(
        self,
        config: Config,
        dynamodb_client: DynamoDBClient,
        bedrock_client: BedrockClient,
        metrics: FraudMetrics,
    ) -> None:
        self._config = config
        self._dynamo = dynamodb_client
        self._bedrock = bedrock_client
        self._metrics = metrics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        transaction_dict: dict,
        persona_dict: dict | None,
        user_history: list[dict],
        current_score: float,
        existing_case: dict | None,
    ) -> dict:
        """Run full AML analysis for one transaction.

        Args:
            transaction_dict: Enriched transaction data (JSON-serialisable dict).
            persona_dict:     Behavioral persona (includes aml_profile), or None.
            user_history:     Last 6 months of user transactions for context.
            current_score:    Current AML risk score from DynamoDB (0–100).
            existing_case:    Open investigation case dict, or None.

        Returns:
            Result dict:
            {
                "score_delta":          int,
                "new_score":            float,
                "typologies_detected":  list[str],
                "reasoning":            str,
                "open_investigation":   bool,
                "escalate_to_compliance": bool,
                "case_id":              str | None,
                "case_status":          str | None,
            }
        """
        transaction_id: str = transaction_dict.get("transaction_id", "unknown")
        user_id: str = transaction_dict.get("user_id", "unknown")
        now_iso = datetime.now(timezone.utc).isoformat()

        # Build and format the AML Specialist prompt.
        prompt = self._load_prompt(
            "aml_specialist",
            transaction_json=json.dumps(transaction_dict, default=str),
            persona_json=json.dumps(persona_dict, default=str) if persona_dict else "null",
            user_history_6mo=json.dumps(user_history, default=str),
            current_aml_score=str(current_score),
            existing_case=json.dumps(existing_case, default=str) if existing_case else "null",
        )

        # Invoke Sonnet — AML reasoning benefits from the larger context window.
        sonnet_response = self._bedrock.invoke_sonnet(prompt, max_tokens=4096)
        tokens_used: int = sonnet_response.get("tokens_used", 0)
        self._metrics.record_token_spend(tokens_used)

        # Parse the JSON response.
        parsed = self._parse_llm_response(
            sonnet_response.get("text", ""),
            context={"transaction_id": transaction_id, "user_id": user_id},
        )

        score_delta: int = int(parsed.get("score_delta", 0))
        typologies_detected: list[str] = parsed.get("typologies_detected", [])
        reasoning: str = sanitize_llm_input(parsed.get("reasoning", ""))
        open_investigation: bool = bool(parsed.get("open_investigation", False))
        escalate_to_compliance: bool = bool(parsed.get("escalate_to_compliance", False))

        # Compute new score, clamped to [0, 100].
        new_score = max(0.0, min(100.0, current_score + score_delta))

        # Update AML risk score in DynamoDB.
        score_update = ScoreUpdate(
            timestamp=now_iso,
            delta=float(score_delta),
            reason=reasoning,
            transaction_id=transaction_id,
        )
        self._update_aml_score(
            user_id=user_id,
            new_score=new_score,
            score_update=score_update,
            current_score=current_score,
            existing_case_id=(
                existing_case.get("case_id") if existing_case else None
            ),
            now_iso=now_iso,
        )
        self._metrics.record_aml_risk_update()

        # Case management.
        case_id: str | None = None
        case_status: str | None = None

        if open_investigation and existing_case is None:
            # Open a new investigation case.
            case_id = _generate_case_id()
            new_case = InvestigationCase(
                case_id=case_id,
                user_id=user_id,
                status="OPEN",
                opened_at=now_iso,
                escalated_at=None,
                closed_at=None,
                risk_score_at_open=new_score,
                typologies_detected=typologies_detected,
                transactions=[transaction_id],
                notes=[reasoning],
            )
            case_item = new_case.model_dump()
            case_item["PK"] = f"CASE#{case_id}"
            self._dynamo.put_investigation_case(
                table_name=self._config.dynamo_table_investigations,
                case=case_item,
            )
            case_status = "OPEN"
            self._metrics.record_investigation_opened()
            logger.info(
                "Investigation case opened",
                extra={
                    "user_id": user_id,
                    "transaction_id": transaction_id,
                    "case_id": case_id,
                    "new_score": new_score,
                },
            )

        elif escalate_to_compliance and existing_case is not None:
            # Escalate the existing open case to compliance.
            case_id = existing_case.get("case_id")
            escalated_case = {
                **existing_case,
                "status": "ESCALATED",
                "escalated_at": now_iso,
                "notes": existing_case.get("notes", []) + [
                    f"Escalated to compliance at {now_iso}. Score: {new_score}. "
                    f"Typologies: {typologies_detected}. Reason: {reasoning}"
                ],
            }
            self._dynamo.put_investigation_case(
                table_name=self._config.dynamo_table_investigations,
                case=escalated_case,
            )
            case_status = "ESCALATED"
            self._metrics.record_investigation_escalated()
            logger.info(
                "Investigation case escalated to compliance",
                extra={
                    "user_id": user_id,
                    "transaction_id": transaction_id,
                    "case_id": case_id,
                    "new_score": new_score,
                },
            )

        elif existing_case is not None:
            # Existing case present — append the transaction and new reasoning.
            case_id = existing_case.get("case_id")
            case_status = existing_case.get("status")
            updated_transactions = list(existing_case.get("transactions", []))
            if transaction_id not in updated_transactions:
                updated_transactions.append(transaction_id)
            updated_notes = list(existing_case.get("notes", []))
            updated_notes.append(
                f"AML analysis at {now_iso}: score delta {score_delta:+d}, "
                f"typologies={typologies_detected}. {reasoning}"
            )
            updated_case = {
                **existing_case,
                "transactions": updated_transactions,
                "notes": updated_notes,
            }
            self._dynamo.put_investigation_case(
                table_name=self._config.dynamo_table_investigations,
                case=updated_case,
            )

        logger.info(
            "AML analysis complete",
            extra={
                "user_id": user_id,
                "transaction_id": transaction_id,
                "score_delta": score_delta,
                "new_score": new_score,
                "typologies": typologies_detected,
                "tokens_used": tokens_used,
            },
        )

        return {
            "score_delta": score_delta,
            "new_score": new_score,
            "typologies_detected": typologies_detected,
            "reasoning": reasoning,
            "open_investigation": open_investigation,
            "escalate_to_compliance": escalate_to_compliance,
            "case_id": case_id,
            "case_status": case_status,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_aml_score(
        self,
        user_id: str,
        new_score: float,
        score_update: ScoreUpdate,
        current_score: float,
        existing_case_id: str | None,
        now_iso: str,
    ) -> None:
        """Fetch, update, and persist the AML risk score record for *user_id*."""
        existing = self._dynamo.get_aml_risk_score(
            table_name=self._config.dynamo_table_aml_risk,
            user_id=user_id,
        )

        if existing is not None:
            history = existing.get("score_history", [])
            history.append(score_update.model_dump())
            updated_score = AMLRiskScore(
                user_id=user_id,
                current_score=new_score,
                last_updated=now_iso,
                score_history=history,
                investigation_case_id=existing_case_id or existing.get("investigation_case_id"),
            )
        else:
            # First AML record for this user.
            updated_score = AMLRiskScore(
                user_id=user_id,
                current_score=new_score,
                last_updated=now_iso,
                score_history=[score_update.model_dump()],
                investigation_case_id=existing_case_id,
            )

        score_item = updated_score.model_dump()
        score_item["PK"] = f"AML#{user_id}"
        self._dynamo.put_aml_risk_score(
            table_name=self._config.dynamo_table_aml_risk,
            score=score_item,
        )

    def _load_prompt(self, template_name: str, **kwargs: object) -> str:
        """Load and format a prompt template from the prompts/ directory."""
        path = os.path.join(_PROMPTS_DIR, f"{template_name}.txt")
        with open(path, encoding="utf-8") as fh:
            template = fh.read()
        return template.format(**kwargs)

    def _parse_llm_response(self, text: str, context: dict) -> dict:
        """Parse the JSON payload from a Sonnet response.

        Falls back to a zero-delta safe response on parse failure so a
        malformed model response does not crash the AML pipeline.
        """
        try:
            json_str = _extract_json(text)
            return json.loads(json_str)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error(
                "Failed to parse AMLSpecialist LLM response — defaulting to zero delta",
                extra={
                    **context,
                    "errorType": type(exc).__name__,
                    "error_detail": str(exc),
                    "raw_text": text[:500],
                },
            )
            return {
                "score_delta": 0,
                "typologies_detected": [],
                "reasoning": "Parse error in AML Specialist response; score unchanged.",
                "open_investigation": False,
                "escalate_to_compliance": False,
            }
