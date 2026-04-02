"""
ArchaeologistService — Tier 3 behavioral persona synthesis agent.

The Archaeologist compresses 24 months of raw transaction history into a
structured BehavioralPersona per user.  It runs:
  - On-demand when a user is first seen (cold-start persona creation).
  - On a weekly EventBridge batch for all active users.

Responsibilities:
  1. Fetch the user's previous persona from DynamoDB (if it exists).
  2. Format and send the archaeologist.txt prompt to Sonnet with the full
     transaction history and previous persona for incremental refinement.
  3. Parse the JSON response into a BehavioralPersona model.
  4. Persist the new persona to DynamoDB with an incremented version.
  5. Refresh the Redis persona cache so the Sentinel sees the updated profile
     on the next transaction.

Version scheme: SK=VERSION#<N> where N is zero-padded to 6 digits (e.g.
VERSION#000002).  get_persona() fetches in descending SK order so it always
returns the latest version without a GSI.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Callable

from src.clients.bedrock_client import BedrockClient
from src.clients.dynamodb_client import DynamoDBClient
from src.clients.redis_client import RedisClient
from src.config import Config
from src.models.persona import BehavioralPersona
from src.utils.metrics import FraudMetrics
from src.utils.sanitizer import sanitize_llm_input

logger = logging.getLogger(__name__)

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")

# Zero-padded width for the version counter in SK values.
_VERSION_PAD = 6


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


def _parse_version(sk: str) -> int:
    """Extract the integer version number from a VERSION#<N> SK value.

    Returns 0 if the SK format is unrecognised.
    """
    try:
        return int(sk.split("#", 1)[1])
    except (IndexError, ValueError):
        return 0


class ArchaeologistService:
    """Builds and incrementally refreshes behavioral personas from transaction history."""

    def __init__(
        self,
        config: Config,
        dynamodb_client: DynamoDBClient,
        bedrock_client: BedrockClient,
        redis_client: RedisClient,
        metrics: FraudMetrics,
    ) -> None:
        self._config = config
        self._dynamo = dynamodb_client
        self._bedrock = bedrock_client
        self._redis = redis_client
        self._metrics = metrics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_persona(
        self,
        user_id: str,
        transaction_history: list[dict],
    ) -> dict:
        """Synthesise a BehavioralPersona from *transaction_history*.

        Args:
            user_id:             The user whose persona is being built.
            transaction_history: Chronologically ordered list of transaction dicts
                                 covering up to 24 months of activity.

        Returns:
            The new persona as a plain dict (model.model_dump() output), suitable
            for JSON serialisation or further processing.

        Raises:
            Exception: Propagated from Bedrock or DynamoDB on unrecoverable failure.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        # Step 1 — fetch previous persona (if any) for incremental refinement.
        previous_item = self._dynamo.get_persona(
            table_name=self._config.dynamo_table_personas,
            user_id=user_id,
        )
        previous_persona_json = (
            json.dumps(previous_item, default=str) if previous_item else "null"
        )

        # Determine the new version number.
        if previous_item:
            sk = previous_item.get("SK", "VERSION#000000")
            previous_version = _parse_version(sk)
        else:
            previous_version = 0
        new_version = previous_version + 1
        new_sk = f"VERSION#{new_version:0{_VERSION_PAD}d}"

        logger.info(
            "Building persona",
            extra={
                "user_id": user_id,
                "history_size": len(transaction_history),
                "new_version": new_version,
                "has_previous": previous_item is not None,
            },
        )

        # Step 2 — format and send the Archaeologist prompt.
        prompt = self._load_prompt(
            "archaeologist",
            user_id=user_id,
            transaction_history_json=json.dumps(transaction_history, default=str),
            previous_persona_json=previous_persona_json,
        )

        # Step 3 — invoke Sonnet with a large token budget for chain-of-thought.
        sonnet_response = self._bedrock.invoke_sonnet(prompt, max_tokens=4096)
        tokens_used: int = sonnet_response.get("tokens_used", 0)
        self._metrics.record_token_spend(tokens_used)

        # Step 4 — parse the JSON persona.
        persona_dict = self._parse_persona_response(
            sonnet_response.get("text", ""),
            user_id=user_id,
        )

        # Validate structure via Pydantic before writing to DynamoDB.
        # This catches schema drift early rather than at Sentinel read time.
        try:
            persona_model = BehavioralPersona(**persona_dict)
        except Exception as exc:
            logger.error(
                "BehavioralPersona validation failed — aborting persona write",
                extra={
                    "errorType": type(exc).__name__,
                    "user_id": user_id,
                    "error_detail": str(exc),
                },
            )
            raise

        # Step 5 — persist to DynamoDB.
        item = persona_model.model_dump()
        item["PK"] = f"USER#{user_id}"
        item["SK"] = new_sk
        item["version"] = new_version
        item["built_at"] = now_iso

        self._dynamo.put_persona(
            table_name=self._config.dynamo_table_personas,
            persona=item,
        )

        # Step 6 — refresh Redis cache.
        try:
            self._redis.set_persona_cache(
                user_id=user_id,
                persona=item,
                ttl=self._config.persona_cache_ttl,
            )
        except Exception:
            # Cache write failure must not abort a successful DynamoDB write.
            logger.warning(
                "Failed to update persona Redis cache after build — Sentinel will "
                "fall back to DynamoDB on next transaction",
                extra={"user_id": user_id},
                exc_info=True,
            )

        logger.info(
            "Persona built and stored",
            extra={
                "user_id": user_id,
                "version": new_version,
                "tokens_used": tokens_used,
            },
        )
        return item

    def run_weekly_batch(
        self,
        user_ids: list[str],
        fetch_history_fn: Callable[[str], list[dict]],
    ) -> dict:
        """Rebuild personas for a list of users (weekly batch run).

        Args:
            user_ids:         List of user IDs to process.
            fetch_history_fn: Callable that accepts a user_id string and returns
                              up to 24 months of that user's transaction history
                              as a list of dicts. In production this queries Athena;
                              in tests it can be a simple mock.

        Returns:
            Summary dict:
            {
                "total":     int,
                "succeeded": int,
                "failed":    int,
                "errors":    [ {"user_id": str, "error": str}, ... ]
            }
        """
        total = len(user_ids)
        succeeded = 0
        failed = 0
        errors: list[dict] = []

        logger.info(
            "Archaeologist weekly batch starting",
            extra={"total_users": total},
        )

        for user_id in user_ids:
            try:
                history = fetch_history_fn(user_id)
                self.build_persona(user_id, history)
                succeeded += 1
            except Exception as exc:
                failed += 1
                errors.append({"user_id": user_id, "error": str(exc)})
                logger.error(
                    "Weekly batch persona build failed for user",
                    extra={
                        "errorType": type(exc).__name__,
                        "user_id": user_id,
                        "error_detail": str(exc),
                    },
                    exc_info=True,
                )

        logger.info(
            "Archaeologist weekly batch complete",
            extra={
                "total": total,
                "succeeded": succeeded,
                "failed": failed,
            },
        )
        return {
            "total": total,
            "succeeded": succeeded,
            "failed": failed,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_persona_response(self, text: str, user_id: str) -> dict:
        """Extract and parse the JSON persona from a Sonnet response.

        The Archaeologist prompt instructs the model to emit chain-of-thought
        reasoning followed by the final JSON object, so we must find the last
        (outermost) JSON object in the response rather than the first.

        Raises:
            ValueError: If parsing fails — callers should handle and abort the write.
        """
        try:
            json_str = _extract_json(text)
            return json.loads(json_str)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error(
                "Failed to parse Archaeologist LLM response",
                extra={
                    "errorType": type(exc).__name__,
                    "user_id": user_id,
                    "error_detail": str(exc),
                    "raw_text": text[:500],
                },
            )
            raise ValueError(
                f"Archaeologist response parse error for user {user_id}: {exc}"
            ) from exc

    def _load_prompt(self, template_name: str, **kwargs: object) -> str:
        """Load and format a prompt template from the prompts/ directory."""
        path = os.path.join(_PROMPTS_DIR, f"{template_name}.txt")
        with open(path, encoding="utf-8") as fh:
            template = fh.read()
        return template.format(**kwargs)
