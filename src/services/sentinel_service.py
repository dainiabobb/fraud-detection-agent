"""
SentinelService — Tier 1 routing agent for the fraud detection pipeline.

Responsibilities:
  1. Sanitize and enrich inbound transaction data (GeoIP, temporal).
  2. Fetch the user's behavioral persona from Redis (with DynamoDB fallback).
  3. Embed the transaction and perform kNN search against historical vectors.
  4. Detect AML structural signals (structuring, round numbers, high-risk
     jurisdictions, economic profile mismatch).
  5. Apply deterministic routing rules; only call Haiku in the gray zone.
  6. Return a FraudDecision (APPROVE / ESCALATE) and publish CloudWatch metrics.

The Sentinel never issues a BLOCK verdict — that is the domain of Tier 2
(FraudAnalystService). Escalations are routed to FRAUD_ONLY, AML_ONLY, or BOTH
depending on which signals fired.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

from src.clients.bedrock_client import BedrockClient
from src.clients.dynamodb_client import DynamoDBClient
from src.clients.opensearch_client import OpenSearchClient
from src.clients.redis_client import RedisClient
from src.config import Config
from src.models.aml import AMLSignal
from src.models.decision import FraudDecision
from src.models.transaction import EnrichedTransaction, Transaction
from src.utils.embedding import build_transaction_text, cosine_similarity
from src.utils.geoip import calculate_local_hour, enrich_geoip
from src.utils.metrics import FraudMetrics
from src.utils.sanitizer import sanitize_llm_input, sanitize_transaction

logger = logging.getLogger(__name__)

# High-risk jurisdictions per OFAC/FATF consolidated list.
_HIGH_RISK_JURISDICTIONS: frozenset[str] = frozenset(
    ["IR", "KP", "SY", "CU", "VE", "MM", "BY", "RU", "AF"]
)

# Amount window that characterises CTR-threshold structuring attempts.
_STRUCTURING_LOW: float = 8_000.0
_STRUCTURING_HIGH: float = 9_999.99

# Round-number transfer threshold (AML signal).
_ROUND_NUMBER_MIN: float = 5_000.0

# Prompts directory relative to project root, resolved at module load time.
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

    # Pick whichever opening delimiter comes first (or the only one present).
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


class SentinelService:
    """Tier 1 routing agent: enriches, embeds, and routes incoming transactions."""

    def __init__(
        self,
        config: Config,
        dynamodb_client: DynamoDBClient,
        opensearch_client: OpenSearchClient,
        redis_client: RedisClient,
        bedrock_client: BedrockClient,
        metrics: FraudMetrics,
    ) -> None:
        self._config = config
        self._dynamo = dynamodb_client
        self._opensearch = opensearch_client
        self._redis = redis_client
        self._bedrock = bedrock_client
        self._metrics = metrics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_transaction(self, raw_transaction: dict) -> FraudDecision:
        """Run the full Sentinel pipeline for one transaction.

        Steps:
            1.  Sanitize raw input.
            2.  Construct Transaction model.
            3.  GeoIP enrichment → EnrichedTransaction.
            4.  Fetch/cache behavioral persona.
            5.  Embed transaction text.
            6.  kNN similarity search (k=5).
            7.  Fetch/cache active fraud patterns.
            8.  Detect AML signals.
            9.  Apply deterministic routing; call Haiku only in gray zone.
            10. Build and return FraudDecision.

        Args:
            raw_transaction: Untrusted inbound transaction payload (dict).

        Returns:
            FraudDecision with verdict APPROVE or ESCALATE.

        Raises:
            ValueError: If required transaction fields are missing or invalid.
            Exception:  Propagated from downstream clients (DynamoDB, Bedrock…).
        """
        start_ms = int(time.monotonic() * 1000)

        # Step 1 — sanitize
        cleaned = sanitize_transaction(raw_transaction)

        # Step 2 — build Transaction model
        txn = Transaction(**cleaned)

        # Step 3 — GeoIP enrichment
        geo_data = enrich_geoip(txn.ip_address)
        local_hour: int | None = None
        geo_tz: str | None = None

        # Attempt to derive a timezone from the persona if geo lookup is sparse.
        # We compute local_hour after persona fetch below if timezone is needed.
        enriched_data = {**cleaned, **geo_data}
        enriched = EnrichedTransaction(**enriched_data)

        # Step 4 — fetch persona (Redis → DynamoDB fallback)
        persona = self._fetch_persona(txn.user_id)

        # Derive local_hour from persona timezone estimate if GeoIP didn't give us
        # longitude/latitude for an independent timezone lookup.
        if enriched.local_hour is None and persona:
            tz_estimate = (
                persona.get("temporal_profile", {}).get("timezone_estimate")
            )
            local_hour = calculate_local_hour(txn.timestamp, tz_estimate)
            # Re-create EnrichedTransaction with the derived local_hour.
            enriched = EnrichedTransaction(**{**enriched_data, "local_hour": local_hour})

        # Step 5 — embed transaction text
        txn_text = build_transaction_text(enriched.model_dump())
        embedding: list[float] = self._bedrock.get_embedding(txn_text)

        # Step 6 — kNN search for similar transactions
        similar_results: list[dict] = self._opensearch.knn_search(
            index_name=self._config.opensearch_index,
            vector=embedding,
            k=5,
        )

        # Step 7 — fetch active patterns (Redis → DynamoDB fallback)
        active_patterns = self._fetch_patterns()

        # Step 8 — detect AML signals
        aml_signals: list[AMLSignal] = self._detect_aml_signals(enriched, persona)

        # Step 9 — routing logic
        decision = self._route(
            txn=enriched,
            persona=persona,
            similar_results=similar_results,
            active_patterns=active_patterns,
            aml_signals=aml_signals,
            start_ms=start_ms,
        )
        return decision

    # ------------------------------------------------------------------
    # AML signal detection
    # ------------------------------------------------------------------

    def _detect_aml_signals(
        self,
        transaction: EnrichedTransaction,
        persona: dict | None,
    ) -> list[AMLSignal]:
        """Identify structural AML signals on the current transaction.

        Checks four independent conditions:
          * Structuring   — amount in [$8k, $9,999.99]
          * Round number  — amount % 1000 == 0 and amount >= $5k
          * High-risk jurisdiction — merchant_country in OFAC list
          * Economic profile mismatch — persona flag indicates mismatch

        Returns:
            List of AMLSignal instances (may be empty).
        """
        signals: list[AMLSignal] = []
        amount = transaction.amount

        # Structuring check
        if _STRUCTURING_LOW <= amount <= _STRUCTURING_HIGH:
            signals.append(
                AMLSignal(
                    signal_type="STRUCTURING",
                    confidence=0.85,
                    details={"amount": amount, "threshold_low": _STRUCTURING_LOW},
                )
            )
            logger.info(
                "AML structuring signal detected",
                extra={
                    "user_id": transaction.user_id,
                    "transaction_id": transaction.transaction_id,
                    "amount": amount,
                },
            )

        # Round-number transfer check
        if amount >= _ROUND_NUMBER_MIN and amount % 1000 == 0:
            signals.append(
                AMLSignal(
                    signal_type="ROUND_TRIPPING",
                    confidence=0.65,
                    details={"amount": amount, "round_number_min": _ROUND_NUMBER_MIN},
                )
            )
            logger.info(
                "AML round-number transfer signal detected",
                extra={
                    "user_id": transaction.user_id,
                    "transaction_id": transaction.transaction_id,
                    "amount": amount,
                },
            )

        # High-risk jurisdiction check (merchant country)
        if transaction.merchant_country.upper() in _HIGH_RISK_JURISDICTIONS:
            signals.append(
                AMLSignal(
                    signal_type="U_TURN",
                    confidence=0.90,
                    details={"merchant_country": transaction.merchant_country},
                )
            )
            logger.info(
                "AML high-risk jurisdiction signal detected",
                extra={
                    "user_id": transaction.user_id,
                    "transaction_id": transaction.transaction_id,
                    "merchant_country": transaction.merchant_country,
                },
            )

        # Economic profile mismatch via persona
        if persona:
            aml_profile = persona.get("aml_profile", {})
            if aml_profile.get("economic_profile_match") is False:
                signals.append(
                    AMLSignal(
                        signal_type="PROFILE_MISMATCH",
                        confidence=0.75,
                        details={
                            "user_id": transaction.user_id,
                            "source": "persona.aml_profile.economic_profile_match",
                        },
                    )
                )
                logger.info(
                    "AML economic profile mismatch signal detected",
                    extra={
                        "user_id": transaction.user_id,
                        "transaction_id": transaction.transaction_id,
                    },
                )

        return signals

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route(
        self,
        txn: EnrichedTransaction,
        persona: dict | None,
        similar_results: list[dict],
        active_patterns: list[dict],
        aml_signals: list[AMLSignal],
        start_ms: int,
    ) -> FraudDecision:
        """Apply deterministic routing rules and return a FraudDecision.

        Routing precedence:
          1. AML signals set escalation_target (may override fraud path).
          2. If max cosine similarity >= auto_approve_threshold → APPROVE.
          3. If max cosine similarity < escalation_threshold → ESCALATE.
          4. Gray zone: check pattern matches; call Haiku if no clear answer.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        ttl_epoch = int(time.time()) + 90 * 86_400  # 90-day DynamoDB TTL

        # Derive max similarity from kNN results.
        max_similarity: float = 0.0
        if similar_results:
            scores = [r.get("score", 0.0) for r in similar_results]
            max_similarity = max(scores)

        # Determine AML escalation routing.
        has_aml = len(aml_signals) > 0
        aml_escalation_target: str | None = None
        if has_aml:
            # The final target may be upgraded to BOTH if fraud path also escalates.
            aml_escalation_target = "AML_ONLY"
            self._metrics.record_aml_escalation()

        # Matched pattern names (used if we reach gray zone or for enrichment).
        matched_patterns: list[str] = []

        # Persona deviation check — even if kNN similarity is high, flag
        # transactions that are anomalous *for this specific user*.
        persona_anomalies: list[str] = self._check_persona_deviation(txn, persona)
        has_persona_anomaly = len(persona_anomalies) > 0

        # Auto-approve path: high similarity to legitimate transactions.
        if max_similarity >= self._config.auto_approve_threshold:
            if has_aml:
                # Cannot auto-approve when AML signal is present.
                verdict = "ESCALATE"
                escalation_target = "AML_ONLY"
                confidence = 0.80
                reasoning = (
                    "High kNN similarity would approve but AML signal overrides; "
                    "routing to AML specialist."
                )
                self._metrics.record_escalation()
            elif has_persona_anomaly:
                # High similarity globally, but anomalous for this user.
                verdict = "ESCALATE"
                escalation_target = "FRAUD_ONLY"
                confidence = 0.70
                reasoning = (
                    f"Similarity {max_similarity:.3f} exceeds auto-approve threshold "
                    f"but persona deviation detected: {'; '.join(persona_anomalies)}. "
                    "Escalating for Tier 2 review."
                )
                self._metrics.record_escalation()
            else:
                verdict = "APPROVE"
                escalation_target = None
                confidence = min(1.0, max_similarity)
                reasoning = (
                    f"Similarity {max_similarity:.3f} exceeds auto-approve "
                    f"threshold {self._config.auto_approve_threshold}; "
                    "no AML signals or persona anomalies detected."
                )
                self._metrics.record_auto_approve()

            return self._build_decision(
                txn=txn,
                verdict=verdict,
                tier="SENTINEL",
                confidence=confidence,
                reasoning=reasoning,
                escalation_target=escalation_target,
                pattern_matches=matched_patterns,
                tokens_used=0,
                timestamp=now_iso,
                latency_ms=int(time.monotonic() * 1000) - start_ms,
                ttl=ttl_epoch,
            )

        # Definite escalation path: similarity too low to trust.
        if max_similarity < self._config.escalation_threshold:
            escalation_target = (
                "BOTH" if has_aml else "FRAUD_ONLY"
            )
            reasoning = (
                f"Similarity {max_similarity:.3f} below escalation threshold "
                f"{self._config.escalation_threshold}; routing to Tier 2."
            )
            self._metrics.record_escalation()
            return self._build_decision(
                txn=txn,
                verdict="ESCALATE",
                tier="SENTINEL",
                confidence=1.0 - max_similarity,
                reasoning=reasoning,
                escalation_target=escalation_target,
                pattern_matches=matched_patterns,
                tokens_used=0,
                timestamp=now_iso,
                latency_ms=int(time.monotonic() * 1000) - start_ms,
                ttl=ttl_epoch,
            )

        # Gray zone (escalation_threshold <= similarity < auto_approve_threshold).
        # Check pattern matches first — a pattern hit forces ESCALATE without Haiku.
        txn_dict = txn.model_dump()
        for pattern in active_patterns:
            pattern_name = pattern.get("pattern_name", "")
            # Patterns carry a natural-language detection_rule.  We include them in
            # the Haiku prompt rather than evaluating the rule in Python; but we can
            # do a lightweight pre-check on country/channel fields if present in the
            # rule text to avoid an unnecessary LLM call.
            # For now, always include matched patterns in the prompt and let Haiku decide.
            matched_patterns.append(pattern_name)

        if matched_patterns:
            self._metrics.record_pattern_match()
            escalation_target = "BOTH" if has_aml else "FRAUD_ONLY"
            reasoning = (
                f"Gray zone (similarity={max_similarity:.3f}); "
                f"pattern matches fired: {matched_patterns}. Escalating to Tier 2."
            )
            self._metrics.record_escalation()
            return self._build_decision(
                txn=txn,
                verdict="ESCALATE",
                tier="SENTINEL",
                confidence=0.75,
                reasoning=reasoning,
                escalation_target=escalation_target,
                pattern_matches=matched_patterns,
                tokens_used=0,
                timestamp=now_iso,
                latency_ms=int(time.monotonic() * 1000) - start_ms,
                ttl=ttl_epoch,
            )

        # No pattern match in gray zone — call Haiku for final decision.
        prompt = self._load_prompt(
            "sentinel",
            transaction_json=json.dumps(txn_dict, default=str),
            persona_json=json.dumps(persona, default=str) if persona else "null",
            similar_transactions=json.dumps(similar_results, default=str),
            pattern_matches=json.dumps(matched_patterns),
        )

        haiku_response = self._bedrock.invoke_haiku(prompt, max_tokens=512)
        tokens_used: int = haiku_response.get("tokens_used", 0)
        self._metrics.record_token_spend(tokens_used)

        parsed = self._parse_llm_response(
            haiku_response.get("text", ""),
            context={"transaction_id": txn.transaction_id},
        )

        llm_verdict: str = parsed.get("verdict", "ESCALATE")
        llm_confidence: float = float(parsed.get("confidence", 0.5))
        llm_reasoning: str = sanitize_llm_input(parsed.get("reasoning", ""))
        llm_target: str | None = parsed.get("escalation_target")

        # Reconcile AML overlay with LLM verdict.
        if has_aml:
            if llm_verdict == "ESCALATE":
                # Upgrade to BOTH if fraud path also wants to escalate.
                llm_target = "BOTH"
            else:
                # LLM approved but AML overrides.
                llm_verdict = "ESCALATE"
                llm_target = "AML_ONLY"
                llm_reasoning += " [AML signal override]"

        if llm_verdict == "ESCALATE":
            self._metrics.record_escalation()
        else:
            self._metrics.record_auto_approve()

        return self._build_decision(
            txn=txn,
            verdict=llm_verdict,
            tier="SENTINEL",
            confidence=llm_confidence,
            reasoning=llm_reasoning,
            escalation_target=llm_target if llm_verdict == "ESCALATE" else None,
            pattern_matches=matched_patterns,
            tokens_used=tokens_used,
            timestamp=now_iso,
            latency_ms=int(time.monotonic() * 1000) - start_ms,
            ttl=ttl_epoch,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_persona(self, user_id: str) -> dict | None:
        """Return the behavioral persona for *user_id*.

        Checks Redis first; on cache miss, queries DynamoDB and caches the result.
        Returns None if no persona exists yet for this user.
        """
        # Redis cache read
        cached = self._redis.get_persona_cache(user_id)
        if cached is not None:
            logger.debug("Persona cache hit", extra={"user_id": user_id})
            return cached

        # DynamoDB fallback
        persona = self._dynamo.get_persona(
            table_name=self._config.dynamo_table_personas,
            user_id=user_id,
        )
        if persona is not None:
            # Populate cache on miss.
            try:
                self._redis.set_persona_cache(
                    user_id, persona, ttl=self._config.persona_cache_ttl
                )
            except Exception:
                # Cache write failure must not block transaction processing.
                logger.warning(
                    "Failed to write persona to Redis cache — continuing",
                    extra={"user_id": user_id},
                    exc_info=True,
                )

        return persona

    def _check_persona_deviation(
        self,
        txn: EnrichedTransaction,
        persona: dict | None,
    ) -> list[str]:
        """Check if a transaction deviates from the user's behavioral persona.

        Returns a list of human-readable anomaly descriptions.  An empty list
        means the transaction is consistent with the persona.  If no persona
        exists (cold user), returns a single "cold user" anomaly.
        """
        anomalies: list[str] = []

        if persona is None:
            anomalies.append("no behavioral persona (cold user)")
            return anomalies

        # --- Amount anomaly ---
        velocity = persona.get("velocity", {})
        max_single = velocity.get("max_single_txn", 0)
        daily_spend = velocity.get("daily_spend_amount", 0)
        if max_single > 0 and txn.amount > max_single * 3:
            anomalies.append(
                f"amount ${txn.amount:.2f} is >3x user max (${max_single:.2f})"
            )
        elif daily_spend > 0 and txn.amount > daily_spend * 5:
            anomalies.append(
                f"amount ${txn.amount:.2f} is >5x daily average (${daily_spend:.2f})"
            )

        # --- Geographic anomaly ---
        geo_footprint = persona.get("geo_footprint", [])
        if geo_footprint and txn.merchant_country:
            # Extract known countries directly from geo_footprint entries.
            known_countries = {
                g.get("country", "").upper()
                for g in geo_footprint
                if g.get("country")
            }
            known_cities = {g.get("city", "").lower() for g in geo_footprint}
            txn_country = (txn.merchant_country or "").upper()
            txn_city = (txn.merchant_city or "").lower()

            if known_countries and txn_country not in known_countries:
                # Country-level mismatch is a strong signal.
                anomalies.append(
                    f"country {txn_country} not in known countries {known_countries}"
                )
            elif txn_city and known_cities and txn_city not in known_cities:
                # City-level mismatch within a known country.
                anomalies.append(
                    f"new city '{txn_city}' in known country {txn_country}"
                )

        # --- Temporal anomaly ---
        temporal = persona.get("temporal_profile", {})
        active_hours = temporal.get("active_hours", [])
        if active_hours and len(active_hours) == 2 and txn.local_hour is not None:
            start_hour, end_hour = active_hours[0], active_hours[1]
            if start_hour <= end_hour:
                in_window = start_hour <= txn.local_hour <= end_hour
            else:
                # Wraps midnight (e.g., [22, 6])
                in_window = txn.local_hour >= start_hour or txn.local_hour <= end_hour
            if not in_window:
                anomalies.append(
                    f"hour {txn.local_hour} outside active window [{start_hour}-{end_hour}]"
                )

        # --- Category anomaly ---
        category_anchors = persona.get("category_anchors", [])
        if category_anchors and txn.merchant_category:
            known_categories = {c.get("category", "").lower() for c in category_anchors}
            if txn.merchant_category.lower() not in known_categories:
                anomalies.append(
                    f"category '{txn.merchant_category}' not in user's known categories"
                )

        # --- IP / device anomaly ---
        ip_footprint = persona.get("ip_footprint", [])
        if ip_footprint and txn.ip_address:
            # We don't have ASN resolution locally, but we can check if the
            # IP region is known.  Since GeoIP may not be available, this is
            # a best-effort check based on geo_city from enrichment.
            if txn.geo_city:
                known_regions = {
                    ip.get("region", "").lower() for ip in ip_footprint
                }
                if known_regions and not any(
                    txn.geo_city.lower() in r for r in known_regions
                ):
                    anomalies.append(
                        f"IP region '{txn.geo_city}' not in known IP footprint"
                    )

        return anomalies

    def _fetch_patterns(self) -> list[dict]:
        """Return active fraud patterns from Redis cache or DynamoDB scan."""
        cached = self._redis.get_pattern_cache()
        if cached is not None:
            logger.debug("Pattern cache hit")
            return cached

        patterns = self._dynamo.scan_patterns(
            table_name=self._config.dynamo_table_patterns,
            status="ACTIVE",
        )
        try:
            self._redis.set_pattern_cache(patterns)
        except Exception:
            logger.warning(
                "Failed to write patterns to Redis cache — continuing",
                exc_info=True,
            )
        return patterns

    def _load_prompt(self, template_name: str, **kwargs: object) -> str:
        """Load a prompt template from the prompts/ directory and format it.

        Args:
            template_name: Template file name without the .txt extension.
            **kwargs:       Placeholder values to substitute in the template.

        Returns:
            Formatted prompt string ready to send to a model.

        Raises:
            FileNotFoundError: If the template file does not exist.
            KeyError:          If a required placeholder is missing from kwargs.
        """
        template_path = os.path.join(_PROMPTS_DIR, f"{template_name}.txt")
        with open(template_path, encoding="utf-8") as fh:
            template = fh.read()
        return template.format(**kwargs)

    def _parse_llm_response(
        self, text: str, context: dict
    ) -> dict:
        """Extract and parse the JSON payload from a model response.

        Args:
            text:    Raw text returned by the model.
            context: Structured context for error logging (e.g. transaction_id).

        Returns:
            Parsed dict.  Falls back to a safe ESCALATE sentinel on parse failure
            so a bad model response never crashes the pipeline.
        """
        try:
            json_str = _extract_json(text)
            return json.loads(json_str)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error(
                "Failed to parse Sentinel LLM response — defaulting to ESCALATE",
                extra={
                    **context,
                    "errorType": type(exc).__name__,
                    "error_detail": str(exc),
                    "raw_text": text[:500],
                },
            )
            return {
                "verdict": "ESCALATE",
                "confidence": 0.5,
                "reasoning": "Parse error — defaulting to escalation for safety.",
                "escalation_target": "FRAUD_ONLY",
            }

    @staticmethod
    def _build_decision(
        txn: EnrichedTransaction,
        verdict: str,
        tier: str,
        confidence: float,
        reasoning: str,
        escalation_target: str | None,
        pattern_matches: list[str],
        tokens_used: int,
        timestamp: str,
        latency_ms: int,
        ttl: int,
    ) -> FraudDecision:
        """Construct a FraudDecision from routing results."""
        return FraudDecision(
            transaction_id=txn.transaction_id,
            user_id=txn.user_id,
            verdict=verdict,
            tier=tier,
            confidence=confidence,
            reasoning=reasoning,
            escalation_target=escalation_target,
            pattern_matches=pattern_matches,
            timestamp=timestamp,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            ttl=ttl,
        )
