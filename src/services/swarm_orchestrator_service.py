"""
SwarmOrchestratorService — Tier 2 multi-agent invocation coordinator.

The Swarm Orchestrator receives an escalation payload from the Sentinel and
fans it out to the appropriate specialist Lambda functions based on the
escalation_target field:

  FRAUD_ONLY  — invoke fraud_analyst Lambda synchronously.
  AML_ONLY    — invoke aml_specialist Lambda synchronously.
  BOTH        — invoke fraud_analyst synchronously (it blocks the payment
                flow) AND aml_specialist asynchronously (fire-and-forget,
                because AML analysis must not delay a blocking decision
                and disclosing the investigation would violate the tipping-off
                doctrine).

The orchestrator does not interpret results from the specialist Lambdas beyond
logging — result handling is the responsibility of the calling handler.
"""

import logging

from src.clients.lambda_client import LambdaInvokeClient
from src.config import Config

logger = logging.getLogger(__name__)


class SwarmOrchestratorService:
    """Routes escalation payloads to specialist Lambda functions."""

    def __init__(
        self,
        config: Config,
        lambda_client: LambdaInvokeClient,
    ) -> None:
        self._config = config
        self._lambda = lambda_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def orchestrate(self, escalation_payload: dict) -> None:
        """Fan out the escalation payload to the appropriate specialist(s).

        Expected fields in *escalation_payload*:
            transaction         — enriched transaction dict
            persona             — behavioral persona dict or None
            similar_transactions — kNN results list from the Sentinel
            escalation_target   — "FRAUD_ONLY" | "AML_ONLY" | "BOTH"
            aml_signals         — list of AMLSignal dicts (may be empty)
            escalation_context  — Sentinel's reasoning string

        Args:
            escalation_payload: Dict with the fields listed above.

        Raises:
            ValueError:    If escalation_target is missing or unrecognised.
            RuntimeError:  Propagated from LambdaInvokeClient on function error
                           (synchronous invocations only).
            Exception:     Propagated from boto3 on transport failure.
        """
        escalation_target: str = escalation_payload.get("escalation_target", "")
        transaction_id: str = escalation_payload.get(
            "transaction", {}
        ).get("transaction_id", "unknown")
        user_id: str = escalation_payload.get(
            "transaction", {}
        ).get("user_id", "unknown")

        if not escalation_target:
            raise ValueError(
                "escalation_payload must contain a non-empty 'escalation_target' field."
            )

        logger.info(
            "Swarm orchestrator routing escalation",
            extra={
                "transaction_id": transaction_id,
                "user_id": user_id,
                "escalation_target": escalation_target,
            },
        )

        if escalation_target == "FRAUD_ONLY":
            self._invoke_fraud_analyst(escalation_payload, transaction_id, user_id)

        elif escalation_target == "AML_ONLY":
            self._invoke_aml_specialist_sync(escalation_payload, transaction_id, user_id)

        elif escalation_target == "BOTH":
            # Fraud analyst runs synchronously — its verdict determines whether the
            # transaction is blocked and must be available before responding to the caller.
            self._invoke_fraud_analyst(escalation_payload, transaction_id, user_id)

            # AML specialist runs asynchronously — fire-and-forget.
            # AML analysis must not delay the fraud decision, and tipping off the
            # subject by blocking based on AML signals is prohibited by law.
            self._invoke_aml_specialist_async(escalation_payload, transaction_id, user_id)

        else:
            raise ValueError(
                f"Unknown escalation_target '{escalation_target}'. "
                "Expected one of: FRAUD_ONLY, AML_ONLY, BOTH."
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_fraud_analyst_payload(self, escalation_payload: dict) -> dict:
        """Construct the payload for the FraudAnalyst Lambda.

        The FraudAnalyst Lambda expects:
            transaction_dict      — enriched transaction dict
            persona_dict          — persona or None
            similar_transactions  — kNN results list
            escalation_context    — Sentinel reasoning string

        The aml_signals list is appended to the escalation_context string so
        the Fraud Analyst has full signal visibility without schema changes.
        """
        aml_signals = escalation_payload.get("aml_signals", [])
        base_context: str = escalation_payload.get("escalation_context", "")
        context_with_aml = base_context
        if aml_signals:
            import json
            context_with_aml = (
                f"{base_context}\n\nAML signals detected by Sentinel: "
                f"{json.dumps(aml_signals, default=str)}"
            )

        return {
            "transaction_dict": escalation_payload.get("transaction", {}),
            "persona_dict": escalation_payload.get("persona"),
            "similar_transactions": escalation_payload.get("similar_transactions", []),
            "escalation_context": context_with_aml,
        }

    def _build_aml_specialist_payload(self, escalation_payload: dict) -> dict:
        """Construct the payload for the AMLSpecialist Lambda.

        The AMLSpecialist Lambda expects:
            transaction_dict  — enriched transaction dict
            persona_dict      — persona or None
            user_history      — list of recent transactions (Sentinel passes this
                                through; may be empty if not pre-fetched)
            current_score     — float AML risk score from DynamoDB
            existing_case     — open case dict or None
            aml_signals       — structural signals detected by Sentinel

        Note: current_score and existing_case are fetched by the Lambda handler
        itself at invocation time from DynamoDB. The Sentinel-detected aml_signals
        are forwarded as advisory context.
        """
        return {
            "transaction_dict": escalation_payload.get("transaction", {}),
            "persona_dict": escalation_payload.get("persona"),
            "user_history": escalation_payload.get("user_history", []),
            "current_score": escalation_payload.get("current_aml_score", 0.0),
            "existing_case": escalation_payload.get("existing_case"),
            "aml_signals": escalation_payload.get("aml_signals", []),
        }

    def _invoke_fraud_analyst(
        self,
        escalation_payload: dict,
        transaction_id: str,
        user_id: str,
    ) -> dict:
        """Invoke the FraudAnalyst Lambda synchronously and return its response."""
        payload = self._build_fraud_analyst_payload(escalation_payload)
        logger.debug(
            "Invoking FraudAnalyst Lambda synchronously",
            extra={
                "function": self._config.fraud_analyst_function_name,
                "transaction_id": transaction_id,
                "user_id": user_id,
            },
        )
        result = self._lambda.invoke_sync(
            function_name=self._config.fraud_analyst_function_name,
            payload=payload,
        )
        logger.info(
            "FraudAnalyst Lambda returned",
            extra={
                "transaction_id": transaction_id,
                "user_id": user_id,
                "verdict": result.get("verdict"),
            },
        )
        return result

    def _invoke_aml_specialist_sync(
        self,
        escalation_payload: dict,
        transaction_id: str,
        user_id: str,
    ) -> dict:
        """Invoke the AMLSpecialist Lambda synchronously."""
        payload = self._build_aml_specialist_payload(escalation_payload)
        logger.debug(
            "Invoking AMLSpecialist Lambda synchronously",
            extra={
                "function": self._config.aml_specialist_function_name,
                "transaction_id": transaction_id,
                "user_id": user_id,
            },
        )
        result = self._lambda.invoke_sync(
            function_name=self._config.aml_specialist_function_name,
            payload=payload,
        )
        logger.info(
            "AMLSpecialist Lambda returned",
            extra={
                "transaction_id": transaction_id,
                "user_id": user_id,
                "new_score": result.get("new_score"),
            },
        )
        return result

    def _invoke_aml_specialist_async(
        self,
        escalation_payload: dict,
        transaction_id: str,
        user_id: str,
    ) -> None:
        """Invoke the AMLSpecialist Lambda asynchronously (fire-and-forget)."""
        payload = self._build_aml_specialist_payload(escalation_payload)
        logger.debug(
            "Invoking AMLSpecialist Lambda asynchronously",
            extra={
                "function": self._config.aml_specialist_function_name,
                "transaction_id": transaction_id,
                "user_id": user_id,
            },
        )
        self._lambda.invoke_async(
            function_name=self._config.aml_specialist_function_name,
            payload=payload,
        )
        logger.info(
            "AMLSpecialist Lambda queued (async)",
            extra={
                "transaction_id": transaction_id,
                "user_id": user_id,
            },
        )
