"""
CloudWatch custom metrics publisher for the fraud detection agent.

All metrics are published under the "FraudDetection/Metrics" namespace.
The FraudMetrics class is designed to be instantiated once per Lambda
cold-start and reused across invocations — the boto3 client is injected at
construction time so it can be mocked in unit tests without patching.

Usage:
    import boto3
    from src.utils.metrics import FraudMetrics

    _cw = boto3.client("cloudwatch", region_name=os.environ["AWS_REGION"])
    metrics = FraudMetrics(_cw)
    metrics.record_escalation()
    metrics.record_latency("evaluate-transaction", latency_ms=342)
"""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Avoid importing boto3 at type-check time so the stubs aren't required.
    import mypy_boto3_cloudwatch as CloudWatchClient  # noqa: F401

logger = logging.getLogger(__name__)

_NAMESPACE = "FraudDetection/Metrics"


class FraudMetrics:
    """Publishes structured CloudWatch custom metrics for fraud detection."""

    def __init__(self, cloudwatch_client: Any) -> None:
        """
        Args:
            cloudwatch_client: A boto3 CloudWatch client.  Injected rather
                               than created internally so tests can pass a
                               mock without patching boto3 globally.
        """
        self._cw = cloudwatch_client

    # ------------------------------------------------------------------
    # Decision outcome metrics
    # ------------------------------------------------------------------

    def record_escalation(self) -> None:
        """Increment the count of transactions escalated for human review."""
        self._put_metric("EscalationRate", 1, "Count")

    def record_auto_approve(self) -> None:
        """Increment the count of transactions auto-approved by the agent."""
        self._put_metric("AutoApproveCount", 1, "Count")

    def record_block(self) -> None:
        """Increment the count of transactions blocked by the agent."""
        self._put_metric("BlockCount", 1, "Count")

    # ------------------------------------------------------------------
    # LLM cost / usage
    # ------------------------------------------------------------------

    def record_token_spend(self, tokens: int) -> None:
        """Publish the number of tokens consumed by a single model invocation.

        Args:
            tokens: Total tokens (input + output) for the invocation.
        """
        self._put_metric("TokenSpend", float(tokens), "Count")

    # ------------------------------------------------------------------
    # Pattern matching
    # ------------------------------------------------------------------

    def record_pattern_match(self) -> None:
        """Increment the count of escalations triggered by a pattern match."""
        self._put_metric("PatternMatchEscalationCount", 1, "Count")

    def record_new_pattern(self) -> None:
        """Increment the count of new fraud patterns discovered and stored."""
        self._put_metric("NewPatternsDiscovered", 1, "Count")

    def record_pattern_retired(self) -> None:
        """Increment the count of fraud patterns retired (expired / superseded)."""
        self._put_metric("PatternsRetired", 1, "Count")

    # ------------------------------------------------------------------
    # AML metrics
    # ------------------------------------------------------------------

    def record_aml_escalation(self) -> None:
        """Increment the count of AML rule-triggered escalations."""
        self._put_metric("AMLEscalationCount", 1, "Count")

    def record_aml_risk_update(self) -> None:
        """Increment the count of AML risk score updates written to the store."""
        self._put_metric("AMLRiskScoreUpdates", 1, "Count")

    # ------------------------------------------------------------------
    # Investigation case management
    # ------------------------------------------------------------------

    def record_investigation_opened(self) -> None:
        """Increment the count of new investigation cases opened."""
        self._put_metric("InvestigationCasesOpened", 1, "Count")

    def record_investigation_escalated(self) -> None:
        """Increment the count of investigations escalated to compliance."""
        self._put_metric("InvestigationCasesEscalatedToCompliance", 1, "Count")

    # ------------------------------------------------------------------
    # Latency
    # ------------------------------------------------------------------

    def record_latency(self, function_name: str, latency_ms: int) -> None:
        """Publish end-to-end latency for a named Lambda function.

        Publishes with a ``FunctionName`` dimension so each function's latency
        can be graphed and alarmed independently in CloudWatch.

        Args:
            function_name: The logical function name (e.g. "evaluate-transaction").
                           Use the same name consistently across invocations.
            latency_ms:    Wall-clock duration in milliseconds.
        """
        try:
            self._cw.put_metric_data(
                Namespace=_NAMESPACE,
                MetricData=[
                    {
                        "MetricName": "Latency",
                        "Dimensions": [
                            {"Name": "FunctionName", "Value": function_name}
                        ],
                        "Value": float(latency_ms),
                        "Unit": "Milliseconds",
                    }
                ],
            )
        except Exception as exc:  # noqa: BLE001
            # Metrics publishing must never crash the main Lambda handler.
            logger.warning(
                "Failed to publish Latency metric for function=%s: %s",
                function_name,
                exc,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _put_metric(self, name: str, value: float, unit: str) -> None:
        """Publish a single metric data point to CloudWatch.

        Args:
            name:  CloudWatch metric name.
            value: Numeric value of the data point.
            unit:  CloudWatch unit string (e.g. "Count", "Milliseconds").
        """
        try:
            self._cw.put_metric_data(
                Namespace=_NAMESPACE,
                MetricData=[
                    {
                        "MetricName": name,
                        "Value": value,
                        "Unit": unit,
                    }
                ],
            )
        except Exception as exc:  # noqa: BLE001
            # Structured log so downstream log-based alarms can catch failures.
            logger.warning(
                "Failed to publish metric name=%s value=%s unit=%s: %s",
                name,
                value,
                unit,
                exc,
                exc_info=True,
            )
