"""
Unit tests for SwarmOrchestratorService.

Verifies that the correct Lambda functions are invoked — synchronously or
asynchronously — based on the escalation_target field in the payload.
No prompt files or real AWS calls are used.
"""

import pytest

from src.services.swarm_orchestrator_service import SwarmOrchestratorService


def _make_service(sample_config, mock_lambda_client) -> SwarmOrchestratorService:
    return SwarmOrchestratorService(
        config=sample_config,
        lambda_client=mock_lambda_client,
    )


def _base_payload(escalation_target: str, extra: dict | None = None) -> dict:
    payload = {
        "escalation_target": escalation_target,
        "transaction": {
            "transaction_id": "txn-orch-001",
            "user_id": "user-orch-001",
        },
        "persona": None,
        "similar_transactions": [],
        "aml_signals": [],
        "escalation_context": "Low similarity detected by Sentinel.",
    }
    if extra:
        payload.update(extra)
    return payload


# ---------------------------------------------------------------------------
# FRAUD_ONLY routing
# ---------------------------------------------------------------------------


class TestFraudOnlyRouting:
    def test_fraud_only_invokes_fraud_analyst_sync(
        self, sample_config, mock_lambda_client
    ):
        service = _make_service(sample_config, mock_lambda_client)
        mock_lambda_client.invoke_sync.return_value = {
            "verdict": "BLOCK",
            "confidence": 0.95,
        }

        service.orchestrate(_base_payload("FRAUD_ONLY"))

        mock_lambda_client.invoke_sync.assert_called_once()
        call_args = mock_lambda_client.invoke_sync.call_args
        assert call_args.kwargs["function_name"] == sample_config.fraud_analyst_function_name

    def test_fraud_only_does_not_invoke_aml_specialist(
        self, sample_config, mock_lambda_client
    ):
        service = _make_service(sample_config, mock_lambda_client)
        mock_lambda_client.invoke_sync.return_value = {"verdict": "APPROVE"}

        service.orchestrate(_base_payload("FRAUD_ONLY"))

        mock_lambda_client.invoke_async.assert_not_called()


# ---------------------------------------------------------------------------
# AML_ONLY routing
# ---------------------------------------------------------------------------


class TestAmlOnlyRouting:
    def test_aml_only_invokes_aml_specialist_sync(
        self, sample_config, mock_lambda_client
    ):
        service = _make_service(sample_config, mock_lambda_client)
        mock_lambda_client.invoke_sync.return_value = {
            "new_score": 45.0,
            "open_investigation": False,
        }

        service.orchestrate(_base_payload("AML_ONLY"))

        mock_lambda_client.invoke_sync.assert_called_once()
        call_args = mock_lambda_client.invoke_sync.call_args
        assert call_args.kwargs["function_name"] == sample_config.aml_specialist_function_name

    def test_aml_only_does_not_invoke_fraud_analyst(
        self, sample_config, mock_lambda_client
    ):
        service = _make_service(sample_config, mock_lambda_client)
        mock_lambda_client.invoke_sync.return_value = {"new_score": 10.0}

        service.orchestrate(_base_payload("AML_ONLY"))

        # Only one sync call (AML), no async call
        assert mock_lambda_client.invoke_sync.call_count == 1
        called_function = mock_lambda_client.invoke_sync.call_args.kwargs["function_name"]
        assert called_function != sample_config.fraud_analyst_function_name


# ---------------------------------------------------------------------------
# BOTH routing
# ---------------------------------------------------------------------------


class TestBothRouting:
    def test_both_invokes_fraud_analyst_sync_and_aml_async(
        self, sample_config, mock_lambda_client
    ):
        service = _make_service(sample_config, mock_lambda_client)
        mock_lambda_client.invoke_sync.return_value = {"verdict": "BLOCK"}

        service.orchestrate(_base_payload("BOTH"))

        # Fraud analyst must have been called synchronously
        mock_lambda_client.invoke_sync.assert_called_once()
        sync_fn = mock_lambda_client.invoke_sync.call_args.kwargs["function_name"]
        assert sync_fn == sample_config.fraud_analyst_function_name

        # AML specialist must have been fired asynchronously
        mock_lambda_client.invoke_async.assert_called_once()
        async_fn = mock_lambda_client.invoke_async.call_args.kwargs["function_name"]
        assert async_fn == sample_config.aml_specialist_function_name

    def test_both_aml_signals_forwarded_in_fraud_analyst_context(
        self, sample_config, mock_lambda_client
    ):
        """AML signals detected by Sentinel are appended to escalation_context."""
        service = _make_service(sample_config, mock_lambda_client)
        mock_lambda_client.invoke_sync.return_value = {"verdict": "BLOCK"}

        payload = _base_payload(
            "BOTH",
            extra={
                "aml_signals": [{"signal_type": "STRUCTURING", "confidence": 0.85, "details": {}}],
                "escalation_context": "Low similarity.",
            },
        )
        service.orchestrate(payload)

        fraud_payload = mock_lambda_client.invoke_sync.call_args.kwargs["payload"]
        # The AML signal text must be embedded in the escalation_context forwarded to fraud analyst
        assert "STRUCTURING" in fraud_payload["escalation_context"]


# ---------------------------------------------------------------------------
# Unknown escalation_target
# ---------------------------------------------------------------------------


class TestUnknownEscalationTarget:
    def test_unknown_target_raises_value_error(
        self, sample_config, mock_lambda_client
    ):
        service = _make_service(sample_config, mock_lambda_client)

        with pytest.raises(ValueError, match="Unknown escalation_target"):
            service.orchestrate(_base_payload("UNKNOWN_TARGET"))

    def test_missing_escalation_target_raises_value_error(
        self, sample_config, mock_lambda_client
    ):
        service = _make_service(sample_config, mock_lambda_client)
        payload = {
            "transaction": {"transaction_id": "txn-x", "user_id": "user-x"},
            # escalation_target intentionally omitted
        }

        with pytest.raises(ValueError, match="escalation_target"):
            service.orchestrate(payload)
