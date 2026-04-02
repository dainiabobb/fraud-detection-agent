"""
Unit tests for AMLSpecialistService (Tier 2B).

Key invariants:
  - AML Specialist never returns a BLOCK verdict (returns a dict, not FraudDecision).
  - Score is clamped to [0.0, 100.0] regardless of LLM delta.
  - Investigation case is opened when open_investigation=True and no existing case.
  - Existing case is escalated when escalate_to_compliance=True and case exists.
  - Existing case is updated (transaction appended) when neither flag is set.
  - Score history entry is always appended (verified via DynamoDB put call).
  - Metrics are published for every analysis run.
"""

import json
from unittest.mock import MagicMock, call, patch

import pytest

from src.services.aml_specialist_service import AMLSpecialistService

_PROMPT_TEMPLATE = (
    "AML analysis: {transaction_json} persona: {persona_json} "
    "history: {user_history_6mo} score: {current_aml_score} case: {existing_case}"
)


def _make_service(
    sample_config,
    mock_dynamodb_client,
    mock_bedrock_client,
    mock_metrics,
) -> AMLSpecialistService:
    return AMLSpecialistService(
        config=sample_config,
        dynamodb_client=mock_dynamodb_client,
        bedrock_client=mock_bedrock_client,
        metrics=mock_metrics,
    )


def _run_analyze(
    service: AMLSpecialistService,
    transaction_dict: dict,
    sonnet_response: dict,
    current_score: float = 0.0,
    existing_case: dict | None = None,
    existing_score_record: dict | None = None,
):
    """Configure mocks and invoke analyze()."""
    service._bedrock.invoke_sonnet.return_value = {
        "text": json.dumps(sonnet_response),
        "tokens_used": 300,
    }
    service._dynamo.get_aml_risk_score.return_value = existing_score_record

    with patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = _PROMPT_TEMPLATE
        return service.analyze(
            transaction_dict=transaction_dict,
            persona_dict=None,
            user_history=[],
            current_score=current_score,
            existing_case=existing_case,
        )


# ---------------------------------------------------------------------------
# Score update
# ---------------------------------------------------------------------------


class TestScoreUpdate:
    def test_score_delta_applied_and_returned(
        self,
        sample_config,
        sample_transaction,
        mock_dynamodb_client,
        mock_bedrock_client,
        mock_metrics,
    ):
        service = _make_service(sample_config, mock_dynamodb_client, mock_bedrock_client, mock_metrics)
        result = _run_analyze(
            service,
            sample_transaction,
            sonnet_response={
                "score_delta": 20,
                "typologies_detected": ["STRUCTURING"],
                "reasoning": "Multiple near-threshold deposits.",
                "open_investigation": False,
                "escalate_to_compliance": False,
            },
            current_score=10.0,
        )
        assert result["score_delta"] == 20
        assert result["new_score"] == 30.0

    def test_score_clamped_at_zero_for_negative_delta(
        self,
        sample_config,
        sample_transaction,
        mock_dynamodb_client,
        mock_bedrock_client,
        mock_metrics,
    ):
        service = _make_service(sample_config, mock_dynamodb_client, mock_bedrock_client, mock_metrics)
        result = _run_analyze(
            service,
            sample_transaction,
            sonnet_response={
                "score_delta": -50,
                "typologies_detected": [],
                "reasoning": "No suspicious activity.",
                "open_investigation": False,
                "escalate_to_compliance": False,
            },
            current_score=5.0,
        )
        # 5 + (-50) = -45 → clamped to 0
        assert result["new_score"] == 0.0

    def test_score_clamped_at_hundred(
        self,
        sample_config,
        sample_transaction,
        mock_dynamodb_client,
        mock_bedrock_client,
        mock_metrics,
    ):
        service = _make_service(sample_config, mock_dynamodb_client, mock_bedrock_client, mock_metrics)
        result = _run_analyze(
            service,
            sample_transaction,
            sonnet_response={
                "score_delta": 60,
                "typologies_detected": ["LAYERING"],
                "reasoning": "Complex layering pattern.",
                "open_investigation": True,
                "escalate_to_compliance": False,
            },
            current_score=80.0,
        )
        # 80 + 60 = 140 → clamped to 100
        assert result["new_score"] == 100.0

    def test_score_history_appended_on_dynamo_write(
        self,
        sample_config,
        sample_transaction,
        mock_dynamodb_client,
        mock_bedrock_client,
        mock_metrics,
    ):
        """When an existing score record exists, the new update is appended to history."""
        existing_score_record = {
            "user_id": sample_transaction["user_id"],
            "current_score": 20.0,
            "last_updated": "2026-03-30T10:00:00Z",
            "score_history": [
                {
                    "timestamp": "2026-03-30T10:00:00Z",
                    "delta": 20.0,
                    "reason": "Prior event.",
                    "transaction_id": "txn-prior",
                }
            ],
            "investigation_case_id": None,
        }
        service = _make_service(sample_config, mock_dynamodb_client, mock_bedrock_client, mock_metrics)
        _run_analyze(
            service,
            sample_transaction,
            sonnet_response={
                "score_delta": 10,
                "typologies_detected": [],
                "reasoning": "Minor event.",
                "open_investigation": False,
                "escalate_to_compliance": False,
            },
            current_score=20.0,
            existing_score_record=existing_score_record,
        )

        mock_dynamodb_client.put_aml_risk_score.assert_called_once()
        written_item = mock_dynamodb_client.put_aml_risk_score.call_args.kwargs["score"]
        # History should now have 2 entries: 1 prior + 1 new
        assert len(written_item["score_history"]) == 2


# ---------------------------------------------------------------------------
# Investigation case management
# ---------------------------------------------------------------------------


class TestInvestigationCaseManagement:
    def test_investigation_opened_when_flag_true_and_no_existing_case(
        self,
        sample_config,
        sample_transaction,
        mock_dynamodb_client,
        mock_bedrock_client,
        mock_metrics,
    ):
        service = _make_service(sample_config, mock_dynamodb_client, mock_bedrock_client, mock_metrics)
        result = _run_analyze(
            service,
            sample_transaction,
            sonnet_response={
                "score_delta": 35,
                "typologies_detected": ["STRUCTURING"],
                "reasoning": "Structuring pattern confirmed.",
                "open_investigation": True,
                "escalate_to_compliance": False,
            },
            current_score=25.0,
            existing_case=None,
        )

        assert result["open_investigation"] is True
        assert result["case_id"] is not None
        assert result["case_status"] == "OPEN"
        mock_dynamodb_client.put_investigation_case.assert_called_once()
        mock_metrics.record_investigation_opened.assert_called_once()

    def test_compliance_escalation_when_flag_true_and_case_exists(
        self,
        sample_config,
        sample_transaction,
        mock_dynamodb_client,
        mock_bedrock_client,
        mock_metrics,
    ):
        existing_case = {
            "case_id": "CASE-EXISTING-001",
            "user_id": sample_transaction["user_id"],
            "status": "OPEN",
            "opened_at": "2026-03-01T10:00:00Z",
            "risk_score_at_open": 52.0,
            "typologies_detected": ["STRUCTURING"],
            "transactions": ["txn-old"],
            "notes": ["Initial note."],
        }
        service = _make_service(sample_config, mock_dynamodb_client, mock_bedrock_client, mock_metrics)
        result = _run_analyze(
            service,
            sample_transaction,
            sonnet_response={
                "score_delta": 30,
                "typologies_detected": ["LAYERING"],
                "reasoning": "Complex layering now evident.",
                "open_investigation": False,
                "escalate_to_compliance": True,
            },
            current_score=55.0,
            existing_case=existing_case,
        )

        assert result["case_status"] == "ESCALATED"
        assert result["case_id"] == "CASE-EXISTING-001"
        mock_dynamodb_client.put_investigation_case.assert_called_once()
        # Verify the written case has ESCALATED status
        written_case = mock_dynamodb_client.put_investigation_case.call_args.kwargs["case"]
        assert written_case["status"] == "ESCALATED"
        assert written_case["escalated_at"] is not None
        mock_metrics.record_investigation_escalated.assert_called_once()

    def test_existing_case_updated_with_new_transaction(
        self,
        sample_config,
        sample_transaction,
        mock_dynamodb_client,
        mock_bedrock_client,
        mock_metrics,
    ):
        existing_case = {
            "case_id": "CASE-EXISTING-002",
            "user_id": sample_transaction["user_id"],
            "status": "OPEN",
            "opened_at": "2026-03-10T10:00:00Z",
            "risk_score_at_open": 51.0,
            "typologies_detected": ["SMURFING"],
            "transactions": ["txn-original"],
            "notes": ["First note."],
        }
        service = _make_service(sample_config, mock_dynamodb_client, mock_bedrock_client, mock_metrics)
        _run_analyze(
            service,
            sample_transaction,
            sonnet_response={
                "score_delta": 5,
                "typologies_detected": [],
                "reasoning": "Continued monitoring.",
                "open_investigation": False,
                "escalate_to_compliance": False,
            },
            current_score=55.0,
            existing_case=existing_case,
        )

        mock_dynamodb_client.put_investigation_case.assert_called_once()
        written = mock_dynamodb_client.put_investigation_case.call_args.kwargs["case"]
        # New transaction appended
        assert sample_transaction["transaction_id"] in written["transactions"]
        # Note appended
        assert len(written["notes"]) == 2


# ---------------------------------------------------------------------------
# AML never blocks (return type is dict, no BLOCK field)
# ---------------------------------------------------------------------------


class TestAMLNeverBlocks:
    def test_result_has_no_block_verdict_field(
        self,
        sample_config,
        sample_transaction,
        mock_dynamodb_client,
        mock_bedrock_client,
        mock_metrics,
    ):
        service = _make_service(sample_config, mock_dynamodb_client, mock_bedrock_client, mock_metrics)
        result = _run_analyze(
            service,
            sample_transaction,
            sonnet_response={
                "score_delta": 25,
                "typologies_detected": ["STRUCTURING"],
                "reasoning": "Structuring detected.",
                "open_investigation": True,
                "escalate_to_compliance": False,
            },
            current_score=30.0,
        )
        # AML returns a dict, not a FraudDecision — no 'verdict' key
        assert "verdict" not in result
        # Core AML result keys present
        assert "new_score" in result
        assert "typologies_detected" in result
        assert "open_investigation" in result
