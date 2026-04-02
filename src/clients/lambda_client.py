"""
LambdaInvokeClient — thin wrapper around a boto3 Lambda client.

Constructor injection keeps the class testable without patching boto3 globally.

Invocation modes:
  - Async (Event)           — fire-and-forget; no response payload is returned.
  - Sync (RequestResponse)  — waits for the function to return; parses the
                              response payload as JSON.

Both modes propagate function-level errors by checking the "FunctionError"
field in the boto3 response and raising a RuntimeError with the error detail
so callers don't silently swallow Lambda execution failures.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class LambdaInvokeClient:
    """Wraps a boto3 Lambda client for cross-function invocations."""

    def __init__(self, lambda_client: Any) -> None:
        """
        Args:
            lambda_client: An initialised boto3 Lambda client
                           (boto3.client("lambda")).  Injected so tests can
                           supply a mock.
        """
        self._client = lambda_client

    # ------------------------------------------------------------------
    # Async invocation
    # ------------------------------------------------------------------

    def invoke_async(self, function_name: str, payload: dict) -> None:
        """Invoke *function_name* asynchronously (fire-and-forget).

        The Lambda service queues the event and returns immediately with
        HTTP 202.  No response payload is available.

        Args:
            function_name: Lambda function name, ARN, or partial ARN.
            payload:       JSON-serialisable dict sent as the event body.

        Raises:
            Exception: If boto3 raises a transport/service error (e.g.
                       ResourceNotFoundException, TooManyRequestsException).
        """
        try:
            response = self._client.invoke(
                FunctionName=function_name,
                InvocationType="Event",
                Payload=json.dumps(payload).encode(),
            )
            status_code: int = response.get("StatusCode", 0)
            # HTTP 202 is the success code for async invocations.
            if status_code != 202:
                logger.warning(
                    "Unexpected status code for async Lambda invocation",
                    extra={"function": function_name, "status_code": status_code},
                )
            else:
                logger.debug(
                    "Async Lambda invocation queued",
                    extra={"function": function_name},
                )
        except Exception:
            logger.exception(
                "LambdaInvokeClient invoke_async failed",
                extra={"function": function_name},
            )
            raise

    # ------------------------------------------------------------------
    # Synchronous invocation
    # ------------------------------------------------------------------

    def invoke_sync(self, function_name: str, payload: dict) -> dict:
        """Invoke *function_name* synchronously and return the parsed response.

        Waits for the function to complete (up to the configured timeout, max
        15 minutes) and deserialises the response payload as JSON.

        Args:
            function_name: Lambda function name, ARN, or partial ARN.
            payload:       JSON-serialisable dict sent as the event body.

        Returns:
            Parsed JSON response payload from the invoked function.

        Raises:
            RuntimeError: If the Lambda function itself errors (FunctionError
                          is set in the response), with the error detail included.
            Exception:    If boto3 raises a transport/service error.
        """
        try:
            response = self._client.invoke(
                FunctionName=function_name,
                InvocationType="RequestResponse",
                Payload=json.dumps(payload).encode(),
            )

            # Check for function-level errors (distinct from transport errors).
            # FunctionError is set to "Handled" or "Unhandled" on failure.
            function_error: str | None = response.get("FunctionError")
            raw_payload: bytes = response["Payload"].read()

            if function_error:
                error_detail: str = raw_payload.decode("utf-8")
                logger.error(
                    "Synchronous Lambda invocation returned a function error",
                    extra={
                        "function": function_name,
                        "function_error": function_error,
                        "detail": error_detail,
                    },
                )
                raise RuntimeError(
                    f"Lambda function '{function_name}' failed "
                    f"({function_error}): {error_detail}"
                )

            result: dict = json.loads(raw_payload)
            logger.debug(
                "Synchronous Lambda invocation succeeded",
                extra={"function": function_name},
            )
            return result

        except RuntimeError:
            # Already logged above — re-raise without double-logging.
            raise
        except Exception:
            logger.exception(
                "LambdaInvokeClient invoke_sync failed",
                extra={"function": function_name},
            )
            raise
