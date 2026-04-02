"""
DynamoDBClient — thin wrapper around a boto3 DynamoDB Table resource.

The constructor accepts the resource (not the low-level client) so that tests
can inject a mock resource without patching boto3 globally.

Access patterns covered:
  - Persona: PK=USER#<user_id>, SK=VERSION#<version> (latest resolved by query + reverse sort)
  - Decision: PK=DECISION#<decision_id>, GSI verdict-timestamp-index
  - Pattern: PK=PATTERN#<pattern_name>
  - AMLRiskScore: PK=AML#<user_id>
  - InvestigationCase: PK=CASE#<case_id>, GSI userId-status-index, status-openedAt-index
"""

import logging
from typing import Any

import boto3
from boto3.dynamodb.conditions import Attr, Key
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class DynamoDBClient:
    """Wraps a boto3 DynamoDB service resource for fraud-detection table operations."""

    def __init__(self, dynamodb_resource: Any) -> None:
        """
        Args:
            dynamodb_resource: A boto3 DynamoDB *resource* object
                               (boto3.resource("dynamodb")), not the low-level client.
                               Injected so tests can supply a mock.
        """
        self._resource = dynamodb_resource

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _table(self, table_name: str) -> Any:
        """Return a Table resource for the given table name."""
        return self._resource.Table(table_name)

    # ------------------------------------------------------------------
    # Persona
    # ------------------------------------------------------------------

    def get_persona(self, table_name: str, user_id: str) -> dict | None:
        """Return the latest persona for *user_id*, or None if no record exists.

        Items are stored with PK=USER#<user_id> and SK=VERSION#<version>.
        Querying in reverse order and taking the first result gives the latest.
        """
        table = self._table(table_name)
        try:
            response = table.query(
                KeyConditionExpression=Key("PK").eq(f"USER#{user_id}") & Key("SK").begins_with("VERSION#"),
                ScanIndexForward=False,  # descending — latest version first
                Limit=1,
            )
            items = response.get("Items", [])
            return items[0] if items else None
        except ClientError:
            logger.exception(
                "DynamoDB get_persona failed",
                extra={"table": table_name, "user_id": user_id},
            )
            raise

    def put_persona(self, table_name: str, persona: dict) -> None:
        """Write a persona item.  The caller is responsible for setting PK/SK."""
        table = self._table(table_name)
        try:
            table.put_item(Item=persona)
        except ClientError:
            logger.exception(
                "DynamoDB put_persona failed",
                extra={"table": table_name},
            )
            raise

    # ------------------------------------------------------------------
    # Fraud decision
    # ------------------------------------------------------------------

    def put_decision(self, table_name: str, decision: dict) -> None:
        """Write a fraud decision.  The item should carry a 'ttl' attribute for
        automatic expiry so that the table does not grow unbounded."""
        table = self._table(table_name)
        try:
            table.put_item(Item=decision)
        except ClientError:
            logger.exception(
                "DynamoDB put_decision failed",
                extra={"table": table_name},
            )
            raise

    def get_decisions_by_verdict(
        self,
        table_name: str,
        verdict: str,
        since_timestamp: str,
        index_name: str = "verdict-timestamp-index",
    ) -> list[dict]:
        """Query the verdict-timestamp GSI for decisions with *verdict* since
        *since_timestamp* (ISO-8601 string).

        Args:
            table_name: DynamoDB table name.
            verdict:    e.g. "FRAUD", "LEGITIMATE", "REVIEW".
            since_timestamp: Lower-bound timestamp (inclusive), ISO-8601.
            index_name: GSI name (default: verdict-timestamp-index).

        Returns:
            List of decision items, sorted ascending by timestamp.
        """
        table = self._table(table_name)
        try:
            response = table.query(
                IndexName=index_name,
                KeyConditionExpression=(
                    Key("verdict").eq(verdict) & Key("timestamp").gte(since_timestamp)
                ),
                ScanIndexForward=True,
            )
            return response.get("Items", [])
        except ClientError:
            logger.exception(
                "DynamoDB get_decisions_by_verdict failed",
                extra={"table": table_name, "verdict": verdict, "since": since_timestamp},
            )
            raise

    # ------------------------------------------------------------------
    # Fraud pattern
    # ------------------------------------------------------------------

    def get_pattern(self, table_name: str, pattern_name: str) -> dict | None:
        """Fetch a single fraud pattern by name, or None if absent."""
        table = self._table(table_name)
        try:
            response = table.get_item(
                Key={"PK": f"PATTERN#{pattern_name}"},
            )
            return response.get("Item")  # None when key does not exist
        except ClientError:
            logger.exception(
                "DynamoDB get_pattern failed",
                extra={"table": table_name, "pattern_name": pattern_name},
            )
            raise

    def put_pattern(self, table_name: str, pattern: dict) -> None:
        """Create or overwrite a fraud pattern."""
        table = self._table(table_name)
        try:
            table.put_item(Item=pattern)
        except ClientError:
            logger.exception(
                "DynamoDB put_pattern failed",
                extra={"table": table_name},
            )
            raise

    def scan_patterns(self, table_name: str, status: str = "ACTIVE") -> list[dict]:
        """Return all patterns whose 'status' attribute equals *status*.

        Full-table scan — acceptable for a small configuration table.  Uses a
        filter expression so that only the requested status is returned.
        Handles DynamoDB pagination automatically.
        """
        table = self._table(table_name)
        try:
            items: list[dict] = []
            scan_kwargs: dict[str, Any] = {
                "FilterExpression": Attr("status").eq(status),
            }
            while True:
                response = table.scan(**scan_kwargs)
                items.extend(response.get("Items", []))
                last_key = response.get("LastEvaluatedKey")
                if not last_key:
                    break
                scan_kwargs["ExclusiveStartKey"] = last_key
            return items
        except ClientError:
            logger.exception(
                "DynamoDB scan_patterns failed",
                extra={"table": table_name, "status": status},
            )
            raise

    # ------------------------------------------------------------------
    # AML risk score
    # ------------------------------------------------------------------

    def get_aml_risk_score(self, table_name: str, user_id: str) -> dict | None:
        """Fetch the AML risk score record for *user_id*, or None."""
        table = self._table(table_name)
        try:
            response = table.get_item(Key={"PK": f"AML#{user_id}"})
            return response.get("Item")
        except ClientError:
            logger.exception(
                "DynamoDB get_aml_risk_score failed",
                extra={"table": table_name, "user_id": user_id},
            )
            raise

    def put_aml_risk_score(self, table_name: str, score: dict) -> None:
        """Create or overwrite an AML risk score record."""
        table = self._table(table_name)
        try:
            table.put_item(Item=score)
        except ClientError:
            logger.exception(
                "DynamoDB put_aml_risk_score failed",
                extra={"table": table_name},
            )
            raise

    # ------------------------------------------------------------------
    # Investigation case
    # ------------------------------------------------------------------

    def get_investigation_case(self, table_name: str, case_id: str) -> dict | None:
        """Fetch a single investigation case by *case_id*, or None."""
        table = self._table(table_name)
        try:
            response = table.get_item(Key={"PK": f"CASE#{case_id}"})
            return response.get("Item")
        except ClientError:
            logger.exception(
                "DynamoDB get_investigation_case failed",
                extra={"table": table_name, "case_id": case_id},
            )
            raise

    def put_investigation_case(self, table_name: str, case: dict) -> None:
        """Create or overwrite an investigation case."""
        table = self._table(table_name)
        try:
            table.put_item(Item=case)
        except ClientError:
            logger.exception(
                "DynamoDB put_investigation_case failed",
                extra={"table": table_name},
            )
            raise

    def query_investigations_by_user(
        self,
        table_name: str,
        user_id: str,
        status: str | None = None,
        index_name: str = "userId-status-index",
    ) -> list[dict]:
        """Return all investigation cases for *user_id*, optionally filtered by
        *status*, using the userId-status GSI.

        Args:
            table_name: DynamoDB table name.
            user_id:    The user whose cases are queried.
            status:     Optional status filter (e.g. "OPEN", "CLOSED").
            index_name: GSI name (default: userId-status-index).
        """
        table = self._table(table_name)
        try:
            key_condition = Key("userId").eq(user_id)
            if status is not None:
                key_condition = key_condition & Key("status").eq(status)

            response = table.query(
                IndexName=index_name,
                KeyConditionExpression=key_condition,
            )
            return response.get("Items", [])
        except ClientError:
            logger.exception(
                "DynamoDB query_investigations_by_user failed",
                extra={"table": table_name, "user_id": user_id, "status": status},
            )
            raise

    def query_open_investigations(
        self,
        table_name: str,
        index_name: str = "status-openedAt-index",
    ) -> list[dict]:
        """Return all cases with status=OPEN ordered by openedAt ascending,
        using the status-openedAt GSI.  Handles pagination automatically.

        Args:
            table_name: DynamoDB table name.
            index_name: GSI name (default: status-openedAt-index).
        """
        table = self._table(table_name)
        try:
            items: list[dict] = []
            query_kwargs: dict[str, Any] = {
                "IndexName": index_name,
                "KeyConditionExpression": Key("status").eq("OPEN"),
                "ScanIndexForward": True,  # oldest open cases first
            }
            while True:
                response = table.query(**query_kwargs)
                items.extend(response.get("Items", []))
                last_key = response.get("LastEvaluatedKey")
                if not last_key:
                    break
                query_kwargs["ExclusiveStartKey"] = last_key
            return items
        except ClientError:
            logger.exception(
                "DynamoDB query_open_investigations failed",
                extra={"table": table_name},
            )
            raise
