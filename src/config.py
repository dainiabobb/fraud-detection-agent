"""
Central configuration for the fraud detection agent system.

All values are read from environment variables at cold-start.
Secrets (Redis password, API keys) must be fetched separately from
Secrets Manager — this module only handles non-secret configuration.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # ------------------------------------------------------------------ #
    # DynamoDB tables                                                      #
    # ------------------------------------------------------------------ #
    dynamo_table_personas: str
    dynamo_table_decisions: str
    dynamo_table_patterns: str
    dynamo_table_aml_risk: str
    dynamo_table_investigations: str

    # ------------------------------------------------------------------ #
    # OpenSearch                                                           #
    # ------------------------------------------------------------------ #
    opensearch_endpoint: str  # required — no default
    opensearch_index: str

    # ------------------------------------------------------------------ #
    # Redis                                                                #
    # ------------------------------------------------------------------ #
    redis_host: str  # required — no default
    redis_port: int

    # ------------------------------------------------------------------ #
    # AWS / Bedrock                                                        #
    # ------------------------------------------------------------------ #
    bedrock_region: str

    # ------------------------------------------------------------------ #
    # Lambda function names                                                #
    # ------------------------------------------------------------------ #
    fraud_analyst_function_name: str
    aml_specialist_function_name: str

    # ------------------------------------------------------------------ #
    # Caching                                                              #
    # ------------------------------------------------------------------ #
    persona_cache_ttl: int  # seconds

    # ------------------------------------------------------------------ #
    # Embeddings                                                           #
    # ------------------------------------------------------------------ #
    embedding_dimension: int

    # ------------------------------------------------------------------ #
    # Bedrock model IDs                                                    #
    # ------------------------------------------------------------------ #
    sentinel_model_id: str
    sonnet_model_id: str
    titan_embedding_model_id: str

    # ------------------------------------------------------------------ #
    # Decision thresholds                                                  #
    # ------------------------------------------------------------------ #
    # Scores at or above this value are auto-approved (0–1 confidence)
    auto_approve_threshold: float
    # Scores at or above this value are escalated for human review
    escalation_threshold: float
    # AML risk scores (0–100): open an investigation above this value
    aml_investigation_threshold: int
    # AML risk scores (0–100): trigger compliance filing above this value
    aml_compliance_threshold: int

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls) -> "Config":
        """Construct a Config by reading environment variables.

        Raises:
            ValueError: if a required variable (OPENSEARCH_ENDPOINT or
                        REDIS_HOST) is not set.
        """
        opensearch_endpoint = os.environ.get("OPENSEARCH_ENDPOINT", "")
        if not opensearch_endpoint:
            raise ValueError(
                "OPENSEARCH_ENDPOINT environment variable is required but not set."
            )

        redis_host = os.environ.get("REDIS_HOST", "")
        if not redis_host:
            raise ValueError(
                "REDIS_HOST environment variable is required but not set."
            )

        return cls(
            # DynamoDB
            dynamo_table_personas=os.environ.get(
                "DYNAMO_TABLE_PERSONAS", "FraudPersonas"
            ),
            dynamo_table_decisions=os.environ.get(
                "DYNAMO_TABLE_DECISIONS", "FraudDecisions"
            ),
            dynamo_table_patterns=os.environ.get(
                "DYNAMO_TABLE_PATTERNS", "FraudPatterns"
            ),
            dynamo_table_aml_risk=os.environ.get(
                "DYNAMO_TABLE_AML_RISK", "AMLRiskScores"
            ),
            dynamo_table_investigations=os.environ.get(
                "DYNAMO_TABLE_INVESTIGATIONS", "InvestigationCases"
            ),
            # OpenSearch
            opensearch_endpoint=opensearch_endpoint,
            opensearch_index=os.environ.get("OPENSEARCH_INDEX", "fraud-vectors"),
            # Redis
            redis_host=redis_host,
            redis_port=int(os.environ.get("REDIS_PORT", "6379")),
            # AWS / Bedrock
            bedrock_region=os.environ.get("BEDROCK_REGION", "us-east-1"),
            # Lambda function names
            fraud_analyst_function_name=os.environ.get(
                "FRAUD_ANALYST_FUNCTION_NAME", "fraud-analyst"
            ),
            aml_specialist_function_name=os.environ.get(
                "AML_SPECIALIST_FUNCTION_NAME", "aml-specialist"
            ),
            # Caching
            persona_cache_ttl=int(os.environ.get("PERSONA_CACHE_TTL", "3600")),
            # Embeddings
            embedding_dimension=int(os.environ.get("EMBEDDING_DIMENSION", "1536")),
            # Model IDs
            sentinel_model_id=os.environ.get(
                "SENTINEL_MODEL_ID",
                "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            ),
            sonnet_model_id=os.environ.get(
                "SONNET_MODEL_ID",
                "us.anthropic.claude-sonnet-4-6",
            ),
            titan_embedding_model_id=os.environ.get(
                "TITAN_EMBEDDING_MODEL_ID",
                "amazon.titan-embed-text-v2:0",
            ),
            # Decision thresholds
            auto_approve_threshold=float(
                os.environ.get("AUTO_APPROVE_THRESHOLD", "0.85")
            ),
            escalation_threshold=float(
                os.environ.get("ESCALATION_THRESHOLD", "0.75")
            ),
            aml_investigation_threshold=int(
                os.environ.get("AML_INVESTIGATION_THRESHOLD", "50")
            ),
            aml_compliance_threshold=int(
                os.environ.get("AML_COMPLIANCE_THRESHOLD", "80")
            ),
        )
