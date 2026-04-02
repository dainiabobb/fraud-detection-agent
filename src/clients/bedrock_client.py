"""
BedrockClient — thin wrapper around a boto3 bedrock-runtime client.

Constructor injection keeps the class testable without patching boto3 globally.

Model IDs (pinned at module level for easy update):
  - Claude 3.5 Haiku:  anthropic.claude-3-5-haiku-20241022-v1:0
  - Claude 3.5 Sonnet: anthropic.claude-3-5-sonnet-20241022-v2:0
  - Titan Embed V2:    amazon.titan-embed-text-v2:0 (1536-dim output)

All Claude calls use the Bedrock Messages API format (anthropic_version
"bedrock-2023-05-31") with a single user-turn prompt so callers do not need to
build the message structure themselves.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_HAIKU_MODEL_ID = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
_SONNET_MODEL_ID = "us.anthropic.claude-sonnet-4-6"
_TITAN_EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
_ANTHROPIC_VERSION = "bedrock-2023-05-31"


class BedrockClient:
    """Wraps a boto3 bedrock-runtime client for LLM inference and embeddings."""

    def __init__(self, bedrock_runtime_client: Any) -> None:
        """
        Args:
            bedrock_runtime_client: An initialised boto3 bedrock-runtime client.
                                    Injected so tests can supply a mock.
        """
        self._client = bedrock_runtime_client

    # ------------------------------------------------------------------
    # Public LLM methods
    # ------------------------------------------------------------------

    def invoke_haiku(self, prompt: str, max_tokens: int = 1024) -> dict:
        """Invoke Claude 3.5 Haiku for fast, cost-efficient inference.

        Args:
            prompt:     User-facing prompt text.
            max_tokens: Maximum tokens in the model response (default 1024).

        Returns:
            {"text": <response text>, "tokens_used": <total token count>}
        """
        return self._invoke_claude(_HAIKU_MODEL_ID, prompt, max_tokens)

    def invoke_sonnet(self, prompt: str, max_tokens: int = 4096) -> dict:
        """Invoke Claude 3.5 Sonnet for higher-quality, complex inference.

        Args:
            prompt:     User-facing prompt text.
            max_tokens: Maximum tokens in the model response (default 4096).

        Returns:
            {"text": <response text>, "tokens_used": <total token count>}
        """
        return self._invoke_claude(_SONNET_MODEL_ID, prompt, max_tokens)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def get_embedding(self, text: str) -> list[float]:
        """Generate a 1536-dimensional embedding vector using Titan Embeddings V2.

        Args:
            text: Input text to embed.

        Returns:
            List of 1536 floats representing the embedding vector.
        """
        try:
            body = json.dumps({"inputText": text})
            response = self._client.invoke_model(
                modelId=_TITAN_EMBED_MODEL_ID,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            response_body: dict = json.loads(response["body"].read())
            embedding: list[float] = response_body["embedding"]
            logger.debug(
                "Titan embedding generated",
                extra={"model": _TITAN_EMBED_MODEL_ID, "dim": len(embedding)},
            )
            return embedding
        except Exception:
            logger.exception(
                "BedrockClient get_embedding failed",
                extra={"model": _TITAN_EMBED_MODEL_ID},
            )
            raise

    # ------------------------------------------------------------------
    # Private shared Claude invocation logic
    # ------------------------------------------------------------------

    def _invoke_claude(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int,
    ) -> dict:
        """Shared invocation path for all Claude models on Bedrock.

        Builds a single-turn Messages API request, invokes the model, and
        parses the response into a normalised dict.

        Args:
            model_id:   Bedrock model identifier string.
            prompt:     The user message text.
            max_tokens: Token budget for the response.

        Returns:
            {"text": <first content block text>, "tokens_used": <input + output tokens>}

        Raises:
            Exception: Propagated from boto3 after structured error logging.
        """
        try:
            request_body = {
                "anthropic_version": _ANTHROPIC_VERSION,
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            }

            response = self._client.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )

            response_body: dict = json.loads(response["body"].read())

            # Extract the first text content block.
            content_blocks: list[dict] = response_body.get("content", [])
            text = next(
                (block["text"] for block in content_blocks if block.get("type") == "text"),
                "",
            )

            usage: dict = response_body.get("usage", {})
            tokens_used: int = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

            logger.debug(
                "Claude invocation complete",
                extra={"model": model_id, "tokens_used": tokens_used},
            )
            return {"text": text, "tokens_used": tokens_used}

        except Exception:
            logger.exception(
                "BedrockClient _invoke_claude failed",
                extra={"model": model_id},
            )
            raise
