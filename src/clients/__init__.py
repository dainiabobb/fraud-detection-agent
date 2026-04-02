"""
src/clients — AWS and cache client wrappers for the fraud-detection agent.

All classes use constructor injection so that test suites can supply mock
clients without patching boto3 or redis at the module level.
"""

from src.clients.bedrock_client import BedrockClient
from src.clients.dynamodb_client import DynamoDBClient
from src.clients.lambda_client import LambdaInvokeClient
from src.clients.opensearch_client import OpenSearchClient
from src.clients.redis_client import RedisClient

__all__ = [
    "BedrockClient",
    "DynamoDBClient",
    "LambdaInvokeClient",
    "OpenSearchClient",
    "RedisClient",
]
