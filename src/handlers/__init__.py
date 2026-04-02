"""
src.handlers — Lambda handler entry points for the fraud detection agent.

Each handler module initialises its AWS clients at module level (outside the
handler function) so they are reused across warm Lambda invocations.

Trigger mapping:
  sentinel_handler           — Kinesis stream (batch size 10)
  swarm_orchestrator_handler — async Lambda invocation from Sentinel
  fraud_analyst_handler      — sync Lambda invocation from the orchestrator
  aml_specialist_handler     — sync Lambda invocation from the orchestrator
  pattern_discovery_handler  — EventBridge daily cron (06:00 UTC)
  archaeologist_handler      — EventBridge weekly cron (Sun 03:00 UTC)
"""

from src.handlers.sentinel_handler import handler as sentinel_handler
from src.handlers.swarm_orchestrator_handler import handler as swarm_orchestrator_handler
from src.handlers.fraud_analyst_handler import handler as fraud_analyst_handler
from src.handlers.aml_specialist_handler import handler as aml_specialist_handler
from src.handlers.pattern_discovery_handler import handler as pattern_discovery_handler
from src.handlers.archaeologist_handler import handler as archaeologist_handler

__all__ = [
    "sentinel_handler",
    "swarm_orchestrator_handler",
    "fraud_analyst_handler",
    "aml_specialist_handler",
    "pattern_discovery_handler",
    "archaeologist_handler",
]
