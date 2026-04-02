"""
src/services — service layer for the fraud detection pipeline.

Each service class encapsulates one agent tier or background job.  All classes
use constructor injection (config + client instances) so they can be unit-tested
without live AWS connectivity.

Service tiers:
  SentinelService          — Tier 1: fast routing (Haiku + kNN + rules)
  FraudAnalystService      — Tier 2A: deep fraud adjudication (Sonnet)
  AMLSpecialistService     — Tier 2B: AML typology analysis and score update (Sonnet)
  SwarmOrchestratorService — Tier 2 coordinator: fans out to Tier 2 specialists
  PatternDiscoveryService  — Offline: daily cluster analysis of BLOCK decisions (Sonnet)
  ArchaeologistService     — Offline: weekly persona synthesis from 24-mo history (Sonnet)

Example import:
    from src.services import SentinelService, SwarmOrchestratorService
"""

from src.services.sentinel_service import SentinelService
from src.services.fraud_analyst_service import FraudAnalystService
from src.services.aml_specialist_service import AMLSpecialistService
from src.services.swarm_orchestrator_service import SwarmOrchestratorService
from src.services.pattern_discovery_service import PatternDiscoveryService
from src.services.archaeologist_service import ArchaeologistService

__all__ = [
    "SentinelService",
    "FraudAnalystService",
    "AMLSpecialistService",
    "SwarmOrchestratorService",
    "PatternDiscoveryService",
    "ArchaeologistService",
]
