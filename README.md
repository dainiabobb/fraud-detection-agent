# Fraud Detection Agent

A three-tiered, RAG-enhanced fraud detection and AML monitoring system on AWS. Transactions flow in via Kinesis, get triaged by Tier 1 (Sentinel/Haiku), then escalated to a Tier 2 agent swarm comprising two specialized analysts: Agent 2A (Fraud Analyst) and Agent 2B (AML Specialist). Tier 3 (Archaeologist) rebuilds behavioral personas weekly and detects AML typologies.

## Architecture Overview

```
                          Kinesis Data Stream
                                │
                    ┌───────────▼───────────┐
                    │   Tier 1: Sentinel    │
                    │   (Haiku / Zero-Shot) │
                    └───┬───────┬───────┬───┘
                        │       │       │
                   APPROVE  ESCALATE  DEFER
                   (>0.85)  (<0.75)  (cold)
                                │
              ┌─────────────────▼─────────────────┐
              │      Swarm Orchestrator            │
              │  (routes by escalation_target)     │
              └────────┬──────────────┬────────────┘
                       │              │
          ┌────────────▼────┐  ┌──────▼───────────┐
          │  Agent 2A:      │  │  Agent 2B:       │
          │  Fraud Analyst  │  │  AML Specialist  │
          │  (Sonnet)       │  │  (Sonnet)        │
          │                 │  │                  │
          │  BLOCK/APPROVE  │  │  Risk Score ++   │
          │  + reasoning    │  │  + Investigation │
          └─────────────────┘  └──────────────────┘

          ┌─────────────────────────────────────────┐
          │  Tier 3: Archaeologist (Weekly Batch)    │
          │  - Persona Synthesis (24mo history)      │
          │  - AML Typology Detection                │
          │  - Pattern Discovery (daily)             │
          └─────────────────────────────────────────┘
```

## Three-Tier Logic Flow

### Tier 1: The Sentinel (Router)
- **Model**: Claude 3.5 Haiku (zero-shot structural)
- **Trigger**: Kinesis Data Stream event
- **Process**:
  1. GeoIP enrichment (MaxMind GeoLite2)
  2. Temporal enrichment (local hour calculation)
  3. Fetch Behavioral Persona (Redis cache -> DynamoDB)
  4. Build embedding via Titan -> kNN search in OpenSearch
  5. **Dual-path routing**:
     - **Fraud path**: Cosine similarity > 0.85 auto-approve, < 0.75 escalate, gray-zone checks discovered patterns then Haiku
     - **AML path**: Structural signal detection (structuring, round-number transfers, high-risk jurisdictions)
  6. Escalation targets: `FRAUD_ONLY`, `AML_ONLY`, or `BOTH`
- **Latency target**: < 3 seconds

### Tier 2: Agent Swarm

#### Agent 2A: Fraud Analyst (Heuristic Specialist)
- **Model**: Claude 3.5 Sonnet (few-shot + RAG context)
- **Focus**: Immediate risk -- stolen credentials, account takeover, rapid-fire spending, geospatial anomalies
- **Decision**: Binary BLOCK or APPROVE with reasoning string
- **Latency**: Seconds

#### Agent 2B: AML Specialist (Relational Specialist)
- **Model**: Claude 3.5 Sonnet (few-shot + deep RAG)
- **Focus**: Structural risk -- structuring, smurfing, layering, round-tripping, U-turn transactions, economic profile mismatch
- **Decision**: Does NOT block transactions. Updates continuous AML risk score (0-100). Opens investigation cases at score > 50. Auto-escalates to compliance at score > 80.
- **Latency**: Up to 90 seconds (deep 6-month RAG lookups)

#### Pattern Discovery (Daily Batch)
- Mines recent fraud decisions across all users
- Discovers emerging attack patterns (clusters of 3+ fraud cases with >70% precision)
- Feeds patterns back to Sentinel for faster Tier 1 detection
- Refines patterns with high false positive rates, retires stale patterns

### Tier 3: The Archaeologist (Optimizer)
- **Model**: Claude 3.5 Sonnet (chain-of-thought)
- **Trigger**: Weekly EventBridge schedule
- **Process**: Analyzes 24 months of S3 logs via Athena
- **Output**: Behavioral Persona (JSON) per user + AML typology detection

## Behavioral Persona

The Archaeologist compresses 24 months of raw logs into a structured JSON per user:

```json
{
  "user_id": "user-123",
  "geo_footprint": [{"city": "Dallas", "state": "TX", "frequency": 0.65}],
  "category_anchors": [{"category": "Groceries", "avg_amount": 120.0, "frequency": "weekly", "std_deviation": 35.0}],
  "velocity": {"daily_txn_count": 4.2, "daily_spend_amount": 185.0, "max_single_txn": 450.0, "hourly_burst_limit": 3},
  "anomaly_history": {"false_positive_triggers": ["travel-to-miami"], "confirmed_fraud_count": 0},
  "ip_footprint": [{"region": "Dallas-Fort Worth, TX", "frequency": 0.65, "typical_asn": "AS7018 AT&T"}],
  "temporal_profile": {"active_hours": [7, 22], "peak_hour": 12, "weekend_ratio": 0.28, "timezone_estimate": "America/Chicago"},
  "aml_profile": {
    "deposit_pattern": {"avg_amount": 850.0, "pct_near_threshold": 0.02},
    "transfer_pattern": {"avg_count": 3, "unique_counterparties": 12, "pct_high_risk_jurisdictions": 0.0},
    "round_trip_score": 0.0,
    "economic_profile_match": true,
    "typology_flags": []
  }
}
```

## Prompting Strategy

| Tier | Model | Strategy | Focus |
|------|-------|----------|-------|
| Sentinel | Haiku | Zero-Shot / Structural | Latency & Routing + AML Signal Detection |
| Fraud Analyst | Sonnet | Few-Shot + RAG Context | Immediate Risk & Accuracy |
| AML Specialist | Sonnet | Few-Shot + Deep RAG (6mo) | Structural Risk & Continuous Scoring |
| Pattern Discovery | Sonnet | Cluster Analysis | Cross-User Intelligence |
| Archaeologist | Sonnet | Chain-of-Thought (10 steps) | Persona Synthesis & AML Typology |

## AWS Tech Stack

| Service | Purpose |
|---------|---------|
| Amazon Kinesis Data Streams | Transaction ingestion |
| Amazon Kinesis Data Firehose | S3 data lake delivery |
| Amazon S3 | Data lake (24mo transaction logs) |
| Amazon DynamoDB | Personas, Decisions, Patterns, AML Risk Scores, Investigation Cases |
| Amazon OpenSearch Serverless | Vector store for RAG (1536-dim Titan Embeddings V2) |
| Amazon ElastiCache (Redis) | Persona cache, rate limiting, pattern cache |
| AWS Lambda | All compute (6 functions) |
| Amazon Bedrock | Claude 3.5 Haiku, Claude 3.5 Sonnet, Titan Embeddings V2 |
| Amazon Athena + Glue | S3 datalake queries (Tier 3) |
| Amazon CloudWatch | Dashboards, alarms, structured logging |
| AWS CDK (TypeScript) | Infrastructure as Code (6 stacks) |

## DynamoDB Tables

| Table | PK | SK | Purpose |
|-------|----|----|---------|
| `FraudPersonas` | `userId` | `version` | Behavioral personas (TTL, PITR) |
| `FraudDecisions` | `transactionId` | `timestamp` | All fraud/approve decisions (90-day TTL) |
| `FraudPatterns` | `patternName` | -- | Discovered attack patterns (no TTL) |
| `AMLRiskScores` | `userId` | -- | Continuous AML risk scores (no TTL) |
| `InvestigationCases` | `caseId` | `userId` | AML investigation cases (permanent) |

### GSIs
- `FraudDecisions.verdict-timestamp-index` -- Efficient "all BLOCK decisions in last 7 days" queries for Pattern Discovery
- `InvestigationCases.userId-status-index` -- "All open cases for user X"
- `InvestigationCases.status-openedAt-index` -- "All OPEN cases sorted by date" for compliance dashboard

## Lambda Functions

| Function | Memory | Timeout | Trigger | VPC |
|----------|--------|---------|---------|-----|
| `fraud-sentinel` | 512MB | 30s | Kinesis (batch 10) | Yes |
| `fraud-swarm-orchestrator` | 512MB | 15s | Async invoke from Sentinel | Yes |
| `fraud-analyst` (2A) | 1024MB | 60s | Invoke from Orchestrator | Yes |
| `aml-specialist` (2B) | 1024MB | 90s | Invoke from Orchestrator | Yes |
| `fraud-pattern-discovery` | 2048MB | 300s | EventBridge daily 06:00 UTC | No |
| `fraud-archaeologist` | 2048MB | 900s | EventBridge weekly Sun 03:00 UTC | No |

## Observability Dashboard

CloudWatch Dashboard `Fraud-Detection-Production`:

1. **Lambda Health** -- Error rates + p50/p99 duration for all 6 functions
2. **Invocations & Cold Starts** -- Per-function invocation counts
3. **DynamoDB Capacity** -- Read/write CU + throttle counts for all 5 tables
4. **Bedrock Performance** -- Haiku/Sonnet/Titan latency + errors + token counts
5. **Fraud Metrics** -- Escalation rate (target <5%), auto-approve rate, block rate, pattern match count, token spend
6. **AML Metrics** -- AML escalation rate, risk score distribution, open investigation cases, cases escalated to compliance

### Custom Metrics (Namespace: `FraudDetection/Metrics`)
- `EscalationRate`, `AutoApproveCount`, `BlockCount`, `TokenSpend`
- `PatternMatchEscalationCount`, `NewPatternsDiscovered`, `PatternsRetired`
- `AMLEscalationCount`, `AMLRiskScoreUpdates`, `InvestigationCasesOpened`, `InvestigationCasesEscalatedToCompliance`

## AML Detection

### Typologies Detected

| Typology | Detection Layer | Description |
|----------|----------------|-------------|
| Structuring | Sentinel + AML Specialist + Archaeologist | Multiple deposits just under $10,000 reporting threshold |
| Smurfing | AML Specialist | Multiple accounts/sources feeding one account |
| Layering | AML Specialist | Complex multi-hop transfer chains obscuring fund origin |
| Round-Tripping | AML Specialist + Archaeologist | Funds leaving and returning to same account via intermediaries |
| U-Turn | Sentinel + AML Specialist | Inbound funds immediately redirected to high-risk jurisdictions |
| Economic Profile Mismatch | AML Specialist + Archaeologist | Transaction volume vs KYC-declared income/occupation |

### AML Risk Score Lifecycle

```
Transaction arrives
       │
  Sentinel detects AML signal (e.g., $9,200 deposit)
       │
  AML Specialist analyzes (deep RAG, 6 months)
       │
  Risk score updated: current_score += score_delta
       │
  ┌────▼────────────────────────────────────────┐
  │  Score < 30  │  30-50     │  50-80    │ >80 │
  │  Normal      │  Elevated  │  Case     │ Auto│
  │              │            │  Opened   │ Esc │
  └─────────────────────────────────────────────┘
```

## End-to-End Testing

### Approach: Hybrid (Option C)
- **Local mocks**: DynamoDB, OpenSearch, Redis -- in-memory Python dicts/lists
- **Real Bedrock**: Actual API calls validate prompt quality against real data
- **Dataset**: IEEE-CIS Fraud Detection (Kaggle) with synthetic AML patterns injected

### Dataset Setup

1. Download IEEE-CIS Fraud Detection dataset from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)
2. Place `train_transaction.csv` and `train_identity.csv` in `data/` directory

### IEEE-CIS Column Mapping

| IEEE-CIS Column | Transaction Field | Transform |
|---|---|---|
| `TransactionID` | `transaction_id` | `str(value)` |
| `card1` | `user_id` | `f"user-{card1}"` (card as user proxy) |
| `TransactionAmt` | `amount` | Direct |
| `ProductCD` | `merchant_category` | W->"Digital Goods", H->"Hotels", C->"Services", S->"Subscriptions", R->"Retail" |
| `addr1` | `merchant_city` | Zip code -> city name lookup |
| `addr2` | `merchant_country` | Country code mapping (87.0->US) |
| `TransactionDT` | `timestamp` | Reference date + timedelta(seconds) |
| `DeviceType` | `channel` | "mobile"->"mobile", "desktop"->"online" |
| `isFraud` | -- | Ground truth label (NOT passed to pipeline) |

### Synthetic AML Injection

AML patterns injected for ~2% of users:

| Pattern | % of Injections | Description |
|---------|----------------|-------------|
| Structuring | 40% | 3-6 deposits of $8,200-$9,800 over 2-4 weeks |
| Smurfing | 20% | 4-8 feeder users sending $2k-$4k within 10 days |
| Layering | 15% | $20k-$50k splits into 3-5 outbound transfers in 24-48hrs |
| Round-Tripping | 10% | $15k-$40k cycle through intermediaries in 30 days |
| Profile Mismatch | 15% | Low-activity user gets $10k-$50k "Consulting" transactions |

### Running E2E Tests

```bash
# Run full pipeline (requires AWS credentials for Bedrock)
pytest tests/e2e/test_runner.py -v -s --tb=short

# Run with smaller sample (cheaper/faster)
pytest tests/e2e/test_runner.py -v -s -k "test_full_pipeline" --sample-size=1000

# View results
cat tests/e2e/results/report.json
```

### Quality Thresholds

| Metric | Minimum |
|--------|---------|
| Fraud F1 Score | >= 0.70 |
| Escalation Rate | <= 10% |
| False Positive Rate | <= 5% |
| AML Structuring Recall | >= 60% |

### Sample E2E Report

```
========== E2E FRAUD DETECTION REPORT ==========
Dataset: IEEE-CIS (5,000 txns, 175 fraud, 42 AML-injected users)
Bedrock calls: 312 Haiku, 89 Sonnet, 5000 Titan Embeddings

FRAUD DETECTION:
  Precision:  0.87  |  Recall: 0.79  |  F1: 0.83
  Auto-approved (Tier 1):  4,622 (92.4%)
  Escalated to Tier 2:       289 (5.8%)
  Blocked:                    89 (1.8%)
  False positives:            13

AML DETECTION:
  Structuring detected:   14/17 (82.4%)
  Smurfing detected:       6/8  (75.0%)
  Layering detected:       5/6  (83.3%)
  Round-tripping detected: 3/4  (75.0%)
  Profile mismatch:        6/7  (85.7%)
  Investigation cases opened: 28
  Cases matching injected users: 24/28 (85.7%)

COST:
  Total tokens: 1,247,000
  Estimated cost: $4.82
  Avg tokens/decision: 249
=================================================
```

## Project Structure

```
fraud-detection-agent/
├── src/
│   ├── config.py                          # Dataclass config from env vars
│   ├── handlers/
│   │   ├── sentinel_handler.py            # Tier 1 -- Kinesis trigger
│   │   ├── fraud_analyst_handler.py       # Tier 2A -- Fraud Analyst
│   │   ├── aml_specialist_handler.py      # Tier 2B -- AML Specialist
│   │   ├── swarm_orchestrator_handler.py  # Tier 2 -- routes to 2A/2B
│   │   ├── pattern_discovery_handler.py   # Tier 2 -- daily pattern discovery
│   │   └── archaeologist_handler.py       # Tier 3 -- weekly persona builder
│   ├── services/                          # Business logic (separated from handlers)
│   ├── models/                            # Pydantic data models
│   ├── clients/                           # AWS service wrappers
│   └── utils/                             # Embedding, GeoIP, sanitizer, metrics
├── cdk/                                   # AWS CDK infrastructure (TypeScript)
│   └── lib/                               # 6 stacks
├── prompts/                               # LLM prompt templates
├── tests/
│   ├── unit/                              # Mocked unit tests
│   ├── integration/                       # Moto-backed integration tests
│   └── e2e/                               # IEEE-CIS dataset + real Bedrock
├── events/                                # Test event payloads
├── data/                                  # Kaggle dataset (gitignored)
└── scripts/                               # Seeding scripts
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| CDK (TypeScript) over SAM | 6 stacks with cross-stack refs -- CDK handles this cleanly |
| Python 3.12 for Lambdas | Matches existing crypto projects |
| Agent swarm over monolithic Arbiter | Fraud (binary) and AML (continuous scoring) have fundamentally different decision models |
| AML never blocks transactions | Regulatory requirement -- blocking tips off the subject ("tipping off") |
| AML risk score as accumulator | Patterns emerge over time; continuous score lets them build until investigation threshold |
| Investigation cases are permanent | Regulatory retention -- compliance team manages lifecycle |
| Dependency injection in all services | E2E swaps in-memory stores while keeping real Bedrock |
| IEEE-CIS + AML injection for E2E | Single unified dataset tests both paths with labeled ground truth |
| Hybrid testing (Option C) | In-memory infra is free/fast; real Bedrock validates prompt quality |

## Getting Started

### Prerequisites
- Python 3.12+
- Node.js 18+ (for CDK)
- AWS CLI configured with Bedrock access (us-east-1)
- Bedrock model access enabled: Claude 3.5 Haiku, Claude 3.5 Sonnet, Titan Embeddings V2

### Setup

```bash
# Install Python dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install CDK dependencies
cd cdk && npm install && cd ..

# Run unit tests
pytest tests/unit/ -v

# Run E2E tests (requires Bedrock credentials + Kaggle dataset)
pytest tests/e2e/test_runner.py -v -s

# Synthesize CDK stacks
cd cdk && npx cdk synth

# Deploy to AWS
cd cdk && npx cdk deploy --all
```
