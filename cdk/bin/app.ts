#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { CacheStack } from '../lib/cache-stack';
import { ComputeStack } from '../lib/compute-stack';
import { NetworkingStack } from '../lib/networking-stack';
import { ObservabilityStack } from '../lib/observability-stack';
import { SearchStack } from '../lib/search-stack';
import { StorageStack } from '../lib/storage-stack';

const app = new cdk.App();

// ----------------------------------------------------------------
// Environment — target account and region resolved from CDK context
// or from environment variables set by the CI role.
//
// Usage:
//   cdk deploy --all \
//     --context account=123456789012 \
//     --context region=us-east-1
//
// Or set AWS_ACCOUNT_ID and AWS_DEFAULT_REGION in the environment.
// ----------------------------------------------------------------
const env: cdk.Environment = {
  account: app.node.tryGetContext('account') ?? process.env.AWS_ACCOUNT_ID,
  region: app.node.tryGetContext('region') ?? process.env.AWS_DEFAULT_REGION ?? 'us-east-1',
};

// ----------------------------------------------------------------
// Stack 1 — Networking
// No dependencies; must be deployed first.
// ----------------------------------------------------------------
const networkingStack = new NetworkingStack(app, 'FraudDetectionNetworking', {
  env,
  description: 'VPC, security groups, and VPC endpoints for fraud detection',
  // Terminate protection prevents accidental destruction of shared network
  terminationProtection: true,
});

// ----------------------------------------------------------------
// Stack 2 — Storage
// No VPC dependency; can deploy in parallel with Networking.
// ----------------------------------------------------------------
const storageStack = new StorageStack(app, 'FraudDetectionStorage', {
  env,
  description: 'DynamoDB tables and S3 data lake for fraud detection',
  terminationProtection: true,
});

// ----------------------------------------------------------------
// Stack 3 — Cache
// Depends on Networking for VPC and Redis security group.
// ----------------------------------------------------------------
const cacheStack = new CacheStack(app, 'FraudDetectionCache', {
  env,
  description: 'ElastiCache Redis cluster for hot-path persona caching',
  vpc: networkingStack.vpc,
  redisSg: networkingStack.redisSg,
});
// Explicit dependency ensures CFN deploys networking before cache
cacheStack.addDependency(networkingStack);

// ----------------------------------------------------------------
// Stack 4 — Search
// Depends on Networking for VPC and OpenSearch security group.
// ----------------------------------------------------------------
const searchStack = new SearchStack(app, 'FraudDetectionSearch', {
  env,
  description: 'OpenSearch Serverless vector collection for RAG',
  vpc: networkingStack.vpc,
  opensearchSg: networkingStack.opensearchSg,
});
searchStack.addDependency(networkingStack);

// ----------------------------------------------------------------
// Stack 5 — Compute
// Depends on all preceding stacks — must deploy last (before Observability).
// ----------------------------------------------------------------
const computeStack = new ComputeStack(app, 'FraudDetectionCompute', {
  env,
  description: 'Lambda functions, Kinesis stream, and Firehose delivery pipeline',
  // Networking
  vpc: networkingStack.vpc,
  lambdaSg: networkingStack.lambdaSecurityGroup,
  // Storage
  fraudPersonasTable: storageStack.fraudPersonasTable,
  fraudDecisionsTable: storageStack.fraudDecisionsTable,
  fraudPatternsTable: storageStack.fraudPatternsTable,
  amlRiskScoresTable: storageStack.amlRiskScoresTable,
  investigationCasesTable: storageStack.investigationCasesTable,
  dataLakeBucket: storageStack.dataLakeBucket,
  // Cache
  redisEndpoint: cacheStack.redisEndpoint,
  redisPort: cacheStack.redisPort,
  // Search
  opensearchEndpoint: searchStack.collectionEndpoint,
});
computeStack.addDependency(networkingStack);
computeStack.addDependency(storageStack);
computeStack.addDependency(cacheStack);
computeStack.addDependency(searchStack);

// ----------------------------------------------------------------
// Stack 6 — Observability
// Depends only on Compute (for function references) and Storage (for table metrics).
// ----------------------------------------------------------------
const observabilityStack = new ObservabilityStack(app, 'FraudDetectionObservability', {
  env,
  description: 'CloudWatch dashboard, alarms, and SNS alert topic',
  // Lambda functions
  fraudSentinelFn: computeStack.fraudSentinelFn,
  fraudSwarmOrchestratorFn: computeStack.fraudSwarmOrchestratorFn,
  fraudAnalystFn: computeStack.fraudAnalystFn,
  amlSpecialistFn: computeStack.amlSpecialistFn,
  fraudPatternDiscoveryFn: computeStack.fraudPatternDiscoveryFn,
  fraudArchaeologistFn: computeStack.fraudArchaeologistFn,
  // DynamoDB tables
  fraudPersonasTable: storageStack.fraudPersonasTable,
  fraudDecisionsTable: storageStack.fraudDecisionsTable,
  fraudPatternsTable: storageStack.fraudPatternsTable,
  amlRiskScoresTable: storageStack.amlRiskScoresTable,
  investigationCasesTable: storageStack.investigationCasesTable,
});
observabilityStack.addDependency(computeStack);
observabilityStack.addDependency(storageStack);

app.synth();
