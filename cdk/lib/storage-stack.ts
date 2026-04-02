import * as cdk from 'aws-cdk-lib';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as s3 from 'aws-cdk-lib/aws-s3';
import { Construct } from 'constructs';

export class StorageStack extends cdk.Stack {
  public readonly fraudPersonasTable: dynamodb.Table;
  public readonly fraudDecisionsTable: dynamodb.Table;
  public readonly fraudPatternsTable: dynamodb.Table;
  public readonly amlRiskScoresTable: dynamodb.Table;
  public readonly investigationCasesTable: dynamodb.Table;
  public readonly dataLakeBucket: s3.Bucket;

  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // ----------------------------------------------------------------
    // DynamoDB Tables — all PAY_PER_REQUEST (on-demand) billing
    // ----------------------------------------------------------------

    // 1. FraudPersonas — stores behavioral profile snapshots per user version
    this.fraudPersonasTable = new dynamodb.Table(this, 'FraudPersonasTable', {
      tableName: 'FraudPersonas',
      partitionKey: { name: 'userId', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'version', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      // PITR enabled — persona data is sensitive and must be recoverable
      pointInTimeRecovery: true,
      // TTL allows persona versions to expire automatically
      timeToLiveAttribute: 'ttl',
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    // 2. FraudDecisions — immutable audit log of every verdict (90-day TTL)
    this.fraudDecisionsTable = new dynamodb.Table(this, 'FraudDecisionsTable', {
      tableName: 'FraudDecisions',
      partitionKey: { name: 'transactionId', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'timestamp', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      // 90-day TTL keeps hot storage lean; Firehose delivers cold copy to S3
      timeToLiveAttribute: 'ttl',
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    // GSI: enables verdict-based queries (e.g., "all BLOCKED decisions today")
    this.fraudDecisionsTable.addGlobalSecondaryIndex({
      indexName: 'verdict-timestamp-index',
      partitionKey: { name: 'verdict', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'timestamp', type: dynamodb.AttributeType.STRING },
      projectionType: dynamodb.ProjectionType.ALL,
    });

    // 3. FraudPatterns — reference data for known fraud signatures; no expiry
    this.fraudPatternsTable = new dynamodb.Table(this, 'FraudPatternsTable', {
      tableName: 'FraudPatterns',
      partitionKey: { name: 'patternName', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    // 4. AMLRiskScores — latest AML score per user; overwritten on each run
    this.amlRiskScoresTable = new dynamodb.Table(this, 'AmlRiskScoresTable', {
      tableName: 'AMLRiskScores',
      partitionKey: { name: 'userId', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    // 5. InvestigationCases — permanent record; requires dual GSIs for ops queries
    this.investigationCasesTable = new dynamodb.Table(this, 'InvestigationCasesTable', {
      tableName: 'InvestigationCases',
      partitionKey: { name: 'caseId', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'userId', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      // No TTL — investigation cases are permanent records for compliance
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    // GSI: look up all cases for a specific user and filter by status
    this.investigationCasesTable.addGlobalSecondaryIndex({
      indexName: 'userId-status-index',
      partitionKey: { name: 'userId', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'status', type: dynamodb.AttributeType.STRING },
      projectionType: dynamodb.ProjectionType.ALL,
    });

    // GSI: ops dashboard — list all open cases sorted by open date
    this.investigationCasesTable.addGlobalSecondaryIndex({
      indexName: 'status-openedAt-index',
      partitionKey: { name: 'status', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'openedAt', type: dynamodb.AttributeType.STRING },
      projectionType: dynamodb.ProjectionType.ALL,
    });

    // ----------------------------------------------------------------
    // S3 Data Lake — 24-month transaction log retention
    // ----------------------------------------------------------------
    this.dataLakeBucket = new s3.Bucket(this, 'DataLakeBucket', {
      // Account-scoped name avoids cross-account conflicts
      bucketName: `fraud-detection-datalake-${this.account}`,
      versioned: true,
      // Enforce encryption at rest
      encryption: s3.BucketEncryption.S3_MANAGED,
      // Block all public access — this bucket must never be public
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      // Enforce HTTPS-only access
      enforceSSL: true,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      lifecycleRules: [
        {
          id: 'TransactionLogTiering',
          enabled: true,
          transitions: [
            {
              // Move to S3 Infrequent Access after 90 days (cheaper, same durability)
              storageClass: s3.StorageClass.INFREQUENT_ACCESS,
              transitionAfter: cdk.Duration.days(90),
            },
            {
              // Archive to Glacier after 365 days — retrieval within hours acceptable
              storageClass: s3.StorageClass.GLACIER,
              transitionAfter: cdk.Duration.days(365),
            },
          ],
          // Expire objects after 24 months per data retention policy
          expiration: cdk.Duration.days(730),
          // Clean up non-current versions to avoid runaway storage costs
          noncurrentVersionExpiration: cdk.Duration.days(30),
        },
      ],
    });

    // ----------------------------------------------------------------
    // CloudFormation Outputs
    // ----------------------------------------------------------------
    new cdk.CfnOutput(this, 'FraudPersonasTableName', {
      value: this.fraudPersonasTable.tableName,
      exportName: 'FraudDetection-FraudPersonasTableName',
    });

    new cdk.CfnOutput(this, 'FraudDecisionsTableName', {
      value: this.fraudDecisionsTable.tableName,
      exportName: 'FraudDetection-FraudDecisionsTableName',
    });

    new cdk.CfnOutput(this, 'FraudPatternsTableName', {
      value: this.fraudPatternsTable.tableName,
      exportName: 'FraudDetection-FraudPatternsTableName',
    });

    new cdk.CfnOutput(this, 'AmlRiskScoresTableName', {
      value: this.amlRiskScoresTable.tableName,
      exportName: 'FraudDetection-AmlRiskScoresTableName',
    });

    new cdk.CfnOutput(this, 'InvestigationCasesTableName', {
      value: this.investigationCasesTable.tableName,
      exportName: 'FraudDetection-InvestigationCasesTableName',
    });

    new cdk.CfnOutput(this, 'DataLakeBucketName', {
      value: this.dataLakeBucket.bucketName,
      exportName: 'FraudDetection-DataLakeBucketName',
    });
  }
}
