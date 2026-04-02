import * as path from 'path';
import * as cdk from 'aws-cdk-lib';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as kinesis from 'aws-cdk-lib/aws-kinesis';
import * as firehose from 'aws-cdk-lib/aws-kinesisfirehose';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as lambdaEventSources from 'aws-cdk-lib/aws-lambda-event-sources';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import { Construct } from 'constructs';

export interface ComputeStackProps extends cdk.StackProps {
  // Networking
  vpc: ec2.Vpc;
  lambdaSg: ec2.SecurityGroup;
  // Storage
  fraudPersonasTable: dynamodb.Table;
  fraudDecisionsTable: dynamodb.Table;
  fraudPatternsTable: dynamodb.Table;
  amlRiskScoresTable: dynamodb.Table;
  investigationCasesTable: dynamodb.Table;
  dataLakeBucket: s3.Bucket;
  // Cache
  redisEndpoint: string;
  redisPort: string;
  // Search
  opensearchEndpoint: string;
}

export class ComputeStack extends cdk.Stack {
  // Expose all functions so ObservabilityStack can attach alarms/dashboards
  public readonly fraudSentinelFn: lambda.Function;
  public readonly fraudSwarmOrchestratorFn: lambda.Function;
  public readonly fraudAnalystFn: lambda.Function;
  public readonly amlSpecialistFn: lambda.Function;
  public readonly fraudPatternDiscoveryFn: lambda.Function;
  public readonly fraudArchaeologistFn: lambda.Function;

  constructor(scope: Construct, id: string, props: ComputeStackProps) {
    super(scope, id, props);

    const {
      vpc,
      lambdaSg,
      fraudPersonasTable,
      fraudDecisionsTable,
      fraudPatternsTable,
      amlRiskScoresTable,
      investigationCasesTable,
      dataLakeBucket,
      redisEndpoint,
      redisPort,
      opensearchEndpoint,
    } = props;

    // ----------------------------------------------------------------
    // Kinesis Data Stream — ingest point for all transaction events
    // Shard count 1 is sufficient for <1 MB/s; scale up for production
    // ----------------------------------------------------------------
    const transactionStream = new kinesis.Stream(this, 'FraudTransactionStream', {
      streamName: 'fraud-transactions',
      shardCount: 1,
      // 24-hour retention ensures no data loss during Lambda outages
      retentionPeriod: cdk.Duration.hours(24),
      encryption: kinesis.StreamEncryption.MANAGED,
    });

    // ----------------------------------------------------------------
    // IAM — shared Bedrock InvokeModel policy attached to all Lambdas
    // Bedrock does not support resource-level ARNs for InvokeModel yet,
    // so the wildcard on the resource is required by the service.
    // ----------------------------------------------------------------
    const bedrockInvokePolicy = new iam.PolicyStatement({
      sid: 'BedrockInvokeModel',
      effect: iam.Effect.ALLOW,
      actions: ['bedrock:InvokeModel', 'bedrock:InvokeModelWithResponseStream'],
      // Bedrock does not support resource-level restrictions for InvokeModel
      resources: ['*'],
    });

    // ----------------------------------------------------------------
    // Common environment variables injected into every Lambda
    // ----------------------------------------------------------------
    const sharedEnv: Record<string, string> = {
      FRAUD_PERSONAS_TABLE: fraudPersonasTable.tableName,
      FRAUD_DECISIONS_TABLE: fraudDecisionsTable.tableName,
      FRAUD_PATTERNS_TABLE: fraudPatternsTable.tableName,
      AML_RISK_SCORES_TABLE: amlRiskScoresTable.tableName,
      INVESTIGATION_CASES_TABLE: investigationCasesTable.tableName,
      OPENSEARCH_ENDPOINT: opensearchEndpoint,
      REDIS_HOST: redisEndpoint,
      REDIS_PORT: redisPort,
      // Bedrock is a global service but the SDK needs a region hint
      BEDROCK_REGION: this.region,
    };

    // ----------------------------------------------------------------
    // VPC config reused by all VPC-attached Lambdas
    // ----------------------------------------------------------------
    const vpcConfig: lambda.FunctionProps['vpc'] = vpc;
    const vpcSubnets: lambda.FunctionProps['vpcSubnets'] = {
      subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
    };
    const securityGroups: lambda.FunctionProps['securityGroups'] = [lambdaSg];

    // Helper — create a CloudWatch log group with 90-day retention
    const makeLogGroup = (name: string): logs.LogGroup =>
      new logs.LogGroup(this, `${name}LogGroup`, {
        logGroupName: `/aws/lambda/${name}`,
        retention: logs.RetentionDays.THREE_MONTHS,
        removalPolicy: cdk.RemovalPolicy.DESTROY,
      });

    // ----------------------------------------------------------------
    // 1. fraud-sentinel — real-time triage; Kinesis-triggered, VPC
    // ----------------------------------------------------------------
    const sentinelLogGroup = makeLogGroup('fraud-sentinel');
    this.fraudSentinelFn = new lambda.Function(this, 'FraudSentinelFn', {
      functionName: 'fraud-sentinel',
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'handlers.sentinel.handler',
      // Source code lives in ../src relative to the cdk/ directory
      code: lambda.Code.fromAsset(path.join(__dirname, '../../src')),
      memorySize: 512,
      timeout: cdk.Duration.seconds(30),
      vpc: vpcConfig,
      vpcSubnets,
      securityGroups,
      environment: {
        ...sharedEnv,
        DATA_LAKE_BUCKET: dataLakeBucket.bucketName,
        TRANSACTION_STREAM_NAME: transactionStream.streamName,
      },
      logGroup: sentinelLogGroup,
      // Tracing helps attribute latency across Kinesis → Lambda → DynamoDB hops
      tracing: lambda.Tracing.ACTIVE,
    });

    // Kinesis trigger: batch 10 records, start from latest position
    this.fraudSentinelFn.addEventSource(
      new lambdaEventSources.KinesisEventSource(transactionStream, {
        batchSize: 10,
        startingPosition: lambda.StartingPosition.LATEST,
        // Bisect on error to isolate poison-pill records rather than stalling the shard
        bisectBatchOnError: true,
        // Retry 3 times before sending to DLQ (not configured here — add SQS DLQ for prod)
        retryAttempts: 3,
      }),
    );

    // ----------------------------------------------------------------
    // 2. fraud-swarm-orchestrator — routes decisions to specialists, VPC
    // ----------------------------------------------------------------
    const orchestratorLogGroup = makeLogGroup('fraud-swarm-orchestrator');
    this.fraudSwarmOrchestratorFn = new lambda.Function(this, 'FraudSwarmOrchestratorFn', {
      functionName: 'fraud-swarm-orchestrator',
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'handlers.orchestrator.handler',
      code: lambda.Code.fromAsset(path.join(__dirname, '../../src')),
      memorySize: 512,
      timeout: cdk.Duration.seconds(15),
      vpc: vpcConfig,
      vpcSubnets,
      securityGroups,
      environment: sharedEnv,
      logGroup: orchestratorLogGroup,
      tracing: lambda.Tracing.ACTIVE,
    });

    // ----------------------------------------------------------------
    // 3. fraud-analyst — deep transaction analysis, VPC
    // ----------------------------------------------------------------
    const analystLogGroup = makeLogGroup('fraud-analyst');
    this.fraudAnalystFn = new lambda.Function(this, 'FraudAnalystFn', {
      functionName: 'fraud-analyst',
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'handlers.analyst.handler',
      code: lambda.Code.fromAsset(path.join(__dirname, '../../src')),
      memorySize: 1024,
      timeout: cdk.Duration.seconds(60),
      vpc: vpcConfig,
      vpcSubnets,
      securityGroups,
      environment: sharedEnv,
      logGroup: analystLogGroup,
      tracing: lambda.Tracing.ACTIVE,
    });

    // ----------------------------------------------------------------
    // 4. aml-specialist — AML/CTF risk scoring, VPC
    // ----------------------------------------------------------------
    const amlLogGroup = makeLogGroup('aml-specialist');
    this.amlSpecialistFn = new lambda.Function(this, 'AmlSpecialistFn', {
      functionName: 'aml-specialist',
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'handlers.aml.handler',
      code: lambda.Code.fromAsset(path.join(__dirname, '../../src')),
      memorySize: 1024,
      timeout: cdk.Duration.seconds(90),
      vpc: vpcConfig,
      vpcSubnets,
      securityGroups,
      environment: sharedEnv,
      logGroup: amlLogGroup,
      tracing: lambda.Tracing.ACTIVE,
    });

    // ----------------------------------------------------------------
    // 5. fraud-pattern-discovery — nightly batch ML job, NO VPC
    // No VPC intentionally: avoids ENI cold-start penalty for long-running
    // batch jobs that don't need Redis or OpenSearch direct access
    // ----------------------------------------------------------------
    const patternDiscoveryLogGroup = makeLogGroup('fraud-pattern-discovery');
    this.fraudPatternDiscoveryFn = new lambda.Function(this, 'FraudPatternDiscoveryFn', {
      functionName: 'fraud-pattern-discovery',
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'handlers.pattern_discovery.handler',
      code: lambda.Code.fromAsset(path.join(__dirname, '../../src')),
      memorySize: 2048,
      timeout: cdk.Duration.seconds(300),
      // Intentionally no VPC — batch job uses DynamoDB and S3 via gateway endpoints
      environment: {
        ...sharedEnv,
        DATA_LAKE_BUCKET: dataLakeBucket.bucketName,
      },
      logGroup: patternDiscoveryLogGroup,
      tracing: lambda.Tracing.ACTIVE,
    });

    // EventBridge rule: daily at 06:00 UTC (off-peak hours for batch workload)
    const patternDiscoveryRule = new events.Rule(this, 'PatternDiscoverySchedule', {
      ruleName: 'fraud-pattern-discovery-daily',
      description: 'Triggers fraud-pattern-discovery at 06:00 UTC daily',
      schedule: events.Schedule.cron({ hour: '6', minute: '0' }),
    });
    patternDiscoveryRule.addTarget(new targets.LambdaFunction(this.fraudPatternDiscoveryFn));

    // ----------------------------------------------------------------
    // 6. fraud-archaeologist — weekly historical analysis, NO VPC
    // ----------------------------------------------------------------
    const archaeologistLogGroup = makeLogGroup('fraud-archaeologist');
    this.fraudArchaeologistFn = new lambda.Function(this, 'FraudArchaeologistFn', {
      functionName: 'fraud-archaeologist',
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'handlers.archaeologist.handler',
      code: lambda.Code.fromAsset(path.join(__dirname, '../../src')),
      memorySize: 2048,
      // 15-minute timeout for deep S3 historical scan
      timeout: cdk.Duration.seconds(900),
      environment: {
        ...sharedEnv,
        DATA_LAKE_BUCKET: dataLakeBucket.bucketName,
      },
      logGroup: archaeologistLogGroup,
      tracing: lambda.Tracing.ACTIVE,
    });

    // EventBridge rule: weekly Sunday at 03:00 UTC (lowest traffic window)
    const archaeologistRule = new events.Rule(this, 'ArchaeologistSchedule', {
      ruleName: 'fraud-archaeologist-weekly',
      description: 'Triggers fraud-archaeologist at 03:00 UTC every Sunday',
      schedule: events.Schedule.cron({ hour: '3', minute: '0', weekDay: 'SUN' }),
    });
    archaeologistRule.addTarget(new targets.LambdaFunction(this.fraudArchaeologistFn));

    // ----------------------------------------------------------------
    // IAM Grants — least-privilege per function
    // ----------------------------------------------------------------

    // All functions can invoke Bedrock
    const allFunctions = [
      this.fraudSentinelFn,
      this.fraudSwarmOrchestratorFn,
      this.fraudAnalystFn,
      this.amlSpecialistFn,
      this.fraudPatternDiscoveryFn,
      this.fraudArchaeologistFn,
    ];
    for (const fn of allFunctions) {
      fn.addToRolePolicy(bedrockInvokePolicy);
    }

    // fraud-sentinel: reads from Kinesis stream (event source does this implicitly,
    // but explicit grant ensures the policy survives event-source removal)
    transactionStream.grantRead(this.fraudSentinelFn);
    // sentinel writes decisions and reads personas
    fraudDecisionsTable.grantReadWriteData(this.fraudSentinelFn);
    fraudPersonasTable.grantReadData(this.fraudSentinelFn);

    // sentinel invokes orchestrator asynchronously (InvokeFunction only — no management)
    this.fraudSentinelFn.addToRolePolicy(
      new iam.PolicyStatement({
        sid: 'InvokeOrchestrator',
        effect: iam.Effect.ALLOW,
        actions: ['lambda:InvokeFunction'],
        resources: [this.fraudSwarmOrchestratorFn.functionArn],
      }),
    );

    // orchestrator routes to analyst and specialist
    this.fraudSwarmOrchestratorFn.addToRolePolicy(
      new iam.PolicyStatement({
        sid: 'InvokeSpecialists',
        effect: iam.Effect.ALLOW,
        actions: ['lambda:InvokeFunction'],
        resources: [
          this.fraudAnalystFn.functionArn,
          this.amlSpecialistFn.functionArn,
        ],
      }),
    );
    fraudDecisionsTable.grantReadWriteData(this.fraudSwarmOrchestratorFn);
    fraudPersonasTable.grantReadWriteData(this.fraudSwarmOrchestratorFn);

    // fraud-analyst: reads patterns, writes decisions and personas
    fraudPatternsTable.grantReadData(this.fraudAnalystFn);
    fraudDecisionsTable.grantReadWriteData(this.fraudAnalystFn);
    fraudPersonasTable.grantReadWriteData(this.fraudAnalystFn);

    // aml-specialist: reads personas, writes AML scores and investigation cases
    fraudPersonasTable.grantReadData(this.amlSpecialistFn);
    amlRiskScoresTable.grantReadWriteData(this.amlSpecialistFn);
    investigationCasesTable.grantReadWriteData(this.amlSpecialistFn);

    // pattern-discovery: reads historical decisions, writes discovered patterns
    fraudDecisionsTable.grantReadData(this.fraudPatternDiscoveryFn);
    fraudPatternsTable.grantReadWriteData(this.fraudPatternDiscoveryFn);
    dataLakeBucket.grantRead(this.fraudPatternDiscoveryFn);

    // archaeologist: read-only access to S3 data lake and all tables for historical analysis
    dataLakeBucket.grantRead(this.fraudArchaeologistFn);
    fraudDecisionsTable.grantReadData(this.fraudArchaeologistFn);
    fraudPersonasTable.grantReadData(this.fraudArchaeologistFn);
    fraudPatternsTable.grantReadData(this.fraudArchaeologistFn);
    amlRiskScoresTable.grantReadData(this.fraudArchaeologistFn);
    investigationCasesTable.grantReadData(this.fraudArchaeologistFn);

    // ----------------------------------------------------------------
    // Kinesis Data Firehose — streams Kinesis records to S3 data lake
    // ----------------------------------------------------------------
    const firehoseRole = new iam.Role(this, 'FirehoseDeliveryRole', {
      roleName: 'fraud-detection-firehose-role',
      assumedBy: new iam.ServicePrincipal('firehose.amazonaws.com'),
      description: 'Role for Firehose to deliver transaction events to S3',
    });

    dataLakeBucket.grantReadWrite(firehoseRole);
    transactionStream.grantRead(firehoseRole);

    // Firehose log group for delivery errors
    const firehoseLogGroup = new logs.LogGroup(this, 'FirehoseLogGroup', {
      logGroupName: '/aws/kinesisfirehose/fraud-transactions',
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const firehoseLogStream = new logs.LogStream(this, 'FirehoseLogStream', {
      logGroup: firehoseLogGroup,
      logStreamName: 'S3Delivery',
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    firehoseRole.addToPolicy(
      new iam.PolicyStatement({
        sid: 'FirehoseCloudWatchLogs',
        effect: iam.Effect.ALLOW,
        actions: ['logs:PutLogEvents'],
        // LogStream does not expose a typed ARN property; construct it from components
        resources: [
          `arn:aws:logs:${this.region}:${this.account}:log-group:${firehoseLogGroup.logGroupName}:log-stream:${firehoseLogStream.logStreamName}`,
        ],
      }),
    );

    new firehose.CfnDeliveryStream(this, 'TransactionFirehose', {
      deliveryStreamName: 'fraud-transactions-firehose',
      deliveryStreamType: 'KinesisStreamAsSource',
      kinesisStreamSourceConfiguration: {
        kinesisStreamArn: transactionStream.streamArn,
        roleArn: firehoseRole.roleArn,
      },
      s3DestinationConfiguration: {
        bucketArn: dataLakeBucket.bucketArn,
        roleArn: firehoseRole.roleArn,
        prefix: 'transactions/year=!{timestamp:yyyy}/month=!{timestamp:MM}/day=!{timestamp:dd}/',
        errorOutputPrefix: 'errors/!{firehose:error-output-type}/year=!{timestamp:yyyy}/month=!{timestamp:MM}/',
        bufferingHints: {
          // Buffer for 5 minutes or 128 MB, whichever comes first
          intervalInSeconds: 300,
          sizeInMBs: 128,
        },
        compressionFormat: 'GZIP',
        cloudWatchLoggingOptions: {
          enabled: true,
          logGroupName: firehoseLogGroup.logGroupName,
          logStreamName: firehoseLogStream.logStreamName,
        },
      },
    });

    // ----------------------------------------------------------------
    // Stack Outputs
    // ----------------------------------------------------------------
    new cdk.CfnOutput(this, 'TransactionStreamArn', {
      value: transactionStream.streamArn,
      exportName: 'FraudDetection-TransactionStreamArn',
    });

    new cdk.CfnOutput(this, 'FraudSentinelArn', {
      value: this.fraudSentinelFn.functionArn,
      exportName: 'FraudDetection-FraudSentinelArn',
    });

    new cdk.CfnOutput(this, 'FraudSwarmOrchestratorArn', {
      value: this.fraudSwarmOrchestratorFn.functionArn,
      exportName: 'FraudDetection-FraudSwarmOrchestratorArn',
    });
  }
}
