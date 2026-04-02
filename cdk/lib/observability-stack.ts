import * as cdk from 'aws-cdk-lib';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as actions from 'aws-cdk-lib/aws-cloudwatch-actions';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as sns from 'aws-cdk-lib/aws-sns';
import { Construct } from 'constructs';

export interface ObservabilityStackProps extends cdk.StackProps {
  // Lambda functions
  fraudSentinelFn: lambda.Function;
  fraudSwarmOrchestratorFn: lambda.Function;
  fraudAnalystFn: lambda.Function;
  amlSpecialistFn: lambda.Function;
  fraudPatternDiscoveryFn: lambda.Function;
  fraudArchaeologistFn: lambda.Function;
  // DynamoDB tables
  fraudPersonasTable: dynamodb.Table;
  fraudDecisionsTable: dynamodb.Table;
  fraudPatternsTable: dynamodb.Table;
  amlRiskScoresTable: dynamodb.Table;
  investigationCasesTable: dynamodb.Table;
}

export class ObservabilityStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: ObservabilityStackProps) {
    super(scope, id, props);

    const {
      fraudSentinelFn,
      fraudSwarmOrchestratorFn,
      fraudAnalystFn,
      amlSpecialistFn,
      fraudPatternDiscoveryFn,
      fraudArchaeologistFn,
      fraudPersonasTable,
      fraudDecisionsTable,
      fraudPatternsTable,
      amlRiskScoresTable,
      investigationCasesTable,
    } = props;

    // All functions as an array for widget loops
    const allFunctions: Array<{ fn: lambda.Function; label: string }> = [
      { fn: fraudSentinelFn, label: 'Sentinel' },
      { fn: fraudSwarmOrchestratorFn, label: 'Orchestrator' },
      { fn: fraudAnalystFn, label: 'Analyst' },
      { fn: amlSpecialistFn, label: 'AML Specialist' },
      { fn: fraudPatternDiscoveryFn, label: 'Pattern Discovery' },
      { fn: fraudArchaeologistFn, label: 'Archaeologist' },
    ];

    const allTables: Array<{ table: dynamodb.Table; label: string }> = [
      { table: fraudPersonasTable, label: 'FraudPersonas' },
      { table: fraudDecisionsTable, label: 'FraudDecisions' },
      { table: fraudPatternsTable, label: 'FraudPatterns' },
      { table: amlRiskScoresTable, label: 'AMLRiskScores' },
      { table: investigationCasesTable, label: 'InvestigationCases' },
    ];

    // ----------------------------------------------------------------
    // SNS Topic — alarm notifications (wire subscriptions manually or via ops runbook)
    // ----------------------------------------------------------------
    const alarmTopic = new sns.Topic(this, 'FraudAlarmTopic', {
      topicName: 'fraud-detection-alarms',
      displayName: 'Fraud Detection Alarms',
    });

    // ----------------------------------------------------------------
    // CloudWatch Alarms
    // ----------------------------------------------------------------

    // Alarm 1: sentinel error rate > 1% over 5-minute window
    // Uses a math expression: errors / max(invocations, 1) to avoid division-by-zero
    const sentinelErrors = new cloudwatch.Metric({
      namespace: 'AWS/Lambda',
      metricName: 'Errors',
      dimensionsMap: { FunctionName: fraudSentinelFn.functionName },
      statistic: 'Sum',
      period: cdk.Duration.minutes(5),
      label: 'Sentinel Errors',
    });

    const sentinelInvocations = new cloudwatch.Metric({
      namespace: 'AWS/Lambda',
      metricName: 'Invocations',
      dimensionsMap: { FunctionName: fraudSentinelFn.functionName },
      statistic: 'Sum',
      period: cdk.Duration.minutes(5),
      label: 'Sentinel Invocations',
    });

    const sentinelErrorRate = new cloudwatch.MathExpression({
      expression: '(errors / MAX([errors, invocations])) * 100',
      usingMetrics: {
        errors: sentinelErrors,
        invocations: sentinelInvocations,
      },
      label: 'Sentinel Error Rate (%)',
      period: cdk.Duration.minutes(5),
    });

    const sentinelErrorRateAlarm = new cloudwatch.Alarm(this, 'SentinelErrorRateAlarm', {
      alarmName: 'fraud-sentinel-error-rate-high',
      alarmDescription: 'Sentinel Lambda error rate exceeded 1% — investigate Kinesis consumer health',
      metric: sentinelErrorRate,
      threshold: 1,
      evaluationPeriods: 1,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });
    sentinelErrorRateAlarm.addAlarmAction(new actions.SnsAction(alarmTopic));
    sentinelErrorRateAlarm.addOkAction(new actions.SnsAction(alarmTopic));

    // Alarm 2: sentinel p99 latency > 3000ms
    const sentinelP99Alarm = new cloudwatch.Alarm(this, 'SentinelP99LatencyAlarm', {
      alarmName: 'fraud-sentinel-p99-latency-high',
      alarmDescription: 'Sentinel p99 duration exceeded 3s — possible Bedrock or DynamoDB slowdown',
      metric: new cloudwatch.Metric({
        namespace: 'AWS/Lambda',
        metricName: 'Duration',
        dimensionsMap: { FunctionName: fraudSentinelFn.functionName },
        statistic: 'p99',
        period: cdk.Duration.minutes(5),
        label: 'Sentinel p99 Duration',
      }),
      threshold: 3000,
      evaluationPeriods: 2,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });
    sentinelP99Alarm.addAlarmAction(new actions.SnsAction(alarmTopic));
    sentinelP99Alarm.addOkAction(new actions.SnsAction(alarmTopic));

    // Alarm 3: escalation rate > 10%
    // Uses a custom metric published by the orchestrator Lambda
    const escalationRateAlarm = new cloudwatch.Alarm(this, 'EscalationRateAlarm', {
      alarmName: 'fraud-escalation-rate-high',
      alarmDescription: 'Escalation rate exceeded 10% — possible fraud wave or model degradation',
      metric: new cloudwatch.Metric({
        namespace: 'FraudDetection',
        metricName: 'EscalationRate',
        statistic: 'Average',
        period: cdk.Duration.minutes(5),
        label: 'Escalation Rate (%)',
      }),
      threshold: 10,
      evaluationPeriods: 3,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
      // If no data, assume OK — metric only published when there are transactions
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });
    escalationRateAlarm.addAlarmAction(new actions.SnsAction(alarmTopic));
    escalationRateAlarm.addOkAction(new actions.SnsAction(alarmTopic));

    // ----------------------------------------------------------------
    // CloudWatch Dashboard
    // ----------------------------------------------------------------
    const dashboard = new cloudwatch.Dashboard(this, 'FraudDetectionDashboard', {
      dashboardName: 'Fraud-Detection-Production',
      // Default to last 3 hours — covers an overnight incident window
      defaultInterval: cdk.Duration.hours(3),
    });

    // ---- Row 1: Lambda Error Rates ----
    dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: 'Lambda Error Rates',
        width: 24,
        height: 6,
        left: allFunctions.map(({ fn, label }) =>
          new cloudwatch.Metric({
            namespace: 'AWS/Lambda',
            metricName: 'Errors',
            dimensionsMap: { FunctionName: fn.functionName },
            statistic: 'Sum',
            period: cdk.Duration.minutes(1),
            label,
          }),
        ),
        view: cloudwatch.GraphWidgetView.TIME_SERIES,
        stacked: false,
      }),
    );

    // ---- Row 2: Lambda Duration p50 / p99 ----
    dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: 'Lambda Duration — p50',
        width: 12,
        height: 6,
        left: allFunctions.map(({ fn, label }) =>
          new cloudwatch.Metric({
            namespace: 'AWS/Lambda',
            metricName: 'Duration',
            dimensionsMap: { FunctionName: fn.functionName },
            statistic: 'p50',
            period: cdk.Duration.minutes(1),
            label: `${label} p50`,
          }),
        ),
      }),
      new cloudwatch.GraphWidget({
        title: 'Lambda Duration — p99',
        width: 12,
        height: 6,
        left: allFunctions.map(({ fn, label }) =>
          new cloudwatch.Metric({
            namespace: 'AWS/Lambda',
            metricName: 'Duration',
            dimensionsMap: { FunctionName: fn.functionName },
            statistic: 'p99',
            period: cdk.Duration.minutes(1),
            label: `${label} p99`,
          }),
        ),
      }),
    );

    // ---- Row 3: Lambda Invocations + Cold Starts ----
    dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: 'Lambda Invocations',
        width: 12,
        height: 6,
        left: allFunctions.map(({ fn, label }) =>
          new cloudwatch.Metric({
            namespace: 'AWS/Lambda',
            metricName: 'Invocations',
            dimensionsMap: { FunctionName: fn.functionName },
            statistic: 'Sum',
            period: cdk.Duration.minutes(1),
            label,
          }),
        ),
      }),
      new cloudwatch.GraphWidget({
        title: 'Lambda Cold Starts (InitDuration)',
        width: 12,
        height: 6,
        left: allFunctions.map(({ fn, label }) =>
          new cloudwatch.Metric({
            namespace: 'AWS/Lambda',
            metricName: 'InitDuration',
            dimensionsMap: { FunctionName: fn.functionName },
            statistic: 'Sum',
            period: cdk.Duration.minutes(1),
            label,
          }),
        ),
      }),
    );

    // ---- Row 4: DynamoDB Read/Write Capacity + Throttles ----
    dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: 'DynamoDB Consumed Read Capacity',
        width: 12,
        height: 6,
        left: allTables.map(({ table, label }) =>
          new cloudwatch.Metric({
            namespace: 'AWS/DynamoDB',
            metricName: 'ConsumedReadCapacityUnits',
            dimensionsMap: { TableName: table.tableName },
            statistic: 'Sum',
            period: cdk.Duration.minutes(1),
            label,
          }),
        ),
      }),
      new cloudwatch.GraphWidget({
        title: 'DynamoDB Consumed Write Capacity',
        width: 12,
        height: 6,
        left: allTables.map(({ table, label }) =>
          new cloudwatch.Metric({
            namespace: 'AWS/DynamoDB',
            metricName: 'ConsumedWriteCapacityUnits',
            dimensionsMap: { TableName: table.tableName },
            statistic: 'Sum',
            period: cdk.Duration.minutes(1),
            label,
          }),
        ),
      }),
    );

    // DynamoDB throttle tracking — surfaced separately for immediate visibility
    dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: 'DynamoDB Throttled Requests',
        width: 24,
        height: 4,
        left: allTables.flatMap(({ table, label }) => [
          new cloudwatch.Metric({
            namespace: 'AWS/DynamoDB',
            metricName: 'ReadThrottleEvents',
            dimensionsMap: { TableName: table.tableName },
            statistic: 'Sum',
            period: cdk.Duration.minutes(1),
            label: `${label} Read Throttles`,
          }),
          new cloudwatch.Metric({
            namespace: 'AWS/DynamoDB',
            metricName: 'WriteThrottleEvents',
            dimensionsMap: { TableName: table.tableName },
            statistic: 'Sum',
            period: cdk.Duration.minutes(1),
            label: `${label} Write Throttles`,
          }),
        ]),
      }),
    );

    // ---- Row 5: Custom Business Metrics — Fraud Decision Outcomes ----
    dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: 'Fraud Decision Outcomes',
        width: 12,
        height: 6,
        left: [
          new cloudwatch.Metric({
            namespace: 'FraudDetection',
            metricName: 'EscalationRate',
            statistic: 'Average',
            period: cdk.Duration.minutes(5),
            label: 'Escalation Rate (%)',
          }),
        ],
        right: [
          new cloudwatch.Metric({
            namespace: 'FraudDetection',
            metricName: 'AutoApproveCount',
            statistic: 'Sum',
            period: cdk.Duration.minutes(5),
            label: 'Auto-Approved',
          }),
          new cloudwatch.Metric({
            namespace: 'FraudDetection',
            metricName: 'BlockCount',
            statistic: 'Sum',
            period: cdk.Duration.minutes(5),
            label: 'Blocked',
          }),
        ],
      }),
      new cloudwatch.GraphWidget({
        title: 'Bedrock Token Spend',
        width: 12,
        height: 6,
        left: [
          new cloudwatch.Metric({
            namespace: 'FraudDetection',
            metricName: 'TokenSpend',
            statistic: 'Sum',
            period: cdk.Duration.minutes(5),
            label: 'Total Tokens Used',
          }),
        ],
      }),
    );

    // ---- Row 6: AML-Specific Metrics ----
    dashboard.addWidgets(
      new cloudwatch.GraphWidget({
        title: 'AML Escalations & Investigations',
        width: 12,
        height: 6,
        left: [
          new cloudwatch.Metric({
            namespace: 'FraudDetection',
            metricName: 'AMLEscalationCount',
            statistic: 'Sum',
            period: cdk.Duration.minutes(5),
            label: 'AML Escalations',
          }),
          new cloudwatch.Metric({
            namespace: 'FraudDetection',
            metricName: 'InvestigationCasesOpened',
            statistic: 'Sum',
            period: cdk.Duration.minutes(5),
            label: 'Investigation Cases Opened',
          }),
        ],
      }),
      // Alarm status widget gives NOC a quick red/green overview
      new cloudwatch.AlarmStatusWidget({
        title: 'Active Alarms',
        width: 12,
        height: 6,
        alarms: [
          sentinelErrorRateAlarm,
          sentinelP99Alarm,
          escalationRateAlarm,
        ],
      }),
    );

    // ----------------------------------------------------------------
    // Stack Outputs
    // ----------------------------------------------------------------
    new cdk.CfnOutput(this, 'DashboardUrl', {
      value: `https://${this.region}.console.aws.amazon.com/cloudwatch/home#dashboards:name=Fraud-Detection-Production`,
      exportName: 'FraudDetection-DashboardUrl',
    });

    new cdk.CfnOutput(this, 'AlarmTopicArn', {
      value: alarmTopic.topicArn,
      exportName: 'FraudDetection-AlarmTopicArn',
    });
  }
}
