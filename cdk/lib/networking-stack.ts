import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { Construct } from 'constructs';

export class NetworkingStack extends cdk.Stack {
  public readonly vpc: ec2.Vpc;
  public readonly lambdaSecurityGroup: ec2.SecurityGroup;
  public readonly redisSg: ec2.SecurityGroup;
  public readonly opensearchSg: ec2.SecurityGroup;

  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // VPC with 2 AZs, public + private subnets
    this.vpc = new ec2.Vpc(this, 'FraudDetectionVpc', {
      maxAzs: 2,
      subnetConfiguration: [
        {
          cidrMask: 24,
          name: 'Public',
          subnetType: ec2.SubnetType.PUBLIC,
        },
        {
          cidrMask: 24,
          name: 'Private',
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
        },
      ],
      // NAT gateway per AZ for HA; reduce to 1 for cost savings in non-prod
      natGateways: 1,
    });

    // VPC endpoint for DynamoDB (Gateway type — no cost, no SG required)
    this.vpc.addGatewayEndpoint('DynamoDbEndpoint', {
      service: ec2.GatewayVpcEndpointAwsService.DYNAMODB,
      subnets: [{ subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS }],
    });

    // VPC endpoint for S3 (Gateway type — no cost, no SG required)
    this.vpc.addGatewayEndpoint('S3Endpoint', {
      service: ec2.GatewayVpcEndpointAwsService.S3,
      subnets: [{ subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS }],
    });

    // Security group for Lambda functions
    this.lambdaSecurityGroup = new ec2.SecurityGroup(this, 'LambdaSg', {
      vpc: this.vpc,
      description: 'Security group for fraud detection Lambda functions',
      allowAllOutbound: true, // Lambdas need outbound to call Bedrock, Kinesis, etc.
    });

    // Security group for ElastiCache Redis
    // Only accepts inbound on 6379 from the Lambda SG
    this.redisSg = new ec2.SecurityGroup(this, 'RedisSg', {
      vpc: this.vpc,
      description: 'Security group for ElastiCache Redis cluster',
      allowAllOutbound: false,
    });
    this.redisSg.addIngressRule(
      this.lambdaSecurityGroup,
      ec2.Port.tcp(6379),
      'Allow Redis access from Lambda functions',
    );

    // Security group for OpenSearch Serverless
    // Only accepts inbound HTTPS from the Lambda SG
    this.opensearchSg = new ec2.SecurityGroup(this, 'OpenSearchSg', {
      vpc: this.vpc,
      description: 'Security group for OpenSearch Serverless collection',
      allowAllOutbound: false,
    });
    this.opensearchSg.addIngressRule(
      this.lambdaSecurityGroup,
      ec2.Port.tcp(443),
      'Allow HTTPS access from Lambda functions',
    );

    // Stack outputs for cross-stack references
    new cdk.CfnOutput(this, 'VpcId', {
      value: this.vpc.vpcId,
      exportName: 'FraudDetection-VpcId',
    });

    new cdk.CfnOutput(this, 'LambdaSgId', {
      value: this.lambdaSecurityGroup.securityGroupId,
      exportName: 'FraudDetection-LambdaSgId',
    });

    new cdk.CfnOutput(this, 'RedisSgId', {
      value: this.redisSg.securityGroupId,
      exportName: 'FraudDetection-RedisSgId',
    });

    new cdk.CfnOutput(this, 'OpenSearchSgId', {
      value: this.opensearchSg.securityGroupId,
      exportName: 'FraudDetection-OpenSearchSgId',
    });
  }
}
