import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as elasticache from 'aws-cdk-lib/aws-elasticache';
import { Construct } from 'constructs';

export interface CacheStackProps extends cdk.StackProps {
  vpc: ec2.Vpc;
  redisSg: ec2.SecurityGroup;
}

export class CacheStack extends cdk.Stack {
  public readonly redisEndpoint: string;
  public readonly redisPort: string;

  constructor(scope: Construct, id: string, props: CacheStackProps) {
    super(scope, id, props);

    const { vpc, redisSg } = props;

    // Subnet group — pin Redis to private subnets only
    const subnetGroup = new elasticache.CfnSubnetGroup(this, 'RedisSubnetGroup', {
      description: 'Subnet group for fraud detection Redis cluster',
      subnetIds: vpc.privateSubnets.map((subnet) => subnet.subnetId),
      cacheSubnetGroupName: 'fraud-detection-redis-subnets',
    });

    // Parameter group — override maxmemory policy to LRU so the cache self-manages
    // when memory is full, evicting least-recently-used keys rather than erroring
    const paramGroup = new elasticache.CfnParameterGroup(this, 'RedisParamGroup', {
      cacheParameterGroupFamily: 'redis7',
      description: 'Fraud detection Redis parameter group — LRU eviction',
      properties: {
        'maxmemory-policy': 'allkeys-lru',
      },
    });

    // Single-node Redis cluster (cache.t3.micro) — suitable for dev/staging
    // For production, replace with a CfnReplicationGroup with Multi-AZ enabled
    const redisCluster = new elasticache.CfnCacheCluster(this, 'RedisCluster', {
      clusterName: 'fraud-detection-redis',
      engine: 'redis',
      engineVersion: '7.1',
      cacheNodeType: 'cache.t3.micro',
      numCacheNodes: 1, // Single node — upgrade to replication group for HA
      cacheSubnetGroupName: subnetGroup.cacheSubnetGroupName,
      vpcSecurityGroupIds: [redisSg.securityGroupId],
      cacheParameterGroupName: paramGroup.ref,
      // Enable at-rest encryption for compliance
      snapshotRetentionLimit: 1,
      // Automatic minor version upgrades keep security patches current
      autoMinorVersionUpgrade: true,
    });

    // Ensure subnet group and param group are created before the cluster
    redisCluster.addDependency(subnetGroup);
    redisCluster.addDependency(paramGroup);

    // Expose endpoint details for ComputeStack environment variables
    this.redisEndpoint = redisCluster.attrRedisEndpointAddress;
    this.redisPort = redisCluster.attrRedisEndpointPort;

    new cdk.CfnOutput(this, 'RedisEndpoint', {
      value: this.redisEndpoint,
      exportName: 'FraudDetection-RedisEndpoint',
    });

    new cdk.CfnOutput(this, 'RedisPort', {
      value: this.redisPort,
      exportName: 'FraudDetection-RedisPort',
    });
  }
}
