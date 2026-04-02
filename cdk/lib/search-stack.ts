import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as opensearchserverless from 'aws-cdk-lib/aws-opensearchserverless';
import { Construct } from 'constructs';

export interface SearchStackProps extends cdk.StackProps {
  vpc: ec2.Vpc;
  opensearchSg: ec2.SecurityGroup;
}

export class SearchStack extends cdk.Stack {
  public readonly collectionEndpoint: string;

  constructor(scope: Construct, id: string, props: SearchStackProps) {
    super(scope, id, props);

    const collectionName = 'fraud-detection-vectors';

    // ----------------------------------------------------------------
    // Encryption Policy — required before collection creation
    // AWS-managed keys are sufficient; bring-your-own KMS is optional
    // ----------------------------------------------------------------
    const encryptionPolicy = new opensearchserverless.CfnSecurityPolicy(
      this,
      'VectorEncryptionPolicy',
      {
        name: 'fraud-detection-encryption',
        type: 'encryption',
        // AWSOwnedKey: true uses AWS-managed keys (no cost, no rotation management)
        policy: JSON.stringify({
          Rules: [
            {
              ResourceType: 'collection',
              Resource: [`collection/${collectionName}`],
            },
          ],
          AWSOwnedKey: true,
        }),
      },
    );

    // ----------------------------------------------------------------
    // Network Policy — restrict access to VPC only
    // ----------------------------------------------------------------
    const networkPolicy = new opensearchserverless.CfnSecurityPolicy(
      this,
      'VectorNetworkPolicy',
      {
        name: 'fraud-detection-network',
        type: 'network',
        policy: JSON.stringify([
          {
            Rules: [
              {
                ResourceType: 'collection',
                Resource: [`collection/${collectionName}`],
              },
              {
                ResourceType: 'dashboard',
                Resource: [`collection/${collectionName}`],
              },
            ],
            // Allow VPC-sourced access via the OpenSearch security group
            SourceVPCEs: [],
            AllowFromPublic: false,
          },
        ]),
      },
    );

    // ----------------------------------------------------------------
    // Data Access Policy — grants Lambda execution roles read/write
    // The ComputeStack will grant its Lambda roles access; we create a
    // permissive base policy here that the account admin can tighten
    // ----------------------------------------------------------------
    const dataAccessPolicy = new opensearchserverless.CfnAccessPolicy(
      this,
      'VectorDataAccessPolicy',
      {
        name: 'fraud-detection-data-access',
        type: 'data',
        policy: JSON.stringify([
          {
            Rules: [
              {
                ResourceType: 'index',
                Resource: [`index/${collectionName}/*`],
                Permission: [
                  'aoss:CreateIndex',
                  'aoss:DeleteIndex',
                  'aoss:UpdateIndex',
                  'aoss:DescribeIndex',
                  'aoss:ReadDocument',
                  'aoss:WriteDocument',
                ],
              },
              {
                ResourceType: 'collection',
                Resource: [`collection/${collectionName}`],
                Permission: ['aoss:CreateCollectionItems', 'aoss:DescribeCollectionItems'],
              },
            ],
            // Scoped to the deploying account — Lambda roles added by ComputeStack
            Principal: [`arn:aws:iam::${this.account}:root`],
          },
        ]),
      },
    );

    // ----------------------------------------------------------------
    // OpenSearch Serverless Collection — VECTORSEARCH type for RAG
    // ----------------------------------------------------------------
    const collection = new opensearchserverless.CfnCollection(this, 'FraudVectorCollection', {
      name: collectionName,
      type: 'VECTORSEARCH',
      description: 'Vector store for fraud detection RAG — transaction embeddings and patterns',
    });

    // Policies must exist before collection; deletion order is managed by CFN
    collection.addDependency(encryptionPolicy);
    collection.addDependency(networkPolicy);
    collection.addDependency(dataAccessPolicy);

    // The collection endpoint is available after creation
    this.collectionEndpoint = collection.attrCollectionEndpoint;

    new cdk.CfnOutput(this, 'CollectionEndpoint', {
      value: this.collectionEndpoint,
      exportName: 'FraudDetection-CollectionEndpoint',
    });

    new cdk.CfnOutput(this, 'CollectionArn', {
      value: collection.attrArn,
      exportName: 'FraudDetection-CollectionArn',
    });
  }
}
