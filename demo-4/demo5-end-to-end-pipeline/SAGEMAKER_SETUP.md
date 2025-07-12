# SageMaker Deployment Setup Guide

## Overview

The end-to-end pipeline now includes automated deployment to AWS SageMaker with manual approval for production deployments. This guide explains how to set up and use the SageMaker deployment feature.

## Prerequisites

### 1. AWS Account Setup
- Active AWS account with SageMaker access
- IAM user with appropriate permissions
- SageMaker execution role

### 2. Required AWS Permissions

Your AWS user needs these permissions:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:*",
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket",
                "iam:PassRole",
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "*"
        }
    ]
}
```

### 3. SageMaker Execution Role

Create a SageMaker execution role with these policies:
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess`

## GitHub Setup

### 1. Required Secrets

Add these secrets to your GitHub repository:

```bash
# AWS Credentials
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key

# SageMaker Role ARN
SAGEMAKER_EXECUTION_ROLE=arn:aws:iam::123456789012:role/service-role/SageMakerExecutionRole
```

### 2. Repository Variables

Set these variables in your repository:

```bash
# AWS Region (optional, defaults to us-east-1)
AWS_REGION=us-east-1
```

### 3. Environment Protection

The workflow uses GitHub Environments for approval:

1. **Go to Settings > Environments**
2. **Create environment**: `sagemaker-production`
3. **Add protection rules**:
   - ✅ Required reviewers (add yourself or team)
   - ✅ Wait timer (optional): 5 minutes
   - ✅ Deployment branches: Selected branches (main only)

## How the Approval Process Works

### 1. **Automatic Trigger**
- Approval step only runs for **production environment**
- Triggered when model passes quality gates
- Requires deployment to be ready

### 2. **Manual Approval Required**
When the pipeline reaches the approval step:

1. **Notification**: You'll receive a GitHub notification
2. **Review Required**: Check the deployment details in the workflow summary
3. **Approve/Reject**: Click "Review deployments" and approve/reject
4. **Automatic Deployment**: Once approved, deployment proceeds automatically

### 3. **Approval Information Displayed**
- Pipeline ID and model accuracy
- Git SHA and triggering user
- Cost warnings and deployment impact
- SageMaker console links

## Workflow Steps

### 1. **Pipeline Execution**
```bash
# Trigger with production environment
gh workflow run end-to-end-pipeline.yml \
  -f environment=production \
  -f pipeline_stages=all \
  -f force_run=true
```

### 2. **Approval Step** (Production Only)
- Job: `sagemaker-deployment-approval`
- Environment: `sagemaker-production`
- Manual approval required
- Displays model details and cost warnings

### 3. **SageMaker Deployment**
- Job: `deploy-to-sagemaker`
- Creates model package (model.tar.gz)
- Uploads to S3
- Creates SageMaker endpoint
- Tests the deployment
- Provides usage examples

## SageMaker Deployment Details

### Model Package Structure
```
model.tar.gz
├── final_model.pkl      # Trained model
├── final_scaler.pkl     # Feature scaler
└── inference.py         # SageMaker inference script
```

### Endpoint Configuration
- **Instance Type**: ml.t2.medium (cost-effective)
- **Instance Count**: 1
- **Framework**: scikit-learn
- **Python Version**: 3.x

### Endpoint Naming
- **Pattern**: `loan-approval-endpoint-YYYYMMDD-HHMMSS`
- **Example**: `loan-approval-endpoint-20240106-143022`

## Usage After Deployment

### 1. **Python SDK**
```python
import boto3
import json

# Initialize SageMaker runtime client
client = boto3.client('sagemaker-runtime', region_name='us-east-1')

# Prepare loan application data
loan_data = {
    "instances": [
        [50000, 700, 120000, 5, 0.35, 8, 1, 1, 0]  # loan features
    ]
}

# Make prediction
response = client.invoke_endpoint(
    EndpointName='your-endpoint-name',
    ContentType='application/json',
    Body=json.dumps(loan_data)
)

# Parse result
result = json.loads(response['Body'].read().decode())
print(f"Prediction: {result['predictions']}")
print(f"Probabilities: {result['probabilities']}")
```

### 2. **AWS CLI**
```bash
# Test endpoint
aws sagemaker-runtime invoke-endpoint \
    --endpoint-name your-endpoint-name \
    --content-type application/json \
    --body '{"instances": [[50000, 700, 120000, 5, 0.35, 8, 1, 1, 0]]}' \
    output.json

# View result
cat output.json
```

### 3. **Web Interface**
Access through [SageMaker Console](https://console.aws.amazon.com/sagemaker/):
- View endpoint status
- Monitor performance metrics
- Check CloudWatch logs
- Manage endpoint configuration

## Cost Management

### Estimated Costs
- **ml.t2.medium**: ~$0.056 per hour
- **Monthly cost**: ~$40 (if running 24/7)
- **S3 storage**: Minimal (~$0.01 per month)

### Cost Optimization
1. **Auto-scaling**: Configure to scale down during low usage
2. **Scheduled deletion**: Delete endpoints when not needed
3. **Instance optimization**: Use smaller instances for testing

### Delete Endpoint
```python
import boto3

sagemaker = boto3.client('sagemaker')
sagemaker.delete_endpoint(EndpointName='your-endpoint-name')
```

## Troubleshooting

### Common Issues

1. **Permission Denied**
   - Verify AWS credentials are correct
   - Check SageMaker execution role permissions
   - Ensure IAM user has SageMaker access

2. **Deployment Timeout**
   - Check CloudWatch logs for errors
   - Verify model package is valid
   - Ensure sufficient resources in region

3. **Approval Not Triggered**
   - Verify environment is set to "production"
   - Check that model passed quality gates
   - Ensure GitHub environment is configured

### Debug Commands

```bash
# Check SageMaker endpoints
aws sagemaker list-endpoints

# View endpoint details
aws sagemaker describe-endpoint --endpoint-name your-endpoint-name

# Check CloudWatch logs
aws logs describe-log-groups --log-group-name-prefix /aws/sagemaker/Endpoints
```

## Security Best Practices

### 1. **Credentials Management**
- Use IAM roles instead of access keys when possible
- Rotate access keys regularly
- Use least privilege principle

### 2. **Network Security**
- Deploy in VPC for production workloads
- Use security groups to restrict access
- Enable VPC endpoints for S3 and SageMaker

### 3. **Model Security**
- Encrypt model artifacts in S3
- Use encrypted endpoints
- Monitor access patterns

## Monitoring and Observability

### 1. **CloudWatch Metrics**
- Endpoint invocation count
- Model latency
- Error rates
- Instance utilization

### 2. **CloudWatch Logs**
- Endpoint logs: `/aws/sagemaker/Endpoints/your-endpoint-name`
- Training logs: `/aws/sagemaker/TrainingJobs`

### 3. **SageMaker Model Monitor**
- Data drift detection
- Model quality monitoring
- Automatic alerts

## Next Steps

1. **Set up monitoring**: Configure CloudWatch dashboards
2. **Implement A/B testing**: Deploy multiple model versions
3. **Add auto-scaling**: Configure based on traffic patterns
4. **Enhance security**: Implement VPC and encryption
5. **Cost optimization**: Set up automated cleanup

## Support

For issues with:
- **AWS setup**: Check AWS documentation
- **GitHub Actions**: Review workflow logs
- **SageMaker**: Check CloudWatch logs
- **Pipeline**: Review MLflow experiment tracking

The deployment includes comprehensive logging and error handling to help diagnose issues quickly.