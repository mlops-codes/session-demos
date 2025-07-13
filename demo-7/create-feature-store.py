#!/usr/bin/env python3
"""
Create SageMaker Feature Store and Feature Groups
This script sets up the feature store infrastructure for our ML pipeline
"""

import boto3
import time
import logging
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config, CUSTOMER_FEATURE_DEFINITIONS, TRANSACTION_FEATURE_DEFINITIONS, AGGREGATED_FEATURE_DEFINITIONS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureStoreManager:
    """Manages SageMaker Feature Store operations"""
    
    def __init__(self, environment='development'):
        self.config = get_config(environment)
        self.sagemaker_client = boto3.client('sagemaker', region_name=self.config.AWS_REGION)
        self.s3_client = boto3.client('s3', region_name=self.config.AWS_REGION)
        
        logger.info(f"Initialized FeatureStoreManager for {environment} environment")
    
    def create_s3_bucket(self):
        """Create S3 bucket for offline feature store"""
        try:
            bucket_name = self.config.S3_BUCKET
            
            # Check if bucket exists
            try:
                self.s3_client.head_bucket(Bucket=bucket_name)
                logger.info(f"✅ S3 bucket {bucket_name} already exists")
                return True
            except:
                pass
            
            # Create bucket
            if self.config.AWS_REGION == 'us-east-1':
                # us-east-1 doesn't need LocationConstraint
                self.s3_client.create_bucket(Bucket=bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.config.AWS_REGION}
                )
            
            logger.info(f"✅ Created S3 bucket: {bucket_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create S3 bucket: {str(e)}")
            return False
    
    def create_feature_group(self, feature_group_name, feature_definitions, 
                           record_identifier_name, event_time_feature_name,
                           description=""):
        """Create a feature group in SageMaker Feature Store"""
        
        try:
            logger.info(f"🏗️ Creating feature group: {feature_group_name}")
            
            # Check if feature group already exists
            try:
                response = self.sagemaker_client.describe_feature_group(
                    FeatureGroupName=feature_group_name
                )
                logger.info(f"✅ Feature group {feature_group_name} already exists")
                return response['FeatureGroupArn']
            except self.sagemaker_client.exceptions.ResourceNotFound:
                pass
            
            # Prepare feature group configuration
            create_params = {
                'RoleArn': self.config.ROLE_ARN,
                'FeatureGroupName': feature_group_name,
                'RecordIdentifierFeatureName': record_identifier_name,
                'EventTimeFeatureName': event_time_feature_name,
                'FeatureDefinitions': feature_definitions,
                'Description': description,
                'Tags': [
                    {'Key': 'Environment', 'Value': self.config.FEATURE_GROUP_PREFIX},
                    {'Key': 'Project', 'Value': 'FeatureStoreDemo'},
                    {'Key': 'CreatedBy', 'Value': 'FeatureStoreManager'}
                ]
            }
            
            # Add online store configuration
            if self.config.ENABLE_ONLINE_STORE:
                create_params['OnlineStoreConfig'] = {
                    'EnableOnlineStore': True
                }
                logger.info(f"   📱 Enabling online store")
            
            # Add offline store configuration
            if self.config.ENABLE_OFFLINE_STORE:
                create_params['OfflineStoreConfig'] = {
                    'S3StorageConfig': {
                        'S3Uri': self.config.get_s3_uri(f'offline-store/{feature_group_name}'),
                    },
                    'DisableGlueTableCreation': True,
                    'DataCatalogConfig': {
                        'TableName': feature_group_name.replace('-', '_'),
                        'Catalog': 'AwsDataCatalog',
                        'Database': self.config.DATABASE_NAME
                    }
                }
                logger.info(f"   💾 Enabling offline store")
            
            # Create the feature group
            response = self.sagemaker_client.create_feature_group(**create_params)
            
            logger.info(f"✅ Created feature group: {feature_group_name}")
            logger.info(f"   ARN: {response['FeatureGroupArn']}")
            
            return response['FeatureGroupArn']
            
        except Exception as e:
            logger.error(f"❌ Failed to create feature group {feature_group_name}: {str(e)}")
            raise
    
    def wait_for_feature_group_creation(self, feature_group_name, timeout=600):
        """Wait for feature group to be created and available"""
        
        logger.info(f"⏳ Waiting for feature group {feature_group_name} to be available...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.sagemaker_client.describe_feature_group(
                    FeatureGroupName=feature_group_name
                )
                
                status = response['FeatureGroupStatus']
                logger.info(f"   Status: {status}")
                
                if status == 'Created':
                    logger.info(f"✅ Feature group {feature_group_name} is ready")
                    return True
                elif status == 'CreateFailed':
                    failure_reason = response.get('FailureReason', 'Unknown')
                    logger.error(f"❌ Feature group creation failed: {failure_reason}")
                    return False
                
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"❌ Error checking feature group status: {str(e)}")
                return False
        
        logger.error(f"❌ Timeout waiting for feature group {feature_group_name}")
        return False
    
    def create_all_feature_groups(self):
        """Create all feature groups for the demo"""
        
        logger.info("🚀 Creating all feature groups...")
        
        feature_groups = [
            {
                'name': self.config.get_feature_group_name('customers'),
                'definitions': CUSTOMER_FEATURE_DEFINITIONS,
                'record_identifier': 'customer_id',
                'event_time': 'event_time',
                'description': 'Customer demographic and profile features'
            },
            {
                'name': self.config.get_feature_group_name('transactions'),
                'definitions': TRANSACTION_FEATURE_DEFINITIONS,
                'record_identifier': 'transaction_id',
                'event_time': 'event_time',
                'description': 'Individual transaction features'
            },
            {
                'name': self.config.get_feature_group_name('aggregated'),
                'definitions': AGGREGATED_FEATURE_DEFINITIONS,
                'record_identifier': 'customer_id',
                'event_time': 'event_time',
                'description': 'Aggregated customer behavior features'
            }
        ]
        
        created_groups = []
        
        for group in feature_groups:
            try:
                # Create feature group
                arn = self.create_feature_group(
                    group['name'],
                    group['definitions'],
                    group['record_identifier'],
                    group['event_time'],
                    group['description']
                )
                
                # Wait for creation to complete
                if self.wait_for_feature_group_creation(group['name']):
                    created_groups.append({
                        'name': group['name'],
                        'arn': arn,
                        'status': 'created'
                    })
                else:
                    created_groups.append({
                        'name': group['name'],
                        'arn': arn,
                        'status': 'failed'
                    })
                
            except Exception as e:
                logger.error(f"❌ Failed to create feature group {group['name']}: {str(e)}")
                created_groups.append({
                    'name': group['name'],
                    'arn': None,
                    'status': 'error',
                    'error': str(e)
                })
        
        return created_groups
    
    def list_feature_groups(self):
        """List all feature groups"""
        try:
            response = self.sagemaker_client.list_feature_groups(
                NameContains=self.config.FEATURE_GROUP_PREFIX
            )
            
            logger.info(f"📋 Found {len(response['FeatureGroupSummaries'])} feature groups:")
            
            for group in response['FeatureGroupSummaries']:
                logger.info(f"   • {group['FeatureGroupName']} - {group['FeatureGroupStatus']}")
            
            return response['FeatureGroupSummaries']
            
        except Exception as e:
            logger.error(f"❌ Failed to list feature groups: {str(e)}")
            return []
    
    def delete_feature_group(self, feature_group_name):
        """Delete a feature group"""
        try:
            logger.info(f"🗑️ Deleting feature group: {feature_group_name}")
            
            self.sagemaker_client.delete_feature_group(
                FeatureGroupName=feature_group_name
            )
            
            logger.info(f"✅ Deleted feature group: {feature_group_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete feature group {feature_group_name}: {str(e)}")
            return False

def main():
    """Main function to create feature store infrastructure"""
    
    print("🏪 SageMaker Feature Store Setup")
    print("=" * 40)
    
    # Initialize manager
    environment = os.getenv('ENVIRONMENT', 'development')
    manager = FeatureStoreManager(environment)
    
    # Validate configuration
    config_errors = manager.config.validate_config()
    if config_errors:
        logger.error("❌ Configuration errors found:")
        for error in config_errors:
            logger.error(f"   • {error}")
        logger.error("💡 Please update config.py with your AWS details")
        return False
    
    try:
        # Create S3 bucket
        logger.info("📦 Setting up S3 bucket...")
        if not manager.create_s3_bucket():
            logger.error("❌ Failed to create S3 bucket")
            return False
        
        # Create feature groups
        logger.info("🏗️ Creating feature groups...")
        created_groups = manager.create_all_feature_groups()
        
        # Summary
        print("\n" + "=" * 40)
        print("📊 Creation Summary:")
        
        success_count = 0
        for group in created_groups:
            status_emoji = "✅" if group['status'] == 'created' else "❌"
            print(f"   {status_emoji} {group['name']}: {group['status']}")
            if group['status'] == 'created':
                success_count += 1
        
        print(f"\n🎯 {success_count}/{len(created_groups)} feature groups created successfully")
        
        if success_count == len(created_groups):
            print("\n🎉 Feature store setup completed successfully!")
            print("\n💡 Next steps:")
            return True
        else:
            print("\n⚠️ Some feature groups failed to create")
            return False
            
    except Exception as e:
        logger.error(f"❌ Setup failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)