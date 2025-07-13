#!/usr/bin/env python3
"""
Configuration for SageMaker Feature Store Examples
"""

import os
from datetime import datetime

class FeatureStoreConfig:
    """Configuration for SageMaker Feature Store"""
    
    # AWS Configuration
    AWS_REGION = os.getenv('AWS_REGION', 'eu-west-2')
    AWS_ACCOUNT_ID = os.getenv('AWS_ACCOUNT_ID', '750952118292')
    
    # SageMaker Configuration
    SAGEMAKER_EXECUTION_ROLE = os.getenv(
        'SAGEMAKER_EXECUTION_ROLE',
        f'arn:aws:iam::{AWS_ACCOUNT_ID}:role/service-role/AmazonSageMaker-ExecutionRole-20250615T204995'
    )
    
    # S3 Configuration
    S3_BUCKET = os.getenv('S3_BUCKET', 'sagemaker-feature-store-demo-fhkajhk3h4hhfsdhfhk')
    S3_PREFIX = os.getenv('S3_PREFIX', 'feature-store')
    
    # Feature Store Configuration
    FEATURE_GROUP_PREFIX = 'demo'
    ENABLE_ONLINE_STORE = True
    ENABLE_OFFLINE_STORE = True
    
    # Database Configuration (for offline store)
    DATABASE_NAME = 'feature_store_db'
    
    # Feature Groups
    CUSTOMER_FEATURE_GROUP = f'{FEATURE_GROUP_PREFIX}-customers'
    TRANSACTION_FEATURE_GROUP = f'{FEATURE_GROUP_PREFIX}-transactions'
    AGGREGATED_FEATURE_GROUP = f'{FEATURE_GROUP_PREFIX}-aggregated'
    
    # Data Generation
    NUM_CUSTOMERS = 10000
    NUM_TRANSACTIONS = 50000
    DATE_RANGE_DAYS = 365
    
    # Model Configuration
    MODEL_NAME = 'fraud-detection-model'
    ENDPOINT_NAME = f'{MODEL_NAME}-endpoint'
    
    # Monitoring
    FEATURE_DRIFT_THRESHOLD = 0.1
    DATA_QUALITY_THRESHOLD = 0.05

    ROLE_ARN = SAGEMAKER_EXECUTION_ROLE
    
    @classmethod
    def validate_config(cls):
        """Validate configuration"""
        errors = []
        
        if cls.AWS_ACCOUNT_ID == 'YOUR_ACCOUNT_ID':
            errors.append("AWS_ACCOUNT_ID not configured")
        
        if 'YOUR_ACCOUNT_ID' in cls.SAGEMAKER_EXECUTION_ROLE:
            errors.append("SAGEMAKER_EXECUTION_ROLE not configured")
        
        return errors
    
    @classmethod
    def get_feature_group_name(cls, base_name):
        """Generate timestamped feature group name"""
        timestamp = datetime.now().strftime('%Y%m%d')
        return f"{cls.FEATURE_GROUP_PREFIX}-{base_name}-{timestamp}"
    
    @classmethod
    def get_s3_uri(cls, key):
        """Generate S3 URI"""
        return f"s3://{cls.S3_BUCKET}/{cls.S3_PREFIX}/{key}"

# Environment-specific configurations
class DevelopmentConfig(FeatureStoreConfig):
    """Development environment configuration"""
    FEATURE_GROUP_PREFIX = 'dev'
    NUM_CUSTOMERS = 1000
    NUM_TRANSACTIONS = 5000
    ENABLE_ONLINE_STORE = False  # Save costs in dev
    ROLE_ARN = FeatureStoreConfig.SAGEMAKER_EXECUTION_ROLE

class ProductionConfig(FeatureStoreConfig):
    """Production environment configuration"""
    FEATURE_GROUP_PREFIX = 'prod'
    NUM_CUSTOMERS = 100000
    NUM_TRANSACTIONS = 1000000
    ENABLE_ONLINE_STORE = True
    
    # Enhanced monitoring in production
    FEATURE_DRIFT_THRESHOLD = 0.05
    DATA_QUALITY_THRESHOLD = 0.02

def get_config(environment='development'):
    """Get configuration based on environment"""
    if environment.lower() == 'production':
        return ProductionConfig()
    else:
        return DevelopmentConfig()

# Feature Definitions
CUSTOMER_FEATURE_DEFINITIONS = [
    {'FeatureName': 'customer_id', 'FeatureType': 'String'},
    {'FeatureName': 'age', 'FeatureType': 'Integral'},
    {'FeatureName': 'income', 'FeatureType': 'Fractional'},
    {'FeatureName': 'credit_score', 'FeatureType': 'Integral'},
    {'FeatureName': 'account_age_days', 'FeatureType': 'Integral'},
    {'FeatureName': 'num_accounts', 'FeatureType': 'Integral'},
    {'FeatureName': 'avg_monthly_balance', 'FeatureType': 'Fractional'},
    {'FeatureName': 'customer_segment', 'FeatureType': 'String'},
    {'FeatureName': 'is_premium', 'FeatureType': 'Integral'},
    {'FeatureName': 'last_login_days', 'FeatureType': 'Integral'},
    {'FeatureName': 'event_time', 'FeatureType': 'Fractional'},
]

TRANSACTION_FEATURE_DEFINITIONS = [
    {'FeatureName': 'transaction_id', 'FeatureType': 'String'},
    {'FeatureName': 'customer_id', 'FeatureType': 'String'},
    {'FeatureName': 'amount', 'FeatureType': 'Fractional'},
    {'FeatureName': 'transaction_type', 'FeatureType': 'String'},
    {'FeatureName': 'merchant_category', 'FeatureType': 'String'},
    {'FeatureName': 'is_weekend', 'FeatureType': 'Integral'},
    {'FeatureName': 'hour_of_day', 'FeatureType': 'Integral'},
    {'FeatureName': 'days_since_last_transaction', 'FeatureType': 'Integral'},
    {'FeatureName': 'amount_vs_avg_ratio', 'FeatureType': 'Fractional'},
    {'FeatureName': 'is_international', 'FeatureType': 'Integral'},
    {'FeatureName': 'event_time', 'FeatureType': 'Fractional'},
]

AGGREGATED_FEATURE_DEFINITIONS = [
    {'FeatureName': 'customer_id', 'FeatureType': 'String'},
    {'FeatureName': 'total_transactions_30d', 'FeatureType': 'Integral'},
    {'FeatureName': 'total_amount_30d', 'FeatureType': 'Fractional'},
    {'FeatureName': 'avg_transaction_amount_30d', 'FeatureType': 'Fractional'},
    {'FeatureName': 'unique_merchants_30d', 'FeatureType': 'Integral'},
    {'FeatureName': 'weekend_transaction_ratio_30d', 'FeatureType': 'Fractional'},
    {'FeatureName': 'international_transaction_ratio_30d', 'FeatureType': 'Fractional'},
    {'FeatureName': 'max_transaction_amount_30d', 'FeatureType': 'Fractional'},
    {'FeatureName': 'transaction_frequency_score', 'FeatureType': 'Fractional'},
    {'FeatureName': 'risk_score', 'FeatureType': 'Fractional'},
    {'FeatureName': 'event_time', 'FeatureType': 'Fractional'},
]

# Test configuration
if __name__ == "__main__":
    print("🔧 SageMaker Feature Store Configuration")
    print("=" * 50)
    
    # Test configurations
    dev_config = get_config('development')
    prod_config = get_config('production')
    
    print(f"📊 Development Configuration:")
    print(f"   Region: {dev_config.AWS_REGION}")
    print(f"   Prefix: {dev_config.FEATURE_GROUP_PREFIX}")
    print(f"   Customers: {dev_config.NUM_CUSTOMERS:,}")
    print(f"   Online Store: {dev_config.ENABLE_ONLINE_STORE}")
    
    print(f"\n🏭 Production Configuration:")
    print(f"   Region: {prod_config.AWS_REGION}")
    print(f"   Prefix: {prod_config.FEATURE_GROUP_PREFIX}")
    print(f"   Customers: {prod_config.NUM_CUSTOMERS:,}")
    print(f"   Online Store: {prod_config.ENABLE_ONLINE_STORE}")
    
    # Test feature group names
    print(f"\n📝 Example Feature Group Names:")
    print(f"   Customers: {dev_config.get_feature_group_name('customers')}")
    print(f"   Transactions: {dev_config.get_feature_group_name('transactions')}")
    
    # Validate configuration
    errors = dev_config.validate_config()
    if errors:
        print(f"\n⚠️ Configuration Issues:")
        for error in errors:
            print(f"   • {error}")
    else:
        print(f"\n✅ Configuration valid!")
    
    print(f"\n📊 Feature Definitions:")
    print(f"   Customer Features: {len(CUSTOMER_FEATURE_DEFINITIONS)}")
    print(f"   Transaction Features: {len(TRANSACTION_FEATURE_DEFINITIONS)}")
    print(f"   Aggregated Features: {len(AGGREGATED_FEATURE_DEFINITIONS)}")