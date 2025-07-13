#!/usr/bin/env python3
"""
SageMaker Feature Store Training
This script demonstrates how to actually retrieve features from SageMaker Feature Store
and use them for model training (not simulation)
"""

import boto3
import pandas as pd
import numpy as np
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_definition import FeatureDefinition
import awswrangler as wr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from datetime import datetime, timedelta
import sys
import os
import time
import argparse


# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureStoreTrainer:
    """ implementation using SageMaker Feature Store APIs"""
    
    def __init__(self, environment='development'):
        self.config = get_config(environment)
        
        # Initialize SageMaker session
        self.sagemaker_session = sagemaker.Session()
        self.region = self.config.AWS_REGION
        self.role = self.config.SAGEMAKER_EXECUTION_ROLE
        
        # Initialize AWS clients
        self.sagemaker_client = boto3.client('sagemaker', region_name=self.region)
        self.featurestore_runtime = boto3.client('sagemaker-featurestore-runtime', region_name=self.region)
        
        # Feature group names
        self.customer_fg_name = self.config.get_feature_group_name('customers')
        self.transaction_fg_name = self.config.get_feature_group_name('transactions')
        self.aggregated_fg_name = self.config.get_feature_group_name('aggregated')
        
        logger.info(f"Initialized  Feature Store Trainer for {environment}")
        logger.info(f"Feature Groups: {[self.customer_fg_name, self.transaction_fg_name, self.aggregated_fg_name]}")
    
    def verify_feature_groups_exist(self):
        """Verify that feature groups exist and are available"""
        logger.info("🔍 Verifying feature groups exist...")
        
        feature_groups = [
            self.customer_fg_name,
            self.transaction_fg_name, 
            self.aggregated_fg_name
        ]
        
        for fg_name in feature_groups:
            try:
                response = self.sagemaker_client.describe_feature_group(FeatureGroupName=fg_name)
                status = response['FeatureGroupStatus']
                
                if status != 'Created':
                    logger.error(f"❌ Feature group {fg_name} is not ready (status: {status})")
                    return False
                
                logger.info(f"   ✅ {fg_name}: {status}")
                
            except Exception as e:
                logger.error(f"❌ Feature group {fg_name} not found: {str(e)}")
                return False
        
        return True
    
    def get_feature_group_s3_uri(self, feature_group_name):
        """Get the S3 URI for offline feature store data"""
        try:
            response = self.sagemaker_client.describe_feature_group(FeatureGroupName=feature_group_name)
            
            if 'OfflineStoreConfig' in response:
                s3_uri = response['OfflineStoreConfig']['S3StorageConfig']['S3Uri']
                return s3_uri
            else:
                logger.error(f"❌ No offline store configured for {feature_group_name}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to get S3 URI for {feature_group_name}: {str(e)}")
            return None
    
    def query_offline_store(self, feature_group_name, customer_ids=None, max_records=None):
        """Query the offline feature store using Athena"""
        logger.info(f"📊 Querying offline store for {feature_group_name}...")
        
        try:
            # Get the feature group details
            response = self.sagemaker_client.describe_feature_group(FeatureGroupName=feature_group_name)
            
            # Check if offline store is configured
            if 'OfflineStoreConfig' not in response:
                logger.error(f"❌ No offline store configured for {feature_group_name}")
                return None
            
            # Get the data catalog info
            catalog_config = response['OfflineStoreConfig']['DataCatalogConfig']
            database_name = catalog_config['Database']
            table_name = catalog_config['TableName']
            
            # Build the query
            write_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            base_query = f"""
            SELECT *
            FROM "{database_name}"."{table_name}"
            """
            
            # Add customer filter if provided
            if customer_ids:
                # Format customer IDs for SQL IN clause
                customer_list = "', '".join(customer_ids[:100])  # Limit to first 100 for query length
                base_query += f" AND customer_id IN ('{customer_list}')"
            
            # Add limit if specified
            if max_records:
                base_query += f" LIMIT {max_records}"
            
            logger.info(f"   Executing Athena query on table: {database_name}.{table_name}")
            
            # Execute query using AWS Data Wrangler
            df = wr.athena.read_sql_query(
                sql=base_query,
                database=database_name,
                s3_output=self.config.get_s3_uri('athena-results/'),
                boto3_session=boto3.Session()
            )
            
            logger.info(f"   ✅ Retrieved {len(df)} records from {feature_group_name}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to query offline store for {feature_group_name}: {str(e)}")
            logger.info(f"💡 Make sure Athena is set up and the feature group has offline store enabled")
            return None
    
    def retrieve_online_features(self, customer_ids, max_batch_size=100):
        """Retrieve features from online feature store for specific customers"""
        logger.info(f"⚡ Retrieving online features for {len(customer_ids)} customers...")
        
        all_features = []
        
        # Process in batches to avoid API limits
        for i in range(0, len(customer_ids), max_batch_size):
            batch_ids = customer_ids[i:i + max_batch_size]
            
            try:
                # Retrieve customer features
                customer_features = self._get_online_features_batch(
                    self.customer_fg_name, batch_ids, 'customer_id'
                )
                
                # Retrieve aggregated features
                aggregated_features = self._get_online_features_batch(
                    self.aggregated_fg_name, batch_ids, 'customer_id'
                )
                
                # Merge features
                for customer_id in batch_ids:
                    customer_record = customer_features.get(customer_id, {})
                    aggregated_record = aggregated_features.get(customer_id, {})
                    
                    # Combine features
                    combined_features = {**customer_record, **aggregated_record}
                    combined_features['customer_id'] = customer_id
                    
                    if combined_features:
                        all_features.append(combined_features)
                
                logger.info(f"   Processed batch {i//max_batch_size + 1}/{(len(customer_ids)-1)//max_batch_size + 1}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to retrieve features for batch starting at {i}: {str(e)}")
                continue
        
        if all_features:
            features_df = pd.DataFrame(all_features)
            logger.info(f"   ✅ Retrieved online features for {len(features_df)} customers")
            return features_df
        else:
            logger.error("❌ No online features retrieved")
            return None
    
    def _get_online_features_batch(self, feature_group_name, record_ids, identifier_name):
        """Get online features for a batch of records"""
        features_dict = {}
        
        for record_id in record_ids:
            try:
                response = self.featurestore_runtime.get_record(
                    FeatureGroupName=feature_group_name,
                    RecordIdentifierValueAsString=str(record_id)
                )
                
                if 'Record' in response and response['Record']:
                    # Convert feature list to dictionary
                    feature_dict = {}
                    for feature in response['Record']:
                        feature_name = feature['FeatureName']
                        feature_value = feature['ValueAsString']
                        
                        # Convert to appropriate type
                        if feature_name != identifier_name and feature_name != 'event_time':
                            try:
                                # Try to convert to float first
                                feature_dict[feature_name] = float(feature_value)
                            except ValueError:
                                # Keep as string if conversion fails
                                feature_dict[feature_name] = feature_value
                        else:
                            feature_dict[feature_name] = feature_value
                    
                    features_dict[record_id] = feature_dict
                
            except Exception as e:
                logger.debug(f"   Could not retrieve record {record_id} from {feature_group_name}: {str(e)}")
                continue
        
        return features_dict
    
    def create_dataset_from_feature_store(self, approach='online', customer_ids=None, max_records=5000):
        """Create training dataset from feature store"""
        logger.info(f"📊 Creating dataset using {approach} approach...")
        
        if approach == 'online':
            # Use online feature store
            if not customer_ids:
                # Get customer IDs from offline store first
                logger.info("   Getting customer IDs from offline store...")
                customer_df = self.query_offline_store(self.customer_fg_name, max_records=max_records)
                if customer_df is None or customer_df.empty:
                    logger.error("❌ Could not get customer IDs")
                    return None
                customer_ids = customer_df['customer_id'].unique()[:max_records]
            
            # Retrieve features from online store
            features_df = self.retrieve_online_features(customer_ids[:max_records])
            
        elif approach == 'offline':
            # Use offline feature store (Athena queries)
            logger.info("   Querying offline store...")
            
            # Query customer features
            customer_df = self.query_offline_store(self.customer_fg_name, customer_ids, max_records)
            if customer_df is None or customer_df.empty:
                logger.error("❌ Could not retrieve customer features from offline store")
                return None
            
            # Query aggregated features
            aggregated_df = self.query_offline_store(self.aggregated_fg_name, customer_ids, max_records)
            if aggregated_df is None or aggregated_df.empty:
                logger.error("❌ Could not retrieve aggregated features from offline store")
                return None
            
            # Merge features
            features_df = pd.merge(
                customer_df, aggregated_df, 
                on='customer_id', 
                how='inner',
                suffixes=('', '_agg')
            )
            
            # Remove duplicate columns and metadata columns
            features_df = features_df.loc[:, ~features_df.columns.duplicated()]
            features_df = features_df.drop(columns=['write_time', 'api_invocation_time', 'is_deleted'], errors='ignore')
            
        else:
            logger.error(f"❌ Unsupported approach: {approach}")
            return None
        
        if features_df is None or features_df.empty:
            logger.error("❌ No features retrieved")
            return None
        
        logger.info(f"   ✅ Created dataset with {len(features_df)} records and {len(features_df.columns)} features")
        
        return features_df
    
    def create_fraud_labels(self, features_df):
        """Create fraud labels based on risk patterns (for demo purposes)"""
        logger.info("🏷️ Creating fraud labels...")
        
        # In a  scenario, you would have actual fraud labels
        # For this demo, we create labels based on risk patterns
        
        labels = []
        for _, row in features_df.iterrows():
            risk_score = 0
            
            # Risk factors
            if 'risk_score' in row and pd.notna(row['risk_score']):
                risk_score += float(row['risk_score']) * 0.4
            
            if 'international_transaction_ratio_30d' in row and pd.notna(row['international_transaction_ratio_30d']):
                risk_score += float(row['international_transaction_ratio_30d']) * 0.3
            
            if 'total_transactions_30d' in row and pd.notna(row['total_transactions_30d']):
                if float(row['total_transactions_30d']) > 50:
                    risk_score += 0.2
            
            if 'avg_transaction_amount_30d' in row and pd.notna(row['avg_transaction_amount_30d']):
                if float(row['avg_transaction_amount_30d']) > 1000:
                    risk_score += 0.3
            
            # Add some randomness
            risk_score += np.random.normal(0, 0.1)
            
            # Convert to binary label
            is_fraud = 1 if risk_score > 0.6 else 0
            labels.append(is_fraud)
        
        fraud_rate = np.mean(labels)
        logger.info(f"   ✅ Created {len(labels)} labels with {fraud_rate:.1%} fraud rate")
        
        return np.array(labels)
    
    def prepare_features_for_training(self, features_df):
        """Prepare features for model training"""
        logger.info("📋 Preparing features for training...")
        
        # Define feature columns (exclude metadata columns)
        exclude_columns = ['customer_id', 'event_time', 'write_time', 'api_invocation_time', 'is_deleted']
        feature_columns = [col for col in features_df.columns if col not in exclude_columns]
        
        # Extract feature matrix
        X = features_df[feature_columns].copy()
        
        # Handle missing values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

        # Fill the rest with a placeholder or drop them, depending on your needs
        X = X.fillna(0)
        
        # Convert all columns to numeric where possible
        for col in X.columns:
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            except:
                pass
        
        # Final fill for any remaining NaN values
        X = X.fillna(0)
        
        logger.info(f"   ✅ Prepared {len(feature_columns)} features for {len(X)} samples")
        logger.info(f"   📊 Feature columns: {feature_columns}")
        
        return X, feature_columns
    
    def train_model_with_feature_store(self, approach='online', max_records=5000):
        """Train model using  SageMaker Feature Store"""
        logger.info("🚀 Training model with  Feature Store data...")
        
        try:
            # Step 1: Verify feature groups exist
            if not self.verify_feature_groups_exist():
                logger.error("❌ Feature groups not ready")
                return False
            
            # Step 2: Create dataset from feature store
            features_df = self.create_dataset_from_feature_store(
                approach=approach, 
                max_records=max_records
            )
            
            if features_df is None:
                logger.error("❌ Could not create dataset from feature store")
                return False
            
            # Step 3: Create labels
            y = self.create_fraud_labels(features_df)
            
            # Step 4: Prepare features
            X, feature_columns = self.prepare_features_for_training(features_df)
            
            # Step 5: Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"   📊 Training: {len(X_train)} samples")
            logger.info(f"   📊 Testing: {len(X_test)} samples")
            
            # Step 6: Train model
            logger.info("   🤖 Training Random Forest model...")
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            model.fit(X_train, y_train)
            
            # Step 7: Evaluate model
            y_pred = model.predict(X_test)
            if len(model.classes_) == 2:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = model.predict_proba(X_test)[:, 0]  # only one class, fallback
            
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            logger.info(f"   ✅ Model trained successfully!")
            logger.info(f"   🎯 AUC Score: {auc_score:.3f}")
            
            # Step 8: Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                logger.info("   📊 Top 10 Feature Importances:")
                for _, row in feature_importance.head(10).iterrows():
                    logger.info(f"      • {row['feature']}: {row['importance']:.3f}")
            
            # Step 9: Save model
            model_info = self._save_trained_model(
                model, feature_columns, auc_score, approach
            )
            
            logger.info(f"   💾 Model saved: {model_info['model_path']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {str(e)}")
            return False
    
    def _save_trained_model(self, model, feature_columns, auc_score, approach):
        """Save the trained model and metadata"""
        # Create models directory
        models_dir = './models'
        os.makedirs(models_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'{models_dir}/fraud_model_featurestore_{approach}_{timestamp}.pkl'
        
        # Save model
        joblib.dump(model, model_filename)
        
        # Create metadata
        metadata = {
            'model_type': 'RandomForestClassifier',
            'training_approach': approach,
            'auc_score': float(auc_score),
            'feature_columns': feature_columns,
            'timestamp': timestamp,
            'feature_groups_used': [
                self.customer_fg_name,
                self.aggregated_fg_name
            ],
            'environment': self.config.FEATURE_GROUP_PREFIX
        }
        
        metadata_filename = f'{models_dir}/fraud_model_featurestore_{approach}_{timestamp}_metadata.json'
        
        import json
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            'model_path': model_filename,
            'metadata_path': metadata_filename,
            'auc_score': auc_score
        }

def main():
    """Main function to run  feature store training"""
    parser = argparse.ArgumentParser(description='Train model using  SageMaker Feature Store')
    parser.add_argument('--approach', choices=['online', 'offline'], default='offline',
                       help='Use online or offline feature store')
    parser.add_argument('--max-records', type=int, default=5000,
                       help='Maximum number of records to use for training')
    parser.add_argument('--environment', choices=['development', 'production'], 
                       default='development',
                       help='Environment configuration')
    
    args = parser.parse_args()
    
    print("🎯  SageMaker Feature Store Model Training")
    print("=" * 60)
    print(f"Approach: {args.approach}")
    print(f"Max Records: {args.max_records}")
    print(f"Environment: {args.environment}")
    
    try:
        # Initialize trainer
        trainer = FeatureStoreTrainer(args.environment)
        
        # Train model
        success = trainer.train_model_with_feature_store(
            approach=args.approach,
            max_records=args.max_records
        )
        
        if success:
            print("\n" + "=" * 60)
            print("🎉  Feature Store training completed successfully!")
            print("\n💡 Key Achievements:")
            print(f"   ✅ Used actual SageMaker Feature Store {args.approach} APIs")
            print("   ✅ Retrieved features from feature groups")
            print("   ✅ Trained model with  feature store data")
            print("   ✅ Saved model with feature store metadata")
            print("\n🚀 Next Steps:")
            print("   1. Deploy model to SageMaker endpoint")
            print("   2. Set up -time inference with feature store")
            print("   3. Monitor feature drift in production")
            return True
        else:
            print("\n❌ Training failed")
            print("\n💡 Troubleshooting:")
            print("   • Ensure feature groups are created and in 'Created' status")
            print("   • Check that data has been ingested into feature store")
            print("   • Verify AWS permissions for Feature Store access")
            print("   • For offline approach, ensure Athena is configured")
            return False
            
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)