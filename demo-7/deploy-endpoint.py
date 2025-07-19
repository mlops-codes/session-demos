#!/usr/bin/env python3

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import joblib
import pandas as pd
import numpy as np
import json
import os
import sys
import time
import logging
import argparse
from datetime import datetime
import tarfile
import tempfile

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelDeployer:
    """Deploys trained models to SageMaker endpoints with Feature Store integration"""
    
    def __init__(self, environment='development'):
        self.config = get_config(environment)
        
        # Initialize SageMaker session
        self.sagemaker_session = sagemaker.Session()
        self.region = self.config.AWS_REGION
        self.role = self.config.SAGEMAKER_EXECUTION_ROLE
        self.bucket = self.config.S3_BUCKET
        
        # Initialize clients
        self.sagemaker_client = boto3.client('sagemaker', region_name=self.region)
        self.s3_client = boto3.client('s3', region_name=self.region)
        
        logger.info(f"Initialized Model Deployer for {environment}")
        logger.info(f"Region: {self.region}")
        logger.info(f"S3 Bucket: {self.bucket}")
    
    def find_latest_model(self, models_dir='./models'):
        """Find the latest trained model from Feature Store training"""
        logger.info("🔍 Finding latest trained model...")
        
        if not os.path.exists(models_dir):
            logger.error(f"❌ Models directory not found: {models_dir}")
            return None, None
        
        # Look for models with 'featurestore' in the name
        model_files = [f for f in os.listdir(models_dir) 
                      if f.endswith('.pkl') and 'featurestore' in f]
        
        if not model_files:
            logger.error("❌ No Feature Store trained models found")
            logger.info("💡 Run _feature_store_training.py first")
            return None, None
        
        # Sort by timestamp to get latest
        model_files.sort(reverse=True)
        latest_model = model_files[0]
        
        # Find corresponding metadata
        metadata_file = latest_model.replace('.pkl', '_metadata.json')
        metadata_path = os.path.join(models_dir, metadata_file)
        
        model_path = os.path.join(models_dir, latest_model)
        
        if not os.path.exists(metadata_path):
            logger.warning(f"⚠️ Metadata file not found: {metadata_path}")
            metadata_path = None
        
        logger.info(f"✅ Found latest model: {latest_model}")
        
        return model_path, metadata_path
    
    def load_model_metadata(self, metadata_path):
        """Load model metadata"""
        if not metadata_path or not os.path.exists(metadata_path):
            logger.warning("⚠️ No metadata available")
            return {}
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            logger.info("📋 Model metadata loaded:")
            logger.info(f"   Model Type: {metadata.get('model_type', 'Unknown')}")
            logger.info(f"   AUC Score: {metadata.get('auc_score', 'Unknown')}")
            logger.info(f"   Training Approach: {metadata.get('training_approach', 'Unknown')}")
            logger.info(f"   Features: {len(metadata.get('feature_columns', []))}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to load metadata: {str(e)}")
            return {}
    
    def create_inference_script(self, feature_columns, feature_groups_used):
        """Create inference script for SageMaker endpoint"""
        logger.info("📝 Creating inference script...")
        
        # Save inference script
        script_path = './inference.py'
        
        logger.info(f"✅ Inference script created: {script_path}")
        return script_path
    
    def create_model_artifact(self, model_path, inference_script_path):
        """Create model artifact (tar.gz) for SageMaker"""
        logger.info("📦 Creating model artifact...")
        
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy model file
                model_filename = 'model.pkl'
                temp_model_path = os.path.join(temp_dir, model_filename)
                
                # Load and save model to ensure compatibility
                model = joblib.load(model_path)
                joblib.dump(model, temp_model_path)
                
                # Copy inference script
                temp_script_path = os.path.join(temp_dir, 'inference.py')
                with open(inference_script_path, 'r') as src, open(temp_script_path, 'w') as dst:
                    dst.write(src.read())
                
                # Create tar.gz
                artifact_path = './model.tar.gz'
                with tarfile.open(artifact_path, 'w:gz') as tar:
                    tar.add(temp_model_path, arcname=model_filename)
                    tar.add(temp_script_path, arcname='inference.py')
                
                logger.info(f"✅ Model artifact created: {artifact_path}")
                return artifact_path
                
        except Exception as e:
            logger.error(f"❌ Failed to create model artifact: {str(e)}")
            return None
    
    def upload_model_to_s3(self, artifact_path):
        """Upload model artifact to S3"""
        logger.info("☁️ Uploading model to S3...")
        
        try:
            # Generate S3 key
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f"models/fraud-detection-{timestamp}/model.tar.gz"
            
            # Upload to S3
            self.s3_client.upload_file(artifact_path, self.bucket, s3_key)
            
            s3_uri = f"s3://{self.bucket}/{s3_key}"
            logger.info(f"✅ Model uploaded to: {s3_uri}")
            
            return s3_uri
            
        except Exception as e:
            logger.error(f"❌ Failed to upload model to S3: {str(e)}")
            return None
    
    def create_sagemaker_model(self, model_s3_uri, model_name=None):
        """Create SageMaker model"""
        logger.info("🤖 Creating SageMaker model...")
        
        if not model_name:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            model_name = f"fraud-detection-{timestamp}"
        
        try:
            # Create SKLearn model
            sklearn_model = SKLearnModel(
                model_data=model_s3_uri,
                role=self.role,
                entry_point='inference.py',
                framework_version='1.0-1',
                py_version='py3',
                name=model_name,
                sagemaker_session=self.sagemaker_session
            )
            
            logger.info(f"✅ SageMaker model created: {model_name}")
            return sklearn_model, model_name
            
        except Exception as e:
            logger.error(f"❌ Failed to create SageMaker model: {str(e)}")
            return None, None
    
    def deploy_endpoint(self, sklearn_model, endpoint_name=None, instance_type='ml.t2.medium'):
        """Deploy model to SageMaker endpoint"""
        logger.info("🚀 Deploying endpoint...")
        
        if not endpoint_name:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            endpoint_name = f"fraud-detection-endpoint-{timestamp}"
        
        try:
            # Deploy endpoint
            predictor = sklearn_model.deploy(
                initial_instance_count=1,
                instance_type=instance_type,
                endpoint_name=endpoint_name,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer()
            )
            
            logger.info(f"✅ Endpoint deployed: {endpoint_name}")
            logger.info(f"   Instance Type: {instance_type}")
            logger.info(f"   Status: InService")
            
            return predictor, endpoint_name
            
        except Exception as e:
            logger.error(f"❌ Failed to deploy endpoint: {str(e)}")
            return None, None
    
    def test_endpoint(self, predictor, test_data=None):
        """Test the deployed endpoint"""
        logger.info("🧪 Testing endpoint...")
        
        try:
            if test_data is None:
                # Create test data
                test_data = {
                    'customer_id': 'cust_000001',
                    'age': 35,
                    'income': 75000,
                    'credit_score': 720,
                    'total_transactions_30d': 25,
                    'avg_transaction_amount_30d': 150.0,
                    'risk_score': 0.3
                }
            
            # Test single prediction
            logger.info("   Testing single prediction...")
            result = predictor.predict(test_data)
            
            logger.info(f"   ✅ Test result: {result}")
            
            # Test batch prediction  
            logger.info("   Testing batch prediction...")
            batch_data = [test_data, {**test_data, 'customer_id': 'cust_000002', 'risk_score': 0.8}]
            batch_result = predictor.predict(batch_data)
            
            logger.info(f"   ✅ Batch result: {batch_result}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Endpoint testing failed: {str(e)}")
            return False
    
    def deploy_complete_pipeline(self, model_path=None, endpoint_name=None, instance_type='ml.t2.medium'):
        """Deploy complete model pipeline"""
        logger.info("🚀 Starting complete model deployment pipeline...")
        
        try:
            # Step 1: Find and load model
            if not model_path:
                model_path, metadata_path = self.find_latest_model()
                if not model_path:
                    return False
            else:
                # Look for metadata file
                metadata_path = model_path.replace('.pkl', '_metadata.json')
                if not os.path.exists(metadata_path):
                    metadata_path = None
            
            # Load metadata
            metadata = self.load_model_metadata(metadata_path)
            feature_columns = metadata.get('feature_columns', [])
            feature_groups_used = metadata.get('feature_groups_used', [])
            
            if not feature_columns:
                logger.error("❌ No feature columns found in metadata")
                return False
            
            # Step 2: Create inference script
            inference_script = self.create_inference_script(feature_columns, feature_groups_used)
            if not inference_script:
                return False
            
            # Step 3: Create model artifact
            artifact_path = self.create_model_artifact(model_path, inference_script)
            if not artifact_path:
                return False
            
            # Step 4: Upload to S3
            s3_uri = self.upload_model_to_s3(artifact_path)
            if not s3_uri:
                return False
            
            # Step 5: Create SageMaker model
            sklearn_model, model_name = self.create_sagemaker_model(s3_uri)
            if not sklearn_model:
                return False
            
            # Step 6: Deploy endpoint
            predictor, endpoint_name = self.deploy_endpoint(sklearn_model, endpoint_name, instance_type)
            if not predictor:
                return False
            
            # Step 7: Test endpoint
            test_success = self.test_endpoint(predictor)
            
            # Step 8: Save deployment info
            deployment_info = {
                'endpoint_name': endpoint_name,
                'model_name': model_name,
                'model_s3_uri': s3_uri,
                'instance_type': instance_type,
                'feature_columns': feature_columns,
                'feature_groups_used': feature_groups_used,
                'deployed_at': datetime.now().isoformat(),
                'test_success': test_success
            }
            
            info_path = f'./deployment_info_{endpoint_name}.json'
            with open(info_path, 'w') as f:
                json.dump(deployment_info, f, indent=2)
            
            # Cleanup
            os.remove(artifact_path)
            os.remove(inference_script)
            
            logger.info("🎉 Deployment completed successfully!")
            self._print_deployment_summary(deployment_info)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment pipeline failed: {str(e)}")
            return False
    
    def _print_deployment_summary(self, deployment_info):
        """Print deployment summary"""
        print("\n" + "=" * 70)
        print("🚀 SAGEMAKER ENDPOINT DEPLOYMENT SUMMARY")
        print("=" * 70)
        
        print(f"✅ Endpoint Name: {deployment_info['endpoint_name']}")
        print(f"🤖 Model Name: {deployment_info['model_name']}")
        print(f"🏗️ Instance Type: {deployment_info['instance_type']}")
        print(f"📊 Features: {len(deployment_info['feature_columns'])}")
        print(f"🏪 Feature Groups: {len(deployment_info['feature_groups_used'])}")
        print(f"🧪 Test Status: {'✅ PASSED' if deployment_info['test_success'] else '❌ FAILED'}")
        
        print(f"\n📋 Feature Groups Used:")
        for fg in deployment_info['feature_groups_used']:
            print(f"   • {fg}")
        
        print(f"\n🔗 Usage Examples:")
        print(f"   # Single prediction with customer ID")
        print(f"   predictor.predict({{'customer_id': 'cust_000001'}})")
        print(f"   ")
        print(f"   # Direct feature prediction")
        print(f"   predictor.predict({{'age': 35, 'income': 75000, 'risk_score': 0.3}})")
        
        print(f"\n💡 Next Steps:")
        print(f"   1. Test endpoint: python test_endpoint.py --endpoint {deployment_info['endpoint_name']}")
        print(f"   2. Monitor performance: Check CloudWatch metrics")
        print(f"   3. Set up auto-scaling if needed")
        print(f"   4. Integrate with application")

def main():
    """Main function for model deployment"""
    parser = argparse.ArgumentParser(description='Deploy trained model to SageMaker endpoint')
    parser.add_argument('--model-path', help='Path to trained model file')
    parser.add_argument('--endpoint-name', help='Name for the endpoint')
    parser.add_argument('--instance-type', default='ml.t2.medium',
                       help='SageMaker instance type for endpoint')
    parser.add_argument('--environment', choices=['development', 'production'],
                       default='development', help='Environment configuration')
    
    args = parser.parse_args()
    
    print("🚀 SageMaker Model Deployment")
    print("=" * 50)
    print(f"Environment: {args.environment}")
    print(f"Instance Type: {args.instance_type}")
    if args.model_path:
        print(f"Model Path: {args.model_path}")
    if args.endpoint_name:
        print(f"Endpoint Name: {args.endpoint_name}")
    
    try:
        # Initialize deployer
        deployer = ModelDeployer(args.environment)
        
        # Deploy model
        success = deployer.deploy_complete_pipeline(
            model_path=args.model_path,
            endpoint_name=args.endpoint_name,
            instance_type=args.instance_type
        )
        
        if success:
            print("\n🎉 Model deployment completed successfully!")
            return True
        else:
            print("\n❌ Model deployment failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Deployment failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)