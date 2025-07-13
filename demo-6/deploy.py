#!/usr/bin/env python3
"""
SageMaker Model Deployment Script for Flight Delay Prediction Model

This script deploys a trained pickle model to Amazon SageMaker for -time inference.
"""

import boto3
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import tarfile
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

class FlightDelayModelDeployer:
    def __init__(self, model_path, role, bucket_name, region='eu-west-2'):
        """
        Initialize the SageMaker deployer
        
        Args:
            model_path: Local path to the trained model pickle file
            role: SageMaker execution role ARN
            bucket_name: S3 bucket for storing model artifacts
            region: AWS region
        """
        self.model_path = model_path
        self.role = role
        self.bucket_name = bucket_name
        self.region = region
        
        # Initialize SageMaker session
        self.sagemaker_session = sagemaker.Session()
        self.boto_session = boto3.Session(region_name=region)
        
        # Model configuration
        self.model_name = f"flight-delay-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.endpoint_config_name = f"flight-delay-config-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.endpoint_name = f"flight-delay-endpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
    def prepare_model_artifacts(self):
        """
        Prepare model artifacts for SageMaker deployment
        """
        import shutil
        import tempfile
        import time

        print("📦 Preparing model artifacts...")

        # Use a temporary directory
        temp_dir = tempfile.mkdtemp()
        model_file_path = os.path.join(temp_dir, "model.pkl")
        tar_path = os.path.join(temp_dir, "model.tar.gz")

        shutil.copy2(self.model_path, model_file_path)

        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(temp_dir, arcname=".")

        # Upload to S3
        s3_model_path = self.sagemaker_session.upload_data(
            path=tar_path,
            bucket=self.bucket_name,
            key_prefix="flight-delay-model"
        )

        print(f"✅ Model uploaded to: {s3_model_path}")

        # Wait briefly to release any lingering file locks
        time.sleep(1)

        # Try cleanup
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"⚠️ Warning during cleanup: {e}")

        return s3_model_path

        
    def deploy_model(self, instance_type="ml.t2.medium", initial_instance_count=1):
        """
        Deploy the model to SageMaker endpoint
        
        Args:
            instance_type: EC2 instance type for hosting
            initial_instance_count: Number of instances
        """
        print("🚀 Deploying model to SageMaker...")
        
        # Prepare model artifacts
        model_data = self.prepare_model_artifacts()
        
        # Create SKLearn model
        model = SKLearnModel(
            model_data=model_data,
            role=self.role,
            entry_point="inference.py",
            framework_version="0.23-1",
            py_version="py3",
            sagemaker_session=self.sagemaker_session
        )
        
        # Deploy model
        predictor = model.deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type,
            endpoint_name=self.endpoint_name,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )
        
        print(f"✅ Model deployed successfully!")
        print(f"📍 Endpoint name: {self.endpoint_name}")
        
        return predictor
        
    def test_endpoint(self, predictor=None):
        """
        Test the deployed endpoint with sample data
        """
        if predictor is None:
            predictor = sagemaker.predictor.Predictor(
                endpoint_name=self.endpoint_name,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer()
            )
        
        print("🧪 Testing endpoint with sample data...")
        
        # Sample test data
        test_data = {
            "instances": [
                {
                    "weather_encoded": 1,  # Rain
                    "departure_hour": 18,  # Rush hour
                    "airline_encoded": 2,  # Airline_C (worst)
                    "distance_miles": 1200,
                    "is_rush_hour": 1
                },
                {
                    "weather_encoded": 0,  # Clear
                    "departure_hour": 10,  # Non-rush hour
                    "airline_encoded": 0,  # Airline_A (best)
                    "distance_miles": 500,
                    "is_rush_hour": 0
                }
            ]
        }
        
        # Make prediction
        response = predictor.predict(test_data)
        
        print("📊 Prediction results:")
        print(f"   Sample 1 (Bad weather, rush hour, worst airline): {response['predictions'][0]}")
        print(f"   Sample 2 (Good weather, non-rush, best airline): {response['predictions'][1]}")
        
        return response
        
    def delete_endpoint(self):
        """
        Delete the SageMaker endpoint to avoid charges
        """
        print("🗑️ Deleting endpoint...")
        
        client = self.boto_session.client('sagemaker')
        
        try:
            # Delete endpoint
            client.delete_endpoint(EndpointName=self.endpoint_name)
            print(f"✅ Endpoint {self.endpoint_name} deleted")
            
            # Delete endpoint configuration
            client.delete_endpoint_config(EndpointConfigName=self.endpoint_config_name)
            print(f"✅ Endpoint config {self.endpoint_config_name} deleted")
            
            # Delete model
            client.delete_model(ModelName=self.model_name)
            print(f"✅ Model {self.model_name} deleted")
            
        except Exception as e:
            print(f"❌ Error during cleanup: {str(e)}")

def main():
    """
    Main deployment function
    """
    # Configuration - Update these with your values
    MODEL_PATH = "./model/model.pkl"
    ROLE = "arn:aws:iam::750952118292:role/service-role/AmazonSageMaker-ExecutionRole-20250615T204995"  # Update this
    BUCKET_NAME = "demo-sagemaker-1207-adasdadasdadsad"  # Update this
    
    # Validate inputs
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        return
        
    if "YOUR_ACCOUNT" in ROLE:
        print("❌ Please update the ROLE with your actual SageMaker execution role ARN")
        return
        
    if BUCKET_NAME == "your-sagemaker-bucket":
        print("❌ Please update BUCKET_NAME with your actual S3 bucket name")
        return
    
    # Deploy model
    deployer = FlightDelayModelDeployer(MODEL_PATH, ROLE, BUCKET_NAME)
    
    try:
        # Deploy
        predictor = deployer.deploy_model()
        
        # Test
        deployer.test_endpoint(predictor)
        
        print("\n🎉 Deployment completed successfully!")
        print(f"🔗 Endpoint name: {deployer.endpoint_name}")
        print("\n💡 Remember to delete the endpoint when done to avoid charges:")
        print(f"   deployer.delete_endpoint()")
        
    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        # Try to cleanup
        deployer.delete_endpoint()

if __name__ == "__main__":
    main()