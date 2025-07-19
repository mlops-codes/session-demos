
import joblib
import pandas as pd
import numpy as np
import json
import boto3
import os
from io import StringIO

# Feature Store configuration
FEATURE_GROUPS = {feature_groups_used}
FEATURE_COLUMNS = {feature_columns}

def model_fn(model_dir):
    """Load model for inference"""
    model_path = os.path.join(model_dir, "model.pkl")
    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    """Parse input for inference"""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        
        # Handle both single prediction and batch prediction
        if isinstance(input_data, dict):
            # Single prediction
            if 'customer_id' in input_data:
                # Retrieve features from Feature Store
                features = retrieve_features_from_store(input_data['customer_id'])
                return features
            else:
                # Direct features provided
                return pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            # Batch prediction
            return pd.DataFrame(input_data)
    else:
        raise ValueError(f"Unsupported content type: {{request_content_type}}")

def retrieve_features_from_store(customer_id):
    """Retrieve features from Feature Store for -time inference"""
    try:
        # Initialize Feature Store runtime client
        featurestore_runtime = boto3.client('sagemaker-featurestore-runtime')
        
        features = {{'customer_id': customer_id}}
        
        # Retrieve from each feature group
        for fg_name in FEATURE_GROUPS:
            try:
                response = featurestore_runtime.get_record(
                    FeatureGroupName=fg_name,
                    RecordIdentifierValueAsString=str(customer_id)
                )
                
                if 'Record' in response:
                    for feature in response['Record']:
                        feature_name = feature['FeatureName']
                        feature_value = feature['ValueAsString']
                        
                        # Convert to appropriate type
                        if feature_name in FEATURE_COLUMNS:
                            try:
                                features[feature_name] = float(feature_value)
                            except ValueError:
                                features[feature_name] = feature_value
                                
            except Exception as e:
                print(f"Warning: Could not retrieve from {{fg_name}}: {{str(e)}}")
        
        # Ensure all required features are present
        for col in FEATURE_COLUMNS:
            if col not in features:
                features[col] = 0.0  # Default value for missing features
        
        # Return as DataFrame
        feature_df = pd.DataFrame([features])
        return feature_df[FEATURE_COLUMNS]
        
    except Exception as e:
        print(f"Error retrieving features: {{str(e)}}")
        # Return default features if retrieval fails
        default_features = {{col: 0.0 for col in FEATURE_COLUMNS}}
        return pd.DataFrame([default_features])

def predict_fn(input_data, model):
    """Make predictions"""
    try:
        # Ensure input_data has the right columns
        if not all(col in input_data.columns for col in FEATURE_COLUMNS):
            missing_cols = [col for col in FEATURE_COLUMNS if col not in input_data.columns]
            print(f"Warning: Missing columns {{missing_cols}}, filling with 0")
            for col in missing_cols:
                input_data[col] = 0.0
        
        # Select only the features used for training
        feature_data = input_data[FEATURE_COLUMNS].fillna(0)
        
        # Make predictions
        predictions = model.predict(feature_data)
        probabilities = model.predict_proba(feature_data)
        
        # Format results
        results = []
        for i in range(len(predictions)):
            result = {{
                'prediction': 'FRAUD' if predictions[i] == 1 else 'NORMAL',
                'fraud_probability': float(probabilities[i][1]),
                'normal_probability': float(probabilities[i][0]),
                'risk_level': get_risk_level(probabilities[i][1])
            }}
            results.append(result)
        
        return results
        
    except Exception as e:
        return [{{'error': str(e)}}]

def get_risk_level(fraud_probability):
    """Convert probability to risk level"""
    if fraud_probability < 0.2:
        return 'LOW'
    elif fraud_probability < 0.5:
        return 'MEDIUM'
    elif fraud_probability < 0.8:
        return 'HIGH'
    else:
        return 'CRITICAL'

def output_fn(prediction, content_type):
    """Format output"""
    if content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {{content_type}}")