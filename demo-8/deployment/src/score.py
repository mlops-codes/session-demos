import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model, scaler, feature_names
    
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = os.environ.get("AZUREML_MODEL_DIR")
    
    if model_path is None:
        # For local testing
        model_path = "../outputs"
    
    try:
        # Load the model and scaler
        model = joblib.load(os.path.join(model_path, "outputs" ,"model.pkl"))
        scaler = joblib.load(os.path.join(model_path, "outputs", "scaler.pkl"))
        
        # Load metadata to get feature names
        with open(os.path.join(model_path, "outputs", "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        # Create feature names (assuming they follow the pattern from training)
        n_features = metadata['data_shape']['features']
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        logging.info(f"Model loaded successfully from {model_path}")
        logging.info(f"Model type: {metadata['model_type']}")
        logging.info(f"Expected features: {len(feature_names)}")
        
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise


def run(raw_data: str) -> str:
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the base image, this function is called after the init() function.
    """
    try:
        # Parse the input data
        data = json.loads(raw_data)
        
        # Handle different input formats
        if "data" in data:
            # Format: {"data": [[1,2,3,...], [4,5,6,...]]}
            input_data = np.array(data["data"])
        elif "instances" in data:
            # Format: {"instances": [{"feature_0": 1, "feature_1": 2, ...}, ...]}
            instances = data["instances"]
            input_data = pd.DataFrame(instances)[feature_names].values
        elif isinstance(data, list):
            # Format: [[1,2,3,...], [4,5,6,...]]
            input_data = np.array(data)
        else:
            raise ValueError("Unsupported input format. Expected 'data', 'instances', or direct array.")
        
        # Ensure input is 2D
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        
        # Validate input shape
        if input_data.shape[1] != len(feature_names):
            raise ValueError(f"Expected {len(feature_names)} features, got {input_data.shape[1]}")
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make predictions
        predictions = model.predict(input_scaled)
        probabilities = model.predict_proba(input_scaled)
        
        # Format the response
        results = []
        for i in range(len(predictions)):
            result = {
                "prediction": int(predictions[i]),
                "probability": {
                    "class_0": float(probabilities[i][0]),
                    "class_1": float(probabilities[i][1])
                },
                "confidence": float(max(probabilities[i]))
            }
            results.append(result)
        
        response = {
            "predictions": results,
            "model_info": {
                "features_used": len(feature_names),
                "samples_processed": len(predictions)
            }
        }
        
        return json.dumps(response)
        
    except Exception as e:
        error_response = {
            "error": str(e),
            "error_type": type(e).__name__
        }
        logging.error(f"Prediction error: {e}")
        return json.dumps(error_response)


# For local testing
if __name__ == "__main__":
    # Initialize the model
    init()
    
    # Test with sample data
    test_data = {
        "data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]]
    }
    
    result = run(json.dumps(test_data))
    print("Test result:")
    print(json.dumps(json.loads(result), indent=2))