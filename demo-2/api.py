from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from typing import List

app = FastAPI(title="Iris Species Prediction API", version="1.0.0")

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    prediction: int
    species: str
    probability: float
    all_probabilities: dict

model = None
model_info = None

def load_model():
    global model, model_info
    try:
        with open("models/iris_model.pkl", 'rb') as f:
            model = pickle.load(f)
        
        with open("models/model_info.pkl", 'rb') as f:
            model_info = pickle.load(f)
        
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def root():
    return {"message": "Iris Species Prediction API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict_species(features: IrisFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        feature_array = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        
        prediction = model.predict(feature_array)
        probability = model.predict_proba(feature_array)
        
        result = {
            "prediction": int(prediction[0]),
            "species": model_info['target_names'][prediction[0]],
            "probability": float(max(probability[0])),
            "all_probabilities": {
                name: float(prob) for name, prob in zip(model_info['target_names'], probability[0])
            }
        }
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(features_list: List[IrisFeatures]):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        feature_arrays = []
        for features in features_list:
            feature_arrays.append([
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width
            ])
        
        feature_matrix = np.array(feature_arrays)
        predictions = model.predict(feature_matrix)
        probabilities = model.predict_proba(feature_matrix)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            result = {
                "prediction": int(pred),
                "species": model_info['target_names'][pred],
                "probability": float(max(prob)),
                "all_probabilities": {
                    name: float(p) for name, p in zip(model_info['target_names'], prob)
                }
            }
            results.append(result)
        
        return {"predictions": results}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    if model_info is None:
        raise HTTPException(status_code=500, detail="Model info not loaded")
    
    return {
        "feature_names": model_info['feature_names'].tolist(),
        "target_names": model_info['target_names'].tolist(),
        "model_type": "RandomForestClassifier"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)