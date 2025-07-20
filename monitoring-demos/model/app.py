#!/usr/bin/env python3
"""
ML Model API with Prometheus Monitoring
=====================================

A FastAPI application that serves an ML model with comprehensive monitoring
using Prometheus metrics for model performance and inference tracking.
"""

import time
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy import stats
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from fastapi.responses import Response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics - use try/except to handle duplicate registration
try:
    REQUEST_COUNT = Counter('ml_model_requests_total', 'Total number of requests', ['method', 'endpoint', 'status'])
    REQUEST_LATENCY = Histogram('ml_model_request_duration_seconds', 'Request latency in seconds', ['endpoint'])
    PREDICTION_COUNT = Counter('ml_model_predictions_total', 'Total number of predictions made')
    MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Current model accuracy')
    MODEL_PRECISION = Gauge('ml_model_precision', 'Current model precision')
    MODEL_RECALL = Gauge('ml_model_recall', 'Current model recall')
    ACTIVE_CONNECTIONS = Gauge('ml_model_active_connections', 'Number of active connections')
    PREDICTION_CONFIDENCE = Histogram('ml_model_prediction_confidence', 'Prediction confidence scores')
    ERROR_COUNT = Counter('ml_model_errors_total', 'Total number of errors', ['error_type'])
    
    # Data drift metrics
    FEATURE_MEAN = Gauge('ml_model_feature_mean', 'Mean of input features', ['feature_index'])
    FEATURE_STD = Gauge('ml_model_feature_std', 'Standard deviation of input features', ['feature_index'])
    KS_STATISTIC = Gauge('ml_model_ks_statistic', 'Kolmogorov-Smirnov test statistic for drift', ['feature_index'])
    PSI_SCORE = Gauge('ml_model_psi_score', 'Population Stability Index for drift detection', ['feature_index'])
    PREDICTION_DISTRIBUTION = Histogram('ml_model_prediction_distribution', 'Distribution of predictions', buckets=[0, 0.5, 1])
    DATA_QUALITY_SCORE = Gauge('ml_model_data_quality', 'Overall data quality score')
except ValueError:
    # Metrics already registered, get existing ones
    REQUEST_COUNT = REGISTRY._names_to_collectors['ml_model_requests_total']
    REQUEST_LATENCY = REGISTRY._names_to_collectors['ml_model_request_duration_seconds']
    PREDICTION_COUNT = REGISTRY._names_to_collectors['ml_model_predictions_total']
    MODEL_ACCURACY = REGISTRY._names_to_collectors['ml_model_accuracy']
    MODEL_PRECISION = REGISTRY._names_to_collectors['ml_model_precision']
    MODEL_RECALL = REGISTRY._names_to_collectors['ml_model_recall']
    ACTIVE_CONNECTIONS = REGISTRY._names_to_collectors['ml_model_active_connections']
    PREDICTION_CONFIDENCE = REGISTRY._names_to_collectors['ml_model_prediction_confidence']
    ERROR_COUNT = REGISTRY._names_to_collectors['ml_model_errors_total']
    
    # Get existing drift metrics
    FEATURE_MEAN = REGISTRY._names_to_collectors['ml_model_feature_mean']
    FEATURE_STD = REGISTRY._names_to_collectors['ml_model_feature_std']
    KS_STATISTIC = REGISTRY._names_to_collectors['ml_model_ks_statistic']
    PSI_SCORE = REGISTRY._names_to_collectors['ml_model_psi_score']
    PREDICTION_DISTRIBUTION = REGISTRY._names_to_collectors['ml_model_prediction_distribution']
    DATA_QUALITY_SCORE = REGISTRY._names_to_collectors['ml_model_data_quality']

app = FastAPI(title="ML Model Monitoring Demo", version="1.0.0")

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    timestamp: str

class ModelTrainer:
    """Simple model trainer for demonstration"""
    
    def __init__(self):
        self.model = None
        self.X_test = None
        self.y_test = None
        self.X_train_reference = None  # Reference data for drift detection
        self.recent_predictions = []  # Store recent inputs for drift analysis
        self.train_model()
    
    def train_model(self):
        """Train a simple classification model"""
        logger.info("Training model...")
        
        # Generate synthetic dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        
        # Split the data
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Store reference data for drift detection
        self.X_train_reference = X_train.copy()
        
        # Calculate initial metrics
        self.update_model_metrics()
        self.update_reference_statistics()
        
        logger.info("Model training completed")
    
    def update_model_metrics(self):
        """Update model performance metrics"""
        if self.model and self.X_test is not None:
            y_pred = self.model.predict(self.X_test)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            
            MODEL_ACCURACY.set(accuracy)
            MODEL_PRECISION.set(precision)
            MODEL_RECALL.set(recall)
            
            logger.info(f"Model metrics updated - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
    
    def update_reference_statistics(self):
        """Update reference statistics for drift detection"""
        if self.X_train_reference is not None:
            for i in range(self.X_train_reference.shape[1]):
                feature_data = self.X_train_reference[:, i]
                FEATURE_MEAN.labels(feature_index=str(i)).set(np.mean(feature_data))
                FEATURE_STD.labels(feature_index=str(i)).set(np.std(feature_data))
    
    def calculate_psi(self, reference_data, current_data, bins=10):
        """Calculate Population Stability Index"""
        # Create bins based on reference data
        _, bin_edges = np.histogram(reference_data, bins=bins)
        
        # Calculate expected percentages (reference)
        expected_percents = np.histogram(reference_data, bins=bin_edges)[0] / len(reference_data)
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)  # Avoid division by zero
        
        # Calculate actual percentages (current)
        actual_percents = np.histogram(current_data, bins=bin_edges)[0] / len(current_data)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)  # Avoid division by zero
        
        # Calculate PSI
        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return psi
    
    def detect_drift(self, new_features):
        """Detect data drift for new input"""
        if self.X_train_reference is None or len(new_features) != self.X_train_reference.shape[1]:
            return
        
        # Add to recent predictions buffer
        self.recent_predictions.append(new_features)
        if len(self.recent_predictions) > 100:  # Keep last 100 predictions
            self.recent_predictions.pop(0)
        
        # Only calculate drift if we have enough recent data
        if len(self.recent_predictions) >= 30:
            recent_array = np.array(self.recent_predictions)
            
            data_quality_issues = 0
            for i in range(len(new_features)):
                reference_feature = self.X_train_reference[:, i]
                recent_feature = recent_array[:, i]
                
                # Calculate KS statistic
                ks_stat, _ = stats.ks_2samp(reference_feature, recent_feature)
                KS_STATISTIC.labels(feature_index=str(i)).set(ks_stat)
                
                # Calculate PSI
                psi = self.calculate_psi(reference_feature, recent_feature)
                PSI_SCORE.labels(feature_index=str(i)).set(psi)
                
                # Update feature statistics
                FEATURE_MEAN.labels(feature_index=str(i)).set(np.mean(recent_feature))
                FEATURE_STD.labels(feature_index=str(i)).set(np.std(recent_feature))
                
                # Check for data quality issues
                if np.isnan(new_features[i]) or np.isinf(new_features[i]):
                    data_quality_issues += 1
            
            # Calculate overall data quality score
            quality_score = 1.0 - (data_quality_issues / len(new_features))
            DATA_QUALITY_SCORE.set(quality_score)

# Initialize model trainer
model_trainer = ModelTrainer()

@app.middleware("http")
async def monitor_requests(request, call_next):
    """Middleware to monitor all requests"""
    start_time = time.time()
    
    # Increment active connections
    ACTIVE_CONNECTIONS.inc()
    
    try:
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        return response
    
    except Exception as e:
        ERROR_COUNT.labels(error_type=type(e).__name__).inc()
        raise
    
    finally:
        # Decrement active connections
        ACTIVE_CONNECTIONS.dec()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ML Model Monitoring Demo",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model_trainer.model is not None,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "accuracy": MODEL_ACCURACY._value._value if MODEL_ACCURACY._value._value else 0,
            "precision": MODEL_PRECISION._value._value if MODEL_PRECISION._value._value else 0,
            "recall": MODEL_RECALL._value._value if MODEL_RECALL._value._value else 0
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction with monitoring"""
    try:
        if model_trainer.model is None:
            ERROR_COUNT.labels(error_type="ModelNotLoaded").inc()
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        if len(request.features) != 10:
            ERROR_COUNT.labels(error_type="InvalidInput").inc()
            raise HTTPException(status_code=400, detail="Expected 10 features")
        
        # Make prediction
        features = np.array(request.features).reshape(1, -1)
        prediction = model_trainer.model.predict(features)[0]
        probabilities = model_trainer.model.predict_proba(features)[0]
        confidence = float(max(probabilities))
        
        # Update metrics
        PREDICTION_COUNT.inc()
        PREDICTION_CONFIDENCE.observe(confidence)
        PREDICTION_DISTRIBUTION.observe(float(prediction))
        
        # Detect data drift
        model_trainer.detect_drift(request.features)
        
        return PredictionResponse(
            prediction=int(prediction),
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        ERROR_COUNT.labels(error_type=type(e).__name__).inc()
        logger.error(f"Prediction error: {str(e)}")
        raise

@app.post("/retrain")
async def retrain_model():
    """Retrain the model and update metrics"""
    try:
        model_trainer.train_model()
        return {
            "status": "success",
            "message": "Model retrained successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        ERROR_COUNT.labels(error_type=type(e).__name__).inc()
        logger.error(f"Retraining error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/stats")
async def get_stats():
    """Get current statistics"""
    return {
        "total_predictions": PREDICTION_COUNT._value._value,
        "active_connections": ACTIVE_CONNECTIONS._value._value,
        "model_metrics": {
            "accuracy": MODEL_ACCURACY._value._value if MODEL_ACCURACY._value._value else 0,
            "precision": MODEL_PRECISION._value._value if MODEL_PRECISION._value._value else 0,
            "recall": MODEL_RECALL._value._value if MODEL_RECALL._value._value else 0
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )