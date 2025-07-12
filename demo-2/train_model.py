import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import joblib
import os

class IrisModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_names = None
        self.target_names = None
        
    def load_data(self):
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = iris.target
        self.feature_names = iris.feature_names
        self.target_names = iris.target_names
        return X, y
    
    def train_model(self):
        X, y = self.load_data()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.target_names))
        
        return accuracy
    
    def save_model(self, model_dir="models"):
        os.makedirs(model_dir, exist_ok=True)
        
        with open(f"{model_dir}/iris_model.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        
        joblib.dump(self.model, f"{model_dir}/iris_model.joblib")
        
        model_info = {
            'feature_names': self.feature_names,
            'target_names': self.target_names
        }
        
        with open(f"{model_dir}/model_info.pkl", 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"Model saved to {model_dir}/")
    
    def load_model(self, model_dir="models"):
        with open(f"{model_dir}/iris_model.pkl", 'rb') as f:
            self.model = pickle.load(f)
        
        with open(f"{model_dir}/model_info.pkl", 'rb') as f:
            model_info = pickle.load(f)
            self.feature_names = model_info['feature_names']
            self.target_names = model_info['target_names']
        
        print("Model loaded successfully")
    
    def predict(self, features):
        prediction = self.model.predict([features])
        probability = self.model.predict_proba([features])
        
        return {
            'prediction': int(prediction[0]),
            'species': self.target_names[prediction[0]],
            'probability': float(max(probability[0])),
            'all_probabilities': {
                name: float(prob) for name, prob in zip(self.target_names, probability[0])
            }
        }

if __name__ == "__main__":
    model = IrisModel()
    accuracy = model.train_model()
    model.save_model()
    
    sample_features = [5.1, 3.5, 1.4, 0.2]
    result = model.predict(sample_features)
    print(f"\nSample prediction: {result}")