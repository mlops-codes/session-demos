import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
    parser.add_argument("--n_estimators", required=False, default=100, type=int)
    parser.add_argument("--max_depth", required=False, default=10, type=int)
    parser.add_argument("--registered_model_name", type=str, help="model name")
    args = parser.parse_args()

    # Start MLflow run
    mlflow.start_run()
    mlflow.sklearn.autolog()

    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_train_ratio, random_state=42
    )

    # Log metrics
    mlflow.log_metric("num_samples", len(X))
    mlflow.log_metric("num_features", X.shape[1])
    mlflow.log_param("test_train_ratio", args.test_train_ratio)

    # Train model
    print(f"Training with data of shape {X_train.shape}")
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators, 
        max_depth=args.max_depth,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Evaluate model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))

    # Register model
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=clf,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name,
    )

    # Save model locally
    mlflow.sklearn.save_model(
        sk_model=clf,
        path=os.path.join(args.registered_model_name, "trained_model"),
    )

    mlflow.end_run()

if __name__ == "__main__":
    main()