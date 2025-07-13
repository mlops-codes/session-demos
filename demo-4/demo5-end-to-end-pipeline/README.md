# Demo 5: End-to-End MLflow Pipeline with CI/CD

## Overview

This demo showcases a complete end-to-end ML pipeline using MLflow with comprehensive CI/CD integration. It demonstrates how to build, test, and deploy machine learning models in a production-ready environment.

## Features

### 🚀 Complete ML Pipeline
- **Data Generation**: Synthetic loan approval dataset with realistic features
- **Data Preprocessing**: Feature scaling and train/test splitting
- **Model Training**: Multiple model comparison (Random Forest, Logistic Regression)
- **Model Evaluation**: Comprehensive performance metrics and validation
- **Deployment Preparation**: Production-ready model packaging

### 🔄 CI/CD Integration
- **GitHub Actions**: Automated pipeline execution on code changes
- **Environment-specific**: Different configurations for dev/staging/production
- **Quality Gates**: Automated quality checks and thresholds
- **Artifact Management**: Automatic artifact storage and versioning

### 📊 MLflow Integration
- **Experiment Tracking**: Nested runs for each pipeline stage
- **Model Registry**: Automatic model registration and versioning
- **Artifact Logging**: Comprehensive logging of datasets, models, and reports
- **Metadata Tracking**: Complete audit trail of pipeline execution

## Quick Start

### 1. Install Dependencies

```bash
cd demo-4/demo5-end-to-end-pipeline
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Run the complete pipeline
python pipeline.py

# Or run the test version (smaller dataset)
python test_pipeline.py
```

### 3. View Results

```bash
# Start MLflow UI
mlflow ui

# Navigate to http://localhost:5000
# Check experiment: "complete_ml_pipeline"
```

## Pipeline Stages

### 📊 Stage 1: Data Preparation
- Generates synthetic loan approval dataset
- Features: income, credit score, loan amount, employment history, etc.
- Target: Binary loan approval decision (0=Denied, 1=Approved)
- Data quality validation and reporting

**Key Features:**
```python
# Environment-controlled data generation
n_samples = int(os.getenv("N_SAMPLES", "2000"))

# Realistic loan approval logic
approval_prob = (
    0.3 * (credit_score - 300) / (850 - 300) +  # Credit score influence
    0.2 * (income - 15000) / (200000 - 15000) +  # Income influence
    0.2 * (1 - debt_to_income) +  # Lower debt-to-income is better
    # ... more factors
)
```

### 🔄 Stage 2: Preprocessing
- Train/test split with stratification
- Feature scaling using StandardScaler
- Artifact logging for reproducibility

### 🏗️ Stage 3: Model Training
- Multiple model comparison
- Hyperparameter configuration
- Performance tracking and logging

**Models Trained:**
- Random Forest (baseline)
- Random Forest (tuned)
- Logistic Regression

### 📊 Stage 4: Evaluation
- Comprehensive metrics calculation
- Confidence analysis
- Classification reports and confusion matrices

### 🚀 Stage 5: Deployment Preparation
- Model packaging for production
- Inference script generation
- Deployment metadata creation

## CI/CD Workflows

### Main Workflow: `end-to-end-pipeline.yml`

**Features:**
- Environment detection (development/staging/production)
- Quality gate enforcement
- Artifact management
- Deployment readiness checks

**Usage:**
```bash
# Manual trigger with specific environment
gh workflow run end-to-end-pipeline.yml \
  -f environment=staging \
  -f pipeline_stages=all \
  -f force_run=true
```

### Simple Workflow: `simple-pipeline.yml`

**Features:**
- Basic pipeline testing
- Artifact upload
- Simplified for debugging

## Environment Configuration

### Environment Variables

```bash
# Data generation
export N_SAMPLES=2000                    # Dataset size
export MIN_ACCURACY_THRESHOLD=0.90       # Quality gate threshold
export ENABLE_HYPERPARAMETER_TUNING=true # Enable tuning

# CI/CD integration
export GITHUB_SHA=abc123                 # Git commit hash
export GITHUB_ACTOR=username            # Triggering user
```

### Environment-Specific Settings

| Environment | Samples | Accuracy Threshold | Hyperparameter Tuning |
|-------------|---------|-------------------|----------------------|
| Development | 1000    | 0.85              | Disabled            |
| Staging     | 2000    | 0.90              | Enabled             |
| Production  | 5000    | 0.95              | Enabled             |

## Pipeline Output

### Artifacts Generated

```
pipeline_artifacts/
├── raw_data.csv                    # Generated dataset
├── data_validation_report.json     # Data quality metrics
├── X_train_processed.csv          # Processed training data
├── X_test_processed.csv           # Processed test data
├── y_train.csv                    # Training labels
├── y_test.csv                     # Test labels
├── scaler.pkl                     # Feature scaler
├── model_*.pkl                    # Trained models
├── model_selection_report.json    # Model comparison
├── evaluation_report.json         # Performance metrics
├── predictions.csv                # Model predictions
├── deployment_package/            # Production-ready package
│   ├── final_model.pkl
│   ├── final_scaler.pkl
│   ├── model_metadata.json
│   └── inference.py
└── pipeline_summary.json          # Complete pipeline summary
```

### Pipeline Summary Example

```json
{
  "pipeline_name": "complete_ml_pipeline",
  "run_id": "abc123def456",
  "status": "completed",
  "stages": {
    "data_preparation": "completed",
    "preprocessing": "completed", 
    "model_training": "completed",
    "evaluation": "completed",
    "deployment_prep": "completed"
  },
  "best_model": "random_forest_tuned",
  "final_accuracy": 0.9234,
  "data_quality_score": 1.0,
  "deployment_ready": true
}
```

## Model Registry Integration

### Registered Models

The pipeline automatically registers models in MLflow:

- `loan_approval_pipeline_random_forest`
- `loan_approval_pipeline_random_forest_tuned` 
- `loan_approval_pipeline_logistic_regression`
- `loan_approval_pipeline_best_model` (final deployment model)

### Model Metadata

Each registered model includes:
- Training parameters and hyperparameters
- Performance metrics (accuracy, F1, precision, recall)
- Data quality information
- Pipeline run information
- Deployment readiness status

## Deployment

### Production Inference

The pipeline generates a complete deployment package:

```python
# Load and use the model
from deployment_package.inference import ModelInference

# Initialize inference
inference = ModelInference(
    "deployment_package/final_model.pkl",
    "deployment_package/final_scaler.pkl"
)

# Make predictions
features = [[50000, 700, 120000, 5, 0.35, 8, 1, 1, 0]]  # Loan features
result = inference.predict(features)

print(result)
# {
#   "prediction": [1],           # 1=Approved, 0=Denied
#   "probabilities": [[0.2, 0.8]], # [Denied, Approved]
#   "confidence": [0.8]          # Confidence score
# }
```

### Docker Deployment

```bash
# Build Docker image (example)
docker build -t loan-approval-model .

# Run inference server
docker run -p 8080:8080 loan-approval-model
```

## Testing

### Unit Testing

```bash
# Test individual components
python test_pipeline.py
```

### Integration Testing

```bash
# Test with different environments
N_SAMPLES=100 MIN_ACCURACY_THRESHOLD=0.7 python pipeline.py
```

### CI/CD Testing

```bash
# Trigger GitHub Actions workflow
git push origin main
```

## Monitoring and Observability

### MLflow Tracking

- **Experiments**: Organized by pipeline name and timestamp
- **Runs**: Nested structure showing pipeline stages
- **Metrics**: Performance tracking across all stages
- **Artifacts**: Complete artifact lineage

### Performance Monitoring

```python
# Key metrics tracked
- Data quality score
- Model accuracy, precision, recall, F1
- Prediction confidence levels
- Training/validation performance gaps
- Pipeline execution time
```

## Troubleshooting

### Common Issues

1. **MLflow Connection**: Ensure MLflow tracking server is accessible
2. **Dependencies**: Install requirements with `pip install -r requirements.txt`
3. **Environment Variables**: Set required environment variables for CI/CD
4. **Disk Space**: Pipeline generates artifacts; ensure adequate storage

### Debug Commands

```bash
# Check MLflow experiments
mlflow experiments list

# Verify pipeline artifacts
ls -la pipeline_artifacts/

# Test individual stages
python -c "from pipeline import MLflowPipeline; p = MLflowPipeline(); p.stage_1_data_preparation()"
```

### CI/CD Debugging

```bash
# Check workflow status
gh workflow list

# View workflow logs
gh run view --log

# Manual workflow trigger
gh workflow run end-to-end-pipeline.yml -f force_run=true
```

## Best Practices Demonstrated

### 1. **Modular Pipeline Design**
- Each stage is independent and testable
- Clear input/output contracts
- Comprehensive error handling

### 2. **Comprehensive Logging**
- All parameters, metrics, and artifacts logged
- Nested run structure for clarity
- Complete audit trail

### 3. **Quality Gates**
- Data quality validation
- Model performance thresholds
- Deployment readiness checks

### 4. **Environment Management**
- Environment-specific configurations
- Parameterized pipeline execution
- CI/CD integration

### 5. **Production Readiness**
- Complete deployment package
- Inference script generation
- Model metadata and versioning

## Next Steps

1. **Extend Pipeline**: Add more sophisticated models or feature engineering
2. **Enhanced CI/CD**: Add deployment automation and monitoring
3. **Model Monitoring**: Implement drift detection and retraining triggers
4. **A/B Testing**: Add support for model comparison in production
5. **Scaling**: Adapt for distributed training and inference

## Key Takeaways

This demo showcases enterprise-grade MLOps practices:
- **End-to-end automation** from data to deployment
- **Comprehensive tracking** and versioning
- **Quality-driven** development with automated gates
- **Production-ready** deployment artifacts
- **CI/CD integration** for continuous delivery

The implementation demonstrates how to build reliable, scalable ML systems with proper governance, testing, and deployment practices.