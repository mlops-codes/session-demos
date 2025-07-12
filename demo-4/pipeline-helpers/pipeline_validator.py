#!/usr/bin/env python3
"""
Pipeline Output Validator
Validates pipeline outputs and artifacts for CI/CD
"""
import json
import os
import sys
from pathlib import Path


def validate_pipeline_outputs():
    """Validate pipeline outputs and artifacts"""
    print("🔍 Validating pipeline outputs...")
    
    # Check for required files
    required_files = [
        "cicd_pipeline_summary.json",
        "pipeline_artifacts/raw_data.csv",
        "pipeline_artifacts/pipeline_summary.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    # Validate summary file
    try:
        with open("cicd_pipeline_summary.json", "r") as f:
            summary = json.load(f)
        
        required_keys = ["pipeline_status", "best_accuracy", "deployment_ready"]
        missing_keys = [key for key in required_keys if key not in summary]
        
        if missing_keys:
            print(f"❌ Missing required keys in summary: {missing_keys}")
            return False
        
        # Validate values
        if summary["pipeline_status"] not in ["completed", "failed"]:
            print(f"❌ Invalid pipeline status: {summary['pipeline_status']}")
            return False
        
        if not isinstance(summary["best_accuracy"], (int, float)):
            print(f"❌ Invalid accuracy value: {summary['best_accuracy']}")
            return False
        
        if summary["best_accuracy"] < 0 or summary["best_accuracy"] > 1:
            print(f"❌ Accuracy out of range: {summary['best_accuracy']}")
            return False
        
        print(f"✅ Pipeline validation passed")
        print(f"   Status: {summary['pipeline_status']}")
        print(f"   Accuracy: {summary['best_accuracy']:.4f}")
        print(f"   Deployment Ready: {summary['deployment_ready']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating summary: {e}")
        return False


def main():
    """Main validation function"""
    success = validate_pipeline_outputs()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())