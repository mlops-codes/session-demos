#!/usr/bin/env python3
"""
Test script for the end-to-end pipeline
"""
import os
import sys
import tempfile
import shutil
from pipeline import MLflowPipeline

def test_pipeline():
    """Test the pipeline execution"""
    print("🧪 Testing End-to-End Pipeline...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        
        # Set environment variables for testing
        test_env = {
            "N_SAMPLES": "500",  # Small dataset for testing
            "MIN_ACCURACY_THRESHOLD": "0.75",
            "ENABLE_HYPERPARAMETER_TUNING": "false"
        }
        
        # Apply test environment
        for key, value in test_env.items():
            os.environ[key] = value
        
        try:
            # Create and run pipeline
            pipeline = MLflowPipeline("test_pipeline")
            summary = pipeline.run_complete_pipeline()
            
            # Validate results
            assert summary["status"] == "completed", "Pipeline should complete successfully"
            assert summary["final_accuracy"] > 0.0, "Should have a valid accuracy"
            assert summary["deployment_ready"] == True, "Should be deployment ready"
            
            print("✅ Pipeline test passed!")
            print(f"   - Final accuracy: {summary['final_accuracy']:.4f}")
            print(f"   - Best model: {summary['best_model']}")
            print(f"   - Deployment ready: {summary['deployment_ready']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline test failed: {e}")
            return False
        
        finally:
            # Restore environment
            for key in test_env.keys():
                if key in os.environ:
                    del os.environ[key]

def main():
    """Main test function"""
    success = test_pipeline()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()