#!/usr/bin/env python3
"""
CI/CD Pipeline Runner
Executes MLflow pipeline with environment variables and CI/CD integration
"""
import os
import sys
import json
from pathlib import Path

# Add parent directory to path to import pipeline
sys.path.append(str(Path(__file__).parent.parent / "demo5-end-to-end-pipeline"))
from pipeline import MLflowPipeline


def main():
    """Main CI/CD pipeline execution function"""
    try:
        # Get environment variables
        environment = os.getenv("ENVIRONMENT", "development")
        pipeline_id = os.getenv("PIPELINE_ID", "default_pipeline")
        
        # Set up pipeline with environment-specific name
        pipeline_name = f"{pipeline_id}_{environment}"
        pipeline = MLflowPipeline(pipeline_name)
        
        # Override environment variables for data generation
        original_env = {}
        env_overrides = {
            "N_SAMPLES": os.getenv("N_SAMPLES", "2000"),
            "MIN_ACCURACY_THRESHOLD": os.getenv("MIN_ACCURACY_THRESHOLD", "0.90"),
            "ENABLE_HYPERPARAMETER_TUNING": os.getenv("ENABLE_HYPERPARAMETER_TUNING", "false")
        }
        
        # Apply environment overrides
        for key, value in env_overrides.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        print(f"🚀 Starting pipeline: {pipeline_name}")
        print(f"📊 Environment: {environment}")
        print(f"🔧 Configuration: {env_overrides}")
        
        # Run the complete pipeline
        summary = pipeline.run_complete_pipeline()
        
        # Extract key metrics
        deployment_ready = summary.get("deployment_ready", False)
        best_accuracy = summary.get("final_accuracy", 0.0)
        pipeline_status = summary.get("status", "unknown")
        
        # Check quality gates
        min_threshold = float(os.getenv("MIN_ACCURACY_THRESHOLD", "0.90"))
        quality_gate_passed = best_accuracy >= min_threshold
        
        print(f"\n📊 Pipeline Results:")
        print(f"   Status: {pipeline_status}")
        print(f"   Best Accuracy: {best_accuracy:.4f}")
        print(f"   Quality Gate: {'PASSED' if quality_gate_passed else 'FAILED'} (>= {min_threshold})")
        print(f"   Deployment Ready: {deployment_ready}")
        
        # Create CI/CD summary
        cicd_summary = {
            "pipeline_name": pipeline_name,
            "environment": environment,
            "pipeline_status": pipeline_status,
            "best_accuracy": best_accuracy,
            "quality_gate_passed": quality_gate_passed,
            "quality_threshold": min_threshold,
            "deployment_ready": deployment_ready and quality_gate_passed,
            "github_sha": os.getenv("GITHUB_SHA", ""),
            "github_actor": os.getenv("GITHUB_ACTOR", ""),
            "pipeline_stages_completed": summary.get("stages", {}),
            "artifacts_location": "pipeline_artifacts/"
        }
        
        # Save CI/CD summary
        with open("cicd_pipeline_summary.json", "w") as f:
            json.dump(cicd_summary, f, indent=2)
        
        # Output for GitHub Actions
        github_output = os.getenv('GITHUB_OUTPUT')
        if github_output:
            with open(github_output, 'a') as f:
                f.write(f"status={pipeline_status}\n")
                f.write(f"best-accuracy={best_accuracy}\n")
                f.write(f"deployment-ready={cicd_summary['deployment_ready']}\n")
                f.write(f"quality-gate-passed={quality_gate_passed}\n")
        
        # Exit with error if quality gate failed
        if not quality_gate_passed:
            print(f"❌ Quality gate failed! Accuracy {best_accuracy:.4f} < {min_threshold}")
            sys.exit(1)
        
        print(f"✅ Pipeline completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        # Save error summary
        error_summary = {
            "pipeline_name": pipeline_name if 'pipeline_name' in locals() else "unknown",
            "environment": environment if 'environment' in locals() else "unknown",
            "pipeline_status": "failed",
            "error_message": str(e),
            "deployment_ready": False
        }
        with open("cicd_pipeline_summary.json", "w") as f:
            json.dump(error_summary, f, indent=2)
        
        github_output = os.getenv('GITHUB_OUTPUT')
        if github_output:
            with open(github_output, 'a') as f:
                f.write(f"status=failed\n")
                f.write(f"deployment-ready=false\n")
        
        sys.exit(1)
    
    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]


if __name__ == "__main__":
    main()