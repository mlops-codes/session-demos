#!/usr/bin/env python3
"""
Demo 5: End-to-End MLflow Pipeline

This demo shows:
1. Complete ML pipeline with MLflow Projects integration
2. Multi-stage pipeline with nested runs
3. Data preparation → Training → Evaluation → Deployment
4. Automated model selection and registry integration
"""

import subprocess
import sys
import os

def main():
    """Main demo runner"""
    print("🎯 MLflow Demo 5: End-to-End Pipeline")
    print("=" * 45)
    
    # Change to demo directory
    os.chdir('demo-4/demo5-end-to-end-pipeline')
    
    print("📦 Installing requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "../requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return
    
    print("\n🚀 Running end-to-end pipeline...")
    try:
        result = subprocess.run([sys.executable, "pipeline.py"], 
                              capture_output=True, text=True, check=True)
        print("✅ End-to-end pipeline completed successfully!")
        print("\n📊 Output:")
        print(result.stdout)
        
        if result.stderr:
            print("\n⚠️  Warnings:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running pipeline: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return
    
    print("\n" + "=" * 45)
    print("🎉 Demo completed successfully!")
    print("\n💡 Next steps:")
    print("1. View results: mlflow ui --backend-store-uri ./mlruns")
    print("2. Explore experiment 'complete_ml_pipeline'")
    print("3. Check nested runs for each pipeline stage")
    print("4. View registered models in Models tab")
    print("5. Download deployment package artifacts")
    print("=" * 45)

if __name__ == "__main__":
    main()