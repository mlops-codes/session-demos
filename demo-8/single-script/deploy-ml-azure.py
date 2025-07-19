import uuid
from config import ml_client
from azure.ai.ml import command, Input
from azure.ai.ml.entities import AmlCompute, ManagedOnlineEndpoint, ManagedOnlineDeployment
from datetime import datetime

def create_compute():
    """Create compute cluster in UK South and return its name"""
    cpu_compute_target = "cpu-cluster"

    try:
        cpu_cluster = ml_client.compute.get(cpu_compute_target)
        print(f"Found existing cluster: {cpu_cluster.name}")
    except Exception:
        print("Creating new compute cluster…")
        cpu_cluster = AmlCompute(
            name=cpu_compute_target,
            type="amlcompute",
            size="STANDARD_E8S_V3",
            min_instances=0,
            max_instances=2,
            idle_time_before_scale_down=180,
            tier="Dedicated",
            location="uksouth"
        )
        cpu_cluster = ml_client.compute.begin_create_or_update(cpu_cluster).result()
        print(f"Created cluster: {cpu_cluster.name}")

    # ← return the name, not the object
    return cpu_cluster.name


def submit_training_job(compute_name: str):
    """Submit a training job and wait for completion."""
    job = command(
        inputs={
            "test_train_ratio": 0.2,
            "n_estimators": 50,
            "max_depth": 10,
            "registered_model_name": "iris_model",
        },
        code="./src/",
        command=(
            "python main.py "
            "--test_train_ratio ${{inputs.test_train_ratio}} "
            "--n_estimators ${{inputs.n_estimators}} "
            "--max_depth ${{inputs.max_depth}} "
            "--registered_model_name ${{inputs.registered_model_name}}"
        ),
        environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",
        compute=compute_name,
        experiment_name="iris-training-experiment",
        display_name="iris_classification_training",
    )

    # submit and stream logs
    job = ml_client.jobs.create_or_update(job)
    print(f"Job submitted: {job.name}")
    print("Waiting for training job to complete…")
    ml_client.jobs.stream(job.name)

    # check final status
    completed = ml_client.jobs.get(job.name)
    print(f"Job status: {completed.status}")
    if completed.status != "Completed":
        raise Exception(f"Training job failed: {completed.status}")
    print("✅ Training job completed successfully")
    return job.name


def list_registered_models():
    """List all registered models with None-safe version handling"""
    try:
        models = list(ml_client.models.list())
        print("\nRegistered models:")
        if not models:
            print("  No models found")
            return []
        
        model_info = []
        for model in models:
            version_str = str(model.version) if model.version is not None else "None"
            print(f"  - {model.name}:{version_str}")
            model_info.append((model.name, model.version))
        return model_info
    except Exception as e:
        print(f"Error listing models: {e}")
        return []

def create_endpoint():
    endpoint_name = "iris-endpoint-" + str(uuid.uuid4())[:8]
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Iris classification endpoint",
        auth_mode="key",
        tags={"model_type": "sklearn.RandomForestClassifier"},
        location="uksouth"            # ← UK South
    )
    return ml_client.online_endpoints.begin_create_or_update(endpoint).result().name

def deploy_model(endpoint_name):
    """Deploy the most-recent iris_model, handling VM-size quota and SKU support."""
    from datetime import datetime
    # 1) grab the latest iris_model
    models_list = list(ml_client.models.list(name="iris_model"))
    if not models_list:
        raise Exception("No iris_model registrations found.")
    def created_at(m):
        return getattr(m.creation_context, "created_at", datetime(1970,1,1))
    model = max(models_list, key=created_at)
    print(f"✅ Using iris_model created at {model.creation_context.created_at}")

    # 2) helper to deploy on a given SKU
    def try_deploy(vm_size):
        dep = ManagedOnlineDeployment(
            name="blue",
            endpoint_name=endpoint_name,
            model=model,
            instance_type=vm_size,
            instance_count=1,
        )
        print(f"🚀 Trying {vm_size}…")
        return ml_client.online_deployments.begin_create_or_update(dep).result()

    # 3) fallback sequence with correct names
    for sku in ("Standard_DS3_v2", "Standard_E4s_v3", "Standard_E2s_v3"):
        try:
            deployment = try_deploy(sku)
            chosen = sku
            break
        except Exception as e:
            if "not enough quota" in str(e).lower() or "not supported" in str(e).lower():
                print(f"⚠️ {sku} failed—falling back…")
                continue
            raise

    # 4) route 100% traffic
    ep = ml_client.online_endpoints.get(endpoint_name)
    ep.traffic = {"blue": 100}
    ml_client.online_endpoints.begin_create_or_update(ep).result()

    print(f"✅ Model deployed successfully on {chosen}")
    return endpoint_name


def test_endpoint(endpoint_name):
    """Test the deployed endpoint"""
    # Sample iris data: [sepal_length, sepal_width, petal_length, petal_width]
    test_data = {
        "data": [
            [5.1, 3.5, 1.4, 0.2],  # Setosa
            [6.2, 3.4, 5.4, 2.3],  # Virginica
            [5.8, 2.7, 4.1, 1.0]   # Versicolor
        ]
    }
    
    response = ml_client.online_endpoints.invoke(
        endpoint_name=endpoint_name,
        deployment_name="blue",
        request_file=test_data
    )
    
    print(f"Prediction results: {response}")
    return response

if __name__ == "__main__":
    # Complete workflow
    print("Step 1: Creating compute cluster...")
    compute_name = create_compute()
    
    print("\nStep 2: Submitting training job...")
    job = submit_training_job(compute_name)
    
    print("\nStep 3: Creating endpoint...")
    endpoint_name = create_endpoint()
    
    print("\nStep 4: Deploying model...")
    deploy_model(endpoint_name)
    
    print("\nStep 5: Testing endpoint...")
    test_endpoint(endpoint_name)
    
    print(f"\nDeployment complete! Endpoint name: {endpoint_name}")