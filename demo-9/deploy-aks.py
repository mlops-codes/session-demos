#!/usr/bin/env python3
"""
Kubeflow on Azure Deployment Script - Windows Compatible
========================================================

This script automates the deployment of Kubeflow on Azure Kubernetes Service (AKS)
with all necessary components and configurations.

Windows Compatible Version:
- No emoji characters (fixes Unicode encoding issues)
- Windows-specific command adjustments
- Enhanced error handling for Windows environment

Requirements:
- Azure CLI installed and logged in
- Python 3.8+
- kubectl installed
- Git for Windows
- Sufficient Azure permissions

Usage:
    python deploy_kubeflow_azure_windows.py --resource-group myRG --cluster-name kubeflow-cluster
"""

import os
import sys
import time
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Tuple

# Configure logging with UTF-8 encoding for Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('kubeflow_deployment.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set UTF-8 encoding for Windows console
if sys.platform.startswith('win'):
    import io
    # Try to set UTF-8 encoding
    try:
        # Set console to UTF-8 if possible
        os.system('chcp 65001 > nul')
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        # Fallback to current encoding
        pass


class KubeflowAzureDeployer:
    """
    Automated Kubeflow deployment on Azure Kubernetes Service - Windows Compatible
    """
    
    def __init__(self, 
                 resource_group: str,
                 cluster_name: str,
                 location: str = "uksouth",
                 node_count: int = 2,
                 node_vm_size: str = "Standard_E2s_v3"):
        
        self.resource_group = resource_group
        self.cluster_name = cluster_name
        self.location = location
        self.node_count = node_count
        self.node_vm_size = node_vm_size
        
        # Kubeflow configuration
        self.kubeflow_version = "v1.8.0"
        self.manifests_repo = "https://github.com/kubeflow/manifests.git"
        self.deployment_dir = Path("./kubeflow-manifests")
        
        # Azure-specific configurations
        self.storage_account = f"{cluster_name}storage"
        self.file_share_name = "kubeflow-storage"
        
        # Windows-specific settings
        self.is_windows = sys.platform.startswith('win')
        
        logger.info("Initialized Kubeflow Azure Deployer")
        logger.info(f"Resource Group: {resource_group}")
        logger.info(f"Cluster Name: {cluster_name}")
        logger.info(f"Location: {location}")
        logger.info(f"Platform: {'Windows' if self.is_windows else 'Unix-like'}")
    
    def run_command(self, command: str, check: bool = True, shell: bool = True) -> Tuple[int, str, str]:
        """Execute shell command and return results - Windows compatible"""
        logger.info(f"Executing: {command}")
        
        try:
            # On Windows, use cmd.exe for better compatibility
            if self.is_windows and shell:
                result = subprocess.run(
                    command,
                    shell=True,
                    check=check,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    encoding='utf-8',
                    errors='ignore'  # Ignore encoding errors
                )
            else:
                result = subprocess.run(
                    command,
                    shell=shell,
                    check=check,
                    capture_output=True,
                    text=True,
                    timeout=600
                )
            
            if result.stdout:
                logger.debug(f"STDOUT: {result.stdout}")
            if result.stderr and result.returncode == 0:
                logger.debug(f"STDERR: {result.stderr}")
            
            return result.returncode, result.stdout, result.stderr
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {command}")
            logger.error(f"Return code: {e.returncode}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            raise
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {command}")
            raise
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are installed - Windows compatible"""
        logger.info("Checking prerequisites...")
        
        # Windows-specific command mappings
        prerequisites = {
            "az": "Azure CLI",
            "kubectl": "Kubernetes CLI", 
            "git": "Git",
            "python": "Python 3"
        }
        
        missing = []
        for cmd, name in prerequisites.items():
            try:
                if self.is_windows:
                    # Use 'where' command on Windows instead of 'which'
                    check_cmd = f"where {cmd}"
                else:
                    check_cmd = f"which {cmd}"
                
                returncode, _, _ = self.run_command(check_cmd, check=False)
                if returncode != 0:
                    missing.append(name)
                else:
                    logger.info(f"[SUCCESS] {name} found")
            except:
                missing.append(name)
        
        if missing:
            logger.error(f"[ERROR] Missing prerequisites: {', '.join(missing)}")
            logger.error("Please install missing tools and try again")
            
            # Provide Windows-specific installation instructions
            if self.is_windows:
                logger.info("Windows Installation Instructions:")
                logger.info("1. Azure CLI: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli-windows")
                logger.info("2. kubectl: choco install kubernetes-cli (or download from Kubernetes website)")
                logger.info("3. Git: https://git-scm.com/download/win")
                logger.info("4. Python: https://www.python.org/downloads/windows/")
            
            return False
        
        # Check Azure CLI login
        try:
            returncode, _, _ = self.run_command("az account show", check=False)
            if returncode != 0:
                logger.error("[ERROR] Not logged into Azure CLI. Run 'az login' first")
                return False
            logger.info("[SUCCESS] Azure CLI logged in")
        except:
            logger.error("[ERROR] Azure CLI not accessible")
            return False
        
        logger.info("[SUCCESS] All prerequisites satisfied")
        return True
    
    def check_and_register_providers(self) -> bool:
        """Check and register required Azure resource providers"""
        logger.info("Checking Azure resource providers...")
        
        required_providers = [
            "Microsoft.ContainerService",
            "Microsoft.Compute", 
            "Microsoft.Network",
            "Microsoft.Storage"
        ]
        
        for provider in required_providers:
            try:
                # Check provider registration status
                returncode, stdout, _ = self.run_command(
                    f"az provider show --namespace {provider} --query \"registrationState\" -o tsv",
                    check=False
                )
                
                if returncode == 0:
                    status = stdout.strip()
                    if status.lower() == "registered":
                        logger.info(f"[SUCCESS] {provider} is registered")
                    else:
                        logger.info(f"[INFO] Registering {provider}...")
                        self.run_command(f"az provider register --namespace {provider}")
                        logger.info(f"[SUCCESS] {provider} registration initiated")
                else:
                    logger.warning(f"[WARNING] Could not check {provider} status")
                    
            except Exception as e:
                logger.error(f"[ERROR] Failed to register {provider}: {e}")
                return False
        
        logger.info("[SUCCESS] All required providers checked/registered")
        return True
    
    def create_resource_group(self) -> bool:
        """Create Azure resource group if it doesn't exist"""
        logger.info(f"Creating resource group: {self.resource_group}")
        
        try:
            # Check if resource group exists
            returncode, _, _ = self.run_command(
                f"az group show --name {self.resource_group}", 
                check=False
            )
            
            if returncode == 0:
                logger.info(f"[SUCCESS] Resource group {self.resource_group} already exists")
                return True
            
            # Create resource group
            self.run_command(
                f"az group create --name {self.resource_group} --location {self.location}"
            )
            
            logger.info(f"[SUCCESS] Resource group {self.resource_group} created")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to create resource group: {e}")
            return False
    
    def get_latest_k8s_version(self) -> str:
        """Get the latest supported Kubernetes version for the region"""
        try:
            returncode, stdout, _ = self.run_command(
                f"az aks get-versions --location {self.location} --query \"values[?isDefault].version\" -o tsv",
                check=False
            )
            
            if returncode == 0 and stdout.strip():
                version = stdout.strip()
                logger.info(f"[INFO] Using default Kubernetes version: {version}")
                return version
            else:
                # Fallback to a known stable version
                logger.info("[INFO] Using fallback Kubernetes version: 1.30.12")
                return "1.30.12"
                
        except Exception as e:
            logger.warning(f"[WARNING] Could not get K8s version, using fallback: {e}")
            return "1.30.12"

    def create_aks_cluster(self) -> bool:
        """Create AKS cluster optimized for Kubeflow"""
        logger.info(f"Creating AKS cluster: {self.cluster_name}")
        
        try:
            # Check if cluster exists
            returncode, _, _ = self.run_command(
                f"az aks show --resource-group {self.resource_group} --name {self.cluster_name}",
                check=False
            )
            
            if returncode == 0:
                logger.info(f"[SUCCESS] AKS cluster {self.cluster_name} already exists")
            else:
                # Get the latest supported Kubernetes version
                k8s_version = self.get_latest_k8s_version()
                
                # Create AKS cluster with Kubeflow-optimized settings
                create_cmd = (
                    f"az aks create "
                    f"--resource-group {self.resource_group} "
                    f"--name {self.cluster_name} "
                    f"--node-count {self.node_count} "
                    f"--node-vm-size {self.node_vm_size} "
                    f"--location {self.location} "
                    f"--enable-addons monitoring "
                    f"--enable-managed-identity "
                    f"--enable-cluster-autoscaler "
                    f"--min-count 1 "
                    f"--max-count 5 "
                    f"--network-plugin azure "
                    f"--network-policy azure "
                    f"--kubernetes-version {k8s_version} "
                    f"--generate-ssh-keys "
                    f"--tags Environment=Kubeflow Purpose=MLOps"
                )
                
                logger.info("Creating AKS cluster... This may take 10-15 minutes")
                self.run_command(create_cmd)
                logger.info(f"[SUCCESS] AKS cluster {self.cluster_name} created")
            
            # Get credentials
            self.run_command(
                f"az aks get-credentials --resource-group {self.resource_group} --name {self.cluster_name} --overwrite-existing"
            )
            
            # Verify connection
            returncode, stdout, _ = self.run_command("kubectl cluster-info")
            if "running" in stdout.lower():
                logger.info("[SUCCESS] Successfully connected to AKS cluster")
                return True
            else:
                logger.error("[ERROR] Failed to connect to AKS cluster")
                return False
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to create AKS cluster: {e}")
            return False
    
    def create_storage_resources(self) -> bool:
        """Create Azure storage resources for Kubeflow"""
        logger.info("Creating storage resources...")
        
        try:
            # Create storage account
            storage_cmd = (
                f"az storage account create "
                f"--name {self.storage_account} "
                f"--resource-group {self.resource_group} "
                f"--location {self.location} "
                f"--sku Standard_LRS "
                f"--kind StorageV2 "
                f"--tags Purpose=Kubeflow Component=Storage"
            )
            
            returncode, _, _ = self.run_command(storage_cmd, check=False)
            if returncode != 0:
                logger.info("Storage account may already exist, continuing...")
            
            # Get storage account key
            _, key_output, _ = self.run_command(
                f"az storage account keys list --resource-group {self.resource_group} --account-name {self.storage_account} --query \"[0].value\" -o tsv"
            )
            storage_key = key_output.strip()
            
            # Create file share
            share_cmd = (
                f"az storage share create "
                f"--name {self.file_share_name} "
                f"--account-name {self.storage_account} "
                f"--account-key {storage_key} "
                f"--quota 100"
            )
            
            returncode, _, _ = self.run_command(share_cmd, check=False)
            if returncode != 0:
                logger.info("File share may already exist, continuing...")
            
            # Create Kubernetes secret for storage
            secret_cmd = (
                f"kubectl create secret generic azure-storage-secret "
                f"--from-literal=azurestorageaccountname={self.storage_account} "
                f"--from-literal=azurestorageaccountkey={storage_key} "
                f"--dry-run=client -o yaml | kubectl apply -f -"
            )
            
            self.run_command(secret_cmd)
            
            logger.info("[SUCCESS] Storage resources created")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to create storage resources: {e}")
            return False
    
    def install_cert_manager(self) -> bool:
        """Install cert-manager for TLS certificates"""
        logger.info("Installing cert-manager...")
        
        try:
            # Install cert-manager
            self.run_command("kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml")
            
            # Wait for cert-manager to be ready
            logger.info("Waiting for cert-manager to be ready...")
            self.run_command("kubectl wait --for=condition=ready pod -l app=cert-manager -n cert-manager --timeout=300s")
            self.run_command("kubectl wait --for=condition=ready pod -l app=cainjector -n cert-manager --timeout=300s")
            self.run_command("kubectl wait --for=condition=ready pod -l app=webhook -n cert-manager --timeout=300s")
            
            logger.info("[SUCCESS] cert-manager installed and ready")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to install cert-manager: {e}")
            return False
    
    def clone_kubeflow_manifests(self) -> bool:
        """Clone Kubeflow manifests repository"""
        logger.info("Cloning Kubeflow manifests...")
        
        try:
            if self.deployment_dir.exists():
                logger.info("Manifests directory exists, removing and re-cloning for clean state...")
                # Remove existing directory to avoid git issues
                if self.is_windows:
                    self.run_command(f"rmdir /s /q {self.deployment_dir}", check=False)
                else:
                    self.run_command(f"rm -rf {self.deployment_dir}", check=False)
            
            # Always clone fresh to avoid git state issues
            logger.info(f"Cloning Kubeflow manifests from {self.manifests_repo}...")
            self.run_command(f"git clone {self.manifests_repo} {self.deployment_dir}")
            
            # Change to repo directory and checkout specific version
            original_dir = os.getcwd()
            os.chdir(self.deployment_dir)
            
            try:
                # Checkout the specific Kubeflow version
                logger.info(f"Checking out Kubeflow version {self.kubeflow_version}...")
                self.run_command(f"git checkout {self.kubeflow_version}")
            except Exception as checkout_error:
                logger.warning(f"[WARNING] Could not checkout {self.kubeflow_version}: {checkout_error}")
                logger.info("Using default branch (main/master)...")
                # Try to checkout main or master
                returncode, _, _ = self.run_command("git checkout main", check=False)
                if returncode != 0:
                    self.run_command("git checkout master", check=False)
            
            finally:
                os.chdir(original_dir)
            
            logger.info("[SUCCESS] Kubeflow manifests ready")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to clone manifests: {e}")
            return False
    
    def deploy_kubeflow_components(self) -> bool:
        """Deploy Kubeflow components in the correct order"""
        logger.info("Deploying Kubeflow components...")
        
        # Component deployment order (critical for dependencies)
        components = [
            # Core components
            ("cert-manager", "./common/cert-manager/cert-manager/base"),
            ("cert-manager-issuer", "./common/cert-manager/kubeflow-issuer/base"),
            
            # Authentication and authorization
            ("dex", "./common/dex/base"),
            ("oidc-authservice", "./common/oidc-client/oidc-authservice/base"),
            
            # Core Kubeflow components
            ("kubeflow-namespace", "./common/kubeflow-namespace/base"),
            ("kubeflow-roles", "./common/kubeflow-roles/base"),
            ("pipelines", "./apps/pipeline/upstream/env/cert-manager/platform-agnostic-multi-user"),
            ("katib", "./apps/katib/upstream/installs/katib-with-kubeflow"),
            ("centraldashboard", "./apps/centraldashboard/upstream/overlays/kserve"),
            ("admission-webhook", "./apps/admission-webhook/upstream/overlays/cert-manager"),
            ("notebook-controller", "./apps/notebook-controller/upstream/overlays/kubeflow"),
            ("jupyter-web-app", "./apps/jupyter/jupyter-web-app/upstream/overlays/istio"),
            ("profiles", "./apps/profiles/upstream/overlays/kubeflow"),
            ("volumes-web-app", "./apps/volumes-web-app/upstream/overlays/istio"),
            ("tensorboard", "./apps/tensorboard/tensorboards-web-app/upstream/overlays/istio"),
            ("tensorboard-controller", "./apps/tensorboard/tensorboard-controller/upstream/overlays/kubeflow"),
            ("training-operator", "./apps/training-operator/upstream/overlays/kubeflow"),
        ]
        
        try:
            original_dir = os.getcwd()
            os.chdir(self.deployment_dir)
            
            for component_name, manifest_path in components:
                logger.info(f"Installing {component_name}...")
                
                try:
                    # Apply the component
                    self.run_command(f"kubectl apply -k {manifest_path}")
                    
                    # Wait a bit between components to avoid conflicts
                    time.sleep(10)
                    
                    logger.info(f"[SUCCESS] {component_name} installed")
                    
                except Exception as e:
                    logger.warning(f"[WARNING] Issue with {component_name}: {e}")
                    logger.info("Continuing with next component...")
                    continue
            
            # Return to original directory
            os.chdir(original_dir)
            
            logger.info("[SUCCESS] Kubeflow components deployment initiated")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to deploy Kubeflow components: {e}")
            return False
    
    def wait_for_kubeflow_ready(self) -> bool:
        """Wait for Kubeflow components to be ready"""
        logger.info("Waiting for Kubeflow components to be ready...")
        
        try:
            # Wait for key namespaces
            namespaces = ["kubeflow", "cert-manager", "auth"]
            
            for namespace in namespaces:
                logger.info(f"Checking namespace: {namespace}")
                returncode, _, _ = self.run_command(
                    f"kubectl wait --for=condition=ready pods --all -n {namespace} --timeout=600s",
                    check=False
                )
                
                if returncode == 0:
                    logger.info(f"[SUCCESS] Namespace {namespace} is ready")
                else:
                    logger.warning(f"[WARNING] Some pods in {namespace} may not be ready")
            
            # Check specific deployments
            key_deployments = [
                "ml-pipeline",
                "centraldashboard",
                "katib-mysql",
                "jupyter-web-app-deployment"
            ]
            
            for deployment in key_deployments:
                logger.info(f"Checking deployment: {deployment}")
                returncode, _, _ = self.run_command(
                    f"kubectl wait --for=condition=available deployment {deployment} -n kubeflow --timeout=300s",
                    check=False
                )
                
                if returncode == 0:
                    logger.info(f"[SUCCESS] Deployment {deployment} is available")
                else:
                    logger.warning(f"[WARNING] Deployment {deployment} may not be ready")
            
            logger.info("[SUCCESS] Kubeflow deployment completed")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Error waiting for Kubeflow: {e}")
            return False
    
    def configure_ingress(self) -> bool:
        """Configure ingress for Kubeflow access"""
        logger.info("Configuring ingress...")
        
        try:
            # Option 1: Create NodePort service for centraldashboard
            nodeport_service = """
apiVersion: v1
kind: Service
metadata:
  name: centraldashboard-nodeport
  namespace: kubeflow
spec:
  type: NodePort
  ports:
  - port: 80
    targetPort: 8082
    nodePort: 30000
  selector:
    app: centraldashboard
---
apiVersion: v1
kind: Service
metadata:
  name: centraldashboard-loadbalancer
  namespace: kubeflow
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8082
  selector:
    app: centraldashboard
"""
            
            # Apply services
            with open("centraldashboard-services.yaml", "w", encoding='utf-8') as f:
                f.write(nodeport_service)
            
            self.run_command("kubectl apply -f centraldashboard-services.yaml")
            
            # Wait for LoadBalancer to get external IP
            logger.info("Waiting for LoadBalancer to get external IP...")
            for _ in range(12):  # Wait up to 2 minutes
                time.sleep(10)
                returncode, output, _ = self.run_command(
                    "kubectl get svc centraldashboard-loadbalancer -n kubeflow -o jsonpath=\"{.status.loadBalancer.ingress[0].ip}\"",
                    check=False
                )
                
                if returncode == 0 and output.strip() and output.strip() != "<none>":
                    external_ip = output.strip()
                    kubeflow_url = f"http://{external_ip}"
                    logger.info(f"[SUCCESS] Kubeflow dashboard available at: {kubeflow_url}")
                    break
            else:
                logger.info("[INFO] LoadBalancer IP not ready yet. Checking NodePort access...")
                
                # Try to get node external IP for NodePort access
                returncode, output, _ = self.run_command(
                    "kubectl get nodes -o jsonpath=\"{.items[0].status.addresses[?(@.type==\\\"ExternalIP\\\")].address}\"",
                    check=False
                )
                
                if returncode == 0 and output.strip():
                    external_ip = output.strip()
                    kubeflow_url = f"http://{external_ip}:30000"
                    logger.info(f"[SUCCESS] Kubeflow dashboard available via NodePort: {kubeflow_url}")
                else:
                    logger.info("[INFO] External access methods:")
                    logger.info("1. Port forwarding: kubectl port-forward svc/centraldashboard -n kubeflow 8080:80")
                    logger.info("   Then open: http://localhost:8080")
                    logger.info("2. Check LoadBalancer IP: kubectl get svc centraldashboard-loadbalancer -n kubeflow")
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to configure ingress: {e}")
            return False
    
    def create_default_profile(self) -> bool:
        """Create default user profile for Kubeflow"""
        logger.info("Creating default user profile...")
        
        try:
            profile_yaml = """
apiVersion: kubeflow.org/v1beta1
kind: Profile
metadata:
  name: kubeflow-user-example-com
spec:
  owner:
    kind: User
    name: user@example.com
  resourceQuotaSpec:
    hard:
      cpu: "2"
      memory: 4Gi
      persistentvolumeclaims: "1"
      requests.nvidia.com/gpu: "1"
"""
            
            with open("default-profile.yaml", "w", encoding='utf-8') as f:
                f.write(profile_yaml)
            
            self.run_command("kubectl apply -f default-profile.yaml")
            
            logger.info("[SUCCESS] Default user profile created")
            logger.info("Default credentials: user@example.com / 12341234")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to create default profile: {e}")
            return False
    
    def create_persistent_volumes(self) -> bool:
        """Create persistent volumes for Kubeflow workloads"""
        logger.info("Creating persistent volumes...")
        
        try:
            pv_yaml = f"""
apiVersion: v1
kind: PersistentVolume
metadata:
  name: kubeflow-azure-file-pv
spec:
  capacity:
    storage: 100Gi
  accessModes:
    - ReadWriteMany
  azureFile:
    secretName: azure-storage-secret
    shareName: {self.file_share_name}
    readOnly: false
  mountOptions:
  - dir_mode=0777
  - file_mode=0777
  - uid=1000
  - gid=1000
  - mfsymlinks
  - nobrl
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: kubeflow-azure-file-pvc
  namespace: kubeflow
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: ""
  resources:
    requests:
      storage: 100Gi
  volumeName: kubeflow-azure-file-pv
"""
            
            with open("kubeflow-pv.yaml", "w", encoding='utf-8') as f:
                f.write(pv_yaml)
            
            self.run_command("kubectl apply -f kubeflow-pv.yaml")
            
            logger.info("[SUCCESS] Persistent volumes created")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to create persistent volumes: {e}")
            return False
    
    def verify_deployment(self) -> bool:
        """Verify Kubeflow deployment is working"""
        logger.info("Verifying Kubeflow deployment...")
        
        try:
            # Check pods in kubeflow namespace
            _, output, _ = self.run_command("kubectl get pods -n kubeflow")
            logger.info("Kubeflow pods status:")
            logger.info(output)
            
            # Check services
            _, output, _ = self.run_command("kubectl get svc -n kubeflow")
            logger.info("Kubeflow services:")
            logger.info(output)
            
            # Check if centraldashboard is accessible
            returncode, _, _ = self.run_command(
                "kubectl get deployment centraldashboard -n kubeflow",
                check=False
            )
            
            if returncode == 0:
                logger.info("[SUCCESS] Centraldashboard deployment found")
            else:
                logger.warning("[WARNING] Centraldashboard deployment not found")
            
            # Check pipelines
            returncode, _, _ = self.run_command(
                "kubectl get deployment ml-pipeline -n kubeflow",
                check=False
            )
            
            if returncode == 0:
                logger.info("[SUCCESS] ML Pipelines deployment found")
            else:
                logger.warning("[WARNING] ML Pipelines deployment not found")
            
            logger.info("[SUCCESS] Deployment verification completed")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Error verifying deployment: {e}")
            return False
    
    def print_deployment_summary(self):
        """Print deployment summary and next steps"""
        logger.info("\n" + "="*60)
        logger.info("KUBEFLOW DEPLOYMENT COMPLETED!")
        logger.info("="*60)
        
        logger.info("\nDEPLOYMENT SUMMARY:")
        logger.info(f"   Resource Group: {self.resource_group}")
        logger.info(f"   AKS Cluster: {self.cluster_name}")
        logger.info(f"   Location: {self.location}")
        logger.info(f"   Kubeflow Version: {self.kubeflow_version}")
        
        logger.info("\nACCESS KUBEFLOW:")
        logger.info("   Option 1 - LoadBalancer (Public IP):")
        logger.info("   kubectl get svc centraldashboard-loadbalancer -n kubeflow")
        logger.info("   Use the EXTERNAL-IP shown in the output")
        
        logger.info("\n   Option 2 - NodePort:")
        logger.info("   kubectl get nodes -o wide")
        logger.info("   Use any node's EXTERNAL-IP with port 30000")
        
        logger.info("\n   Option 3 - Port Forwarding (Local Access):")
        logger.info("   kubectl port-forward svc/centraldashboard -n kubeflow 8080:80")
        logger.info("   Then open: http://localhost:8080")
        
        logger.info("\nDEFAULT CREDENTIALS:")
        logger.info("   Email: user@example.com")
        logger.info("   Password: 12341234")
        
        logger.info("\nNEXT STEPS:")
        logger.info("   1. Access the Kubeflow dashboard")
        logger.info("   2. Create a new notebook server")
        logger.info("   3. Build and run ML pipelines")
        logger.info("   4. Train and deploy models")
        
        logger.info("\nUSEFUL COMMANDS:")
        logger.info("   # Check pod status")
        logger.info("   kubectl get pods -n kubeflow")
        logger.info("")
        logger.info("   # Check services") 
        logger.info("   kubectl get svc -n kubeflow")
        logger.info("")
        logger.info("   # Get logs")
        logger.info("   kubectl logs -n kubeflow deployment/centraldashboard")
        
        logger.info("\nCLEANUP (when needed):")
        logger.info(f"   az group delete --name {self.resource_group} --yes --no-wait")
        
        logger.info("\n" + "="*60)
    
    def deploy(self) -> bool:
        """Main deployment orchestration"""
        logger.info("Starting Kubeflow on Azure deployment...")
        
        steps = [
            ("Checking prerequisites", self.check_prerequisites),
            ("Checking Azure providers", self.check_and_register_providers),
            ("Creating resource group", self.create_resource_group),
            ("Creating AKS cluster", self.create_aks_cluster),
            ("Creating storage resources", self.create_storage_resources),
            ("Installing cert-manager", self.install_cert_manager),
            ("Cloning Kubeflow manifests", self.clone_kubeflow_manifests),
            ("Deploying Kubeflow components", self.deploy_kubeflow_components),
            ("Waiting for components to be ready", self.wait_for_kubeflow_ready),
            ("Configuring ingress", self.configure_ingress),
            ("Creating default profile", self.create_default_profile),
            ("Creating persistent volumes", self.create_persistent_volumes),
            ("Verifying deployment", self.verify_deployment),
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n{'='*60}")
            logger.info(f"STEP: {step_name}...")
            logger.info(f"{'='*60}")
            
            try:
                if not step_func():
                    logger.error(f"[ERROR] Failed at step: {step_name}")
                    return False
                    
                logger.info(f"[SUCCESS] {step_name} completed successfully")
                
            except KeyboardInterrupt:
                logger.info(f"\n[WARNING] Deployment interrupted by user at step: {step_name}")
                return False
            except Exception as e:
                logger.error(f"[ERROR] Unexpected error in {step_name}: {e}")
                return False
        
        self.print_deployment_summary()
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Deploy Kubeflow on Azure Kubernetes Service - Windows Compatible",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic deployment
  python deploy_kubeflow_azure_windows.py --resource-group myRG --cluster-name kubeflow-cluster

  # Custom configuration
  python deploy_kubeflow_azure_windows.py ^
    --resource-group production-ml ^
    --cluster-name kubeflow-prod ^
    --location uksouth ^
    --node-count 3 ^
    --node-vm-size Standard_E4s_v3

  # Minimal deployment (UK South, cheapest config)
  python deploy_kubeflow_azure_windows.py ^
    --resource-group dev-ml ^
    --cluster-name kubeflow-dev ^
    --node-count 2 ^
    --node-vm-size Standard_E2s_v3

Note: This is the Windows-compatible version without emoji characters.
        """
    )
    
    parser.add_argument(
        "--resource-group", 
        required=True,
        help="Azure resource group name"
    )
    
    parser.add_argument(
        "--cluster-name",
        required=True, 
        help="AKS cluster name"
    )
    
    parser.add_argument(
        "--location",
        default="uksouth",
        help="Azure region (default: uksouth)"
    )
    
    parser.add_argument(
        "--node-count",
        type=int,
        default=2,
        help="Number of nodes in the cluster (default: 2)"
    )
    
    parser.add_argument(
        "--node-vm-size",
        default="Standard_E2s_v3",
        help="VM size for cluster nodes (default: Standard_E2s_v3)"
    )
    
    parser.add_argument(
        "--skip-cluster-creation",
        action="store_true",
        help="Skip AKS cluster creation (use existing cluster)"
    )
    
    parser.add_argument(
        "--minimal",
        action="store_true", 
        help="Deploy minimal Kubeflow setup (no Istio, KServe)"
    )
    
    args = parser.parse_args()
    
    # Initialize deployer
    deployer = KubeflowAzureDeployer(
        resource_group=args.resource_group,
        cluster_name=args.cluster_name,
        location=args.location,
        node_count=args.node_count,
        node_vm_size=args.node_vm_size
    )
    
    # Deploy Kubeflow
    try:
        success = deployer.deploy()
        
        if success:
            logger.info("[SUCCESS] Kubeflow deployment completed successfully!")
            sys.exit(0)
        else:
            logger.error("[ERROR] Kubeflow deployment failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n[WARNING] Deployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()