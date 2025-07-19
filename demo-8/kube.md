# Using Chocolatey (if installed)
choco install kustomize

# Or download manually from:
# https://github.com/kubernetes-sigs/kustomize/releases

kustomize version

git clone https://github.com/kubeflow/manifests.git
cd manifests

kubectl apply -k \"./example\"

or

kubectl apply -k \"./common/cert-manager/cert-manager/base\"
kubectl apply -k \"./common/cert-manager/kubeflow-issuer/base\"
kubectl apply -k \"./common/istio-1-17/istio-crds/base\"
kubectl apply -k \"./common/istio-1-17/istio-namespace/base\"
kubectl apply -k \"./common/istio-1-17/istio-install/base\"
kubectl apply -k \"./common/dex/base\"
kubectl apply -k \"./common/oidc-client/oidc-authservice/base\"
kubectl apply -k \"./common/knative/knative-serving/overlays/gateways\"
kubectl apply -k \"./common/knative/knative-eventing/base\"
kubectl apply -k \"./common/istio-1-17/cluster-local-gateway/base\"
kubectl apply -k \"./apps/pipeline/upstream/env/cert-manager/platform-agnostic-multi-user\"
kubectl apply -k \"./apps/katib/upstream/installs/katib-with-kubeflow\"
kubectl apply -k \"./apps/centraldashboard/upstream/overlays/kserve\"
kubectl apply -k \"./apps/admission-webhook/upstream/overlays/cert-manager\"
kubectl apply -k \"./apps/notebook-controller/upstream/overlays/kubeflow\"
kubectl apply -k \"./apps/jupyter/notebook-controller/upstream/overlays/kubeflow\"
kubectl apply -k \"./contrib/notebook-controller/upstream/overlays/kubeflow\"
kubectl apply -k \"./apps/profiles/upstream/overlays/kubeflow\"
kubectl apply -k \"./apps/volumes-web-app/upstream/overlays/istio\"
kubectl apply -k \"./apps/tensorboard/tensorboards-web-app/upstream/overlays/istio\"
kubectl apply -k \"./apps/tensorboard/tensorboard-controller/upstream/overlays/kubeflow\"
kubectl apply -k \"./apps/training-operator/upstream/overlays/kubeflow\"
kubectl apply -k \"./apps/kserve/upstream/overlays/kubeflow\"