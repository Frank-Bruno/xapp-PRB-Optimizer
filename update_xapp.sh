#!/bin/bash

namespace="ricxapp"
xapp_name="xappnori"
deployment_name="$namespace-$xapp_name"
mode="$(echo "${MODE:-train}" | tr '[:upper:]' '[:lower:]')"

case "$mode" in
    train)
        rl_inference_only="false"
        llm_agent="true"
        ;;
    infer)
        rl_inference_only="true"
        llm_agent="true"
        ;;
    pf|proportional_fair|proportional-fair)
        rl_inference_only="false"
        llm_agent="false"
        ;;
    llm)
        rl_inference_only="false"
        llm_agent="true"
        ;;
    *)
        echo "ERROR: invalid MODE '$mode'."
        echo "Use one of: train | infer | llm | pf | proportional_fair "
        exit 1
        ;;
esac

model_checkpoint_path="${MODEL_CHECKPOINT_PATH:-/opt/rl-models/export/latest}"

echo "namespace=$namespace"
echo "xapp_name=$xapp_name"
echo "deployment_name=$deployment_name"
echo "mode=$mode"
echo "rl_inference_only=$rl_inference_only"
echo "llm_agent=$llm_agent"

echo "----------------- Ensuring PVC for RL models -----------------"
kubectl apply -f k8s/rl-models-pv.yaml
if kubectl get pvc rl-models-pvc -n "$namespace" >/dev/null 2>&1; then
    echo "PVC rl-models-pvc already exists in namespace $namespace. Reusing existing volume."
else
    echo "PVC rl-models-pvc not found in namespace $namespace. Creating it now."
fi
kubectl apply -f k8s/rl-models-pvc.yaml

echo -n "Waiting PVC rl-models-pvc to bind"
for _ in $(seq 1 30)
do
    pvc_phase=$(kubectl get pvc rl-models-pvc -n "$namespace" -o jsonpath='{.status.phase}' 2>/dev/null)
    if [ "$pvc_phase" = "Bound" ]; then
        echo " - Bound"
        break
    fi
    sleep 1
    echo -n "."
done

if [ "$pvc_phase" != "Bound" ]; then
    echo ""
    echo "ERROR: PVC rl-models-pvc is not Bound (current: ${pvc_phase:-unknown})."
    kubectl get pvc rl-models-pvc -n "$namespace" -o wide
    exit 1
fi

echo "----------------- Onboarding the xApp chart -----------------"
dms_cli onboard init/config-file.json init/schema.json

echo "----------------- Terminating xApp pod -----------------"
dms_cli uninstall $xapp_name $namespace

echo -n "Waiting pod termination"
while kubectl get pods -n $namespace | grep -q $namespace-$xapp_name-
do
    sleep 1 # seconds
    echo -n "."
done

printf "\n"
echo "----------------- Removing previous image -----------------"
docker image rm 127.0.0.1:5001/$xapp_name:1.0.0

echo "----------------- Building new image -----------------"
docker build . -t 127.0.0.1:5001/$xapp_name:1.0.0 --network host

echo "----------------- Pushing new image -----------------"
docker push 127.0.0.1:5001/$xapp_name:1.0.0

echo "----------------- Installing the xApp -----------------"
dms_cli install $xapp_name 1.0.0 $namespace

echo -n "Waiting pod creation"
while ! kubectl get pods -n $namespace | grep $namespace-$xapp_name- | grep -q "1/1";
do
    if kubectl get pods -n $namespace | grep $namespace-$xapp_name- | grep -q CrashLoopBackOff; then 
        printf "\n%s" "INSTALLATION ERROR: CrashLoopBackOff"
        break
    fi
    sleep 1 # seconds
    echo -n .
done

printf "\n"

echo "----------------- Forcing deployment strategy to Recreate -----------------"
kubectl patch deployment "$deployment_name" -n "$namespace" --type='merge' -p '{"spec":{"strategy":{"type":"Recreate","rollingUpdate":null}}}'

echo "----------------- Patching deployment with persistent model volume -----------------"
patch_payload=$(cat <<EOF
{
    "spec": {
        "template": {
            "spec": {
                "volumes": [
                    {
                        "name": "rl-models",
                        "persistentVolumeClaim": {
                            "claimName": "rl-models-pvc"
                        }
                    }
                ],
                "containers": [
                    {
                        "name": "xappnoricontainer",
                        "env": [
                            {"name": "RAY_STORAGE", "value": "/opt/rl-models/ray_results"},
                            {"name": "MODEL_EXPORT_DIR", "value": "/opt/rl-models/export/latest"},
                            {"name": "RL_INFERENCE_ONLY", "value": "$rl_inference_only"},
                            {"name": "MODEL_CHECKPOINT_PATH", "value": "$model_checkpoint_path"},
                            {"name": "LLM_AGENT", "value": "$llm_agent"}
                        ],
                        "volumeMounts": [
                            {"name": "rl-models", "mountPath": "/opt/rl-models"}
                        ]
                    }
                ]
            }
        }
    }
}
EOF
)

kubectl patch deployment "$deployment_name" -n "$namespace" --type='strategic' -p "$patch_payload"

echo "----------------- Waiting rollout with persistence enabled -----------------"
if ! kubectl rollout status deployment/$deployment_name -n $namespace --timeout=300s; then
        echo "WARNING: rollout timed out. Continuing with checks anyway."
fi

echo "----------------- Cleaning stale pending xApp pods -----------------"
pending_pods=$(kubectl get pods -n "$namespace" --no-headers | awk '/^'"$namespace-$xapp_name"'-/ && $3=="Pending" {print $1}')
if [ -n "$pending_pods" ]; then
    echo "Deleting pending pod(s): $pending_pods"
    kubectl delete pod -n "$namespace" $pending_pods --ignore-not-found
fi

echo "----------------- Verifying model mount -----------------"
echo -n "Waiting for a Running+Ready xApp pod"
pod_name=""
for _ in $(seq 1 60)
do
    pod_name=$(kubectl get pods -n "$namespace" -o jsonpath='{range .items[*]}{.metadata.name}{"|"}{.status.phase}{"|"}{range .status.conditions[*]}{.type}{"="}{.status}{";"}{end}{"\n"}{end}' \
        | grep "^$namespace-$xapp_name-" \
        | awk -F'|' '$2=="Running" && $3 ~ /Ready=True/ {print $1}' \
        | tail -n1)
    if [ -n "$pod_name" ]; then
        break
    fi
    sleep 2
    echo -n "."
done
printf "\n"

if [ -z "$pod_name" ]; then
    echo "ERROR: no Running+Ready xApp pod found in namespace $namespace"
    kubectl get pods -n "$namespace" | grep "$namespace-$xapp_name-"
    exit 1
fi

if ! kubectl exec -n "$namespace" "$pod_name" -- ls -la /opt/rl-models; then
    echo "WARNING: could not list /opt/rl-models on pod $pod_name"
fi

echo "----------------- Verifying exported RL checkpoint -----------------"
checkpoint_file=$(kubectl exec -n "$namespace" "$pod_name" -- sh -lc '
if [ -d /opt/rl-models/export/latest ] && [ "$(ls -A /opt/rl-models/export/latest 2>/dev/null)" ]; then
    echo "/opt/rl-models/export/latest/$(ls -1 /opt/rl-models/export/latest | head -n1)"
elif [ -d /opt/rl-models/ray_results ] && [ "$(ls -A /opt/rl-models/ray_results 2>/dev/null)" ]; then
    echo "/opt/rl-models/ray_results/$(ls -1 /opt/rl-models/ray_results | head -n1)"
fi
')

if [ -n "$checkpoint_file" ]; then
    echo "Checkpoint found (example: $checkpoint_file)"
else
    echo "No checkpoint found yet in /opt/rl-models/export/latest or /opt/rl-models/ray_results."
fi

echo "----------------- Cleaning old ReplicaSets -----------------"
old_rs=$(kubectl get rs -n "$namespace" -l app="$deployment_name" \
    -o custom-columns=NAME:.metadata.name,REV:.metadata.annotations.deployment\.kubernetes\.io/revision --no-headers \
    | sort -k2,2n \
    | awk 'NF>0 {print $1}' \
    | head -n -1)

if [ -n "$old_rs" ]; then
    echo "Deleting old ReplicaSet(s):"
    echo "$old_rs"
    echo "$old_rs" | xargs -r kubectl delete rs -n "$namespace" --ignore-not-found
else
    echo "No old ReplicaSets to clean."
fi

echo "----------------- Getting pod's logs -----------------"
sleep 1
if ! kubectl logs "$pod_name" -n "$namespace" | tail -n 3; then
    echo "WARNING: could not get logs from pod $pod_name"
fi