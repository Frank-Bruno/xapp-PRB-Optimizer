# Kubernetes manifests for RL model persistence

This folder contains a minimal example to persist RL checkpoints from the xApp pod.

## Files

- `rl-models-pv.yaml`: creates a static hostPath persistent volume for model files.
- `rl-models-pvc.yaml`: creates a persistent volume claim for model files.
- `xappnori-deployment-persistent.yaml`: deployment mounting the PVC at `/opt/rl-models`.

## Apply

```bash
kubectl apply -f k8s/rl-models-pvc.yaml
kubectl apply -f k8s/xappnori-deployment-persistent.yaml
```

If your cluster has no default dynamic provisioner, apply PV first:

```bash
kubectl apply -f k8s/rl-models-pv.yaml
kubectl apply -f k8s/rl-models-pvc.yaml
```

If you use `update_xapp.sh`, this is automatic now: the script applies the PVC and patches the deployed xApp with the `/opt/rl-models` mount plus `RAY_STORAGE` and `MODEL_EXPORT_DIR` env vars.

## Required edit

Before applying the deployment, replace this placeholder image:

- `YOUR_REGISTRY/xapp-prb-optimizer:latest`

## Verify persistence

```bash
kubectl -n ricxapp exec deploy/xappnori -- ls -la /opt/rl-models
kubectl -n ricxapp exec deploy/xappnori -- ls -la /opt/rl-models/export/latest
```

If pod restarts and these files remain, model persistence is working.
