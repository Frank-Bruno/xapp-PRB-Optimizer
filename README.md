# xapp-nori
The xApp-NORI is an OSC xApp for interacting with gNBs simulated by the NORI NS-3 module.

This xApp sends a subscription request for the RAN Function ID 200 to all E2 nodes registered at the Near-RT RIC. The E2 Nodes should then start sending RIC Indication messages containing information about the RAN. The xApp responds to each message with a XXXXXXXXXXXX message.

All commands assume:
- You are running an [OpenRAN@Brasil Blueprint v1](https://github.com/LABORA-INF-UFG/openran-br-blueprint/wiki/OpenRAN@Brasil-Blueprint-v1) VM
- NORI is already installed, configured, running and registered at the Near-RT RIC
- You are inside the repository folder `xapp-nori/`

## How to use

Install the xApp:

```bash
bash update_xapp.sh
```

Get the xApp logs:

```bash
bash log_xapp.sh
```

If you want to test how NORI handles deleting subscriptions and accepting new ones:

```bash
bash resubscribe.sh
```

Uninstall the xApp:

```bash
dms_cli uninstall xappnori ricxapp
```

## Troubleshoot

If the xApp says no gNBs are registered, try redeploying the Near-RT RIC:

```bash
bash redeploy_ric.sh
```

Then, wait for all `ricplt` pods to be ready (with `1/1` or `2/2` on the `READY` column). You can check this with:

```bash
watch kubectl get pods -n ricplt
```

After the Near-RT RIC is ready, you can start NORI. Still, sometimes the gNBs do not register. This can happen because some Near-RT RIC components are still executing starting routines. As a workaround, stop and restart NORI again until it registers.