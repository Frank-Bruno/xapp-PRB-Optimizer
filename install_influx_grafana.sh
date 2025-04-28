#!/bin/bash

# Install InfluxDB
kubectl create namespace influxdb
helm repo add influxdata https://helm.influxdata.com/
helm upgrade --install influxdb influxdata/influxdb2 --namespace influxdb --set persistence.enabled=false --set adminUser.username=admin --set adminUser.password=admin1234 --set adminUser.token=admin --set adminUser.organization=openranbr --set adminUser.bucket=openranbr --set adminUser.retention_policy="1h" --set service.type=NodePort --set service.port=8086 --set service.nodePort=30086 --set persistence.size="1Gi"

