#!/bin/bash
# === LLM Refinery — Tailscale Port Forwarder ===
# Forwards traffic from Tailscale IP to Docker services on localhost.
# Required because Docker in WSL2 doesn't bind on the Tailscale interface.
# Usage: bash scripts/tailscale-forward.sh

set -euo pipefail

TAILSCALE_IP=$(tailscale ip -4)

# Ports to forward: service_name:tailscale_port:localhost_port
# Docker Desktop in WSL2 blocks standard ports on Tailscale IP,
# so we use alternate ports (1xxxx) on the Tailscale interface.
PORTS=(
  "FastAPI:18080:8080"
  "MLflow:15000:5000"
  "Redis:16379:6379"
  "MinIO-API:19000:9000"
  "MinIO-Console:19001:9001"
)

# Reverse tunnel: local → pod (for vLLM chat proxy)
# GPU_POD_IP is exported by start-local.sh from .env
POD_IP="${GPU_POD_IP:?GPU_POD_IP not set — source .env or run via start-local.sh}"
REVERSE_PORTS=(
  "vLLM:18000:${POD_IP}:8000"
)

echo "=== Tailscale Port Forwarder ==="
echo "Tailscale IP: ${TAILSCALE_IP}"
echo ""

# Kill any existing forwarders
pkill -f "socat.*${TAILSCALE_IP}" 2>/dev/null || true
sleep 1

for entry in "${PORTS[@]}"; do
  IFS=':' read -r name ext_port int_port <<< "$entry"
  socat TCP-LISTEN:${ext_port},bind=${TAILSCALE_IP},reuseaddr,fork TCP:127.0.0.1:${int_port} 2>/dev/null &
  echo "  ${name}: ${TAILSCALE_IP}:${ext_port} → 127.0.0.1:${int_port} (PID $!)"
done

echo ""
echo "Reverse tunnels (local → pod via Tailscale):"
for entry in "${REVERSE_PORTS[@]}"; do
  IFS=':' read -r name local_port remote_ip remote_port <<< "$entry"
  socat TCP-LISTEN:${local_port},reuseaddr,fork EXEC:"tailscale nc ${remote_ip} ${remote_port}" 2>/dev/null &
  echo "  ${name}: 127.0.0.1:${local_port} → ${remote_ip}:${remote_port} (PID $!)"
done

echo ""
echo "All ports forwarded. Press Ctrl+C to stop all."
wait
