#!/bin/bash
# === LLM Refinery — Local Control Plane Startup ===
# Run from WSL: bash scripts/start-local.sh
set -euo pipefail

cd "$(dirname "$0")/.."
echo "=== LLM Refinery Local Control Plane Starting ==="

# 1. Docker services
echo "[1/4] Starting Docker services..."
docker compose up -d --build
echo "  Docker services up"

# 2. Tailscale
echo "[2/4] Checking Tailscale..."
if tailscale status > /dev/null 2>&1; then
  LOCAL_IP=$(tailscale ip -4)
  echo "  Tailscale up: ${LOCAL_IP}"
else
  echo "  Tailscale not running — start with: sudo tailscale up"
  exit 1
fi

# 3. Firewall rule for pod
echo "[3/4] Adding iptables rule for pod..."
POD_IP="${GPU_POD_IP:-100.127.171.76}"
sudo iptables -C ts-input -s "${POD_IP}" -j ACCEPT 2>/dev/null \
  || sudo iptables -I ts-input 2 -s "${POD_IP}" -j ACCEPT
echo "  Allowing traffic from ${POD_IP}"

# 4. Socat forwarders
echo "[4/4] Starting socat forwarders..."
pkill -f "socat.*TCP-LISTEN.*fork" 2>/dev/null || true
sleep 1
bash scripts/tailscale-forward.sh &
sleep 2
echo "  Socat forwarders running"

echo ""
echo "=== Local control plane ready ==="
echo "  Backend:  http://localhost:8080"
echo "  MLflow:   http://localhost:5000"
echo "  MinIO:    http://localhost:9001"
echo "  Flower:   http://localhost:5555"
echo ""
echo "Next: Start RunPod and run 'bash /workspace/start.sh' on the pod."
