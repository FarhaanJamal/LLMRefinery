#!/bin/bash
# === LLM Refinery — GPU Pod Startup Script ===
# Run after every pod restart: bash /workspace/start.sh
set -euo pipefail

echo "=== LLM Refinery GPU Pod Starting ==="

# 0. Install system deps (don't survive pod restarts)
echo "[0/5] Checking system dependencies..."
if ! command -v socat &>/dev/null || ! command -v redis-cli &>/dev/null; then
  apt-get update -qq && apt-get install -y -qq socat redis-tools > /dev/null 2>&1
  echo "  Installed socat + redis-tools"
else
  echo "  Already socat + redis-tools are installed"
fi

if ! command -v tailscale &>/dev/null; then
  curl -fsSL https://tailscale.com/install.sh | sh
  echo "  Installed Tailscale"
fi

if ! command -v celery &>/dev/null; then
  pip install -q --root-user-action=ignore torch==2.11.0 2>&1 | tail -1 || true
  pip install -q --no-build-isolation --root-user-action=ignore -r /workspace/compute_node/requirements.txt 2>&1 | tail -1 || true
  echo "  Installed Python dependencies"
fi

# 1. Load config
echo "[1/5] Loading config..."
set -a && source /workspace/config.env && set +a
echo "  Control plane IP: ${CONTROL_PLANE_IP}"

# 2. Start Tailscale
echo "[2/5] Starting Tailscale..."
if ! pgrep -x tailscaled > /dev/null; then
  tailscaled --tun=userspace-networking --socks5-server=localhost:1055 &
  sleep 3
fi

if ! tailscale status > /dev/null 2>&1; then
  # Replace with your auth key from https://login.tailscale.com/admin/settings/keys
  tailscale up --authkey=tskey-auth-XXXXXXXXXXXXX --hostname=llm-refinery-gpu
fi

POD_IP=$(tailscale ip -4)
echo "  Pod Tailscale IP: ${POD_IP}"

# 3. Start socat tunnels (userspace Tailscale can't bind directly)
echo "[3/5] Starting socat tunnels..."
pkill -f "socat.*tailscale nc" 2>/dev/null || true
sleep 1
socat TCP-LISTEN:18080,bind=127.0.0.1,reuseaddr,fork EXEC:"tailscale nc ${CONTROL_PLANE_IP} 18080" &
socat TCP-LISTEN:15000,bind=127.0.0.1,reuseaddr,fork EXEC:"tailscale nc ${CONTROL_PLANE_IP} 15000" &
socat TCP-LISTEN:16379,bind=127.0.0.1,reuseaddr,fork EXEC:"tailscale nc ${CONTROL_PLANE_IP} 16379" &
socat TCP-LISTEN:19000,bind=127.0.0.1,reuseaddr,fork EXEC:"tailscale nc ${CONTROL_PLANE_IP} 19000" &
sleep 2
echo "  4 socat tunnels started"

# 4. Test connectivity
echo "[4/5] Testing connectivity to control plane..."
FAIL=0
curl -sf "http://127.0.0.1:18080/docs" > /dev/null 2>&1 && echo "  FastAPI:  OK" || { echo "  FastAPI:  FAIL"; FAIL=1; }
curl -sf "http://127.0.0.1:15000" > /dev/null 2>&1 && echo "  MLflow:   OK" || { echo "  MLflow:   FAIL"; FAIL=1; }
redis-cli -h 127.0.0.1 -p 16379 ping > /dev/null 2>&1 && echo "  Redis:    OK" || { echo "  Redis:    FAIL"; FAIL=1; }
curl -sf "http://127.0.0.1:19000/minio/health/live" > /dev/null 2>&1 && echo "  MinIO:    OK" || { echo "  MinIO:    FAIL"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then
  echo ""
  echo "WARNING: Some services unreachable. Check Tailscale and local Docker."
  echo "Continuing anyway — worker will retry connections."
fi

# 5. Start Celery worker
echo "[5/5] Starting Celery worker (solo pool — no fork, CUDA-safe)..."
cd /workspace/compute_node
exec celery -A worker.app worker \
  --loglevel=info \
  --pool=solo \
  --hostname=gpu-worker@%h
