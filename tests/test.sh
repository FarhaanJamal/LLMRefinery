#!/usr/bin/env bash
# LLM Refinery — Integration Tests
# Usage:
#   bash tests/test.sh          # full tests including pod connectivity & job flow
#   bash tests/test.sh --local   # local-only tests (no pod needed)

set -euo pipefail

PASS=0
FAIL=0
API="http://localhost:8080"
LOCAL_ONLY=false
[[ "${1:-}" == "--local" ]] && LOCAL_ONLY=true

green() { echo -e "\033[32m✓ $1\033[0m"; }
red()   { echo -e "\033[31m✗ $1\033[0m"; }

check() {
  if [ $1 -eq 0 ]; then
    green "$2"
    PASS=$((PASS + 1))
  else
    red "$2"
    FAIL=$((FAIL + 1))
  fi
}

echo "========================================="
echo " LLM Refinery — Integration Tests"
[[ "$LOCAL_ONLY" == "true" ]] && echo " (local only)" || echo " (full end-to-end)"
echo "========================================="
echo ""

# =============================================
#  LOCAL INFRASTRUCTURE
# =============================================

# --- 1. Docker Services ---
echo "--- Docker Services ---"

for svc in llm-refinery-backend llm-refinery-mongodb llm-refinery-redis llm-refinery-minio llm-refinery-mlflow llm-refinery-flower; do
  STATUS=$(docker inspect -f '{{.State.Running}}' "$svc" 2>/dev/null || echo "false")
  check $([[ "$STATUS" == "true" ]] && echo 0 || echo 1) "Container $svc is running"
done

echo ""

# --- 2. Service Health Checks ---
echo "--- Service Health ---"

curl -sf "$API/docs" > /dev/null 2>&1
check $? "FastAPI responding at :8080"

curl -sf "http://localhost:5000" > /dev/null 2>&1
check $? "MLflow responding at :5000"

PONG=$(docker exec llm-refinery-redis redis-cli ping 2>/dev/null)
check $([[ "$PONG" == "PONG" ]] && echo 0 || echo 1) "Redis responding"

MONGO_OK=$(docker exec llm-refinery-mongodb mongosh --quiet --eval "db.runCommand({ping:1}).ok" 2>/dev/null)
check $([[ "$MONGO_OK" == "1" ]] && echo 0 || echo 1) "MongoDB responding"

echo ""

# --- 3. Upload Endpoint ---
echo "--- Upload Endpoint ---"

TMPFILE=$(mktemp /tmp/test_XXXX.jsonl)
printf '{"instruction":"What is 2+2?","output":"4"}\n{"instruction":"Hello","output":"Hi"}\n' > "$TMPFILE"

UPLOAD_RES=$(curl -s -X POST "$API/api/dataset/upload" -F "file=@$TMPFILE")
DATASET_ID=$(echo "$UPLOAD_RES" | python3 -c "import sys,json; print(json.load(sys.stdin).get('dataset_id',''))" 2>/dev/null)
ROW_COUNT=$(echo "$UPLOAD_RES" | python3 -c "import sys,json; print(json.load(sys.stdin).get('row_count',0))" 2>/dev/null)
S3_PATH=$(echo "$UPLOAD_RES" | python3 -c "import sys,json; print(json.load(sys.stdin).get('s3_path',''))" 2>/dev/null)

check $([[ -n "$DATASET_ID" ]] && echo 0 || echo 1) "Upload returns dataset_id"
check $([[ "$ROW_COUNT" == "2" ]] && echo 0 || echo 1) "Upload returns correct row_count (2)"
check $([[ "$S3_PATH" == s3://datasets/* ]] && echo 0 || echo 1) "Upload returns valid s3_path"

# Reject non-.jsonl
BADFILE=$(mktemp /tmp/test_XXXX.txt)
echo "not json" > "$BADFILE"
BAD_RES=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API/api/dataset/upload" -F "file=@$BADFILE")
check $([[ "$BAD_RES" == "400" ]] && echo 0 || echo 1) "Rejects non-.jsonl file (400)"

# Reject invalid JSON lines
BADJSONL=$(mktemp /tmp/test_XXXX.jsonl)
printf '{"valid":"json"}\nnot json at all\n' > "$BADJSONL"
BAD_JSON_RES=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API/api/dataset/upload" -F "file=@$BADJSONL")
check $([[ "$BAD_JSON_RES" == "400" ]] && echo 0 || echo 1) "Rejects invalid JSON lines (400)"

rm -f "$TMPFILE" "$BADFILE" "$BADJSONL"

echo ""

# --- 4. Experiment Start Endpoint ---
echo "--- Experiment Start ---"

EXP_RES=$(curl -s -X POST "$API/api/experiment/start" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"meta-llama/Meta-Llama-3-8B\",\"task\":\"qlora\",\"params\":{\"r\":16,\"alpha\":32,\"quant_type\":\"awq\"},\"dataset_path\":\"$S3_PATH\"}")

LOCAL_JOB_ID=$(echo "$EXP_RES" | python3 -c "import sys,json; print(json.load(sys.stdin).get('job_id',''))" 2>/dev/null)
JOB_STATUS=$(echo "$EXP_RES" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)

check $([[ -n "$LOCAL_JOB_ID" ]] && echo 0 || echo 1) "Experiment returns job_id"
check $([[ "$JOB_STATUS" == "queued" ]] && echo 0 || echo 1) "Experiment status is 'queued'"

# Invalid quant_type
BAD_EXP=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API/api/experiment/start" \
  -H "Content-Type: application/json" \
  -d '{"model":"test","params":{"r":16,"alpha":32,"quant_type":"invalid"},"dataset_path":"s3://datasets/x.jsonl"}')
check $([[ "$BAD_EXP" == "400" ]] && echo 0 || echo 1) "Rejects invalid quant_type (400)"

echo ""

# --- 5. Results Endpoint ---
echo "--- Results Endpoint ---"

RESULTS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API/api/experiments/results")
check $([[ "$RESULTS_CODE" == "200" ]] && echo 0 || echo 1) "Results endpoint returns 200"

RESULTS_BODY=$(curl -s "$API/api/experiments/results")
HAS_EXPERIMENTS=$(echo "$RESULTS_BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); print('yes' if 'experiments' in d else 'no')" 2>/dev/null)
check $([[ "$HAS_EXPERIMENTS" == "yes" ]] && echo 0 || echo 1) "Results body contains 'experiments' key"

echo ""

# --- 6. MongoDB Records ---
echo "--- MongoDB Persistence ---"

DS_COUNT=$(docker exec llm-refinery-mongodb mongosh --quiet --eval "db.getSiblingDB('llm_refinery').datasets.countDocuments()" 2>/dev/null)
check $([[ "$DS_COUNT" -ge 1 ]] && echo 0 || echo 1) "MongoDB has dataset records ($DS_COUNT)"

JOB_COUNT=$(docker exec llm-refinery-mongodb mongosh --quiet --eval "db.getSiblingDB('llm_refinery').jobs.countDocuments()" 2>/dev/null)
check $([[ "$JOB_COUNT" -ge 1 ]] && echo 0 || echo 1) "MongoDB has job records ($JOB_COUNT)"

echo ""

# --- Local Cleanup ---
echo "--- Local Cleanup ---"

if [[ -n "$DATASET_ID" ]]; then
  docker exec llm-refinery-minio mc alias set local http://localhost:9000 minioadmin minioadmin > /dev/null 2>&1
  docker exec llm-refinery-minio mc rm "local/datasets/${DATASET_ID}.jsonl" > /dev/null 2>&1

  # If pod is running, wait for worker to process the local job before cleanup
  if [[ "$LOCAL_ONLY" != "true" ]]; then
    echo "  Waiting for worker to process local test job..."
    for i in $(seq 1 12); do
      sleep 5
      LOCAL_DB_STATUS=$(docker exec llm-refinery-mongodb mongosh --quiet --eval \
        "db.getSiblingDB('llm_refinery').jobs.findOne({job_id:'$LOCAL_JOB_ID'},{status:1,_id:0}).status" 2>/dev/null || echo "")
      if [[ "$LOCAL_DB_STATUS" == "completed" || "$LOCAL_DB_STATUS" == "failed" ]]; then
        break
      fi
    done
  fi

  # Get MLflow run_id for the local test job
  LOCAL_MLFLOW_RUN_ID=$(curl -s --max-time 5 "$API/api/experiments/results" 2>/dev/null | python3 -c "
import sys,json
data = json.load(sys.stdin).get('experiments',[])
match = [e for e in data if e.get('job_id') == '$LOCAL_JOB_ID']
print(match[0]['run_id'] if match else '')
" 2>/dev/null || echo "")

  docker exec llm-refinery-mongodb mongosh --quiet --eval "
    db.getSiblingDB('llm_refinery').datasets.deleteMany({dataset_id: '$DATASET_ID'});
    db.getSiblingDB('llm_refinery').jobs.deleteOne({job_id: '$LOCAL_JOB_ID'});
  " > /dev/null 2>&1

  if [[ -n "$LOCAL_MLFLOW_RUN_ID" ]]; then
    curl -s --max-time 5 -X POST "http://localhost:5000/api/2.0/mlflow/runs/delete" \
      -H "Content-Type: application/json" \
      -d "{\"run_id\": \"$LOCAL_MLFLOW_RUN_ID\"}" > /dev/null 2>&1
  fi

  green "Cleaned up local test data from MinIO, MongoDB, and MLflow"
fi

echo ""

# =============================================
#  END-TO-END (POD CONNECTIVITY & JOB FLOW)
# =============================================

if [[ "$LOCAL_ONLY" != "true" ]]; then

  echo "============================================="
  echo " End-to-End Tests (Pod Connectivity & Jobs)"
  echo "============================================="
  echo ""

  # --- 7. Tailscale ---
  echo "--- Tailscale ---"

  tailscale status > /dev/null 2>&1
  check $? "Tailscale is running locally"

  LOCAL_IP=$(tailscale ip -4 2>/dev/null || echo "")
  check $([[ -n "$LOCAL_IP" ]] && echo 0 || echo 1) "Local Tailscale IP: ${LOCAL_IP:-none}"

  echo ""

  # --- 8. Socat Forwarders ---
  echo "--- Local Socat Forwarders ---"

  SOCAT_COUNT=$(pgrep -c -f "socat.*TCP-LISTEN" 2>/dev/null || echo "0")
  check $([[ "$SOCAT_COUNT" -ge 4 ]] && echo 0 || echo 1) "Socat forwarders running ($SOCAT_COUNT processes)"

  echo ""

  # --- 9. Flower / Celery Workers ---
  echo "--- Celery Monitor ---"

  FLOWER_RESP=$(curl -sf "http://localhost:5555/api/workers?refresh=true" 2>/dev/null || echo "")
  check $([[ -n "$FLOWER_RESP" ]] && echo 0 || echo 1) "Flower API responding"

  WORKER_COUNT=$(echo "$FLOWER_RESP" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
  if [[ "$WORKER_COUNT" -eq 0 ]]; then
    sleep 5
    FLOWER_RESP=$(curl -sf "http://localhost:5555/api/workers?refresh=true" 2>/dev/null || echo "")
    WORKER_COUNT=$(echo "$FLOWER_RESP" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
  fi
  check $([[ "$WORKER_COUNT" -ge 1 ]] && echo 0 || echo 1) "Celery workers connected: $WORKER_COUNT"

  echo ""

  # --- 10. End-to-End Job Flow ---
  echo "--- End-to-End Job Flow ---"

  BEFORE=$(curl -s --max-time 5 "$API/api/experiments/results" 2>/dev/null | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('experiments',[])))" 2>/dev/null || echo "0")

  JOB_RESP=$(curl -s --max-time 10 -X POST "$API/api/experiment/start" \
    -H "Content-Type: application/json" \
    -d '{"model":"test/e2e-model","task":"qlora","params":{"r":8,"alpha":16,"quant_type":"awq","eval_mode":"quick"},"dataset_path":"s3://datasets/test.jsonl"}' 2>/dev/null)

  E2E_JOB_ID=$(echo "$JOB_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('job_id',''))" 2>/dev/null || echo "")
  E2E_STATUS=$(echo "$JOB_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "")

  check $([[ "$E2E_STATUS" == "queued" ]] && echo 0 || echo 1) "Job submitted: ${E2E_JOB_ID:-none} (status=$E2E_STATUS)"

  MONGO_STATUS=$(docker exec llm-refinery-mongodb mongosh --quiet --eval \
    "db.getSiblingDB('llm_refinery').jobs.findOne({job_id:'$E2E_JOB_ID'},{status:1,_id:0}).status" 2>/dev/null || echo "")
  check $([[ "$MONGO_STATUS" == "queued" ]] && echo 0 || echo 1) "MongoDB job record created (status=$MONGO_STATUS)"

  echo ""

  # --- 11. Wait for Worker ---
  echo "--- Waiting for Worker (up to 60s) ---"

  COMPLETED=false
  for i in $(seq 1 12); do
    sleep 5
    DB_STATUS=$(docker exec llm-refinery-mongodb mongosh --quiet --eval \
      "db.getSiblingDB('llm_refinery').jobs.findOne({job_id:'$E2E_JOB_ID'},{status:1,_id:0}).status" 2>/dev/null || echo "")
    echo "  [$((i*5))s] MongoDB status: $DB_STATUS"
    if [[ "$DB_STATUS" == "completed" ]]; then
      COMPLETED=true
      break
    fi
  done

  check $([[ "$COMPLETED" == "true" ]] && echo 0 || echo 1) "Worker completed job"

  echo ""

  # --- 12. MLflow Results ---
  echo "--- MLflow Results ---"

  AFTER=$(curl -s --max-time 5 "$API/api/experiments/results" 2>/dev/null | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('experiments',[])))" 2>/dev/null || echo "0")
  check $([[ "$AFTER" -gt "$BEFORE" ]] && echo 0 || echo 1) "New MLflow run logged ($BEFORE → $AFTER experiments)"

  RUN_DATA=$(curl -s --max-time 5 "$API/api/experiments/results" 2>/dev/null | python3 -c "
import sys,json
data = json.load(sys.stdin).get('experiments',[])
match = [e for e in data if e.get('job_id') == '$E2E_JOB_ID']
if match:
    r = match[0]
    print(f\"model={r.get('model','')} quant={r.get('quantization_type','')} acc={r.get('accuracy',0)} lat={r.get('latency',0)}\")
else:
    print('NOT_FOUND')
" 2>/dev/null || echo "NOT_FOUND")

  check $([[ "$RUN_DATA" != "NOT_FOUND" ]] && echo 0 || echo 1) "MLflow run has correct data: $RUN_DATA"

  # Get MLflow run_id for cleanup
  MLFLOW_RUN_ID=$(curl -s --max-time 5 "$API/api/experiments/results" 2>/dev/null | python3 -c "
import sys,json
data = json.load(sys.stdin).get('experiments',[])
match = [e for e in data if e.get('job_id') == '$E2E_JOB_ID']
print(match[0]['run_id'] if match else '')
" 2>/dev/null || echo "")

  echo ""

  # --- E2E Cleanup ---
  echo "--- E2E Cleanup ---"

  if [[ -n "$E2E_JOB_ID" ]]; then
    docker exec llm-refinery-mongodb mongosh --quiet --eval \
      "db.getSiblingDB('llm_refinery').jobs.deleteOne({job_id: '$E2E_JOB_ID'})" > /dev/null 2>&1

    if [[ -n "$MLFLOW_RUN_ID" ]]; then
      curl -s --max-time 5 -X POST "http://localhost:5000/api/2.0/mlflow/runs/delete" \
        -H "Content-Type: application/json" \
        -d "{\"run_id\": \"$MLFLOW_RUN_ID\"}" > /dev/null 2>&1
    fi

    green "Cleaned up E2E test data from MongoDB and MLflow"
  fi

  echo ""
fi

# =============================================
#  SUMMARY
# =============================================

TOTAL=$((PASS + FAIL))
echo "========================================="
echo " Results: $PASS/$TOTAL passed"
if [ $FAIL -eq 0 ]; then
  echo -e " \033[32mAll tests passed!\033[0m"
else
  echo -e " \033[31m$FAIL test(s) failed\033[0m"
fi
echo "========================================="

exit $FAIL
