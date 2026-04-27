#!/usr/bin/env bash
# LLM Refinery — Phase 1 Integration Tests
# Usage: bash tests/test_phase1.sh

set -euo pipefail

PASS=0
FAIL=0
API="http://localhost:8080"

green() { echo -e "\033[32m✓ $1\033[0m"; }
red() { echo -e "\033[31m✗ $1\033[0m"; }

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
echo " LLM Refinery — Phase 1 Integration Tests"
echo "========================================="
echo ""

# --- 1. Docker Services ---
echo "--- Docker Services ---"

for svc in llm-refinery-backend llm-refinery-mongodb llm-refinery-redis llm-refinery-minio llm-refinery-mlflow; do
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

# Valid .jsonl upload
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

JOB_ID=$(echo "$EXP_RES" | python3 -c "import sys,json; print(json.load(sys.stdin).get('job_id',''))" 2>/dev/null)
JOB_STATUS=$(echo "$EXP_RES" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)

check $([[ -n "$JOB_ID" ]] && echo 0 || echo 1) "Experiment returns job_id"
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

# --- Cleanup Test Data ---
echo "--- Cleanup ---"

if [[ -n "$DATASET_ID" ]]; then
  # Remove test object from MinIO
  docker exec llm-refinery-minio mc alias set local http://localhost:9000 minioadmin minioadmin > /dev/null 2>&1
  docker exec llm-refinery-minio mc rm "local/datasets/${DATASET_ID}.jsonl" > /dev/null 2>&1

  # Remove test records from MongoDB
  docker exec llm-refinery-mongodb mongosh --quiet --eval "
    db.getSiblingDB('llm_refinery').datasets.deleteMany({dataset_id: '$DATASET_ID'});
    db.getSiblingDB('llm_refinery').jobs.deleteMany({});
  " > /dev/null 2>&1

  green "Cleaned up test data from MinIO and MongoDB"
fi

echo ""

# --- Summary ---
echo "========================================="
echo " Results: $PASS passed, $FAIL failed"
echo "========================================="

exit $FAIL
