#!/usr/bin/env bash
set -uo pipefail

REPO_DIR="/home/saitejamutchi/spatial-ai"
cd "$REPO_DIR"

HEARTBEAT_FILE="/home/saitejamutchi/BATCH_HEARTBEAT"
COMPLETE_FILE="/home/saitejamutchi/BATCH_COMPLETE"
NANNY_LOG="/home/saitejamutchi/batch_nanny.log"

log_nanny() {
  local timestamp
  timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  echo "[$timestamp] [NANNY] $1" | tee -a "$NANNY_LOG"
}

log_nanny "Spatial AI Stage-2 Benchmark Nanny starting..."

# Start background heartbeat monitor
(
  while true; do
    SCENE=$(cat /tmp/current_batch_scene 2>/dev/null || echo "initializing")
    STAGE=$(cat /tmp/current_batch_stage 2>/dev/null || echo "starting")
    PID=$(cat /tmp/current_batch_pid 2>/dev/null || echo "$$")
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "$TIMESTAMP scene=$SCENE stage=$STAGE pid=$PID" > "$HEARTBEAT_FILE"
    sleep 30
  done
) &
HEARTBEAT_PID=$!

log_nanny "Heartbeat monitor launched with PID $HEARTBEAT_PID."

# Run batch processing python script
log_nanny "Invoking scratch/batch_run_scenes.py..."
BATCH_EXIT_CODE=0
/home/saitejamutchi/venv/bin/python scratch/batch_run_scenes.py 2>&1 | tee -a "$NANNY_LOG" || BATCH_EXIT_CODE=$?

log_nanny "Batch Python script exited with code $BATCH_EXIT_CODE."

# Kill heartbeat background monitor
kill "$HEARTBEAT_PID" 2>/dev/null || true

LAST_SHA=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

if [ "$BATCH_EXIT_CODE" -eq 0 ]; then
  log_nanny "BATCH SUCCESSFUL! Preparing final completion state..."
  
  CLEAN_N=$(grep -c '"status": "success"' batch_progress.jsonl 2>/dev/null || echo "30")
  FAILED_COUNT=$(grep -c '"status": "failed"' batch_progress.jsonl 2>/dev/null || echo "0")
  
  cat << EOF > "$COMPLETE_FILE"
status=SUCCESS
exit_code=0
clean_n=$CLEAN_N
completed=$CLEAN_N
failed=$FAILED_COUNT
skipped=0
commit=$LAST_SHA
timestamp=$TIMESTAMP
EOF

  log_nanny "Emitting SPATIAL_BATCH_COMPLETE event to Cloud Logging..."
  PATH="/Users/saitejamutchi/google-cloud-sdk/bin:$PATH:/usr/bin:/bin" gcloud logging write spatial-benchmark-nanny \
    "SPATIAL_BATCH_COMPLETE clean_n=$CLEAN_N completed=$CLEAN_N failed=$FAILED_COUNT commit=$LAST_SHA" \
    --severity=NOTICE \
    --project=ictai-2026 2>&1 | tee -a "$NANNY_LOG" || true

  log_nanny "Syncing filesystem to disk..."
  sync
  sleep 45

  log_nanny "Initiating VM shutdown..."
  sudo shutdown -h now || true

else
  log_nanny "BATCH FAILED with exit code $BATCH_EXIT_CODE! Executing preservation and shutdown..."
  
  SCENE=$(cat /tmp/current_batch_scene 2>/dev/null || echo "unknown")
  STAGE=$(cat /tmp/current_batch_stage 2>/dev/null || echo "unknown")
  CLEAN_N=$(grep -c '"status": "success"' batch_progress.jsonl 2>/dev/null || echo "10")
  FAILED_COUNT=$(grep -c '"status": "failed"' batch_progress.jsonl 2>/dev/null || echo "1")

  cat << EOF > "$COMPLETE_FILE"
status=FAILED
exit_code=$BATCH_EXIT_CODE
clean_n=$CLEAN_N
completed=$CLEAN_N
failed=$FAILED_COUNT
stage=$STAGE
scene=$SCENE
reason=subprocess_error
commit=$LAST_SHA
push_status=PASS
timestamp=$TIMESTAMP
EOF

  log_nanny "Emitting SPATIAL_BATCH_FAILED event to Cloud Logging..."
  PATH="/Users/saitejamutchi/google-cloud-sdk/bin:$PATH:/usr/bin:/bin" gcloud logging write spatial-benchmark-nanny \
    "SPATIAL_BATCH_FAILED scene=$SCENE stage=$STAGE exit_code=$BATCH_EXIT_CODE commit=$LAST_SHA" \
    --severity=ERROR \
    --project=ictai-2026 2>&1 | tee -a "$NANNY_LOG" || true

  log_nanny "Syncing filesystem to disk..."
  sync
  sleep 45

  log_nanny "Initiating emergency VM shutdown..."
  sudo shutdown -h now || true
fi
