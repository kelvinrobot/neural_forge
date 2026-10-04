#!/usr/bin/env bash
# Container startup: download missing models, start both vLLM servers, wait for health, then start the API.
set -euo pipefail

APP_DIR="${APP_DIR:-/app}"
SUPERVISOR_CONF="${APP_DIR}/deploy/supervisord.conf"

if [ -d /data ] && [ -w /data ]; then
  export HF_HOME="/data/huggingface"
  echo "[entrypoint] Persistent storage detected; HF_HOME=${HF_HOME}"
else
  export HF_HOME="${HF_HOME:-/home/user/.cache/huggingface}"
  echo "[entrypoint] No persistent storage; HF_HOME=${HF_HOME}"
fi
export NF_MODEL_CACHE_DIR="${NF_MODEL_CACHE_DIR:-${HF_HOME}/neural_forge}"
export NF_TRAINED_HEADS_DIR="${NF_TRAINED_HEADS_DIR:-${NF_MODEL_CACHE_DIR}/trained_heads}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/home/user/.config/Ultralytics}"
mkdir -p "${HF_HOME}" "${NF_MODEL_CACHE_DIR}" "${NF_TRAINED_HEADS_DIR}" "${YOLO_CONFIG_DIR}"

export REASONING_PORT="${REASONING_PORT:-8001}"
export CONSULTANT_PORT="${CONSULTANT_PORT:-8002}"
export NF_REASONING_BASE_URL="${NF_REASONING_BASE_URL:-http://127.0.0.1:${REASONING_PORT}}"
export NF_CONSULTANT_BASE_URL="${NF_CONSULTANT_BASE_URL:-http://127.0.0.1:${CONSULTANT_PORT}}"
LLM_STARTUP_TIMEOUT="${LLM_STARTUP_TIMEOUT:-1800}"

echo "[entrypoint] Downloading missing model assets"
python "${APP_DIR}/scripts/download_models.py" || echo "[entrypoint] Some downloads failed; continuing so the logs stay visible"

echo "[entrypoint] Starting supervisord"
supervisord -c "${SUPERVISOR_CONF}" &
SUPERVISORD_PID=$!

shutdown() {
  echo "[entrypoint] Shutting down"
  supervisorctl -c "${SUPERVISOR_CONF}" shutdown >/dev/null 2>&1 || true
  wait "${SUPERVISORD_PID}" 2>/dev/null || true
  exit 0
}
trap shutdown SIGTERM SIGINT

for _ in $(seq 1 60); do
  if supervisorctl -c "${SUPERVISOR_CONF}" status >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

wait_for_health() {
  local name="$1"
  local url="$2"
  local deadline=$(( $(date +%s) + LLM_STARTUP_TIMEOUT ))
  echo "[entrypoint] Waiting for ${name} at ${url}"
  until curl -fsS "${url}" >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "${deadline}" ]; then
      echo "[entrypoint] ${name} did not become healthy within ${LLM_STARTUP_TIMEOUT}s"
      return 1
    fi
    sleep 5
  done
  echo "[entrypoint] ${name} is healthy"
}

supervisorctl -c "${SUPERVISOR_CONF}" start vllm_reasoning || echo "[entrypoint] vllm_reasoning failed to start; check the logs"
wait_for_health "reasoning LLM" "http://127.0.0.1:${REASONING_PORT}/health" || echo "[entrypoint] Continuing without a healthy reasoning LLM"

supervisorctl -c "${SUPERVISOR_CONF}" start vllm_consultant || echo "[entrypoint] vllm_consultant failed to start; check the logs"
wait_for_health "consultant LLM" "http://127.0.0.1:${CONSULTANT_PORT}/health" || echo "[entrypoint] Continuing without a healthy consultant LLM"

supervisorctl -c "${SUPERVISOR_CONF}" start api || echo "[entrypoint] api failed to start; check the logs"
echo "[entrypoint] API started on port ${NF_PORT:-7860}"

wait "${SUPERVISORD_PID}"
