#!/usr/bin/env bash
# Launch one vLLM OpenAI-compatible server on an internal localhost port. Usage: run_vllm.sh reasoning|consultant
set -euo pipefail

ROLE="${1:?role required: reasoning or consultant}"

REASONING_DEFAULT_MM_LIMIT='{"image": 1, "video": 0}'
CONSULTANT_DEFAULT_MM_LIMIT='{"image": 1}'

case "${ROLE}" in
  reasoning)
    MODEL_ID="${NF_REASONING_MODEL_ID:-Qwen/Qwen3-VL-4B-Instruct}"
    SERVED_NAME="${NF_REASONING_MODEL_NAME:-qwen3-vl-4b}"
    PORT="${REASONING_PORT:-8001}"
    GPU_UTILIZATION="${REASONING_GPU_MEMORY_UTILIZATION:-0.32}"
    QUANTIZATION="${REASONING_QUANTIZATION:-fp8}"
    MAX_MODEL_LEN="${REASONING_MAX_MODEL_LEN:-8192}"
    MAX_NUM_SEQS="${REASONING_MAX_NUM_SEQS:-4}"
    MM_LIMIT="${REASONING_MM_LIMIT:-$REASONING_DEFAULT_MM_LIMIT}"
    ;;
  consultant)
    MODEL_ID="${NF_CONSULTANT_MODEL_ID:-google/medgemma-1.5-4b-it}"
    SERVED_NAME="${NF_CONSULTANT_MODEL_NAME:-medgemma-1.5-4b}"
    PORT="${CONSULTANT_PORT:-8002}"
    GPU_UTILIZATION="${CONSULTANT_GPU_MEMORY_UTILIZATION:-0.30}"
    QUANTIZATION="${CONSULTANT_QUANTIZATION:-fp8}"
    MAX_MODEL_LEN="${CONSULTANT_MAX_MODEL_LEN:-4096}"
    MAX_NUM_SEQS="${CONSULTANT_MAX_NUM_SEQS:-4}"
    MM_LIMIT="${CONSULTANT_MM_LIMIT:-$CONSULTANT_DEFAULT_MM_LIMIT}"
    ;;
  *)
    echo "Unknown role: ${ROLE}" >&2
    exit 2
    ;;
esac

ARGUMENTS=(
  serve "${MODEL_ID}"
  --host 127.0.0.1
  --port "${PORT}"
  --served-model-name "${SERVED_NAME}"
  --gpu-memory-utilization "${GPU_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --limit-mm-per-prompt "${MM_LIMIT}"
  --dtype auto
)

if [ -n "${QUANTIZATION}" ] && [ "${QUANTIZATION}" != "none" ]; then
  ARGUMENTS+=(--quantization "${QUANTIZATION}")
fi

if [ "${VLLM_ENFORCE_EAGER:-0}" = "1" ]; then
  ARGUMENTS+=(--enforce-eager)
fi

echo "[vllm:${ROLE}] vllm ${ARGUMENTS[*]}"
exec vllm "${ARGUMENTS[@]}"
