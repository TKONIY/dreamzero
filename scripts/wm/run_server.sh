#!/bin/bash
# DreamZero WebSocket Policy Server with debugpy (rank 0 only)
#
# Usage:
#   bash scripts/wm/run_server.sh                          # debug mode (default)
#   bash scripts/wm/run_server.sh --no-debug               # without debugpy
#   bash scripts/wm/run_server.sh --model_path /path/to/ckpt --port 8000

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-./checkpoints/dreamzero}"
PORT="${PORT:-8000}"
DEBUG_PORT="${DEBUG_PORT:-5678}"
NPROC="${NPROC:-8}"
ENABLE_DIT_CACHE="${ENABLE_DIT_CACHE:-false}"
INDEX="${INDEX:-0}"
ENABLE_DEBUGPY="1"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)       MODEL_PATH="$2"; shift 2 ;;
        --port)             PORT="$2"; shift 2 ;;
        --debug_port)       DEBUG_PORT="$2"; shift 2 ;;
        --nproc)            NPROC="$2"; shift 2 ;;
        --index)            INDEX="$2"; shift 2 ;;
        --enable_dit_cache) ENABLE_DIT_CACHE=true; shift ;;
        --no-debug)         ENABLE_DEBUGPY=""; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== DreamZero Server ==="
echo "Model path:     ${MODEL_PATH}"
echo "WebSocket port: ${PORT}"
echo "GPUs (nproc):   ${NPROC}"
if [ -n "${ENABLE_DEBUGPY}" ]; then
    echo "Debugpy:        ON (port ${DEBUG_PORT}, attach VSCode before server starts)"
else
    echo "Debugpy:        OFF"
fi
echo "========================"

DIT_FLAG=""
if [ "${ENABLE_DIT_CACHE}" = "true" ]; then
    DIT_FLAG="--enable_dit_cache"
fi

ENABLE_DEBUGPY="${ENABLE_DEBUGPY}" DEBUG_PORT="${DEBUG_PORT}" \
torchrun --nproc_per_node="${NPROC}" \
    scripts/wm/server_debug_wrapper.py \
    --port "${PORT}" \
    --model_path "${MODEL_PATH}" \
    --index "${INDEX}" \
    ${DIT_FLAG}
