#!/bin/bash
# DreamZero WebSocket Test Client
# Usage: bash scripts/wm/run_client.sh [--host HOST] [--port PORT] [--num_chunks N] [--use_zero_images]

set -euo pipefail

HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
NUM_CHUNKS="${NUM_CHUNKS:-15}"
PROMPT="${PROMPT:-Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan}"
USE_ZERO_IMAGES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --num_chunks) NUM_CHUNKS="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --use_zero_images) USE_ZERO_IMAGES="--use-zero-images"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== DreamZero Client ==="
echo "Server: ${HOST}:${PORT}"
echo "Chunks: ${NUM_CHUNKS}"
echo "Prompt: ${PROMPT}"
echo "========================"

python test_client_AR.py \
    --host "${HOST}" \
    --port "${PORT}" \
    --num-chunks "${NUM_CHUNKS}" \
    --prompt "${PROMPT}" \
    ${USE_ZERO_IMAGES}
