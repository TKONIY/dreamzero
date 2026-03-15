#!/bin/bash
# DreamZero Inference Profiling Script
# Usage: bash run_profile.sh [NUM_GPUS] [MODEL_PATH] [PORT]
#
# Example:
#   bash run_profile.sh 4 ./checkpoints/dreamzero 5000

set -e

NUM_GPUS=${1:-4}
MODEL_PATH=${2:-./checkpoints/dreamzero}
PORT=${3:-5000}
NUM_CHUNKS=${4:-10}

LOG_DIR="profile_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

SERVER_LOG="$LOG_DIR/server.log"
CLIENT_LOG="$LOG_DIR/client.log"

echo "=== DreamZero Inference Profiling ==="
echo "GPUs: $NUM_GPUS"
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "Chunks: $NUM_CHUNKS"
echo "Logs: $LOG_DIR/"
echo ""

# Start server in background
echo "[1/3] Starting inference server..."
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1))) \
    torchrun --standalone --nproc_per_node=$NUM_GPUS \
    socket_test_optimized_AR.py \
    --port $PORT \
    --enable-dit-cache \
    --model-path "$MODEL_PATH" \
    2>&1 | tee "$SERVER_LOG" &

SERVER_PID=$!

# Wait for server to be ready (health check endpoint)
echo "[2/3] Waiting for server to be ready..."
MAX_WAIT=600  # 10 minutes for model loading + torch.compile warmup
for i in $(seq 1 $MAX_WAIT); do
    if curl -s "http://localhost:$PORT/healthz" > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process died. Check $SERVER_LOG"
        exit 1
    fi
    if [ $i -eq $MAX_WAIT ]; then
        echo "ERROR: Server did not become ready in ${MAX_WAIT}s"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# Run test client
echo "[3/3] Running test client ($NUM_CHUNKS chunks)..."
python test_client_AR.py \
    --host localhost \
    --port $PORT \
    --num-chunks $NUM_CHUNKS \
    2>&1 | tee "$CLIENT_LOG"

# Give server a moment to flush logs
sleep 2

# Kill server
echo ""
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

# Parse results
echo ""
echo "=== Performance Report ==="
python profile_inference.py --server-log "$SERVER_LOG" --client-log "$CLIENT_LOG"

echo ""
echo "Raw logs saved to: $LOG_DIR/"
