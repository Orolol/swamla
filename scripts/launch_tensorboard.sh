#!/bin/bash

# Script to launch TensorBoard automatically
# Usage: ./launch_tensorboard.sh [LOG_DIR] [PORT]

LOG_DIR=${1:-"runs"}
PORT=${2:-6006}

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "Checking for existing TensorBoard on port $PORT..."

# Kill any process listening on the port
PID=$(lsof -ti:$PORT 2>/dev/null)
if [ -n "$PID" ]; then
    echo "Stopping existing process on port $PORT (PID: $PID)..."
    kill $PID 2>/dev/null || true
    sleep 1
fi

echo "Starting TensorBoard on port $PORT with logdir $LOG_DIR..."

# Try to find a working tensorboard command
TB_CMD=""
if command -v tensorboard &> /dev/null; then
    TB_CMD="tensorboard"
elif python -m tensorboard --version &> /dev/null; then
    TB_CMD="python -m tensorboard"
elif python3 -m tensorboard --version &> /dev/null; then
    TB_CMD="python3 -m tensorboard"
else
    echo "Error: Could not find tensorboard command."
    echo "Please ensure tensorboard is installed (pip install tensorboard)."
    exit 1
fi

echo "Using command: $TB_CMD"

# Launch in background
$TB_CMD --logdir="$LOG_DIR" --port=$PORT --bind_all > tb_launch.log 2>&1 &
TB_PID=$!

# Wait a moment to check if it stays up
sleep 2
if ps -p $TB_PID > /dev/null 2>&1; then
    echo "TensorBoard started successfully!"
    echo "URL: http://localhost:$PORT"
    echo "PID: $TB_PID"
else
    echo "Warning: Failed to start TensorBoard."
    echo "Check if tensorboard is installed and reachable."
fi
