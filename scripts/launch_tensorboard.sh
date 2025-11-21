#!/bin/bash

# Script to launch TensorBoard automatically
# It checks if TensorBoard is already running, and if not, launches it.

LOG_DIR="runs"
PORT=6006

# Create log directory if it doesn't exist
mkdir -p $LOG_DIR

# Check if TensorBoard is already running
if pgrep -x "tensorboard" > /dev/null; then
    echo "TensorBoard is already running."
else
    echo "Starting TensorBoard on port $PORT..."
    # Launch in background and redirect output to /dev/null to avoid cluttering the terminal
    tensorboard --logdir=$LOG_DIR --port=$PORT --bind_all > /dev/null 2>&1 &
    echo "TensorBoard started at http://localhost:$PORT"
fi
