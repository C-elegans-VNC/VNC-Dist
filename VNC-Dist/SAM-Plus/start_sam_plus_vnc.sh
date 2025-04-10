#!/bin/bash
# Updated script to run the SAM-Plus-VNC container in the foreground using exec

PORT=8502

# Remove any existing container with the same name
if docker ps -a -q --filter "name=sam-plus-vnc-container" | grep -q .; then
  echo "Removing existing SAM-Plus-VNC container..."
  docker rm -f sam-plus-vnc-container
fi

echo "Starting SAM-Plus-VNC container on port $PORT..."
# Use exec to replace the shell with the docker run process,
# so that signals (e.g., SIGTERM when the terminal is closed) are sent to docker run.
exec docker run --rm -it --gpus all -p $PORT:$PORT --name sam-plus-vnc-container \
  sam_plus_vnc /app/venv/bin/streamlit run /app/SAM-Plus_GUI.py \
  --server.port=$PORT --server.address=0.0.0.0

