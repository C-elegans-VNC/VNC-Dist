#!/bin/bash

# Stop and remove any running container named "vnc-quant-gui"
docker stop vnc-quant-gui &>/dev/null || true
docker rm vnc-quant-gui &>/dev/null || true

# Check if port 8501 is already in use and free it if necessary
if sudo ss -tuln | grep -q ":8501"; then
    echo "Port 8501 is already in use. Freeing the port..."
    # Stop any container publishing on port 8501
    docker ps --filter "publish=8501" -q | xargs -r docker stop
fi

# Start the container in detached mode
echo "Starting the VNC-Quant application..."
docker run --rm --name vnc-quant-gui -p 8501:8501 vnc_quant_gui:latest &

# Get the container ID of the running container
CONTAINER_ID=$(docker ps -qf "name=vnc-quant-gui")

# Monitor the container; exit and clean up when the terminal or app is closed
trap "echo 'Stopping the container...'; docker stop vnc-quant-gui &>/dev/null || true; exit" SIGINT SIGTERM

while docker ps | grep -q "$CONTAINER_ID"; do
    sleep 2
done

