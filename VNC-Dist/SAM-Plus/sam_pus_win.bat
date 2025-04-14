@echo off
REM Set the port to use
set PORT=8502

REM Remove any existing container named "sam-plus-vnc-container"
for /f "usebackq delims=" %%i in (`docker ps -a -q --filter "name=sam-plus-vnc-container"`) do (
    echo Removing existing SAM-Plus-VNC container with ID %%i...
    docker rm -f %%i
)

echo.
echo Starting SAM-Plus-VNC container on port %PORT%...
echo.

REM Run the container in interactive mode with GPU access.
docker run --rm -it --gpus all -p %PORT%:%PORT% --name sam-plus-vnc-container sam_plus_vnc /app/venv/bin/streamlit run /app/SAM-Plus_GUI.py --server.port=%PORT% --server.address=0.0.0.0

pause
