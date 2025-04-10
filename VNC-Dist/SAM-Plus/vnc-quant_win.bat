@echo off

REM Stop and remove any existing container named "vnc-quant-gui"
docker stop vnc-quant-gui >nul 2>&1
docker rm vnc-quant-gui >nul 2>&1

REM Open the app URL in your default browser
start "" http://localhost:8501

REM Run the container using your updated image (note the underscore)
docker run --rm --name vnc-quant-gui -p 8501:8501 vnc_quant_gui:latest >nul 2>&1

REM Monitor the container until it exits
:loop
docker ps -q --filter "name=vnc-quant-gui" >nul 2>&1
if errorlevel 1 goto cleanup
timeout /t 2 >nul
goto loop

:cleanup
REM Stop and remove the container (for cleanup)
docker stop vnc-quant-gui >nul 2>&1
docker rm vnc-quant-gui >nul 2>&1

exit
