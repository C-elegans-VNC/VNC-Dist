# Use an official Python 3.9 slim-based image
FROM python:3.9-slim

# Set working directory within the container
WORKDIR /app

# Install required system dependencies, including libGL for OpenCV and Tk for tkinter
RUN apt-get update && apt-get install -y \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files in the repository to the container
COPY . .

# Ensure the Linux shortcut creation script is executable
RUN chmod +x run_vnc_quant_gui.sh

# Optional: Set up the Ubuntu desktop shortcut automatically
RUN chmod +x /app/run_vnc_quant_gui.sh && \
    mkdir -p ~/.local/share/applications && \
    printf "[Desktop Entry]\n\
Name=VNC-Quant\n\
Comment=Run the VNC-Quant GUI app\n\
Exec=/app/run_vnc_quant_gui.sh\n\
Terminal=true\n\
Type=Application\n\
Icon=/app/VNC-Quant.PNG\n" > ~/.local/share/applications/VNC-Quant.desktop && \
    chmod +x ~/.local/share/applications/VNC-Quant.desktop

# Expose port 8501 for the app
EXPOSE 8501

# Set the container entrypoint to launch the Streamlit app
ENTRYPOINT ["streamlit", "run", "VNC-Quant_GUI.py", "--server.port=8501", "--server.address=0.0.0.0"]

