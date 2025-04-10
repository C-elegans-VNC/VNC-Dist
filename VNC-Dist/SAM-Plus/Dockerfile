# Use the latest CUDA-enabled image as the base image
FROM nvidia/cuda:12.6.3-base-ubuntu24.04

# Set the working directory
WORKDIR /app

# Copy the application files into the container
COPY . /app

# Ensure the SAM model weights are accessible within the build context
# Copy the model weights into the container
COPY sam_vit_h_4b8939.pth /app/models/sam_vit_h_4b8939.pth

# Update and install required system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a Python virtual environment
RUN python3 -m venv /app/venv

# Upgrade pip, setuptools, and wheel in the virtual environment
RUN /app/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies from requirements.txt
RUN /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Install compatible PyTorch, torchvision, and torchaudio versions for CUDA 12.4
RUN /app/venv/bin/pip install --no-cache-dir \
    torch==2.5.1+cu124 \
    torchvision==0.20.1+cu124 \
    torchaudio==2.5.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Expose the default port for Streamlit
EXPOSE 8501

# Set environment variables for Streamlit to prevent credentials prompt
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Run the application with Streamlit using the virtual environment
ENTRYPOINT ["/app/venv/bin/streamlit", "run", "SAM-Plus_GUI.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0"]

