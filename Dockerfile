# Use CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the package in development mode
RUN pip3 install -e .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/models data/metrics/reports

# Set environment variables
ENV PYTHONPATH=/app
ENV LEGONIZER_API_HOST=0.0.0.0
ENV LEGONIZER_API_PORT=8000

# Expose port
EXPOSE 8000

# Run API server
CMD ["python3", "-m", "src.api.main"]
